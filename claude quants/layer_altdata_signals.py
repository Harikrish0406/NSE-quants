# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS — Alternative-Data Signal Layer (layer_altdata_signals.py)

New signal-scoring functions built from alt-data sources that the live
production pipeline (parent folder) already pulls but never turned into a
signal: institutional bulk/block deals, the earnings event calendar (plus a
precomputed post-earnings drift stat), and F&O positioning (PCR / OI / max
pain).

READ-ONLY DATA ACCESS, WRITE ISOLATION
---------------------------------------
Per instructions, this file must not touch anything in the parent folder
(`..\\`, `D:\\MBA\\STOCK MARKET RESEARCH\\NSE quants py\\`) — that is a
separate live production pipeline. This module only READS the following
parent-folder parquet files and never writes/edits them:
    bulk_deals.parquet, block_deals.parquet,
    earnings_calendar.parquet, earnings_dna.parquet,
    earnings_dna_historical.parquet, earnings_dna_drift.parquet,
    fno_pcr.parquet, fno_oi.parquet, fno_maxpain.parquet
Everything this module writes (if ever run) stays inside this
`claude quants\\` folder, matching the convention already used by
layer_swing_signals.py.

HONEST DATA-RANGE FINDINGS (checked before writing any code)
--------------------------------------------------------------
Every alt-data source pulled by the live pipeline turned out to be a
current-snapshot (or near-snapshot) export, NOT a historical archive:

  bulk_deals.parquet              deal_date 2026-04-20 -> 2026-04-21
                                   (2 unique dates, 21 unique tickers, 70 rows)
  block_deals.parquet             deal_date 2026-04-24 -> 2026-05-08
                                   (4 unique dates, 5 unique tickers, 70 rows)
  earnings_calendar.parquet       board_meeting_date 2026-06-13 -> 2026-08-14
                                   (55 unique dates, 986 rows) — a FORWARD
                                   calendar of upcoming board meetings, single pull
  earnings_dna.parquet            next_result_date 2026-07-10 -> 2026-08-14
                                   (31 unique dates, 294 rows) — FORWARD calendar,
                                   single pull
  earnings_dna_historical.parquet next_result_date 2026-05-18 -> 2026-05-30
                                   (12 unique dates, 893 rows) — despite the
                                   filename this is ALSO just a forward
                                   "next result date" snapshot from an earlier
                                   pull, not a dated archive of PAST results
  earnings_dna_drift.parquet      NO date column at all. 1551 tickers, columns
                                   Ticker / AvgDrift_D1_D10 / ReliabilityFlag /
                                   Direction — a precomputed cross-sectional stat
                                   (presumably built internally by the live
                                   pipeline from its own multi-quarter history).
                                   Only 48/1551 rows have ReliabilityFlag=True.
                                   Consumed here as-is; NOT independently
                                   re-validated since the underlying dated
                                   observations aren't available in this file.
  fno_pcr.parquet                 fetch_date 2026-07-13 only (1 unique date, 69 rows)
  fno_oi.parquet                  fetch_date 2026-07-13 only (1 unique date, 69 rows)
  fno_maxpain.parquet             fetch_date 2026-07-13 only (1 unique date, 69 rows)

None of these has the months-scale, repeated-over-time history that
validation_framework.run_walkforward's fold generator needs (WF_CONFIGS:
swing = 12mo train / 3mo test, long = 24mo train / 6mo test, folds generated
2010-2026). bulk_deals/block_deals come closest — a real, if tiny,
institutional-flow window — but even that is only 2-4 distinct dates spread
over at most ~2.5 weeks: nowhere near enough to build genuine train/test
folds without fabricating history that doesn't exist. The F&O snapshots and
every earnings-calendar file are single-pull, forward-looking data with zero
repeat observations, so a walk-forward test isn't just weak there, it is not
constructible at all yet.

Per an explicit mid-task instruction from the coordinator, NO validation or
backtest of any kind was executed for this file (no run_walkforward call, no
ad-hoc cross-sectional IC check, foreground or background — nothing). Every
function below is a LIVE-SCORING function only, ready to call against
today's snapshot, and is explicitly commented UNVALIDATED. Do not treat any
of these as PASS/FAIL-tested; treat them exactly as what they are:
reasonable, leak-free-by-construction scoring logic that has not yet been
run against real forward returns.

The institutional-flow score is also exposed via a `*_signal_fn_factory`
helper so it can be dropped straight into validation_framework.run_walkforward
once enough daily deal snapshots have accumulated on disk to form real
folds. That wiring is written but has not been exercised.

Run: not run by this build. If ever run manually (`python
layer_altdata_signals.py`), the __main__ block below only prints live scores
for today's snapshot — it is a smoke print, not a validation run.
"""

import os

import numpy as np
import pandas as pd

from validation_framework import PROJECT_DIR, log

# Parent folder = live production pipeline. READ-ONLY from here on — this
# module must never write/edit anything under PARENT_DIR.
PARENT_DIR = os.path.dirname(PROJECT_DIR)

ALTDATA_FILES = {
    "bulk_deals": os.path.join(PARENT_DIR, "bulk_deals.parquet"),
    "block_deals": os.path.join(PARENT_DIR, "block_deals.parquet"),
    "earnings_calendar": os.path.join(PARENT_DIR, "earnings_calendar.parquet"),
    "earnings_dna": os.path.join(PARENT_DIR, "earnings_dna.parquet"),
    "earnings_dna_historical": os.path.join(PARENT_DIR, "earnings_dna_historical.parquet"),
    "earnings_dna_drift": os.path.join(PARENT_DIR, "earnings_dna_drift.parquet"),
    "fno_pcr": os.path.join(PARENT_DIR, "fno_pcr.parquet"),
    "fno_oi": os.path.join(PARENT_DIR, "fno_oi.parquet"),
    "fno_maxpain": os.path.join(PARENT_DIR, "fno_maxpain.parquet"),
}


def _safe_read(key):
    """Read-only load of a parent-folder alt-data file. Returns an empty
    DataFrame (never raises) if the file is missing or unreadable, since the
    live pipeline refreshes these daily and a given file may not exist at
    call time."""
    path = ALTDATA_FILES[key]
    if not os.path.exists(path):
        log.warning("altdata file not found (read-only lookup): %s", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log.warning("failed to read %s: %s", path, e)
        return pd.DataFrame()


def _sentiment_to_score(val):
    """Best-effort numeric coercion for sentiment-style fields that may be
    stored as strings (e.g. 'Bullish'/'Bearish'/'Neutral') or already
    numeric (-1/0/+1-style). Returns NaN if it can't be interpreted."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or (isinstance(val, str) and not val.strip()):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip().lower()
    if "bull" in s or s in ("buy", "long", "up", "positive"):
        return 1.0
    if "bear" in s or s in ("sell", "short", "down", "negative"):
        return -1.0
    if "neutral" in s:
        return 0.0
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — INSTITUTIONAL FLOW (bulk_deals + block_deals)
#
# score = (buy_value_cr - sell_value_cr) / (buy_value_cr + sell_value_cr)
# over a trailing lookback window, per ticker. +1 = fully one-sided buying,
# -1 = fully one-sided selling, NaN = no deals for that ticker in the window
# (caller should treat NaN as "no signal", not neutral — 0.0 is reserved for
# a genuinely balanced buy/sell tape).
#
# UNVALIDATED — see module docstring: bulk_deals/block_deals as pulled cover
# only 2-4 distinct dates over <=2.5 weeks. This has NOT been walk-forward
# tested against forward returns.
# ─────────────────────────────────────────────────────────────────────────────
def load_institutional_deals():
    """Combine bulk_deals + block_deals into one normalized frame:
    ticker, deal_date, value_cr, is_buy, is_sell, deal_type, client_name."""
    frames = []
    for key in ("bulk_deals", "block_deals"):
        df = _safe_read(key)
        if df.empty:
            continue
        df = df.copy()
        df["deal_date"] = pd.to_datetime(df["deal_date"])
        cols = ["ticker", "deal_date", "value_cr", "is_buy", "is_sell", "deal_type", "client_name"]
        cols = [c for c in cols if c in df.columns]
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=["ticker", "deal_date", "value_cr", "is_buy", "is_sell", "deal_type", "client_name"])
    return pd.concat(frames, ignore_index=True, sort=False)


def institutional_flow_score(ticker, deals_df, as_of_date, lookback_days=30):
    """
    Net institutional flow score for `ticker` as of `as_of_date`, using
    bulk/block deal value_cr over the trailing `lookback_days` calendar days
    (window is (as_of_date - lookback_days, as_of_date], so it only looks
    backward from as_of_date — safe to call with as_of_date = end of a
    training window without leaking future deals).

    UNVALIDATED — see module docstring.
    """
    if deals_df is None or deals_df.empty:
        return np.nan
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(days=lookback_days)
    sub = deals_df[(deals_df["ticker"] == ticker) &
                    (deals_df["deal_date"] > window_start) &
                    (deals_df["deal_date"] <= as_of)]
    if sub.empty:
        return np.nan
    buy_val = sub.loc[sub["is_buy"], "value_cr"].sum()
    sell_val = sub.loc[sub["is_sell"], "value_cr"].sum()
    denom = buy_val + sell_val
    if denom <= 0:
        return np.nan
    return float((buy_val - sell_val) / denom)


def institutional_deal_count_tilt(ticker, deals_df, as_of_date, lookback_days=30):
    """Secondary, simpler cut: (buy deal count - sell deal count) over the
    trailing window, unweighted by value_cr. UNVALIDATED, same caveats."""
    if deals_df is None or deals_df.empty:
        return np.nan
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(days=lookback_days)
    sub = deals_df[(deals_df["ticker"] == ticker) &
                    (deals_df["deal_date"] > window_start) &
                    (deals_df["deal_date"] <= as_of)]
    if sub.empty:
        return np.nan
    return float(sub["is_buy"].sum() - sub["is_sell"].sum())


def make_institutional_flow_signal_fn(ticker, deals_df, lookback_days=30):
    """
    Factory producing a validation_framework-compatible signal_fn(tr_px) ->
    float for a single ticker, so this CAN be plugged into
    run_walkforward(..., signal_fn=...) once enough daily deal snapshots
    have accumulated on disk to form real (months-scale) folds.

    tr_px is the training-window price Series (index=Date); its LAST date
    (always <= the fold's train_end) is used as as_of_date for the deals
    lookup, so this stays leak-free by construction — it only ever looks at
    deals up to the end of the training window, exactly like every other
    signal_fn in this codebase.

    NOT wired into any actual run_walkforward(...) call anywhere in this
    file — that would require months of daily deal snapshots to build real
    folds, which do not exist yet (see module docstring). This factory
    exists purely so a future validation run needs zero rework.
    """
    def signal_fn(tr_px):
        if tr_px is None or len(tr_px) == 0:
            return np.nan
        as_of_date = tr_px.index[-1]
        return institutional_flow_score(ticker, deals_df, as_of_date, lookback_days)
    return signal_fn


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 2 — EARNINGS EVENT / PEAD-STYLE SCORE
#
# UNVALIDATED — insufficient historical data as of this build. earnings_dna
# / earnings_calendar / earnings_dna_historical are each a single
# current-snapshot pull of a FORWARD "next result date" calendar (no
# repeated dated archive), so there is no way to walk-forward test any
# earnings-proximity signal yet. earnings_dna_drift.parquet has a
# precomputed AvgDrift_D1_D10 per ticker but no date column of its own — it
# is consumed here as a prior, as-is, not re-derived or re-validated.
# ─────────────────────────────────────────────────────────────────────────────
def load_earnings_dna():
    df = _safe_read("earnings_dna")
    if df.empty:
        return df
    df = df.copy()
    df["next_result_date"] = pd.to_datetime(df["next_result_date"])
    return df


def load_earnings_drift():
    df = _safe_read("earnings_dna_drift")
    if df.empty:
        return df
    return df.rename(columns={"Ticker": "ticker"})


def earnings_event_score(ticker, earnings_dna_df, drift_df, as_of_date=None,
                          pre_earnings_window_days=5):
    """
    UNVALIDATED — insufficient historical data as of this build (see module
    docstring). Live-scoring logic only, ready to use once real forward
    history exists to test it against:

      1. Pre-earnings proximity: if `ticker` has 0 < days_to_result <=
         pre_earnings_window_days in earnings_dna.parquet, treat it as
         inside the pre-earnings run-up window (flag = 1), else 0.
      2. Historical drift prior: pulls AvgDrift_D1_D10 and ReliabilityFlag
         for `ticker` from earnings_dna_drift.parquet — a precomputed,
         un-dated stat carried over as-is from the live pipeline (NOT
         re-derived or re-validated here).
      3. score = sign(AvgDrift_D1_D10), but ONLY exposed (non-zero) when
         `ticker` is inside the pre-earnings window AND the live pipeline
         itself marked that ticker's drift stat ReliabilityFlag=True.
         Returns 0.0 when outside the window or unreliable, NaN when the
         ticker isn't in the earnings calendar at all (no known upcoming
         event to score).

    `as_of_date` is accepted for interface symmetry with the other score
    functions but is currently unused: earnings_dna.parquet's
    days_to_result is already computed relative to whenever the live
    pipeline last refreshed it (there is no dated history to recompute it
    against an arbitrary as_of_date).
    """
    if earnings_dna_df is None or earnings_dna_df.empty or "ticker" not in earnings_dna_df.columns:
        return np.nan
    matches = earnings_dna_df.loc[earnings_dna_df["ticker"] == ticker]
    if matches.empty:
        return np.nan
    row = matches.iloc[0]
    days_to_result = row.get("days_to_result", np.nan)
    if pd.isna(days_to_result) or not (0 < days_to_result <= pre_earnings_window_days):
        return 0.0
    if drift_df is None or drift_df.empty or "ticker" not in drift_df.columns:
        return np.nan
    dmatches = drift_df.loc[drift_df["ticker"] == ticker]
    if dmatches.empty:
        return np.nan
    drow = dmatches.iloc[0]
    if not bool(drow.get("ReliabilityFlag", False)):
        return 0.0
    drift = drow.get("AvgDrift_D1_D10", np.nan)
    if pd.isna(drift):
        return 0.0
    return float(np.sign(drift))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 3 — F&O POSITIONING (PCR / OI / Max Pain)
#
# UNVALIDATED — insufficient historical data as of this build.
# fno_pcr.parquet / fno_oi.parquet / fno_maxpain.parquet are each a single
# fetch_date snapshot (2026-07-13 at the time of this build, 69 tickers, 1
# unique date) — zero history to compute even one honest forward-return
# fold against. Live-scoring only.
# ─────────────────────────────────────────────────────────────────────────────
def load_fno_snapshot():
    """Outer-join PCR + OI + max-pain on ticker (all single-day snapshots as
    pulled — see module docstring). Returns empty df if none are available."""
    pcr = _safe_read("fno_pcr")
    oi = _safe_read("fno_oi")
    mp = _safe_read("fno_maxpain")
    if pcr.empty and oi.empty and mp.empty:
        return pd.DataFrame()

    def _cols(df, wanted):
        return [c for c in wanted if c in df.columns]

    df = pcr[_cols(pcr, ["ticker", "pcr", "pcr_sentiment", "fetch_date"])].copy() if not pcr.empty else pd.DataFrame(columns=["ticker"])
    if not oi.empty:
        df = df.merge(oi[_cols(oi, ["ticker", "net_oi_bias", "oi_signal"])], on="ticker", how="outer")
    if not mp.empty:
        df = df.merge(mp[_cols(mp, ["ticker", "max_pain_strike", "cmp", "max_pain_distance_pct", "pinning_risk"])], on="ticker", how="outer")
    return df


def fno_positioning_score(ticker, fno_df):
    """
    UNVALIDATED — insufficient historical data as of this build (see module
    docstring). score is roughly in [-1, +1]: positive = bullish-leaning
    positioning, negative = bearish-leaning positioning. Built as the
    unweighted mean of whichever of these are available for `ticker`:
      - pcr_sentiment / net_oi_bias / oi_signal, coerced via
        _sentiment_to_score (handles either string labels like 'Bullish' or
        already-numeric -1/0/+1 fields — the live pipeline's exact dtype for
        these columns wasn't independently verified here beyond schema)
      - sign(pcr - 1): PCR > 1 (more puts written than calls) is
        conventionally read as bullish/support-below
      - -sign(max_pain_distance_pct): price pulled toward max pain, so a
        positive distance (price above max pain) reads bearish and vice versa
    Returns NaN if `ticker` isn't in the snapshot or none of the above
    fields are populated for it.
    """
    if fno_df is None or fno_df.empty or "ticker" not in fno_df.columns:
        return np.nan
    matches = fno_df.loc[fno_df["ticker"] == ticker]
    if matches.empty:
        return np.nan
    row = matches.iloc[0]
    parts = []
    for col in ("pcr_sentiment", "net_oi_bias", "oi_signal"):
        if col in row.index:
            s = _sentiment_to_score(row[col])
            if pd.notna(s):
                parts.append(s)
    pcr = row.get("pcr", np.nan)
    if pd.notna(pcr):
        parts.append(float(np.sign(pcr - 1.0)))
    dist = row.get("max_pain_distance_pct", np.nan)
    if pd.notna(dist):
        parts.append(float(-np.sign(dist)))
    if not parts:
        return np.nan
    return float(np.mean(parts))


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE (convenience only) — simple mean of whichever of the three
# alt-data scores are available for a ticker as of a given date. UNVALIDATED,
# same caveats as every component above: this has not been backtested,
# standalone or combined.
# ─────────────────────────────────────────────────────────────────────────────
def altdata_composite_score(ticker, deals_df, earnings_dna_df, drift_df, fno_df, as_of_date=None):
    as_of_date = as_of_date or pd.Timestamp.today().normalize()
    parts = []
    flow = institutional_flow_score(ticker, deals_df, as_of_date)
    if pd.notna(flow):
        parts.append(flow)
    ev = earnings_event_score(ticker, earnings_dna_df, drift_df, as_of_date)
    if pd.notna(ev) and ev != 0.0:
        parts.append(ev)
    fno = fno_positioning_score(ticker, fno_df)
    if pd.notna(fno):
        parts.append(fno)
    if not parts:
        return np.nan
    return float(np.mean(parts))


if __name__ == "__main__":
    # Live-scoring smoke print only — NOT a validation/backtest run. None of
    # the signals above have been walk-forward tested (see module docstring);
    # this just demonstrates the scoring path against today's alt-data
    # snapshot for whichever tickers appear in the deal/earnings/FNO files.
    deals = load_institutional_deals()
    edna = load_earnings_dna()
    drift = load_earnings_drift()
    fno = load_fno_snapshot()

    tickers = sorted(
        set(deals["ticker"]) if "ticker" in deals.columns else set()
        | (set(edna["ticker"]) if "ticker" in edna.columns else set())
        | (set(fno["ticker"]) if "ticker" in fno.columns else set())
    )
    log.info("altdata snapshot loaded: %d deal tickers, %d earnings-calendar tickers, %d fno tickers",
             deals["ticker"].nunique() if "ticker" in deals.columns else 0,
             edna["ticker"].nunique() if "ticker" in edna.columns else 0,
             fno["ticker"].nunique() if "ticker" in fno.columns else 0)
    for t in tickers[:20]:
        score = altdata_composite_score(t, deals, edna, drift, fno)
        log.info("  %-15s composite=%s", t, "n/a" if pd.isna(score) else f"{score:+.3f}")
