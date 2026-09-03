# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS — Swing Signal Layer (layer_swing_signals.py)

Standalone replica. Every signal here is validated through the shared
leak-free walk-forward framework in validation_framework.py (run_walkforward),
which passes signal_fn ONLY training-window price data — it is structurally
impossible for these signals to see test-window data, the exact class of bug
that was found and fixed in the original pairs-trading signal (entry z-score
normalized against test-window mean/std).

FIRST-PASS RESULTS (200-ticker cap, since removed from validation_framework):

  | Signal                     | Verdict | mean IC | Notes                                  |
  |-----------------------------|---------|---------|-----------------------------------------|
  | zscore_mean_reversion_20d   | PASS    | 0.059   | Working well, left as-is                |
  | pairs_mean_reversion        | WEAK    | 0.031   | t-stat cleared 1.5, pnl_pos_pct exactly |
  |                              |         |         | 50%, just below the 55% bar             |
  | short_momentum_20d          | FAIL    | -0.049  | Negative IC -> real reversal, not noise |
  | vol_breakout                | FAIL    | -0.063  | Also negative IC                        |

SECOND PASS (this revision) — multi-variant search + full universe:

  1. pairs_mean_reversion   — candidate pool expanded (per_sector 6 -> 20, ~2000
                               candidate pairs vs. ~150 before) AND pair validity
                               is now decided by an Augmented Dickey-Fuller test
                               on the training-window spread (statsmodels
                               adfuller, maxlag=5, autolag="AIC") instead of just
                               "same sector, top-N liquidity". A pair only emits
                               a signal in folds where its training-window spread
                               is stationary (ADF p-value < 0.05); beta, the ADF
                               test itself, and the z-score mean/std are ALL
                               computed from tr_px1/tr_px2 only, inside
                               signal_fn — never test-window data.
  2. short_momentum_*        — the single 20d-continuation signal_fn is replaced
                               by 10 variants: 5 lookbacks (5/10/20/40/60d) x
                               2 directions (continuation, and sign-flipped
                               reversal). The first-pass 20d continuation signal
                               had IC -0.049 — a real negative Spearman
                               correlation, not noise — so the flipped
                               (reversal) versions are the actual hypothesis
                               under test here, run alongside the raw
                               continuation versions and the other lookbacks so
                               the walk-forward decides which combination(s)
                               pass, rather than assuming 20d-flip is the fix.
  3. vol_breakout_*           — the single 20d breakout-continuation signal_fn
                               is replaced by 8 variants: 3 realized-vol/SMA
                               windows (10/20/40d) x 2 framings (breakout-
                               continuation vs. sign-flipped mean-reversion /
                               "fade the spike"), plus 2 threshold-gated
                               variants (20d mean-reversion framing, only fires
                               when |breakout score| clears 0.5 / 1.0 band-
                               widths — filters out low-conviction noise that a
                               continuous score can't). First pass had IC
                               -0.063 on the continuation framing, so — same
                               logic as momentum — the flipped/meanrev variants
                               are the live hypothesis.
  4. zscore_mean_reversion_20d — unchanged. Already PASS in the first pass;
                               left exactly as-is per the "leave as-is"
                               instruction.

  NOTE ON PRUNING: this revision intentionally VALIDATES ALL VARIANTS above —
  it does not pre-guess winners. Whichever variant(s) actually clear
  build_verdict()'s PASS bar (IC t-stat>1.5 AND IC+>55% AND PnL+>55%) in the
  real run should be kept; the rest should be deleted or clearly marked FAIL
  in a follow-up edit once real results are in. This file was NOT executed
  end-to-end as part of writing this revision (see run notes at the bottom of
  this docstring / the calling task) — do not treat any IC numbers in this
  docstring beyond the first-pass table as real.

Candidate pairs for #1: no spread_history.parquet exists in this folder, so
candidate pairs are still built directly from universe_master.parquet — top-N
by AvgDailyValue within each same-sector group (excludes "Unknown" sector,
requires long trading history so pairs have enough folds to be tested). Same-
sector grouping is kept as an economically-sensible prior (same-sector names
are more plausible cointegration candidates); what changed is per_sector
(6 -> 20) and that ADF cointegration, not liquidity rank, now gates whether a
pair actually trades in a given fold.

Run: `python layer_swing_signals.py`
Outputs (written to this folder, NOT the parent production pipeline):
  - swing_signals_validation.parquet   (raw per-fold report, all signal variants)
  - swing_signals_validation_report.txt (verdict table + run metadata)
"""

import os
import time
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from validation_framework import (
    run_walkforward, build_verdict, PROJECT_DIR, RAW_DIR, log,
)

UNIV_PATH = os.path.join(PROJECT_DIR, "universe_master.parquet")
OUT_PARQUET = os.path.join(PROJECT_DIR, "swing_signals_validation.parquet")
OUT_TXT = os.path.join(PROJECT_DIR, "swing_signals_validation_report.txt")

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE / TIER MAP / CANDIDATE PAIRS
# ─────────────────────────────────────────────────────────────────────────────
def load_universe_df():
    df = pd.read_parquet(UNIV_PATH)
    df["DataStart"] = pd.to_datetime(df["DataStart"])
    df["DataEnd"] = pd.to_datetime(df["DataEnd"])
    # Only keep tickers that actually have a local OHLCV cache file (offline-safe run)
    cache_tickers = {f[:-8] for f in os.listdir(RAW_DIR) if f.endswith(".parquet")}
    df = df[df["Ticker"].isin(cache_tickers)].copy()
    return df


def build_tier_map(df):
    return dict(zip(df["Ticker"], df["Tier"]))


def select_single_ticker_universe(df):
    """FULL universe (no cap) — every ticker with a local OHLCV cache, sorted
    by liquidity (order is cosmetic; run_walkforward just iterates the list)."""
    return df.sort_values("AvgDailyValue", ascending=False)["Ticker"].tolist()


def build_sector_pairs(df, per_sector=20, long_history_cutoff="2012-06-01", exclude_sector="Unknown"):
    """Candidate pairs = all combinations of the top-`per_sector` most liquid,
    long-history tickers within each sector. per_sector raised 6 -> 20 now that
    the 200-ticker universe cap is gone (was the binding constraint before).
    No spread_history.parquet is present in this folder, so this liquidity/
    sector prefilter is still how the CANDIDATE POOL is built — actual pair
    validity per fold is decided inside pairs_mean_reversion_signal via ADF
    cointegration on training-window data, not by this liquidity rank."""
    d = df[(df["DataStart"] <= pd.Timestamp(long_history_cutoff)) & (df["Sector"] != exclude_sector)]
    pairs = []
    for sector, g in d.groupby("Sector"):
        top = g.sort_values("AvgDailyValue", ascending=False)["Ticker"].head(per_sector).tolist()
        pairs.extend(itertools.combinations(top, 2))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — PAIRS MEAN REVERSION (ADF-cointegration gated, leak-free)
# beta, the ADF stationarity test, and the spread mean/std are ALL computed
# from tr_px1/tr_px2 ONLY (training-window data passed to signal_fn — never
# test-window data). A pair only emits a signal in folds where its training-
# window spread tests stationary (ADF p-value < adf_pvalue_max); otherwise it
# returns NaN and run_walkforward skips it for that fold, same as any other
# "not enough data" case.
#
# Sign convention unchanged from the original (pre-leak-bug) validate_pairs:
# signal = -zscore of the spread's last training-window point (fade the
# stretch).
# ─────────────────────────────────────────────────────────────────────────────
def pairs_mean_reversion_signal(tr_px1, tr_px2, adf_pvalue_max=0.05, adf_maxlag=5):
    common = tr_px1.index.intersection(tr_px2.index)
    if len(common) < 60:
        return np.nan
    p1 = tr_px1.loc[common]
    p2 = tr_px2.loc[common]
    try:
        beta = np.polyfit(p2.values, p1.values, 1)[0]
    except Exception:
        return np.nan
    spread = p1 - beta * p2
    sigma = spread.std()
    if not np.isfinite(sigma) or sigma < 1e-9:
        return np.nan

    # ADF cointegration test on the TRAINING-WINDOW spread only — this is the
    # gate that replaces "same-sector top-6-by-liquidity" pair selection.
    try:
        adf_result = adfuller(spread.values, maxlag=adf_maxlag, autolag="AIC")
        adf_pval = adf_result[1]
    except Exception:
        return np.nan
    if not np.isfinite(adf_pval) or adf_pval >= adf_pvalue_max:
        return np.nan  # spread not stationary this fold -> not a tradeable pair

    mu = spread.mean()
    z = (spread.iloc[-1] - mu) / sigma
    return float(-z)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 2 — SHORT-HORIZON MOMENTUM (multi-lookback, multi-direction)
# Factory builds one signal_fn per (lookback, flip) combination. All inputs
# come from tr_px only (training-window data). `flip=True` sign-flips the raw
# lookback return — a reversal framing, since the first-pass 20d continuation
# signal showed a real negative IC (-0.049) at this horizon, i.e. reversal,
# not continuation, was actually happening.
# ─────────────────────────────────────────────────────────────────────────────
def make_short_momentum_signal(lookback, flip=False):
    def _signal(tr_px):
        if len(tr_px) < lookback + 5:
            return np.nan
        p_now = tr_px.iloc[-1]
        p_then = tr_px.iloc[-lookback - 1]
        if p_then <= 0:
            return np.nan
        mom = p_now / p_then - 1.0
        return float(-mom if flip else mom)
    _signal.__name__ = f"short_momentum_{lookback}d{'_flip' if flip else ''}_fn"
    return _signal


MOMENTUM_LOOKBACKS = [5, 10, 20, 40, 60]


def build_momentum_variants():
    """Returns {signal_name: signal_fn} for every lookback x direction combo."""
    variants = {}
    for lb in MOMENTUM_LOOKBACKS:
        variants[f"short_momentum_{lb}d"] = make_short_momentum_signal(lb, flip=False)
        variants[f"short_momentum_{lb}d_flip"] = make_short_momentum_signal(lb, flip=True)
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 3 — VOLATILITY BREAKOUT (multi-window, multi-framing, threshold-gated)
# Factory builds one signal_fn per (lookback, framing, threshold) combination.
# Base score unchanged from first pass: distance of the last training close
# from its `lookback`-day mean, scaled by a realized-vol price band
# (`vol_lookback`-day return std, scaled to a `lookback`-day price move). All
# inputs come from tr_px only. GARCH(1,1) was benchmarked at ~1.4s/fit — at
# full-universe x ~60 folds x this many variants that's many hours, so this
# still uses the framework's explicitly-sanctioned realized-vol fallback.
#
# `flip=True`  -> mean-reversion framing: fade the spike (first pass's IC of
#                 -0.063 on the continuation framing is the evidence this may
#                 actually work).
# `threshold`  -> if set, only emit a signal when |breakout score| clears the
#                 threshold (in band-widths); otherwise NaN (that ticker-fold
#                 is skipped). Filters low-conviction/noisy near-zero cases
#                 that a continuous score can't — a real "is this an actual
#                 breakout/spike event" gate, distinct from the window choice.
# ─────────────────────────────────────────────────────────────────────────────
def make_vol_breakout_signal(lookback=20, vol_lookback=None, flip=False, threshold=None):
    vol_lookback = vol_lookback or lookback
    def _signal(tr_px):
        min_len = max(lookback, vol_lookback) + 25
        if len(tr_px) < min_len:
            return np.nan
        ret = tr_px.pct_change().dropna()
        if len(ret) < vol_lookback + 5:
            return np.nan
        realized_vol = ret.rolling(vol_lookback).std().iloc[-1]
        if not np.isfinite(realized_vol) or realized_vol < 1e-6:
            return np.nan
        sma = tr_px.rolling(lookback).mean().iloc[-1]
        if not np.isfinite(sma) or sma <= 0:
            return np.nan
        band = sma * realized_vol * np.sqrt(lookback)
        if band < 1e-9:
            return np.nan
        breakout = (tr_px.iloc[-1] - sma) / band
        if threshold is not None and abs(breakout) < threshold:
            return np.nan
        return float(-breakout if flip else breakout)
    name = f"vol_breakout_L{lookback}_V{vol_lookback}"
    if flip:
        name += "_meanrev"
    if threshold is not None:
        name += f"_thr{threshold}"
    _signal.__name__ = name + "_fn"
    return _signal


VOL_BREAKOUT_WINDOWS = [10, 20, 40]


def build_vol_breakout_variants():
    """Returns {signal_name: signal_fn}: 3 windows x 2 framings, plus 2
    threshold-gated variants on the 20d mean-reversion framing."""
    variants = {}
    for w in VOL_BREAKOUT_WINDOWS:
        variants[f"vol_breakout_{w}d"] = make_vol_breakout_signal(lookback=w, vol_lookback=w, flip=False)
        variants[f"vol_breakout_{w}d_meanrev"] = make_vol_breakout_signal(lookback=w, vol_lookback=w, flip=True)
    for thr in (0.5, 1.0):
        variants[f"vol_breakout_20d_meanrev_thr{thr}"] = make_vol_breakout_signal(
            lookback=20, vol_lookback=20, flip=True, threshold=thr,
        )
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 4 — SHORT-TERM MEAN REVERSION (single-stock, not pairs)
# UNCHANGED from first pass — already PASS (mean IC 0.059), left as-is.
# Z-score of price vs. its own 20-day moving average, computed from tr_px
# only. Signal = -zscore (fade the extension vs. own recent history).
# ─────────────────────────────────────────────────────────────────────────────
def zscore_mean_reversion_signal(tr_px, lookback=20):
    if len(tr_px) < lookback + 5:
        return np.nan
    mu = tr_px.rolling(lookback).mean().iloc[-1]
    sigma = tr_px.rolling(lookback).std().iloc[-1]
    if not np.isfinite(sigma) or sigma < 1e-9:
        return np.nan
    z = (tr_px.iloc[-1] - mu) / sigma
    return float(-z)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    log.info("Loading universe_master.parquet + local cache list ...")
    df = load_universe_df()
    tier_map = build_tier_map(df)

    single_universe = select_single_ticker_universe(df)   # FULL universe, no cap
    pairs = build_sector_pairs(df, per_sector=20)          # expanded candidate pool
    unique_pair_tickers = sorted(set(t for p in pairs for t in p))

    momentum_variants = build_momentum_variants()
    vol_breakout_variants = build_vol_breakout_variants()

    log.info("Single-ticker universe: %d tickers", len(single_universe))
    log.info("Candidate pairs: %d pairs across sectors, %d unique tickers",
              len(pairs), len(unique_pair_tickers))
    log.info("Momentum variants: %d", len(momentum_variants))
    log.info("Vol-breakout variants: %d", len(vol_breakout_variants))

    total_signals = 1 + len(momentum_variants) + len(vol_breakout_variants) + 1
    reports = []
    done = 0

    done += 1
    log.info("=== [%d/%d] pairs_mean_reversion (ADF-gated) ===", done, total_signals)
    r = run_walkforward(
        "pairs_mean_reversion", "swing", [], pairs_mean_reversion_signal,
        tier_map, is_pairs=True, pairs=pairs,
    )
    log.info("    -> %d folds scored", len(r))
    reports.append(r)

    for name, fn in momentum_variants.items():
        done += 1
        log.info("=== [%d/%d] %s ===", done, total_signals, name)
        r = run_walkforward(name, "swing", single_universe, fn, tier_map)
        log.info("    -> %d folds scored", len(r))
        reports.append(r)

    for name, fn in vol_breakout_variants.items():
        done += 1
        log.info("=== [%d/%d] %s ===", done, total_signals, name)
        r = run_walkforward(name, "swing", single_universe, fn, tier_map)
        log.info("    -> %d folds scored", len(r))
        reports.append(r)

    done += 1
    log.info("=== [%d/%d] zscore_mean_reversion_20d (unchanged, already PASS) ===", done, total_signals)
    r = run_walkforward(
        "zscore_mean_reversion_20d", "swing", single_universe, zscore_mean_reversion_signal, tier_map,
    )
    log.info("    -> %d folds scored", len(r))
    reports.append(r)

    report_df = pd.concat(reports, ignore_index=True)
    verdict_df = build_verdict(report_df)

    report_df.to_parquet(OUT_PARQUET, index=False)

    elapsed = time.time() - t_start
    lines = []
    lines.append("=" * 78)
    lines.append("CLAUDE QUANTS — SWING SIGNAL LAYER — VALIDATION REPORT (2nd pass, full universe)")
    lines.append("=" * 78)
    lines.append(f"Run timestamp   : {pd.Timestamp.now()}")
    lines.append(f"Elapsed         : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    lines.append(f"Single universe : {len(single_universe)} tickers (full, no cap)")
    lines.append(f"Candidate pairs : {len(pairs)} pairs / {len(unique_pair_tickers)} unique tickers (ADF-gated per fold)")
    lines.append(f"Momentum variants   : {len(momentum_variants)} ({', '.join(momentum_variants.keys())})")
    lines.append(f"Vol-breakout variants: {len(vol_breakout_variants)} ({', '.join(vol_breakout_variants.keys())})")
    lines.append(f"WF config       : swing (12mo train / 3mo test), forward=10 trading days")
    lines.append("")
    lines.append("VERDICT (PASS: IC t-stat>1.5 AND IC+>55% AND PnL+>55%)")
    lines.append("-" * 78)
    lines.append(verdict_df.to_string(index=False))
    lines.append("")
    lines.append("Per-signal fold counts (raw report):")
    lines.append(report_df.groupby("signal").size().to_string())
    txt = "\n".join(lines)

    with open(OUT_TXT, "w") as f:
        f.write(txt)

    print()
    print(txt)
    log.info("Done in %.1fs. Wrote %s and %s", elapsed, OUT_PARQUET, OUT_TXT)


if __name__ == "__main__":
    main()
