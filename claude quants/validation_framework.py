# -*- coding: utf-8 -*-
# CLAUDE QUANTS — Leak-Free Walk-Forward Validation Framework
#
# WHY THIS FILE EXISTS:
# The original layer1_5_validation.py had a real lookahead bug in its pairs-trading
# signal (confirmed via code review): the entry z-score was normalized using
# mean/std computed over the ENTIRE test window, which includes future data the
# signal shouldn't be able to see at entry time. That's why it scored a suspicious
# Sharpe of 7.7 / IC t-stat of 60 — not a real edge, a leak.
#
# This framework fixes that by construction, not just by convention: every
# signal_fn here receives ONLY the training-window price data as its argument.
# It structurally cannot see test-window data, because test-window data is never
# passed to it. Forward returns are computed separately, downstream, from data
# the signal function never touches.
#
# Reused as-is from the original (confirmed leak-free, no changes needed):
# generate_folds, compute_ic, simulate_pnl, make_row, WF_CONFIGS,
# FORWARD_RETURN_DAYS, SLIPPAGE — same logic, copied here so this replica has
# zero import dependency on the original pipeline's files.

import os
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ValidationFramework")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (identical to original — same fold structure, same horizons)
# ─────────────────────────────────────────────────────────────────────────────
WF_CONFIGS = {
    "intraday": {"train_months": 6,  "test_months": 1},
    "swing":    {"train_months": 12, "test_months": 3},
    "long":     {"train_months": 24, "test_months": 6},
}
FORWARD_RETURN_DAYS = {"intraday": 1, "swing": 10, "long": 60}

SLIPPAGE = {
    "T1": {"swing": 0.0015, "long": 0.0015, "intraday": 0.0008},
    "T2": {"swing": 0.0025, "long": 0.0025, "intraday": 0.0013},
    "T3": {"swing": 0.0050, "long": 0.0050, "intraday": 0.0025},
}

def get_slippage(tier, horizon):
    t = tier if tier in SLIPPAGE else "T3"
    h = horizon if horizon in SLIPPAGE["T1"] else "swing"
    return SLIPPAGE[t][h]

# ─────────────────────────────────────────────────────────────────────────────
# FOLD GENERATION (unchanged from original — this part was never the bug)
# ─────────────────────────────────────────────────────────────────────────────
def generate_folds(wf_type, start_year=2010, end_year=2026):
    cfg = WF_CONFIGS[wf_type]
    folds, cursor, end = [], datetime(start_year, 1, 1), datetime(end_year, 1, 1)
    while True:
        tr_s = cursor
        tr_e = cursor + relativedelta(months=cfg["train_months"])
        te_s = tr_e
        te_e = te_s + relativedelta(months=cfg["test_months"])
        if te_e > end:
            break
        folds.append((tr_s, tr_e, te_s, te_e))
        cursor += relativedelta(months=cfg["test_months"])
    return folds

# ─────────────────────────────────────────────────────────────────────────────
# PRICE LOADING — reads from local data/raw/*.parquet cache first (fast, no
# network), falls back to yfinance only for tickers not in the cache.
#
# PERF NOTE (added for the full-universe / multi-variant swing-signal re-run):
# the original _PRICE_CACHE was keyed by (tickers, start, end), which meant a
# 100%-identical disk read was repeated for every walk-forward fold (each fold
# has a different (start,end)) and again for every additional signal_fn run
# over the same universe (e.g. testing 5 momentum lookbacks back-to-back).
# With ~1550 tickers x 60 folds x several signal variants that was tens of
# thousands of redundant parquet reads. _TICKER_CACHE below caches each
# ticker's FULL local price history ONCE (keyed by ticker only, not by date
# range), and fetch_prices() just slices the in-memory Series per call. This
# is a pure I/O optimization: it changes nothing about what data reaches
# signal_fn (still exactly the [start,end]-sliced window, same as before),
# so the leak-free contract is untouched.
# ─────────────────────────────────────────────────────────────────────────────
_TICKER_CACHE = {}

def _load_local(ticker):
    path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")["Close"].sort_index()
    except Exception:
        return None

def _get_cached_series(ticker):
    if ticker not in _TICKER_CACHE:
        _TICKER_CACHE[ticker] = _load_local(ticker)
    return _TICKER_CACHE[ticker]

def fetch_prices(tickers, start, end, max_tickers=2500):
    """Returns a DataFrame: index=Date, columns=tickers, values=Close price."""
    tickers = list(tickers)[:max_tickers]

    series = {}
    missing = []
    for t in tickers:
        s = _get_cached_series(t)
        if s is not None:
            series[t] = s
        else:
            missing.append(t)

    if missing:
        try:
            import yfinance as yf
            ns_tickers = [t + ".NS" for t in missing]
            raw = yf.download(ns_tickers, start=start - timedelta(days=5), end=end + timedelta(days=5),
                               auto_adjust=True, progress=False, threads=True)
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            prices.columns = [c.replace(".NS", "") for c in prices.columns]
            for t in prices.columns:
                s = prices[t].dropna()
                series[t] = s
                _TICKER_CACHE[t] = s  # cache the fallback fetch too, ticker-keyed
        except Exception as e:
            log.warning("yfinance fallback failed for %d missing tickers: %s", len(missing), e)

    if not series:
        return pd.DataFrame()
    result = pd.DataFrame(series)
    result = result.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna(axis=1, how="all")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SCORING UTILS (unchanged from original — confirmed leak-free)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic(sig, fwd):
    df = pd.DataFrame({"s": sig, "f": fwd}).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    return spearmanr(df["s"], df["f"])

def simulate_pnl(sig, fwd, tiers, horizon):
    df = pd.DataFrame({"s": sig, "f": fwd, "t": tiers}).dropna(subset=["s", "f"])
    if len(df) < 10:
        return np.nan
    longs  = df[df["s"] >= df["s"].quantile(0.75)]
    shorts = df[df["s"] <= df["s"].quantile(0.25)]
    def net(rows, d):
        if len(rows) == 0:
            return 0.0
        return rows["f"].mean() * d - rows["t"].apply(lambda t: get_slippage(t, horizon)).mean()
    return net(longs, 1) + net(shorts, -1)

def make_row(signal, i, tr_s, te_s, ic, pval, pnl, n, wf_type="swing"):
    return {"signal": signal, "fold": i, "train_start": tr_s, "test_start": te_s,
            "ic": ic, "pval": pval, "pnl": pnl, "n_stocks": n, "wf_type": wf_type}

# ─────────────────────────────────────────────────────────────────────────────
# THE CORE FIX: generic, leak-free walk-forward runner.
#
# signal_fn is called with ONLY training-window data. It never receives test-
# window data as an argument, so it is structurally impossible for it to
# normalize (mean/std/etc.) against future information — the exact bug found
# in the original pairs signal.
#
# Single-ticker contract:
#   signal_fn(tr_prices: pd.Series) -> float
#
# Pairs contract (is_pairs=True):
#   signal_fn(tr_prices_1: pd.Series, tr_prices_2: pd.Series) -> float
#   (beta, spread mean/std — everything — must be computed inside signal_fn
#    from tr_prices_1/tr_prices_2 only, since that's all it's given)
# ─────────────────────────────────────────────────────────────────────────────
def run_walkforward(name, wf_type, universe, signal_fn, tier_map,
                     is_pairs=False, pairs=None, start_year=2010, end_year=2026,
                     min_train_days=60):
    folds = generate_folds(wf_type, start_year, end_year)
    horizon = wf_type
    fd = FORWARD_RETURN_DAYS[wf_type]
    rows = []

    items = pairs if is_pairs else list(universe)

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        flat_universe = set()
        if is_pairs:
            for a, b in items:
                flat_universe.add(a); flat_universe.add(b)
        else:
            flat_universe.update(items)

        prices = fetch_prices(flat_universe, tr_s, te_e)
        if prices.empty:
            continue

        signals, fwd_rets, tiers = {}, {}, {}

        if is_pairs:
            for t1, t2 in items:
                if t1 not in prices.columns or t2 not in prices.columns:
                    continue
                tr_px1 = prices.loc[(prices.index >= tr_s) & (prices.index < tr_e), t1].dropna()
                tr_px2 = prices.loc[(prices.index >= tr_s) & (prices.index < tr_e), t2].dropna()
                if len(tr_px1) < min_train_days or len(tr_px2) < min_train_days:
                    continue

                # signal_fn sees ONLY training-window data for both legs — cannot leak
                try:
                    sig_val = signal_fn(tr_px1, tr_px2)
                except Exception:
                    continue
                if sig_val is None or (isinstance(sig_val, float) and np.isnan(sig_val)):
                    continue

                # forward return computed independently, downstream, from test-window
                # data the signal function never saw
                te_px1 = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t1].dropna()
                te_px2 = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t2].dropna()
                cte = te_px1.index.intersection(te_px2.index)
                if len(cte) < 2:
                    continue
                beta = np.polyfit(tr_px2.reindex(tr_px1.index.intersection(tr_px2.index)),
                                   tr_px1.reindex(tr_px1.index.intersection(tr_px2.index)), 1)[0]
                te_spread = te_px1.loc[cte] - beta * te_px2.loc[cte]
                fwd_spread = te_spread.iloc[min(fd, len(te_spread) - 1)] - te_spread.iloc[0]

                key = f"{t1}_{t2}"
                signals[key] = sig_val
                fwd_rets[key] = fwd_spread
                tiers[key] = tier_map.get(t1, "T3")
        else:
            for t in items:
                if t not in prices.columns:
                    continue
                tr_px = prices.loc[(prices.index >= tr_s) & (prices.index < tr_e), t].dropna()
                if len(tr_px) < min_train_days:
                    continue

                # signal_fn sees ONLY training-window data — cannot leak
                try:
                    sig_val = signal_fn(tr_px)
                except Exception:
                    continue
                if sig_val is None or (isinstance(sig_val, float) and np.isnan(sig_val)):
                    continue

                # forward return computed independently, downstream
                te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t].dropna()
                if len(te_px) < 2:
                    continue
                entry = te_px.iloc[0]
                exit_ = te_px.iloc[min(fd, len(te_px) - 1)]
                fwd_ret = (exit_ - entry) / entry

                signals[t] = sig_val
                fwd_rets[t] = fwd_ret
                tiers[t] = tier_map.get(t, "T3")

        if len(signals) < 10:
            continue

        sig_series = pd.Series(signals)
        fwd_series = pd.Series(fwd_rets)
        tier_series = pd.Series(tiers)

        ic, pval = compute_ic(sig_series, fwd_series)
        pnl = simulate_pnl(sig_series, fwd_series, tier_series, horizon)
        rows.append(make_row(name, i, tr_s, te_s, ic, pval, pnl, len(signals), wf_type=wf_type))
        log.info("    %s fold %d/%d | IC=%.3f | PnL=%.4f | n=%d",
                 name, i + 1, len(folds), ic or 0, pnl or 0, len(signals))

    return pd.DataFrame(rows)


def run_walkforward_ml(name, wf_type, universe, fit_predict_fn, tier_map,
                        start_year=2010, end_year=2026, min_train_days=60):
    """
    For ML models that need to fit ONCE per fold across the whole cross-sectional
    training panel (not per-ticker like run_walkforward's simple signal_fn).

    fit_predict_fn(train_panel: dict[ticker -> pd.Series]) -> dict[ticker -> float]
      Receives ONLY training-window price series for every ticker in the fold.
      Must fit and predict entirely from this — no test-window data is ever
      passed in, so it's structurally impossible to leak into the fit.
      Feature engineering happens inside fit_predict_fn; keep every feature
      derived only from the training_window series it's given per ticker.

    Forward returns (the label used to score predictions) are computed
    downstream from test-window data the fit_predict_fn never sees, exactly
    like run_walkforward.
    """
    folds = generate_folds(wf_type, start_year, end_year)
    horizon = wf_type
    fd = FORWARD_RETURN_DAYS[wf_type]
    rows = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(universe, tr_s, te_e)
        if prices.empty:
            continue

        train_panel = {}
        for t in prices.columns:
            tr_px = prices.loc[(prices.index >= tr_s) & (prices.index < tr_e), t].dropna()
            if len(tr_px) >= min_train_days:
                train_panel[t] = tr_px

        if len(train_panel) < 10:
            continue

        try:
            predictions = fit_predict_fn(train_panel)
        except Exception as e:
            log.warning("    %s fold %d fit_predict_fn failed: %s", name, i + 1, e)
            continue
        if not predictions:
            continue

        signals, fwd_rets, tiers = {}, {}, {}
        for t, sig_val in predictions.items():
            if sig_val is None or (isinstance(sig_val, float) and np.isnan(sig_val)):
                continue
            if t not in prices.columns:
                continue
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t].dropna()
            if len(te_px) < 2:
                continue
            entry = te_px.iloc[0]
            exit_ = te_px.iloc[min(fd, len(te_px) - 1)]
            signals[t] = sig_val
            fwd_rets[t] = (exit_ - entry) / entry
            tiers[t] = tier_map.get(t, "T3")

        if len(signals) < 10:
            continue

        sig_series = pd.Series(signals)
        fwd_series = pd.Series(fwd_rets)
        tier_series = pd.Series(tiers)
        ic, pval = compute_ic(sig_series, fwd_series)
        pnl = simulate_pnl(sig_series, fwd_series, tier_series, horizon)
        rows.append(make_row(name, i, tr_s, te_s, ic, pval, pnl, len(signals), wf_type=wf_type))
        log.info("    %s fold %d/%d | IC=%.3f | PnL=%.4f | n=%d",
                 name, i + 1, len(folds), ic or 0, pnl or 0, len(signals))

    return pd.DataFrame(rows)


def build_verdict(report_df):
    """Same PASS/WEAK/FAIL criteria as the original: IC t-stat > 1.5 AND IC+ > 55% AND PnL+ > 55%."""
    out = []
    for signal, g in report_df.groupby("signal"):
        g = g.dropna(subset=["ic"])
        if len(g) < 5:
            continue
        mean_ic = g["ic"].mean()
        ic_t = mean_ic / (g["ic"].std() / np.sqrt(len(g)) + 1e-9)
        ic_pos = (g["ic"] > 0).mean() * 100
        pnl_pos = (g["pnl"] > 0).mean() * 100
        tot_pnl = g["pnl"].sum()
        row_wf_type = g["wf_type"].iloc[0] if "wf_type" in g.columns else "swing"
        sharpe = g["pnl"].mean() / (g["pnl"].std() + 1e-9) * np.sqrt(252 / FORWARD_RETURN_DAYS.get(row_wf_type, 10))
        verdict = "PASS" if (ic_t > 1.5 and ic_pos > 55 and pnl_pos > 55) else \
                  ("WEAK" if (ic_t > 1.0 or (ic_pos > 50 and pnl_pos > 50)) else "FAIL")
        out.append({"signal": signal, "folds": len(g), "mean_ic": mean_ic, "ic_t": ic_t,
                     "ic_pos_pct": ic_pos, "total_pnl": tot_pnl, "pnl_pos_pct": pnl_pos,
                     "sharpe": sharpe, "verdict": verdict})
    return pd.DataFrame(out)
