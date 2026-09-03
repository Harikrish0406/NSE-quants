# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS — Live Signal Scoring

Computes TODAY's live signal value per ticker for every validated signal in
layer_swing_signals.py and layer_ml_ensemble.py, and writes them to
latest_signals.parquet in the (ticker, signal_name, value) format
layer_combiner.py already expects. This is the missing piece the
2026-07-14 full run flagged: validation was real, but nothing produced a
live score, so the combiner had 0 tickers to work with.

Reuses the EXACT SAME signal_fn / fit_predict_fn implementations that were
walk-forward validated — no reimplementation. Each single-ticker signal_fn
is called with the full available local price history as its argument
(there's no "test window" in live scoring — today IS the point being
scored, not a held-out historical fold).

KNOWN GAP — pairs_mean_reversion: this signal scores a PAIR, not a single
ticker. Consistent with how it was backtested, this writes one row keyed by
a synthetic "TICKERA_TICKERB" identifier, not a real tradeable ticker.
Actual pairs execution (long one leg, short the other) isn't implemented in
paper_tracker.py yet — treat these rows as informational only for now.

ML ensemble scoring calls fit_predict_fn() once, which fits both models on
the full current universe panel — the same real operation as one
walk-forward fold (~1 minute in testing, not instant), because it's an
actual model fit, not a cached lookup.

Run manually:  python live_scoring.py
"""
import os
import time

import numpy as np
import pandas as pd

from validation_framework import PROJECT_DIR, log, _get_cached_series
from layer_swing_signals import (
    load_universe_df,
    select_single_ticker_universe,
    build_sector_pairs,
    pairs_mean_reversion_signal,
    build_momentum_variants,
    build_vol_breakout_variants,
    zscore_mean_reversion_signal,
)
from layer_ml_ensemble import load_universe as ml_load_universe, fit_predict_fn

OUTPUT_PATH = os.path.join(PROJECT_DIR, "latest_signals.parquet")
MIN_HISTORY_DAYS = 60


def _full_history(ticker):
    """Full local price history for a ticker — this IS the scoring window
    for live signals, since there's no held-out future period when scoring
    'today'."""
    return _get_cached_series(ticker)


def score_single_ticker_signals(universe, signal_variants):
    """signal_variants: dict[name -> signal_fn(tr_px) -> float].
    Returns a list of (ticker, signal_name, value) rows."""
    rows = []
    for name, fn in signal_variants.items():
        t0 = time.time()
        n_scored = 0
        for ticker in universe:
            px = _full_history(ticker)
            if px is None or len(px) < MIN_HISTORY_DAYS:
                continue
            try:
                val = fn(px)
            except Exception as e:
                log.warning("  %s failed for %s: %s", name, ticker, e)
                continue
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            rows.append((ticker, name, float(val)))
            n_scored += 1
        log.info("  [%s] scored %d/%d tickers in %.1fs", name, n_scored, len(universe), time.time() - t0)
    return rows


def score_pairs_signal(pairs):
    """One row per pair, keyed by a synthetic 'TICKERA_TICKERB' id — see
    module docstring KNOWN GAP note before treating these as tradeable."""
    rows = []
    t0 = time.time()
    n_scored = 0
    for t1, t2 in pairs:
        px1 = _full_history(t1)
        px2 = _full_history(t2)
        if px1 is None or px2 is None or len(px1) < MIN_HISTORY_DAYS or len(px2) < MIN_HISTORY_DAYS:
            continue
        try:
            val = pairs_mean_reversion_signal(px1, px2)
        except Exception as e:
            log.warning("  pairs_mean_reversion failed for %s/%s: %s", t1, t2, e)
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        rows.append((f"{t1}_{t2}", "pairs_mean_reversion", float(val)))
        n_scored += 1
    log.info("  [pairs_mean_reversion] scored %d/%d pairs in %.1fs", n_scored, len(pairs), time.time() - t0)
    return rows


def score_ml_ensemble(universe):
    """Reuses fit_predict_fn directly — fits both models on the full
    current panel (same operation as one walk-forward fold), then returns
    predictions for the latest available point per ticker."""
    t0 = time.time()
    train_panel = {}
    for ticker in universe:
        px = _full_history(ticker)
        if px is not None and len(px) >= MIN_HISTORY_DAYS:
            train_panel[ticker] = px
    if len(train_panel) < 10:
        log.warning("  [ml_ensemble] too few tickers with history (%d) -- skipping", len(train_panel))
        return []
    try:
        predictions = fit_predict_fn(train_panel)
    except Exception as e:
        log.warning("  [ml_ensemble] fit_predict_fn failed: %s", e)
        return []
    rows = [
        (t, "ml_ensemble", float(v))
        for t, v in predictions.items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    log.info("  [ml_ensemble] scored %d tickers in %.1fs", len(rows), time.time() - t0)
    return rows


def main():
    log.info("=" * 72)
    log.info("CLAUDE QUANTS -- LIVE SIGNAL SCORING")
    log.info("=" * 72)
    overall_start = time.time()

    df = load_universe_df()
    single_universe = select_single_ticker_universe(df)
    pairs = build_sector_pairs(df, per_sector=20)

    all_rows = []

    log.info("Scoring momentum variants (10)...")
    all_rows += score_single_ticker_signals(single_universe, build_momentum_variants())

    log.info("Scoring vol-breakout variants (8)...")
    all_rows += score_single_ticker_signals(single_universe, build_vol_breakout_variants())

    log.info("Scoring zscore_mean_reversion_20d...")
    all_rows += score_single_ticker_signals(
        single_universe, {"zscore_mean_reversion_20d": zscore_mean_reversion_signal}
    )

    log.info("Scoring pairs_mean_reversion (%d candidate pairs)...", len(pairs))
    all_rows += score_pairs_signal(pairs)

    log.info("Scoring ml_ensemble (fits both models on current panel, ~1 min)...")
    ml_universe, _ml_tier_map, _ml_sector_map = ml_load_universe()
    all_rows += score_ml_ensemble(ml_universe)

    out = pd.DataFrame(all_rows, columns=["ticker", "signal_name", "value"])
    out.to_parquet(OUTPUT_PATH, index=False)

    elapsed = time.time() - overall_start
    log.info("=" * 72)
    log.info(
        "Wrote %d rows (%d unique ticker/pair keys, %d signals) -> %s",
        len(out), out["ticker"].nunique(), out["signal_name"].nunique(), OUTPUT_PATH,
    )
    log.info("Total time: %.1f min", elapsed / 60)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
