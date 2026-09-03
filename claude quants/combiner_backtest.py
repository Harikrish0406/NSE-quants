# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS -- Combiner-Level Backtest (added 2026-07-18)

WHY THIS FILE EXISTS
---------------------
layer_swing_signals.py and layer_ml_ensemble.py walk-forward validated each
signal INDIVIDUALLY. But the thing that actually decides FIRE/WAIT/KILL day
to day is layer_combiner.py's assembled composite score (SIGNAL_GROUPS
two-stage aggregation + performance weights + threshold) -- and that
assembled decision layer had never itself been tested against history. A
ticker could clear every individual signal's bar and still have the
COMBINATION behave differently than any one ingredient (that's the whole
point of combining signals). This file closes that gap: same leak-free
per-fold contract as run_walkforward (every signal_fn / fit_predict_fn call
below receives ONLY that fold's training-window price data), but the score
being tested is the live combiner's composite, not any single signal.

CAVEAT -- read before trusting the verdict
--------------------------------------------
compute_signal_weights() weights are FIXED from the full 2010-2025 Round 2
validation (the same weights currently live in combiner_output.parquet) and
applied UNIFORMLY to every fold here, including folds years before that
validation run existed. Deciding "which signals survive and how much they're
weighted" with full-sample hindsight, then testing across the whole sample,
is a real methodological optimism -- this is NOT a fully nested walk-forward
re-derivation of signal selection itself. What IS leak-free and genuinely
tested: every signal VALUE and the ML model FIT going into each fold's
composite score is computed from that fold's training-window prices only
(identical contract to validation_framework.run_walkforward /
run_walkforward_ml), and forward returns are computed independently
downstream from test-window data no signal_fn ever saw.

Run: python combiner_backtest.py  (ml_ensemble refits per fold -- expect a
similar order of runtime to layer_ml_ensemble.py's own ~40min deep-pass run)
"""
import os
import time

import numpy as np
import pandas as pd

from validation_framework import (
    generate_folds, fetch_prices, compute_ic, simulate_pnl, build_verdict,
    log, PROJECT_DIR, FORWARD_RETURN_DAYS,
)
from layer_swing_signals import (
    load_universe_df, build_tier_map, select_single_ticker_universe,
    build_momentum_variants, build_vol_breakout_variants, zscore_mean_reversion_signal,
)
import layer_ml_ensemble as ml_mod
from layer_combiner import get_current_weights, _signal_group_map, VERDICT_THRESHOLDS

OUT_PARQUET = os.path.join(PROJECT_DIR, "combiner_backtest.parquet")
OUT_FIRE_PARQUET = os.path.join(PROJECT_DIR, "combiner_backtest_fire.parquet")
OUT_TXT = os.path.join(PROJECT_DIR, "combiner_backtest_report.txt")

WF_TYPE = "swing"
FD = FORWARD_RETURN_DAYS[WF_TYPE]


def zscore_dict(d):
    s = pd.Series(d)
    std = s.std(ddof=0)
    if not std or np.isnan(std):
        return {t: 0.0 for t in s.index}
    return ((s - s.mean()) / std).to_dict()


def main():
    t_start = time.time()

    weights, _ = get_current_weights()
    log.info("Using FIXED weights from Round 2 full-history validation (%d survivors, see "
              "module docstring CAVEAT): %s", len(weights), {k: round(v, 3) for k, v in weights.items()})
    sig_to_group = _signal_group_map(weights.keys())
    group_total_weight = {}
    for sig, w in weights.items():
        grp = sig_to_group[sig]
        group_total_weight[grp] = group_total_weight.get(grp, 0.0) + w

    df = load_universe_df()
    tier_map = build_tier_map(df)
    universe = select_single_ticker_universe(df)
    ml_mod.TICKER_SECTOR = dict(zip(df["Ticker"], df["Sector"].fillna("Unknown")))

    single_signal_variants = {}
    single_signal_variants.update(build_momentum_variants())
    single_signal_variants.update(build_vol_breakout_variants())
    single_signal_variants["zscore_mean_reversion_20d"] = zscore_mean_reversion_signal
    # Only score variants that actually survived validation and carry weight
    # in the live combiner -- no point computing FAIL/unweighted signals here.
    single_signal_variants = {k: v for k, v in single_signal_variants.items() if k in weights}

    folds = generate_folds(WF_TYPE)
    log.info("Universe: %d tickers | %d surviving single-ticker signal variants + ml_ensemble",
              len(universe), len(single_signal_variants))
    log.info("Running combiner backtest over %d swing folds ...", len(folds))

    rows, fire_rows = [], []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        t0 = time.time()
        prices = fetch_prices(universe, tr_s, te_e)
        if prices.empty:
            continue

        raw_by_signal = {name: {} for name in single_signal_variants}
        train_panel = {}
        for t in universe:
            if t not in prices.columns:
                continue
            tr_px = prices.loc[(prices.index >= tr_s) & (prices.index < tr_e), t].dropna()
            if len(tr_px) < 60:
                continue
            train_panel[t] = tr_px
            for name, fn in single_signal_variants.items():
                try:
                    v = fn(tr_px)
                except Exception:
                    continue
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                raw_by_signal[name][t] = float(v)

        ml_preds = {}
        if "ml_ensemble" in weights and len(train_panel) >= 10:
            try:
                ml_preds = ml_mod.fit_predict_fn(train_panel)
            except Exception as e:
                log.warning("  fold %d ml_ensemble fit failed: %s", i + 1, e)
        raw_by_signal["ml_ensemble"] = {
            t: float(v) for t, v in ml_preds.items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        }

        z_by_signal = {name: zscore_dict(d) for name, d in raw_by_signal.items() if d}
        if not z_by_signal:
            continue

        all_tickers = set()
        for d in z_by_signal.values():
            all_tickers.update(d.keys())

        composite = {}
        for t in all_tickers:
            grp_vals = {}
            for name, zd in z_by_signal.items():
                if t not in zd:
                    continue
                grp_vals.setdefault(sig_to_group[name], []).append(zd[t])
            if not grp_vals:
                continue
            grp_z = {g: np.mean(vs) for g, vs in grp_vals.items()}
            grp_w = {g: group_total_weight.get(g, 0.0) for g in grp_z}
            w_sum = sum(grp_w.values())
            if w_sum <= 0:
                continue
            composite[t] = sum(grp_z[g] * grp_w[g] for g in grp_z) / w_sum

        fwd_rets, tiers = {}, {}
        for t in composite:
            te_px = (prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t].dropna()
                     if t in prices.columns else pd.Series(dtype=float))
            if len(te_px) < 2:
                continue
            entry = te_px.iloc[0]
            exit_ = te_px.iloc[min(FD, len(te_px) - 1)]
            fwd_rets[t] = (exit_ - entry) / entry
            tiers[t] = tier_map.get(t, "T3")

        if len(fwd_rets) < 10:
            continue

        comp_series = pd.Series({t: composite[t] for t in fwd_rets})
        fwd_series = pd.Series(fwd_rets)
        tier_series = pd.Series(tiers)

        ic, pval = compute_ic(comp_series, fwd_series)
        pnl = simulate_pnl(comp_series, fwd_series, tier_series, WF_TYPE)
        rows.append({"signal": "combiner_composite", "fold": i, "train_start": tr_s, "test_start": te_s,
                      "ic": ic, "pval": pval, "pnl": pnl, "n_stocks": len(comp_series), "wf_type": WF_TYPE})

        fire_mask = comp_series.abs() >= VERDICT_THRESHOLDS["Default"]["fire"]
        fire_t = comp_series[fire_mask].index
        fire_win_rate = fire_mean_ret = np.nan
        if len(fire_t) >= 5:
            fire_dir = np.sign(comp_series.loc[fire_t])
            directional_ret = fwd_series.loc[fire_t] * fire_dir
            fire_win_rate = (directional_ret > 0).mean() * 100
            fire_mean_ret = directional_ret.mean() * 100
            fire_rows.append({"fold": i, "test_start": te_s, "n_fire": len(fire_t),
                                "fire_win_rate": fire_win_rate, "fire_mean_ret_pct": fire_mean_ret})

        log.info("  fold %d/%d [%s] composite IC=%.3f PnL=%.4f n=%d | FIRE n=%d win%%=%s | %.1fs",
                  i + 1, len(folds), te_s.date(), ic or 0, pnl or 0, len(comp_series), len(fire_t),
                  f"{fire_win_rate:.1f}" if len(fire_t) >= 5 else "n/a", time.time() - t0)

    report_df = pd.DataFrame(rows)
    fire_df = pd.DataFrame(fire_rows)
    verdict = build_verdict(report_df) if not report_df.empty else pd.DataFrame()

    report_df.to_parquet(OUT_PARQUET, index=False)
    fire_df.to_parquet(OUT_FIRE_PARQUET, index=False)

    elapsed = time.time() - t_start
    lines = [
        "=" * 78,
        "CLAUDE QUANTS -- COMBINER-LEVEL BACKTEST (assembled decision layer)",
        "=" * 78,
        f"Run timestamp : {pd.Timestamp.now()}",
        f"Elapsed       : {elapsed/60:.1f} min",
        f"Folds scored  : {len(report_df)} / {len(folds)}",
        "",
        "CAVEAT: weights are FIXED from the full 2010-2025 Round 2 validation "
        "(same weights live in combiner_output.parquet today), applied uniformly "
        "to every fold including years before that validation existed. This tests "
        "whether the grouping/aggregation math and FIRE threshold, computed leak-free "
        "per fold from that fold's own training-window prices, produce a profitable "
        "actionable list -- it is NOT a fully nested re-derivation of which signals "
        "to include per era.",
        "",
        "OVERALL COMPOSITE (every ticker scored each fold)",
        "-" * 78,
        verdict.to_string(index=False) if not verdict.empty else "NO VERDICT ROWS",
        "",
        "FIRE-ONLY SUBSET (the actionable list -- does trading it make money?)",
        "-" * 78,
    ]
    if not fire_df.empty:
        lines.append(f"Folds with >=5 FIRE candidates : {len(fire_df)} / {len(folds)}")
        lines.append(f"Mean FIRE directional win rate : {fire_df['fire_win_rate'].mean():.1f}%")
        lines.append(f"Mean FIRE directional fwd return: {fire_df['fire_mean_ret_pct'].mean():+.3f}%")
        lines.append(f"Median n_fire per fold          : {fire_df['n_fire'].median():.0f}")
        lines.append("")
        lines.append(fire_df.to_string(index=False))
    else:
        lines.append("No folds had >=5 FIRE candidates.")
    txt = "\n".join(lines)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)

    print("\n" + txt)
    log.info("Done in %.1f min. Wrote %s, %s, %s", elapsed / 60, OUT_PARQUET, OUT_FIRE_PARQUET, OUT_TXT)


if __name__ == "__main__":
    main()
