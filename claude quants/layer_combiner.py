# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS -- Layer Combiner (scoring / FIRE-WAIT-KILL decision layer)
===========================================================================

WHAT THIS FILE DOES
--------------------
Takes validated signals (from layer_swing_signals.py and layer_ml_ensemble.py,
each scored via validation_framework.build_verdict()) and combines them into
one composite score per ticker, weighted by how well each signal ACTUALLY
performed in walk-forward validation -- not by hand-picked weights. A signal
that fails validation (verdict == FAIL) is excluded from voting ENTIRELY; it
does not just get down-weighted. That is the whole point of this rebuild: the
old production pipeline (layer4_portfolio.py, parent folder -- reference
only) hardcoded weights such as ~42% of the score for one signal regardless
of how it actually performed, and kept nonzero weight on a signal that had
FAILED validation with a negative Sharpe. Here, 100% of a signal's weight is
earned from its measured mean_ic / sharpe, and a failing signal gets zero.

CONTRACTS -- the interfaces this file depends on / produces
--------------------------------------------------------------------------
1. VALIDATION REPORTS (read).
   Any file in this folder matching `*_validation.parquet` or
   `*_validation_report.parquet` is treated as a validation report and is
   loaded automatically -- no hardcoded single filename. Each file is
   expected to be EITHER:
     (a) already in validation_framework.build_verdict() output shape
         (columns: signal, folds, mean_ic, ic_t, ic_pos_pct, total_pnl,
          pnl_pos_pct, sharpe, verdict), OR
     (b) a raw per-fold report_df (columns: signal, fold, ic, pval, pnl,
         n_stocks) -- in which case build_verdict() is run on it here.
   Known (expected, not required) producers -- this file is defensive and
   works with zero, one, or both present:
     - swing_signals_validation.parquet      (layer_swing_signals.py)
     - ml_ensemble_validation_report.parquet (layer_ml_ensemble.py)

2. LIVE SIGNAL SCORES (read).
   `latest_signals.parquet` -- long format, columns EXACTLY:
       ticker (str), signal_name (str), value (float)
   `value` is each signal's raw live output, in the SAME orientation it used
   during validation -- i.e. whatever was passed as `sig` into compute_ic /
   simulate_pnl there (higher = more bullish, lower = more bearish). This
   matches run_walkforward's own long/short convention: top quartile of a
   signal's value = long candidates, bottom quartile = short candidates.
   Neither upstream layer had a live-scoring function built at the time this
   file was written, so this parquet is the agreed hand-off contract. A
   ticker may appear once per signal_name that scored it; not every signal
   has to score every ticker.

3. COMPOSITE OUTPUT (write).
   `combiner_output.parquet` -- one row per ticker with >=1 contributing
   signal:
       ticker, sector, tier, composite_score, contributing_signals,
       n_signals, n_groups, direction, verdict
   sector/tier are joined from universe_master.parquet (already present in
   this folder) purely for downstream convenience (paper_tracker.py reads
   this file to log FIRE trades).

SCORING METHOD
--------------
Each surviving signal's live `value` is z-scored cross-sectionally (within
the current candidate batch) so heterogeneous signal scales become
comparable. Per-ticker composite = weighted average of the available
z-scores, using performance-based weights from compute_signal_weights(),
RE-NORMALIZED over just the signals that actually scored that ticker (since
not every signal scores every ticker). direction = LONG if composite >= 0
else SHORT (mirrors the long/short quantile split simulate_pnl already
uses). verdict is FIRE/WAIT/KILL by |composite| vs a threshold set that CAN
be looked up per-regime (a hook -- no regime-detection layer exists yet in
this replica, so 'Default' is used unless a regime string is passed in).

This file has no import dependency on the parent production pipeline. It was
read only for pattern/inspiration (threshold shape, FIRE/WAIT/KILL naming).
"""

import os
import glob
import logging

import numpy as np
import pandas as pd

from validation_framework import build_verdict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

UNIVERSE_FILE      = os.path.join(PROJECT_DIR, "universe_master.parquet")
LATEST_SIGNALS_FILE = os.path.join(PROJECT_DIR, "latest_signals.parquet")
COMBINER_OUTPUT_PATH = os.path.join(PROJECT_DIR, "combiner_output.parquet")

VALIDATION_GLOB_PATTERNS = ["*_validation.parquet", "*_validation_report.parquet"]

VERDICT_COLUMNS = ["signal", "folds", "mean_ic", "ic_t", "ic_pos_pct",
                    "total_pnl", "pnl_pos_pct", "sharpe", "verdict"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("LayerCombiner")

# ---------------------------------------------------------------------------
# FIRE / WAIT / KILL thresholds, on |composite_score| (composite is in
# cross-sectional z-score units, so these are "how many sigma from the pack").
# Keyed by regime; 'Default' is used when no regime is supplied. Shape is
# borrowed from the old pipeline's regime-aware FIRE/WAIT thresholds
# (layer4_portfolio.py, reference only) but simplified to a single dict pair.
# ---------------------------------------------------------------------------
VERDICT_THRESHOLDS = {
    "Default":  {"fire": 1.00, "wait": 0.40},
    "Bull":     {"fire": 1.10, "wait": 0.45},
    "Sideways": {"fire": 0.85, "wait": 0.35},
    "Crisis":   {"fire": 1.40, "wait": 0.60},
}
MIN_CONTRIBUTING_SIGNALS = 1   # a ticker with 0 surviving-signal coverage can never FIRE/WAIT

# ---------------------------------------------------------------------------
# SIGNAL GROUPS -- fix for vote-stacking (added 2026-07-18).
#
# Live cross-sectional correlation check on latest_signals.parquet found the
# 11 "independent" swing signals below at r=0.32-1.00 (several pairs at
# EXACTLY 1.00 -- e.g. vol_breakout_20d_meanrev vs. its _thr0.5/_thr1.0
# variants are the same computation with only a threshold filter changed).
# They are ~10 measurements of ONE underlying phenomenon (recent move
# reverses over a 10-day horizon), not 10 independent confirmations. Left
# ungrouped, compute_signal_weights() gave this one cluster ~93% of total
# weight (vs. 4.5% for the genuinely uncorrelated ml_ensemble, r=~0), so any
# ticker where the cluster agreed with itself -- which it almost always does,
# by construction -- produced an artificially extreme composite z-score.
# 427/2064 tickers (~20%) hit FIRE in the first run, which is not a
# plausible base rate for a "genuinely quick swing trades" quality filter.
#
# Fix: aggregate within a group FIRST (mean z-score across whichever member
# signals scored a given ticker), then treat each group as ONE vote in the
# composite, weighted by the group's total performance-based weight. This
# keeps the "weight comes from measured Sharpe/IC, FAIL is excluded" design
# intact -- it just stops one phenomenon from being counted eleven times.
# A signal not listed here is its own singleton group (e.g. ml_ensemble,
# pairs_mean_reversion, and any future alt-data signal once validated).
# ---------------------------------------------------------------------------
SIGNAL_GROUPS = {
    "short_term_reversal": [
        "short_momentum_5d_flip", "short_momentum_10d_flip", "short_momentum_20d_flip",
        "short_momentum_40d_flip", "short_momentum_60d_flip",
        "vol_breakout_10d_meanrev", "vol_breakout_20d_meanrev",
        "vol_breakout_20d_meanrev_thr0.5", "vol_breakout_20d_meanrev_thr1.0",
        "vol_breakout_40d_meanrev", "zscore_mean_reversion_20d",
    ],
}

# ---------------------------------------------------------------------------
# NOT-YET-EXECUTABLE signals (added 2026-07-18).
#
# pairs_mean_reversion scores a synthetic "TICKERA_TICKERB" pair id, not a
# real tradeable ticker (see live_scoring.py docstring, KNOWN GAP), and
# long-one-leg/short-the-other execution isn't implemented in
# paper_tracker.py yet. It also only rated WEAK in validation (pnl_pos_pct
# 45% < 55% needed -- historically lost money more often than not despite a
# statistically significant IC). On the first live run, 154/396 (39%) of all
# FIRE verdicts rested on this ONE weak, unexecutable signal alone (it has no
# group-mates, so nothing else ever confirms or dilutes it). A FIRE verdict
# is supposed to mean "actionable" -- so until pairs execution exists, this
# signal is excluded from voting entirely rather than being allowed to
# single-handedly FIRE. Re-include it here once paper_tracker.py supports
# pairs trades.
# ---------------------------------------------------------------------------
NOT_YET_EXECUTABLE = {"pairs_mean_reversion"}


def _signal_group_map(signal_names):
    """signal_name -> group_name for every name in signal_names. Names not
    listed in SIGNAL_GROUPS map to themselves (singleton group)."""
    lookup = {sig: grp for grp, members in SIGNAL_GROUPS.items() for sig in members}
    return {sig: lookup.get(sig, sig) for sig in signal_names}


# ===========================================================================
# STEP 1 -- discover + load validation reports -> combined verdict_df
# ===========================================================================
def discover_validation_reports(directory=PROJECT_DIR):
    """Return sorted, de-duplicated paths of every validation-report parquet in `directory`."""
    found = set()
    for pat in VALIDATION_GLOB_PATTERNS:
        found.update(glob.glob(os.path.join(directory, pat)))
    return sorted(found)


def _load_validation_file(path):
    """Load one validation parquet, coercing a raw report_df into verdict shape if needed."""
    df = pd.read_parquet(path)
    if "verdict" in df.columns:
        return df
    if {"signal", "ic", "pnl"}.issubset(df.columns):
        log.info("  %s looks like a raw fold report -- running build_verdict() on it.", os.path.basename(path))
        return build_verdict(df)
    log.warning("  %s has neither verdict-shape nor raw-report-shape columns -- skipping.", os.path.basename(path))
    return pd.DataFrame(columns=VERDICT_COLUMNS)


def load_all_verdicts(directory=PROJECT_DIR):
    """
    Discover and load every validation report present, concatenated into one
    verdict_df (build_verdict() output shape). Missing files are NOT an
    error -- returns an empty (correctly-columned) frame if nothing is found
    yet, since the other two layers may still be running in parallel.
    """
    paths = discover_validation_reports(directory)
    if not paths:
        log.warning("No validation report parquet files found in %s -- "
                     "nothing to weight yet (expected until the other layers finish).", directory)
        return pd.DataFrame(columns=VERDICT_COLUMNS)

    frames = []
    for p in paths:
        try:
            df = _load_validation_file(p)
            if not df.empty:
                frames.append(df)
                log.info("Loaded validation report: %s (%d signal(s))", os.path.basename(p), len(df))
        except Exception as e:
            log.warning("Failed to read %s: %s", p, e)

    if not frames:
        return pd.DataFrame(columns=VERDICT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # If the same signal name shows up in more than one report file, keep the
    # last-loaded copy (later files = more recently produced, in glob-sorted order).
    combined = combined.drop_duplicates(subset="signal", keep="last").reset_index(drop=True)
    return combined


# ===========================================================================
# STEP 2 -- verdict_df -> performance-based weights
# (pure function, no I/O -- can be re-run/inspected independently, e.g. from
#  a notebook, as more signals get validated over time)
# ===========================================================================
def compute_signal_weights(verdict_df, metric="sharpe"):
    """
    Convert a build_verdict()-shaped DataFrame into {signal_name: weight}.

    Rules:
      - verdict == 'FAIL'          -> excluded entirely (not just down-weighted;
                                       the signal gets no key in the returned dict)
      - verdict in ('PASS','WEAK') -> included, weighted proportional to `metric`
      - `metric` values are floored at 0 before normalizing (a WEAK signal with
        a slightly negative sharpe/IC shouldn't get a negative weight)
      - if every surviving signal's `metric` is <=0, falls back to the other
        metric ('mean_ic' <-> 'sharpe'); if that's also all <=0, falls back to
        equal-weighting the survivors so validated signals still produce a
        usable combiner instead of an empty dict
      - weights always sum to 1.0

    Returns {} if verdict_df is None/empty or nothing survives the FAIL filter.
    """
    if verdict_df is None or verdict_df.empty:
        return {}

    survivors = verdict_df[verdict_df["verdict"].isin(["PASS", "WEAK"])].copy()
    survivors = survivors[~survivors["signal"].isin(NOT_YET_EXECUTABLE)]
    if survivors.empty:
        return {}

    def _basis(col):
        if col not in survivors.columns:
            return None
        b = pd.to_numeric(survivors[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        return b if b.sum() > 0 else None

    basis = _basis(metric)
    if basis is None:
        alt = "mean_ic" if metric != "mean_ic" else "sharpe"
        log.info("compute_signal_weights: '%s' unusable (all <=0) -- falling back to '%s'.", metric, alt)
        basis = _basis(alt)

    if basis is None:
        log.info("compute_signal_weights: no usable performance metric -- equal-weighting %d survivor(s).", len(survivors))
        n = len(survivors)
        return {sig: 1.0 / n for sig in survivors["signal"]}

    weights = basis / basis.sum()
    return dict(zip(survivors["signal"], weights))


def get_current_weights(directory=PROJECT_DIR, metric="sharpe"):
    """Convenience wrapper: discover + load + weight, in one call."""
    verdict_df = load_all_verdicts(directory)
    return compute_signal_weights(verdict_df, metric=metric), verdict_df


# ===========================================================================
# STEP 3 -- live signal scores -> composite score per ticker
# ===========================================================================
def load_latest_signals(path=LATEST_SIGNALS_FILE):
    """Read the live-scoring hand-off file. See module docstring for the contract."""
    if not os.path.exists(path):
        log.warning("latest_signals.parquet not found at %s -- "
                     "upstream layers haven't produced a live run yet.", path)
        return pd.DataFrame(columns=["ticker", "signal_name", "value"])
    df = pd.read_parquet(path)
    required = {"ticker", "signal_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"latest_signals.parquet is missing required column(s): {missing}")
    return df


def load_universe_meta(path=UNIVERSE_FILE):
    """Ticker -> (sector, tier) lookup, for annotating combiner output."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "sector", "tier"])
    u = pd.read_parquet(path)
    u = u.rename(columns={"Ticker": "ticker", "Sector": "sector", "Tier": "tier"})
    keep = [c for c in ["ticker", "sector", "tier"] if c in u.columns]
    return u[keep]


def classify_verdict(composite_score, n_contributing, regime="Default", min_signals=MIN_CONTRIBUTING_SIGNALS):
    """FIRE / WAIT / KILL from |composite_score| (z-score units) + signal coverage."""
    if n_contributing < min_signals or pd.isna(composite_score):
        return "KILL"
    thr = VERDICT_THRESHOLDS.get(regime, VERDICT_THRESHOLDS["Default"])
    mag = abs(composite_score)
    if mag >= thr["fire"]:
        return "FIRE"
    if mag >= thr["wait"]:
        return "WAIT"
    return "KILL"


def compute_composite_scores(latest_df, weights, regime="Default"):
    """
    latest_df : output of load_latest_signals() -- columns ticker, signal_name, value
    weights   : output of compute_signal_weights() -- {signal_name: weight}, sums to 1.0

    Two-stage aggregation (see SIGNAL_GROUPS above): signals are first
    z-scored individually, then averaged WITHIN their group (so 11 correlated
    reversal variants collapse to one group-level z-score per ticker), and
    only then combined ACROSS groups using each group's total performance
    weight (sum of its surviving members' weights). This is the same
    "weight comes from measured Sharpe/IC" design as before -- it just
    prevents one phenomenon measured 11 ways from casting 11 votes.

    Returns a DataFrame: ticker, composite_score, contributing_signals,
    n_signals, n_groups, direction, verdict. Empty (correctly-columned) if
    either input is empty, or if none of the signals in latest_df survived
    validation.
    """
    out_cols = ["ticker", "composite_score", "contributing_signals", "n_signals", "n_groups", "direction", "verdict"]
    if latest_df is None or latest_df.empty or not weights:
        return pd.DataFrame(columns=out_cols)

    df = latest_df[latest_df["signal_name"].isin(weights.keys())].copy()
    if df.empty:
        log.warning("None of the signals in latest_signals.parquet survived validation "
                     "(PASS/WEAK) -- composite score cannot be computed.")
        return pd.DataFrame(columns=out_cols)

    # Cross-sectional z-score per signal, within this candidate batch.
    def _zscore(g):
        std = g["value"].std(ddof=0)
        if not std or np.isnan(std):
            return pd.Series(0.0, index=g.index)
        return (g["value"] - g["value"].mean()) / std

    sig_to_group = _signal_group_map(weights.keys())
    df["z"] = df.groupby("signal_name", group_keys=False).apply(_zscore)
    df["w"] = df["signal_name"].map(weights)
    df["group"] = df["signal_name"].map(sig_to_group)

    # Total performance-based weight per group = sum of its surviving
    # members' individual weights -- preserves total weight mass, just
    # re-cast as one vote per group instead of one vote per signal.
    group_total_weight = {}
    for sig, w in weights.items():
        grp = sig_to_group[sig]
        group_total_weight[grp] = group_total_weight.get(grp, 0.0) + w

    rows = []
    for ticker, g in df.groupby("ticker"):
        # Stage 1: mean z-score within each group, using only the members
        # that actually scored this ticker.
        grp_z = g.groupby("group")["z"].mean()
        grp_w = grp_z.index.to_series().map(group_total_weight).fillna(0.0)
        w_sum = grp_w.sum()
        if w_sum <= 0:
            continue
        composite = (grp_z * grp_w).sum() / w_sum   # re-normalized over groups present for this ticker
        contributing = sorted(g["signal_name"].tolist())
        rows.append({
            "ticker": ticker,
            "composite_score": round(float(composite), 4),
            "contributing_signals": ",".join(contributing),
            "n_signals": len(contributing),
            "n_groups": g["group"].nunique(),
            "direction": "LONG" if composite >= 0 else "SHORT",
        })

    result = pd.DataFrame(rows, columns=["ticker", "composite_score", "contributing_signals", "n_signals", "n_groups", "direction"])
    if result.empty:
        return pd.DataFrame(columns=out_cols)

    result["verdict"] = result.apply(
        lambda r: classify_verdict(r["composite_score"], r["n_signals"], regime=regime), axis=1
    )
    return result[out_cols].sort_values("composite_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# ===========================================================================
# STEP 4 -- orchestration entry point (called by orchestrator.py)
# ===========================================================================
def run_combiner(regime="Default", write_output=True):
    """
    Full pipeline: load verdicts -> weights -> live signals -> composite
    scores -> (optionally) write combiner_output.parquet.

    Returns (result_df, weights, verdict_df) so callers can inspect any stage.
    Safe to call even if upstream files don't exist yet -- returns empty
    frames rather than raising, so the orchestrator can log a clear "nothing
    to do yet" message instead of crashing.
    """
    verdict_df = load_all_verdicts()
    weights = compute_signal_weights(verdict_df)
    log.info("Signal weights (%d survivor(s) of %d validated): %s",
              len(weights), len(verdict_df), {k: round(v, 3) for k, v in weights.items()})

    latest_df = load_latest_signals()
    result = compute_composite_scores(latest_df, weights, regime=regime)

    if not result.empty:
        meta = load_universe_meta()
        if not meta.empty:
            result = result.merge(meta, on="ticker", how="left")
        else:
            result["sector"] = ""
            result["tier"] = ""
        col_order = ["ticker", "sector", "tier", "composite_score", "contributing_signals",
                     "n_signals", "n_groups", "direction", "verdict"]
        result = result[[c for c in col_order if c in result.columns]]

        vc = result["verdict"].value_counts().to_dict()
        log.info("Composite scores computed for %d ticker(s): FIRE=%d  WAIT=%d  KILL=%d",
                  len(result), vc.get("FIRE", 0), vc.get("WAIT", 0), vc.get("KILL", 0))

        if write_output:
            result.to_parquet(COMBINER_OUTPUT_PATH, index=False)
            log.info("Wrote %s", COMBINER_OUTPUT_PATH)
    else:
        log.info("No composite scores produced this run (see warnings above for why).")

    return result, weights, verdict_df


if __name__ == "__main__":
    result_df, current_weights, current_verdicts = run_combiner()
    print("\n=== SIGNAL WEIGHTS ===")
    if current_weights:
        for sig, w in sorted(current_weights.items(), key=lambda kv: -kv[1]):
            print(f"  {sig:<25} {w:.3f}")
    else:
        print("  (none -- no validation reports found or nothing passed)")

    print("\n=== FIRE CANDIDATES ===")
    if not result_df.empty:
        fire = result_df[result_df["verdict"] == "FIRE"]
        print(fire.to_string(index=False) if not fire.empty else "  (none this run)")
    else:
        print("  (no composite scores this run -- see log above)")
