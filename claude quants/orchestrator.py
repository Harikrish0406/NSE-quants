# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS -- Orchestrator
================================

Ties the pipeline together: data refresh -> signal layers' live scoring ->
layer_combiner's FIRE/WAIT/KILL decision -> paper_tracker's log/update/report.

Modelled loosely on the parent production pipeline's layer5_orchestrator.py
(read for pattern only, not imported): a sequence of named steps, each timed
and logged, with a final summary. Deliberately NOT that file's subprocess-
per-script design, though -- every step here is an in-process Python module
already living in this folder, so a direct function call is simpler and
faster than spawning a subprocess per step. Also deliberately NOT
layer6_dashboard.py's ~2000-line HTML dashboard -- just a short, readable
runner with clear console logging.

Steps
-----
1. Data refresh                    -- PLACEHOLDER. This replica's data layers
                                       (layer0_nse_data.py, layer1_daily.py,
                                       layer1_heavy_compute.py) already exist
                                       in this folder but are out of scope for
                                       this build; wire this hook up to them
                                       when a live daily run is needed.
2. Signal layers' live scoring     -- Defensively tries to import
                                       layer_swing_signals.py and
                                       layer_ml_ensemble.py (built in parallel
                                       by other agents) and call a recognized
                                       live-scoring entry point on each, so
                                       they write latest_signals.parquet. If
                                       either module or entry point doesn't
                                       exist yet, logs that clearly and moves
                                       on -- this step is expected to no-op
                                       until those builds land.
3. Layer combiner                  -- layer_combiner.run_combiner(): reads
                                       validation reports + latest_signals.
                                       parquet (if present), writes
                                       combiner_output.parquet.
4. Paper tracker                   -- paper_tracker.run_daily(): log new FIRE
                                       trades from combiner_output.parquet,
                                       update open trades against live OHLC,
                                       print the performance report.

Every step is wrapped so one step failing does NOT abort the run -- later
steps (especially 3 and 4) are already written to degrade gracefully when
their inputs are missing, which is the expected state until every layer in
this parallel build exists. The point of this runner is to always finish and
tell you clearly what did and didn't happen, not to hard-gate on upstream
layers the way the production pipeline's layer5 does.

Usage
-----
  python orchestrator.py
"""

import importlib
import logging
import sys
import time
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Orchestrator")


def banner(msg: str):
    log.info("")
    log.info("=" * 78)
    log.info("  %s", msg)
    log.info("=" * 78)


def run_step(name: str, fn) -> dict:
    """Time + log one pipeline step. Never raises -- failures are captured and reported."""
    log.info("")
    log.info("-" * 78)
    log.info("  STEP: %s", name)
    log.info("-" * 78)
    t0 = time.time()
    try:
        detail = fn()
        elapsed = time.time() - t0
        log.info("  [OK]   %s finished in %.1fs%s", name, elapsed, f"  -- {detail}" if detail else "")
        return {"ok": True, "elapsed": elapsed, "detail": detail or ""}
    except Exception as e:
        elapsed = time.time() - t0
        log.error("  [FAIL] %s failed after %.1fs: %s", name, elapsed, e)
        log.error(traceback.format_exc())
        return {"ok": False, "elapsed": elapsed, "detail": str(e)}


# ===========================================================================
# STEP 1 -- data refresh (placeholder hook)
# ===========================================================================
def step_data_refresh():
    log.info("  [placeholder] Not wired to a live refresh in this build.")
    log.info("  [placeholder] Hook point for layer0_nse_data.py / layer1_daily.py / "
              "layer1_heavy_compute.py when a live daily run is needed.")
    return "no-op (placeholder)"


# ===========================================================================
# STEP 2 -- signal layers' live scoring (defensive, best-effort)
# ===========================================================================
LIVE_SCORING_ENTRY_POINTS = ("run_live_scoring", "live_score", "score_live", "run")


def _try_live_score(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        log.warning("  %s.py not importable yet -- skipping (expected if that build isn't done).", module_name)
        return f"skipped ({module_name} not found)"

    for fn_name in LIVE_SCORING_ENTRY_POINTS:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            log.info("  Found %s.%s() -- calling it.", module_name, fn_name)
            fn()
            return f"called {module_name}.{fn_name}()"

    log.warning("  %s.py found but has no recognized live-scoring entry point %s -- skipping.",
                module_name, LIVE_SCORING_ENTRY_POINTS)
    return f"skipped ({module_name} has no live-scoring entry point yet)"


def step_signal_live_scoring():
    results = []
    for module_name in ("layer_swing_signals", "layer_ml_ensemble"):
        results.append(_try_live_score(module_name))
    return "; ".join(results)


# ===========================================================================
# STEP 3 -- layer combiner (real)
# ===========================================================================
def step_combiner():
    from layer_combiner import run_combiner
    result_df, weights, verdict_df = run_combiner()
    n_fire = int((result_df["verdict"] == "FIRE").sum()) if not result_df.empty else 0
    return f"{len(weights)} signal(s) weighted, {len(result_df)} ticker(s) scored, {n_fire} FIRE"


# ===========================================================================
# STEP 4 -- paper tracker (real)
# ===========================================================================
def step_paper_tracker():
    from paper_tracker import run_daily
    run_daily()
    return "log -> update -> report complete"


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t_start = time.time()
    banner("CLAUDE QUANTS ORCHESTRATOR -- daily run")

    steps = [
        ("Data Refresh",              step_data_refresh),
        ("Signal Live Scoring",       step_signal_live_scoring),
        ("Layer Combiner",            step_combiner),
        ("Paper Tracker",             step_paper_tracker),
    ]

    results = {}
    for name, fn in steps:
        results[name] = run_step(name, fn)

    total_elapsed = time.time() - t_start
    banner(f"ORCHESTRATOR COMPLETE  |  total {total_elapsed:.1f}s")
    for name, r in results.items():
        status = "OK  " if r["ok"] else "FAIL"
        log.info("  [%s]  %-24s %6.1fs   %s", status, name, r["elapsed"], r["detail"])

    if not all(r["ok"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
