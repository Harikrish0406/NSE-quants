"""
================================================================================
  NSE DECISION ENGINE — LAYER 5 DAILY ORCHESTRATOR
  nse_layer5_daily_runner.py

  Purpose : Master daily runner — executes L2 → L3 → L4 in sequence,
            enforces timing guards, writes the consolidated EOD report,
            and appends to the persistent run-log for the feedback loop.

  Run     : python nse_layer5_daily_runner.py
  Schedule: Task Scheduler / cron — 21:00 IST (after NSE close + bhavcopy)

  Output files (all in BASE dir)
    nse_layer2_candidates.parquet
    nse_layer3_signals.parquet
    nse_layer4_portfolio.parquet
    nse_layer5_eod_report_YYYY-MM-DD.txt   ← consolidated daily report
    nse_run_log.csv                         ← persistent cross-day log
================================================================================
"""

import subprocess
import sys
import os
import time
import logging
import traceback
from datetime import datetime, date
from pathlib import Path
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these paths to match your machine
# ──────────────────────────────────────────────────────────────────────────────
BASE          = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
PYTHON        = r"C:/Users/harik/AppData/Local/Programs/Python/Python312/python.exe"

LAYER2_SCRIPT = BASE / "layer2_daily_filter.py"
LAYER3_SCRIPT = BASE / "layer3_rules_engine.py"
LAYER4_SCRIPT = BASE / "layer4_portfolio.py"

LAYER2_OUT    = BASE / "nse_layer2_candidates.parquet"
LAYER3_OUT    = BASE / "nse_layer3_signals.parquet"
LAYER4_OUT    = BASE / "nse_layer4_portfolio.parquet"

RUN_LOG       = BASE / "nse_run_log.csv"

# Hard timeout per layer (seconds) — L3 is slowest due to yfinance calls
TIMEOUTS = {
    "Layer2": 300,   # 5 min
    "Layer3": 600,   # 10 min
    "Layer4": 120,   # 2 min
}

TODAY = date.today().isoformat()

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING — console + file
# ──────────────────────────────────────────────────────────────────────────────
log_path = BASE / f"nse_layer5_run_{TODAY}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger("L5")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def banner(msg: str):
    log.info("")
    log.info("=" * 80)
    log.info(f"  {msg}")
    log.info("=" * 80)


def run_layer(name: str, script: Path, timeout: int) -> tuple[bool, float, str]:
    """
    Run a layer script as a subprocess.
    Returns (success, elapsed_sec, stderr_tail).
    """
    log.info(f"[{name}] Starting  →  {script.name}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BASE),
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-1500:]
            log.error(f"[{name}] FAILED  (rc={result.returncode})  in {elapsed:.1f}s")
            log.error(f"[{name}] Tail:\n{tail}")
            return False, elapsed, tail
        log.info(f"[{name}] OK  in {elapsed:.1f}s")
        return True, elapsed, ""
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        msg = f"Timed out after {timeout}s"
        log.error(f"[{name}] {msg}")
        return False, elapsed, msg
    except Exception as exc:
        elapsed = time.time() - t0
        msg = traceback.format_exc()
        log.error(f"[{name}] Exception: {exc}")
        return False, elapsed, msg


def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log.warning(f"Could not load {path.name}: {e}")
        return None


def parquet_row_count(path: Path) -> int:
    df = load_parquet_safe(path)
    return len(df) if df is not None else -1


# ──────────────────────────────────────────────────────────────────────────────
# EOD CONSOLIDATED REPORT
# ──────────────────────────────────────────────────────────────────────────────

def build_eod_report(
    results: dict,
    l2_df: pd.DataFrame | None,
    l3_df: pd.DataFrame | None,
    l4_df: pd.DataFrame | None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    W = 80

    # ── Normalize stop/target column names from Layer 4 ──
    if l4_df is not None:
        if "stop_price" in l4_df.columns and "stop_loss" not in l4_df.columns:
            l4_df = l4_df.copy()
            l4_df["stop_loss"] = l4_df["stop_price"]
        if "entry_price" in l4_df.columns and "price_close" not in l4_df.columns:
            l4_df = l4_df.copy()
            l4_df["price_close"] = l4_df["entry_price"]

    def sep():
        lines.append("=" * W)

    def sub():
        lines.append("─" * W)

    sep()
    lines.append(f"  NSE DECISION ENGINE — LAYER 5 EOD CONSOLIDATED REPORT")
    lines.append(f"  Date   : {now}")
    lines.append(f"  Status : {'✓ ALL LAYERS OK' if all(r['ok'] for r in results.values()) else '✗ ONE OR MORE LAYERS FAILED'}")
    sep()

    # ── pipeline timing
    lines.append("")
    lines.append("  PIPELINE TIMING")
    sub()
    total_sec = sum(r["elapsed"] for r in results.values())
    for name, r in results.items():
        status = "OK " if r["ok"] else "FAIL"
        lines.append(f"  {name:<12}  [{status}]  {r['elapsed']:6.1f}s")
    lines.append(f"  {'Total':<12}           {total_sec:6.1f}s")
    lines.append("")

    # ── Layer 2 summary
    sep()
    lines.append("  LAYER 2 — UNIVERSE FILTER")
    sub()
    if l2_df is not None:
        regime = l2_df["regime"].iloc[0] if "regime" in l2_df.columns else "N/A"
        vix    = l2_df["vix"].iloc[0]    if "vix"    in l2_df.columns else float("nan")
        lines.append(f"  Regime : {regime}      VIX : {float(vix):.2f}" if vix is not None else f"  Regime : {regime}      VIX : N/A")
        lines.append(f"  Candidates passed all gates : {len(l2_df)}")
        if "tier" in l2_df.columns:
            tc = l2_df["tier"].value_counts().to_dict()
            lines.append(f"  Tier breakdown : T1={tc.get('T1',0)}  T2={tc.get('T2',0)}  T3={tc.get('T3',0)}")
        if "pairs_eligible" in l2_df.columns:
            lines.append(f"  Pairs-eligible : {l2_df['pairs_eligible'].sum()}")
    else:
        lines.append("  [Layer 2 output not available]")
    lines.append("")

    # ── Layer 3 summary
    sep()
    lines.append("  LAYER 3 — RULES ENGINE (A / C / D / E)")
    sub()
    if l3_df is not None:
        lines.append(f"  Total signals ranked : {len(l3_df)}")
        for col, label in [
            ("rule_a_fires",  "Rule A  (Vol Breakout)  "),
            ("rule_c_fires",  "Rule C  (Pairs MR)      "),
            ("rule_d_fires",  "Rule D  (Event-driven)  "),
            ("rule_e_fires",  "Rule E  (CUSUM Momentum)"),
        ]:
            if col in l3_df.columns:
                n = int(l3_df[col].sum())
                pct = n / len(l3_df) * 100
                lines.append(f"  {label} : {n:5d}  ({pct:.1f}%)")

        if "num_rules_firing" in l3_df.columns:
            lines.append("")
            lines.append("  Multi-rule synergy :")
            for k in range(5):
                n = int((l3_df["num_rules_firing"] == k).sum())
                label = ["0 rules (watch)","1 rule (weak)","2 rules (confirmed)","3 rules (strong)","4 rules (very strong)"][k]
                lines.append(f"    {label:<28} : {n}")

        lines.append("")
        lines.append("  TOP 25 SIGNALS (by final_score)")
        sub()
        score_col = "final_score" if "final_score" in l3_df.columns else None
        if score_col:
            l3_df[score_col] = pd.to_numeric(l3_df[score_col], errors='coerce').fillna(0.0)
            top25 = l3_df.nlargest(25, score_col)
            for i, row in enumerate(top25.itertuples(), 1):
                ticker = getattr(row, "ticker", "?")
                tier   = getattr(row, "tier", "?")
                sector = getattr(row, "sector", "Unknown")[:18]
                price  = getattr(row, "price_close", float("nan"))
                score  = getattr(row, score_col, 0.0)
                # rule tag
                tags = ""
                for col, ch in [("rule_a_fires","A"),("rule_c_fires","C"),
                                 ("rule_d_fires","D"),("rule_e_fires","E")]:
                    tags += ch if (col in l3_df.columns and getattr(row, col, False)) else "."
                lines.append(
                    f"  {i:>3}  {ticker:<14} {tier}  {sector:<20}  ₹{price:>8.2f}  [{tags}]  Score:{score:.4f}"
                )
    else:
        lines.append("  [Layer 3 output not available]")
    lines.append("")

    # ── Layer 4 / portfolio
    sep()
    lines.append("  LAYER 4 — PORTFOLIO CONSTRUCTOR")
    sub()
    if l4_df is not None:
        fire_mask = l4_df["timing_verdict"].str.upper() == "FIRE" if "timing_verdict" in l4_df.columns else pd.Series(False, index=l4_df.index)
        wait_mask = l4_df["timing_verdict"].str.upper() == "WAIT" if "timing_verdict" in l4_df.columns else pd.Series(False, index=l4_df.index)
        kill_mask = l4_df["timing_verdict"].str.upper() == "KILL" if "timing_verdict" in l4_df.columns else pd.Series(False, index=l4_df.index)

        fire_df = l4_df[fire_mask]
        lines.append(f"  Timing verdicts : FIRE={fire_mask.sum()}  WAIT={wait_mask.sum()}  KILL={kill_mask.sum()}")
        lines.append("")

        if len(fire_df) > 0:
            lines.append("  ✅  FIRE — EXECUTE THESE POSITIONS")
            sub()
            for row in fire_df.itertuples():
                ticker  = getattr(row, "ticker", "?")
                tier    = getattr(row, "tier", "?")
                entry   = getattr(row, "entry_price",  getattr(row, "price_close", float("nan")))
                sl      = getattr(row, "stop_loss",    float("nan"))
                tgt     = getattr(row, "target_price", float("nan"))
                qty     = getattr(row, "qty",          getattr(row, "position_qty", "?"))
                score   = getattr(row, "composite_score", getattr(row, "final_score", 0.0))
                sector  = getattr(row, "sector", "?")[:16]
                lines.append(
                    f"  {ticker:<14} {tier}  {sector:<18}"
                    f"  E:₹{entry:>8.2f}  SL:₹{sl:>8.2f}  T:₹{tgt:>8.2f}"
                    f"  Qty:{qty:>6}  Score:{score:.4f}"
                )
            # exposure / risk totals
            if "exposure" in l4_df.columns:
                total_exp  = fire_df["exposure"].sum()
                total_risk = fire_df["risk_inr"].sum() if "risk_inr" in l4_df.columns else float("nan")
                lines.append("")
                lines.append(f"  Total exposure : ₹{total_exp:>12,.0f}")
                lines.append(f"  Total risk     : ₹{total_risk:>12,.0f}")
        else:
            lines.append("  No FIRE signals today — check WATCHLIST below.")

        # watchlist
        lines.append("")
        lines.append("  📋  WATCHLIST — TOP 30 WAIT signals")
        sub()
        wait_df = l4_df[wait_mask]
        score_col = "composite_score" if "composite_score" in l4_df.columns else "final_score"
        if len(wait_df) > 0 and score_col in wait_df.columns:
            top_wait = wait_df.nlargest(30, score_col)
            for i, row in enumerate(top_wait.itertuples(), 1):
                ticker = getattr(row, "ticker", "?")
                tier   = getattr(row, "tier", "?")
                price  = getattr(row, "entry_price", getattr(row, "price_close", float("nan")))
                score  = getattr(row, score_col, 0.0)
                sector = getattr(row, "sector", "?")[:18]
                lines.append(
                    f"  {i:>3}  {ticker:<14} {tier}  {sector:<20}  ₹{price:>8.2f}  Score:{score:.4f}"
                )
        else:
            lines.append("  [No WAIT signals]")
    else:
        lines.append("  [Layer 4 output not available]")
    lines.append("")

    # ── regime context
    sep()
    lines.append("  REGIME + MARKET CONTEXT")
    sub()
    if l2_df is not None and "regime" in l2_df.columns:
        regime = l2_df["regime"].iloc[0]
        vix    = l2_df["vix"].iloc[0] if "vix" in l2_df.columns else float("nan")
        lines.append(f"  Regime : {regime}      VIX : {float(vix):.2f}" if vix is not None else f"  Regime : {regime}      VIX : N/A")
        if "sector" in l2_df.columns and "sector_rank" in l2_df.columns:
            top_sect = (
                l2_df[l2_df["sector"].str.strip().ne("")]
                      .sort_values("sector_rank")
                      .groupby("sector")["sector_rank"]
                      .first()
                      .sort_values()
                      .head(3)
                      .index.tolist()
            )
            lines.append(f"  Top sectors today : {top_sect}")
    lines.append("")

    # ── footer
    sep()
    lines.append(f"  NEXT : Execute FIRE signals at market open tomorrow")
    lines.append(f"         Log fills in nse_run_log.csv for feedback loop")
    sep()

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# RUN LOG  — persistent cross-day CSV for feedback loop
# ──────────────────────────────────────────────────────────────────────────────

def append_run_log(results: dict, l4_df: pd.DataFrame | None):
    """
    Appends one row per day to nse_run_log.csv.
    Columns: date, l2_ok, l3_ok, l4_ok, l2_sec, l3_sec, l4_sec,
             candidates, fire_count, wait_count, kill_count,
             regime, vix, top_score
    """
    row = {"date": TODAY}
    for name in ["Layer2", "Layer3", "Layer4"]:
        key = name.lower()
        if name not in results:
            row[f"{key}_ok"]  = False
            row[f"{key}_elapsed"] = 0.0
            row[f"{key}_error"] = "did_not_run"
            continue
        row[f"{key}_ok"]  = results[name]["ok"]
        row[f"{key}_sec"] = round(results[name]["elapsed"], 1)

    row["candidates"] = parquet_row_count(LAYER2_OUT)

    if l4_df is not None:
        if "timing_verdict" in l4_df.columns:
            vc = l4_df["timing_verdict"].str.upper().value_counts().to_dict()
            row["fire_count"] = vc.get("FIRE", 0)
            row["wait_count"] = vc.get("WAIT", 0)
            row["kill_count"] = vc.get("KILL", 0)
        sc = "composite_score" if "composite_score" in l4_df.columns else "final_score"
        row["top_score"] = round(float(l4_df[sc].max()), 4) if sc in l4_df.columns else None

    l2 = load_parquet_safe(LAYER2_OUT)
    if l2 is not None:
        row["regime"] = l2["regime"].iloc[0] if "regime" in l2.columns else None
        vix_val = l2["vix"].iloc[0] if "vix" in l2.columns else None
        row["vix"] = round(float(vix_val), 2) if vix_val is not None else None

    new_row = pd.DataFrame([row])
    if RUN_LOG.exists():
        existing = pd.read_csv(RUN_LOG)
        # avoid duplicate for same date
        existing = existing[existing["date"] != TODAY]
        updated  = pd.concat([existing, new_row], ignore_index=True).dropna(axis=1, how="all")
    else:
        updated = new_row

    updated.to_csv(RUN_LOG, index=False)
    log.info(f"Run log updated: {RUN_LOG.name}  ({len(updated)} days recorded)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    banner(f"NSE DECISION ENGINE — LAYER 5 DAILY ORCHESTRATOR  |  {TODAY}")

    # ── sanity checks
    for label, path in [
        ("Layer2 script", LAYER2_SCRIPT),
        ("Layer3 script", LAYER3_SCRIPT),
        ("Layer4 script", LAYER4_SCRIPT),
    ]:
        if not path.exists():
            log.error(f"ABORT — {label} not found: {path}")
            sys.exit(1)
    log.info("All layer scripts found. Starting pipeline...")
    log.info("")

    results = {}

    # ── Layer 2
    ok2, t2, err2 = run_layer("Layer2", LAYER2_SCRIPT, TIMEOUTS["Layer2"])
    results["Layer2"] = {"ok": ok2, "elapsed": t2, "error": err2}
    if not ok2:
        log.error("Layer 2 failed — aborting pipeline. Fix error above and rerun.")
        append_run_log(results, None)
        sys.exit(1)

    l2_count = parquet_row_count(LAYER2_OUT)
    log.info(f"Layer 2 output: {l2_count} candidates")
    if l2_count < 1:
        log.error("Layer 2 produced 0 candidates — aborting.")
        sys.exit(1)

    # ── Layer 3
    ok3, t3, err3 = run_layer("Layer3", LAYER3_SCRIPT, TIMEOUTS["Layer3"])
    results["Layer3"] = {"ok": ok3, "elapsed": t3, "error": err3}
    if not ok3:
        log.error("Layer 3 failed — aborting pipeline.")
        append_run_log(results, None)
        sys.exit(1)

    l3_count = parquet_row_count(LAYER3_OUT)
    log.info(f"Layer 3 output: {l3_count} signals")

    # ── Layer 4
    ok4, t4, err4 = run_layer("Layer4", LAYER4_SCRIPT, TIMEOUTS["Layer4"])
    results["Layer4"] = {"ok": ok4, "elapsed": t4, "error": err4}
    if not ok4:
        log.error("Layer 4 failed — portfolio not constructed.")
        # still write partial report
    
    # ── Load outputs for report
    l2_df = load_parquet_safe(LAYER2_OUT)
    l3_df = load_parquet_safe(LAYER3_OUT)
    l4_df = load_parquet_safe(LAYER4_OUT) if ok4 else None

    # ── Build + save consolidated report
    banner("BUILDING CONSOLIDATED EOD REPORT")
    report_text = build_eod_report(results, l2_df, l3_df, l4_df)
    report_path = BASE / f"nse_layer5_eod_report_{TODAY}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log.info(f"EOD report saved: {report_path.name}")

    # Print report to console
    print("\n" + report_text)

    # ── Append to run log
    append_run_log(results, l4_df)

    total_elapsed = time.time() - t_start
    banner(f"LAYER 5 COMPLETE  |  Total runtime: {total_elapsed:.1f}s  |  {TODAY}")

    # Exit with error code if any layer failed
    if not all(r["ok"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()


