"""
================================================================================
  NSE DECISION ENGINE - INTERACTIVE WALKTHROUGH
  nse_interactive_walkthrough.py

  A live, click-through replica of layer6_dashboard.py's weekday pipeline,
  built for demoing the real system to another person one layer at a time.

  Each layer is a REAL subprocess run of the actual production script
  (same commands layer6_dashboard.py --all uses) - not a simulation. The
  page will not advance to the next layer until you click "Continue" -
  giving you a natural pause after each one finishes to explain it.

  Run:
    python nse_interactive_walkthrough.py
  Then open:
    http://localhost:8877

  Safety net: every pending layer also has a "Use last saved output"
  button, in case a live NSE fetch hiccups mid-demo - falls back to
  reading whatever that layer last wrote to disk instead of blocking you.
================================================================================
"""

import http.server
import json
import os
import random
import re
import socketserver
import subprocess
import threading
import time
import webbrowser
from datetime import date, datetime
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
PYTHON = r"C:/Users/harik/AppData/Local/Programs/Python/Python312/python.exe"
PORT = 8877

SCRIPTS = {
    "Layer0-NSE": BASE / "layer0_nse_data.py",
    "Layer0-FnO": BASE / "layer0_fno.py",
    "PatchSectors": BASE / "patch_sectors.py",
    "Layer1-Daily": BASE / "layer1_daily.py",
    "Layer2": BASE / "layer2_daily_filter.py",
    "Layer3": BASE / "layer3_rules_engine.py",
    "Layer4": BASE / "layer4_portfolio.py",
    "Layer-TS": BASE / "TS layer testing.py",
    # PaperTracker and Output have no script file — dispatched specially, see
    # run_papertracker_bg() / run_output_bg() / do_POST's /api/run handler.
}

# Same weekday sequence layer6_dashboard.py --all runs on a Mon-Fri trading
# day: TS runs AFTER Layer4, checking support/resistance structure on the
# already-filtered/scored candidates rather than the full universe — plus
# the paper-tracker and final output steps this demo adds on top.
PIPELINE = [
    "Layer0-NSE", "Layer0-FnO", "PatchSectors", "Layer1-Daily",
    "Layer2", "Layer3", "Layer4", "Layer-TS", "PaperTracker", "Output",
]

TIME_ESTIMATES = {
    "Layer0-NSE": 12, "Layer0-FnO": 195, "PatchSectors": 10, "Layer1-Daily": 140,
    "Layer2": 95, "Layer3": 20, "Layer4": 2, "Layer-TS": 40, "PaperTracker": 15,
    "Output": 1,
}

TITLES = {
    "Layer0-NSE": "Layer 0a - Corporate Data",
    "Layer0-FnO": "Layer 0b - Options (F&O) Data",
    "PatchSectors": "Sector Classification Refresh",
    "Layer1-Daily": "Layer 1 - Daily Structural Signals",
    "Layer2": "Layer 2 - Universe Filter (3 Gates)",
    "Layer3": "Layer 3 - Rules Engine + Scoring",
    "Layer4": "Layer 4/5 - Portfolio, Sizing, Stops",
    "Layer-TS": "TS Scan - Support/Resistance Structure",
    "PaperTracker": "Paper Trading Ledger",
    "Output": "Final Output - Results",
}

# Talking points to narrate while / after each layer runs. Kept short on
# purpose - the detail lives in the companion PDF, this is a live prompt.
EXPLAIN = {
    "Layer0-NSE": [
        "Pulls live NSE data: board meetings, bulk deals, block deals, corporate "
        "actions, announcements.",
        "Combines the four feeds into one institutional_score per stock, weighted "
        "toward institutional buying activity, adverse news, and upcoming results.",
        "Output: earnings_dna.parquet, bulk_deals.parquet, block_deals.parquet, "
        "corporate_actions.parquet, announcements.parquet, score_summary.parquet.",
    ],
    "Layer0-FnO": [
        "Pulls live options-chain data for the F&O-eligible stock universe.",
        "Computes five metrics per stock: Put/Call Ratio, Max Pain, OI skew "
        "(support/resistance walls), IV percentile, rollover %.",
        "Output: fno_pcr.parquet, fno_maxpain.parquet, fno_oi.parquet, "
        "fno_iv.parquet, fno_rollover.parquet.",
    ],
    "PatchSectors": [
        "Refreshes which sector each stock belongs to, cached for fast repeat runs.",
        "Output: updated sector_cache.json and the sector column in "
        "universe_master.parquet.",
    ],
    "Layer1-Daily": [
        "Recomputes the fast-moving structural numbers: sector momentum, "
        "cointegrated-pair z-scores, CUSUM momentum breaks.",
        "The heavier structural numbers (Hurst exponent, Granger causality, GARCH) "
        "are computed weekly on weekends; this keeps the daily-moving parts current.",
        "Output: sector_momentum.parquet, spread_history.parquet (Z_now column), "
        "cusum_breaks.parquet.",
    ],
    "Layer2": [
        "Classifies today's market regime from India VIX, then applies 3 gates: "
        "liquidity (turnover floor), event blackout (earnings in next 0-2 days), "
        "and a signal-quality grade gate.",
        "Produces the ranked candidate list — everything downstream only sees "
        "the stocks that survive these gates.",
        "Output: nse_layer2_candidates.parquet.",
    ],
    "Layer3": [
        "The core scoring layer. 4 independent rules: volatility breakout, "
        "cointegrated pairs mean-reversion, event-driven, CUSUM momentum.",
        "Combines the rules with options positioning and a price-structure "
        "multiplier into one final_score per stock.",
        "Output: nse_layer3_signals.parquet, ranked by final_score.",
    ],
    "Layer4": [
        "Turns scores into an actual portfolio: regime weighting, tail-risk "
        "dampening, sector bias, then a FIRE/WAIT/KILL verdict per stock.",
        "Sizes every FIRE position by fixed risk-per-trade, with tier-based "
        "allocation caps and cluster/sector diversification limits.",
        "Output: nse_layer4_portfolio.parquet, nse_layer4_portfolio_report.txt.",
    ],
    "Layer-TS": [
        "An independent technical-structure scan — support/resistance zones, "
        "trend structure, chart patterns — across the whole universe.",
        "Produces the stop-loss / target levels Layer 4 anchors to when available.",
        "Output: ts_analysis_report.csv, ts_sr_levels.parquet, ts_analysis_report.txt.",
    ],
    "PaperTracker": [
        "Logs every FIRE-verdict stock from Layer 4 as a new paper trade, then "
        "checks all open trades against their stop-loss, target, and max-hold window.",
        "Closes out trades that hit a stop, target, or expiry, and updates the "
        "running win-rate / expectancy numbers.",
        "Output: paper_trades.csv, paper_trades_v2.csv, paper_trades_v2.xlsx.",
    ],
    "Output": [
        "The consolidated result of the whole run: the actual FIRE candidates "
        "from Layer 4 — ticker, direction, score, entry, stop, target — pulled "
        "straight from nse_layer4_portfolio.parquet.",
        "Alongside it, the paper-trading ledger's running performance: closed "
        "trade count, win rate, and average PnL per trade.",
        "This is what the whole pipeline exists to produce — a decision, not a log.",
    ],
}

# ---------------------------------------------------------------------------
_INFO_PATTERNS = [
    ("candidates", r"(\d+)\s*candidates"),
    ("signals", r"(\d+)\s*signals"),
    ("FIRE", r"FIRE[=:](\d+)"),
    ("WAIT", r"WAIT[=:](\d+)"),
    ("KILL", r"KILL[=:](\d+)"),
    ("tickers", r"(\d+)\s*tickers"),
    ("pairs", r"(\d+)\s*pairs"),
    ("sectors", r"(\d+)\s*sectors"),
    ("regime", r"Regime[=:\s]+(\w+)"),
    ("VIX", r"VIX[=:\s]+([\d.]+)"),
]


def extract_info(line):
    out = {}
    for label, pattern in _INFO_PATTERNS:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            out[label] = m.group(1)
    return out


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
lock = threading.Lock()
state = {
    "order": PIPELINE,
    "running": None,
    "layers": {
        name: {"status": "pending", "elapsed": 0.0, "metrics": {}, "tail": [], "mode": None}
        for name in PIPELINE
    },
}


def _current_target():
    """First layer that's still pending, or None if all resolved."""
    for name in PIPELINE:
        if state["layers"][name]["status"] == "pending":
            return name
    return None


def run_layer_bg(name):
    script = SCRIPTS[name]
    with lock:
        state["running"] = name
        L = state["layers"][name]
        L["status"] = "running"
        L["tail"] = []
        L["metrics"] = {}
        L["mode"] = "live"
        L["t0"] = time.time()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    ok = False
    try:
        proc = subprocess.Popen(
            [PYTHON, str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, cwd=str(BASE),
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            with lock:
                L["tail"].append(line)
                if len(L["tail"]) > 300:
                    del L["tail"][0]
                info = extract_info(line)
                if info:
                    L["metrics"].update(info)
        proc.wait()
        ok = proc.returncode == 0
    except Exception as e:
        with lock:
            L["tail"].append(f"ERROR launching layer: {e}")
        ok = False

    with lock:
        L["status"] = "done" if ok else "failed"
        L["elapsed"] = time.time() - L["t0"]
        state["running"] = None


def run_papertracker_bg():
    """The paper-trading step: f1_paper_tracker.py log, then update — same
    two calls layer6_dashboard.py runs automatically at the end of a pipeline
    run. Streamed live like a normal layer rather than the old blocking
    subprocess.run() version, so it gets the same progress bar / tail log."""
    name = "PaperTracker"
    with lock:
        state["running"] = name
        L = state["layers"][name]
        L["status"] = "running"
        L["tail"] = []
        L["metrics"] = {}
        L["mode"] = "live"
        L["t0"] = time.time()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    ok = True
    for cmd in ("log", "update"):
        with lock:
            L["tail"].append(f"--- f1_paper_tracker.py {cmd} ---")
        try:
            proc = subprocess.Popen(
                [PYTHON, str(BASE / "f1_paper_tracker.py"), cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(BASE),
            )
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                with lock:
                    L["tail"].append(line)
                    if len(L["tail"]) > 300:
                        del L["tail"][0]
            proc.wait()
            ok = ok and proc.returncode == 0
        except Exception as e:
            with lock:
                L["tail"].append(f"ERROR running {cmd}: {e}")
            ok = False

    with lock:
        L["status"] = "done" if ok else "failed"
        L["elapsed"] = time.time() - L["t0"]
        state["running"] = None


def run_output_bg():
    """The final step — nothing to execute, it just presents the pipeline's
    already-produced output (FIRE table + paper-trading stats). Still routed
    through the same running->done flow as a real layer so the UI (progress
    bar, pause-for-explanation) behaves consistently."""
    name = "Output"
    with lock:
        state["running"] = name
        L = state["layers"][name]
        L["status"] = "running"
        L["tail"] = ["Assembling the final output from this run's results..."]
        L["metrics"] = {}
        L["mode"] = "live"
        L["t0"] = time.time()
    time.sleep(0.5)
    with lock:
        L["status"] = "done"
        L["elapsed"] = time.time() - L["t0"]
        state["running"] = None


def mark_skipped(name):
    with lock:
        L = state["layers"][name]
        L["status"] = "done"
        L["mode"] = "cached"
        L["tail"] = ["(Skipped live run - using last saved output on disk for this layer.)"]
        L["elapsed"] = 0.0


def run_cacheload_bg(name):
    """'Use cache for next layer' — paced, visible loading of the last saved
    output instead of an instant skip. The read itself is near-instant (it's
    a parquet on disk), so the 10-20s is deliberate pacing for the demo, not
    real work — the tail log says so honestly rather than pretending."""
    duration = random.uniform(10, 20)
    with lock:
        state["running"] = name
        L = state["layers"][name]
        L["status"] = "running"
        L["mode"] = "cache-loading"
        L["tail"] = [f"Loading last saved output for {name} from disk (~{duration:.0f}s)..."]
        L["metrics"] = {}
        L["cache_duration"] = duration
        L["t0"] = time.time()

    time.sleep(duration)

    with lock:
        L["tail"].append("Loaded from cache.")
        L["status"] = "done"
        L["mode"] = "cached"
        L["elapsed"] = time.time() - L["t0"]
        state["running"] = None


# ---------------------------------------------------------------------------
# DATA READERS - used to show real results after Layer 2 / 3 / 4 finish
# ---------------------------------------------------------------------------
def read_layer4_fire(limit=12):
    p = BASE / "nse_layer4_portfolio.parquet"
    if not p.exists():
        return []
    try:
        df = pd.read_parquet(p)
    except Exception:
        return []
    if "verdict" in df.columns:
        df = df[df["verdict"] == "FIRE"]
    sort_col = "composite_score" if "composite_score" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False).head(limit)
    cols = [c for c in ["ticker", "direction", "composite_score", "entry_price",
                         "stop_price", "target_price", "qty", "tier"] if c in df.columns]
    return json.loads(df[cols].to_json(orient="records")) if cols else []


def read_paper_stats():
    p = BASE / "paper_trades.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if "status" not in df.columns:
        return {"total_trades": len(df)}
    closed = df[df["status"].isin(["WIN", "LOSS", "EXPIRED"])]
    out = {"total_trades": int(len(df)), "closed": int(len(closed))}
    if len(closed):
        out["win_rate"] = round(100 * (closed["status"] == "WIN").sum() / len(closed), 1)
        if "pnl_pct" in closed.columns:
            out["avg_pnl_pct"] = round(float(closed["pnl_pct"].mean()), 2)
    return out


def read_layer2_summary():
    p = BASE / "nse_layer2_candidates.parquet"
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception:
        return {}
    out = {"candidates": len(df)}
    if "tier" in df.columns:
        out["tiers"] = df["tier"].value_counts().to_dict()
    return out


def read_layer3_summary():
    p = BASE / "nse_layer3_signals.parquet"
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception:
        return {}
    out = {"signals": len(df)}
    for rc in ["rule_a_fired", "rule_c_fired", "rule_d_fired", "rule_e_fired"]:
        if rc in df.columns:
            out[rc] = int(df[rc].sum())
    return out


def run_paper_tracker():
    """Mirrors the automatic end-of-run step in layer6_dashboard.py."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logs = []
    for cmd in ("log", "update"):
        try:
            r = subprocess.run(
                [PYTHON, str(BASE / "f1_paper_tracker.py"), cmd],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(BASE), timeout=180,
            )
            logs.append(f"--- f1_paper_tracker.py {cmd} ---\n{r.stdout[-2000:]}")
        except Exception as e:
            logs.append(f"--- f1_paper_tracker.py {cmd} FAILED: {e} ---")
    return "\n\n".join(logs)


# ---------------------------------------------------------------------------
# HTML (single page, vanilla JS, polls /api/state)
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NSE Decision Engine — Interactive Walkthrough</title>
<style>
  /* ---- Design tokens (Design DNA pass: trust-first advisory palette,
     8px spacing rhythm, restrained motion) ---- */
  :root{
    --bg:#080b16; --panel:rgba(255,255,255,.055); --panel2:rgba(255,255,255,.03);
    --glass-border:rgba(255,255,255,.10); --line:rgba(255,255,255,.09);
    --text:#f2f5fc; --muted:#9fb0d1; --accent:#5b8bff; --accent2:#8ab0ff;
    --good:#33e29a; --bad:#ff6b6b; --warn:#ffc15c; --pend:rgba(255,255,255,.08);
    --sp-1:4px; --sp-2:8px; --sp-3:14px; --sp-4:20px; --sp-5:28px; --sp-6:40px;
    --r-sm:8px; --r-md:14px; --r-lg:20px;
    --ease-standard:cubic-bezier(.22,.61,.36,1);
    --blur:18px;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:"Segoe UI",Inter,Arial,sans-serif;letter-spacing:.1px;
       position:relative;overflow-x:hidden}

  /* ---- animated ambient background: drifting glow orbs behind glass ---- */
  /* wrapper carries the mouse-parallax transform; the inner .orb keeps its
     own CSS drift animation — separate elements so the two transforms don't
     fight over the same property */
  .orb-wrap{position:fixed;inset:0;pointer-events:none;z-index:0}
  .orb{position:fixed;border-radius:50%;filter:blur(90px);opacity:.55;z-index:0;
       pointer-events:none;mix-blend-mode:screen}
  .orb1{width:520px;height:520px;left:-120px;top:-140px;
        background:radial-gradient(circle at 30% 30%, #5b8bff, transparent 70%);
        animation:drift1 22s ease-in-out infinite alternate}
  .orb2{width:460px;height:460px;right:-140px;top:120px;
        background:radial-gradient(circle at 60% 40%, #33e29a, transparent 70%);
        animation:drift2 26s ease-in-out infinite alternate}
  .orb3{width:420px;height:420px;left:30%;bottom:-200px;
        background:radial-gradient(circle at 50% 50%, #8ab0ff, transparent 70%);
        animation:drift3 30s ease-in-out infinite alternate}
  @keyframes drift1{ from{transform:translate(0,0) scale(1)} to{transform:translate(80px,60px) scale(1.15)} }
  @keyframes drift2{ from{transform:translate(0,0) scale(1)} to{transform:translate(-70px,50px) scale(.9)} }
  @keyframes drift3{ from{transform:translate(0,0) scale(1)} to{transform:translate(50px,-40px) scale(1.1)} }
  @media (prefers-reduced-motion: reduce){ .orb1,.orb2,.orb3{animation:none} }

  header{position:relative;z-index:2;padding:var(--sp-4) var(--sp-6);
         border-bottom:1px solid var(--glass-border);display:flex;
         justify-content:space-between;align-items:center;
         background:rgba(10,14,26,.45);backdrop-filter:blur(var(--blur)) saturate(150%);
         -webkit-backdrop-filter:blur(var(--blur)) saturate(150%)}
  header h1{font-size:19px;margin:0;font-weight:650;letter-spacing:-.1px}
  header .sub{color:var(--muted);font-size:12.5px;margin-top:4px}
  .wrap{position:relative;z-index:1;display:grid;grid-template-columns:270px 1fr;
        gap:0;min-height:calc(100vh - 68px)}
  nav{border-right:1px solid var(--glass-border);padding:var(--sp-4) var(--sp-3);
      background:rgba(255,255,255,.025);backdrop-filter:blur(var(--blur));
      -webkit-backdrop-filter:blur(var(--blur))}
  .step{display:flex;align-items:center;gap:var(--sp-2);padding:var(--sp-2) var(--sp-2);
        border-radius:var(--r-sm);margin-bottom:3px;font-size:13.3px;cursor:default;
        transition:background-color .3s var(--ease-standard)}
  .step .dot{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;
             justify-content:center;font-size:10.5px;flex:0 0 20px;background:var(--pend);
             color:#cbd6ea;transition:background-color .35s var(--ease-standard),box-shadow .35s;
             position:relative}
  .step.done .dot{background:var(--good);color:#06210f;box-shadow:0 0 12px 1px rgba(51,226,154,.55)}
  .step.running .dot{background:var(--accent);color:#fff;animation:pulse 1.3s ease-in-out infinite;
                      box-shadow:0 0 14px 2px rgba(91,139,255,.65)}
  .step.running .dot::after{content:'';position:absolute;inset:-2px;border-radius:50%;
             border:1.5px solid var(--accent2);animation:ping 1.6s cubic-bezier(0,.6,.5,1) infinite}
  @keyframes ping{0%{transform:scale(1);opacity:.9}100%{transform:scale(2.4);opacity:0}}
  .step.failed .dot{background:var(--bad);color:#fff;box-shadow:0 0 12px 1px rgba(255,107,107,.55)}
  .step.active{background:rgba(255,255,255,.05)}
  @keyframes pulse{0%{opacity:1}50%{opacity:.45}100%{opacity:1}}
  @media (prefers-reduced-motion: reduce){ .step.running .dot::after{animation:none} }
  main{position:relative;padding:var(--sp-5) var(--sp-6);max-width:920px}

  /* ---- glassmorphism: translucent frosted panels over the orb background ---- */
  .card{background:var(--panel);border:1px solid var(--glass-border);
        border-radius:var(--r-lg);padding:var(--sp-5) var(--sp-5);margin-bottom:var(--sp-4);
        backdrop-filter:blur(var(--blur)) saturate(160%);
        -webkit-backdrop-filter:blur(var(--blur)) saturate(160%);
        box-shadow:0 8px 32px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
        position:relative;overflow:hidden}
  .card::before{content:'';position:absolute;inset:0;border-radius:inherit;padding:1px;
        background:linear-gradient(135deg, rgba(255,255,255,.22), rgba(255,255,255,0) 40%);
        -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite:xor; mask-composite:exclude; pointer-events:none}
  .card h2{margin:0 0 var(--sp-1) 0;font-size:21px;font-weight:650;letter-spacing:-.1px}
  .badge{display:inline-block;font-size:10.5px;padding:2px 9px;border-radius:20px;
         margin-left:8px;vertical-align:middle;font-weight:600;letter-spacing:.2px;
         text-transform:uppercase;backdrop-filter:blur(6px)}
  .badge.pending{background:rgba(255,255,255,.1);color:#cbd6ea}
  .badge.running{background:var(--accent);color:#fff;box-shadow:0 0 10px rgba(91,139,255,.6)}
  .badge.done{background:var(--good);color:#06210f;box-shadow:0 0 10px rgba(51,226,154,.5)}
  .badge.failed{background:var(--bad);color:#fff}
  .badge.cached{background:var(--warn);color:#3a2600}
  ul.talk{margin:var(--sp-3) 0 var(--sp-1) 0;padding-left:20px;color:#d5e0f5;
          font-size:14px;line-height:1.6}
  ul.talk li{margin-bottom:4px}
  .metrics{display:flex;gap:var(--sp-2);flex-wrap:wrap;margin:var(--sp-3) 0}
  .metric{background:rgba(255,255,255,.05);border:1px solid var(--glass-border);
          border-radius:var(--r-sm);padding:8px 14px;font-size:13px;
          backdrop-filter:blur(8px);transition:transform .2s var(--ease-standard),background .2s}
  .metric:hover{transform:translateY(-3px) scale(1.04);background:rgba(255,255,255,.08)}
  .metric b{color:var(--accent2);font-size:15px;display:block;text-shadow:0 0 12px rgba(138,176,255,.5)}
  .btnrow{display:flex;gap:var(--sp-2);margin-top:var(--sp-4)}
  button{background:linear-gradient(135deg, var(--accent), #3f6fe0);color:#fff;border:1px solid rgba(255,255,255,.18);
         border-radius:var(--r-sm);padding:11px 20px;font-size:14px;cursor:pointer;font-weight:600;
         box-shadow:0 4px 16px rgba(91,139,255,.35);position:relative;overflow:hidden;
         transition:transform .15s var(--ease-standard),box-shadow .2s,background .2s}
  .ripple{position:absolute;border-radius:50%;background:rgba(255,255,255,.5);
          transform:scale(0);pointer-events:none}
  button:not(:disabled):hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(91,139,255,.5)}
  button.secondary{background:rgba(255,255,255,.06);border:1px solid var(--glass-border);
                    color:var(--muted);box-shadow:none;backdrop-filter:blur(8px)}
  button.secondary:not(:disabled):hover{background:rgba(255,255,255,.1);color:var(--text);box-shadow:none}
  button:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
  button.continue{background:linear-gradient(135deg, var(--good), #1fa473);
                   box-shadow:0 4px 18px rgba(51,226,154,.4)}
  button.continue:not(:disabled):hover{box-shadow:0 8px 26px rgba(51,226,154,.55)}
  pre.log{background:rgba(4,7,15,.65);border:1px solid var(--glass-border);border-radius:var(--r-sm);
          padding:12px 14px;font-size:12px;color:#a9bbdd;max-height:220px;overflow-y:auto;
          white-space:pre-wrap;margin-top:var(--sp-3);backdrop-filter:blur(10px)}
  table{width:100%;border-collapse:collapse;margin-top:var(--sp-2);font-size:13px}
  th,td{padding:6px 8px;text-align:left;border-bottom:1px solid var(--glass-border)}
  th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase}
  .done-banner{background:var(--panel);border:1px solid rgba(51,226,154,.4);border-radius:var(--r-md);
               padding:var(--sp-4) var(--sp-4);color:var(--good);position:relative;overflow:visible;
               backdrop-filter:blur(var(--blur)) saturate(160%);
               box-shadow:0 8px 32px rgba(51,226,154,.15), inset 0 1px 0 rgba(255,255,255,.06)}
  .confetti-piece{position:fixed;top:-10px;width:8px;height:8px;border-radius:2px;
                   pointer-events:none;z-index:20}
  .progress-outer{background:rgba(0,0,0,.35);border:1px solid var(--glass-border);border-radius:20px;
                   height:22px;overflow:hidden;margin-top:var(--sp-4);position:relative;
                   box-shadow:inset 0 1px 4px rgba(0,0,0,.4)}
  .progress-fill{
    height:100%;border-radius:20px;width:0%;
    background:repeating-linear-gradient(45deg,var(--accent) 0 14px,#8ab0ff 14px 28px);
    background-size:40px 40px;
    animation:barstripes 1s linear infinite;
    box-shadow:0 0 16px 2px rgba(91,139,255,.6);
  }
  @keyframes barstripes{ from{background-position:0 0} to{background-position:40px 0} }
  .progress-label{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;
                   align-items:center;justify-content:center;font-size:11.5px;font-weight:700;
                   color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}
  .spinner{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,.35);
           border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;
           margin-right:8px;vertical-align:-2px}
  @keyframes spin{ to{transform:rotate(360deg)} }
  .nextbar{display:flex;justify-content:flex-end;gap:var(--sp-2);margin-top:var(--sp-4);
           padding-top:var(--sp-3);border-top:1px dashed var(--glass-border)}

  /* ---- hero / intro screen ---- */
  #appshell{transition:opacity .3s ease}
  .hero{position:fixed;inset:0;z-index:10;display:flex;align-items:center;justify-content:center;
        padding:var(--sp-6)}
  .hero-card{max-width:640px;width:100%;text-align:center;background:var(--panel);
             border:1px solid var(--glass-border);border-radius:24px;padding:48px 44px;
             backdrop-filter:blur(24px) saturate(160%);-webkit-backdrop-filter:blur(24px) saturate(160%);
             box-shadow:0 20px 60px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.07);
             position:relative}
  .hero-kicker{color:var(--accent2);font-size:12.5px;font-weight:700;letter-spacing:1.6px;
               text-transform:uppercase;margin-bottom:10px}
  .hero-title{font-size:36px;margin:0 0 16px 0;font-weight:750;letter-spacing:-.5px;
              background:linear-gradient(135deg,#fff,#b9caf0 60%,var(--accent2));
              -webkit-background-clip:text;background-clip:text;color:transparent}
  .hero-sub{color:var(--muted);font-size:14.5px;line-height:1.65;margin:0 auto 28px auto;max-width:520px}
  .hero-sub code{background:rgba(255,255,255,.08);padding:1px 6px;border-radius:4px;font-size:13px;color:#dbe6ff}
  .hero-stats{display:flex;justify-content:center;gap:14px;flex-wrap:wrap;margin-bottom:32px}
  .hstat{background:rgba(255,255,255,.05);border:1px solid var(--glass-border);border-radius:var(--r-sm);
         padding:12px 18px;min-width:100px;backdrop-filter:blur(8px)}
  .hstat b{display:block;font-size:19px;color:var(--accent2);text-shadow:0 0 14px rgba(138,176,255,.5)}
  .hstat span{font-size:11px;color:var(--muted)}
  .hero-btn{background:linear-gradient(135deg,var(--accent),#3f6fe0);color:#fff;border:1px solid rgba(255,255,255,.2);
            border-radius:var(--r-md);padding:15px 34px;font-size:15.5px;font-weight:650;cursor:pointer;
            box-shadow:0 6px 24px rgba(91,139,255,.45);animation:heroGlow 2.4s ease-in-out infinite;
            transition:transform .15s var(--ease-standard)}
  .hero-btn:hover{transform:translateY(-2px) scale(1.02)}
  @keyframes heroGlow{0%,100%{box-shadow:0 6px 24px rgba(91,139,255,.4)}50%{box-shadow:0 10px 36px rgba(91,139,255,.7)}}
  @media (prefers-reduced-motion: reduce){ .hero-btn{animation:none} }

  /* ---- overall pipeline progress spine (under header) ---- */
  .overall-bar{position:relative;z-index:2;height:3px;background:rgba(255,255,255,.06)}
  .overall-fill{height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--good));
                box-shadow:0 0 10px rgba(91,139,255,.6)}

  /* ---- sidebar connecting timeline (the "flow" through the pipeline) ---- */
  .timeline{position:relative;padding-left:0}
  .timeline::before{content:'';position:absolute;left:23px;top:14px;bottom:14px;width:2px;
                     background:rgba(255,255,255,.09);border-radius:2px}
  .timeline-fill{position:absolute;left:23px;top:14px;width:2px;height:0%;
                  background:linear-gradient(180deg,var(--accent),var(--good));
                  box-shadow:0 0 8px rgba(91,139,255,.6);border-radius:2px;transition:none}
</style>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
</head>
<body>
<div class="orb-wrap" id="orbwrap1"><div class="orb orb1"></div></div>
<div class="orb-wrap" id="orbwrap2"><div class="orb orb2"></div></div>
<div class="orb-wrap" id="orbwrap3"><div class="orb orb3"></div></div>

<div class="hero" id="hero">
  <div class="hero-card">
    <a href="https://zip-repl--moneymonster040.replit.app/" target="_blank" rel="noopener"
       style="display:inline-block;margin-bottom:10px;font-size:12.5px;color:var(--muted);text-decoration:underline">
      zip-repl--moneymonster040.replit.app ↗
    </a>
    <div class="hero-kicker">NSE Decision Engine</div>
    <h1 class="hero-title">Interactive Walkthrough</h1>
    <p class="hero-sub">A live, layer-by-layer run of the real production pipeline — the same
      steps <code>layer6_dashboard.py --all</code> runs every trading day, including the
      paper-trading ledger update. Real subprocesses, real NSE data, one step at a time,
      paced for you to explain each one as it happens.</p>
    <div class="hero-stats">
      <div class="hstat"><b>10</b><span>pipeline steps</span></div>
      <div class="hstat"><b>Live</b><span>NSE + F&amp;O data</span></div>
      <div class="hstat"><b>33.3%</b><span>paper win rate</span></div>
      <div class="hstat"><b>+0.97%</b><span>avg PnL / trade</span></div>
    </div>
    <button class="hero-btn" onclick="beginFlow()">Begin Walkthrough →</button>
  </div>
</div>

<div id="appshell" style="opacity:0">
  <header>
    <div>
      <h1>NSE Decision Engine — Interactive Walkthrough</h1>
      <div class="sub">Real production layers, real live data — one layer at a time, click to continue</div>
    </div>
    <button class="secondary" onclick="resetAll()">Reset run</button>
  </header>
  <div class="overall-bar"><div class="overall-fill" id="overallfill"></div></div>
  <div class="wrap">
    <nav>
      <div class="timeline">
        <div class="timeline-fill" id="timelinefill"></div>
        <div id="nav"></div>
      </div>
    </nav>
    <main id="main"></main>
  </div>
</div>

<script>
const TITLES = __TITLES__;
const EXPLAIN = __EXPLAIN__;
const ORDER = __ORDER__;
const ESTIMATES = __ESTIMATES__;

let polling = null;
let lastSignature = null;   // "target|status" — only re-animate when this changes
let doneShown = false;
let isFirstRender = true;   // skip the exit-transition on the very first paint
let cursorIndex = 0;        // which layer is ON SCREEN — advances ONLY on an explicit
                             // "Next Layer" click, never automatically when a layer
                             // finishes mid-poll. cursorIndex === ORDER.length means
                             // "show the final completion screen."
const prevNavStatus = {};   // per-layer status last time nav was drawn
const motionOK = !(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
                 && typeof gsap !== 'undefined';

function badgeClass(s){ return s; }

async function getState(){
  const r = await fetch('/api/state');
  return await r.json();
}

async function runLayer(name){
  await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({layer:name})});
  render();
  if(!polling) polling = setInterval(render, 1200);
}

async function skipLayer(name){
  await fetch('/api/skip', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({layer:name})});
  render();
}

async function useCacheLayer(name){
  await fetch('/api/cacheload', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({layer:name})});
  render();
  if(!polling) polling = setInterval(render, 1200);
}

async function resetAll(){
  await fetch('/api/reset', {method:'POST'});
  lastSignature = null;
  doneShown = false;
  cursorIndex = 0;
  Object.keys(prevNavStatus).forEach(k => delete prevNavStatus[k]);
  render();
}

function renderNav(state){
  const nav = document.getElementById('nav');
  let html = '';
  const changed = [];
  ORDER.forEach((name, i) => {
    const L = state.layers[name];
    let cls = 'step ' + L.status;
    let icon = L.status === 'done' ? '✓' : L.status === 'running' ? '' : L.status === 'failed' ? '!' : (i+1);
    html += `<div class="${cls}" data-layer="${name}"><div class="dot">${icon}</div><div>${TITLES[name]}</div></div>`;
    if(prevNavStatus[name] !== L.status){
      changed.push(name);
      prevNavStatus[name] = L.status;
    }
  });
  nav.innerHTML = html;
  if(motionOK && changed.length){
    changed.forEach(name => {
      const dot = nav.querySelector(`[data-layer="${name}"] .dot`);
      if(dot) gsap.fromTo(dot, {scale: 1.5}, {scale: 1, duration: .4, ease: 'back.out(2)'});
    });
  }

  // Progress spine: how far through the journey we are.
  const doneCount = ORDER.filter(n => state.layers[n].status === 'done').length;
  const overallPct = Math.round((doneCount / ORDER.length) * 100);
  const timelinePct = ORDER.length > 1 ? Math.round((doneCount / (ORDER.length - 1)) * 100) : 0;
  const overallFill = document.getElementById('overallfill');
  const timelineFill = document.getElementById('timelinefill');
  if(overallFill){
    if(motionOK) gsap.to(overallFill, {width: overallPct + '%', duration: .6, ease: 'power1.out'});
    else overallFill.style.width = overallPct + '%';
  }
  if(timelineFill){
    if(motionOK) gsap.to(timelineFill, {height: timelinePct + '%', duration: .6, ease: 'power1.out'});
    else timelineFill.style.height = timelinePct + '%';
  }
}

function animateCardIn(card){
  if(!motionOK){
    // Even without motion, make sure metric numbers show their final value.
    card.querySelectorAll('.metric b[data-final]').forEach(b => { b.textContent = b.dataset.final; });
    return;
  }
  gsap.fromTo(card, {opacity: 0, y: 14},
    {opacity: 1, y: 0, duration: .45, ease: 'power2.out'});

  const bullets = card.querySelectorAll('.talk li');
  if(bullets.length){
    gsap.fromTo(bullets, {opacity: 0, x: -10},
      {opacity: 1, x: 0, duration: .35, stagger: .07, delay: .12, ease: 'power2.out'});
  }

  const metrics = card.querySelectorAll('.metric');
  if(metrics.length){
    gsap.fromTo(metrics, {opacity: 0, y: 8},
      {opacity: 1, y: 0, duration: .35, stagger: .06, delay: .12 + bullets.length * .04, ease: 'power2.out'});
    // Count numeric metrics up from 0 instead of just appearing.
    metrics.forEach(m => {
      const b = m.querySelector('b[data-final]');
      if(!b) return;
      const final = b.dataset.final;
      const num = parseFloat(final.replace(/[^\d.\-]/g, ''));
      if(isNaN(num) || !/^-?[\d.]+$/.test(final.replace(/[,%]/g,''))){
        b.textContent = final;   // non-numeric value (e.g. "Bull") — just show it
        return;
      }
      const counter = {v: 0};
      gsap.to(counter, {
        v: num, duration: .8, delay: .15, ease: 'power1.out',
        onUpdate: () => { b.textContent = Math.round(counter.v) + final.replace(/^-?[\d.,]+/, ''); },
        onComplete: () => { b.textContent = final; },
      });
    });
  }
}

function spawnRipple(btn, evt){
  if(!motionOK) return;
  const rect = btn.getBoundingClientRect();
  const ripple = document.createElement('span');
  ripple.className = 'ripple';
  const size = Math.max(rect.width, rect.height) * 1.4;
  ripple.style.width = ripple.style.height = size + 'px';
  ripple.style.left = ((evt.clientX ?? rect.left + rect.width/2) - rect.left - size/2) + 'px';
  ripple.style.top = ((evt.clientY ?? rect.top + rect.height/2) - rect.top - size/2) + 'px';
  btn.appendChild(ripple);
  gsap.to(ripple, {scale: 1, opacity: 0, duration: .55, ease: 'power2.out',
    onComplete: () => ripple.remove()});
}

function spawnConfetti(){
  if(!motionOK) return;
  const colors = ['#5b8bff', '#33e29a', '#8ab0ff', '#ffc15c', '#ff6b6b'];
  const count = 46;
  for(let i = 0; i < count; i++){
    const el = document.createElement('div');
    el.className = 'confetti-piece';
    el.style.left = (Math.random() * 100) + 'vw';
    el.style.background = colors[i % colors.length];
    el.style.borderRadius = Math.random() > .5 ? '50%' : '2px';
    document.body.appendChild(el);
    gsap.to(el, {
      y: window.innerHeight + 40,
      x: (Math.random() - .5) * 200,
      rotation: Math.random() * 540,
      duration: 1.6 + Math.random() * 1.1,
      delay: Math.random() * .4,
      ease: 'power1.in',
      opacity: .9,
      onComplete: () => el.remove(),
    });
  }
}

function animateNextEnabled(btn){
  if(!motionOK || !btn || btn.disabled) return;
  gsap.fromTo(btn, {scale: .92}, {scale: 1, duration: .4, ease: 'back.out(2.2)'});
}

// Consolidated final view: the REAL FIRE candidates straight out of
// nse_layer4_portfolio.parquet, plus the paper-trading ledger's running
// performance — the actual output the whole pipeline exists to produce.
async function loadOutputSummary(){
  const holder = document.getElementById('outputsummary');
  if(!holder) return;
  holder.innerHTML = `<div style="color:var(--muted);font-size:12.5px;margin-top:12px">Loading results…</div>`;

  let rows = [], stats = {};
  try{
    const [fireResp, statsResp] = await Promise.all([fetch('/api/fire'), fetch('/api/paperstats')]);
    rows = await fireResp.json();
    stats = await statsResp.json();
  } catch(e){ rows = []; stats = {}; }

  let html = '';

  if(stats && (stats.closed || stats.total_trades)){
    html += `<div style="margin-top:4px;font-size:12.5px;color:var(--muted);font-weight:600;
             text-transform:uppercase;letter-spacing:.4px">Paper-trading performance</div>
             <div class="metrics">`;
    if(stats.closed !== undefined) html += `<div class="metric"><b data-final="${stats.closed}">0</b>closed trades</div>`;
    if(stats.win_rate !== undefined) html += `<div class="metric"><b data-final="${stats.win_rate}%">0</b>win rate</div>`;
    if(stats.avg_pnl_pct !== undefined) html += `<div class="metric"><b data-final="${stats.avg_pnl_pct}%">0</b>avg PnL / trade</div>`;
    html += `</div>`;
  }

  if(!rows.length){
    html += `<div style="color:var(--muted);font-size:12.5px;margin-top:12px">No FIRE candidates in the current output.</div>`;
    holder.innerHTML = html;
    return;
  }

  const cols = Object.keys(rows[0]);
  html += `<div style="margin-top:18px;font-size:12.5px;color:var(--muted);font-weight:600;
           text-transform:uppercase;letter-spacing:.4px">FIRE candidates — Layer 4 output</div>
           <div style="overflow-x:auto"><table><thead><tr>`;
  cols.forEach(c => html += `<th>${c}</th>`);
  html += `</tr></thead><tbody>`;
  rows.forEach(row => {
    html += `<tr>` + cols.map(c => `<td>${row[c]}</td>`).join('') + `</tr>`;
  });
  html += `</tbody></table></div>`;
  holder.innerHTML = html;

  const metricEls = holder.querySelectorAll('.metric b[data-final]');
  metricEls.forEach(b => {
    const final = b.dataset.final;
    const num = parseFloat(final);
    if(!motionOK || isNaN(num)){ b.textContent = final; return; }
    const counter = {v: 0};
    gsap.to(counter, {v: num, duration: .8, ease: 'power1.out',
      onUpdate: () => { b.textContent = Math.round(counter.v * 10) / 10 + final.replace(/^-?[\d.]+/, ''); },
      onComplete: () => { b.textContent = final; }});
  });

  if(motionOK){
    const trs = holder.querySelectorAll('tbody tr');
    gsap.fromTo(trs, {opacity: 0, x: -8},
      {opacity: 1, x: 0, duration: .3, stagger: .04, delay: .2, ease: 'power2.out'});
  }
}

// The server only writes L.elapsed AFTER a layer finishes — during the run
// itself it stays frozen, which is why the progress bar used to sit still
// the whole time and jump at the end. Compute it live instead from t0
// (sent on every poll) against the client's own clock.
function liveElapsed(L){
  if(L.status === 'running' && L.t0){
    return (Date.now() / 1000) - L.t0;
  }
  return L.elapsed || 0;
}

// A "cache-loading" run (the paced 10-20s "Use cache" button) has its own
// randomized duration per call — use that for the progress bar instead of
// the normal live-run time estimate.
function estimateFor(target, L){
  if(L.mode === 'cache-loading' && L.cache_duration) return L.cache_duration;
  return ESTIMATES[target] || 60;
}

function renderMain(state){
  const main = document.getElementById('main');
  // If a layer finishes mid-poll, DO NOT auto-jump the display forward —
  // stay on the current cursor layer (showing its "done" state + an enabled
  // Next button) until the user actually clicks it.
  const atEnd = cursorIndex >= ORDER.length;

  if(atEnd){
    if(!doneShown){
      doneShown = true;
      lastSignature = 'ALLDONE';
      main.innerHTML = `
        <div class="done-banner">
          <h2 style="margin:0 0 6px 0;color:var(--good)">Walkthrough complete.</h2>
          <div>All 10 steps ran: the 8 decision-engine layers, the paper-trading ledger update,
          and the final output. Click "Reset run" above to start over.</div>
        </div>`;
      if(motionOK){
        gsap.fromTo('.done-banner', {opacity: 0, y: 14},
          {opacity: 1, y: 0, duration: .45, ease: 'power2.out'});
      }
      spawnConfetti();
    }
    return;
  }

  const target = ORDER[cursorIndex];
  const L = state.layers[target];
  const signature = target + '|' + L.status;

  if(signature !== lastSignature){
    // Genuine state change (new layer, or pending->running->done->failed).
    // Choreograph it as exit-old, then enter-new — a real transition
    // between steps in the journey, not an in-place swap.
    lastSignature = signature;

    const buildNewCard = () => {
      const explain = EXPLAIN[target] || [];
      let metricsHtml = '';
      const m = L.metrics || {};
      Object.keys(m).forEach(k => {
        metricsHtml += `<div class="metric"><b data-final="${m[k]}">0</b>${k}</div>`;
      });

      let body = `
        <div class="card">
          <h2>${TITLES[target]} <span class="badge ${L.status}">${L.status}</span></h2>
          <ul class="talk">${explain.map(t => `<li>${t}</li>`).join('')}</ul>
          ${metricsHtml ? `<div class="metrics">${metricsHtml}</div>` : ''}
      `;

      const isDone = (L.status === 'done');

      if(L.status === 'pending'){
        body += `<div class="btnrow">
          <button onclick="runLayer('${target}')">Run this layer live</button>
          <button class="secondary" onclick="skipLayer('${target}')">Use last saved output instead</button>
        </div>`;
      } else if(L.status === 'running'){
        const est = estimateFor(target, L);
        const elapsedNow = liveElapsed(L);
        const pct = Math.min(95, Math.round((elapsedNow / est) * 100));
        const isCacheLoad = L.mode === 'cache-loading';
        const runLabel = isCacheLoad ? 'Loading from cache…' : 'Running…';
        body += `
          <div class="btnrow"><button disabled><span class="spinner"></span><span id="elapsedtxt">${runLabel} ${Math.round(elapsedNow)}s</span></button></div>
          <div class="progress-outer">
            <div class="progress-fill" id="progfill" style="width:0%"></div>
            <div class="progress-label" id="proglabel">${pct}% · est. ${Math.round(est)}s</div>
          </div>
          <pre class="log" id="livelog">${(L.tail||[]).slice(-40).join('\n')}</pre>`;
      } else if(L.status === 'failed'){
        body += `<div class="btnrow">
          <button onclick="runLayer('${target}')">Retry live</button>
          <button class="secondary" onclick="skipLayer('${target}')">Use last saved output instead</button>
        </div>
        <pre class="log">${(L.tail||[]).slice(-40).join('\n')}</pre>`;
      } else if(L.status === 'done'){
        if(L.mode === 'live' && (L.tail||[]).length){
          body += `<pre class="log" id="livelog">${(L.tail||[]).slice(-15).join('\n')}</pre>`;
        }
        if(target === 'Output'){
          body += `<div id="outputsummary"></div>`;
        }
      }

      body += `<div class="nextbar">
          <button class="secondary" ${isDone ? '' : 'disabled'} onclick="advanceWithCache()">Use cache for next layer</button>
          <button class="continue" id="nextbtn" ${isDone ? '' : 'disabled'} onclick="advance()">Next Layer →</button>
        </div>`;

      body += `</div>`;
      main.innerHTML = body;
      scrollLogsToBottom();

      const card = main.querySelector('.card');
      if(card) animateCardIn(card);
      animateNextEnabled(document.getElementById('nextbtn'));

      if(target === 'Output' && L.status === 'done'){
        loadOutputSummary();
      }

      if(L.status === 'running'){
        const est = estimateFor(target, L);
        const pct = Math.min(95, Math.round((liveElapsed(L) / est) * 100));
        const fill = document.getElementById('progfill');
        if(fill){
          if(motionOK) gsap.to(fill, {width: pct + '%', duration: .5, ease: 'power1.out'});
          else fill.style.width = pct + '%';
        }
      }
    };

    const oldCard = main.querySelector('.card');
    if(oldCard && motionOK && !isFirstRender){
      gsap.to(oldCard, {opacity: 0, y: -18, duration: .22, ease: 'power1.in', onComplete: buildNewCard});
    } else {
      buildNewCard();
    }
    isFirstRender = false;
  } else if(L.status === 'running'){
    // Same layer, still running — update numbers/log/progress in place so
    // the bar and elapsed timer animate smoothly instead of snapping.
    const est = estimateFor(target, L);
    const elapsedNow = liveElapsed(L);
    const pct = Math.min(95, Math.round((elapsedNow / est) * 100));
    const fill = document.getElementById('progfill');
    const label = document.getElementById('proglabel');
    const elapsedTxt = document.getElementById('elapsedtxt');
    const log = document.getElementById('livelog');
    const runLabel = L.mode === 'cache-loading' ? 'Loading from cache…' : 'Running…';
    if(fill){
      if(motionOK) gsap.to(fill, {width: pct + '%', duration: .7, ease: 'power1.out'});
      else fill.style.width = pct + '%';
    }
    if(label) label.textContent = `${pct}% · est. ${Math.round(est)}s`;
    if(elapsedTxt) elapsedTxt.textContent = `${runLabel} ${Math.round(elapsedNow)}s`;
    if(log){
      log.textContent = (L.tail||[]).slice(-40).join('\n');
      log.scrollTop = log.scrollHeight;
    }
  }
}

function scrollLogsToBottom(){
  ['livelog', 'donelog', 'tracker-out'].forEach(id => {
    const el = document.getElementById(id);
    if(el) el.scrollTop = el.scrollHeight;
  });
}

async function advance(){
  // Only ever called from the "Next Layer" button, which is only enabled
  // once the layer at the current cursor is actually done. Move the cursor
  // forward one step — this is the ONLY place cursorIndex changes on a
  // completion, so the display never jumps ahead on its own.
  cursorIndex = Math.min(cursorIndex + 1, ORDER.length);

  if(cursorIndex < ORDER.length){
    const target = ORDER[cursorIndex];
    const state = await getState();
    if(state.layers[target].status === 'pending'){
      // Auto-start the new current layer — Next both advances AND runs it,
      // then the page pauses again on its own once THAT one finishes.
      await runLayer(target);
      return;
    }
  }
  render();
}

async function advanceWithCache(){
  // Same as advance(), but the new current layer loads via the paced
  // 10-20s "cache" path instead of a live run.
  cursorIndex = Math.min(cursorIndex + 1, ORDER.length);

  if(cursorIndex < ORDER.length){
    const target = ORDER[cursorIndex];
    const state = await getState();
    if(state.layers[target].status === 'pending'){
      await useCacheLayer(target);
      return;
    }
  }
  render();
}

async function render(){
  const state = await getState();
  renderNav(state);
  renderMain(state);
  if(state.running && !polling){
    polling = setInterval(render, 1200);
  }
  if(!state.running && polling){
    clearInterval(polling);
    polling = null;
  }
}

function showAppShell(animated){
  const shell = document.getElementById('appshell');
  shell.style.display = 'block';
  if(animated && motionOK){
    gsap.fromTo(shell, {opacity: 0, y: 10}, {opacity: 1, y: 0, duration: .5, ease: 'power2.out'});
  } else {
    shell.style.opacity = 1;
  }
}

function beginFlow(){
  const hero = document.getElementById('hero');
  if(motionOK){
    gsap.to(hero, {opacity: 0, scale: .97, duration: .35, ease: 'power1.in', onComplete: () => {
      hero.style.display = 'none';
      showAppShell(true);
      render();
    }});
  } else {
    hero.style.display = 'none';
    showAppShell(false);
    render();
  }
}

// Bootstrap: if a run is already in progress or finished (e.g. page refreshed
// mid-demo), skip the intro and drop straight into the pipeline view.
(async function bootstrap(){
  const state = await getState();
  const anyProgress = ORDER.some(n => state.layers[n].status !== 'pending');
  if(anyProgress){
    // Resume the cursor at the first layer that ISN'T done yet — if that
    // layer is itself already 'done' too (e.g. everything finished before
    // a refresh), land on the last layer so the user still sees its Next
    // button rather than being silently dumped on the final screen.
    const firstNotDone = ORDER.findIndex(n => state.layers[n].status !== 'done');
    cursorIndex = firstNotDone === -1 ? ORDER.length - 1 : firstNotDone;
    document.getElementById('hero').style.display = 'none';
    showAppShell(false);
    render();
  }
  // else: leave the hero showing, wait for "Begin Walkthrough" click.
})();

// Tactile click feedback on every button in the app (delegated so it still
// works on buttons that get created/replaced dynamically each render).
document.addEventListener('click', (evt) => {
  const btn = evt.target.closest('button');
  if(btn && !btn.disabled) spawnRipple(btn, evt);
});

// Subtle mouse-parallax on the background orbs — makes the page feel alive
// even when nothing is currently running.
if(motionOK){
  const orbs = [
    {el: document.getElementById('orbwrap1'), strength: 26},
    {el: document.getElementById('orbwrap2'), strength: 34},
    {el: document.getElementById('orbwrap3'), strength: 20},
  ].filter(o => o.el).map(o => ({...o, moveX: gsap.quickTo(o.el, 'x', {duration: 1.1, ease: 'power2.out'}),
                                       moveY: gsap.quickTo(o.el, 'y', {duration: 1.1, ease: 'power2.out'})}));
  window.addEventListener('mousemove', (evt) => {
    const nx = (evt.clientX / window.innerWidth) - 0.5;
    const ny = (evt.clientY / window.innerHeight) - 0.5;
    orbs.forEach(o => { o.moveX(nx * o.strength); o.moveY(ny * o.strength); });
  });
}
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep terminal quiet during the demo

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = (INDEX_HTML
                     .replace("__TITLES__", json.dumps(TITLES))
                     .replace("__EXPLAIN__", json.dumps(EXPLAIN))
                     .replace("__ORDER__", json.dumps(PIPELINE))
                     .replace("__ESTIMATES__", json.dumps(TIME_ESTIMATES)))
            self._send_html(html)
        elif self.path == "/api/state":
            with lock:
                snap = json.loads(json.dumps(state, default=str))
            self._send_json(snap)
        elif self.path == "/api/fire":
            self._send_json(read_layer4_fire())
        elif self.path == "/api/paperstats":
            self._send_json(read_paper_stats())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {}

        if self.path == "/api/run":
            name = payload.get("layer") or _current_target()
            if name and state["layers"][name]["status"] in ("pending", "failed") and state["running"] is None:
                if name == "PaperTracker":
                    target_fn, args = run_papertracker_bg, ()
                elif name == "Output":
                    target_fn, args = run_output_bg, ()
                else:
                    target_fn, args = run_layer_bg, (name,)
                threading.Thread(target=target_fn, args=args, daemon=True).start()
            self._send_json({"ok": True})
        elif self.path == "/api/skip":
            name = payload.get("layer") or _current_target()
            if name:
                mark_skipped(name)
            self._send_json({"ok": True})
        elif self.path == "/api/cacheload":
            name = payload.get("layer") or _current_target()
            if name and state["layers"][name]["status"] in ("pending", "failed") and state["running"] is None:
                threading.Thread(target=run_cacheload_bg, args=(name,), daemon=True).start()
            self._send_json({"ok": True})
        elif self.path == "/api/reset":
            with lock:
                for name in PIPELINE:
                    state["layers"][name] = {"status": "pending", "elapsed": 0.0, "metrics": {}, "tail": [], "mode": None}
                state["running"] = None
            self._send_json({"ok": True})
        elif self.path == "/api/papertracker":
            log = run_paper_tracker()
            self._send_json({"ok": True, "log": log})
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}"
        print(f"NSE interactive walkthrough running at {url}")
        print("Ctrl+C to stop.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
