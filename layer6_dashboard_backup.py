"""
================================================================================
  NSE DECISION ENGINE — LAYER 6 DASHBOARD
  layer6_dashboard.py

  Modes:
    python layer6_dashboard.py          — interactive layer selector + dashboard
    python layer6_dashboard.py --dash   — dashboard only (use last parquets)
    python layer6_dashboard.py --monitor — monitor open trades vs live prices
    python layer6_dashboard.py --all    — run all layers without prompting
================================================================================
"""

import subprocess, sys, os, time, argparse, webbrowser, json
import logging
from datetime import datetime, date
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BASE   = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
PYTHON = r"C:/Users/harik/AppData/Local/Programs/Python/Python312/python.exe"
TODAY  = date.today().isoformat()
NOW    = datetime.now().strftime("%Y-%m-%d %H:%M")

SCRIPTS = {
    "Layer0-NSE"  : BASE / "layer0_nse_data.py",
    "Layer0-FnO"  : BASE / "layer0_fno.py",
    "Layer1-Heavy": BASE / "layer1_heavy_compute.py",
    "Layer2"      : BASE / "layer2_daily_filter.py",
    "Layer3"      : BASE / "layer3_rules_engine.py",
    "Layer4"      : BASE / "layer4_portfolio.py",
}

SCRIPTS_WEEKLY = {
    "Layer1-Val"  : BASE / "layer1_5_validation.py",
    "Layer1-Sig"  : BASE / "layer 1.55.py",
}

LAYER_DESCRIPTIONS = {
    "Layer0-NSE"  : "Fetch NSE price/volume data for all tickers",
    "Layer0-FnO"  : "Fetch F&O options chain / PCR / OI / IV data",
    "Layer1-Heavy": "GARCH, sector momentum, pairs, clusters (SLOW ~5min)",
    "Layer1-Val"  : "Signal validation / IC backtest",
    "Layer1-Sig"  : "Signal generation (momentum_12_1 etc.)",
    "Layer2"      : "Daily filter — price, liquidity, regime gate",
    "Layer3"      : "Rules engine A/C/D/E — score all candidates",
    "Layer4"      : "Portfolio construction — FIRE/WAIT/KILL + sizing",
}

PARQUETS = {
    "l2"    : BASE / "nse_layer2_candidates.parquet",
    "l3"    : BASE / "nse_layer3_signals.parquet",
    "l4"    : BASE / "nse_layer4_portfolio.parquet",
    "trades": BASE / "paper_trades.csv",
    "sector": BASE / "sector_momentum.parquet",
    "runlog": BASE / "nse_run_log.csv",
}

DASHBOARD_OUT = BASE / f"nse_dashboard_{TODAY}.html"

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("L6")

# ──────────────────────────────────────────────────────────────────────────────
# COLOURS
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "reset" : "\033[0m",
    "green" : "\033[92m",
    "yellow": "\033[93m",
    "red"   : "\033[91m",
    "cyan"  : "\033[96m",
    "bold"  : "\033[1m",
    "dim"   : "\033[2m",
    "white" : "\033[97m",
    "orange": "\033[38;5;214m",
    "fire"  : "\033[38;5;202m",
}

def cp(color, msg, end="\n"):
    print(f"{C.get(color,'')}{msg}{C['reset']}", end=end, flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# INTERACTIVE LAYER SELECTOR
# ──────────────────────────────────────────────────────────────────────────────

def select_layers():
    all_layers = list(SCRIPTS.items())

    print()
    cp("bold", "╔══════════════════════════════════════════════════════════════╗")
    cp("bold", "║   NSE DECISION ENGINE  ·  Layer 6  ·  SELECT LAYERS TO RUN  ║")
    cp("bold", "╚══════════════════════════════════════════════════════════════╝")
    print()
    cp("dim",  "  Enter layer numbers separated by spaces  (e.g. 1 2 6 7 8)")
    cp("dim",  "  Press Enter with no input to run ALL layers")
    cp("dim",  "  Type 'q' to exit")
    print()

    cp("cyan", "  #   Layer             Exists   Description")
    cp("dim",  "  " + "─" * 70)

    for i, (name, path) in enumerate(all_layers, 1):
        exists  = path.exists()
        ex_mark = f"{C['green']}✓{C['reset']}" if exists else f"{C['red']}✗{C['reset']}"
        dim_s   = C['dim'] if not exists else ""
        desc    = LAYER_DESCRIPTIONS.get(name, "")
        print(f"  {C['cyan']}{i}{C['reset']}   {dim_s}{name:<16}{C['reset']}  {ex_mark}      {C['dim']}{desc}{C['reset']}")

    print()
    cp("dim",  "  " + "─" * 70)

    while True:
        try:
            raw = input(f"  {C['yellow']}▶ Layers to run: {C['reset']}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            cp("red", "  Cancelled.")
            sys.exit(0)

        if raw.lower() == "q":
            cp("dim", "  Exiting.")
            sys.exit(0)

        if raw == "":
            selected = [(name, path) for name, path in all_layers if path.exists()]
            cp("yellow", f"\n  Running ALL {len(selected)} available layers")
            return selected

        try:
            nums = [int(x) for x in raw.split()]
        except ValueError:
            cp("red", "  ✗ Invalid input — enter numbers only (e.g. 1 2 6 7 8)")
            continue

        invalid = [n for n in nums if n < 1 or n > len(all_layers)]
        if invalid:
            cp("red", f"  ✗ Out of range: {invalid} — valid range 1-{len(all_layers)}")
            continue

        selected = []
        for n in nums:
            name, path = all_layers[n - 1]
            if not path.exists():
                cp("yellow", f"  ⚠  {name}: script not found — skipping")
            else:
                selected.append((name, path))

        if not selected:
            cp("red", "  ✗ No valid layers selected.")
            continue

        print()
        cp("cyan", "  Selected layers:")
        for name, _ in selected:
            cp("white", f"    • {name}  —  {LAYER_DESCRIPTIONS.get(name,'')}")
        print()

        try:
            confirm = input(f"  {C['yellow']}▶ Confirm? (Y/n): {C['reset']}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            cp("red", "  Cancelled.")
            sys.exit(0)

        if confirm in ("", "y", "yes"):
            return selected
        else:
            cp("dim", "  Re-selecting...")
            print()
            continue

# ──────────────────────────────────────────────────────────────────────────────
# PROGRESS BAR
# ──────────────────────────────────────────────────────────────────────────────

BAR_WIDTH = 40

import re as _re
_INFO_PATTERNS = [
    ("candidates", r"(\d+)\s*candidates"),
    ("signals",    r"(\d+)\s*signals"),
    ("FIRE",       r"FIRE[=:](\d+)"),
    ("tickers",    r"(\d+)\s*tickers"),
    ("pairs",      r"(\d+)\s*pairs"),
    ("sectors",    r"(\d+)\s*sectors"),
    ("regime",     r"Regime[=:\s]+(\w+)"),
    ("VIX",        r"VIX[=:\s]+([\d\.]+)"),
]

def extract_info(line):
    for label, pattern in _INFO_PATTERNS:
        m = _re.search(pattern, line, _re.IGNORECASE)
        if m:
            return f"{label}={m.group(1)}"
    return None

def progress_bar(step, total, label, status="running", elapsed=None, info=""):
    pct    = step / total
    filled = int(BAR_WIDTH * pct)
    bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
    pct_s  = f"{pct*100:5.1f}%"
    ela_s  = f"  {elapsed:.1f}s" if elapsed else ""
    col    = "green" if status == "ok" else "red" if status == "fail" else "cyan"
    icon   = "✅" if status == "ok" else "❌" if status == "fail" else "⏳"
    info_s = f"  [{info}]" if info else ""
    print(f"\r\033[2K{C[col]}[{bar}] {pct_s}  {icon} {label:<18}{ela_s}{info_s}{C['reset']}",
          end="", flush=True)
    if status in ("ok", "fail"):
        print()

# ──────────────────────────────────────────────────────────────────────────────
# RUN A SINGLE LAYER
# ──────────────────────────────────────────────────────────────────────────────

def run_layer(name, script, step, total):
    import threading

    progress_bar(step - 0.5, total, name, "running", 0, "")
    t0           = time.time()
    stdout_lines = []
    err_lines    = []
    last_info    = ""

    _env = os.environ.copy()
    _env["PYTHONUTF8"]       = "1"
    _env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.Popen(
            [PYTHON, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_env,
            cwd=str(BASE),
        )

        def read_stdout():
            nonlocal last_info
            for line in proc.stdout:
                line = line.rstrip()
                stdout_lines.append(line)
                info = extract_info(line)
                if info:
                    last_info = info
                elapsed = time.time() - t0
                progress_bar(step - 0.5, total, name, "running", elapsed, last_info)

        def read_stderr():
            for line in proc.stderr:
                err_lines.append(line.rstrip())

        t1 = threading.Thread(target=read_stdout, daemon=True)
        t2 = threading.Thread(target=read_stderr, daemon=True)
        t1.start(); t2.start()

        proc.wait()
        t1.join(timeout=5)
        t2.join(timeout=5)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            tail = "\n".join(err_lines[-10:] or stdout_lines[-10:])
            progress_bar(step, total, name, "fail", elapsed, last_info or "FAILED")
            log.error(f"[{name}] FAILED rc={proc.returncode}:\n{tail}")
            return False, elapsed, tail

        progress_bar(step, total, name, "ok", elapsed, last_info)
        return True, elapsed, ""

    except Exception as exc:
        elapsed = time.time() - t0
        progress_bar(step, total, name, "fail", elapsed, "ERROR")
        log.error(f"[{name}] Exception: {exc}")
        return False, elapsed, str(exc)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD PARQUETS SAFELY
# ──────────────────────────────────────────────────────────────────────────────

def load(key):
    p = PARQUETS.get(key)
    if p and p.exists():
        try:
            if str(p).endswith(".csv"):
                return pd.read_csv(p)
            return pd.read_parquet(p)
        except Exception as e:
            log.warning(f"Could not load {p.name}: {e}")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_dashboard(results, mode):
    l2     = load("l2")
    l3     = load("l3")
    l4     = load("l4")
    trades = load("trades")
    sector = load("sector")
    runlog = load("runlog")

    if l4 is not None:
        if "stop_price"  in l4.columns and "stop_loss"   not in l4.columns:
            l4["stop_loss"]   = l4["stop_price"]
        if "entry_price" in l4.columns and "price_close" not in l4.columns:
            l4["price_close"] = l4["entry_price"]

    regime = "N/A"; vix = "N/A"
    if l2 is not None:
        regime = l2["regime"].iloc[0] if "regime" in l2.columns else "N/A"
        vix    = f"{float(l2['vix'].iloc[0]):.2f}" if "vix" in l2.columns else "N/A"

    fire_df = pd.DataFrame(); wait_df = pd.DataFrame()
    fire_count = wait_count = kill_count = 0
    if l4 is not None and "timing_verdict" in l4.columns:
        vc         = l4["timing_verdict"].str.upper()
        fire_df    = l4[vc == "FIRE"]
        wait_df    = l4[vc == "WAIT"]
        fire_count = int((vc == "FIRE").sum())
        wait_count = int((vc == "WAIT").sum())
        kill_count = int((vc == "KILL").sum())

    sc = "composite_score" if l4 is not None and "composite_score" in l4.columns else "final_score"

    top_sectors = []
    if sector is not None and "Sector" in sector.columns:
        top_sectors = sector[sector["Sector"].str.strip() != ""].sort_values("Rank").head(3)["Sector"].tolist()

    timing_rows = ""
    total_sec = 0
    for name, r in results.items():
        ok_cls = "ok" if r["ok"] else "fail"
        ok_txt = "OK"   if r["ok"] else "FAIL"
        total_sec += r["elapsed"]
        timing_rows += f"""
        <tr>
          <td>{name}</td>
          <td class="status-{ok_cls}">{ok_txt}</td>
          <td>{r['elapsed']:.1f}s</td>
        </tr>"""
    if timing_rows:
        timing_rows += f"<tr class='total-row'><td>TOTAL</td><td></td><td>{total_sec:.1f}s</td></tr>"

    fire_rows = ""
    if len(fire_df) > 0:
        for row in fire_df.itertuples():
            ticker = getattr(row, "ticker", "?")
            tier   = getattr(row, "tier", "?")
            entry  = getattr(row, "entry_price",  getattr(row, "price_close", 0))
            sl     = getattr(row, "stop_loss",     0)
            tgt    = getattr(row, "target_price",  0)
            qty    = getattr(row, "qty",            getattr(row, "position_qty", "?"))
            score  = getattr(row, sc, 0.0)
            sec_n  = str(getattr(row, "sector", ""))[:20]
            rr     = round((tgt - entry) / (entry - sl), 2) if (entry - sl) > 0 else 0
            risk_r = round((entry - sl) * float(qty), 0)    if qty != "?" else 0
            fire_rows += f"""
            <tr class="fire-row">
              <td><span class="ticker-badge">{ticker}</span></td>
              <td><span class="tier tier-{str(tier).lower()}">{tier}</span></td>
              <td>{sec_n}</td>
              <td class="num">₹{entry:,.2f}</td>
              <td class="num sl">₹{sl:,.2f}</td>
              <td class="num tgt">₹{tgt:,.2f}</td>
              <td class="num">{qty}</td>
              <td class="num">1:{rr}</td>
              <td class="num risk">₹{risk_r:,.0f}</td>
              <td><div class="score-bar-wrap"><div class="score-bar" style="width:{min(score*100,100):.0f}%"></div><span>{score:.3f}</span></div></td>
            </tr>"""
    else:
        fire_rows = "<tr><td colspan='10' class='no-data'>No FIRE signals today</td></tr>"

    wait_rows = ""
    if len(wait_df) > 0 and sc in wait_df.columns:
        for i, row in enumerate(wait_df.nlargest(15, sc).itertuples(), 1):
            ticker = getattr(row, "ticker", "?")
            tier   = getattr(row, "tier", "?")
            price  = getattr(row, "entry_price", getattr(row, "price_close", 0))
            score  = getattr(row, sc, 0.0)
            sec_n  = str(getattr(row, "sector", ""))[:20]
            wait_rows += f"""
            <tr>
              <td class="rank-num">{i}</td>
              <td><span class="ticker-badge small">{ticker}</span></td>
              <td><span class="tier tier-{str(tier).lower()}">{tier}</span></td>
              <td>{sec_n}</td>
              <td class="num">₹{price:,.2f}</td>
              <td><div class="score-bar-wrap"><div class="score-bar wait-bar" style="width:{min(score*100,100):.0f}%"></div><span>{score:.3f}</span></div></td>
            </tr>"""

    l3_rows = ""
    if l3 is not None and "final_score" in l3.columns:
        for i, row in enumerate(l3.nlargest(10, "final_score").itertuples(), 1):
            ticker = getattr(row, "ticker", "?")
            tier   = getattr(row, "tier", "?")
            price  = getattr(row, "price_close", 0)
            score  = getattr(row, "final_score", 0)
            sec_n  = str(getattr(row, "sector", ""))[:18]
            tags   = ""
            for col, ch, cls in [("rule_a_fires","A","tag-a"),("rule_c_fires","C","tag-c"),
                                  ("rule_d_fires","D","tag-d"),("rule_e_fires","E","tag-e")]:
                if col in l3.columns and getattr(row, col, False):
                    tags += f'<span class="rule-tag {cls}">{ch}</span>'
            l3_rows += f"""
            <tr>
              <td class="rank-num">{i}</td>
              <td><span class="ticker-badge small">{ticker}</span></td>
              <td><span class="tier tier-{str(tier).lower()}">{tier}</span></td>
              <td>{sec_n}</td>
              <td class="num">₹{price:,.2f}</td>
              <td>{tags if tags else '<span class="tag-none">—</span>'}</td>
              <td><div class="score-bar-wrap"><div class="score-bar l3-bar" style="width:{min(score*100,100):.0f}%"></div><span>{score:.3f}</span></div></td>
            </tr>"""

    sector_rows = ""
    if sector is not None:
        for row in sector[sector["Sector"].str.strip() != ""].sort_values("Rank").itertuples():
            ret   = float(row.Rolling20dReturn) * 100
            r_cls = "pos" if ret >= 0 else "neg"
            rank  = int(row.Rank)
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else str(rank)
            sector_rows += f"""
            <tr>
              <td class="rank-num">{medal}</td>
              <td>{row.Sector}</td>
              <td class="num {r_cls}">{ret:+.2f}%</td>
            </tr>"""

    runlog_labels = "[]"; runlog_fire = "[]"; runlog_vix = "[]"
    if runlog is not None and len(runlog) > 0:
        runlog_labels = json.dumps(runlog["date"].tolist()[-14:])
        runlog_fire   = json.dumps([float(x) if pd.notna(x) else 0 for x in runlog["fire_count"].tolist()[-14:]])
        runlog_vix    = json.dumps([float(x) if pd.notna(x) else 0 for x in runlog["vix"].tolist()[-14:]])

    open_trades_html = ""
    if trades is not None and len(trades) > 0:
        open_t = trades[trades["status"] == "OPEN"] if "status" in trades.columns else trades
        if len(open_t) > 0:
            trade_rows = ""
            for row in open_t.itertuples():
                ticker = getattr(row, "ticker", "?")
                entry  = float(getattr(row, "entry_price", 0))
                sl     = float(getattr(row, "stop_price",  0))
                tgt    = float(getattr(row, "target_price", 0))
                days   = getattr(row, "days_held", "?")
                trade_rows += f"""
                <tr>
                  <td><span class="ticker-badge small">{ticker}</span></td>
                  <td class="num">₹{entry:,.2f}</td>
                  <td class="num sl">₹{sl:,.2f}</td>
                  <td class="num tgt">₹{tgt:,.2f}</td>
                  <td class="num">{days}d</td>
                </tr>"""
            open_trades_html = f"""
            <section class="section">
              <h2 class="section-title">📂 Open Paper Trades <span class="badge">{len(open_t)}</span></h2>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Ticker</th><th>Entry</th><th>SL</th><th>Target</th><th>Days</th></tr></thead>
                  <tbody>{trade_rows}</tbody>
                </table>
              </div>
            </section>"""

    rule_a = rule_c = rule_d = rule_e = 0
    if l3 is not None:
        rule_a = int(l3["rule_a_fires"].sum()) if "rule_a_fires" in l3.columns else 0
        rule_c = int(l3["rule_c_fires"].sum()) if "rule_c_fires" in l3.columns else 0
        rule_d = int(l3["rule_d_fires"].sum()) if "rule_d_fires" in l3.columns else 0
        rule_e = int(l3["rule_e_fires"].sum()) if "rule_e_fires" in l3.columns else 0

    candidates = len(l2) if l2 is not None else 0
    top_score  = float(l4[sc].max()) if l4 is not None and sc in l4.columns else 0.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSE Decision Engine — {TODAY}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0a0e14;--bg2:#0f1520;--bg3:#141c2e;--border:#1e2d47;
    --text:#c8d8f0;--dim:#5a7a9a;--green:#00e676;--red:#ff4444;
    --yellow:#ffd740;--cyan:#40c4ff;--orange:#ff9100;--fire:#ff6b35;
    --accent:#0d47a1;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.5;}}
  .header{{background:linear-gradient(135deg,#0a1628,#0d1f3c,#0a1628);border-bottom:1px solid var(--border);padding:20px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}}
  .header h1{{font-family:var(--mono);font-size:17px;font-weight:600;color:var(--cyan);letter-spacing:.05em;}}
  .header p{{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:2px;}}
  .header-right{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;}}
  .stat-pill{{display:flex;flex-direction:column;align-items:center;background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:7px 14px;min-width:72px;}}
  .stat-pill .val{{font-family:var(--mono);font-size:20px;font-weight:600;line-height:1;}}
  .stat-pill .lbl{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}
  .val-fire{{color:var(--fire);}}.val-wait{{color:var(--yellow);}}.val-kill{{color:var(--dim);}}.val-green{{color:var(--green);}}.val-cyan{{color:var(--cyan);}}
  .regime-badge{{font-family:var(--mono);font-size:11px;padding:5px 12px;border-radius:20px;border:1px solid;font-weight:600;text-transform:uppercase;letter-spacing:.1em;}}
  .regime-sideways{{color:var(--yellow);border-color:var(--yellow);background:rgba(255,215,64,.08);}}
  .regime-bull{{color:var(--green);border-color:var(--green);background:rgba(0,230,118,.08);}}
  .regime-crisis{{color:var(--red);border-color:var(--red);background:rgba(255,68,68,.08);}}
  .container{{max-width:1600px;margin:0 auto;padding:20px 32px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
  .section{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:16px;animation:fadeIn .3s ease both;}}
  .section-title{{font-family:var(--mono);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:var(--dim);padding:12px 18px;border-bottom:1px solid var(--border);background:var(--bg3);display:flex;align-items:center;gap:8px;}}
  .badge{{background:var(--accent);color:var(--cyan);font-size:10px;padding:1px 7px;border-radius:10px;font-weight:600;}}
  .badge-fire{{background:rgba(255,107,53,.2);color:var(--fire);}}
  .table-wrap{{overflow-x:auto;}}
  table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px;}}
  thead tr{{background:var(--bg3);border-bottom:1px solid var(--border);}}
  thead th{{padding:9px 12px;text-align:left;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);white-space:nowrap;}}
  tbody tr{{border-bottom:1px solid rgba(30,45,71,.5);transition:background .15s;}}
  tbody tr:hover{{background:rgba(13,71,161,.15);}}
  tbody td{{padding:8px 12px;white-space:nowrap;}}
  .fire-row{{background:rgba(255,107,53,.04);}}.fire-row:hover{{background:rgba(255,107,53,.1)!important;}}
  .total-row td{{font-weight:600;color:var(--cyan);border-top:1px solid var(--border);}}
  .num{{text-align:right;}}.sl{{color:var(--red)!important;}}.tgt{{color:var(--green)!important;}}
  .pos{{color:var(--green);}}.neg{{color:var(--red);}}.risk{{color:var(--orange);}}
  .no-data{{text-align:center;color:var(--dim);padding:20px;}}.rank-num{{color:var(--dim);width:32px;}}
  .ticker-badge{{background:rgba(64,196,255,.1);color:var(--cyan);border:1px solid rgba(64,196,255,.2);border-radius:4px;padding:2px 7px;font-weight:600;font-size:12px;}}
  .ticker-badge.small{{font-size:11px;padding:1px 5px;}}
  .tier{{border-radius:3px;padding:1px 5px;font-size:10px;font-weight:600;}}
  .tier-t1{{background:rgba(0,230,118,.12);color:var(--green);}}.tier-t2{{background:rgba(255,215,64,.12);color:var(--yellow);}}.tier-t3{{background:rgba(90,122,154,.2);color:var(--dim);}}
  .rule-tag{{display:inline-block;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:700;margin-right:2px;}}
  .tag-a{{background:rgba(255,145,0,.15);color:var(--orange);}}.tag-c{{background:rgba(64,196,255,.15);color:var(--cyan);}}.tag-d{{background:rgba(255,215,64,.15);color:var(--yellow);}}.tag-e{{background:rgba(0,230,118,.15);color:var(--green);}}.tag-none{{color:var(--dim);}}
  .score-bar-wrap{{display:flex;align-items:center;gap:6px;min-width:110px;}}
  .score-bar{{height:5px;background:linear-gradient(90deg,var(--fire),var(--yellow));border-radius:3px;flex:1;max-width:70px;}}
  .wait-bar{{background:linear-gradient(90deg,var(--accent),var(--cyan));}}.l3-bar{{background:linear-gradient(90deg,#1565c0,var(--cyan));}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:16px;}}
  .kpi-card{{background:var(--bg2);padding:16px 18px;display:flex;flex-direction:column;gap:4px;}}
  .kpi-card .kpi-val{{font-family:var(--mono);font-size:24px;font-weight:600;line-height:1;}}
  .kpi-card .kpi-lbl{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}}
  .rule-pills{{display:flex;gap:10px;padding:14px 18px;flex-wrap:wrap;}}
  .rule-pill{{display:flex;flex-direction:column;align-items:center;background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:10px 18px;min-width:90px;}}
  .rule-pill .rp-count{{font-family:var(--mono);font-size:26px;font-weight:600;}}
  .rule-pill .rp-name{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}
  .status-ok{{color:var(--green);}}.status-fail{{color:var(--red);}}
  .chart-wrap{{padding:18px;height:190px;position:relative;}}
  .footer{{text-align:center;padding:16px;color:var(--dim);font-family:var(--mono);font-size:10px;border-top:1px solid var(--border);margin-top:16px;}}
  @keyframes fadeIn{{from{{opacity:0;transform:translateY(6px);}}to{{opacity:1;transform:none;}}}}
  .section:nth-child(2){{animation-delay:.04s;}}.section:nth-child(3){{animation-delay:.08s;}}.section:nth-child(4){{animation-delay:.12s;}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>⚡ NSE DECISION ENGINE</h1>
    <p>Layer 6 Dashboard &nbsp;|&nbsp; {NOW} &nbsp;|&nbsp; {mode}</p>
  </div>
  <div class="header-right">
    <span class="regime-badge regime-{regime.lower()}">{regime} · VIX {vix}</span>
    <div class="stat-pill"><span class="val val-fire">{fire_count}</span><span class="lbl">FIRE</span></div>
    <div class="stat-pill"><span class="val val-wait">{wait_count}</span><span class="lbl">WAIT</span></div>
    <div class="stat-pill"><span class="val val-kill">{kill_count}</span><span class="lbl">KILL</span></div>
    <div class="stat-pill"><span class="val val-cyan">{candidates}</span><span class="lbl">Universe</span></div>
    <div class="stat-pill"><span class="val val-green">{top_score:.3f}</span><span class="lbl">Top Score</span></div>
  </div>
</div>

<div class="container">

  <div class="kpi-grid">
    <div class="kpi-card"><span class="kpi-val val-fire">{fire_count}</span><span class="kpi-lbl">🔥 Fire</span></div>
    <div class="kpi-card"><span class="kpi-val val-wait">{wait_count}</span><span class="kpi-lbl">👁 Wait</span></div>
    <div class="kpi-card"><span class="kpi-val val-kill">{kill_count}</span><span class="kpi-lbl">❌ Kill</span></div>
    <div class="kpi-card"><span class="kpi-val val-cyan">{candidates}</span><span class="kpi-lbl">📊 Universe</span></div>
    <div class="kpi-card"><span class="kpi-val" style="color:var(--yellow)">{vix}</span><span class="kpi-lbl">📈 VIX</span></div>
    <div class="kpi-card"><span class="kpi-val val-green">{top_score:.3f}</span><span class="kpi-lbl">🏆 Top Score</span></div>
  </div>

  <section class="section">
    <h2 class="section-title">🔥 FIRE SIGNALS — Execute at Market Open <span class="badge badge-fire">{fire_count}</span></h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Tier</th><th>Sector</th><th>Entry</th><th>Stop Loss</th><th>Target</th><th>Qty</th><th>R:R</th><th>Risk ₹</th><th>Score</th></tr></thead>
        <tbody>{fire_rows}</tbody>
      </table>
    </div>
  </section>

  <div class="grid-2">
    <section class="section">
      <h2 class="section-title">📡 Layer 3 — Top 10 Signals</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Ticker</th><th>Tier</th><th>Sector</th><th>Price</th><th>Rules</th><th>Score</th></tr></thead>
          <tbody>{l3_rows}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2 class="section-title">👁 Watchlist — Top 15 WAIT</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Ticker</th><th>Tier</th><th>Sector</th><th>Price</th><th>Score</th></tr></thead>
          <tbody>{wait_rows}</tbody>
        </table>
      </div>
    </section>
  </div>

  <div class="grid-2">
    <section class="section">
      <h2 class="section-title">📐 Rule Engine Breakdown</h2>
      <div class="rule-pills">
        <div class="rule-pill"><span class="rp-count" style="color:var(--orange)">{rule_a}</span><span class="rp-name">Rule A · Vol</span></div>
        <div class="rule-pill"><span class="rp-count" style="color:var(--cyan)">{rule_c}</span><span class="rp-name">Rule C · Pairs</span></div>
        <div class="rule-pill"><span class="rp-count" style="color:var(--yellow)">{rule_d}</span><span class="rp-name">Rule D · Event</span></div>
        <div class="rule-pill"><span class="rp-count" style="color:var(--green)">{rule_e}</span><span class="rp-name">Rule E · Mom</span></div>
      </div>
    </section>
    <section class="section">
      <h2 class="section-title">🏭 Sector Momentum (20d)</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Sector</th><th>20d Return</th></tr></thead>
          <tbody>{sector_rows if sector_rows else '<tr><td colspan="3" class="no-data">No sector data</td></tr>'}</tbody>
        </table>
      </div>
    </section>
  </div>

  <div class="grid-2">
    <section class="section">
      <h2 class="section-title">⚙️ Pipeline Timing</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Layer</th><th>Status</th><th>Time</th></tr></thead>
          <tbody>{timing_rows if timing_rows else '<tr><td colspan="3" class="no-data">Dashboard-only mode</td></tr>'}</tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2 class="section-title">📊 14-Day Run History</h2>
      <div class="chart-wrap"><canvas id="histChart"></canvas></div>
    </section>
  </div>

  {open_trades_html}

</div>

<div class="footer">NSE Decision Engine · Layer 6 · {NOW} · Capital ₹10,00,000 · Max Risk/Trade 1%</div>

<script>
const labels={runlog_labels},fireData={runlog_fire},vixData={runlog_vix};
if(labels.length>0){{
  const ctx=document.getElementById('histChart').getContext('2d');
  new Chart(ctx,{{type:'bar',data:{{labels,datasets:[
    {{label:'FIRE',data:fireData,backgroundColor:'rgba(255,107,53,.7)',borderColor:'rgba(255,107,53,1)',borderWidth:1,yAxisID:'y'}},
    {{label:'VIX',data:vixData,type:'line',borderColor:'rgba(64,196,255,.8)',backgroundColor:'rgba(64,196,255,.05)',borderWidth:2,pointRadius:3,fill:false,yAxisID:'y1',tension:.3}}
  ]}},options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#5a7a9a',font:{{family:'IBM Plex Mono',size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#5a7a9a',font:{{family:'IBM Plex Mono',size:9}},maxRotation:45}},grid:{{color:'rgba(30,45,71,.5)'}}}},
      y:{{position:'left',ticks:{{color:'#ff6b35',font:{{family:'IBM Plex Mono',size:10}}}},grid:{{color:'rgba(30,45,71,.5)'}},title:{{display:true,text:'FIRE',color:'#ff6b35',font:{{size:10}}}}}},
      y1:{{position:'right',ticks:{{color:'#40c4ff',font:{{family:'IBM Plex Mono',size:10}}}},grid:{{drawOnChartArea:false}},title:{{display:true,text:'VIX',color:'#40c4ff',font:{{size:10}}}}}}
    }}
  }}}}); 
}}
</script>
</body></html>"""
    return html


# ──────────────────────────────────────────────────────────────────────────────
# EOD REPORT + RUN LOG
# ──────────────────────────────────────────────────────────────────────────────

def _build_eod_report(results, l2_df, l3_df, l4_df):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    W   = 80
    lines = []

    if l4_df is not None:
        if "stop_price" in l4_df.columns and "stop_loss" not in l4_df.columns:
            l4_df = l4_df.copy(); l4_df["stop_loss"] = l4_df["stop_price"]
        if "entry_price" in l4_df.columns and "price_close" not in l4_df.columns:
            l4_df = l4_df.copy(); l4_df["price_close"] = l4_df["entry_price"]

    lines += ["=" * W,
              "  NSE DECISION ENGINE — EOD CONSOLIDATED REPORT",
              f"  Date   : {now}",
              f"  Status : {'ALL OK' if all(r['ok'] for r in results.values()) else 'ONE OR MORE FAILED'}",
              "=" * W, ""]

    lines.append("  PIPELINE TIMING"); lines.append("─" * W)
    total_s = sum(r["elapsed"] for r in results.values())
    for name, r in results.items():
        lines.append(f"  {name:<14}  [{'OK' if r['ok'] else 'FAIL'}]  {r['elapsed']:6.1f}s")
    lines += [f"  {'Total':<14}           {total_s:6.1f}s", ""]

    lines += ["=" * W, "  LAYER 2 — UNIVERSE FILTER", "─" * W]
    if l2_df is not None:
        regime = l2_df["regime"].iloc[0] if "regime" in l2_df.columns else "N/A"
        vix    = float(l2_df["vix"].iloc[0]) if "vix" in l2_df.columns else 0
        lines.append(f"  Candidates : {len(l2_df)}   Regime : {regime}   VIX : {vix:.2f}")
        if "tier" in l2_df.columns:
            tc = l2_df["tier"].value_counts().to_dict()
            lines.append(f"  Tiers : T1={tc.get('T1',0)}  T2={tc.get('T2',0)}  T3={tc.get('T3',0)}")
    else:
        lines.append("  [Not available]")
    lines.append("")

    lines += ["=" * W, "  LAYER 4 — FIRE SIGNALS", "─" * W]
    if l4_df is not None and "timing_verdict" in l4_df.columns:
        vc   = l4_df["timing_verdict"].str.upper()
        fire = l4_df[vc == "FIRE"]
        lines.append(f"  FIRE={int((vc=='FIRE').sum())}  WAIT={int((vc=='WAIT').sum())}  KILL={int((vc=='KILL').sum())}")
        lines.append("")
        sc = "composite_score" if "composite_score" in l4_df.columns else "final_score"
        for row in fire.itertuples():
            t  = getattr(row, "ticker", "?")
            ti = getattr(row, "tier", "?")
            e  = getattr(row, "entry_price",  getattr(row, "price_close", 0))
            sl = getattr(row, "stop_loss",    0)
            tg = getattr(row, "target_price", 0)
            q  = getattr(row, "qty",          getattr(row, "position_qty", "?"))
            s  = getattr(row, sc, 0.0)
            lines.append(f"  {t:<14} {ti}  E:₹{e:>8.2f}  SL:₹{sl:>8.2f}  T:₹{tg:>8.2f}  Qty:{q}  Score:{s:.4f}")
    else:
        lines.append("  [Not available]")
    lines += ["", "=" * W]
    return "\n".join(lines)


def _append_run_log(results, l4_df, l2_df):
    RUN_LOG = BASE / "nse_run_log.csv"
    row = {"date": TODAY}

    for name in ["Layer2", "Layer3", "Layer4"]:
        key = name.lower()
        row[f"{key}_ok"]  = results.get(name, {}).get("ok",      False)
        row[f"{key}_sec"] = round(results.get(name, {}).get("elapsed", 0.0), 1)

    if l2_df is not None:
        row["candidates"] = len(l2_df)
        row["regime"]     = l2_df["regime"].iloc[0] if "regime" in l2_df.columns else None
        row["vix"]        = round(float(l2_df["vix"].iloc[0]), 2) if "vix" in l2_df.columns else None
    else:
        row["candidates"] = -1

    if l4_df is not None and "timing_verdict" in l4_df.columns:
        vc = l4_df["timing_verdict"].str.upper().value_counts().to_dict()
        row["fire_count"] = vc.get("FIRE", 0)
        row["wait_count"] = vc.get("WAIT", 0)
        row["kill_count"] = vc.get("KILL", 0)
        sc = "composite_score" if "composite_score" in l4_df.columns else "final_score"
        row["top_score"]  = round(float(l4_df[sc].max()), 4) if sc in l4_df.columns else None

    import pandas as _pd
    new_row = _pd.DataFrame([row])
    if RUN_LOG.exists():
        try:
            existing = _pd.read_csv(RUN_LOG)
            existing = existing[existing["date"] != TODAY]
            updated  = _pd.concat([existing, new_row], ignore_index=True).dropna(axis=1, how="all")
        except Exception:
            updated = new_row
    else:
        updated = new_row
    updated.to_csv(RUN_LOG, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_cli_report(results, total_elapsed):
    import pandas as _pd

    l2     = load("l2")
    l3     = load("l3")
    l4     = load("l4")
    trades = load("trades")
    sector = load("sector")

    if l4 is not None:
        if "stop_price"  in l4.columns and "stop_loss"   not in l4.columns:
            l4["stop_loss"]   = l4["stop_price"]
        if "entry_price" in l4.columns and "price_close" not in l4.columns:
            l4["price_close"] = l4["entry_price"]

    W  = 72
    sc = "composite_score" if l4 is not None and "composite_score" in l4.columns else "final_score"

    regime = l2["regime"].iloc[0] if l2 is not None and "regime" in l2.columns else "N/A"
    vix    = f"{float(l2['vix'].iloc[0]):.2f}" if l2 is not None and "vix" in l2.columns else "N/A"
    cands  = len(l2) if l2 is not None else 0

    fire_df = _pd.DataFrame(); wait_df = _pd.DataFrame()
    fire_count = wait_count = kill_count = 0
    if l4 is not None and "timing_verdict" in l4.columns:
        vc         = l4["timing_verdict"].str.upper()
        fire_df    = l4[vc == "FIRE"]
        wait_df    = l4[vc == "WAIT"]
        fire_count = int((vc == "FIRE").sum())
        wait_count = int((vc == "WAIT").sum())
        kill_count = int((vc == "KILL").sum())

    print()
    cp("bold", "═" * W)
    cp("cyan",  f"  ⚡ NSE DECISION ENGINE  ·  {NOW}")
    cp("bold", "═" * W)

    print()
    regime_col = "green" if regime == "Bull" else "yellow" if regime == "Sideways" else "red"
    cp(regime_col, f"  Regime: {regime}   VIX: {vix}   Universe: {cands} stocks")
    print()
    print(f"  {C['fire']}🔥 FIRE: {fire_count}{C['reset']}   "
          f"{C['yellow']}👁  WAIT: {wait_count}{C['reset']}   "
          f"{C['dim']}❌ KILL: {kill_count}{C['reset']}")

    print()
    cp("bold", "─" * W)
    cp("orange", f"  🔥 FIRE SIGNALS  ({fire_count})")
    cp("bold", "─" * W)
    if len(fire_df) > 0:
        print(f"  {'TICKER':<12} {'TIER':<4} {'SECTOR':<20} {'ENTRY':>8} {'SL':>8} {'TARGET':>9} {'QTY':>5} {'R:R':>5} {'SCORE':>7}")
        print("  " + "─" * 70)
        for row in fire_df.itertuples():
            ticker = getattr(row, "ticker",      "?")
            tier   = getattr(row, "tier",         "?")
            entry  = getattr(row, "entry_price",  getattr(row, "price_close", 0))
            sl     = getattr(row, "stop_loss",    0)
            tgt    = getattr(row, "target_price", 0)
            qty    = getattr(row, "qty",           getattr(row, "position_qty", "?"))
            score  = getattr(row, sc,              0.0)
            sec_n  = str(getattr(row, "sector",   ""))[:18]
            rr     = round((tgt-entry)/(entry-sl), 2) if (entry-sl) > 0 else 0
            risk   = round((entry-sl)*float(qty), 0)  if qty != "?" else 0
            print(f"  {C['cyan']}{ticker:<12}{C['reset']} {tier:<4} {sec_n:<20} "
                  f"{C['white']}₹{entry:>7.2f}{C['reset']} "
                  f"{C['red']}₹{sl:>7.2f}{C['reset']} "
                  f"{C['green']}₹{tgt:>8.2f}{C['reset']} "
                  f"{qty:>5} {C['yellow']}1:{rr:<4}{C['reset']} "
                  f"{C['orange']}₹{risk:>5.0f}{C['reset']}  {score:.3f}")
    else:
        cp("dim", "  No FIRE signals today.")

    print()
    cp("bold", "─" * W)
    cp("yellow", f"  👁  WATCHLIST — Top 10 WAIT")
    cp("bold", "─" * W)
    if len(wait_df) > 0 and sc in wait_df.columns:
        print(f"  {'#':<3} {'TICKER':<12} {'TIER':<4} {'SECTOR':<20} {'PRICE':>8} {'SCORE':>7}")
        print("  " + "─" * 56)
        for i, row in enumerate(wait_df.nlargest(10, sc).itertuples(), 1):
            ticker = getattr(row, "ticker",     "?")
            tier   = getattr(row, "tier",        "?")
            price  = getattr(row, "entry_price", getattr(row, "price_close", 0))
            score  = getattr(row, sc,             0.0)
            sec_n  = str(getattr(row, "sector",  ""))[:18]
            print(f"  {i:<3} {C['cyan']}{ticker:<12}{C['reset']} {tier:<4} {sec_n:<20} ₹{price:>7.2f}  {score:.3f}")
    else:
        cp("dim", "  No WAIT signals.")

    print()
    cp("bold", "─" * W)
    cp("dim",  "  🏭 SECTOR MOMENTUM (20d)")
    cp("bold", "─" * W)
    if sector is not None:
        for row in sector[sector["Sector"].str.strip() != ""].sort_values("Rank").itertuples():
            ret   = float(row.Rolling20dReturn) * 100
            rank  = int(row.Rank)
            medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"  {rank}."
            col   = "green" if ret >= 0 else "red"
            print(f"  {medal}  {row.Sector:<28}  {C[col]}{ret:+.2f}%{C['reset']}")

    print()
    cp("bold", "─" * W)
    cp("dim",  "  📐 RULE ENGINE")
    cp("bold", "─" * W)
    if l3 is not None:
        ra = int(l3["rule_a_fires"].sum()) if "rule_a_fires" in l3.columns else 0
        rc = int(l3["rule_c_fires"].sum()) if "rule_c_fires" in l3.columns else 0
        rd = int(l3["rule_d_fires"].sum()) if "rule_d_fires" in l3.columns else 0
        re = int(l3["rule_e_fires"].sum()) if "rule_e_fires" in l3.columns else 0
        print(f"  {C['orange']}Rule A (Vol)     : {ra}{C['reset']}")
        print(f"  {C['cyan']}Rule C (Pairs)   : {rc}{C['reset']}")
        print(f"  {C['yellow']}Rule D (Event)   : {rd}{C['reset']}")
        print(f"  {C['green']}Rule E (Momentum): {re}{C['reset']}")

    print()
    cp("bold", "─" * W)
    cp("dim",  "  📂 OPEN PAPER TRADES")
    cp("bold", "─" * W)
    if trades is not None and "status" in trades.columns:
        open_t = trades[trades["status"] == "OPEN"]
        if len(open_t) > 0:
            print(f"  {'TICKER':<12} {'ENTRY':>8} {'SL':>8} {'TARGET':>9} {'DAYS':>5}")
            print("  " + "─" * 46)
            for row in open_t.itertuples():
                ticker = getattr(row, "ticker",       "?")
                entry  = float(getattr(row, "entry_price",  0))
                sl     = float(getattr(row, "stop_price",   0))
                tgt    = float(getattr(row, "target_price", 0))
                days   = getattr(row, "days_held", "?")
                print(f"  {C['cyan']}{ticker:<12}{C['reset']} "
                      f"₹{entry:>7.2f} "
                      f"{C['red']}₹{sl:>7.2f}{C['reset']} "
                      f"{C['green']}₹{tgt:>8.2f}{C['reset']} "
                      f"{days:>5}d")
        else:
            cp("dim", "  No open trades.")
    else:
        cp("dim", "  No paper trades file found.")

    if results:
        print()
        cp("bold", "─" * W)
        cp("dim",  "  ⚙️  PIPELINE TIMING")
        cp("bold", "─" * W)
        for name, r in results.items():
            col = "green" if r["ok"] else "red"
            print(f"  {C[col]}{'OK' if r['ok'] else 'FAIL'}{C['reset']}  {name:<16}  {r['elapsed']:.1f}s")
        print(f"  {'':6} {'TOTAL':<16}  {total_elapsed:.1f}s")

    print()
    cp("bold", "═" * W)
    cp("green", f"  ✅  DONE  |  {total_elapsed:.1f}s  |  {NOW}")
    cp("bold", "═" * W)
    print()


def run_full_pipeline(selected_layers=None):
    if selected_layers is None:
        selected_layers = select_layers()

    print()
    cp("bold", "─" * 65)
    cp("cyan",  f"  Running {len(selected_layers)} layer(s)  ·  {NOW}")
    cp("bold", "─" * 65)
    print()

    total   = len(selected_layers) + 1
    results = {}
    t_start = time.time()

    for i, (name, script) in enumerate(selected_layers, 1):
        ok, elapsed, err = run_layer(name, script, i, total)
        results[name] = {"ok": ok, "elapsed": elapsed, "error": err}
        if not ok and name in ("Layer2", "Layer3", "Layer4"):
            cp("red", f"\n  ABORT — {name} failed. Fix error and rerun.")
            break

    total_elapsed = time.time() - t_start

    try:
        _l2 = load("l2")
        _l3 = load("l3")
        _l4 = load("l4")
        _report = _build_eod_report(results, _l2, _l3, _l4)
        _rpath  = BASE / f"nse_layer5_eod_report_{TODAY}.txt"
        _rpath.write_text(_report, encoding="utf-8")
        _append_run_log(results, _l4, _l2)
    except Exception as _e:
        log.warning(f"EOD/run-log write failed: {_e}")

    try:
        subprocess.run(
            [PYTHON, str(BASE / "f1_paper_tracker.py"), "log"],
            cwd=str(BASE),
            timeout=30,
        )
        log.info("Paper tracker: signals logged.")
    except Exception as _e:
        log.warning(f"Paper tracker log failed: {_e}")

    print_cli_report(results, total_elapsed)


def run_dash_only():
    print()
    cp("bold", "─" * 65)
    cp("cyan",  f"  Dashboard-only mode  ·  {NOW}")
    cp("bold", "─" * 65)
    print_cli_report({}, 0)


def run_monitor():
    print()
    cp("bold", "─" * 65)
    cp("cyan",  f"  Monitor mode — open trades  ·  {NOW}")
    cp("bold", "─" * 65)
    print()

    trades = load("trades")
    if trades is None or "status" not in trades.columns:
        cp("yellow", "  No paper trades file found.")
        return

    open_t = trades[trades["status"] == "OPEN"]
    if len(open_t) == 0:
        cp("yellow", "  No open trades to monitor.")
        return

    try:
        import yfinance as yf
    except ImportError:
        cp("red", "  yfinance not installed.")
        return

    print(f"  {'TICKER':<14} {'ENTRY':>8} {'SL':>8} {'TARGET':>8} {'CMP':>8} {'PnL%':>7} {'ToTgt%':>7}  Status")
    print("  " + "─" * 72)

    for row in open_t.itertuples():
        ticker = getattr(row, "ticker", "?")
        entry  = float(getattr(row, "entry_price", 0))
        sl     = float(getattr(row, "stop_price",  0))
        tgt    = float(getattr(row, "target_price", 0))
        try:
            data = yf.download(ticker + ".NS", period="2d", progress=False, auto_adjust=True)
            cmp  = float(data["Close"].dropna().iloc[-1]) if not data.empty else None
        except Exception:
            cmp = None

        if cmp is None:
            print(f"  {ticker:<14} — fetch failed")
            continue

        pnl_pct    = (cmp - entry) / entry * 100
        to_tgt_pct = (tgt - cmp)  / cmp    * 100
        status     = f"{C['green']}🟢 OK{C['reset']}"  if cmp > entry else f"{C['red']}🔴 DOWN{C['reset']}"
        if abs(pnl_pct) < 0.5: status = f"{C['yellow']}🟡 FLAT{C['reset']}"
        if cmp >= tgt: status = f"{C['green']}✅ TARGET{C['reset']}"
        if cmp <= sl:  status = f"{C['red']}❌ SL HIT{C['reset']}"

        print(f"  {ticker:<14} {entry:>8.2f} {sl:>8.2f} {tgt:>8.2f} {cmp:>8.2f} {pnl_pct:>+6.1f}% {to_tgt_pct:>+6.1f}%  {status}")

    print()
    cp("dim", f"  {NOW}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dash",    action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--all",     action="store_true", help="Run all layers without prompting")
    parser.add_argument("--weekly",  action="store_true", help="Run Layer1-Val + Layer1-Sig (weekly maintenance)")
    args = parser.parse_args()

    if args.monitor:
        run_monitor()
    elif args.dash:
        run_dash_only()
    elif args.weekly:
        weekly = [(n, p) for n, p in SCRIPTS_WEEKLY.items() if p.exists()]
        if not weekly:
            cp("red", "  No weekly scripts found.")
        else:
            cp("cyan", f"  Weekly maintenance run: {[n for n,_ in weekly]}")
            run_full_pipeline(selected_layers=weekly)
    elif args.all:
        all_layers = [(n, p) for n, p in {**SCRIPTS, **SCRIPTS_WEEKLY}.items() if p.exists()]
        run_full_pipeline(selected_layers=all_layers)
    else:
        run_full_pipeline()