

"""
================================================================================
  NSE DECISION ENGINE  LAYER 6 DASHBOARD
  layer6_dashboard.py

  Modes:
    python layer6_dashboard.py           interactive layer selector + dashboard
    python layer6_dashboard.py --dash    dashboard only (use last parquets)
    python layer6_dashboard.py --monitor  monitor open trades vs live prices
    python layer6_dashboard.py --all     run all layers without prompting
================================================================================
"""

import subprocess, sys, os, time, argparse, webbrowser, json
import logging
from datetime import datetime, date
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE   = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
PYTHON = r"C:/Users/harik/AppData/Local/Programs/Python/Python312/python.exe"
TODAY    = date.today().isoformat()
NOW      = datetime.now().strftime("%Y-%m-%d %H:%M")
V2_START = date(2026, 6, 3)   # fixed pipeline start

SCRIPTS = {
    "Layer0-NSE"    : BASE / "layer0_nse_data.py",
    "Layer0-FnO"    : BASE / "layer0_fno.py",
    "Layer1-Heavy"  : BASE / "layer1_heavy_compute.py",
    "PatchSectors"  : BASE / "patch_sectors.py",       # runs before Layer1-Daily — keeps sectors fresh
    "Layer1-Daily"  : BASE / "layer1_daily.py",
    "Layer2"        : BASE / "layer2_daily_filter.py",
    "Layer3"        : BASE / "layer3_rules_engine.py",
    "Layer4"        : BASE / "layer4_portfolio.py",
}

# Weekly scripts — run in order after Layer1-Heavy
SCRIPTS_WEEKLY = {
    "Layer1-Val"  : BASE / "layer1_5_validation.py",   # IC backtest (fixed column names)
    "Layer1-Sig"  : BASE / "layer 1.55.py",             # backup signal scorer
    "Layer-TS"    : BASE / "TS layer testing.py",       # technical structure scan (~10 min)
}

LAYER_DESCRIPTIONS = {
    "Layer0-NSE"  : "Fetch NSE price/volume data for all tickers",
    "Layer0-FnO"  : "Fetch F&O options chain / PCR / OI / IV data",
    "Layer1-Heavy": "GARCH, sector momentum, pairs, clusters (SLOW ~5min)",
    "PatchSectors": "Fetch & cache sector classifications from Yahoo Finance (~1-2min)",
    "Layer1-Daily": "Sector momentum + spread Z + CUSUM incremental (FAST ~60s)",
    "Layer1-Val"  : "Signal validation / IC backtest",
    "Layer1-Sig"  : "Signal generation (momentum_12_1 etc.)",
    "Layer-TS"    : "Technical structure scan — S/R zones, entry/exit, weekly trend (~10 min)",
    "Layer2"      : "Daily filter  price, liquidity, regime gate",
    "Layer3"      : "Rules engine A/C/D/E  score all candidates",
    "Layer4"      : "Portfolio construction  FIRE/WAIT/KILL + sizing",
}

# Historical average runtimes (seconds) — used only to WEIGHT the overall
# progress bar so slow layers (Layer0-FnO, Layer-TS) don't count the same as
# fast ones (Layer4). Source: nse_run_log.csv / eod report timings.
LAYER_TIME_ESTIMATES = {
    "Layer0-NSE"   : 12,
    "Layer0-FnO"   : 470,
    "Layer1-Heavy" : 2700,
    "PatchSectors" : 10,
    "Layer1-Daily" : 240,
    "Layer1-Val"   : 4200,
    "Layer1-Sig"   : 30,
    "Layer-TS"     : 860,
    "Layer2"       : 60,
    "Layer3"       : 60,
    "Layer4"       : 2,
}
_DEFAULT_LAYER_ESTIMATE = 60

PARQUETS = {
    "l2"    : BASE / "nse_layer2_candidates.parquet",
    "l3"    : BASE / "nse_layer3_signals.parquet",
    "l4"    : BASE / "nse_layer4_portfolio.parquet",
    "trades":    BASE / "paper_trades.csv",
    "trades_v2": BASE / "paper_trades_v2.csv",
    "sector": BASE / "sector_momentum.parquet",
    "runlog": BASE / "nse_run_log.csv",
    "ts"    : BASE / "ts_analysis_report.csv",     # TS layer output
}

DASHBOARD_OUT = BASE / f"nse_dashboard_{TODAY}.html"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("L6")

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE LAYER SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_layers():
    all_layers = list(SCRIPTS.items())

    print()
    cp("bold", "─" * 72)
    cp("bold", "   NSE DECISION ENGINE    Layer 6    SELECT LAYERS TO RUN  ")
    cp("bold", "─" * 72)
    print()
    cp("dim",  "  Enter layer numbers separated by spaces  (e.g. 1 2 6 7 8)")
    cp("dim",  "  Press Enter with no input to run ALL layers")
    cp("dim",  "  Type 'q' to exit")
    print()

    cp("cyan", "  #   Layer             Exists   Description")
    cp("dim",  "  " + "─" * 70)

    for i, (name, path) in enumerate(all_layers, 1):
        exists  = path.exists()
        ex_mark = f"{C['green']}✔{C['reset']}" if exists else f"{C['red']}✗{C['reset']}"
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
            cp("red", "   Invalid input  enter numbers only (e.g. 1 2 6 7 8)")
            continue

        invalid = [n for n in nums if n < 1 or n > len(all_layers)]
        if invalid:
            cp("red", f"   Out of range: {invalid}  valid range 1-{len(all_layers)}")
            continue

        selected = []
        for n in nums:
            name, path = all_layers[n - 1]
            if not path.exists():
                cp("yellow", f"    {name}: script not found  skipping")
            else:
                selected.append((name, path))

        if not selected:
            cp("red", "   No valid layers selected.")
            continue

        print()
        cp("cyan", "  Selected layers:")
        for name, _ in selected:
            cp("white", f"     {name}    {LAYER_DESCRIPTIONS.get(name,'')}")
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

# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────

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

from rich.console import Console as _RConsole, Group as _RGroup
from rich.live import Live as _RLive
from rich.table import Table as _RTable
from rich.text import Text as _RText
from rich.spinner import Spinner as _RSpinner

# Set NSE_FORCE_RICH=1 to force the animated block even when stdout doesn't
# report as a TTY (e.g. some embedded terminals) — set NSE_FORCE_RICH=0 to
# force the plain compact-line fallback.
_force = os.environ.get("NSE_FORCE_RICH")
_console = _RConsole(
    force_terminal=(True if _force == "1" else False if _force == "0" else None)
)


def _fmt_dur(sec):
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class PipelineProgress:
    """Live weighted progress across all selected layers.

    On a real terminal: one self-updating block — an overall weighted bar plus
    a row per layer (done ✔ / running ⠹+spinner / pending), with elapsed + ETA.
    Finished layers weigh by ACTUAL elapsed; the in-flight layer weighs by
    elapsed/estimate (capped at 99% of its slice) so the bar never hits 100%
    before the last script exits.

    Off a terminal (output redirected/captured): the Live block is skipped and
    a single compact status line is emitted instead — only on a layer change or
    every HEARTBEAT_SECS — so a captured log stays ~20 lines, not thousands."""

    HEARTBEAT_SECS = 90   # non-TTY: min seconds between "still running" lines

    def __init__(self, layer_names, step_total=None, budget_seconds=None):
        self.names          = list(layer_names)
        self.step_total     = step_total or len(self.names)
        self.budget_seconds = budget_seconds
        self.t_start        = time.time()
        self.done_actual    = 0.0
        self.remaining      = list(layer_names)
        self.state          = {n: {"status": "pending", "elapsed": 0.0, "info": ""}
                               for n in self.names}
        self._is_tty        = _console.is_terminal
        self._live          = None
        self._spinner       = _RSpinner("dots", style="cyan")
        self._last_paint    = 0.0
        self._last_name     = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        if self._is_tty and self._live is None:
            self._live = _RLive(self._block(), console=_console,
                                refresh_per_second=8, transient=False)
            self._live.start()

    def stop(self):
        if self._live is not None:
            try:
                self._live.update(self._block())
                self._live.stop()
            finally:
                self._live = None

    # ── math ───────────────────────────────────────────────────────────────
    def _estimate(self, name):
        return LAYER_TIME_ESTIMATES.get(name, _DEFAULT_LAYER_ESTIMATE)

    def _pct(self, current_name=None, current_elapsed=0.0):
        remaining_sum = sum(self._estimate(n) for n in self.remaining)
        total_weight  = self.done_actual + remaining_sum
        cur_contrib   = 0.0
        if current_name and current_name in self.remaining:
            cur_est = self._estimate(current_name)
            if cur_est > 0:
                cur_contrib = min(current_elapsed / cur_est, 0.99) * cur_est
        return (self.done_actual + cur_contrib) / total_weight if total_weight > 0 else 0.0

    def _eta(self, pct):
        # Blend a fixed budget estimate with a pace extrapolation, handing off
        # from budget → pace as we cross 0..30 %. Extrapolation alone is wild
        # when pct is tiny (el/pct amplifies every wobble); budget alone never
        # adapts. Together they settle quickly and stay sane.
        el   = time.time() - self.t_start
        budg = max(0.0, self.budget_seconds - el) if self.budget_seconds else None
        pace = max(0.0, el / pct - el) if pct > 0.02 else None
        if pace is None:
            return budg or 0.0
        if budg is None:
            return pace
        w = min(pct / 0.30, 1.0)          # 0 at the start → 1 by 30 %
        return (1.0 - w) * budg + w * pace

    # ── renderable ─────────────────────────────────────────────────────────
    @staticmethod
    def _mkbar(frac, fill_color, width=22):
        frac   = max(0.0, min(frac, 1.0))
        filled = int(round(width * frac))
        return (f"[{fill_color}]" + "━" * filled + "[/]"
                f"[grey30]" + "━" * (width - filled) + "[/]")

    def _block(self, pct=None):
        el  = time.time() - self.t_start
        if pct is None:
            pct = self._pct()
        eta = self._eta(pct)

        head = _RText.from_markup(
            f"  [bold]NSE PIPELINE[/bold]  {self._mkbar(pct, 'cyan', 28)}"
            f"  [bold]{pct*100:4.0f}%[/bold]   {_fmt_dur(el)} • eta {_fmt_dur(eta)}"
        )

        grid = _RTable.grid(padding=(0, 1))
        grid.add_column(width=2, justify="center")   # icon / spinner
        grid.add_column(width=13)                     # layer name
        grid.add_column(width=22)                     # per-layer bar
        grid.add_column(width=6, justify="right")     # per-layer %
        grid.add_column(overflow="fold")             # elapsed + info
        for n in self.names:
            st  = self.state[n]
            est = self._estimate(n)
            if st["status"] == "done":
                grid.add_row("[green]✔[/]", f"[green]{n}[/]",
                             self._mkbar(1.0, "green"), "[green]100%[/]",
                             f"[grey62]{_fmt_dur(st['elapsed'])}[/]")
            elif st["status"] == "fail":
                frac = min(st["elapsed"] / est, 1.0) if est else 1.0
                grid.add_row("[red]✗[/]", f"[red]{n}[/]",
                             self._mkbar(frac, "red"), "",
                             f"[red]failed after {_fmt_dur(st['elapsed'])}[/]")
            elif st["status"] == "running":
                frac = min(st["elapsed"] / est, 0.99) if est else 0.0
                tail = f"[cyan]{_fmt_dur(st['elapsed'])}[/] / ~{_fmt_dur(est)}"
                if st["info"]:
                    tail += f"   [grey62]{st['info']}[/]"
                grid.add_row(self._spinner, f"[cyan]{n}[/]",
                             self._mkbar(frac, "cyan"), f"[cyan]{frac*100:3.0f}%[/]",
                             _RText.from_markup(tail))
            else:
                grid.add_row(" ", f"[grey42]{n}[/]",
                             self._mkbar(0.0, "grey30"), "",
                             "[grey42]pending[/]")

        return _RGroup(head, _RText(""), grid)

    # ── update hooks (called from run_layer) ───────────────────────────────
    def render(self, current_name, current_elapsed, step=None, status="running", info=""):
        st = self.state.get(current_name)
        if st is None:
            return
        if status == "ok":
            st["status"], st["elapsed"] = "done", current_elapsed
        elif status == "fail":
            st["status"], st["elapsed"] = "fail", current_elapsed
        else:
            st["status"], st["elapsed"] = "running", current_elapsed
            if info:
                st["info"] = info

        pct = self._pct(current_name, current_elapsed)

        if self._is_tty:
            if self._live is not None:
                self._live.update(self._block(pct))
            return

        # non-TTY: compact single line, throttled
        terminal = status in ("ok", "fail")
        changed  = current_name != self._last_name
        if not terminal and not changed and \
           (time.time() - self._last_paint) < self.HEARTBEAT_SECS:
            return
        self._last_name  = current_name
        self._last_paint = time.time()
        idx = self.names.index(current_name) + 1
        if terminal:
            tag = "OK  " if status == "ok" else "FAIL"
            print(f"  [{idx:>2}/{self.step_total}] {current_name:<16} {tag} {_fmt_dur(current_elapsed)}",
                  flush=True)
        else:
            extra = f"  ·  {info}" if info else ""
            print(f"  [{idx:>2}/{self.step_total}] {current_name:<16} running {_fmt_dur(current_elapsed)}"
                  f"  ·  {pct*100:.0f}%  ·  eta {_fmt_dur(self._eta(pct))}{extra}", flush=True)

    def finish_layer(self, name, elapsed):
        self.done_actual += elapsed
        if name in self.remaining:
            self.remaining.remove(name)
        self.state[name]["status"]  = "done"
        self.state[name]["elapsed"] = elapsed
        if self._is_tty and self._live is not None:
            self._live.update(self._block())

# ─────────────────────────────────────────────────────────────────────────────
# RUN A SINGLE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def run_layer(name, script, step, total, progress=None):
    t0           = time.time()
    stdout_lines = []
    last_info    = ""

    _env = os.environ.copy()
    _env["PYTHONUTF8"]       = "1"
    _env["PYTHONIOENCODING"] = "utf-8"

    # No banner, no per-layer log scroll — this layer's activity is folded
    # into the ONE progress line that spans the whole pipeline.
    try:
        proc = subprocess.Popen(
            [PYTHON, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_env,
            cwd=str(BASE),
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            stdout_lines.append(line)
            info = extract_info(line)
            if info:
                last_info = info
            if progress is not None:
                progress.render(name, time.time() - t0, step=step, status="running", info=last_info)

        proc.wait()
        elapsed = time.time() - t0

        if proc.returncode != 0:
            tail = "\n".join(stdout_lines[-10:])
            if progress is not None:
                progress.render(name, elapsed, step=step, status="fail", info=last_info)
                progress.stop()
            print()
            print(f"{C['red']}  └─ {name} FAILED  rc={proc.returncode}  ({elapsed:.1f}s){C['reset']}")
            print(f"{C['dim']}  ── last output before failure ──{C['reset']}")
            for _l in stdout_lines[-10:]:
                print(f"  {C['dim']}│{C['reset']} {_l}")
            log.error(f"[{name}] FAILED rc={proc.returncode}:\n{tail}")
            return False, elapsed, tail

        if progress is not None:
            progress.render(name, elapsed, step=step, status="ok", info=last_info)
            progress.finish_layer(name, elapsed)
        return True, elapsed, ""

    except Exception as exc:
        elapsed = time.time() - t0
        if progress is not None:
            progress.render(name, elapsed, step=step, status="fail", info=str(exc))
            progress.stop()
        print()
        print(f"{C['red']}  └─ {name} ERROR: {exc}  ({elapsed:.1f}s){C['reset']}")
        log.error(f"[{name}] Exception: {exc}")
        return False, elapsed, str(exc)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PARQUETS SAFELY
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD BUILDER
# ─────────────────────────────────────────────────────────────────────────────

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

    last_run_time    = "Never"
    last_run_layers  = "N/A"
    last_run_elapsed = "N/A"
    if runlog is not None and len(runlog) > 0:
        try:
            last_row = runlog.iloc[-1]
            last_run_time = str(last_row.get("date", "N/A"))
            sec_cols = [c for c in runlog.columns if c.endswith("_sec")]
            if sec_cols:
                total_sec_log = float(last_row[sec_cols].fillna(0).sum())
                last_run_elapsed = f"{int(total_sec_log)}s" if total_sec_log > 0 else "N/A"
            ok_cols = [c for c in runlog.columns if c.endswith("_ok")]
            if ok_cols:
                n_ok = int(last_row[ok_cols].fillna(False).sum())
                last_run_layers = f"{n_ok}/{len(ok_cols)} OK"
        except Exception:
            pass

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
            ticker   = getattr(row, "ticker", "?")
            tier     = getattr(row, "tier", "?")
            entry    = float(getattr(row, "entry_price",  getattr(row, "price_close", 0)))
            sl       = float(getattr(row, "stop_loss",    0))
            tgt      = float(getattr(row, "target_price", 0))
            qty      = getattr(row, "qty",            getattr(row, "position_qty", "?"))
            score    = getattr(row, sc, 0.0)
            sec_n    = str(getattr(row, "sector", ""))[:18]
            direction = str(getattr(row, "direction", "LONG")).upper()
            if direction == "BOTH": direction = "LONG"
            is_short  = direction == "SHORT"
            if is_short:
                rr     = round((entry - tgt) / (sl - entry), 2) if (sl - entry) > 0 else 0
                risk_r = round((sl - entry) * float(qty), 0)    if qty != "?" else 0
                dir_badge = '<span class="dir-badge-short">▼ SHORT</span>'
            else:
                rr     = round((tgt - entry) / (entry - sl), 2) if (entry - sl) > 0 else 0
                risk_r = round((entry - sl) * float(qty), 0)    if qty != "?" else 0
                dir_badge = '<span class="dir-badge-long">▲ LONG</span>'
            smc_sig  = int(getattr(row, "smc_signal", 0))
            smc_conv = float(getattr(row, "smc_conviction", 0.0))
            smc_wyck = str(getattr(row, "smc_wyckoff", ""))[:12]
            if smc_sig == 1:
                smc_badge = f'<span class="smc-badge smc-long">▲ {smc_conv:+.1f}</span>'
            elif smc_sig == -1:
                smc_badge = f'<span class="smc-badge smc-short">▼ {smc_conv:+.1f}</span>'
            else:
                smc_badge = f'<span class="smc-badge smc-neut">— {smc_conv:+.1f}</span>'
            ts_used  = bool(getattr(row, "ts_used", False))
            ts_t1_v  = getattr(row, "ts_t1", None)
            ts_src   = str(getattr(row, "ts_stop_source", ""))
            if ts_used and ts_t1_v and str(ts_t1_v) not in ("None", "nan"):
                t1_cell = f'<span class="ts-t1">{float(ts_t1_v):,.2f}</span>'
                sl_cell = f'<span class="sl-anchor" title="S/R-anchored ({ts_src})">{sl:,.2f} <span class="anchor-icon">⚓</span></span>'
            else:
                t1_cell = '<span class="dim-text">—</span>'
                sl_cell = f'<span class="num sl">{sl:,.2f}</span>'
            # Rule tags for fire row
            fire_rule_tags = ""
            for _rc, _ch, _cls in [("rule_a_fires","A","tag-a"),("rule_c_fires","C","tag-c"),
                                    ("rule_d_fires","D","tag-d"),("rule_e_fires","E","tag-e"),
                                    ("smc_rule_fires","S","tag-s")]:
                if _rc in fire_df.columns and getattr(row, _rc, False):
                    fire_rule_tags += f'<span class="rule-tag {_cls}">{_ch}</span>'
            if not fire_rule_tags:
                fire_rule_tags = '<span class="rule-tag tag-none">—</span>'
            fire_rows += f"""
            <tr class="fire-row">
              <td><span class="ticker-badge">{ticker}</span><div class="fire-rules">{fire_rule_tags}</div></td>
              <td>{dir_badge}</td>
              <td><span class="tier tier-{str(tier).lower()}">{tier}</span></td>
              <td class="dim-text">{sec_n}</td>
              <td class="num fw">{entry:,.2f}</td>
              <td class="num">{sl_cell}</td>
              <td class="num t1-col">{t1_cell}</td>
              <td class="num tgt fw">{tgt:,.2f}</td>
              <td class="num">{qty}</td>
              <td class="num rr-col">1:{rr}</td>
              <td class="num risk">{risk_r:,.0f}</td>
              <td>{smc_badge}<span class="wyck-tag">{smc_wyck}</span></td>
              <td><div class="score-bar-wrap"><div class="score-bar" style="width:{min(score*100,100):.0f}%"></div><span class="score-num">{score:.3f}</span></div></td>
            </tr>"""
    else:
        fire_rows = "<tr><td colspan='13' class='no-data'>No FIRE signals today — check WAIT list below</td></tr>"

    wait_rows = ""
    if len(wait_df) > 0 and sc in wait_df.columns:
        for i, row in enumerate(wait_df.nlargest(15, sc).itertuples(), 1):
            ticker   = getattr(row, "ticker", "?")
            tier     = getattr(row, "tier", "?")
            price    = getattr(row, "entry_price", getattr(row, "price_close", 0))
            score    = getattr(row, sc, 0.0)
            sec_n    = str(getattr(row, "sector", ""))[:18]
            w_dir    = str(getattr(row, "direction", "LONG")).upper()
            if w_dir == "SHORT":
                w_dir_badge = '<span class="dir-badge-short" style="font-size:9px;padding:1px 4px">▼S</span>'
            else:
                w_dir_badge = '<span class="dir-badge-long" style="font-size:9px;padding:1px 4px">▲L</span>'
            wait_rows += f"""
            <tr>
              <td class="rank-num">{i}</td>
              <td><span class="ticker-badge small">{ticker}</span></td>
              <td>{w_dir_badge}</td>
              <td><span class="tier tier-{str(tier).lower()}">{tier}</span></td>
              <td>{sec_n}</td>
              <td class="num">{price:,.2f}</td>
              <td><div class="score-bar-wrap"><div class="score-bar wait-bar" style="width:{min(score*100,100):.0f}%"></div><span>{score:.3f}</span></div></td>
            </tr>"""

    l3_rows = ""
    if l3 is not None and "final_score" in l3.columns:
        for i, row in enumerate(l3.nlargest(10, "final_score").itertuples(), 1):
            ticker  = getattr(row, "ticker", "?")
            tier    = getattr(row, "tier", "?")
            price   = getattr(row, "price_close", 0)
            score   = getattr(row, "final_score", 0)
            sec_n   = str(getattr(row, "sector", ""))[:18]
            l3_dir  = str(getattr(row, "direction", "LONG")).upper()
            if l3_dir == "SHORT":
                l3_dir_badge = '<span class="dir-badge-short" style="font-size:9px;padding:1px 4px">▼S</span>'
            else:
                l3_dir_badge = '<span class="dir-badge-long" style="font-size:9px;padding:1px 4px">▲L</span>'
            tags   = ""
            for col, ch, cls in [("rule_a_fires","A","tag-a"),("rule_c_fires","C","tag-c"),
                                  ("rule_d_fires","D","tag-d"),("rule_e_fires","E","tag-e"),
                                  ("rule_a_short_fires","As","tag-a"),("rule_c_short_fires","Cs","tag-c"),
                                  ("rule_e_short_fires","Es","tag-e"),("smc_rule_fires","S","tag-s")]:
                if col in l3.columns and getattr(row, col, False):
                    tags += f'<span class="rule-tag {cls}">{ch}</span>'
            l3_rows += f"""
            <tr>
              <td class="rank-num">{i}</td>
              <td><span class="ticker-badge small">{ticker}</span></td>
              <td>{l3_dir_badge}</td>
              <td><span class="tier tier-{str(tier).lower()}">{tier}</span></td>
              <td>{sec_n}</td>
              <td class="num">{price:,.2f}</td>
              <td>{tags if tags else '<span class="tag-none">—</span>'}</td>
              <td><div class="score-bar-wrap"><div class="score-bar l3-bar" style="width:{min(score*100,100):.0f}%"></div><span>{score:.3f}</span></div></td>
            </tr>"""

    sector_rows = ""
    if sector is not None:
        valid_sectors = sector[
            (sector["Sector"].str.strip() != "") &
            (sector["Sector"].str.strip().str.lower() != "unknown")
        ].sort_values("Rank")
        if len(valid_sectors) == 0:
            sector_rows = "<tr><td colspan='3' class='no-data'>Run <code>patch_sectors.py</code> then Layer1-Daily</td></tr>"
        else:
            for row in valid_sectors.itertuples():
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
            for row in open_t.sort_values("days_held", ascending=False).itertuples() if "days_held" in open_t.columns else open_t.itertuples():
                ticker   = getattr(row, "ticker",       "?")
                entry    = float(getattr(row, "entry_price",  0))
                sl       = float(getattr(row, "stop_price",   0))
                tgt      = float(getattr(row, "target_price", 0))
                days_raw = getattr(row, "days_held", "")
                days     = int(float(str(days_raw))) if str(days_raw).strip() not in ("","nan","NaN","?") else 0
                rule     = str(getattr(row, "primary_rule", "?"))
                _dir     = str(getattr(row, "direction", "LONG")).upper()
                score    = float(getattr(row, "composite_score", getattr(row, "final_score", 0)))
                sdate    = str(getattr(row, "signal_date", ""))[:10]
                dir_badge = f'<span class="dir-badge-short" style="font-size:9px;padding:1px 4px">▼ S</span>' if _dir == "SHORT" else f'<span class="dir-badge-long" style="font-size:9px;padding:1px 4px">▲ L</span>'
                if _dir == "SHORT":
                    pct_to_tgt = round((entry - tgt) / entry * 100, 1)
                    pct_to_sl  = round((sl - entry)  / entry * 100, 1)
                    rng        = abs(sl - tgt)
                else:
                    pct_to_tgt = round((tgt - entry) / entry * 100, 1)
                    pct_to_sl  = round((entry - sl)  / entry * 100, 1)
                    rng        = abs(tgt - sl)
                # Days bar — max 25 days
                max_hold = 25
                days_pct = min(100, round(days / max_hold * 100))
                days_col = "var(--green)" if days_pct < 40 else "var(--yellow)" if days_pct < 75 else "var(--red)"
                trade_rows += f"""
                <tr class="open-trade-row">
                  <td><span class="ticker-badge small">{ticker}</span></td>
                  <td>{dir_badge}</td>
                  <td><span class="rule-tag tag-{rule.lower()}">{rule}</span></td>
                  <td class="num fw">{entry:,.2f}</td>
                  <td class="num sl">{sl:,.2f}</td>
                  <td class="num tgt">{tgt:,.2f}</td>
                  <td class="num neg">-{pct_to_sl:.1f}%</td>
                  <td class="num pos">+{pct_to_tgt:.1f}%</td>
                  <td>
                    <div class="days-wrap">
                      <div class="days-bar"><div class="days-fill" style="width:{days_pct}%;background:{days_col}"></div></div>
                      <span class="days-num" style="color:{days_col}">{days}d</span>
                    </div>
                  </td>
                  <td class="dim-text">{sdate}</td>
                  <td class="dim-text">{score:.3f}</td>
                </tr>"""
            open_trades_html = f"""
            <section class="section">
              <h2 class="section-title">📋 Open Trades <span class="badge">{len(open_t)}</span>
                <span style="margin-left:auto;font-size:10px;color:var(--dim);font-weight:400">sorted by days held</span>
              </h2>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Ticker</th><th>Dir</th><th>Rule</th><th>Entry</th><th>SL</th><th>Target</th><th>Risk%</th><th>Reward%</th><th>Days</th><th>Since</th><th>Score</th></tr></thead>
                  <tbody>{trade_rows}</tbody>
                </table>
              </div>
            </section>"""

    rule_a = rule_c = rule_d = rule_e = rule_s = 0
    smc_bull_cnt = smc_bear_cnt = smc_neut_cnt = 0
    wyckoff_dist = {}
    if l3 is not None:
        rule_a = int(l3["rule_a_fires"].sum())   if "rule_a_fires"   in l3.columns else 0
        rule_c = int(l3["rule_c_fires"].sum())   if "rule_c_fires"   in l3.columns else 0
        rule_d = int(l3["rule_d_fires"].sum())   if "rule_d_fires"   in l3.columns else 0
        rule_e = int(l3["rule_e_fires"].sum())   if "rule_e_fires"   in l3.columns else 0
        rule_s = int(l3["smc_rule_fires"].sum()) if "smc_rule_fires" in l3.columns else 0
        if "smc_signal" in l3.columns:
            smc_bull_cnt = int((l3["smc_signal"] ==  1).sum())
            smc_bear_cnt = int((l3["smc_signal"] == -1).sum())
            smc_neut_cnt = int((l3["smc_signal"] ==  0).sum())
        if "smc_wyckoff" in l3.columns:
            wyckoff_dist = l3[l3["smc_wyckoff"] != "Undefined"]["smc_wyckoff"].value_counts().head(5).to_dict()

    candidates = len(l2) if l2 is not None else 0
    top_score  = float(l4[sc].max()) if l4 is not None and sc in l4.columns else 0.0

    wyckoff_html = ""
    if wyckoff_dist:
        for phase, cnt in wyckoff_dist.items():
            wyckoff_html += f'<span class="wyck-item">{phase}<span class="wyck-count">×{cnt}</span></span>'
    else:
        wyckoff_html = '<span style="color:var(--dim);font-size:11px;font-family:var(--mono)">No Wyckoff data</span>'

    # ── Paper trades stats (all trades) ─────────────────────────────────────
    pt_open = pt_win = pt_loss = pt_exp = 0
    pt_wr = 0.0
    pt_avg_pnl = 0.0
    if trades is not None and "status" in trades.columns:
        pt_open   = int((trades["status"] == "OPEN").sum())
        pt_win    = int((trades["status"] == "WIN").sum())
        pt_loss   = int((trades["status"] == "LOSS").sum())
        pt_exp    = int((trades["status"] == "EXPIRED").sum())
        _closed_n = pt_win + pt_loss
        pt_wr     = round(pt_win / _closed_n * 100, 1) if _closed_n > 0 else 0.0
        _pnl_vals = pd.to_numeric(trades.get("pnl_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        pt_avg_pnl = round(float(_pnl_vals.mean()), 2) if len(_pnl_vals) > 0 else 0.0
    pt_wr_col  = "val-green" if pt_wr >= 38 else "val-wait" if pt_wr >= 33 else "val-fire"
    pt_pnl_col = "val-green" if pt_avg_pnl >= 0 else "val-fire"

    # ── v2 pipeline stats (fixed pipeline only) ──────────────────────────────
    trades_v2 = load("trades_v2")
    v2_open = v2_win = v2_loss = v2_exp = v2_total = 0
    v2_wr = v2_avg_pnl = 0.0
    v2_html = ""
    if trades_v2 is not None and "status" in trades_v2.columns and len(trades_v2) > 0:
        v2_total  = len(trades_v2)
        v2_open   = int((trades_v2["status"] == "OPEN").sum())
        v2_win    = int((trades_v2["status"] == "WIN").sum())
        v2_loss   = int((trades_v2["status"] == "LOSS").sum())
        v2_exp    = int((trades_v2["status"] == "EXPIRED").sum())
        _v2_cl    = v2_win + v2_loss
        v2_wr     = round(v2_win / _v2_cl * 100, 1) if _v2_cl > 0 else 0.0
        _v2_pnl   = pd.to_numeric(trades_v2["pnl_pct"], errors="coerce").dropna()
        v2_avg_pnl = round(float(_v2_pnl.mean()), 2) if len(_v2_pnl) > 0 else 0.0
        _v2_wr_col  = "c-green" if v2_wr >= 38 else "c-wait" if v2_wr >= 33 else "c-fire"
        _v2_pnl_col = "c-green" if v2_avg_pnl >= 0 else "c-fire"

        # Build v2 open trades rows
        v2_open_t = trades_v2[trades_v2["status"] == "OPEN"]
        v2_tr = ""
        for row in v2_open_t.sort_values("signal_date", ascending=False).itertuples() if "signal_date" in v2_open_t.columns else v2_open_t.itertuples():
            _tk   = getattr(row, "ticker", "?")
            _dir  = str(getattr(row, "direction", "LONG")).upper()
            _rule = str(getattr(row, "primary_rule", "?"))
            _ent  = float(getattr(row, "entry_price", 0))
            _sl   = float(getattr(row, "stop_price",  0))
            _tgt  = float(getattr(row, "target_price", 0))
            _scr  = float(getattr(row, "composite_score", 0))
            _tier = str(getattr(row, "tier", "?"))
            _sec  = str(getattr(row, "sector", ""))[:16]
            _sdate = str(getattr(row, "signal_date", ""))[:10]
            _db   = f'<span class="dir-badge-short" style="font-size:9px;padding:1px 4px">▼S</span>' if _dir == "SHORT" else f'<span class="dir-badge-long" style="font-size:9px;padding:1px 4px">▲L</span>'
            if _dir == "SHORT":
                _pct_sl  = round((_sl - _ent) / _ent * 100, 1)
                _pct_tgt = round((_ent - _tgt) / _ent * 100, 1)
            else:
                _pct_sl  = round((_ent - _sl) / _ent * 100, 1)
                _pct_tgt = round((_tgt - _ent) / _ent * 100, 1)
            v2_tr += f"""
            <tr class="open-trade-row">
              <td><span class="ticker-badge small">{_tk}</span></td>
              <td>{_db}</td>
              <td><span class="rule-tag tag-{_rule.lower()}">{_rule}</span></td>
              <td><span class="tier tier-{_tier.lower()}">{_tier}</span></td>
              <td class="dim-text">{_sec}</td>
              <td class="num fw">{_ent:,.2f}</td>
              <td class="num sl">{_sl:,.2f}</td>
              <td class="num tgt">{_tgt:,.2f}</td>
              <td class="num neg">-{_pct_sl:.1f}%</td>
              <td class="num pos">+{_pct_tgt:.1f}%</td>
              <td class="dim-text">{_sdate}</td>
              <td class="dim-text">{_scr:.3f}</td>
            </tr>"""

        # v2 closed rows
        v2_cl_t = trades_v2[trades_v2["status"].isin(["WIN","LOSS","EXPIRED"])].copy()
        v2_cl_rows = ""
        if len(v2_cl_t) > 0:
            v2_cl_t["pnl_pct"] = pd.to_numeric(v2_cl_t["pnl_pct"], errors="coerce").fillna(0)
            for crow in v2_cl_t.sort_values("exit_date", ascending=False).itertuples():
                _tk  = getattr(crow, "ticker", "?")
                _st  = getattr(crow, "status", "?")
                _dir = str(getattr(crow, "direction", "LONG")).upper()
                _ent = float(getattr(crow, "entry_price", 0))
                _ex  = getattr(crow, "exit_price", "")
                _ex  = float(_ex) if str(_ex).strip() not in ("","nan","NaN") else 0
                _pnl = float(getattr(crow, "pnl_pct", 0))
                _d   = int(float(str(getattr(crow, "days_held", 0)))) if str(getattr(crow, "days_held", "0")).strip() not in ("","nan") else 0
                _rule = str(getattr(crow, "primary_rule", "?"))
                _edate = str(getattr(crow, "exit_date", ""))[:10]
                _reason = str(getattr(crow, "exit_reason", ""))[:20]
                st_cls  = "win" if _st == "WIN" else "loss" if _st == "LOSS" else "exp"
                pnl_cls = "pos" if _pnl >= 0 else "neg"
                _db = f'<span class="dir-badge-short" style="font-size:9px;padding:1px 4px">▼S</span>' if _dir == "SHORT" else f'<span class="dir-badge-long" style="font-size:9px;padding:1px 4px">▲L</span>'
                v2_cl_rows += f"""
                <tr>
                  <td><span class="ticker-badge small">{_tk}</span></td>
                  <td>{_db}</td>
                  <td><span class="rule-tag tag-{_rule.lower()}">{_rule}</span></td>
                  <td class="status-{st_cls} fw">{_st}</td>
                  <td class="num">{_ent:,.2f}</td>
                  <td class="num">{_ex:,.2f}</td>
                  <td class="num {pnl_cls} fw">{_pnl:+.2f}%</td>
                  <td class="num">{_d}d</td>
                  <td class="dim-text">{_edate}</td>
                  <td class="dim-text">{_reason}</td>
                </tr>"""

        _v2_wr_display = f"{v2_wr}%" if _v2_cl > 0 else "—"
        _be_color = "c-green" if v2_wr >= 33.3 else "c-fire"
        v2_html = f"""
        <div class="card accent-green" style="border-top-color:var(--teal)">
          <div class="card-header" style="background:linear-gradient(90deg,rgba(38,198,218,.08),transparent)">
            <h2 style="color:var(--teal)">🆕 v2 Pipeline — Fixed System</h2>
            <div class="ch-right">
              <span class="badge" style="background:rgba(38,198,218,.1);color:var(--teal);border:1px solid rgba(38,198,218,.2)">TS stops · SHORT support · since {V2_START}</span>
              <span class="badge">{v2_total} trades</span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)">
            <div class="kpi-cell"><span class="kpi-val c-cyan">{v2_open}</span><span class="kpi-lbl">Open</span></div>
            <div class="kpi-cell"><span class="kpi-val c-green">{v2_win}</span><span class="kpi-lbl">Won</span></div>
            <div class="kpi-cell"><span class="kpi-val c-fire">{v2_loss}</span><span class="kpi-lbl">Lost</span></div>
            <div class="kpi-cell"><span class="kpi-val c-kill">{v2_exp}</span><span class="kpi-lbl">Expired</span></div>
            <div class="kpi-cell"><span class="kpi-val {_v2_wr_col} fw">{_v2_wr_display}</span><span class="kpi-lbl">🎯 Win Rate</span></div>
            <div class="kpi-cell"><span class="kpi-val {_v2_pnl_col}">{v2_avg_pnl:+.2f}%</span><span class="kpi-lbl">💰 Avg PnL</span></div>
            <div class="kpi-cell"><span class="kpi-val" style="color:var(--muted);font-size:12px">≥33.3%</span><span class="kpi-lbl">Breakeven WR</span></div>
          </div>
          <div class="grid-2" style="gap:0;border-bottom:1px solid var(--border)">
            <div>
              <div style="padding:8px 16px 4px;font-size:9px;color:var(--teal);font-weight:700;text-transform:uppercase;letter-spacing:.1em">Open Positions</div>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Ticker</th><th>Dir</th><th>Rule</th><th>Tier</th><th>Sector</th><th class="num">Entry</th><th class="num">SL</th><th class="num">Target</th><th class="num">Risk%</th><th class="num">Reward%</th><th>Since</th><th>Score</th></tr></thead>
                  <tbody>{''.join([v2_tr]) if v2_tr else '<tr><td colspan="12" class="no-data">No open v2 trades</td></tr>'}</tbody>
                </table>
              </div>
            </div>
            <div style="border-left:1px solid var(--border)">
              <div style="padding:8px 16px 4px;font-size:9px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.1em">Closed Trades</div>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Ticker</th><th>Dir</th><th>Rule</th><th>Status</th><th class="num">Entry</th><th class="num">Exit</th><th class="num">PnL%</th><th class="num">Days</th><th>Date</th><th>Reason</th></tr></thead>
                  <tbody>{''.join([v2_cl_rows]) if v2_cl_rows else '<tr><td colspan="10" class="no-data">No closed v2 trades yet — keep running</td></tr>'}</tbody>
                </table>
              </div>
            </div>
          </div>
        </div>"""

    # ── TS Analysis HTML section ──────────────────────────────────────────────
    ts_html = ""
    ts_df   = load("ts")
    if ts_df is not None and len(ts_df) > 0:
        def _ts_rows(subset, max_n=5):
            rows = ""
            for _, r in subset.head(max_n).iterrows():
                sig   = str(r.get("entry_signal",""))
                sig_c = ("val-green" if "BUY" in sig else "val-fire" if "SHORT" in sig else "val-wait")
                et    = str(r.get("entry_type",""))
                fno   = "🌙" if r.get("fno_eligible") else ""
                rows += f"""
                <tr>
                  <td><span class="ticker-badge small">{r['ticker']}</span></td>
                  <td><span class="tier tier-{str(r.get('tier','t1')).lower()}">{r.get('tier','?')}</span></td>
                  <td class="{sig_c}" style="font-size:10px;font-weight:700">{sig}</td>
                  <td style="font-size:10px">{et}</td>
                  <td class="num">{float(r.get('key_level',0)):,.2f}</td>
                  <td class="num sl">{float(r.get('stop',0)):,.2f}</td>
                  <td class="num" style="color:var(--yellow)">{float(r.get('t1',0)):,.2f}</td>
                  <td class="num tgt">{float(r.get('t2',0)):,.2f}</td>
                  <td class="num">{float(r.get('rr',0)):.2f}</td>
                  <td style="font-size:10px">{fno}{str(r.get('patterns',''))[:22]}</td>
                </tr>"""
            return rows

        ts_long  = ts_df[(ts_df.get("direction","")=="LONG")  & (ts_df.get("entry_signal","").isin(["STRONG_BUY","BUY"]))  & (ts_df.get("rr_valid",False)==True)].nlargest(5,"setup_score")
        ts_short = ts_df[(ts_df.get("direction","")=="SHORT") & (ts_df.get("entry_signal","").isin(["STRONG_SHORT","SHORT"])) & (ts_df.get("rr_valid",False)==True)].nlargest(5,"setup_score")

        tbl_hdr = "<thead><tr><th>Ticker</th><th>Tier</th><th>Signal</th><th>Type</th><th>Key Level</th><th>Stop</th><th>T1</th><th>T2</th><th>RR</th><th>Patterns</th></tr></thead>"
        ts_html = f"""
        <div class="card accent-cyan">
          <div class="card-header">
            <h2>📡 TS Analysis — S/R Setups</h2>
            <div class="ch-right">
              <span class="badge badge-ts">Weekly scan · {len(ts_df)} stocks</span>
            </div>
          </div>
          <div class="grid-2">
            <div>
              <div class="ts-subheader" style="color:var(--green)">▲ Long Setups</div>
              <div class="table-wrap">
                <table>{tbl_hdr}<tbody>{_ts_rows(ts_long)}</tbody></table>
              </div>
            </div>
            <div>
              <div class="ts-subheader" style="color:var(--red)">▼ Short Setups</div>
              <div class="table-wrap">
                <table>{tbl_hdr}<tbody>{_ts_rows(ts_short)}</tbody></table>
              </div>
            </div>
          </div>
        </div>"""

    # ── Closed trades HTML section ────────────────────────────────────────────
    closed_trades_html = ""
    if trades is not None and "status" in trades.columns:
        closed_t = trades[trades["status"].isin(["WIN","LOSS","EXPIRED"])].copy()
        if len(closed_t) > 0:
            closed_t["pnl_pct"]   = pd.to_numeric(closed_t.get("pnl_pct",   pd.Series(dtype=float)), errors="coerce").fillna(0)
            closed_t["days_held"] = pd.to_numeric(closed_t.get("days_held", pd.Series(dtype=float)), errors="coerce").fillna(0)
            # Rule breakdown
            rule_stats = {}
            if "primary_rule" in closed_t.columns:
                for rule, grp in closed_t.groupby("primary_rule"):
                    wins  = int((grp["status"] == "WIN").sum())
                    total = len(grp)
                    rule_stats[rule] = (wins, total, round(wins/total*100,1) if total else 0)
            rule_pills_html = ""
            for rule, (w, t, wr) in sorted(rule_stats.items()):
                col = "#00e676" if wr >= 38 else "#ffd740" if wr >= 33 else "#ff4455"
                rule_pills_html += f'<div class="rule-pill"><span class="rp-count" style="color:{col}">{wr}%</span><span class="rp-name">Rule {rule} ({w}/{t})</span></div>'

            closed_rows_html = ""
            for crow in closed_t.sort_values("exit_date", ascending=False).head(20).itertuples():
                _tk     = getattr(crow, "ticker",      "?")
                _status = getattr(crow, "status",      "?")
                _entry  = float(getattr(crow, "entry_price", 0))
                _exit_v = getattr(crow, "exit_price", "")
                _exit   = float(_exit_v) if str(_exit_v).strip() not in ("","nan","NaN") else 0
                _pnl    = float(getattr(crow, "pnl_pct",    0))
                _days   = int(getattr(crow, "days_held",    0))
                _edate  = str(getattr(crow, "exit_date",    ""))[:10]
                _rule   = str(getattr(crow, "primary_rule", "?"))
                _dir    = str(getattr(crow, "direction",    "LONG")).upper()
                _reason = str(getattr(crow, "exit_reason",  ""))
                st_cls  = "win" if _status == "WIN" else "loss" if _status == "LOSS" else "exp"
                pnl_cls = "pos" if _pnl >= 0 else "neg"
                dir_badge = f'<span class="dir-short">▼</span>' if _dir == "SHORT" else f'<span class="dir-long">▲</span>'
                closed_rows_html += f"""
                <tr>
                  <td><span class="ticker-badge small">{_tk}</span></td>
                  <td>{dir_badge}</td>
                  <td><span class="rule-tag tag-{_rule.lower()}">{_rule}</span></td>
                  <td class="status-{st_cls}">{_status}</td>
                  <td class="num">{_entry:,.2f}</td>
                  <td class="num">{_exit:,.2f}</td>
                  <td class="num {pnl_cls}">{_pnl:+.2f}%</td>
                  <td class="num">{_days}d</td>
                  <td class="num dim-text">{_edate}</td>
                  <td class="dim-text">{_reason[:22]}</td>
                </tr>"""

            _wr_color = 'var(--green)' if pt_wr>=38 else 'var(--yellow)' if pt_wr>=33 else 'var(--red)'
            closed_trades_html = f"""
            <div class="card">
              <div class="card-header">
                <h2>📊 Closed Trades</h2>
                <div class="ch-right">
                  <span class="badge">{len(closed_t)} total</span>
                  <span style="font-family:var(--mono);font-size:11px;color:var(--green)">{pt_win}W</span>
                  <span style="font-family:var(--mono);font-size:11px;color:var(--red)">{pt_loss}L</span>
                  <span style="font-family:var(--mono);font-size:11px;color:var(--muted)">{pt_exp}EXP</span>
                  <span style="font-family:var(--mono);font-size:13px;font-weight:700;color:{_wr_color}">{pt_wr}% WR</span>
                </div>
              </div>
              {'<div class="rule-pills-wrap">' + rule_pills_html + '</div>' if rule_pills_html else ''}
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Ticker</th><th>Dir</th><th>Rule</th><th>Status</th><th class="num">Entry</th><th class="num">Exit</th><th class="num">PnL%</th><th class="num">Days</th><th>Date</th><th>Reason</th></tr></thead>
                  <tbody>{closed_rows_html}</tbody>
                </table>
              </div>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSE Decision Engine — {TODAY}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root{{
  --bg:#060a10;--bg2:#0a1018;--bg3:#0e1520;--bg4:#121b28;
  --border:#182235;--border2:#1e2d44;
  --text:#d0e0f4;--dim:#3d5a7a;--muted:#5a7a9a;
  --green:#00e676;--red:#ff3d57;--yellow:#ffc107;
  --cyan:#29b6f6;--orange:#ff7043;--fire:#ff5722;
  --purple:#ce93d8;--indigo:#7986cb;--teal:#26c6da;
  --mono:'JetBrains Mono',monospace;--sans:'Inter',sans-serif;
  --radius:8px;--radius-sm:5px;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.6;-webkit-font-smoothing:antialiased;}}

/* ── HEADER ── */
.header{{
  background:linear-gradient(180deg,rgba(10,16,24,1) 0%,rgba(6,10,16,.95) 100%);
  border-bottom:1px solid var(--border2);
  padding:14px 28px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:200;backdrop-filter:blur(12px);
}}
.header-left h1{{font-family:var(--mono);font-size:15px;font-weight:600;color:var(--cyan);letter-spacing:.05em;}}
.header-left p{{font-size:10px;color:var(--muted);margin-top:1px;font-family:var(--mono);}}
.header-right{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}}
.regime-pill{{font-family:var(--mono);font-size:10px;padding:4px 12px;border-radius:20px;border:1px solid;font-weight:600;text-transform:uppercase;letter-spacing:.1em;}}
.regime-sideways{{color:var(--yellow);border-color:rgba(255,193,7,.35);background:rgba(255,193,7,.07);}}
.regime-bull{{color:var(--green);border-color:rgba(0,230,118,.35);background:rgba(0,230,118,.07);}}
.regime-crisis{{color:var(--red);border-color:rgba(255,61,87,.35);background:rgba(255,61,87,.07);}}
.h-stat{{display:flex;flex-direction:column;align-items:center;padding:5px 14px;border-radius:var(--radius-sm);background:rgba(255,255,255,.03);border:1px solid var(--border);min-width:58px;}}
.h-stat .hv{{font-family:var(--mono);font-size:18px;font-weight:700;line-height:1;}}
.h-stat .hl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:2px;}}
.c-fire{{color:var(--fire);}}.c-wait{{color:var(--yellow);}}.c-kill{{color:var(--muted);}}
.c-green{{color:var(--green);}}.c-cyan{{color:var(--cyan);}}.c-purple{{color:var(--purple);}}
.val-fire{{color:var(--fire);}}.val-wait{{color:var(--yellow);}}.val-green{{color:var(--green);}}

/* ── LAYOUT ── */
.container{{max-width:1800px;margin:0 auto;padding:18px 28px;}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}}
.grid-4{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;}}

/* ── CARDS ── */
.card{{
  background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);
  overflow:hidden;margin-bottom:12px;
  animation:slideUp .3s ease both;
  transition:border-color .2s,box-shadow .2s;
}}
.card:hover{{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.4);}}
.card-header{{
  display:flex;align-items:center;gap:8px;
  padding:10px 16px;border-bottom:1px solid var(--border);
  background:var(--bg3);
}}
.card-header h2{{font-family:var(--mono);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);}}
.card-header .ch-right{{margin-left:auto;display:flex;align-items:center;gap:6px;}}
.accent-fire{{border-top:2px solid var(--fire)!important;}}
.accent-cyan{{border-top:2px solid var(--cyan)!important;}}
.accent-green{{border-top:2px solid var(--green)!important;}}
.accent-purple{{border-top:2px solid var(--purple)!important;}}
.accent-yellow{{border-top:2px solid var(--yellow)!important;}}

/* ── KPI STRIP ── */
.kpi-strip{{display:grid;grid-template-columns:repeat(10,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:12px;}}
.kpi-cell{{background:var(--bg2);padding:14px 16px;display:flex;flex-direction:column;gap:3px;transition:background .15s;cursor:default;}}
.kpi-cell:hover{{background:var(--bg3);}}
.kpi-val{{font-family:var(--mono);font-size:24px;font-weight:700;line-height:1;}}
.kpi-lbl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}}
.kpi-cell.fire-kpi{{background:linear-gradient(180deg,rgba(255,87,34,.08) 0%,var(--bg2) 100%);}}
.kpi-cell.green-kpi{{background:linear-gradient(180deg,rgba(0,230,118,.06) 0%,var(--bg2) 100%);}}

/* ── TABLES ── */
.table-wrap{{overflow-x:auto;}}
table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px;}}
thead tr{{background:var(--bg3);border-bottom:1px solid var(--border2);}}
thead th{{padding:7px 12px;text-align:left;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:var(--dim);white-space:nowrap;}}
tbody tr{{border-bottom:1px solid rgba(24,34,53,.7);transition:background .1s;}}
tbody tr:hover{{background:rgba(41,182,246,.05);}}
tbody td{{padding:7px 12px;white-space:nowrap;}}
.num{{text-align:right;}}
.fw{{font-weight:600;}}
.sl{{color:var(--red)!important;}}
.tgt{{color:var(--green)!important;}}
.t1-col{{color:var(--yellow)!important;}}
.rr-col{{color:var(--cyan);}}
.pos{{color:var(--green);}}.neg{{color:var(--red);}}
.risk{{color:var(--orange);}}
.rank-num{{color:var(--dim);width:24px;font-size:10px;}}
.no-data{{text-align:center;color:var(--dim);padding:28px;font-size:11px;}}
.total-row td{{font-weight:600;color:var(--cyan);border-top:1px solid var(--border2);}}

/* ── FIRE ROWS ── */
.fire-row{{
  background:linear-gradient(90deg,rgba(255,87,34,.06) 0%,transparent 40%);
  border-left:3px solid rgba(255,87,34,.6)!important;
}}
.fire-row:hover{{background:linear-gradient(90deg,rgba(255,87,34,.12) 0%,rgba(255,87,34,.04) 100%)!important;}}
.fire-rules{{margin-top:3px;display:flex;gap:2px;flex-wrap:wrap;}}
.open-trade-row{{border-left:2px solid rgba(41,182,246,.2)!important;}}
.open-trade-row:hover{{background:rgba(41,182,246,.04)!important;}}

/* ── BADGES & TAGS ── */
.badge{{background:rgba(41,182,246,.12);color:var(--cyan);font-size:9px;padding:1px 7px;border-radius:10px;font-weight:600;font-family:var(--mono);}}
.badge-fire{{background:rgba(255,87,34,.15);color:var(--fire);}}
.badge-smc{{background:rgba(206,147,216,.12);color:var(--purple);}}
.badge-ts{{background:rgba(38,198,218,.1);color:var(--teal);border:1px solid rgba(38,198,218,.2);}}
.ticker-badge{{background:rgba(41,182,246,.07);color:var(--cyan);border:1px solid rgba(41,182,246,.15);border-radius:4px;padding:2px 7px;font-weight:600;font-size:12px;letter-spacing:.02em;}}
.ticker-badge.small{{font-size:11px;padding:1px 5px;}}
.tier{{border-radius:3px;padding:1px 6px;font-size:9px;font-weight:700;text-transform:uppercase;}}
.tier-t1{{background:rgba(0,230,118,.08);color:var(--green);border:1px solid rgba(0,230,118,.2);}}
.tier-t2{{background:rgba(255,193,7,.08);color:var(--yellow);border:1px solid rgba(255,193,7,.2);}}
.tier-t3{{background:rgba(61,90,122,.15);color:var(--muted);border:1px solid rgba(61,90,122,.25);}}
.rule-tag{{display:inline-block;border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700;margin-right:2px;}}
.tag-a{{background:rgba(255,112,67,.1);color:var(--orange);border:1px solid rgba(255,112,67,.2);}}
.tag-c{{background:rgba(41,182,246,.1);color:var(--cyan);border:1px solid rgba(41,182,246,.2);}}
.tag-d{{background:rgba(255,193,7,.1);color:var(--yellow);border:1px solid rgba(255,193,7,.2);}}
.tag-e{{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);}}
.tag-s{{background:rgba(206,147,216,.1);color:var(--purple);border:1px solid rgba(206,147,216,.2);}}
.tag-none{{color:var(--dim);font-size:10px;}}
.dir-badge-long{{background:rgba(0,230,118,.08);color:var(--green);border:1px solid rgba(0,230,118,.2);border-radius:3px;padding:2px 7px;font-size:10px;font-weight:700;font-family:var(--mono);}}
.dir-badge-short{{background:rgba(255,61,87,.08);color:var(--red);border:1px solid rgba(255,61,87,.2);border-radius:3px;padding:2px 7px;font-size:10px;font-weight:700;font-family:var(--mono);}}
.dir-long{{color:var(--green);font-weight:700;font-size:10px;}}
.dir-short{{color:var(--red);font-weight:700;font-size:10px;}}

/* ── SMC BADGES ── */
.smc-badge{{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:600;padding:1px 6px;border-radius:3px;margin-right:3px;}}
.smc-long{{color:var(--green);background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.2);}}
.smc-short{{color:var(--red);background:rgba(255,61,87,.08);border:1px solid rgba(255,61,87,.2);}}
.smc-neut{{color:var(--muted);background:rgba(61,90,122,.1);border:1px solid rgba(61,90,122,.2);}}
.wyck-tag{{font-size:9px;color:var(--purple);opacity:.7;font-family:var(--mono);margin-left:2px;}}

/* ── SCORE BAR ── */
.score-bar-wrap{{display:flex;align-items:center;gap:6px;min-width:90px;}}
.score-bar{{height:3px;background:linear-gradient(90deg,var(--fire),var(--yellow));border-radius:2px;flex:1;max-width:56px;}}
.wait-bar{{background:linear-gradient(90deg,var(--indigo),var(--cyan));}}
.l3-bar{{background:linear-gradient(90deg,var(--indigo),var(--purple));}}
.score-num{{font-size:11px;color:var(--text);font-weight:500;}}

/* ── ANCHOR STOP ── */
.sl-anchor{{color:var(--red);cursor:help;}}
.anchor-icon{{font-size:9px;opacity:.7;}}
.ts-t1{{color:var(--yellow);font-weight:600;}}
.dim-text{{color:var(--muted);font-size:10px;}}

/* ── DAYS BAR ── */
.days-wrap{{display:flex;align-items:center;gap:6px;}}
.days-bar{{height:3px;background:rgba(255,255,255,.06);border-radius:2px;width:48px;}}
.days-fill{{height:100%;border-radius:2px;transition:width .3s;}}
.days-num{{font-size:10px;font-weight:600;font-family:var(--mono);min-width:24px;}}

/* ── RULE PILLS ── */
.rule-pills-wrap{{display:flex;gap:8px;padding:14px 16px;flex-wrap:wrap;}}
.rule-pill{{
  display:flex;flex-direction:column;align-items:center;
  background:rgba(255,255,255,.02);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:12px 18px;min-width:80px;
  transition:border-color .2s,background .2s;
}}
.rule-pill:hover{{background:rgba(255,255,255,.04);border-color:var(--border2);}}
.rp-count{{font-family:var(--mono);font-size:30px;font-weight:700;line-height:1;}}
.rp-name{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:4px;}}

/* ── SMC STATS ── */
.smc-stats-wrap{{display:flex;gap:10px;padding:14px 16px;flex-wrap:wrap;}}
.smc-stat{{display:flex;flex-direction:column;align-items:center;background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px 16px;min-width:72px;}}
.sv{{font-family:var(--mono);font-size:24px;font-weight:700;}}
.slb{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:3px;}}
.wyck-list{{padding:10px 16px;display:flex;flex-wrap:wrap;gap:6px;}}
.wyck-item{{background:rgba(206,147,216,.07);border:1px solid rgba(206,147,216,.15);border-radius:3px;padding:2px 9px;font-family:var(--mono);font-size:10px;color:var(--purple);}}
.wyck-count{{color:var(--muted);font-size:9px;margin-left:3px;}}

/* ── STATUS ── */
.status-ok{{color:var(--green);}}.status-fail{{color:var(--red);}}
.status-win{{color:var(--green);font-weight:600;}}.status-loss{{color:var(--red);font-weight:600;}}.status-exp{{color:var(--muted);}}

/* ── CHART ── */
.chart-wrap{{padding:16px;height:200px;position:relative;}}

/* ── FOOTER ── */
.footer{{text-align:center;padding:14px;color:var(--dim);font-family:var(--mono);font-size:10px;border-top:1px solid var(--border);margin-top:8px;letter-spacing:.04em;}}

/* ── TS SECTION ── */
.ts-subheader{{padding:7px 16px 3px;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;}}

@keyframes slideUp{{from{{opacity:0;transform:translateY(8px);}}to{{opacity:1;transform:none;}}}}
.card:nth-child(2){{animation-delay:.04s;}}.card:nth-child(3){{animation-delay:.08s;}}
.card:nth-child(4){{animation-delay:.12s;}}.card:nth-child(5){{animation-delay:.16s;}}
</style>
</head>
<body>

<!-- ══ HEADER ══ -->
<div class="header">
  <div class="header-left">
    <h1>⚡ NSE DECISION ENGINE</h1>
    <p>{NOW} &nbsp;·&nbsp; {mode} &nbsp;·&nbsp; Last run: {last_run_time} ({last_run_elapsed})</p>
  </div>
  <div class="header-right">
    <span class="regime-pill regime-{regime.lower()}">{regime} · VIX {vix}</span>
    <div class="h-stat fire-kpi"><span class="hv c-fire">{fire_count}</span><span class="hl">🔥 Fire</span></div>
    <div class="h-stat"><span class="hv c-wait">{wait_count}</span><span class="hl">⏳ Wait</span></div>
    <div class="h-stat"><span class="hv c-kill">{kill_count}</span><span class="hl">Kill</span></div>
    <div class="h-stat"><span class="hv c-cyan">{candidates}</span><span class="hl">Universe</span></div>
    <div class="h-stat"><span class="hv c-green">{top_score:.3f}</span><span class="hl">Top Score</span></div>
  </div>
</div>

<div class="container">

<!-- ══ KPI STRIP ══ -->
<div class="kpi-strip">
  <div class="kpi-cell fire-kpi"><span class="kpi-val c-fire">{fire_count}</span><span class="kpi-lbl">🔥 Fire Today</span></div>
  <div class="kpi-cell"><span class="kpi-val c-wait">{wait_count}</span><span class="kpi-lbl">⏳ Watching</span></div>
  <div class="kpi-cell"><span class="kpi-val c-kill">{kill_count}</span><span class="kpi-lbl">💀 Killed</span></div>
  <div class="kpi-cell"><span class="kpi-val c-cyan">{candidates}</span><span class="kpi-lbl">🌐 Universe</span></div>
  <div class="kpi-cell"><span class="kpi-val" style="color:var(--yellow)">{vix}</span><span class="kpi-lbl">📊 VIX</span></div>
  <div class="kpi-cell green-kpi"><span class="kpi-val c-green">{top_score:.3f}</span><span class="kpi-lbl">🏆 Top Score</span></div>
  <div class="kpi-cell"><span class="kpi-val c-purple">{smc_bull_cnt}</span><span class="kpi-lbl">📐 SMC Bull</span></div>
  <div class="kpi-cell {'green-kpi' if pt_wr >= 38 else ''}"><span class="kpi-val {pt_wr_col}">{pt_wr}%</span><span class="kpi-lbl">🎯 Win Rate</span></div>
  <div class="kpi-cell"><span class="kpi-val {pt_pnl_col}">{pt_avg_pnl:+.2f}%</span><span class="kpi-lbl">💰 Avg PnL</span></div>
  <div class="kpi-cell"><span class="kpi-val c-cyan">{pt_open}</span><span class="kpi-lbl">📋 Open</span></div>
</div>

<!-- ══ FIRE SIGNALS — HERO ══ -->
<div class="card accent-fire">
  <div class="card-header">
    <h2>🔥 Fire Signals — Execute at Market Open</h2>
    <div class="ch-right">
      <span class="badge badge-fire">{fire_count} signals</span>
      <span class="badge" style="background:rgba(38,198,218,.1);color:var(--teal);border:1px solid rgba(38,198,218,.2)">⚓ = S/R anchored stop</span>
    </div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Ticker · Rules</th><th>Dir</th><th>Tier</th><th>Sector</th>
        <th class="num">Entry</th><th class="num">Stop ⚓</th><th class="num">T1 Partial</th><th class="num">Target T2</th>
        <th class="num">Qty</th><th class="num">R:R</th><th class="num">Risk ₹</th>
        <th>SMC</th><th>Score</th>
      </tr></thead>
      <tbody>{fire_rows}</tbody>
    </table>
  </div>
</div>

<!-- ══ SIGNALS GRID ══ -->
<div class="grid-2">
  <div class="card accent-cyan">
    <div class="card-header">
      <h2>📡 Layer 3 — Top Signals</h2>
      <div class="ch-right"><span class="badge">rules engine output</span></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>Ticker</th><th>Dir</th><th>Tier</th><th>Sector</th><th class="num">Price</th><th>Rules</th><th>Score</th></tr></thead>
        <tbody>{l3_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="card accent-yellow">
    <div class="card-header">
      <h2>👁 Watchlist — Top WAIT</h2>
      <div class="ch-right"><span class="badge">{wait_count} waiting</span></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>Ticker</th><th>Dir</th><th>Tier</th><th>Sector</th><th class="num">Price</th><th>Score</th></tr></thead>
        <tbody>{wait_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══ INTEL ROW ══ -->
<div class="grid-4">
  <div class="card">
    <div class="card-header"><h2>⚙️ Rules Fired Today</h2></div>
    <div class="rule-pills-wrap">
      <div class="rule-pill"><span class="rp-count" style="color:var(--orange)">{rule_a}</span><span class="rp-name">Rule A · Vol</span></div>
      <div class="rule-pill"><span class="rp-count" style="color:var(--cyan)">{rule_c}</span><span class="rp-name">Rule C · Pairs</span></div>
      <div class="rule-pill"><span class="rp-count" style="color:var(--yellow)">{rule_d}</span><span class="rp-name">Rule D · Event</span></div>
      <div class="rule-pill"><span class="rp-count" style="color:var(--green)">{rule_e}</span><span class="rp-name">Rule E · Mom</span></div>
    </div>
  </div>
  <div class="card accent-purple">
    <div class="card-header">
      <h2>📐 SMC Overview</h2>
      <div class="ch-right"><span class="badge badge-smc">{rule_s} SMC fires</span></div>
    </div>
    <div class="smc-stats-wrap">
      <div class="smc-stat"><span class="sv" style="color:var(--green)">{smc_bull_cnt}</span><span class="slb">▲ Bull</span></div>
      <div class="smc-stat"><span class="sv" style="color:var(--red)">{smc_bear_cnt}</span><span class="slb">▼ Bear</span></div>
      <div class="smc-stat"><span class="sv" style="color:var(--muted)">{smc_neut_cnt}</span><span class="slb">— Neutral</span></div>
    </div>
    <div class="wyck-list">{wyckoff_html}</div>
  </div>
  <div class="card">
    <div class="card-header"><h2>🏭 Sector Momentum (20d)</h2></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Rank</th><th>Sector</th><th class="num">Return</th></tr></thead>
        <tbody>{sector_rows if sector_rows else '<tr><td colspan="3" class="no-data">No sector data — run PatchSectors</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><h2>⏱ Pipeline Run</h2></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Layer</th><th>Status</th><th class="num">Time</th></tr></thead>
        <tbody>{timing_rows if timing_rows else '<tr><td colspan="3" class="no-data">Dashboard-only mode</td></tr>'}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══ 14-DAY HISTORY ══ -->
<div class="card">
  <div class="card-header">
    <h2>📈 14-Day Run History</h2>
    <div class="ch-right"><span class="badge">FIRE count · VIX overlay</span></div>
  </div>
  <div class="chart-wrap"><canvas id="histChart"></canvas></div>
</div>

<!-- ══ v2 PIPELINE PERFORMANCE ══ -->
{v2_html}

<!-- ══ TS ANALYSIS ══ -->
{ts_html}

<!-- ══ OPEN TRADES ══ -->
{open_trades_html}

<!-- ══ CLOSED TRADES ══ -->
{closed_trades_html}

</div><!-- /container -->

<div class="footer">
  ⚡ NSE Decision Engine · Layer 6 · {NOW} · Capital ₹10,00,000 · Max Risk/Trade 1%
</div>

<script>
const labels={runlog_labels},fireData={runlog_fire},vixData={runlog_vix};
if(labels.length>0){{
  const ctx=document.getElementById('histChart').getContext('2d');
  new Chart(ctx,{{
    type:'bar',
    data:{{labels,datasets:[
      {{label:'FIRE signals',data:fireData,backgroundColor:'rgba(255,87,34,.65)',borderColor:'rgba(255,87,34,.9)',borderWidth:1,borderRadius:3,yAxisID:'y'}},
      {{label:'VIX',data:vixData,type:'line',borderColor:'rgba(41,182,246,.85)',backgroundColor:'rgba(41,182,246,.06)',borderWidth:2,pointRadius:3,pointBackgroundColor:'rgba(41,182,246,1)',fill:true,yAxisID:'y1',tension:.4}}
    ]}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{color:'#5a7a9a',font:{{family:'JetBrains Mono',size:10}},padding:16}}}},
        tooltip:{{backgroundColor:'rgba(14,21,32,.95)',borderColor:'rgba(30,45,68,.8)',borderWidth:1,titleFont:{{family:'JetBrains Mono',size:10}},bodyFont:{{family:'JetBrains Mono',size:10}}}}
      }},
      scales:{{
        x:{{ticks:{{color:'#3d5a7a',font:{{family:'JetBrains Mono',size:9}},maxRotation:45}},grid:{{color:'rgba(24,34,53,.8)'}}}},
        y:{{position:'left',ticks:{{color:'#ff5722',font:{{family:'JetBrains Mono',size:10}}}},grid:{{color:'rgba(24,34,53,.8)'}},title:{{display:true,text:'FIRE',color:'#ff5722',font:{{size:10}}}}}},
        y1:{{position:'right',ticks:{{color:'#29b6f6',font:{{family:'JetBrains Mono',size:10}}}},grid:{{drawOnChartArea:false}},title:{{display:true,text:'VIX',color:'#29b6f6',font:{{size:10}}}}}}
      }}
    }}
  }});
}}
</script>
</body></html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# EOD REPORT + RUN LOG
# ─────────────────────────────────────────────────────────────────────────────

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
            lines.append(f"  {t:<14} {ti}  E:{e:>8.2f}  SL:{sl:>8.2f}  T:{tg:>8.2f}  Qty:{q}  Score:{s:.4f}")
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI REPORT
# ─────────────────────────────────────────────────────────────────────────────

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
    cp("cyan",  f"   NSE DECISION ENGINE  ⚡  {NOW}")
    cp("bold", "═" * W)

    print()
    regime_col = "green" if regime == "Bull" else "yellow" if regime == "Sideways" else "red"
    cp(regime_col, f"  Regime: {regime}   VIX: {vix}   Universe: {cands} stocks")
    print()
    print(f"  {C['fire']}🔥 FIRE: {fire_count}{C['reset']}   "
          f"{C['yellow']}⏳ WAIT: {wait_count}{C['reset']}   "
          f"{C['dim']}💀 KILL: {kill_count}{C['reset']}")

    # ── FIRE SIGNALS ─────────────────────────────────────────────────────────
    print()
    cp("bold", "═" * W)
    cp("orange", f"   FIRE SIGNALS  ({fire_count})")
    cp("bold", "═" * W)

    if len(fire_df) > 0:
        _open_tickers = set()
        if trades is not None and "ticker" in trades.columns and "status" in trades.columns:
            _open_tickers = set(trades[trades["status"] == "OPEN"]["ticker"].str.upper().tolist())

        _cooldown_tickers = set()
        if trades is not None and "ticker" in trades.columns and "status" in trades.columns:
            _losses = trades[trades["status"].str.upper() == "LOSS"].copy()
            if not _losses.empty and "exit_date" in _losses.columns:
                _losses["exit_date"] = _pd.to_datetime(_losses["exit_date"], errors="coerce", format="%Y-%m-%d")
                _cutoff = _pd.Timestamp.today().normalize() - _pd.Timedelta(days=3)
                _recent = _losses[_losses["exit_date"] >= _cutoff]
                _cooldown_tickers = set(_recent["ticker"].str.upper().tolist())

        _hdr = (f"  {'TICKER':<12} {'TIER':<4} {'SECTOR':<20} "
                f"{'ENTRY':>8} {'SL⚓':>8} {'T1':>9} {'T2':>9} {'QTY':>5} {'R:R':>5} {'SCORE':>7}")
        _div = "  " + chr(9472) * 76

        def _print_fire_row(row):
            ticker    = getattr(row, "ticker",      "?")
            tier      = getattr(row, "tier",         "?")
            entry     = float(getattr(row, "entry_price",  getattr(row, "price_close", 0)))
            sl        = float(getattr(row, "stop_loss",    0))
            tgt       = float(getattr(row, "target_price", 0))
            qty       = getattr(row, "qty",           getattr(row, "position_qty", "?"))
            score     = getattr(row, sc,              0.0)
            sec_n     = str(getattr(row, "sector",   ""))[:18]
            direction = str(getattr(row, "direction", "LONG")).upper()
            if direction == "BOTH": direction = "LONG"
            is_short  = direction == "SHORT"
            ts_t1_v   = getattr(row, "ts_t1", None)
            t1_s      = f"{float(ts_t1_v):>9.2f}" if ts_t1_v and str(ts_t1_v) not in ("None","nan") else f"{'—':>9}"
            if is_short:
                rr   = round((entry - tgt) / (sl - entry), 2) if (sl - entry) > 0 else 0
                risk = round((sl - entry) * float(qty), 0)    if qty != "?" else 0
                dir_tag = f"{C['red']}▼SHORT{C['reset']}"
            else:
                rr   = round((tgt - entry) / (entry - sl), 2) if (entry - sl) > 0 else 0
                risk = round((entry - sl) * float(qty), 0)    if qty != "?" else 0
                dir_tag = f"{C['green']}▲LONG {C['reset']}"
            print(f"  {C['cyan']}{ticker:<12}{C['reset']} {dir_tag} {tier:<4} {sec_n:<20} "
                  f"{C['white']}{entry:>7.2f}{C['reset']} "
                  f"{C['red']}{sl:>7.2f}{C['reset']} "
                  f"{C['yellow']}{t1_s}{C['reset']} "
                  f"{C['green']}{tgt:>8.2f}{C['reset']} "
                  f"{qty:>5} {C['yellow']}1:{rr:<4}{C['reset']} "
                  f"{C['orange']}{risk:>5.0f}{C['reset']}  {score:.3f}")

        fresh = [r for r in fire_df.itertuples()
                 if getattr(r, "ticker", "?").upper() not in _open_tickers
                 and getattr(r, "ticker", "?").upper() not in _cooldown_tickers]
        still = [r for r in fire_df.itertuples()
                 if getattr(r, "ticker", "?").upper() in _open_tickers]

        print()
        cp("green", f"  ✦ FRESH FIRE  ({len(fresh)} new signals)")
        print(_hdr)
        print(_div)
        if fresh:
            for row in fresh:
                _print_fire_row(row)
        else:
            cp("dim", "  No new entries today.")

        print()
        cp("yellow", f"  ↻ STILL FIRE  ({len(still)} continuing)")
        print(_hdr)
        print(_div)
        if still:
            for row in still:
                _print_fire_row(row)
        else:
            cp("dim", "  No continuing signals.")

    else:
        cp("dim", "  No FIRE signals today.")

    # ── WATCHLIST ─────────────────────────────────────────────────────────────
    print()
    cp("bold", "═" * W)
    cp("yellow", f"    WATCHLIST  Top 10 WAIT")
    cp("bold", "═" * W)
    if len(wait_df) > 0 and sc in wait_df.columns:
        print(f"  {'#':<3} {'TICKER':<12} {'TIER':<4} {'SECTOR':<20} {'PRICE':>8} {'SCORE':>7}")
        print("  " + "─" * 56)
        for i, row in enumerate(wait_df.nlargest(10, sc).itertuples(), 1):
            ticker = getattr(row, "ticker",     "?")
            tier   = getattr(row, "tier",        "?")
            price  = getattr(row, "entry_price", getattr(row, "price_close", 0))
            score  = getattr(row, sc,             0.0)
            sec_n  = str(getattr(row, "sector",  ""))[:18]
            print(f"  {i:<3} {C['cyan']}{ticker:<12}{C['reset']} {tier:<4} {sec_n:<20} {price:>7.2f}  {score:.3f}")
    else:
        cp("dim", "  No WAIT signals.")

    # ── SHORT SIGNALS ─────────────────────────────────────────────────────────
    print()
    cp("bold", "═" * W)
    cp("red",  "   SHORT SIGNALS  INTRADAY + F&O")
    cp("bold", "═" * W)

    if l3 is not None and "intraday_short_eligible" in l3.columns:
        shorts = l3[l3["intraday_short_eligible"] == True].copy()
        if "short_score" in shorts.columns:
            shorts = shorts.sort_values("short_score", ascending=False)
        if len(shorts) > 0:
            fno_sh = shorts[shorts["fno_short_eligible"] == True]  if "fno_short_eligible" in shorts.columns else shorts.iloc[:0]
            int_sh = shorts[shorts["fno_short_eligible"] == False] if "fno_short_eligible" in shorts.columns else shorts
            print(f"  {C['red']}Short candidates: {len(shorts)}   "
                  f"F&O overnight: {len(fno_sh)}   "
                  f"Intraday only: {len(int_sh)}{C['reset']}")
            print()
            print(f"  {C['cyan']}  #   {'TICKER':<12} {'TIER':<6} {'SECTOR':<20} {'PRICE':>8}  {'SCORE':>6}  RULES  TYPE{C['reset']}")
            print("  " + chr(9472) * 74)
            for i, row in enumerate(shorts.head(15).itertuples(), 1):
                tk     = getattr(row, "ticker",               "?")
                tier   = getattr(row, "tier",                  "?")
                price  = getattr(row, "price_close",            0)
                score  = getattr(row, "short_score",           0.0)
                sec_n  = str(getattr(row, "sector",            ""))[:18]
                fno_ok = getattr(row, "fno_short_eligible",  False)
                ra_s   = "A" if getattr(row, "rule_a_short_fires", False) else "-"
                rc_s   = "C" if getattr(row, "rule_c_short_fires", False) else "-"
                re_s   = "E" if getattr(row, "rule_e_short_fires", False) else "-"
                stype  = (f"{C['orange']}F&O+INTRADAY{C['reset']}"
                          if fno_ok else f"{C['yellow']}INTRADAY ONLY{C['reset']}")
                print(f"  {i:<3} {C['cyan']}{tk:<12}{C['reset']} {tier:<6} {sec_n:<20} "
                      f"{price:>8.2f}  {C['red']}{score:.3f}{C['reset']}   "
                      f"{ra_s}{rc_s}{re_s}    {stype}")
        else:
            cp("dim", "  No short signals today.")
    else:
        cp("dim", "  Run Layer 3 to see short signals.")

    # ── SECTOR MOMENTUM ───────────────────────────────────────────────────────
    print()
    cp("bold", "═" * W)
    cp("dim",  "   SECTOR MOMENTUM (20d)")
    cp("bold", "═" * W)
    if sector is not None:
        valid_sec = sector[
            (sector["Sector"].str.strip() != "") &
            (sector["Sector"].str.strip().str.lower() != "unknown")
        ].sort_values("Rank")
        if len(valid_sec) == 0:
            cp("dim", "  No sector data — run patch_sectors.py then Layer1-Daily")
        else:
            for row in valid_sec.itertuples():
                ret   = float(row.Rolling20dReturn) * 100
                rank  = int(row.Rank)
                medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"  {rank}."
                col   = "green" if ret >= 0 else "red"
                print(f"  {medal}  {row.Sector:<28}  {C[col]}{ret:+.2f}%{C['reset']}")

    # ── RULE ENGINE ───────────────────────────────────────────────────────────
    print()
    cp("bold", "═" * W)
    cp("dim",  "   RULE ENGINE")
    cp("bold", "═" * W)
    if l3 is not None:
        ra = int(l3["rule_a_fires"].sum())   if "rule_a_fires"   in l3.columns else 0
        rc = int(l3["rule_c_fires"].sum())   if "rule_c_fires"   in l3.columns else 0
        rd = int(l3["rule_d_fires"].sum())   if "rule_d_fires"   in l3.columns else 0
        re = int(l3["rule_e_fires"].sum())   if "rule_e_fires"   in l3.columns else 0
        rs = int(l3["smc_rule_fires"].sum()) if "smc_rule_fires" in l3.columns else 0
        smc_b = int((l3["smc_signal"] ==  1).sum()) if "smc_signal" in l3.columns else 0
        smc_r = int((l3["smc_signal"] == -1).sum()) if "smc_signal" in l3.columns else 0
        print(f"  {C['orange']}Rule A (Vol)     : {ra}{C['reset']}")
        print(f"  {C['cyan']}Rule C (Pairs)   : {rc}{C['reset']}")
        print(f"  {C['yellow']}Rule D (Event)   : {rd}{C['reset']}")
        print(f"  {C['green']}Rule E (Momentum): {re}{C['reset']}")
        print(f"  \033[38;5;183mRule S (SMC)     : {rs}  [{C['green']}▲{smc_b}{C['reset']}\033[38;5;183m / {C['red']}▼{smc_r}{C['reset']}\033[38;5;183m]{C['reset']}")

    # ── TS ANALYSIS ───────────────────────────────────────────────────────────
    ts_data = load("ts")
    if ts_data is not None and len(ts_data) > 0:
        print()
        cp("bold", "═" * W)
        n_strong_buy   = int((ts_data.get("entry_signal","") == "STRONG_BUY").sum())
        n_strong_short = int((ts_data.get("entry_signal","") == "STRONG_SHORT").sum())
        n_retest       = int((ts_data.get("entry_type","") == "RETEST").sum())
        n_breakout     = int((ts_data.get("entry_type","") == "BREAKOUT").sum())
        cp("dim", f"   TS ANALYSIS  ({len(ts_data)} stocks)  "
                  f"🟢🟢{n_strong_buy} STRONG_BUY  🔴🔴{n_strong_short} STRONG_SHORT  "
                  f"🟡{n_retest} RETEST  ⚡{n_breakout} BREAKOUT")
        cp("bold", "═" * W)

        # Top LONG
        ts_long = ts_data[
            (ts_data.get("direction","")=="LONG") &
            (ts_data.get("entry_signal","").isin(["STRONG_BUY","BUY"])) &
            (ts_data.get("rr_valid",False)==True)
        ].nlargest(8, "setup_score")

        if len(ts_long):
            cp("green", f"  ▲ TOP LONG SETUPS")
            print(f"  {C['cyan']}  {'TICKER':<13} {'TIER':<4} {'TYPE':<13} {'KEY_LVL':>9} "
                  f"{'STOP':>8} {'T1':>8} {'T2':>8} {'RR':>5} PATTERNS{C['reset']}")
            print("  " + "─" * 82)
            for _, r in ts_long.iterrows():
                print(f"    {C['cyan']}{str(r['ticker']):<13}{C['reset']} {str(r.get('tier','?')):<4} "
                      f"{str(r.get('entry_type','')):<13} {float(r.get('key_level',0)):>9.2f} "
                      f"{C['red']}{float(r.get('stop',0)):>8.2f}{C['reset']} "
                      f"{C['yellow']}{float(r.get('t1',0)):>8.2f}{C['reset']} "
                      f"{C['green']}{float(r.get('t2',0)):>8.2f}{C['reset']} "
                      f"{float(r.get('rr',0)):>5.2f}  {str(r.get('patterns',''))[:28]}")

        # Top SHORT
        ts_short = ts_data[
            (ts_data.get("direction","")=="SHORT") &
            (ts_data.get("entry_signal","").isin(["STRONG_SHORT","SHORT"])) &
            (ts_data.get("rr_valid",False)==True)
        ].nlargest(8, "setup_score")

        if len(ts_short):
            print()
            cp("red", f"  ▼ TOP SHORT SETUPS")
            print(f"  {C['cyan']}  {'TICKER':<13} {'TIER':<4} {'TYPE':<13} {'KEY_LVL':>9} "
                  f"{'STOP':>8} {'T1':>8} {'T2':>8} {'RR':>5} {'FnO?':<7} PATTERNS{C['reset']}")
            print("  " + "─" * 89)
            for _, r in ts_short.iterrows():
                fno_tag = "🌙FnO" if r.get("fno_eligible") else "📅Intra"
                print(f"    {C['cyan']}{str(r['ticker']):<13}{C['reset']} {str(r.get('tier','?')):<4} "
                      f"{str(r.get('entry_type','')):<13} {float(r.get('key_level',0)):>9.2f} "
                      f"{C['red']}{float(r.get('stop',0)):>8.2f}{C['reset']} "
                      f"{C['yellow']}{float(r.get('t1',0)):>8.2f}{C['reset']} "
                      f"{C['green']}{float(r.get('t2',0)):>8.2f}{C['reset']} "
                      f"{float(r.get('rr',0)):>5.2f}  {fno_tag:<7} {str(r.get('patterns',''))[:28]}")

    # ── PAPER TRADES (OPEN) ───────────────────────────────────────────────────
    print()
    cp("bold", "═" * W)
    if trades is not None and "status" in trades.columns:
        _open_n   = int((trades["status"] == "OPEN").sum())
        _win_n    = int((trades["status"] == "WIN").sum())
        _loss_n   = int((trades["status"] == "LOSS").sum())
        _exp_n    = int((trades["status"] == "EXPIRED").sum())
        _total_n  = len(trades)
        _wr       = round(_win_n / (_win_n + _loss_n) * 100, 1) if (_win_n + _loss_n) > 0 else 0
        cp("dim", f"   PAPER TRADES    {_open_n} open  |  {_win_n}W / {_loss_n}L / {_exp_n}EXP  |  WinRate:{_wr}%  |  {_total_n} total")
    else:
        cp("dim",  "   PAPER TRADES")
    cp("bold", "═" * W)

    if trades is not None and "status" in trades.columns:
        open_t = trades[trades["status"] == "OPEN"]
        if len(open_t) > 0:
            try:
                import yfinance as _yf
                _tickers = open_t["ticker"].tolist()
                _cmp_map = {}
                _high_map = {}
                _low_map = {}
                print(f"  {C['dim']}fetching live prices for {len(_tickers)} trades...{C['reset']}", flush=True)
                for _tk in _tickers:
                    try:
                        _d = _yf.download(_tk + ".NS", period="2d", progress=False, auto_adjust=True)
                        if not _d.empty:
                            _cmp_map[_tk]  = float(_d["Close"].dropna().iloc[-1].item())
                            _high_map[_tk] = float(_d["High"].dropna().iloc[-1].item())
                            _low_map[_tk]  = float(_d["Low"].dropna().iloc[-1].item())
                    except Exception:
                        pass
            except ImportError:
                _cmp_map = {}; _high_map = {}; _low_map = {}

            print(f"  {C['cyan']}{'TICKER':<13} {'ENTRY':>9} {'CMP':>9} {'SL':>9} {'TARGET':>9}  {'PROGRESS %':^20}  {'PNL%':>6}  {'H->T':>6}  {'L->SL':>6}  DAYS{C['reset']}")
            print("  " + chr(9472) * 104)

            # ── collector for avg PnL ──────────────────────────────────────
            _open_pnl_pcts = []

            for row in open_t.itertuples():
                ticker   = getattr(row, "ticker",       "?")
                entry    = float(getattr(row, "entry_price",  0))
                sl       = float(getattr(row, "stop_price",   0))
                tgt      = float(getattr(row, "target_price", 0))
                days_raw = getattr(row, "days_held", "")
                days     = int(days_raw) if str(days_raw).strip() not in ("", "nan", "NaN", "?") else 0

                cmp  = _cmp_map.get(ticker)
                high = _high_map.get(ticker)
                low  = _low_map.get(ticker)

                _direction = str(getattr(row, "direction", "LONG")).upper()
                _is_short  = _direction == "SHORT"
                if cmp is not None:
                    if _is_short:
                        # SHORT: progress goes from stop(top) to target(bottom)
                        range_total = sl - tgt
                        progress    = max(0, min(100, (sl - cmp) / range_total * 100)) if range_total > 0 else 0
                        pnl_pct     = round((entry - cmp) / entry * 100, 2)
                        sl_hit      = high >= sl  if high else False
                        tgt_hit     = low  <= tgt if low  else False
                        h_to_tgt    = round((high - tgt) / tgt * 100, 2) if high else 0
                        l_to_sl     = round((sl - low)   / sl  * 100, 2) if low  else 0
                    else:
                        range_total = tgt - sl
                        progress    = max(0, min(100, (cmp - sl) / range_total * 100)) if range_total > 0 else 0
                        pnl_pct     = round((cmp - entry) / entry * 100, 2)
                        sl_hit      = low  <= sl  if low  else False
                        tgt_hit     = high >= tgt if high else False
                        h_to_tgt    = round((tgt - high) / tgt * 100, 2) if high else 0
                        l_to_sl     = round((low - sl)   / sl  * 100, 2) if low  else 0
                    filled      = int(progress / 6.25)
                    bar         = chr(9608) * filled + chr(9617) * (16 - filled)
                    _open_pnl_pcts.append(pnl_pct)
                    bar_col     = C["green"] if progress > 60 else C["yellow"] if progress > 30 else C["red"]
                    pnl_col     = C["green"] if pnl_pct >= 0 else C["red"]
                    h_col       = C["green"] if h_to_tgt > 2 else C["yellow"]
                    l_col       = C["green"] if l_to_sl  > 2 else C["red"]
                    dir_prefix  = f"{C['red']}▼{C['reset']}" if _is_short else f"{C['green']}▲{C['reset']}"
                    alert       = f" {C['red']}⚠ SL DANGER{C['reset']}" if sl_hit else f" {C['green']}✔ TGT HIT{C['reset']}" if tgt_hit else ""
                    print(f"  {C['cyan']}{ticker:<13}{C['reset']}{dir_prefix}"
                          f"{entry:>8.2f}  "
                          f" {C['cyan']}{cmp:>9.2f}{C['reset']}"
                          f" {C['red']}{sl:>9.2f}{C['reset']}"
                          f" {C['green']}{tgt:>9.2f}{C['reset']}"
                          f"  {bar_col}[{bar}]{progress:>3.0f}%{C['reset']}"
                          f"  {pnl_col}{pnl_pct:>+6.2f}%{C['reset']}"
                          f"  {h_col}{h_to_tgt:>+6.2f}%{C['reset']}"
                          f"  {l_col}{l_to_sl:>+6.2f}%{C['reset']}"
                          f"  {days:>3}d{alert}")
                else:
                    print(f"  {C['cyan']}{ticker:<13}{C['reset']}"
                          f"{entry:>8.2f}  "
                          f" {C['red']}{sl:>9.2f}{C['reset']}"
                          f" {C['green']}{tgt:>9.2f}{C['reset']}"
                          f"{C['dim']}[no price]{C['reset']}  {days:>3}d")

            # ── print avg after all fetches ────────────────────────────────
            if _open_pnl_pcts:
                _avg  = sum(_open_pnl_pcts) / len(_open_pnl_pcts)
                _sign = "+" if _avg >= 0 else ""
                _col  = "green" if _avg >= 0 else "red"
                print()
                cp(_col, f"  ── Avg open-trade PnL : {_sign}{_avg:.2f}%  ({len(_open_pnl_pcts)} positions)")

        else:
            cp("dim", "  No open trades.")

    # ── CLOSED TRADES ─────────────────────────────────────────────────────────
    if trades is not None and "status" in trades.columns:
        closed_t = trades[trades["status"].isin(["WIN", "LOSS", "EXPIRED"])].copy()
        if len(closed_t) > 0:
            print()
            cp("bold", "═" * W)
            cp("dim",  "   CLOSED TRADES  (what-if: no SL scenario)")
            cp("bold", "═" * W)

            try:
                import yfinance as _yf2
                _wtickers = closed_t["ticker"].tolist()
                _wcmp = {}
                print(f"  {C['dim']}fetching current prices for what-if analysis...{C['reset']}", flush=True)
                for _tk in _wtickers:
                    try:
                        _d2 = _yf2.download(_tk + ".NS", period="2d", progress=False, auto_adjust=True)
                        if not _d2.empty:
                            _wcmp[_tk] = float(_d2["Close"].dropna().iloc[-1].item())
                    except Exception:
                        pass
            except ImportError:
                _wcmp = {}

            _h = f"  {'TICKER':<12} {'DATE':<11} {'STATUS':<7} {'ENTRY':>8} {'EXIT':>8} {'PnL%':>7} {'DAYS':>5}  {'WHATIF_CMP':>10}  {'WHATIF_PnL%':>11}  SAVED?"
            print(f"{C['cyan']}{_h}{C['reset']}")
            print("  " + chr(9472) * 110)

            closed_t["pnl_pct"]   = pd.to_numeric(closed_t["pnl_pct"],   errors="coerce").fillna(0)
            closed_t["days_held"] = pd.to_numeric(closed_t["days_held"],  errors="coerce").fillna(0)

            for crow in closed_t.sort_values("exit_date", ascending=False).itertuples():
                _tk     = getattr(crow, "ticker", "?")
                _entry  = float(getattr(crow, "entry_price", 0))
                _exit   = float(getattr(crow, "exit_price",  0)) if str(getattr(crow, "exit_price", "")).strip() not in ("", "nan") else 0
                _pnl    = float(getattr(crow, "pnl_pct",    0))
                _days   = int(getattr(crow, "days_held",    0))
                _status = getattr(crow, "status", "?")
                _edate  = str(getattr(crow, "exit_date", ""))[:10]

                _wcmp_val = _wcmp.get(_tk)
                if _wcmp_val and _entry > 0:
                    _wi_pnl = round((_wcmp_val - _entry) / _entry * 100, 2)
                    _wi_col = C["green"] if _wi_pnl >= 0 else C["red"]
                    _wi_str = f"{_wcmp_val:>10.2f}  {_wi_col}{_wi_pnl:>+10.2f}%{C['reset']}"
                    if _status == "LOSS" and _wi_pnl < _pnl:
                        _saved = f"  {C['green']}SL SAVED {abs(_wi_pnl - _pnl):.1f}%{C['reset']}"
                    elif _status == "LOSS" and _wi_pnl >= 0:
                        _saved = f"  {C['yellow']}recovered (no SL){C['reset']}"
                    elif _status == "WIN":
                        _saved = f"  {C['dim']}won anyway{C['reset']}"
                    else:
                        _saved = ""
                else:
                    _wi_str = f"  {C['dim']}no price{C['reset']}"
                    _saved  = ""

                _st_col  = C["green"] if _status == "WIN" else C["red"] if _status == "LOSS" else C["dim"]
                _pnl_col = C["green"] if _pnl >= 0 else C["red"]
                print(f"  {C['cyan']}{_tk:<12}{C['reset']} "
                      f"{_edate:<11} "
                      f"{_st_col}{_status:<7}{C['reset']} "
                      f"{_entry:>8.2f} "
                      f"{_exit:>8.2f} "
                      f"{_pnl_col}{_pnl:>+6.2f}%{C['reset']} "
                      f"{_days:>5}d  "
                      f"{_wi_str}{_saved}")
        else:
            print()
            cp("dim", "  No closed trades yet.")
    else:
        cp("dim", "  No paper trades file found.")

    # ── PIPELINE TIMING ───────────────────────────────────────────────────────
    if results:
        print()
        cp("bold", "═" * W)
        cp("dim",  "    PIPELINE TIMING")
        cp("bold", "═" * W)
        for name, r in results.items():
            col = "green" if r["ok"] else "red"
            print(f"  {C[col]}{'OK' if r['ok'] else 'FAIL'}{C['reset']}  {name:<16}  {r['elapsed']:.1f}s")
        print(f"  {'':6} {'TOTAL':<16}  {total_elapsed:.1f}s")

    print()
    cp("bold", "═" * W)
    cp("green", f"    DONE  |  {total_elapsed:.1f}s  |  {NOW}")
    cp("bold", "═" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(selected_layers=None):
    if selected_layers is None:
        selected_layers = select_layers()

    print()
    cp("bold", "─" * 65)
    cp("cyan",  f"  Running {len(selected_layers)} layer(s)  —  {NOW}")
    cp("bold", "─" * 65)
    print()

    total    = len(selected_layers)
    results  = {}
    t_start  = time.time()

    _BUDGET  = 5400  # 90 min — covers weekend full rebuild + TS scan
    progress = PipelineProgress([name for name, _ in selected_layers],
                                 step_total=total, budget_seconds=_BUDGET)

    progress.start()
    try:
        for i, (name, script) in enumerate(selected_layers, 1):
            ok, elapsed, err = run_layer(name, script, i, total, progress=progress)
            results[name] = {"ok": ok, "elapsed": elapsed, "error": err}

            if not ok and name in ("Layer2", "Layer3", "Layer4"):
                print()
                cp("red", f"\n  ABORT — {name} failed. Fix error and rerun.")
                break
    finally:
        progress.stop()

    total_elapsed = time.time() - t_start
    print()  # close out the single progress line before the report below
    print()

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

    try:
        subprocess.run(
            [PYTHON, str(BASE / "f1_paper_tracker.py"), "update"],
            cwd=str(BASE),
            timeout=180,
        )
        log.info("Paper tracker: open trades checked against SL/target/MAX_HOLD.")
    except Exception as _e:
        log.warning(f"Paper tracker update failed: {_e}")

    print_cli_report(results, total_elapsed)


def run_dash_only():
    print()
    cp("bold", "─" * 65)
    cp("cyan",  f"  Dashboard-only mode  —  {NOW}")
    cp("bold", "─" * 65)
    print_cli_report({}, 0)


def run_monitor():
    print()
    cp("bold", "─" * 65)
    cp("cyan",  f"  Monitor mode — open trades  —  {NOW}")
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
            print(f"  {ticker:<14}  fetch failed")
            continue

        pnl_pct    = (cmp - entry) / entry * 100
        to_tgt_pct = (tgt - cmp)  / cmp    * 100
        status     = f"{C['green']}✔ OK{C['reset']}"  if cmp > entry else f"{C['red']}↓ DOWN{C['reset']}"
        if abs(pnl_pct) < 0.5: status = f"{C['yellow']}— FLAT{C['reset']}"
        if cmp >= tgt: status = f"{C['green']}✔ TARGET{C['reset']}"
        if cmp <= sl:  status = f"{C['red']}✗ SL HIT{C['reset']}"

        print(f"  {ticker:<14} {entry:>8.2f} {sl:>8.2f} {tgt:>8.2f} {cmp:>8.2f} {pnl_pct:>+6.1f}% {to_tgt_pct:>+6.1f}%  {status}")

    print()
    cp("dim", f"  {NOW}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dash",    action="store_true", help="Dashboard only — no pipeline")
    parser.add_argument("--monitor", action="store_true", help="Live open-trade monitor")
    parser.add_argument("--all",     action="store_true", help="Run all layers (auto weekday/weekend)")
    parser.add_argument("--weekly",  action="store_true", help="Run weekly scripts only (Layer1-Val + TS)")
    parser.add_argument("--weekend", action="store_true", help="Force weekend pipeline (heavy rebuild + TS scan)")
    parser.add_argument("--select",  action="store_true", help="Interactive layer selector")
    args = parser.parse_args()

    # ── Pipeline definitions ──────────────────────────────────────────────────
    # WEEKDAY: fast incremental — runs every trading day (~12-15 min)
    WEEKDAY_PIPELINE = [
        "Layer0-NSE",    # fetch earnings, announcements, corporate actions
        "Layer0-FnO",    # fetch F&O chain, OI, PCR, IV
        "PatchSectors",  # refresh sector classifications (fast — uses cache)
        "Layer1-Daily",  # incremental CUSUM + sector momentum (~60s)
        "Layer2",        # daily filter: liquidity, regime, R2 gate
        "Layer3",        # rules engine: A/C/D/E signals
        "Layer4",        # portfolio: FIRE/WAIT/KILL + sizing
        "Layer-TS",      # technical structure scan — S/R zones, stops/targets (~10 min)
    ]

    # WEEKEND: full rebuild + structural analysis (~60-75 min total)
    WEEKEND_PIPELINE = [
        "Layer0-NSE",    # fetch earnings, announcements
        "Layer0-FnO",    # fetch F&O data
        "Layer1-Heavy",  # GARCH, Hurst, pairs, clusters (~43 min)
        "PatchSectors",  # refresh all sector classifications
        "Layer1-Daily",  # incremental update after heavy compute
        "Layer2",        # daily filter
        "Layer3",        # rules engine
        "Layer4",        # portfolio
        "Layer1-Val",    # walk-forward signal validation (weekly IC check)
        "Layer-TS",      # technical structure scan — S/R, entry/exit (~10 min)
    ]

    def _build_pipeline(names):
        """Resolve pipeline names to (name, path) pairs, skip missing scripts."""
        all_scripts = {**SCRIPTS, **SCRIPTS_WEEKLY}
        selected = []
        for name in names:
            path = all_scripts.get(name)
            if path and path.exists():
                selected.append((name, path))
            elif path:
                cp("yellow", f"    {name}: not found — skipping")
        return selected

    def _auto_pipeline(force_weekend=False):
        today      = date.today()
        is_weekend = force_weekend or today.weekday() >= 5   # 5=Sat, 6=Sun
        day_name   = today.strftime("%A")

        if is_weekend:
            cp("bold",  "\n  ═══════════════════════════════════════════════")
            cp("cyan",  f"  ⚡  WEEKEND RUN — {day_name}  |  Full rebuild + TS scan")
            cp("bold",  "  ═══════════════════════════════════════════════")
            cp("dim",   "  Pipeline: NSE data → FnO → Layer1-Heavy → PatchSectors")
            cp("dim",   "            → Layer1-Daily → L2 → L3 → L4")
            cp("dim",   "            → Layer1-Val (IC backtest) → Layer-TS (S/R scan)")
            cp("dim",   "  Estimated runtime: ~60-75 minutes")
            return _build_pipeline(WEEKEND_PIPELINE)
        else:
            cp("bold",  "\n  ═══════════════════════════════════════════════")
            cp("cyan",  f"  ⚡  WEEKDAY RUN — {day_name}  |  Daily incremental")
            cp("bold",  "  ═══════════════════════════════════════════════")
            cp("dim",   "  Pipeline: NSE data → FnO → PatchSectors → Layer1-Daily")
            cp("dim",   "            → L2 → L3 → L4 → Layer-TS (S/R scan)")
            cp("dim",   "  Estimated runtime: ~22-25 minutes")
            return _build_pipeline(WEEKDAY_PIPELINE)

    # ── Route to correct mode ─────────────────────────────────────────────────
    if args.monitor:
        run_monitor()
    elif args.dash:
        run_dash_only()
    elif args.weekly:
        # Manual: run only the weekly maintenance scripts (validation + TS)
        cp("cyan", "\n  Weekly maintenance — Layer1-Val + Layer-TS")
        run_full_pipeline(selected_layers=_build_pipeline(["Layer1-Val", "Layer-TS"]))
    elif args.weekend:
        # Manual: force full weekend pipeline even on a weekday
        cp("yellow", "\n  Forcing weekend pipeline on non-weekend day")
        run_full_pipeline(selected_layers=_auto_pipeline(force_weekend=True))
    elif args.all or not any([args.monitor, args.dash, args.weekly, args.weekend, args.select]):
        # Default: smart auto-detect weekday/weekend — no prompting
        run_full_pipeline(selected_layers=_auto_pipeline())
    elif args.select:
        # Manual layer picker
        run_full_pipeline()