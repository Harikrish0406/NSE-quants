# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS -- live daily run
==============================

The replica's real daily runner (the old orchestrator.py had a placeholder
data step and a no-op live-scoring step). Chains, as subprocesses:

  1. refresh_cache.py    -- incremental yfinance pull -> data/raw/*.parquet
  2. live_scoring.py     -- validated swing signals + ML ensemble refit
                            -> latest_signals.parquet
  3. layer_combiner.py   -- weighted composite + FIRE/WAIT/KILL
                            -> combiner_output.parquet
  4. paper_tracker.py    -- log new FIRE trades, update opens, print report
                            -> paper_trades.csv  (this folder only)

Progress shows the same rich block the parent pipeline uses (see
reference-rich-progress-ui-pattern): overall weighted bar + a per-step
mini-bar. Set CQ_FORCE_RICH=1 to force it when stdout isn't a detected TTY.

Run:  python live_run.py
      python live_run.py --skip-refresh     # reuse existing cache
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("Cache refresh", [PY, "refresh_cache.py"],      1500),
    ("Live scoring",  [PY, "live_scoring.py"],       1800),
    ("Combiner",      [PY, "layer_combiner.py"],       30),
    ("Paper tracker", [PY, "paper_tracker.py", "daily"], 60),
]

# ── compact rich progress (same style as layer6_dashboard.PipelineProgress) ──
from rich.console import Console as _RC, Group as _RG
from rich.live import Live as _RL
from rich.table import Table as _RT
from rich.text import Text as _RTx
from rich.spinner import Spinner as _RSp

_f = os.environ.get("CQ_FORCE_RICH")
_con = _RC(force_terminal=(True if _f == "1" else False if _f == "0" else None))


def _dur(s):
    s = int(max(0, s))
    if s < 60:  return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def _bar(frac, color, w=22):
    frac = max(0.0, min(frac, 1.0)); n = int(round(w * frac))
    return f"[{color}]" + "━" * n + "[/][grey30]" + "━" * (w - n) + "[/]"


class Progress:
    def __init__(self, steps):
        self.names = [n for n, _, _ in steps]
        self.est   = {n: e for n, _, e in steps}
        self.state = {n: ["pending", 0.0, ""] for n in self.names}
        self.t0    = time.time()
        self.done  = 0.0
        self.left  = list(self.names)
        self.tty   = _con.is_terminal
        self.live  = None
        self.spin  = _RSp("dots", style="cyan")

    def start(self):
        if self.tty:
            self.live = _RL(self._block(), console=_con, refresh_per_second=8)
            self.live.start()

    def stop(self):
        if self.live:
            self.live.update(self._block()); self.live.stop(); self.live = None

    def _pct(self, cur=None, el=0.0):
        rem = sum(self.est[n] for n in self.left)
        tot = self.done + rem
        c = min(el / self.est[cur], 0.99) * self.est[cur] if cur in self.left and self.est.get(cur) else 0
        return (self.done + c) / tot if tot else 0.0

    def _eta(self, pct):
        el = time.time() - self.t0
        return max(0.0, el / pct - el) if pct > 0.03 else 0.0

    def _block(self, pct=None):
        el = time.time() - self.t0
        if pct is None: pct = self._pct()
        head = _RTx.from_markup(
            f"  [bold]CLAUDE QUANTS -- live[/bold]  {_bar(pct, 'cyan', 28)}"
            f"  [bold]{pct*100:4.0f}%[/bold]   {_dur(el)} • eta {_dur(self._eta(pct))}")
        g = _RT.grid(padding=(0, 1))
        g.add_column(width=2, justify="center"); g.add_column(width=14)
        g.add_column(width=22); g.add_column(width=6, justify="right"); g.add_column()
        for n in self.names:
            stt, e, info = self.state[n]; est = self.est[n]
            if stt == "done":
                g.add_row("[green]✔[/]", f"[green]{n}[/]", _bar(1, "green"), "[green]100%[/]",
                          f"[grey62]{_dur(e)}[/]")
            elif stt == "fail":
                g.add_row("[red]✗[/]", f"[red]{n}[/]", _bar(min(e / est, 1) if est else 1, "red"),
                          "", f"[red]failed after {_dur(e)}[/]")
            elif stt == "running":
                fr = min(e / est, 0.99) if est else 0.0
                t = f"[cyan]{_dur(e)}[/] / ~{_dur(est)}" + (f"   [grey62]{info}[/]" if info else "")
                g.add_row(self.spin, f"[cyan]{n}[/]", _bar(fr, "cyan"), f"[cyan]{fr*100:3.0f}%[/]",
                          _RTx.from_markup(t))
            else:
                g.add_row(" ", f"[grey42]{n}[/]", _bar(0, "grey30"), "", "[grey42]pending[/]")
        return _RG(head, _RTx(""), g)

    def upd(self, name, el, status="running", info=""):
        s = self.state[name]
        s[0] = {"ok": "done", "fail": "fail"}.get(status, "running")
        s[1] = el
        if info: s[2] = info
        if self.tty and self.live:
            self.live.update(self._block(self._pct(name, el)))
        elif not self.tty and status in ("ok", "fail"):
            print(f"  [{self.names.index(name)+1}/{len(self.names)}] {name:<14} "
                  f"{status.upper():<4} {_dur(el)}", flush=True)

    def finish(self, name, el):
        self.done += el
        if name in self.left: self.left.remove(name)
        self.state[name][0] = "done"; self.state[name][1] = el


_INFO = __import__("re").compile(
    r"(\d+)\s*(?:rows|tickers|pairs|trades|FIRE|signals)", __import__("re").I)


def run_step(name, cmd, prog):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace")
    tail, last = [], ""
    for line in p.stdout:
        tail.append(line.rstrip())
        m = _INFO.search(line)
        if m: last = m.group(0).strip()
        prog.upd(name, time.time() - t0, info=last)
    p.wait()
    el = time.time() - t0
    if p.returncode != 0:
        prog.upd(name, el, status="fail", info=last); prog.stop()
        print(f"\n  {name} FAILED (rc={p.returncode}) -- last output:")
        for l in tail[-12:]:
            print("   |", l)
        return False
    prog.upd(name, el, status="ok", info=last); prog.finish(name, el)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refresh", action="store_true")
    args = ap.parse_args()
    steps = [s for s in STEPS if not (args.skip_refresh and s[0] == "Cache refresh")]

    print(f"\n  CLAUDE QUANTS -- live run  {time.strftime('%Y-%m-%d %H:%M')}\n")
    prog = Progress(steps)
    prog.start()
    ok_all = True
    try:
        for name, cmd, _ in steps:
            if not run_step(name, cmd, prog):
                ok_all = False
                break
    finally:
        prog.stop()

    out = os.path.join(PROJECT_DIR, "combiner_output.parquet")
    if ok_all and os.path.exists(out):
        import pandas as pd
        d = pd.read_parquet(out)
        v = d["verdict"].value_counts().to_dict()
        print(f"\n  combiner_output.parquet: FIRE={v.get('FIRE',0)}  "
              f"WAIT={v.get('WAIT',0)}  KILL={v.get('KILL',0)}  ({len(d)} scored)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
