"""
CLAUDE QUANTS — run everything in sequence.

Runs each layer script as its own subprocess, in order, streaming its output
live to the console so long steps (the ML search can take ~80 min) are
visible while they run. Does NOT stop on a step's failure — independent
later steps still get their turn. Prints a timing/status summary at the end.

Usage:  python run_all.py
"""
import subprocess
import sys
import time
import os
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Swing signals validation",     [sys.executable, "layer_swing_signals.py"]),
    ("ML ensemble validation",       [sys.executable, "layer_ml_ensemble.py", "swing"]),
    ("Alt-data signals",             [sys.executable, "layer_altdata_signals.py"]),
    ("Signal combiner",              [sys.executable, "layer_combiner.py"]),
    ("Orchestrator (paper tracker)", [sys.executable, "orchestrator.py"]),
]


def main():
    print("=" * 72)
    print(f"CLAUDE QUANTS — full sequence run started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Working directory: {PROJECT_DIR}")
    print("=" * 72)

    results = []
    overall_start = time.time()

    for name, cmd in STEPS:
        print(f"\n{'-' * 72}\n[{name}] starting: {' '.join(cmd)}\n{'-' * 72}")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_DIR)
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            status = "OK" if ok else f"FAILED (exit code {proc.returncode})"
        except Exception as e:
            elapsed = time.time() - t0
            ok = False
            status = f"ERROR: {e}"
        print(f"\n[{name}] finished in {elapsed / 60:.1f} min — {status}")
        results.append((name, status, elapsed, ok))

    total = time.time() - overall_start

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, status, elapsed, ok in results:
        marker = "OK  " if ok else "FAIL"
        print(f"  [{marker}] {name:35s} {status:30s} {elapsed / 60:6.1f} min")
    print(f"\nTotal wall time: {total / 60:.1f} min")
    print(f"Finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    n_failed = sum(1 for _, _, _, ok in results if not ok)
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
