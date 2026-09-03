# NSE Layer 1.5 - Live Progress Monitor
# Run in a SECOND terminal while layer 1.5 is running
# python nse_monitor.py

import os
import time
import pandas as pd
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer1_5_run.log")
IC_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer1_5_ic_results.parquet")

SIGNALS = ["hurst","granger","garch","cusum","earnings_drift","pairs"]

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def read_tail(path, n=30):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []

def parse_log_stats(lines):
    stats = {}
    for line in lines:
        for sig in SIGNALS:
            if sig + " fold" in line or sig + " done" in line:
                parts = line.strip().split("|")
                fold_part = parts[0] if parts else ""
                ic_val, pnl_val, n_val = None, None, None
                for p in parts:
                    p = p.strip()
                    if p.startswith("IC="):
                        try: ic_val = float(p.split("=")[1])
                        except: pass
                    if p.startswith("PnL="):
                        try: pnl_val = float(p.split("=")[1])
                        except: pass
                    if p.startswith("n="):
                        try: n_val = int(p.split("=")[1])
                        except: pass
                if ic_val is not None:
                    stats[sig] = {"ic": ic_val, "pnl": pnl_val, "n": n_val, "line": line.strip()}
    return stats

def load_ic_results():
    try:
        df = pd.read_parquet(IC_FILE)
        return df
    except Exception:
        return None

def get_current_signal(lines):
    current = "waiting..."
    for line in lines:
        for sig in SIGNALS:
            if "[" in line and sig.upper() in line.upper() and "/11]" in line:
                current = line.strip().split("INFO")[-1].strip()
    return current

def draw_bar(val, min_val=-0.1, max_val=0.1, width=20):
    if val is None:
        return "[" + "-" * width + "]"
    norm = (val - min_val) / (max_val - min_val)
    norm = max(0, min(1, norm))
    filled = int(norm * width)
    bar = "#" * filled + "-" * (width - filled)
    return "[" + bar + "]"

def main():
    print("NSE Layer 1.5 Monitor - starting...")
    time.sleep(1)

    while True:
        clear()
        now = datetime.now().strftime("%H:%M:%S")
        lines = read_tail(LOG_FILE, 50)
        stats = parse_log_stats(lines)
        current = get_current_signal(lines)
        ic_df = load_ic_results()

        print("=" * 62)
        print("  NSE DECISION ENGINE - LAYER 1.5 LIVE MONITOR")
        print("  Time: {}    Log: {}".format(now, "LIVE" if lines else "waiting for log..."))
        print("=" * 62)
        print()

        # Signal progress table
        print("  {:<18} {:>8} {:>9} {:>6}  {}".format(
            "Signal", "Last IC", "Last PnL", "n", "IC Bar"))
        print("  " + "-" * 58)

        for sig in SIGNALS:
            if sig in stats:
                s = stats[sig]
                ic  = s["ic"]  if s["ic"]  is not None else 0
                pnl = s["pnl"] if s["pnl"] is not None else 0
                n   = s["n"]   if s["n"]   is not None else 0
                bar = draw_bar(ic)
                ic_str  = "{:+.4f}".format(ic)
                pnl_str = "{:+.4f}".format(pnl)
                print("  {:<18} {:>8} {:>9} {:>6}  {}".format(
                    sig, ic_str, pnl_str, n, bar))
            else:
                print("  {:<18} {:>8} {:>9} {:>6}  {}".format(
                    sig, "pending", "pending", "-", "[" + "." * 20 + "]"))

        print()
        print("  Currently running: {}".format(current))
        print()

        # Fold summary from parquet if available
        if ic_df is not None and not ic_df.empty:
            print("  FOLD SUMMARY (from saved results)")
            print("  " + "-" * 58)
            for sig, grp in ic_df.groupby("signal"):
                ic_vals  = grp["ic"].dropna()
                pnl_vals = grp["pnl"].dropna()
                mean_ic  = ic_vals.mean() if len(ic_vals) > 0 else float("nan")
                pnl_pos  = (pnl_vals > 0).mean() if len(pnl_vals) > 0 else float("nan")
                n_folds  = len(grp)
                status   = "OK" if (not pd.isna(mean_ic) and mean_ic > 0.02) else "WATCH"
                print("  {:<18} folds={:>3}  meanIC={:>+.4f}  PnL+%={:.0%}  [{}]".format(
                    sig, n_folds, mean_ic, pnl_pos, status))
            print()

        # Last 8 log lines
        print("  RECENT LOG")
        print("  " + "-" * 58)
        for line in lines[-8:]:
            line = line.strip()
            if line:
                # trim timestamp prefix for display
                parts = line.split("  ", 2)
                display = parts[-1][:56] if len(parts) >= 2 else line[:56]
                print("  " + display)

        print()
        print("  Refresh every 15s | Ctrl+C to exit monitor (does NOT stop L1.5)")
        print("=" * 62)

        try:
            time.sleep(15)
        except KeyboardInterrupt:
            print("\n  Monitor stopped. Layer 1.5 still running in other terminal.")
            break

if __name__ == "__main__":
    main()
