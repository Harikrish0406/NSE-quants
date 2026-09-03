# ==============================================================================
#  NSE DECISION ENGINE v2 — LAYER 1 DAILY
#  Runs every weekday (Mon–Fri) after Layer 0 NSE + FnO
#  Fast: only recomputes what actually changes day-to-day
#
#  Outputs (overwrites same parquets Layer 2+ read):
#    sector_momentum.parquet   — full recalc (20d rolling, pure math, ~10s)
#    spread_history.parquet    — Z_now column only, pairs/half-life untouched
#    cusum_breaks.parquet      — accumulate mode, last 5 trading days only
#
#  Does NOT touch:
#    earnings_dna.parquet      — handled by layer0_nse_data.py (real NSE calendar)
#    fractal_dna.parquet       — weekend only (Hurst/ApEn/FD, structurally stable)
#    stable_edges.parquet      — weekend only (Granger, needs 300+ days)
#    clusters.parquet          — weekend only (Louvain community detection)
#    r2_grades.parquet         — weekend only (walk-forward R², MR tickers)
#    garch_params.parquet      — weekend only (GARCH(1,1) params)
#    crisis_alpha.parquet      — weekend only (crisis period returns)
#    tier_beta.parquet         — weekend only (cross-tier beta/lag map)
#    spread_history.parquet    — pairs/half-life weekend, Z_now daily (here)
# ==============================================================================

import os, sys, time, warnings, logging, gc
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ==============================================================================
#  CONFIG  — must match layer1_heavy_compute.py exactly
# ==============================================================================
PROJECT_DIR = r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py\claude quants"
RAW_DIR     = os.path.join(PROJECT_DIR, "data", "raw")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")
LOG_DIR     = os.path.join(PROJECT_DIR, "logs")

MOMENTUM_WINDOW  = 20       # days for sector momentum rolling return
CUSUM_LOOKBACK   = 5        # trading days to scan for new CUSUM breaks
CUSUM_THRESHOLD  = 5.0      # same threshold as layer1_heavy_compute
MIN_SPREAD_ROWS  = 30       # minimum rows to compute Z_now

# ==============================================================================
#  HELPERS
# ==============================================================================

def ppath(filename):
    return os.path.join(PROJECT_DIR, filename)

def _raw_path(ticker):
    return os.path.join(RAW_DIR, f"{ticker}.parquet")

def _load_close(ticker, tail=None):
    p = _raw_path(ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p, columns=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).set_index("Date")
        s  = df["Close"]
        return s.tail(tail) if tail else s
    except Exception:
        return None

def _load_returns(ticker, tail=None):
    close = _load_close(ticker, tail=tail)
    if close is None or len(close) < 10:
        return None
    return close.pct_change().dropna()

def print_section(title):
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")

def print_done(label, n, t0):
    print(f"  ✔  {label} — {n} rows  ({time.time()-t0:.1f}s)")

# ==============================================================================
#  D1 : SECTOR MOMENTUM  (full recalc, fast — only reads last 20 rows/ticker)
# ==============================================================================

def run_sector_momentum():
    print_section("D1 — Sector Momentum (20d)")
    t0 = time.time()

    universe_path = ppath("universe_master.parquet")
    if not os.path.exists(universe_path):
        print("  ✗  universe_master.parquet not found — skipping")
        return

    age_days = (date.today() - date.fromtimestamp(os.path.getmtime(universe_path))).days
    if age_days > 7:
        print(f"  ⚠  universe_master.parquet is {age_days}d old — consider running Layer 1 Heavy")

    universe = pd.read_parquet(universe_path)
    t1t2     = universe[universe["Tier"].isin(["T1", "T2"])]
    sector_groups = t1t2.groupby("Sector")["Ticker"].apply(list).to_dict()

    sec_rows = []
    for sector, tickers in tqdm(sector_groups.items(), desc="  Sectors"):
        rets_all = []
        for t in tickers:
            r = _load_returns(t, tail=MOMENTUM_WINDOW + 5)
            if r is not None and len(r) >= MOMENTUM_WINDOW:
                rets_all.append(r.tail(MOMENTUM_WINDOW))
        if not rets_all:
            continue
        aligned     = pd.concat(rets_all, axis=1).mean(axis=1)
        rolling_ret = float((1 + aligned).prod() - 1)
        sec_rows.append({
            "Sector":           sector,
            "Rolling20dReturn": rolling_ret,
            "AsOf":             date.today(),
        })

    if not sec_rows:
        print("  ✗  No sector data computed")
        return

    df         = pd.DataFrame(sec_rows)
    df["Rank"] = df["Rolling20dReturn"].rank(ascending=False).astype(int)
    df.to_parquet(ppath("sector_momentum.parquet"), index=False)
    print_done("sector_momentum.parquet", len(df), t0)
    print(df.sort_values("Rank").head(5)[["Rank","Sector","Rolling20dReturn"]].to_string(index=False))

# ==============================================================================
#  D2 : SPREAD Z_NOW  (recalc Z_now only — pairs/half-life from weekend run)
# ==============================================================================

def run_spread_znow():
    print_section("D2 — Spread Z_now (today's z-score only)")
    t0 = time.time()

    spread_path = ppath("spread_history.parquet")
    if not os.path.exists(spread_path):
        print("  ✗  spread_history.parquet not found — skipping (run weekend Layer 1 first)")
        return

    spread_df = pd.read_parquet(spread_path)
    if spread_df.empty:
        print("  ✗  spread_history.parquet is empty — skipping")
        return

    required = {"Ticker_A", "Ticker_B", "HalfLife"}
    if not required.issubset(spread_df.columns):
        print(f"  ✗  Missing columns {required - set(spread_df.columns)} — skipping")
        return

    updated, skipped = 0, 0
    z_values = []

    for i, row in tqdm(spread_df.iterrows(), total=len(spread_df), desc="  Pairs"):
        src = row["Ticker_A"]
        tgt = row["Ticker_B"]
        try:
            close_src = _load_close(src, tail=300)
            close_tgt = _load_close(tgt, tail=300)
            if close_src is None or close_tgt is None:
                skipped += 1
                z_values.append(np.nan)
                continue

            common = close_src.index.intersection(close_tgt.index)
            if len(common) < MIN_SPREAD_ROWS:
                skipped += 1
                z_values.append(np.nan)
                continue

            ratio  = np.log(close_src.loc[common] / close_tgt.loc[common]).dropna()
            z_now  = float((ratio.iloc[-1] - ratio.mean()) / (ratio.std() + 1e-10))

            z_values.append(z_now)
            updated += 1
        except Exception:
            skipped += 1
            z_values.append(np.nan)
            continue

    spread_df["Z_now"] = z_values
    spread_df.to_parquet(spread_path, index=False)
    print_done("spread_history.parquet", len(spread_df), t0)
    print(f"  Z_now updated: {updated}  skipped: {skipped}")

    # Show top mean-reversion opportunities right now
    if "Z_now" in spread_df.columns and len(spread_df):
        extremes = spread_df[spread_df["Z_now"].abs() > 2.0].copy()
        if len(extremes):
            extremes["AbsZ"] = extremes["Z_now"].abs()
            top = extremes.nlargest(5, "AbsZ")[["Ticker_A","Ticker_B","HalfLife","Z_now"]]
            print(f"\n  Top spread opportunities (|Z| > 2.0):")
            print(top.to_string(index=False))

# ==============================================================================
#  D3 : CUSUM BREAKS  (accumulate — only scan last N trading days)
# ==============================================================================

def _get_recent_trading_days(n=5):
    """Return last n trading days — excludes weekends and known NSE holidays."""
    NSE_HOLIDAYS = {
        # Add/update annually
        date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14),
        date(2025, 3, 31), date(2025, 4, 10), date(2025, 4, 14),
        date(2025, 4, 18), date(2025, 5,  1), date(2025, 8, 15),
        date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 2),
        date(2025, 10,24), date(2025,11,  5), date(2025,12, 25),
        date(2026, 1, 26), date(2026, 3, 20), date(2026, 4,  2),
        date(2026, 4,  3), date(2026, 4, 14), date(2026, 4, 30),
        date(2026, 8, 15), date(2026,10, 21), date(2026,11, 24),
        date(2026,12, 25),
    }
    days = []
    d    = date.today()
    while len(days) < n:
        if d.weekday() < 5 and d not in NSE_HOLIDAYS:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)

def run_cusum_incremental():
    print_section("D3 — CUSUM Breaks (incremental, last 5 trading days)")
    t0 = time.time()

    universe_path = ppath("universe_master.parquet")
    cusum_path    = ppath("cusum_breaks.parquet")

    if not os.path.exists(universe_path):
        print("  ✗  universe_master.parquet not found — skipping")
        return

    universe    = pd.read_parquet(universe_path)
    all_tickers = universe["Ticker"].tolist()

    # Load existing breaks to accumulate into
    if os.path.exists(cusum_path):
        existing = pd.read_parquet(cusum_path)
        existing["BreakDate"] = pd.to_datetime(existing["BreakDate"])
    else:
        print("  ⚠  No existing cusum_breaks.parquet — will create fresh")
        existing = pd.DataFrame(columns=["Ticker","BreakDate","Direction"])

    recent_days  = _get_recent_trading_days(CUSUM_LOOKBACK)
    cutoff_date  = pd.Timestamp(recent_days[0])
    new_breaks   = []

    for ticker in tqdm(all_tickers, desc="  CUSUM scan"):
        p = _raw_path(ticker)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_parquet(p, columns=["Date", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date").dropna()

            if len(df) < 252:
                continue

            ret    = df["Close"].pct_change().dropna()
            mean_r = ret.rolling(252, min_periods=60).mean()
            std_r  = ret.rolling(252, min_periods=60).std()
            z      = (ret - mean_r) / (std_r + 1e-10)

            # Only scan recent window — but carry forward cumulative sums
            # from before the window so we don't miss breaks at the boundary
            # Use last 252 rows for CUSUM state, but only record breaks in recent_days
            z_full = z.dropna()
            if len(z_full) < 60:
                continue

            cp, cn = 0.0, 0.0
            for idx, zval in z_full.items():
                cp = max(0, cp + zval)
                cn = min(0, cn + zval)

                # Check upside and downside independently — elif was blocking
                # downside detection for any date where upside threshold crossed
                if cp > CUSUM_THRESHOLD:
                    if idx >= cutoff_date:
                        new_breaks.append({
                            "Ticker":    ticker,
                            "BreakDate": idx,
                            "Direction": "UP"
                        })
                    cp = 0.0
                if cn < -CUSUM_THRESHOLD:
                    if idx >= cutoff_date:
                        new_breaks.append({
                            "Ticker":    ticker,
                            "BreakDate": idx,
                            "Direction": "DOWN"
                        })
                    cn = 0.0

        except Exception:
            continue

    new_df = pd.DataFrame(new_breaks) if new_breaks else pd.DataFrame(
        columns=["Ticker", "BreakDate", "Direction"])

    if len(new_df):
        new_df["BreakDate"] = pd.to_datetime(new_df["BreakDate"])

    # Merge: drop old rows for recent dates, append new ones, deduplicate
    if len(existing):
        old_trimmed = existing[existing["BreakDate"] < cutoff_date]
    else:
        old_trimmed = existing

    combined = pd.concat([old_trimmed, new_df], ignore_index=True)
    combined  = combined.drop_duplicates(subset=["Ticker","BreakDate","Direction"])
    combined  = combined.sort_values("BreakDate").reset_index(drop=True)
    combined.to_parquet(cusum_path, index=False)

    print_done("cusum_breaks.parquet", len(combined), t0)
    print(f"  New breaks found   : {len(new_df)}")
    print(f"  Total breaks stored: {len(combined)}")

    if len(new_df):
        recent = new_df.sort_values("BreakDate", ascending=False).head(10)
        print(f"\n  Most recent breaks:")
        print(recent[["Ticker","BreakDate","Direction"]].to_string(index=False))

# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":

    for d in [RAW_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  NSE DECISION ENGINE v2 — LAYER 1 DAILY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Weekday incremental compute (fast)")
    print("=" * 60)

    T0 = time.time()

    run_sector_momentum()
    gc.collect()

    run_spread_znow()
    gc.collect()

    run_cusum_incremental()
    gc.collect()

    total = time.time() - T0
    print(f"\n{'='*60}")
    print(f"  LAYER 1 DAILY COMPLETE  —  {total:.1f}s total")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ready for Layer 2.")
    print(f"{'='*60}")

