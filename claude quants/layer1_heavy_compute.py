# ==============================================================================
#  NSE DECISION ENGINE v2 - LAYER 1 : HEAVY COMPUTE
#  Run weekly.
#  H2 strategy: one pool per sector (small cache per pool = fast worker startup)
#               batched submission within each pool (tqdm responsive)
#               checkpoint every H2_CHECKPOINT_EVERY completions
#               resumes automatically from stable_edges.parquet
# ==============================================================================

import os, sys, time, warnings, logging, io, gc
import requests
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from arch import arch_model
import nolds
import hurst as hurst_lib
import networkx as nx
import community as community_louvain
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ==============================================================================
#  CONFIG
# ==============================================================================
PROJECT_DIR  = r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py\claude quants"
N_WORKERS    = max(1, os.cpu_count() - 2)
COOLDOWN_SEC = 7

RAW_DIR    = os.path.join(PROJECT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
LOG_DIR    = os.path.join(PROJECT_DIR, "logs")

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

CRISIS_PERIODS = [
    ("2016-11-08", "2016-12-31"),
    ("2018-09-01", "2018-11-30"),
    ("2020-02-15", "2020-04-15"),
]

H2_CHECKPOINT_EVERY = 5000
H2_BATCH_SIZE       = 500

# ==============================================================================
#  MODULE-LEVEL WORKER FUNCTIONS  (zero indentation — required for Windows)
# ==============================================================================

def _raw_path(ticker):
    return os.path.join(RAW_DIR, f"{ticker}.parquet")

def _load_close(ticker):
    p = _raw_path(ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p, columns=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).set_index("Date")
        return df["Close"]
    except Exception:
        return None

def _load_returns(ticker):
    close = _load_close(ticker)
    if close is None or len(close) < 60:
        return None
    return close.pct_change().dropna()

def fractal_worker(ticker):
    try:
        close = _load_close(ticker)
        if close is None or len(close) < 252:
            return None
        prices  = close.values[-252:]
        log_ret = np.diff(np.log(prices + 1e-10))
        try:
            H, _, _ = hurst_lib.compute_Hc(prices, kind="price", simplified=True)
        except Exception:
            H = np.nan
        try:
            apen = nolds.sampen(log_ret)
        except Exception:
            apen = np.nan
        try:
            fd = nolds.dfa(log_ret)
        except Exception:
            fd = np.nan
        if   np.isnan(H):  ftype = "Unknown"
        elif H < 0.45:     ftype = "MR"
        elif H > 0.55:     ftype = "Trend"
        else:              ftype = "Random"
        return {"Ticker": ticker, "Hurst": H, "ApEn": apen, "FD": fd, "Type": ftype}
    except Exception:
        return None

# ==============================================================================
#  H2 GRANGER WORKER
#  Sector cache sent once per worker via initializer.
#  args = (src, tgt, sector) — strings only, no arrays in task queue.
# ==============================================================================

_RETURNS_CACHE = {}

def _init_worker_cache(cache):
    global _RETURNS_CACHE
    _RETURNS_CACHE = cache

def granger_pair_worker_ram(args):
    src, tgt, sector = args
    rx_v = _RETURNS_CACHE.get(src)
    ry_v = _RETURNS_CACHE.get(tgt)
    if rx_v is None or ry_v is None:
        return None
    try:
        n    = min(len(rx_v), len(ry_v))
        rx_v = rx_v[-n:]
        ry_v = ry_v[-n:]
        if n < 300:
            return None
        mid = n // 2
        windows = [
            (ry_v,       rx_v),
            (ry_v[:mid], rx_v[:mid]),
            (ry_v[mid:], rx_v[mid:]),
        ]
        min_f = float("inf")
        for yv, xv in windows:
            if len(yv) < 40:
                return None
            data   = np.column_stack([yv, xv])
            result = grangercausalitytests(data, maxlag=2, verbose=False)
            f, p   = result[2][0]["ssr_ftest"][:2]
            if f < 8.0 or p > 0.05:
                return None
            min_f = min(min_f, f)
        return {"Source": src, "Target": tgt,
                "F_stat": min_f, "Lag": 2, "Sector": sector}
    except Exception:
        return None

def r2_worker(ticker):
    try:
        close = _load_close(ticker)
        if close is None or len(close) < 300:
            return None
        ret    = close.pct_change()
        zscore = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
        df     = pd.DataFrame({"ret": ret, "zscore_lag": zscore.shift(1),
                               "next_ret": ret.shift(-1)}).dropna()
        if len(df) < 100:
            return None
        X = df[["zscore_lag"]].values
        y = df["next_ret"].values
        r2_scores = []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
            m  = LinearRegression().fit(X[train_idx], y[train_idx])
            yp = m.predict(X[test_idx])
            ss_res = np.sum((y[test_idx] - yp) ** 2)
            ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
            r2_scores.append(1 - ss_res / (ss_tot + 1e-10))
        r2_mean = float(np.mean(r2_scores))
        if   r2_mean > 0.40: grade = "A"
        elif r2_mean > 0.25: grade = "B"
        elif r2_mean > 0.10: grade = "C"
        else:                grade = "D"
        return {"Ticker": ticker, "Grade": grade, "R2_score": r2_mean}
    except Exception:
        return None

def garch_worker(ticker):
    try:
        close = _load_close(ticker)
        if close is None or len(close) < 252:
            return None
        log_ret = 100 * np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 200:
            return None
        res    = arch_model(log_ret, vol="Garch", p=1, q=1,
                            dist="Normal").fit(disp="off", show_warning=False)
        params = res.params
        return {
            "Ticker": ticker,
            "omega": float(params.get("omega",    np.nan)),
            "alpha": float(params.get("alpha[1]", np.nan)),
            "beta":  float(params.get("beta[1]",  np.nan)),
        }
    except Exception:
        return None

def cusum_crisis_worker(ticker):
    try:
        p = _raw_path(ticker)
        if not os.path.exists(p):
            return [], None
        df = pd.read_parquet(p, columns=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date").dropna()
        if len(df) < 252:
            return [], None
        ret    = df["Close"].pct_change().dropna()
        mean_r = ret.rolling(252, min_periods=60).mean()
        std_r  = ret.rolling(252, min_periods=60).std()
        z      = (ret - mean_r) / (std_r + 1e-10)
        cp, cn = 0.0, 0.0
        breaks = []
        for i, (idx, zval) in enumerate(z.items()):
            cp = max(0, cp + zval)
            cn = min(0, cn + zval)
            if cp > 5:
                breaks.append({"Ticker": ticker, "BreakDate": idx, "Direction": "UP"})
                cp = 0.0
            elif cn < -5:
                breaks.append({"Ticker": ticker, "BreakDate": idx, "Direction": "DOWN"})
                cn = 0.0
        crisis_rets = []
        for cs, ce in CRISIS_PERIODS:
            window = ret.loc[cs:ce]
            if len(window) > 5:
                crisis_rets.append(float(window.mean()))
        crisis_rec = None
        if crisis_rets:
            avg = float(np.mean(crisis_rets))
            if avg > 0:
                crisis_rec = {"Ticker": ticker, "CrisisReturnAvg": avg}
        return breaks, crisis_rec
    except Exception:
        return [], None

def beta_lag_worker(args):
    ticker, leader, sector = args
    try:
        ret_t3   = _load_returns(ticker)
        ret_lead = _load_returns(leader)
        if ret_t3 is None or ret_lead is None:
            return None
        common = ret_t3.index.intersection(ret_lead.index)
        if len(common) < 60 + 7:
            return None
        best_lag, best_corr = 0, -1.0
        for lag in range(1, 8):
            c = ret_t3.loc[common].corr(ret_lead.loc[common].shift(lag))
            if abs(c) > best_corr:
                best_corr, best_lag = abs(c), lag
        x    = ret_lead.loc[common].values[-60:].reshape(-1, 1)
        y    = ret_t3.loc[common].values[-60:]
        beta = float(LinearRegression().fit(x, y).coef_[0])
        return {"Ticker": ticker, "SectorLeader": leader,
                "BetaLag": best_lag, "RollingBeta": beta}
    except Exception:
        return None

def earnings_worker(ticker):
    try:
        p = _raw_path(ticker)
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p, columns=["Date", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date").dropna()
        if len(df) < 300:
            return None
        vol_z        = (df["Volume"] - df["Volume"].rolling(60).mean()) / \
                       (df["Volume"].rolling(60).std() + 1e-10)
        abs_ret      = df["Close"].pct_change().abs()
        result_flags = (vol_z > 3.0) & (abs_ret > 0.03)
        result_dates = df.index[result_flags].tolist()
        if len(result_dates) < 3:
            return None
        drifts = []
        for rd in result_dates:
            idx = df.index.get_loc(rd)
            if idx + 10 >= len(df):
                continue
            drift = (df["Close"].iloc[idx + 10] - df["Close"].iloc[idx]) / \
                    (df["Close"].iloc[idx] + 1e-10)
            drifts.append(drift)
        if len(drifts) < 3:
            return None
        drifts    = np.array(drifts)
        avg_drift = float(np.mean(drifts))
        direction = "UP" if avg_drift > 0 else "DOWN"
        pct_same  = np.mean(drifts > 0) if direction == "UP" else np.mean(drifts < 0)
        return {
            "Ticker": ticker,
            "AvgDrift_D1_D10": avg_drift,
            "ReliabilityFlag": bool(pct_same >= 0.70),
            "Direction": direction,
        }
    except Exception:
        return None

def spread_worker(args):
    src, tgt, c1, c2 = args
    try:
        close_src = _load_close(src)
        close_tgt = _load_close(tgt)
        if close_src is None or close_tgt is None:
            return None
        common = close_src.index.intersection(close_tgt.index)
        if len(common) < 252:
            return None
        ratio = np.log(close_src.loc[common] / close_tgt.loc[common])
        if adfuller(ratio.dropna(), maxlag=5)[1] > 0.05:
            return None
        spread = ratio.dropna()
        lag_s  = spread.shift(1).dropna()
        d_s    = spread.diff().dropna()
        idx    = lag_s.index.intersection(d_s.index)
        if len(idx) < 30:
            return None
        X   = sm.add_constant(lag_s.loc[idx])
        res = sm.OLS(d_s.loc[idx], X).fit()
        lam = res.params.iloc[1]
        if lam >= 0 or np.isnan(lam):
            return None
        half_life = float(-np.log(2) / lam)
        if half_life <= 0 or half_life > 252:
            return None
        z_now = float((ratio.iloc[-1] - ratio.mean()) / (ratio.std() + 1e-10))
        return {"Ticker_A": src, "Ticker_B": tgt,
                "Cluster_A": c1, "Cluster_B": c2,
                "HalfLife": half_life, "Z_now": z_now}
    except Exception:
        return None

# ==============================================================================
#  UTILITY HELPERS
# ==============================================================================

def ppath(filename):
    return os.path.join(PROJECT_DIR, filename)

def log_failure(ticker, reason):
    with open(os.path.join(LOG_DIR, "failed_tickers.txt"), "a") as f:
        f.write(f"{datetime.now().isoformat()}  {ticker}  {reason}\n")

def cooldown(label="next block"):
    print(f"\n  Cooling down {COOLDOWN_SEC}s before {label} ...", end="", flush=True)
    for _ in range(COOLDOWN_SEC):
        time.sleep(1)
        print(".", end="", flush=True)
    print(" done\n")
    gc.collect()

def run_parallel(worker_fn, items, desc, n_workers=N_WORKERS):
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(worker_fn, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception:
                pass
    return results

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ==============================================================================
#  H2 — ONE POOL PER SECTOR + BATCHED SUBMISSION
#
#  Why per-sector pools?
#  On Windows, ProcessPoolExecutor spawns workers via "spawn" (not fork).
#  Each worker re-imports the entire module (statsmodels, arch, nolds, hurst…).
#  With a single pool and initializer=full_cache, all 10 workers must start
#  before the first batch runs — 30–90s of silence.
#  Per-sector pools: each pool only pickles that sector's cache (tiny), workers
#  start faster, and you see output immediately after each sector begins.
# ==============================================================================

def _run_sector(sector, sector_pairs, sector_cache, n_workers,
                results, done_count, total, t_start,
                stable_path, pbar):
    """Process one sector's pairs in a fresh pool. Updates results in-place."""
    n_sec = len(sector_pairs)
    tqdm.write(f"  [{sector}]  {len(sector_cache)} tickers  {n_sec:,} pairs  "
               f"— pool starting ...", end="", )
    sys.stdout.flush()

    w = min(n_workers, n_sec)   # no point spawning more workers than pairs

    with ProcessPoolExecutor(max_workers=w,
                             initializer=_init_worker_cache,
                             initargs=(sector_cache,)) as exe:

        tqdm.write(" ready")

        for batch_start in range(0, n_sec, H2_BATCH_SIZE):
            batch   = sector_pairs[batch_start : batch_start + H2_BATCH_SIZE]
            futures = [exe.submit(granger_pair_worker_ram, a) for a in batch]

            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r is not None:
                        results.append(r)
                except Exception:
                    pass

                done_count[0] += 1
                pbar.update(1)

                if done_count[0] % H2_CHECKPOINT_EVERY == 0:
                    if results:
                        pd.DataFrame(results).to_parquet(stable_path, index=False)
                    elapsed = time.time() - t_start
                    remaining = total - done_count[0]
                    eta_s = (elapsed / done_count[0]) * remaining if done_count[0] else 0
                    tqdm.write(
                        f"  Checkpoint: {len(results)} edges | "
                        f"this-run {done_count[0]/total*100:.1f}% | ETA {eta_s/60:.1f} min"
                    )


def run_h2_granger(sector_groups, stable_path, n_workers=N_WORKERS):
    """
    Granger causality pipeline with per-sector pools:
    1. Preload ALL returns into RAM once
    2. Resume from checkpoint if available
    3. For each sector: create a small pool (just that sector's tickers),
       submit pairs in batches of H2_BATCH_SIZE, checkpoint every
       H2_CHECKPOINT_EVERY completions
    """

    # ── STEP 1: Preload ──────────────────────────────────────────────────────
    print("  [H2] Preloading returns into RAM...")
    all_sector_tickers = list({t for tickers in sector_groups.values() for t in tickers})

    returns_cache = {}
    for t in tqdm(all_sector_tickers, desc="  Loading returns"):
        r = _load_returns(t)
        if r is not None and len(r) >= 300:
            returns_cache[t] = r.values

    print(f"  Loaded {len(returns_cache)} tickers into RAM")

    # ── STEP 2: Resume from checkpoint ──────────────────────────────────────
    done_pairs       = set()
    existing_results = []

    if os.path.exists(stable_path):
        try:
            existing_df = pd.read_parquet(stable_path)
            if len(existing_df) > 0 and "Source" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    done_pairs.add((row["Source"], row["Target"]))
                existing_results = existing_df.to_dict("records")
                print(f"  Resuming: {len(done_pairs):,} pairs already done")
        except Exception:
            print("  Could not read checkpoint — starting fresh")

    # ── STEP 3: Build per-sector pair lists and total count ──────────────────
    sector_work = {}   # sector -> list of (src, tgt, sector)
    for sector, tickers in sector_groups.items():
        valid = [t for t in tickers if t in returns_cache]
        if len(valid) < 2:
            continue
        pairs = [
            (src, tgt, sector)
            for src, tgt in itertools.permutations(valid, 2)
            if (src, tgt) not in done_pairs
        ]
        if pairs:
            sector_work[sector] = (valid, pairs)

    total_remaining = sum(len(p) for _, p in sector_work.values())
    total_all       = total_remaining + len(done_pairs)
    print(f"  Sectors to process : {len(sector_work)}")
    print(f"  Remaining pairs    : {total_remaining:,}  (total incl. done: {total_all:,})")

    if total_remaining == 0:
        print("  All pairs already computed — loading from checkpoint")
        return pd.read_parquet(stable_path)

    # ── STEP 4: Per-sector pool execution ────────────────────────────────────
    results    = list(existing_results)
    done_count = [0]    # counts NEW completions only (not pre-existing done_pairs)
    t_start    = time.time()

    print(f"\n  Batch size : {H2_BATCH_SIZE} pairs/batch")
    print(f"  Checkpoint : every {H2_CHECKPOINT_EVERY} completions")
    print(f"  Workers    : {n_workers}")
    print()

    with tqdm(total=total_remaining, desc="  H2 Granger", unit="pair") as pbar:
        for sector, (valid, sector_pairs) in sector_work.items():
            sector_cache = {t: returns_cache[t] for t in valid}
            _run_sector(
                sector, sector_pairs, sector_cache, n_workers,
                results, done_count, total_remaining, t_start,
                stable_path, pbar,
            )

    final_df = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Source", "Target", "F_stat", "Lag", "Sector"])
    final_df.to_parquet(stable_path, index=False)
    return final_df

# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":

    for d in [RAW_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  NSE DECISION ENGINE v2 - LAYER 1")
    print("  Weekly run | H2: per-sector pools + batched submission")
    print("=" * 60)
    print(f"\n  PROJECT_DIR  : {PROJECT_DIR}")
    print(f"  CPU cores    : {os.cpu_count()}  ->  using {N_WORKERS} workers")
    print(f"  Cooldown     : {COOLDOWN_SEC}s between blocks")
    print(f"  H2 batch     : {H2_BATCH_SIZE} pairs/batch")
    print(f"  H2 checkpoint: every {H2_CHECKPOINT_EVERY} completions")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION 1 - UNIVERSE BUILDER
    # ══════════════════════════════════════════════════════════════════════════
    print_section("SESSION 1 - UNIVERSE BUILDER")

    print("\n[S1-1] Fetching NSE equity list ...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        resp = requests.get(NSE_EQUITY_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        nse_list = pd.read_csv(io.StringIO(resp.text))
        nse_list.columns = nse_list.columns.str.strip()
        if "SERIES" in nse_list.columns:
            nse_list = nse_list[nse_list["SERIES"].str.strip() == "EQ"]
        rename = {}
        for c in nse_list.columns:
            cu = c.strip().upper()
            if "SYMBOL"     in cu: rename[c] = "Symbol"
            elif "NAME"     in cu: rename[c] = "CompanyName"
            elif "INDUSTRY" in cu or "SECTOR" in cu: rename[c] = "Sector"
        nse_list = nse_list.rename(columns=rename)
        for col in ["Symbol", "CompanyName", "Sector"]:
            if col not in nse_list.columns:
                nse_list[col] = "Unknown"
        nse_list["Symbol"] = nse_list["Symbol"].str.strip()
        nse_list = nse_list[["Symbol", "CompanyName", "Sector"]].drop_duplicates("Symbol")
        print(f"  {len(nse_list)} EQ-series tickers")
    except Exception as e:
        print(f"  NSE fetch failed: {e}")
        nse_list = pd.DataFrame(columns=["Symbol", "CompanyName", "Sector"])

    print(f"\n[S1-2] Downloading OHLCV for {len(nse_list)} tickers ...")
    print("       (already-saved tickers skipped automatically)\n")

    MIN_TRADING_DAYS = 252
    downloaded_ok, skipped = [], []
    t0 = time.time()

    for _, row in tqdm(nse_list.iterrows(), total=len(nse_list), desc="OHLCV download"):
        sym  = row["Symbol"]
        path = _raw_path(sym)
        if os.path.exists(path):
            downloaded_ok.append(sym)
            continue
        try:
            df = yf.download(sym + ".NS", period="max",
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < MIN_TRADING_DAYS:
                log_failure(sym, f"only {len(df)} rows")
                skipped.append(sym)
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.index.name = "Date"
            df.reset_index(inplace=True)
            df["Ticker"] = sym
            df.to_parquet(path, index=False)
            downloaded_ok.append(sym)
        except Exception as e:
            log_failure(sym, str(e))
            skipped.append(sym)

    print(f"\n  Downloaded : {len(downloaded_ok)}")
    print(f"  Skipped    : {len(skipped)}")
    print(f"  Time       : {(time.time()-t0)/60:.1f} min")

    cooldown("availability filter")

    print("[S1-3] Availability filter (>=3yr continuous, gap <30d) ...")
    survivors = []
    for sym in tqdm(downloaded_ok, desc="Filter"):
        path = _raw_path(sym)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=["Date", "Close"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
            span = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
            if span < 3 * 365:
                continue
            gaps = df["Date"].diff().dt.days.dropna()
            if (gaps > 30).any():
                continue
            survivors.append({"Symbol": sym,
                               "DataStart": df["Date"].iloc[0].date(),
                               "DataEnd":   df["Date"].iloc[-1].date()})
        except Exception:
            pass
    survivors_df = pd.DataFrame(survivors)
    print(f"  Survivors: {len(survivors_df)}")

    print("\n[S1-4] Computing liquidity tiers ...")
    tier_rows = []
    for _, row in tqdm(survivors_df.iterrows(), total=len(survivors_df), desc="Tiers"):
        sym = row["Symbol"]
        try:
            df  = pd.read_parquet(_raw_path(sym), columns=["Date", "Close", "Volume"])
            df  = df.sort_values("Date").tail(252)
            avg = (df["Close"] * df["Volume"]).mean()
            tier = "T1" if avg >= 5e7 else "T2" if avg >= 1e7 else "T3"
            tier_rows.append({"Symbol": sym, "Tier": tier, "AvgDailyValue": avg})
        except Exception:
            pass
    tier_df = pd.DataFrame(tier_rows)

    nse_meta    = nse_list.set_index("Symbol")
    master_rows = []
    for _, row in survivors_df.iterrows():
        sym  = row["Symbol"]
        meta = nse_meta.loc[sym] if sym in nse_meta.index else \
               pd.Series({"CompanyName": "Unknown", "Sector": "Unknown"})
        tr   = tier_df[tier_df["Symbol"] == sym]
        tier = tr["Tier"].values[0]          if len(tr) else "T3"
        avg  = tr["AvgDailyValue"].values[0] if len(tr) else 0.0
        master_rows.append({
            "Ticker":        sym,
            "CompanyName":   meta["CompanyName"],
            "Sector":        meta["Sector"],
            "Tier":          tier,
            "DataStart":     row["DataStart"],
            "DataEnd":       row["DataEnd"],
            "AvgDailyValue": avg,
        })
    universe_master = pd.DataFrame(master_rows)
    universe_master.to_parquet(ppath("universe_master.parquet"), index=False)
    print(f"\n  universe_master.parquet - {len(universe_master)} tickers")
    if len(universe_master):
        print(universe_master.groupby("Tier").size().to_string())

    all_tickers   = universe_master["Ticker"].tolist()
    sectors_of    = dict(zip(universe_master["Ticker"], universe_master["Sector"]))
    t1t2_master   = universe_master[universe_master["Tier"].isin(["T1", "T2"])]
    t3_tickers    = universe_master[universe_master["Tier"] == "T3"]["Ticker"].tolist()
    sector_groups = t1t2_master.groupby("Sector")["Ticker"].apply(list).to_dict()

    cooldown("H1 Fractal DNA")

    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION 2 - HEAVY COMPUTE A
    # ══════════════════════════════════════════════════════════════════════════
    print_section("SESSION 2 - HEAVY COMPUTE A")

    fractal_path = ppath("fractal_dna.parquet")
    print(f"[H1] Fractal DNA - {len(all_tickers)} tickers ({N_WORKERS} workers) ...")
    results    = run_parallel(fractal_worker, all_tickers, "H1 Fractal DNA")
    fractal_df = pd.DataFrame(results)
    fractal_df.to_parquet(fractal_path, index=False)
    print(f"  fractal_dna.parquet - {len(fractal_df)} tickers")
    if "Type" in fractal_df.columns:
        for t, n in fractal_df["Type"].value_counts().items():
            print(f"    {t}: {n}")

    cooldown("H2 Granger network")

    stable_path = ppath("stable_edges.parquet")
    print(f"[H2] Granger network - within-sector pairs, T1+T2 only ...")
    print(f"     Per-sector pools | Batch={H2_BATCH_SIZE} | Checkpoint every {H2_CHECKPOINT_EVERY}")
    print(f"     Resumes automatically from checkpoint if interrupted\n")

    stable_edges_df = run_h2_granger(sector_groups, stable_path, n_workers=N_WORKERS)

    print(f"\n  stable_edges.parquet - {len(stable_edges_df)} stable edges")
    if len(stable_edges_df):
        print(stable_edges_df.groupby("Sector").size()
              .sort_values(ascending=False).head(5).to_string())

    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION 3 - HEAVY COMPUTE B
    # ══════════════════════════════════════════════════════════════════════════
    print_section("SESSION 3 - HEAVY COMPUTE B")

    clusters_path = ppath("clusters.parquet")
    print("[H3A] Louvain community detection ...")
    G = nx.Graph()
    for _, row in stable_edges_df.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["F_stat"])
    if G.number_of_nodes() == 0:
        print("  No edges - falling back to sector-based communities")
        clusters_df = t1t2_master[["Ticker", "Sector"]].copy()
        sector_ids  = {s: i for i, s in enumerate(clusters_df["Sector"].unique())}
        clusters_df["CommunityID"] = clusters_df["Sector"].map(sector_ids)
        clusters_df = clusters_df[["Ticker", "CommunityID"]]
    else:
        partition   = community_louvain.best_partition(G, weight="weight")
        clusters_df = pd.DataFrame([{"Ticker": t, "CommunityID": c}
                                    for t, c in partition.items()])
    clusters_df.to_parquet(clusters_path, index=False)
    print(f"  clusters.parquet - {clusters_df['CommunityID'].nunique()} communities")

    r2_path    = ppath("r2_grades.parquet")
    mr_tickers = fractal_df[fractal_df["Type"] == "MR"]["Ticker"].tolist() \
                 if "Type" in fractal_df.columns else []
    print(f"[H3B] R2 grades - {len(mr_tickers)} MR tickers ({N_WORKERS} workers) ...")
    results = run_parallel(r2_worker, mr_tickers, "H3B R2 grades")
    r2_df   = pd.DataFrame(results)
    r2_df.to_parquet(r2_path, index=False)
    print(f"  r2_grades.parquet - {len(r2_df)} tickers")
    if len(r2_df):
        print(r2_df.groupby("Grade").size().to_string())

    cooldown("H3C GARCH")

    garch_path = ppath("garch_params.parquet")
    print(f"[H3C] GARCH(1,1) - {len(all_tickers)} tickers ({N_WORKERS} workers) ...")
    results  = run_parallel(garch_worker, all_tickers, "H3C GARCH")
    garch_df = pd.DataFrame(results)
    garch_df.to_parquet(garch_path, index=False)
    print(f"  garch_params.parquet - {len(garch_df)} tickers")

    cooldown("H3D CUSUM + crisis alpha")

    cusum_path  = ppath("cusum_breaks.parquet")
    crisis_path = ppath("crisis_alpha.parquet")
    print(f"[H3D] CUSUM + crisis alpha - {len(all_tickers)} tickers ({N_WORKERS} workers) ...")
    cusum_rows, crisis_rows = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(cusum_crisis_worker, t): t for t in all_tickers}
        for fut in tqdm(as_completed(futures), total=len(all_tickers), desc="H3D CUSUM"):
            try:
                breaks, crisis_rec = fut.result()
                cusum_rows.extend(breaks)
                if crisis_rec:
                    crisis_rows.append(crisis_rec)
            except Exception:
                pass
    cusum_df  = pd.DataFrame(cusum_rows)
    crisis_df = pd.DataFrame(crisis_rows)
    cusum_df.to_parquet(cusum_path, index=False)
    crisis_df.to_parquet(crisis_path, index=False)
    print(f"  cusum_breaks.parquet  - {len(cusum_df)} break events")
    print(f"  crisis_alpha.parquet  - {len(crisis_df)} positive-crisis tickers")

    cooldown("H4 cross-tier beta map")

    tier_beta_path = ppath("tier_beta.parquet")
    print(f"[H4] Cross-tier beta - {len(t3_tickers)} T3 tickers ({N_WORKERS} workers) ...")
    t1_df      = universe_master[universe_master["Tier"] == "T1"] \
                     .sort_values("AvgDailyValue", ascending=False)
    leader_map = dict(t1_df.groupby("Sector").first()["Ticker"])
    tasks = []
    for sym in t3_tickers:
        sec = sectors_of.get(sym, "Unknown")
        if sec in leader_map:
            tasks.append((sym, leader_map[sec], sec))
    results      = run_parallel(beta_lag_worker, tasks, "H4 Beta/lag")
    tier_beta_df = pd.DataFrame(results)
    tier_beta_df.to_parquet(tier_beta_path, index=False)
    print(f"  tier_beta.parquet - {len(tier_beta_df)} T3 tickers mapped")

    cooldown("H5 earnings DNA")

    earnings_path = ppath("earnings_dna_drift.parquet")
    print(f"[H5] Earnings DNA - {len(all_tickers)} tickers ({N_WORKERS} workers) ...")
    results     = run_parallel(earnings_worker, all_tickers, "H5 Earnings DNA")
    earnings_df = pd.DataFrame(results)
    earnings_df.to_parquet(earnings_path, index=False)
    n_rel = earnings_df["ReliabilityFlag"].sum() if len(earnings_df) else 0
    print(f"  earnings_dna_drift.parquet - {len(earnings_df)} tickers ({n_rel} reliable)")

    cooldown("H6 cross-community OU spreads")

    spread_path = ppath("spread_history.parquet")
    print("[H6] Building cross-community pairs ...")
    ticker_community = dict(zip(clusters_df["Ticker"], clusters_df["CommunityID"]))
    seen, cross_pairs = set(), []
    for _, row in stable_edges_df.iterrows():
        src, tgt = row["Source"], row["Target"]
        c1 = ticker_community.get(src)
        c2 = ticker_community.get(tgt)
        if c1 is not None and c2 is not None and c1 != c2:
            key = tuple(sorted([src, tgt]))
            if key not in seen:
                seen.add(key)
                cross_pairs.append((src, tgt, c1, c2))
    print(f"  Cross-community pairs: {len(cross_pairs)}")
    results   = run_parallel(spread_worker, cross_pairs, "H6 OU spreads")
    spread_df = pd.DataFrame(results)
    spread_df.to_parquet(spread_path, index=False)
    print(f"  spread_history.parquet - {len(spread_df)} cointegrated pairs")

    cooldown("sector momentum seed")

    print("[Seed] sector_momentum.parquet ...")
    sec_mom_path    = ppath("sector_momentum.parquet")
    MOMENTUM_WINDOW = 20
    sec_rows        = []
    for sector, tickers in tqdm(sector_groups.items(), desc="Sector momentum"):
        rets_all = []
        for t in tickers:
            r = _load_returns(t)
            if r is not None and len(r) > MOMENTUM_WINDOW:
                rets_all.append(pd.Series(r.values[-MOMENTUM_WINDOW:]))
        if not rets_all:
            continue
        aligned     = pd.concat(rets_all, axis=1).mean(axis=1)
        rolling_ret = float((1 + aligned).prod() - 1)
        sec_rows.append({"Sector": sector,
                          "Rolling20dReturn": rolling_ret,
                          "AsOf": datetime.today().date()})
    sec_mom_df         = pd.DataFrame(sec_rows)
    sec_mom_df["Rank"] = sec_mom_df["Rolling20dReturn"].rank(ascending=False).astype(int)
    sec_mom_df.to_parquet(sec_mom_path, index=False)
    print(f"  sector_momentum.parquet - {len(sec_mom_df)} sectors")
    if len(sec_mom_df):
        print(sec_mom_df.sort_values("Rank").head(5).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  NSE DECISION ENGINE v2 - LAYER 1 COMPLETE")
    print("=" * 60)

    parquet_files = [
        "universe_master.parquet",
        "fractal_dna.parquet",
        "stable_edges.parquet",
        "clusters.parquet",
        "r2_grades.parquet",
        "garch_params.parquet",
        "cusum_breaks.parquet",
        "crisis_alpha.parquet",
        "tier_beta.parquet",
        "earnings_dna_drift.parquet",
        "spread_history.parquet",
        "sector_momentum.parquet",
    ]

    print(f"\n  {'File':<35} {'Rows':>7}  {'Size':>8}")
    print(f"  {'-'*54}")
    for fname in parquet_files:
        fp = ppath(fname)
        if os.path.exists(fp):
            try:
                n    = len(pd.read_parquet(fp))
                size = f"{os.path.getsize(fp)//1024} KB"
                print(f"  OK  {fname:<35} {n:>7}  {size:>8}")
            except Exception:
                print(f"  OK  {fname:<35} (exists)")
        else:
            print(f"  XX  {fname:<35} MISSING")

    print(f"\n  Finished at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  All outputs : {PROJECT_DIR}")
    print("\n  Ready for Layer 2 - run layer2_daily_filter.py each evening.")
    print("=" * 60)
