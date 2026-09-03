"""
patch_sectors.py — One-time sector enrichment for universe_master.parquet
=========================================================================
Yahoo Finance has sector data for NSE stocks that the NSE equity CSV lacks.
This script fetches sector info in parallel and patches universe_master.parquet,
then regenerates sector_momentum.parquet from the updated data.

Run once (takes ~1-2 minutes):
    python patch_sectors.py
"""

from __future__ import annotations
import os, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import yfinance as yf

BASE = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
UNIVERSE_PATH = BASE / "universe_master.parquet"
SECTOR_MOM_PATH = BASE / "sector_momentum.parquet"
CACHE_PATH = BASE / "sector_cache.json"

MOMENTUM_WINDOW = 20
MAX_WORKERS = 30

# Yahoo Finance sector → cleaner label map
_YF_SECTOR_MAP = {
    "Financial Services":       "Financial Services",
    "Technology":               "Technology",
    "Healthcare":               "Healthcare",
    "Consumer Cyclical":        "Consumer Cyclical",
    "Consumer Defensive":       "Consumer Defensive",
    "Industrials":              "Industrials",
    "Basic Materials":          "Basic Materials",
    "Energy":                   "Energy",
    "Utilities":                "Utilities",
    "Communication Services":   "Communication Services",
    "Real Estate":              "Real Estate",
}

def fetch_sector(ticker: str) -> tuple[str, str]:
    """Return (ticker, sector_string) — falls back to 'Unknown'."""
    try:
        info = yf.Ticker(ticker + ".NS").fast_info
        # fast_info doesn't have sector; fall back to info
        full = yf.Ticker(ticker + ".NS").info
        sector = full.get("sector", "Unknown") or "Unknown"
        return ticker, _YF_SECTOR_MAP.get(sector, sector)
    except Exception:
        return ticker, "Unknown"


def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def save_cache(cache: dict[str, str]):
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def patch_universe():
    um = pd.read_parquet(UNIVERSE_PATH)
    print(f"Universe: {len(um)} tickers  (T1={int((um['Tier']=='T1').sum())} T2={int((um['Tier']=='T2').sum())} T3={int((um['Tier']=='T3').sum())})")

    # Load previously cached sectors to avoid re-fetching
    cache = load_cache()
    needs_fetch = [t for t in um["Ticker"].tolist() if t not in cache]

    if needs_fetch:
        print(f"Fetching sector for {len(needs_fetch)} tickers  (cached: {len(cache)})  workers={MAX_WORKERS}")
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(fetch_sector, t): t for t in needs_fetch}
            for fut in as_completed(futs):
                ticker, sector = fut.result()
                cache[ticker] = sector
                done += 1
                if done % 50 == 0 or done == len(needs_fetch):
                    pct = done / len(needs_fetch) * 100
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  {done}/{len(needs_fetch)}  ({pct:.0f}%)  {elapsed:.0f}s  {rate:.1f}/s")
                    save_cache(cache)  # save progress incrementally
        save_cache(cache)
        print(f"  Done in {time.time()-t0:.0f}s")
    else:
        print("  All sectors already cached.")

    # Apply to dataframe
    um["Sector"] = um["Ticker"].map(cache).fillna("Unknown")

    # Summary
    dist = um["Sector"].value_counts()
    print(f"\nSector distribution ({um['Sector'].nunique()} unique):")
    for s, n in dist.head(15).items():
        print(f"  {s:<28} {n}")

    um.to_parquet(UNIVERSE_PATH, index=False)
    print(f"\nSaved: {UNIVERSE_PATH}")
    return um


def regenerate_sector_momentum(um: pd.DataFrame):
    """Recompute sector_momentum.parquet from updated universe_master."""
    print("\nRegenerating sector_momentum.parquet ...")

    raw_dir = BASE / "raw_prices"
    if not raw_dir.exists():
        # Try alternate path
        raw_dir = BASE / "data" / "raw"
    if not raw_dir.exists():
        print("  raw_prices directory not found — skipping sector_momentum regen")
        print("  Run Layer1-Daily to regenerate it.")
        return

    t1t2 = um[um["Tier"].isin(["T1", "T2"])]
    sector_groups = t1t2.groupby("Sector")["Ticker"].apply(list).to_dict()
    sector_groups.pop("Unknown", None)

    sec_rows = []
    for sector, tickers in sector_groups.items():
        rets_all = []
        for t in tickers:
            # Try to load returns from raw parquet
            for fname in [f"{t}.parquet", f"{t}.NS.parquet"]:
                p = raw_dir / fname
                if p.exists():
                    try:
                        df = pd.read_parquet(p)
                        close_col = next((c for c in df.columns if c.lower() == "close"), None)
                        if close_col and len(df) >= MOMENTUM_WINDOW + 5:
                            r = df[close_col].pct_change().dropna().tail(MOMENTUM_WINDOW + 5)
                            if len(r) >= MOMENTUM_WINDOW:
                                rets_all.append(r.tail(MOMENTUM_WINDOW))
                    except Exception:
                        pass
                    break

        if not rets_all:
            continue
        aligned = pd.concat(rets_all, axis=1).mean(axis=1)
        rolling_ret = float((1 + aligned).prod() - 1)
        sec_rows.append({
            "Sector": sector,
            "Rolling20dReturn": rolling_ret,
            "AsOf": date.today(),
        })

    if not sec_rows:
        print("  Could not compute sector returns — no raw price files found.")
        print("  Run Layer1-Daily after patching sectors to regenerate.")
        return

    df = pd.DataFrame(sec_rows)
    df["Rank"] = df["Rolling20dReturn"].rank(ascending=False).astype(int)
    df = df.sort_values("Rank")
    df.to_parquet(SECTOR_MOM_PATH, index=False)
    print(f"  Saved: {SECTOR_MOM_PATH}  ({len(df)} sectors)")
    print(df[["Rank", "Sector", "Rolling20dReturn"]].to_string(index=False))


if __name__ == "__main__":
    print("=" * 60)
    print("  SECTOR PATCH UTILITY")
    print("=" * 60)
    um = patch_universe()
    regenerate_sector_momentum(um)
    print("\nDone. Run layer1_daily.py to get fresh sector momentum.")
