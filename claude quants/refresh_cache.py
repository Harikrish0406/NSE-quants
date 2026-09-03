# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS -- incremental price-cache refresh
================================================

The replica scores off `data/raw/{ticker}.parquet` via
`validation_framework._get_cached_series`. That loader (and `fetch_prices`)
only ever yfinance-fetches tickers that are *entirely missing* from the
cache -- a ticker that is present but months stale is used as-is. So the
cache silently rots and `live_scoring.py` ends up scoring today off old
bars. This script is the missing piece: for every ticker in the universe it
finds the last cached bar and pulls only the gap from Yahoo, appending rows
in the exact existing schema (Date, Close, High, Low, Open, Volume, Ticker).

Run:  python refresh_cache.py            # refresh everything stale
      python refresh_cache.py --full     # ignore cache, re-pull full history
      python refresh_cache.py --max 50   # cap ticker count (debug)
"""
import argparse
import glob
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd

from validation_framework import RAW_DIR, PROJECT_DIR, log

SCHEMA   = ["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]
BATCH    = 80           # tickers per yfinance call
BATCH_PAUSE = 2.0       # seconds between batches (Yahoo rate-limits hard)
MAX_RETRY   = 4         # per-batch retries on download failure / rate limit
FULL_START = "2005-01-01"


def _download(yf, ns, start, end):
    """yf.download with exponential backoff on rate-limit / transient errors."""
    delay = 5.0
    for attempt in range(1, MAX_RETRY + 1):
        try:
            raw = yf.download(ns, start=start, end=end, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
            if raw is not None and not raw.empty:
                return raw
            log.warning("  empty result (attempt %d/%d) -- backing off %.0fs",
                        attempt, MAX_RETRY, delay)
        except Exception as e:
            log.warning("  download error (attempt %d/%d): %s -- backing off %.0fs",
                        attempt, MAX_RETRY, e, delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return None


def _universe():
    """Union of universe_master tickers and whatever raw parquets already exist."""
    tickers = set()
    um = os.path.join(PROJECT_DIR, "universe_master.parquet")
    if os.path.exists(um):
        tickers |= set(pd.read_parquet(um)["Ticker"].astype(str))
    tickers |= {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(RAW_DIR, "*.parquet"))}
    return sorted(t for t in tickers if t and t.upper() == t)


def _last_cached_date(ticker):
    path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_parquet(path, columns=["Date"])
        return pd.to_datetime(d["Date"]).max()
    except Exception:
        return None


def _write_merged(ticker, new_rows):
    """Append new_rows (schema-shaped) to the ticker's parquet, dedupe on Date."""
    path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    new_rows = new_rows[SCHEMA].copy()
    new_rows["Date"] = pd.to_datetime(new_rows["Date"])
    if os.path.exists(path):
        old = pd.read_parquet(path)
        old["Date"] = pd.to_datetime(old["Date"])
        merged = pd.concat([old, new_rows], ignore_index=True)
    else:
        merged = new_rows
    merged = (merged.dropna(subset=["Close"])
                    .drop_duplicates(subset=["Date"], keep="last")
                    .sort_values("Date")
                    .reset_index(drop=True))
    merged["Ticker"] = ticker
    merged.to_parquet(path, index=False)
    return len(merged)


def _slice_ticker(raw, tkr_ns):
    """Pull one ticker's OHLCV frame out of a yf.download(group_by='ticker') result."""
    if isinstance(raw.columns, pd.MultiIndex):
        if tkr_ns not in raw.columns.get_level_values(0):
            return None
        sub = raw[tkr_ns].copy()
    else:
        sub = raw.copy()                       # single-ticker batch
    sub = sub.reset_index().rename(columns={"index": "Date"})
    if "Date" not in sub.columns and "Datetime" in sub.columns:
        sub = sub.rename(columns={"Datetime": "Date"})
    need = {"Date", "Close", "High", "Low", "Open", "Volume"}
    if not need.issubset(sub.columns):
        return None
    return sub[list(need)].dropna(subset=["Close"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignore cache, re-pull full history")
    ap.add_argument("--max", type=int, default=0, help="cap ticker count (debug)")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed -- `pip install yfinance`"); sys.exit(1)

    os.makedirs(RAW_DIR, exist_ok=True)
    today = date.today()
    universe = _universe()
    if args.max:
        universe = universe[:args.max]
    log.info("Refreshing %d tickers  (cache dir: %s)", len(universe), RAW_DIR)

    # bucket tickers by how far back we need to pull
    jobs = []                                   # (ticker, start_str)
    fresh = 0
    for t in universe:
        if args.full:
            jobs.append((t, FULL_START)); continue
        last = _last_cached_date(t)
        if last is None:
            jobs.append((t, FULL_START))
        elif last.date() >= today - timedelta(days=1):
            fresh += 1                          # already current
        else:
            jobs.append((t, (last + timedelta(days=1)).strftime("%Y-%m-%d")))

    log.info("  %d already fresh, %d to pull", fresh, len(jobs))
    updated = failed = rows_added = 0
    t_start = time.time()

    for i in range(0, len(jobs), BATCH):
        batch = jobs[i:i + BATCH]
        # one download call spanning the batch's earliest needed start
        start = min(s for _, s in batch)
        ns = [t + ".NS" for t, _ in batch]
        raw = _download(yf, ns, start,
                        (today + timedelta(days=1)).strftime("%Y-%m-%d"))
        if raw is None:
            log.warning("  batch %d-%d gave up after %d retries", i, i + len(batch), MAX_RETRY)
            failed += len(batch)
            continue

        for t, s in batch:
            sub = _slice_ticker(raw, t + ".NS")
            if sub is None or sub.empty:
                failed += 1
                continue
            sub = sub[pd.to_datetime(sub["Date"]) >= pd.to_datetime(s)]
            if sub.empty:
                continue
            sub["Ticker"] = t
            try:
                _write_merged(t, sub)
                updated += 1
                rows_added += len(sub)
            except Exception as e:
                log.warning("  write failed for %s: %s", t, e)
                failed += 1

        done = min(i + BATCH, len(jobs))
        log.info("  [%d/%d]  updated=%d  failed=%d  rows+=%d  (%.0fs)",
                 done, len(jobs), updated, failed, rows_added, time.time() - t_start)
        if done < len(jobs):
            time.sleep(BATCH_PAUSE)

    log.info("=" * 64)
    log.info("Cache refresh done in %.1f min  |  updated %d, fresh %d, failed %d, rows added %d",
             (time.time() - t_start) / 60, updated, fresh, failed, rows_added)
    log.info("=" * 64)
    # non-zero exit only if we got basically nothing
    if updated == 0 and fresh == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
