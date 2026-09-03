# -*- coding: utf-8 -*-
"""
CLAUDE QUANTS -- Independent Paper-Trading Tracker
====================================================

Fully independent of the parent production pipeline's f1_paper_tracker.py
(D:\\MBA\\STOCK MARKET RESEARCH\\NSE quants py\\f1_paper_tracker.py) -- that
file was read only for the pattern (duplicate-open guard, cooldown, direction-
aware SL/target-hit detection, day-0 lookahead guard, report shape). Nothing
is imported from it or from anywhere in the parent folder. This file reads
and writes ONLY inside this ("claude quants") folder: paper_trades.csv here
is a separate ledger from the parent's paper_trades.csv.

Usage
-----
  python paper_tracker.py log      # log today's FIRE verdicts from layer_combiner
  python paper_tracker.py update   # check open trades against live OHLC, close hits/expiries
  python paper_tracker.py report   # win rate / avg win / avg loss / R:R / expectancy
  python paper_tracker.py daily    # log -> update -> report, in sequence

Input contract
--------------
Reads `combiner_output.parquet` (written by layer_combiner.run_combiner()) --
columns: ticker, sector, tier, composite_score, contributing_signals,
n_signals, direction, verdict. Only verdict == 'FIRE' rows are logged as new
trades.

Entry/stop/target/qty are NOT part of layer_combiner's output (that layer
only scores and classifies -- no dedicated position-sizing/risk layer exists
yet in this replica). This file:
  - fetches the live close via yfinance as entry_price
  - sizes stop/target off a real ATR(14)-based % distance (atr_stop_pct(),
    added 2026-07-18 -- 2x the ticker's own 14-day ATR as a % of price,
    computed from the local data/raw/{ticker}.parquet High/Low/Close cache,
    clipped to [MIN_STOP_PCT, MAX_STOP_PCT] to avoid pathological cases on
    illiquid/near-zero-vol names) at a fixed reward:risk of TARGET_RR. Falls
    back to the flat STOP_PCT_BY_TIER guess only if local OHLC history is
    unavailable for that ticker (e.g. cache miss).
  - sizes qty off a fixed rupee risk-per-trade (RISK_PER_TRADE_INR)
qty sizing is still the remaining placeholder -- swap RISK_PER_TRADE_INR for
real portfolio-level capital allocation / max-concurrent-position rules
later without touching anything else in this file.

CSV schema (paper_trades.csv, this folder)
-------------------------------------------
trade_id, signal_date, ticker, sector, tier, entry_price, stop_price,
target_price, qty, composite_score, contributing_signals, direction, status,
exit_date, exit_price, exit_reason, pnl_pts, pnl_pct, rr_achieved, days_held
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from layer_combiner import COMBINER_OUTPUT_PATH

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV  = os.path.join(PROJECT_DIR, "paper_trades.csv")
RAW_DIR     = os.path.join(PROJECT_DIR, "data", "raw")

TODAY = date.today()

COLS = [
    "trade_id", "signal_date", "ticker", "sector", "tier",
    "entry_price", "stop_price", "target_price", "qty",
    "composite_score", "contributing_signals", "direction",
    "status", "exit_date", "exit_price", "exit_reason",
    "pnl_pts", "pnl_pct", "rr_achieved", "days_held",
]

# -- position-sizing config ---------------------------------------------------
STOP_PCT_BY_TIER = {"T1": 0.03, "T2": 0.045, "T3": 0.06}   # fallback only -- see atr_stop_pct()
DEFAULT_STOP_PCT = 0.05
ATR_WINDOW       = 14
ATR_MULTIPLIER   = 2.0      # stop distance = ATR_MULTIPLIER x 14-day ATR, as % of entry
MIN_STOP_PCT     = 0.02     # clip: avoid near-zero stops on very low-vol names
MAX_STOP_PCT     = 0.12     # clip: avoid absurd stops on very high-vol/illiquid names
TARGET_RR        = 2.0      # target distance = stop distance * TARGET_RR
RISK_PER_TRADE_INR = 2000.0 # fixed rupee risk per trade used to size qty (still placeholder --
                             # see module docstring; no portfolio-level allocation yet)


# ===========================================================================
# ATR-BASED STOP SIZING (added 2026-07-18, replaces flat tier-% placeholder)
# ===========================================================================
def atr_stop_pct(ticker: str) -> float | None:
    """14-day ATR as a % of the latest close, from the local price cache
    (same data/raw/{ticker}.parquet used by validation_framework.py -- this
    is historical data through the last refresh, not today's live bar, which
    is fine since ATR is a slow-moving volatility estimate). Returns None if
    the ticker has no local cache or too little history, so the caller can
    fall back to STOP_PCT_BY_TIER."""
    path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=["Date", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").tail(ATR_WINDOW + 5)
        if len(df) < ATR_WINDOW + 1:
            return None
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(ATR_WINDOW).mean().iloc[-1]
        last_close = close.iloc[-1]
        if not np.isfinite(atr) or last_close <= 0:
            return None
        return float(atr / last_close)
    except Exception:
        return None


def resolve_stop_pct(ticker: str, tier: str) -> float:
    """ATR-based stop distance (2x 14d ATR%), clipped to a sane range;
    falls back to the flat tier-based guess only if ATR is unavailable."""
    atr_pct = atr_stop_pct(ticker)
    if atr_pct is None or atr_pct <= 0:
        return STOP_PCT_BY_TIER.get(tier, DEFAULT_STOP_PCT)
    return float(np.clip(atr_pct * ATR_MULTIPLIER, MIN_STOP_PCT, MAX_STOP_PCT))

# -- cooldown rule ------------------------------------------------------------
COOLDOWN_DAYS = 3            # block re-entry into a ticker for this many days after a LOSS

# -- hold-time rule -------------------------------------------------------------
MAX_HOLD_DAYS = 15           # single flat hold limit for this system (no regime-specific hold lengths)


# ===========================================================================
# DUPLICATE-OPEN GUARD  (logic reused from f1_paper_tracker.py, generic)
# ===========================================================================
def check_already_open(ticker: str, trades_df: pd.DataFrame) -> tuple[bool, str]:
    """Block a new trade if `ticker` already has an OPEN position."""
    if trades_df is None or trades_df.empty:
        return False, ""
    open_trades = trades_df[
        (trades_df["ticker"].str.upper() == ticker.upper()) &
        (trades_df["status"].str.upper() == "OPEN")
    ]
    if not open_trades.empty:
        existing = open_trades.iloc[0]
        reason = (f"DUPLICATE BLOCK: {ticker} already has an OPEN trade "
                  f"(entry={existing['entry_price']}, id={existing['trade_id']}). "
                  f"Close or exit that position first.")
        return True, reason
    return False, ""


# ===========================================================================
# COOLDOWN GUARD  (logic reused from f1_paper_tracker.py, generic)
# ===========================================================================
def check_cooldown(ticker: str, trades_df: pd.DataFrame) -> tuple[bool, str]:
    """Block re-entry into `ticker` for COOLDOWN_DAYS after its most recent LOSS."""
    if trades_df is None or trades_df.empty:
        return False, ""
    losses = trades_df[
        (trades_df["ticker"].str.upper() == ticker.upper()) &
        (trades_df["status"].str.upper() == "LOSS")
    ].copy()
    if losses.empty:
        return False, ""

    losses["exit_date"] = pd.to_datetime(losses["exit_date"], errors="coerce")
    most_recent_loss = losses["exit_date"].max()
    if pd.isna(most_recent_loss):
        return False, ""

    days_since = (pd.Timestamp.today().normalize() - most_recent_loss.normalize()).days
    if days_since < COOLDOWN_DAYS:
        remaining = COOLDOWN_DAYS - days_since
        reason = (f"COOLDOWN BLOCK: {ticker} had a LOSS {days_since}d ago "
                  f"({most_recent_loss.date()}). Wait {remaining} more day(s) before re-entry.")
        return True, reason
    return False, ""


# ===========================================================================
# CSV I/O
# ===========================================================================
def load_trades() -> pd.DataFrame:
    if os.path.exists(TRADES_CSV):
        df = pd.read_csv(TRADES_CSV, dtype={"trade_id": str})
        # Force text columns to object dtype -- an all-empty column infers as
        # float64, and writing a string into it later raises LossySetitemError.
        for col in ("exit_date", "exit_reason", "contributing_signals"):
            if col in df.columns:
                df[col] = df[col].astype(object)
        return df
    return pd.DataFrame(columns=COLS)


def save_trades(df: pd.DataFrame):
    df.to_csv(TRADES_CSV, index=False)
    print(f"  Saved {len(df)} trades -> {os.path.basename(TRADES_CSV)}")


def make_trade_id(signal_date: str, ticker: str) -> str:
    return f"{signal_date}_{ticker}"


# ===========================================================================
# PRICE FETCH  (logic reused from f1_paper_tracker.py: yfinance, ticker+".NS")
# ===========================================================================
def fetch_ohlc(ticker: str):
    """Returns (high, low, close) for the latest session, or (None, None, None) on failure."""
    try:
        data = yf.download(ticker + ".NS", period="5d", progress=False, auto_adjust=True)
        if data.empty:
            return None, None, None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        high  = float(data["High"].dropna().iloc[-1])
        low   = float(data["Low"].dropna().iloc[-1])
        close = float(data["Close"].dropna().iloc[-1])
        return high, low, close
    except Exception:
        return None, None, None


# ===========================================================================
# COMMAND 1: LOG  -- read layer_combiner's FIRE verdicts, open new trades
# ===========================================================================
def log_new_trades():
    if not os.path.exists(COMBINER_OUTPUT_PATH):
        print(f"ERROR: {os.path.basename(COMBINER_OUTPUT_PATH)} not found. "
              f"Run layer_combiner.py (via orchestrator.py) first.")
        return

    combiner = pd.read_parquet(COMBINER_OUTPUT_PATH)
    fire = combiner[combiner["verdict"].str.upper() == "FIRE"].copy()
    if fire.empty:
        print("No FIRE verdicts in the latest combiner output.")
        return

    trades = load_trades()
    signal_date = str(TODAY)
    new_count = 0

    for _, row in fire.iterrows():
        ticker = row["ticker"]
        tid = make_trade_id(signal_date, ticker)

        if len(trades) > 0 and tid in trades["trade_id"].values:
            print(f"  SKIP (already logged): {ticker}")
            continue

        blocked, reason = check_cooldown(ticker, trades)
        if blocked:
            print(f"\n  {reason}\n  Trade NOT logged.\n")
            continue

        dup_blocked, dup_reason = check_already_open(ticker, trades)
        if dup_blocked:
            print(f"\n  {dup_reason}\n  Trade NOT logged.\n")
            continue

        _, _, close = fetch_ohlc(ticker)
        if close is None:
            print(f"  SKIP (price fetch failed): {ticker}")
            continue

        direction = str(row.get("direction", "LONG")).upper()
        tier = str(row.get("tier", "T3"))
        stop_pct = resolve_stop_pct(ticker, tier)
        entry = round(float(close), 2)

        if direction == "SHORT":
            stop_price   = round(entry * (1 + stop_pct), 2)
            target_price = round(entry * (1 - stop_pct * TARGET_RR), 2)
        else:
            direction = "LONG"
            stop_price   = round(entry * (1 - stop_pct), 2)
            target_price = round(entry * (1 + stop_pct * TARGET_RR), 2)

        risk_per_share = abs(entry - stop_price)
        qty = max(1, int(RISK_PER_TRADE_INR // risk_per_share)) if risk_per_share > 0 else 1

        new_row = {
            "trade_id": tid,
            "signal_date": signal_date,
            "ticker": ticker,
            "sector": row.get("sector", ""),
            "tier": tier,
            "entry_price": entry,
            "stop_price": stop_price,
            "target_price": target_price,
            "qty": qty,
            "composite_score": round(float(row.get("composite_score", 0.0)), 4),
            "contributing_signals": row.get("contributing_signals", ""),
            "direction": direction,
            "status": "OPEN",
            "exit_date": "",
            "exit_price": "",
            "exit_reason": "",
            "pnl_pts": "",
            "pnl_pct": "",
            "rr_achieved": "",
            "days_held": "",
        }
        trades = pd.concat([trades, pd.DataFrame([new_row])], ignore_index=True)
        new_count += 1
        print(f"  LOGGED [{direction}]: {ticker:12s} E:{entry:.2f}  SL:{stop_price:.2f}  "
              f"T:{target_price:.2f}  Qty:{qty}  Score:{row.get('composite_score', 0):.3f}  "
              f"Signals:{row.get('contributing_signals', '')}")

    save_trades(trades)
    print(f"\n  {new_count} new trade(s) logged. Total in tracker: {len(trades)}")


# ===========================================================================
# COMMAND 2: UPDATE  -- direction-aware SL/target-hit detection + day-0 guard
# (logic reused from f1_paper_tracker.py)
# ===========================================================================
def update_open_trades():
    trades = load_trades()
    open_trades = trades[trades["status"] == "OPEN"]
    if open_trades.empty:
        print("No open trades to update.")
        return

    print(f"Checking {len(open_trades)} open trade(s)...")
    updated = 0
    open_pnl_pcts = []

    for idx, row in open_trades.iterrows():
        ticker = row["ticker"]
        entry  = float(row["entry_price"])
        stop   = float(row["stop_price"])
        target = float(row["target_price"])
        signal_dt = pd.to_datetime(row["signal_date"]).date()
        days_held = (TODAY - signal_dt).days

        direction = str(row.get("direction", "LONG")).upper()
        is_short  = (direction == "SHORT")

        high, low, close = fetch_ohlc(ticker)
        if close is None:
            print(f"  {ticker:12s} price fetch failed - skipping")
            continue

        # Day 0 -- allow SL exit but never a same-day WIN (avoids lookahead illusion)
        if days_held < 1:
            risk_pts0 = (stop - entry) if is_short else (entry - stop)
            sl_hit_d0 = (high >= stop) if is_short else (low <= stop)
            pnl_open  = round((entry - close) if is_short else (close - entry), 2)
            pnl_pct_open = round(pnl_open / entry * 100, 2)
            if sl_hit_d0:
                pnl_pts0 = round((entry - stop) if is_short else (stop - entry), 2)
                pnl_pct0 = round(pnl_pts0 / entry * 100, 2)
                rr0 = round(pnl_pts0 / risk_pts0, 2) if risk_pts0 > 0 else 0
                trades.at[idx, "status"] = "LOSS"
                trades.at[idx, "exit_date"] = str(TODAY)
                trades.at[idx, "exit_price"] = stop
                trades.at[idx, "exit_reason"] = "SL_HIT (day0)"
                trades.at[idx, "pnl_pts"] = pnl_pts0
                trades.at[idx, "pnl_pct"] = pnl_pct0
                trades.at[idx, "rr_achieved"] = rr0
                trades.at[idx, "days_held"] = 0
                updated += 1
                print(f"  {ticker:12s} [{direction}] LOSS | SL hit on entry day! "
                      f"Exit:{stop:.2f}  PnL:{pnl_pts0:+.2f}pts ({pnl_pct0:+.2f}%)  [SL_HIT day0]")
            else:
                open_pnl_pcts.append(pnl_pct_open)
                print(f"  {ticker:12s} [{direction}] DAY0 | CMP:{close:.2f}  "
                      f"OpenPnL:{pnl_open:+.2f}pts ({pnl_pct_open:+.2f}%)  [no exit today]")
            continue

        risk_pts   = (stop - entry)   if is_short else (entry - stop)

        def pnl(exit_px):
            pts = (entry - exit_px) if is_short else (exit_px - entry)
            pct = round(pts / entry * 100, 2)
            rr  = round(pts / risk_pts, 2) if risk_pts > 0 else 0
            return round(pts, 2), pct, rr

        sl_hit     = (high >= stop)   if is_short else (low  <= stop)
        target_hit = (low  <= target) if is_short else (high >= target)

        status = "OPEN"
        exit_reason = ""
        exit_price = ""
        pnl_pts = pnl_pct = rr_achieved = ""

        if sl_hit and target_hit:
            status, exit_reason, exit_price = "LOSS", "SL_HIT (same candle as target)", stop
            pnl_pts, pnl_pct, rr_achieved = pnl(stop)
        elif target_hit:
            status, exit_reason, exit_price = "WIN", "TARGET_HIT (intraday)", target
            pnl_pts, pnl_pct, rr_achieved = pnl(target)
        elif sl_hit:
            status, exit_reason, exit_price = "LOSS", "SL_HIT (intraday)", stop
            pnl_pts, pnl_pct, rr_achieved = pnl(stop)
        elif days_held >= MAX_HOLD_DAYS:
            status, exit_reason, exit_price = "EXPIRED", "MAX_HOLD", close
            pnl_pts, pnl_pct, rr_achieved = pnl(close)

        if status != "OPEN":
            trades.at[idx, "status"] = status
            trades.at[idx, "exit_date"] = str(TODAY)
            trades.at[idx, "exit_price"] = exit_price
            trades.at[idx, "exit_reason"] = exit_reason
            trades.at[idx, "pnl_pts"] = pnl_pts
            trades.at[idx, "pnl_pct"] = pnl_pct
            trades.at[idx, "rr_achieved"] = rr_achieved
            trades.at[idx, "days_held"] = days_held
            updated += 1
            print(f"  {ticker:12s} {status:8s} | Exit:{exit_price:.2f}  PnL:{pnl_pts:+.2f}pts "
                  f"({pnl_pct:+.2f}%)  RR:{rr_achieved}  Days:{days_held}  [{exit_reason}]")
        else:
            pnl_open = round((entry - close) if is_short else (close - entry), 2)
            pnl_pct_open = round(pnl_open / entry * 100, 2)
            open_pnl_pcts.append(pnl_pct_open)
            print(f"  {ticker:12s} OPEN | CMP:{close:.2f}  PnL:{pnl_open:+.2f}pts "
                  f"({pnl_pct_open:+.2f}%)  Days:{days_held}/{MAX_HOLD_DAYS}")

    save_trades(trades)
    if open_pnl_pcts:
        avg_open = sum(open_pnl_pcts) / len(open_pnl_pcts)
        print(f"\n  -- Avg open-trade PnL : {avg_open:+.2f}%  ({len(open_pnl_pcts)} positions)")
    print(f"\n  {updated} trade(s) closed today.")


# ===========================================================================
# COMMAND 3: REPORT  (shape kept close to f1_paper_tracker.py's cmd_report --
# that reporting logic was sound; only the breakdown dimension changes from
# 'primary_rule' to 'contributing signal', since this schema has no single
# rule flag, just a comma-joined list of which signals fired)
# ===========================================================================
def generate_report():
    trades = load_trades()
    if trades.empty:
        print("No trades logged yet.")
        return

    closed = trades[trades["status"].isin(["WIN", "LOSS", "EXPIRED"])].copy()
    open_t = trades[trades["status"] == "OPEN"]

    print()
    print("=" * 65)
    print("  CLAUDE QUANTS PAPER TRADING REPORT -", str(TODAY))
    print("=" * 65)
    print(f"  Total signals logged : {len(trades)}")
    print(f"  Open trades          : {len(open_t)}")
    print(f"  Closed trades        : {len(closed)}")

    if closed.empty:
        print("\n  No closed trades yet - keep paper trading.")
        return

    closed["pnl_pct"]     = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed["rr_achieved"] = pd.to_numeric(closed["rr_achieved"], errors="coerce")
    closed["days_held"]   = pd.to_numeric(closed["days_held"], errors="coerce")

    wins   = closed[closed["status"] == "WIN"]
    losses = closed[closed["status"] == "LOSS"]
    exp    = closed[closed["status"] == "EXPIRED"]

    win_rate = len(wins) / len(closed) * 100
    avg_win  = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    avg_rr   = closed["rr_achieved"].mean()
    avg_hold = closed["days_held"].mean()
    total_pnl = closed["pnl_pct"].sum()

    print()
    print("  OVERALL PERFORMANCE")
    print(f"  {'Win Rate':<25} {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L / {len(exp)}EXP)")
    print(f"  {'Avg Win':<25} {avg_win:+.2f}%")
    print(f"  {'Avg Loss':<25} {avg_loss:+.2f}%")
    print(f"  {'Avg R:R Achieved':<25} {avg_rr:.2f}")
    print(f"  {'Avg Hold Days':<25} {avg_hold:.1f}")
    print(f"  {'Total PnL (sum %)':<25} {total_pnl:+.2f}%")

    if len(losses) > 0 and avg_loss != 0:
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        print(f"  {'Expectancy per trade':<25} {expectancy:+.2f}%")

    # Per contributing-signal breakdown (multi-label: a trade can count under >1 signal)
    print()
    print("  BY CONTRIBUTING SIGNAL")
    print(f"  {'Signal':<20} {'Trades':<8} {'WinRate':<10} {'AvgPnL':<10} {'AvgRR'}")
    exploded = closed.assign(
        _sig=closed["contributing_signals"].fillna("").astype(str).str.split(",")
    ).explode("_sig")
    exploded["_sig"] = exploded["_sig"].str.strip()
    exploded = exploded[exploded["_sig"] != ""]
    for sig, sub in exploded.groupby("_sig"):
        wr = len(sub[sub["status"] == "WIN"]) / len(sub) * 100
        ap = sub["pnl_pct"].mean()
        arr = sub["rr_achieved"].mean()
        print(f"  {sig:<20} {len(sub):<8} {wr:<10.1f} {ap:<+10.2f} {arr:.2f}")

    # Per direction breakdown
    print()
    print("  BY DIRECTION")
    for direction in closed["direction"].dropna().unique():
        sub = closed[closed["direction"] == direction]
        wr = len(sub[sub["status"] == "WIN"]) / len(sub) * 100
        ap = sub["pnl_pct"].mean()
        print(f"  {direction:<12} {len(sub)} trades  WinRate:{wr:.1f}%  AvgPnL:{ap:+.2f}%")

    # Score vs outcome IC check
    print()
    print("  SCORE vs OUTCOME (IC CHECK)")
    closed["outcome_bin"] = (closed["status"] == "WIN").astype(int)
    if closed["composite_score"].nunique() > 1:
        ic = closed["composite_score"].corr(closed["pnl_pct"])
        print(f"  composite_score vs pnl_pct  IC = {ic:.3f}")
        ic2 = closed["composite_score"].corr(closed["outcome_bin"])
        print(f"  composite_score vs win/loss IC = {ic2:.3f}")
    else:
        print("  Not enough variance in scores yet.")

    # Last 10 closed trades
    print()
    print("  LAST 10 CLOSED TRADES")
    print(f"  {'Date':<12} {'Ticker':<12} {'Dir':<6} {'Status':<8} {'PnL%':<10} {'RR':<6} {'Days'}")
    recent = closed.sort_values("exit_date", ascending=False).head(10)
    for _, r in recent.iterrows():
        days = int(r["days_held"]) if pd.notna(r["days_held"]) else "-"
        print(f"  {str(r['exit_date']):<12} {r['ticker']:<12} {r['direction']:<6} {r['status']:<8} "
              f"{r['pnl_pct']:+.2f}%     {r['rr_achieved']:<6} {days}")

    print()
    print("=" * 65)

    # Readiness check (same bar as the pattern this was forked from)
    print()
    print("  LIVE TRADING READINESS")
    ready = True
    checks = [
        (len(closed) >= 30, f"30 closed trades ({len(closed)}/30)"),
        (win_rate >= 50,    f"Win rate >= 50% ({win_rate:.1f}%)"),
        (avg_rr >= 1.5,     f"Avg R:R >= 1.5 ({avg_rr:.2f})"),
    ]
    for passed, label in checks:
        mark = "[PASS]" if passed else "[FAIL]"
        print(f"  {mark} {label}")
        if not passed:
            ready = False

    print()
    print("  READY for next phase" if ready else "  Not ready - continue paper trading")
    print()


# ===========================================================================
# COMMAND 4: DAILY
# ===========================================================================
def run_daily():
    print("\n-- STEP 1: LOG TODAY'S FIRE VERDICTS --")
    log_new_trades()
    print("\n-- STEP 2: UPDATE OPEN TRADES --")
    update_open_trades()
    print("\n-- STEP 3: PERFORMANCE REPORT --")
    generate_report()


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    dispatch = {
        "log": log_new_trades,
        "update": update_open_trades,
        "report": generate_report,
        "daily": run_daily,
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}")
        print("Usage: python paper_tracker.py [log|update|report|daily]")
        sys.exit(1)
    dispatch[cmd]()
