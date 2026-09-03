"""
F1 PAPER TRADING TRACKER - NSE Decision Engine
================================================
Usage:
  1. Log signals daily (after layer5 run):
       python f1_paper_tracker.py log

  2. Update outcomes (run daily after market close):
       python f1_paper_tracker.py update

  3. Print performance report:
       python f1_paper_tracker.py report

  4. All in one (log + update + report):
       python f1_paper_tracker.py daily

CSV file: paper_trades.csv (auto-created in same folder)
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, date, timedelta

# -- NO DUPLICATE OPEN RULE ------------------------------------------------
def check_already_open(ticker: str, trades_df) -> tuple[bool, str]:
    """Block new trade if ticker already has an OPEN position."""
    if trades_df is None or trades_df.empty:
        return False, ""

    open_trades = trades_df[
        (trades_df["ticker"].str.upper() == ticker.upper()) &
        (trades_df["status"].str.upper() == "OPEN")
    ]

    if not open_trades.empty:
        existing = open_trades.iloc[0]
        reason = (
            f"DUPLICATE BLOCK: {ticker} already has an OPEN trade "
            f"(entry={existing['entry_price']}, "
            f"id={existing['trade_id']}). "
            f"Close or exit that position first."
        )
        return True, reason

    return False, ""
# -------------------------------------------------------------------------

# -- COOLDOWN RULE ---------------------------------------------------------
COOLDOWN_DAYS = 3  # block re-entry for this many days after a LOSS

def check_cooldown(ticker: str, trades_df) -> tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str)
    Blocked = True means DO NOT enter this trade.
    """
    if trades_df is None or trades_df.empty:
        return False, ""

    losses = trades_df[
        (trades_df["ticker"].str.upper() == ticker.upper()) &
        (trades_df["status"].str.upper() == "LOSS")
    ].copy()

    if losses.empty:
        return False, ""

    losses["exit_date"] = pd.to_datetime(losses["exit_date"], errors="coerce")
    most_recent_loss    = losses["exit_date"].max()

    if pd.isna(most_recent_loss):
        return False, ""

    days_since = (pd.Timestamp.today().normalize() - most_recent_loss.normalize()).days

    if days_since < COOLDOWN_DAYS:
        remaining = COOLDOWN_DAYS - days_since
        reason = (
            f"COOLDOWN BLOCK: {ticker} had a LOSS {days_since}d ago "
            f"({most_recent_loss.date()}).  "
            f"Wait {remaining} more day(s) before re-entry."
        )
        return True, reason

    return False, ""

# -------------------------------------------------------------------------


# -- CONFIG --------------------------------------------------------------------
BASE_PATH      = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
L4_PARQUET     = BASE_PATH / "nse_layer4_portfolio.parquet"
TRADES_CSV     = BASE_PATH / "paper_trades.csv"
TRADES_V2_CSV  = BASE_PATH / "paper_trades_v2.csv"
TRADES_V2_XLSX = BASE_PATH / "paper_trades_v2.xlsx"
V2_START       = date(2026, 6, 3)   # date fixes went live
TODAY          = date.today()

# -- COLUMNS -------------------------------------------------------------------
COLS = [
    "trade_id", "signal_date", "ticker", "companyname", "tier", "sector",
    "entry_price", "stop_price", "target_price", "qty",
    "composite_score", "final_score", "fractal_score",
    "rule_c_fires", "rule_d_fires", "rule_e_fires",
    "rule_c_entry_score", "rule_d_entry_score", "rule_e_entry_score",
    "primary_rule", "direction",
    "regime", "vix", "r2_grade", "hurst",
    "pipeline_ver",     # v1 = old pipeline | v2 = fixed pipeline (TS stops + SHORT support)
    "status",
    "exit_date", "exit_price", "exit_reason",
    "pnl_pts", "pnl_pct", "rr_achieved",
    "days_held", "notes"
]

# -- HELPERS -------------------------------------------------------------------
def load_trades() -> pd.DataFrame:
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV, dtype={"trade_id": str})
        # Force text columns to object dtype -- when all-empty, pandas infers
        # float64, and later writing a string (e.g. exit_date/exit_reason on
        # the first trade to close) raises pandas.errors.LossySetitemError.
        for _col in ("exit_date", "exit_reason", "notes"):
            if _col in df.columns:
                df[_col] = df[_col].astype(object)
        # Backfill pipeline_ver for old trades that don't have the column yet
        if "pipeline_ver" not in df.columns:
            df["pipeline_ver"] = "v1"
        else:
            df["pipeline_ver"] = df["pipeline_ver"].fillna("v1")
            # Auto-tag as v2 if signal_date >= V2_START
            try:
                parsed = pd.to_datetime(df["signal_date"], errors="coerce", dayfirst=False)
                # Some old dates stored as DD-MM-YYYY — re-parse those
                bad = parsed.isna() | (parsed.dt.year < 2020)
                if bad.any():
                    parsed[bad] = pd.to_datetime(df.loc[bad, "signal_date"], errors="coerce", dayfirst=True)
                df.loc[parsed.dt.date >= V2_START, "pipeline_ver"] = "v2"
            except Exception:
                pass
        return df
    return pd.DataFrame(columns=COLS)


def save_trades(df: pd.DataFrame):
    df.to_csv(TRADES_CSV, index=False)
    print(f"  Saved {len(df)} trades -> {TRADES_CSV.name}")
    _save_v2(df)


def _save_v2(df: pd.DataFrame):
    """Write v2 trades to separate CSV + Excel for clean performance tracking."""
    v2 = df[df.get("pipeline_ver", pd.Series("v1", index=df.index)) == "v2"].copy() \
        if "pipeline_ver" in df.columns else pd.DataFrame()
    if v2.empty:
        return
    # CSV
    v2.to_csv(TRADES_V2_CSV, index=False)
    # Excel — two sheets: Active (OPEN) and History (closed)
    try:
        with pd.ExcelWriter(TRADES_V2_XLSX, engine="openpyxl") as w:
            open_t   = v2[v2["status"] == "OPEN"]
            closed_t = v2[v2["status"].isin(["WIN", "LOSS", "EXPIRED"])]
            display_cols = ["trade_id","signal_date","ticker","direction","primary_rule",
                            "tier","sector","entry_price","stop_price","target_price",
                            "qty","composite_score","status","exit_date","exit_price",
                            "exit_reason","pnl_pct","rr_achieved","days_held"]
            dc = [c for c in display_cols if c in v2.columns]
            open_t[dc].to_excel(w,   sheet_name="Active Trades", index=False)
            closed_t[dc].to_excel(w, sheet_name="Trade History", index=False)
            # Summary sheet
            wins   = int((v2["status"] == "WIN").sum())
            losses = int((v2["status"] == "LOSS").sum())
            exps   = int((v2["status"] == "EXPIRED").sum())
            wr     = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0
            pnl_s  = pd.to_numeric(v2["pnl_pct"], errors="coerce").dropna()
            avg_pnl = round(float(pnl_s.mean()), 2) if len(pnl_s) > 0 else 0
            summary = pd.DataFrame([
                ["Pipeline Version", "v2 (TS-anchored stops + SHORT support)"],
                ["Start Date", str(V2_START)],
                ["Total Trades", len(v2)],
                ["Open", int((v2["status"] == "OPEN").sum())],
                ["WIN", wins], ["LOSS", losses], ["EXPIRED", exps],
                ["Win Rate (W/W+L)", f"{wr}%"],
                ["Avg PnL (all closed)", f"{avg_pnl:+.2f}%"],
            ], columns=["Metric", "Value"])
            summary.to_excel(w, sheet_name="Summary", index=False)
        print(f"  v2 Excel saved -> {TRADES_V2_XLSX.name}  ({len(v2)} trades)")
    except Exception as e:
        print(f"  Excel save failed: {e}")


def get_primary_rule(row) -> str:
    """Return which rule drove the signal — checks long and short variants."""
    if row.get("rule_a_fires"):         return "A"
    if row.get("rule_c_fires"):         return "C"
    if row.get("rule_d_fires"):         return "D"
    if row.get("rule_e_fires"):         return "E"
    if row.get("rule_a_short_fires"):   return "As"
    if row.get("rule_c_short_fires"):   return "Cs"
    if row.get("rule_e_short_fires"):   return "Es"
    return "?"


def make_trade_id(signal_date: str, ticker: str) -> str:
    return f"{signal_date}_{ticker}"


def fetch_ohlc(ticker: str):
    """Returns (high, low, close) for latest session. None if fetch fails."""
    try:
        data = yf.download(ticker + ".NS", period="5d", progress=False, auto_adjust=True)
        if data.empty:
            return None, None, None
        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        high  = float(data["High"].dropna().iloc[-1])
        low   = float(data["Low"].dropna().iloc[-1])
        close = float(data["Close"].dropna().iloc[-1])
        return high, low, close
    except Exception:
        pass
    return None, None, None


# -- COMMAND 1: LOG ------------------------------------------------------------
def cmd_log():
    """Read today's FIRE signals from L4 and append new trades to CSV."""
    if not L4_PARQUET.exists():
        print("ERROR: nse_layer4_portfolio.parquet not found. Run layer5 first.")
        return

    try:
        l4 = pd.read_parquet(L4_PARQUET)
    except ImportError:
        print("ERROR: pyarrow is required to read parquet files.")
        print("  Run: pip install pyarrow")
        return

    fire = l4[l4["timing_verdict"].str.upper() == "FIRE"].copy()

    if fire.empty:
        print("No FIRE signals in today's L4 output.")
        return

    trades = load_trades()
    signal_date = str(TODAY)
    new_count = 0

    for _, row in fire.iterrows():
        ticker = row["ticker"]
        tid    = make_trade_id(signal_date, ticker)

        if len(trades) > 0 and tid in trades["trade_id"].values:
            print(f"  SKIP (already logged): {ticker}")
            continue

        _blocked, _reason = check_cooldown(ticker, trades)
        if _blocked:
            print(f"\n  {_reason}")
            print(f"  Trade NOT logged.\n")
            continue

        _dup_blocked, _dup_reason = check_already_open(ticker, trades)
        if _dup_blocked:
            print(f"\n  {_dup_reason}")
            print(f"  Trade NOT logged.\n")
            continue

        primary   = get_primary_rule(row)
        direction = str(row.get("direction", "LONG")).upper()
        # Treat BOTH as LONG for paper tracking (take the long leg)
        if direction == "BOTH":
            direction = "LONG"

        entry        = round(float(row["entry_price"]), 2)
        stop_price   = round(float(row["stop_price"]), 2)
        target_price = round(float(row["target_price"]), 2)
        # L4 now writes direction-correct stops:
        #   LONG:  stop < entry, target > entry
        #   SHORT: stop > entry, target < entry

        if direction == "SHORT":
            target_pct = round((entry - target_price) / entry * 100, 2)
            stop_pct   = round((stop_price - entry)   / entry * 100, 2)
        else:
            target_pct = round((target_price - entry) / entry * 100, 2)
            stop_pct   = round((entry - stop_price)   / entry * 100, 2)
        rr_ratio = round(target_pct / stop_pct, 2) if stop_pct > 0 else 0

        new_row = {
            "trade_id"           : tid,
            "signal_date"        : signal_date,
            "ticker"             : ticker,
            "companyname"        : row.get("companyname", ""),
            "tier"               : row.get("tier", ""),
            "sector"             : row.get("sector", ""),
            "entry_price"        : entry,
            "stop_price"         : stop_price,
            "target_price"       : target_price,
            "qty"                : int(row["qty"]),
            "composite_score"    : round(float(row["composite_score"]), 4),
            "final_score"        : round(float(row["final_score"]), 4),
            "fractal_score"      : round(float(row.get("fractal_score", 0)), 1),
            "rule_c_fires"       : bool(row.get("rule_c_fires", False)),
            "rule_d_fires"       : bool(row.get("rule_d_fires", False)),
            "rule_e_fires"       : bool(row.get("rule_e_fires", False)),
            "rule_c_entry_score" : round(float(row.get("rule_c_entry_score", 0)), 4),
            "rule_d_entry_score" : round(float(row.get("rule_d_entry_score", 0)), 4),
            "rule_e_entry_score" : round(float(row.get("rule_e_entry_score", 0)), 4),
            "primary_rule"       : primary,
            "direction"          : direction,
            "regime"             : row.get("regime", ""),
            "vix"                : round(float(row.get("vix", 0)), 2),
            "r2_grade"           : row.get("r2_grade", ""),
            "hurst"              : round(float(row.get("hurst", 0)), 4),
            "status"             : "OPEN",
            "exit_date"          : "",
            "exit_price"         : "",
            "exit_reason"        : "",
            "pnl_pts"            : "",
            "pnl_pct"            : "",
            "rr_achieved"        : "",
            "days_held"          : "",
            "pipeline_ver"       : "v2",
            "notes"              : "",
        }

        trades = pd.concat([trades, pd.DataFrame([new_row])], ignore_index=True)
        new_count += 1
        dir_tag = "SHORT" if direction == "SHORT" else "LONG"
        print(f"  LOGGED [{dir_tag}]: {ticker:12s} E:{entry:.2f}  SL:{stop_price:.2f}({'-' if direction=='SHORT' else '-'}{stop_pct:.1f}%)  T:{target_price:.2f}(+{target_pct:.1f}%)  RR:{rr_ratio:.1f}  Rule:{primary}  Score:{row['composite_score']:.3f}")

    save_trades(trades)
    print(f"\n  {new_count} new trade(s) logged. Total in tracker: {len(trades)}")

# -- COMMAND 2: UPDATE ---------------------------------------------------------
def cmd_update():
    """Check open trades against current prices, mark WIN/LOSS/EXPIRED."""
    trades = load_trades()
    open_trades = trades[trades["status"] == "OPEN"]

    if open_trades.empty:
        print("No open trades to update.")
        return

    print(f"Checking {len(open_trades)} open trade(s)...")
    updated = 0
    open_pnl_pcts = []

    for idx, row in open_trades.iterrows():
        ticker     = row["ticker"]
        entry      = float(row["entry_price"])
        stop       = float(row["stop_price"])
        target     = float(row["target_price"])
        signal_dt = pd.to_datetime(row["signal_date"], dayfirst=False).date()
        days_held  = (TODAY - signal_dt).days

        direction = str(row.get("direction", "LONG")).upper()
        is_short  = (direction == "SHORT")

        # Day 0 - allow SL exit but not WIN (no lookahead on upside)
        if days_held < 1:
            high, low, close = fetch_ohlc(ticker)
            if close is not None:
                risk_pts0    = (stop - entry) if is_short else (entry - stop)
                sl_hit_d0    = (high >= stop) if is_short else (low <= stop)
                pnl_open     = round((entry - close) if is_short else (close - entry), 2)
                pnl_pct_open = round(pnl_open / entry * 100, 2)
                if sl_hit_d0:
                    pnl_pts0    = round((entry - stop) if is_short else (stop - entry), 2)
                    pnl_pct0    = round(pnl_pts0 / entry * 100, 2)
                    rr0         = round(pnl_pts0 / risk_pts0, 2) if risk_pts0 > 0 else 0
                    trades.at[idx, "status"]      = "LOSS"
                    trades.at[idx, "exit_date"]   = str(TODAY)
                    trades.at[idx, "exit_price"]  = stop
                    trades.at[idx, "exit_reason"] = "SL_HIT (day0)"
                    trades.at[idx, "pnl_pts"]     = pnl_pts0
                    trades.at[idx, "pnl_pct"]     = pnl_pct0
                    trades.at[idx, "rr_achieved"] = rr0
                    trades.at[idx, "days_held"]   = 0
                    updated += 1
                    print(f"  {ticker:12s} [{direction}] LOSS | SL hit on entry day! Exit:{stop:.2f}  PnL:{pnl_pts0:+.2f}pts ({pnl_pct0:+.2f}%)  [SL_HIT day0]")
                else:
                    open_pnl_pcts.append(pnl_pct_open)
                    print(f"  {ticker:12s} [{direction}] DAY1 | CMP:{close:.2f}  OpenPnL:{pnl_open:+.2f}pts ({pnl_pct_open:+.2f}%)  [no exit today]")
            else:
                print(f"  {ticker:12s} DAY1 | price fetch failed")
            continue

        high, low, close = fetch_ohlc(ticker)
        if close is None:
            print(f"  {ticker:12s} price fetch failed - skipping")
            continue

        direction  = str(row.get("direction", "LONG")).upper()
        is_short   = (direction == "SHORT")

        # For LONG: risk = entry - stop, reward = target - entry
        # For SHORT: risk = stop - entry (stop is above), reward = entry - target (target is below)
        risk_pts   = (stop - entry)   if is_short else (entry - stop)
        reward_pts = (entry - target) if is_short else (target - entry)

        status      = "OPEN"
        exit_reason = ""
        exit_price  = ""
        pnl_pts     = ""
        pnl_pct     = ""
        rr_achieved = ""

        # Direction-aware hit detection
        # LONG:  SL = price drops to stop,  Target = price rises to target
        # SHORT: SL = price rises to stop,  Target = price drops to target
        sl_hit     = (high >= stop)   if is_short else (low  <= stop)
        target_hit = (low  <= target) if is_short else (high >= target)

        def pnl(exit_px):
            pts = (entry - exit_px) if is_short else (exit_px - entry)
            pct = round(pts / entry * 100, 2)
            rr  = round(pts / risk_pts, 2) if risk_pts > 0 else 0
            return round(pts, 2), pct, rr

        if sl_hit and target_hit:
            status      = "LOSS"
            exit_reason = "SL_HIT (same candle as target)"
            exit_price  = stop
            pnl_pts, pnl_pct, rr_achieved = pnl(stop)

        elif target_hit:
            status      = "WIN"
            exit_reason = "TARGET_HIT (intraday)"
            exit_price  = target
            pnl_pts, pnl_pct, rr_achieved = pnl(target)

        elif sl_hit:
            status      = "LOSS"
            exit_reason = "SL_HIT (intraday)"
            exit_price  = stop
            pnl_pts, pnl_pct, rr_achieved = pnl(stop)

        elif days_held >= {"Crisis": 5, "Sideways": 10, "Bull": 15}.get(
                str(row.get("regime", "Sideways")), 20):
            status      = "EXPIRED"
            exit_reason = "MAX_HOLD"
            exit_price  = close
            pnl_pts, pnl_pct, rr_achieved = pnl(close)

        if status != "OPEN":
            trades.at[idx, "status"]      = status
            trades.at[idx, "exit_date"]   = str(TODAY)
            trades.at[idx, "exit_price"]  = exit_price
            trades.at[idx, "exit_reason"] = exit_reason
            trades.at[idx, "pnl_pts"]     = pnl_pts
            trades.at[idx, "pnl_pct"]     = pnl_pct
            trades.at[idx, "rr_achieved"] = rr_achieved
            trades.at[idx, "days_held"]   = days_held
            updated += 1
            print(f"  {ticker:12s} {status:8s} | Exit:{exit_price:.2f}  PnL:{pnl_pts:+.2f}pts ({pnl_pct:+.2f}%)  RR:{rr_achieved}  Days:{days_held}  [{exit_reason}]")
        else:
            # Direction-aware open PnL and progress
            if is_short:
                pnl_open     = round(entry - close, 2)
                pnl_pct_open = round((entry - close) / entry * 100, 2)
                range_total  = stop - target   # stop above entry, target below
                progress     = max(0, min(100, (stop - close) / range_total * 100)) if range_total > 0 else 0
            else:
                pnl_open     = round(close - entry, 2)
                pnl_pct_open = round((close - entry) / entry * 100, 2)
                range_total  = target - stop
                progress     = max(0, min(100, (close - stop) / range_total * 100)) if range_total > 0 else 0
            high_pct_to_target = round((target - high) / target * 100, 2)
            low_pct_to_sl      = round((low - stop) / stop * 100, 2)
            filled = int(progress / 5)
            bar    = chr(9608) * filled + chr(9617) * (20 - filled)
            open_pnl_pcts.append(pnl_pct_open)
            print(f"  {ticker:12s} OPEN | CMP:{close:.2f} [{bar}] {progress:.0f}% of SL-Target range")
            print(f"               PnL:{pnl_open:+.2f}pts ({pnl_pct_open:+.2f}%)  DayHigh->Target:{high_pct_to_target:+.2f}%  DayLow->SL:{low_pct_to_sl:+.2f}%  Days:{days_held}")

    save_trades(trades)
    if open_pnl_pcts:
        avg_open_pnl = sum(open_pnl_pcts) / len(open_pnl_pcts)
        sign = "+" if avg_open_pnl >= 0 else ""
        print(f"\n  -- Avg open-trade PnL : {sign}{avg_open_pnl:.2f}%  ({len(open_pnl_pcts)} positions)")
    print(f"\n  {updated} trade(s) closed today.")


# -- COMMAND 3: REPORT ---------------------------------------------------------
def cmd_report():
    """Print full performance report."""
    trades = load_trades()

    if trades.empty:
        print("No trades logged yet.")
        return

    closed = trades[trades["status"].isin(["WIN", "LOSS", "EXPIRED"])].copy()
    open_t = trades[trades["status"] == "OPEN"]

    print()
    print("=" * 65)
    print("  F1 PAPER TRADING REPORT -", str(TODAY))
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

    # Expectancy
    if len(losses) > 0 and avg_loss != 0:
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        print(f"  {'Expectancy per trade':<25} {expectancy:+.2f}%")

    # Per rule breakdown
    print()
    print("  BY PRIMARY RULE")
    print(f"  {'Rule':<8} {'Trades':<8} {'WinRate':<10} {'AvgPnL':<10} {'AvgRR'}")
    for rule in ["C", "D", "E", "?"]:
        sub = closed[closed["primary_rule"] == rule]
        if sub.empty:
            continue
        wr  = len(sub[sub["status"] == "WIN"]) / len(sub) * 100
        ap  = sub["pnl_pct"].mean()
        arr = sub["rr_achieved"].mean()
        print(f"  {rule:<8} {len(sub):<8} {wr:<10.1f} {ap:<+10.2f} {arr:.2f}")

    # Per regime breakdown
    print()
    print("  BY REGIME")
    for regime in closed["regime"].unique():
        sub = closed[closed["regime"] == regime]
        wr  = len(sub[sub["status"] == "WIN"]) / len(sub) * 100
        ap  = sub["pnl_pct"].mean()
        print(f"  {regime:<12} {len(sub)} trades  WinRate:{wr:.1f}%  AvgPnL:{ap:+.2f}%")

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

    # Recent 10 trades
    print()
    print("  LAST 10 CLOSED TRADES")
    print(f"  {'Date':<12} {'Ticker':<12} {'Rule':<6} {'Status':<8} {'PnL%':<10} {'RR':<6} {'Days'}")
    recent = closed.sort_values("exit_date", ascending=False).head(10)
    for _, r in recent.iterrows():
        print(f"  {str(r['exit_date']):<12} {r['ticker']:<12} {r['primary_rule']:<6} {r['status']:<8} {r['pnl_pct']:+.2f}%     {r['rr_achieved']:<6} {int(r['days_held']) if pd.notna(r['days_held']) else '-'}")

    print()
    print("=" * 65)

    # Readiness check
    print()
    print("  LIVE TRADING READINESS")
    ready = True
    checks = [
        (len(closed) >= 30,       f"30 closed trades ({len(closed)}/30)"),
        (win_rate >= 50,          f"Win rate >= 50% ({win_rate:.1f}%)"),
        (avg_rr >= 1.5,           f"Avg R:R >= 1.5 ({avg_rr:.2f})"),
    ]
    for passed, label in checks:
        mark = "[PASS]" if passed else "[FAIL]"
        print(f"  {mark} {label}")
        if not passed:
            ready = False

    print()
    if ready:
        print("  READY for Phase 3 - 1-2L live, Rule C only, 2-3 positions max")
    else:
        print("  Not ready, continue paper trading Lil Bro")
    print()


# -- COMMAND 4: DAILY ----------------------------------------------------------
def cmd_daily():
    print("\n-- STEP 1: LOG TODAY'S FIRE SIGNALS --")
    cmd_log()
    print("\n-- STEP 2: UPDATE OPEN TRADES --")
    cmd_update()
    print("\n-- STEP 3: PERFORMANCE REPORT --")
    cmd_report()


# -- MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    dispatch = {
        "log"    : cmd_log,
        "update" : cmd_update,
        "report" : cmd_report,
        "daily"  : cmd_daily,
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}")
        print("Usage: python f1_paper_tracker.py [log|update|report|daily]")
        sys.exit(1)

    dispatch[cmd]()
