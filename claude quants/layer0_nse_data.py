# ============================================================
# layer0_nse_data.py  â€”  NSE Corporate Data Fetcher v2
#
# Uses the `nse` pip package (BennyThadikaran/NseIndiaApi)
# which handles Akamai TLS fingerprinting via curl_cffi.
# Raw requests always fail â€” don't fight Akamai manually.
#
# Outputs (all to BASE_PATH):
#   earnings_dna.parquet       â€” next result date per ticker
#   earnings_calendar.parquet  â€” full board meeting calendar
#   bulk_deals.parquet         â€” 30d bulk deal history
#   block_deals.parquet        â€” 30d block deal history
#   corporate_actions.parquet  â€” dividends, splits, bonuses
#   announcements.parquet      â€” 7d corporate filings
#   insider_trading.parquet    â€” PIT disclosures (placeholder)
#   score_summary.parquet      â€” per-ticker institutional score
#
# Run: after 4 PM IST daily (post-market)
# Runtime: ~60-90 seconds
# ============================================================

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_PATH   = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py\claude quants")
TODAY       = datetime.today()
TODAY_DATE  = TODAY.date()

DAYS_AHEAD  = 90   # earnings calendar lookahead
DAYS_BEHIND = 30   # historical lookback for deals

# Scoring thresholds
BLOCK_MIN_VALUE_CR = 5.0
BULK_MIN_VALUE_CR  = 1.0

# â”€â”€ NSE session folder (nse lib stores cookies here) â”€â”€â”€â”€â”€â”€â”€â”€â”€
COOKIE_DIR  = BASE_PATH / "nse_cookies"
COOKIE_DIR.mkdir(exist_ok=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SESSION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def get_nse_client():
    """
    Returns an NSE client using the nse package.
    server=False = local mode (uses requests + curl_cffi for TLS).
    Cookie folder persists session across runs.
    """
    try:
        from nse import NSE
    except ImportError:
        log.error("nse package not installed. Run:")
        log.error("  pip install nse curl_cffi --break-system-packages")
        raise

    # NSE() takes download_folder as first arg
    # It handles cookie warmup internally â€” no manual session needed
    nse = NSE(download_folder=str(COOKIE_DIR))
    log.info("NSE client initialised")
    return nse


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. EARNINGS CALENDAR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_earnings_calendar(nse) -> pd.DataFrame:
    log.info("=== Fetching earnings calendar (board meetings) ===")

    from_dt = TODAY_DATE - timedelta(days=DAYS_BEHIND)
    to_dt   = TODAY_DATE + timedelta(days=DAYS_AHEAD)

    try:
        # nse.boardMeetings returns list of dicts
        # symbol, purpose, meetingDate, description
        data = nse.boardMeetings(
            index     = "equities",
            from_date = datetime.combine(from_dt, datetime.min.time()),
            to_date   = datetime.combine(to_dt,   datetime.min.time()),
        )
        log.info(f"  Raw board meetings: {len(data)}")
    except Exception as e:
        log.error(f"  boardMeetings failed: {e}")
        return pd.DataFrame()

    if not data:
        log.warning("  No board meeting data returned")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    log.info(f"  Columns: {df.columns.tolist()}")

    # â”€â”€ Normalise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col_map = {
        "symbol"     : "ticker",
        "bm_symbol"  : "ticker",
        "company"    : "company_name",
        "meetingDate": "board_meeting_date",
        "bm_date"    : "board_meeting_date",
        "purpose"    : "purpose",
        "bm_purpose" : "purpose",
        "desc"       : "description",
        "bm_desc"    : "description",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # â”€â”€ Filter to result meetings only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    RESULT_KEYWORDS = [
        "financial result", "quarterly result", "annual result",
        r"half.?year", r"\bq[1-4]\b", "earnings", "unaudited", "audited",
        "results", "revenue"
    ]
    if "purpose" in df.columns:
        mask = df["purpose"].str.lower().str.contains(
            "|".join(RESULT_KEYWORDS), regex=True, na=False
        )
        log.info(f"  Result meetings: {mask.sum()} | Other: {(~mask).sum()}")
        df_all   = df.copy()
        df       = df[mask].copy()
    else:
        df_all = df.copy()

    # â”€â”€ Parse dates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    date_col = "board_meeting_date"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

    today_ts = pd.Timestamp.today().normalize()
    df["days_to_result"] = (df[date_col] - today_ts).dt.days

    # â”€â”€ Per-ticker: soonest upcoming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df_future = df[df["days_to_result"] >= -3].copy()

    if df_future.empty:
        log.warning("  No upcoming result meetings")
        df_all.to_parquet(BASE_PATH / "earnings_calendar.parquet", index=False)
        return pd.DataFrame()

    idx    = df_future.groupby("ticker")["days_to_result"].idxmin()
    lookup = df_future.loc[idx, ["ticker", date_col, "days_to_result", "purpose"]].copy()
    lookup = lookup.rename(columns={date_col: "next_result_date"})
    lookup = lookup.sort_values("days_to_result").reset_index(drop=True)

    # Save both
    df_all.to_parquet(BASE_PATH / "earnings_calendar.parquet", index=False)
    lookup.to_parquet(BASE_PATH / "earnings_dna.parquet", index=False)

    log.info(f"  âœ… earnings_dna: {len(lookup)} tickers | "
             f"calendar: {len(df_all)} total meetings")

    # Preview upcoming 14 days
    soon = lookup[lookup["days_to_result"].between(0, 14)]
    if not soon.empty:
        log.info(f"  Results in next 14 days:\n{soon[['ticker','days_to_result','purpose']].to_string(index=False)}")

    return lookup


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. BULK DEALS (30-day history)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_bulk_deals(nse) -> pd.DataFrame:
    log.info("=== Fetching bulk deals (30d) ===")

    from_dt = datetime.combine(TODAY_DATE - timedelta(days=DAYS_BEHIND), datetime.min.time())
    to_dt   = datetime.combine(TODAY_DATE, datetime.min.time())

    try:
        # nse.bulkdeals(option_type, fromdate, todate) â†’ list[dict]
        data = nse.bulkdeals(
            option_type = "bulk_deals",
            fromdate    = from_dt,
            todate      = to_dt,
        )
        log.info(f"  Raw bulk deal records: {len(data)}")
    except RuntimeError as e:
        log.warning(f"  No bulk deal data available: {e}")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"  bulkdeals failed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    log.info(f"  Columns: {df.columns.tolist()}")
    df = _normalise_deals(df, "bulk")
    df.to_parquet(BASE_PATH / "bulk_deals.parquet", index=False)
    log.info(f"  âœ… bulk_deals.parquet: {len(df)} rows")
    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. BLOCK DEALS (30-day history)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_block_deals(nse) -> pd.DataFrame:
    log.info("=== Fetching block deals (30d) ===")

    from_dt = datetime.combine(TODAY_DATE - timedelta(days=DAYS_BEHIND), datetime.min.time())
    to_dt   = datetime.combine(TODAY_DATE, datetime.min.time())

    try:
        data = nse.bulkdeals(
            option_type = "block_deals",
            fromdate    = from_dt,
            todate      = to_dt,
        )
        log.info(f"  Raw block deal records: {len(data)}")
    except RuntimeError as e:
        log.warning(f"  No block deal data available: {e}")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"  block deals failed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = _normalise_deals(df, "block")
    df.to_parquet(BASE_PATH / "block_deals.parquet", index=False)
    log.info(f"  âœ… block_deals.parquet: {len(df)} rows")
    return df


def _normalise_deals(df: pd.DataFrame, deal_type: str) -> pd.DataFrame:
    """Shared normalisation for bulk and block deal DataFrames."""
    # nse package returns consistent column names across versions
    col_map = {
        # nse package v2.x field names
        "symbol"        : "ticker",
        "clientName"    : "client_name",
        "dealType"      : "trade_type",   # BUY / SELL
        "quantityTraded": "qty",
        "tradePrice"    : "price",
        "dealDate"      : "deal_date",
        # alternate field names (older versions)
        "SYMBOL"        : "ticker",
        "CLIENT NAME"   : "client_name",
        "TRADE TYPE"    : "trade_type",
        "QUANTITY TRD"  : "qty",
        "TRADE PRICE"   : "price",
        "DATE"          : "deal_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Numeric cleanup
    for col in ["qty", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # Value in crores
    if "qty" in df.columns and "price" in df.columns:
        df["value_cr"] = (df["qty"] * df["price"] / 1e7).round(2)

    # Parse dates
    if "deal_date" in df.columns:
        df["deal_date"] = pd.to_datetime(df["deal_date"], errors="coerce")

    # Trade direction flags
    if "trade_type" in df.columns:
        tt = df["trade_type"].str.upper().fillna("")
        df["is_buy"]  = tt.str.contains("BUY|B$", regex=True, na=False)
        df["is_sell"] = tt.str.contains("SELL|S$", regex=True, na=False)

    df["deal_type"] = deal_type

    dedup = [c for c in ["ticker", "client_name", "deal_date", "qty"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup).reset_index(drop=True)
    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. CORPORATE ACTIONS (dividends, splits, bonuses)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_corporate_actions(nse) -> pd.DataFrame:
    log.info("=== Fetching corporate actions ===")

    from_dt = datetime.combine(TODAY_DATE - timedelta(days=7),  datetime.min.time())
    to_dt   = datetime.combine(TODAY_DATE + timedelta(days=30), datetime.min.time())

    try:
        data = nse.actions(
            segment   = "equities",
            from_date = from_dt,
            to_date   = to_dt,
        )
        log.info(f"  Raw corporate actions: {len(data)}")
    except Exception as e:
        log.error(f"  actions() failed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    log.info(f"  Columns: {df.columns.tolist()}")

    col_map = {
        "symbol"    : "ticker",
        "company"   : "company_name",
        "subject"   : "action_type",    # Dividend / Bonus / Split / Rights
        "exDate"    : "ex_date",
        "recordDate": "record_date",
        "bc_startDate": "bc_start",
        "bc_endDate"  : "bc_end",
        "series"    : "series",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for dc in ["ex_date", "record_date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    if "action_type" in df.columns:
        at = df["action_type"].str.lower().fillna("")
        df["is_dividend"] = at.str.contains("dividend|div", na=False)
        df["is_split"]    = at.str.contains("split|sub.div", regex=True, na=False)
        df["is_bonus"]    = at.str.contains("bonus", na=False)
        df["is_rights"]   = at.str.contains("rights", na=False)

    df.to_parquet(BASE_PATH / "corporate_actions.parquet", index=False)
    log.info(f"  âœ… corporate_actions.parquet: {len(df)} rows")
    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. ANNOUNCEMENTS (7-day corporate filings)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_announcements(nse) -> pd.DataFrame:
    log.info("=== Fetching corporate announcements (7d) ===")

    from_dt = datetime.combine(TODAY_DATE - timedelta(days=7), datetime.min.time())
    to_dt   = datetime.combine(TODAY_DATE, datetime.min.time())

    try:
        data = nse.announcements(
            index     = "equities",
            from_date = from_dt,
            to_date   = to_dt,
        )
        log.info(f"  Raw announcements: {len(data)}")
    except Exception as e:
        log.error(f"  announcements() failed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    log.info(f"  Columns: {df.columns.tolist()}")

    col_map = {
        "symbol"  : "ticker",
        "company" : "company_name",
        "subject" : "subject",
        "desc"    : "description",
        "an_dt"   : "announcement_date",
        "dt"      : "announcement_date",
        "sort_date": "announcement_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "announcement_date" in df.columns:
        df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")

    if "subject" in df.columns:
        s = df["subject"].str.lower().fillna("")
        df["is_result"]    = s.str.contains(r"result|earnings|q[1-4]", regex=True, na=False)
        df["is_order_win"] = s.str.contains("order|contract|win|award|l1", na=False)
        df["is_guidance"]  = s.str.contains("guidance|outlook|forecast|upgrade", na=False)
        df["is_adverse"]   = s.str.contains(
            "nclt|insolvency|fraud|penalty|sebi|show cause|enquiry|investigation",
            na=False
        )
        df["is_mgmt_change"] = s.str.contains(
            "md|ceo|cfo|director|resign|appoint|management change", na=False
        )

    df.to_parquet(BASE_PATH / "announcements.parquet", index=False)
    log.info(f"  âœ… announcements.parquet: {len(df)} rows")
    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. SCORE SUMMARY  (institutional signal per ticker)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def build_score_summary(
    df_bulk    : pd.DataFrame,
    df_block   : pd.DataFrame,
    df_earnings: pd.DataFrame,
    df_announce: pd.DataFrame,
    df_actions : pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-ticker institutional score consumed by Layer 3 Rule F.

    Score components (additive, clipped to [-1, 1]):
      +0.30  Block deal BUY â‰¥ â‚¹5cr in last 10 days
      +0.20  Bulk deal BUY â‰¥ â‚¹1cr in last 5 days
      +0.10  Corporate action upcoming (ex-date within 10 days)
      -0.30  Adverse announcement (NCLT / fraud / penalty / SEBI)
      +0.10  Result in 3-5 days (pre-result momentum window)
      -0.15  Result in 0-2 days (blackout â€” too risky)
      -0.10  Ex-dividend in 1-2 days (price drop expected)
    """
    log.info("=== Building institutional score summary ===")

    today_ts = pd.Timestamp.today().normalize()

    # Collect all tickers across all datasets
    all_tickers = set()
    for df in [df_bulk, df_block, df_earnings, df_announce, df_actions]:
        if not df.empty and "ticker" in df.columns:
            all_tickers.update(df["ticker"].dropna().unique())

    if not all_tickers:
        log.warning("  No tickers found across any dataset")
        return pd.DataFrame()

    records = []

    for tk in all_tickers:
        score = 0.0
        flags = []
        days_to = 999

        # â”€â”€ Block deal BUY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not df_block.empty and "ticker" in df_block.columns:
            b = df_block[
                (df_block["ticker"] == tk) &
                (df_block.get("is_buy", pd.Series(False, index=df_block.index)))
            ]
            if not b.empty and "deal_date" in b.columns:
                recent = b[b["deal_date"] >= today_ts - timedelta(days=10)]
                if not recent.empty:
                    val = recent.get("value_cr", pd.Series([0])).fillna(0).sum()
                    if val >= BLOCK_MIN_VALUE_CR:
                        score += 0.30
                        flags.append(f"block_buy_{val:.0f}cr")

        # â”€â”€ Bulk deal BUY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not df_bulk.empty and "ticker" in df_bulk.columns:
            b = df_bulk[
                (df_bulk["ticker"] == tk) &
                (df_bulk.get("is_buy", pd.Series(False, index=df_bulk.index)))
            ]
            if not b.empty and "deal_date" in b.columns:
                recent = b[b["deal_date"] >= today_ts - timedelta(days=5)]
                if not recent.empty:
                    val = recent.get("value_cr", pd.Series([0])).fillna(0).sum()
                    if val >= BULK_MIN_VALUE_CR:
                        score += 0.20
                        flags.append(f"bulk_buy_{val:.0f}cr")

        # â”€â”€ Adverse announcement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not df_announce.empty and "ticker" in df_announce.columns:
            ann = df_announce[df_announce["ticker"] == tk]
            if not ann.empty and "is_adverse" in ann.columns:
                if ann["is_adverse"].any():
                    score -= 0.30
                    flags.append("adverse_ann")

        # â”€â”€ Corporate action (ex-date upcoming) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not df_actions.empty and "ticker" in df_actions.columns:
            ca = df_actions[df_actions["ticker"] == tk]
            if not ca.empty and "ex_date" in ca.columns:
                upcoming = ca[
                    ca["ex_date"].notna() &
                    (ca["ex_date"] >= today_ts) &
                    (ca["ex_date"] <= today_ts + timedelta(days=10))
                ]
                if not upcoming.empty:
                    # Ex-div in 1-2 days = price drop risk
                    min_days = (upcoming["ex_date"] - today_ts).dt.days.min()
                    if min_days <= 2:
                        score -= 0.10
                        flags.append(f"ex_div_{min_days}d")
                    else:
                        score += 0.10
                        flags.append(f"corp_action_{min_days}d")

        # â”€â”€ Earnings timing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not df_earnings.empty and "ticker" in df_earnings.columns:
            row = df_earnings[df_earnings["ticker"] == tk]
            if not row.empty and "days_to_result" in row.columns:
                days_to = int(row["days_to_result"].iloc[0])
                if 3 <= days_to <= 5:
                    score += 0.10
                    flags.append(f"pre_result_{days_to}d")
                elif days_to <= 2:
                    score -= 0.15
                    flags.append(f"blackout_{days_to}d")

        records.append({
            "ticker"              : tk,
            "institutional_score" : float(np.clip(score, -1.0, 1.0)),
            "days_to_result"      : days_to,
            "flags"               : "|".join(flags) if flags else "",
            "has_block_buy"       : any("block_buy" in f for f in flags),
            "has_bulk_buy"        : any("bulk_buy" in f for f in flags),
            "has_adverse"         : "adverse_ann" in flags,
            "has_blackout"        : any("blackout" in f for f in flags),
            "score_date"          : str(TODAY_DATE),
        })

    df_out = (
        pd.DataFrame(records)
        .sort_values("institutional_score", ascending=False)
        .reset_index(drop=True)
    )
    df_out.to_parquet(BASE_PATH / "score_summary.parquet", index=False)
    log.info(f"  âœ… score_summary.parquet: {len(df_out)} tickers")
    return df_out


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info(f"layer0_nse_data.py v2  |  {TODAY.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    nse = get_nse_client()

    results = {
        "earnings" : pd.DataFrame(),
        "bulk"     : pd.DataFrame(),
        "block"    : pd.DataFrame(),
        "actions"  : pd.DataFrame(),
        "announce" : pd.DataFrame(),
    }

    for key, fn, label in [
        ("earnings", fetch_earnings_calendar, "Earnings calendar"),
        ("bulk",     fetch_bulk_deals,        "Bulk deals"),
        ("block",    fetch_block_deals,       "Block deals"),
        ("actions",  fetch_corporate_actions, "Corporate actions"),
        ("announce", fetch_announcements,     "Announcements"),
    ]:
        try:
            results[key] = fn(nse)
            time.sleep(2)   # polite delay between calls
        except Exception as e:
            log.error(f"{label} failed: {e}")

    # Score summary
    try:
        df_summary = build_score_summary(
            df_bulk     = results["bulk"],
            df_block    = results["block"],
            df_earnings = results["earnings"],
            df_announce = results["announce"],
            df_actions  = results["actions"],
        )
    except Exception as e:
        log.error(f"Score summary failed: {e}")
        df_summary = pd.DataFrame()

    # Close NSE session cleanly
    try:
        nse.exit()
    except Exception:
        pass

    # â”€â”€ Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("LAYER 0 â€” NSE DATA FETCH REPORT")
    print("=" * 60)
    print(f"  earnings_dna       : {len(results['earnings']):>5} tickers")
    print(f"  bulk_deals         : {len(results['bulk']):>5} records (30d)")
    print(f"  block_deals        : {len(results['block']):>5} records (30d)")
    print(f"  corporate_actions  : {len(results['actions']):>5} actions")
    print(f"  announcements      : {len(results['announce']):>5} filings (7d)")
    print(f"  score_summary      : {len(df_summary):>5} tickers scored")
    print(f"  Runtime            : {elapsed:.1f}s")
    print("=" * 60)

    if not df_summary.empty:
        top = df_summary[df_summary["institutional_score"] > 0].head(20)
        if not top.empty:
            print("\nTOP INSTITUTIONAL SIGNALS:")
            print(top[["ticker","institutional_score","days_to_result","flags"]]
                  .to_string(index=False))

        blackout = df_summary[df_summary["has_blackout"]]
        if not blackout.empty:
            print(f"\nâš ï¸  EARNINGS BLACKOUT ({len(blackout)} tickers â€” avoid):")
            print(blackout[["ticker","days_to_result"]].to_string(index=False))

        adverse = df_summary[df_summary["has_adverse"]]
        if not adverse.empty:
            print(f"\nðŸš¨ ADVERSE ANNOUNCEMENTS ({len(adverse)} tickers):")
            print(adverse[["ticker","flags"]].to_string(index=False))

    print("\nâœ… layer0 complete.")


if __name__ == "__main__":
    main()
