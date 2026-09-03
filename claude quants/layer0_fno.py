# ============================================================
# layer0_fno.py  â€”  NSE F&O Data Fetcher
# Version: 1.0  |  2026-05-20
#
# Fetches F&O signals for all ~200 NSE F&O eligible stocks.
# Uses the same `nse` package (BennyThadikaran/NseIndiaApi)
# as layer0_nse_data.py â€” no new dependencies.
#
# Output parquets (all to BASE_PATH):
#   fno_oi.parquet        â€” OI buildup/unwind per ticker
#   fno_pcr.parquet       â€” Put/Call Ratio per ticker
#   fno_iv.parquet        â€” IV percentile (52-week context) per ticker
#   fno_maxpain.parquet   â€” Max Pain strike + distance from CMP per ticker
#   fno_rollover.parquet  â€” Rollover % (current vs prev series)
#
# Run: after 4 PM IST daily, AFTER layer0_nse_data.py
# Runtime: ~90-120 seconds (~200 tickers Ã— option chain calls)
#
# Daily flow:
#   Step 1 â†’ layer0_nse_data.py   (~18s)
#   Step 2 â†’ layer0_fno.py        (~90-120s)   â† THIS FILE
#   Step 3 â†’ layer5_orchestrator  (~62s)
# ============================================================

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=__import__("sys").stdout)
log = logging.getLogger(__name__)

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_PATH  = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py\claude quants")
TODAY      = datetime.today()
TODAY_DATE = TODAY.date()

COOKIE_DIR = BASE_PATH / "nse_cookies"
COOKIE_DIR.mkdir(exist_ok=True)

# Per-ticker call: how long to wait between option chain requests
# NSE rate limits aggressively â€” 1.2s is safe for ~200 tickers in ~4 minutes
RATE_LIMIT_SLEEP = 1.2

# IV percentile lookback window (trading days)
# We use 90-day historical vol as proxy when live IV history unavailable
IV_LOOKBACK_DAYS = 90

# Rollover: consider previous series OI for rollover % calculation
ROLLOVER_LOOKBACK_DAYS = 30


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# NSE CLIENT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def get_nse_client():
    try:
        from nse import NSE
    except ImportError:
        log.error("nse package not installed. Run: pip install nse curl_cffi --break-system-packages")
        raise
    nse = NSE(download_folder=str(COOKIE_DIR))
    log.info("NSE client initialised")
    return nse


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 1: GET F&O ELIGIBLE TICKER LIST
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def get_fno_tickers(nse) -> list[str]:
    """
    Returns list of F&O eligible NSE tickers.

    Strategy:
      Primary  â†’ nse.fnoLots() returns dict {ticker: lot_size} for all F&O stocks
      Fallback â†’ filter universe_master.parquet if fnoLots fails
    """
    log.info("=== Fetching F&O eligible ticker list ===")

    # Primary: fnoLots() is the cleanest source â€” returns every F&O stock
    try:
        lots = nse.fnoLots()
        if lots and isinstance(lots, dict):
            # Keys are tickers (e.g. "RELIANCE", "HDFCBANK")
            # Remove index entries like NIFTY, BANKNIFTY etc
            index_names = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT",
                           "SENSEX", "BANKEX", "CRUDEOIL", "GOLD", "SILVER"}
            tickers = [t.strip() for t in lots.keys() if t.strip().upper() not in index_names]
            log.info(f"  fnoLots(): {len(tickers)} F&O stocks")
            # Intersect with Layer2 candidates if available
            l2_path = BASE_PATH / "nse_layer2_candidates.parquet"
            if l2_path.exists():
                l2_tickers = set(pd.read_parquet(l2_path)["ticker"].tolist())
                tickers = [t for t in tickers if t in l2_tickers]
                log.info(f"  After Layer2 intersect: {len(tickers)} tickers")
            return sorted(tickers)
    except Exception as e:
        log.warning(f"  fnoLots() failed: {e}")

    # Fallback: use universe_master and flag tickers in spread_history
    # (cointegrated pairs = likely F&O, decent proxy)
    log.warning("  Falling back to universe_master.parquet for F&O list")
    try:
        um = pd.read_parquet(BASE_PATH / "universe_master.parquet")
        # If there's an 'is_fno' or 'fno' column, use it
        fno_cols = [c for c in um.columns if "fno" in c.lower() or "derivative" in c.lower()]
        if fno_cols:
            tickers = um[um[fno_cols[0]].astype(bool)]["Ticker"].tolist()
        else:
            # Last resort: T1 tickers are mostly F&O
            tickers = um[um["Tier"] == "T1"]["Ticker"].tolist()
        log.info(f"  Fallback: {len(tickers)} tickers from universe_master")
        return sorted(tickers)
    except Exception as e:
        log.error(f"  Could not load universe_master: {e}")
        return []


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 2: FETCH OPTION CHAIN PER TICKER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_option_chain(nse, ticker: str) -> dict | None:
    """
    Fetches compiled option chain for a single ticker.
    Returns compiled dict or None on failure.

    nse.compileOptionChain() returns:
      {
        'atm': float,               â† ATM strike
        'underlyingValue': float,   â† CMP
        'expiryDate': str,
        'data': [
          {
            'strikePrice': float,
            'CE': {'openInterest': int, 'impliedVolatility': float, 'lastPrice': float, ...},
            'PE': {'openInterest': int, 'impliedVolatility': float, 'lastPrice': float, ...},
          }, ...
        ],
        'totalCE_OI': int,
        'totalPE_OI': int,
      }
    """
    try:
        # Step 1: get nearest expiry from raw optionChain
        raw = nse.optionChain(ticker)
        if not raw:
            log.debug(f"  {ticker}: optionChain returned empty")
            return None

        # Extract nearest expiry date from raw records
        # raw structure: {'records': {'expiryDates': [...], 'data': [...], 'underlyingValue': float}}
        records = raw.get("records", {})
        expiry_dates = records.get("expiryDates", [])
        if not expiry_dates:
            # fallback: try filtered key
            expiry_dates = raw.get("filtered", {}).get("expiryDates", [])

        if not expiry_dates:
            log.debug(f"  {ticker}: no expiryDates found in optionChain response")
            return _parse_raw_chain(raw, ticker)   # parse raw directly

        # Parse nearest expiry â€” NSE returns strings like "24-Apr-2025"
        from datetime import datetime as _dt
        nearest_expiry = None
        today_ts = _dt.today()
        for exp_str in expiry_dates:
            try:
                exp_dt = _dt.strptime(str(exp_str).strip(), "%d-%b-%Y")
                if exp_dt >= today_ts:
                    nearest_expiry = exp_dt
                    break
            except Exception:
                continue

        if nearest_expiry is None:
            log.debug(f"  {ticker}: could not parse any future expiry â€” using raw")
            return _parse_raw_chain(raw, ticker)

        # Step 2: compileOptionChain with the expiry date
        chain = nse.compileOptionChain(ticker, expiryDate=nearest_expiry)
        if chain:
            return chain

        # Step 3: fallback â€” parse raw chain ourselves
        log.debug(f"  {ticker}: compileOptionChain returned empty, parsing raw")
        return _parse_raw_chain(raw, ticker)

    except Exception as e:
        log.debug(f"  {ticker}: fetch_option_chain failed â€” {e}")
        # Last resort: try parsing raw if we got it
        try:
            raw = nse.optionChain(ticker)
            return _parse_raw_chain(raw, ticker) if raw else None
        except Exception:
            return None


def _parse_raw_chain(raw: dict, ticker: str) -> dict | None:
    """
    Parse raw optionChain response into the same structure compileOptionChain returns.
    Handles the case where compileOptionChain needs an expiry but raw always works.

    raw structure:
      records.data        â†’ list of strike dicts with CE/PE sub-dicts
      records.underlyingValue â†’ CMP
      records.expiryDates â†’ list of expiry strings
      filtered.data       â†’ nearest expiry strikes only (use this preferentially)
    """
    try:
        records  = raw.get("records", {})
        filtered = raw.get("filtered", {})

        # Use filtered (nearest expiry) if available, else all records
        data_list = filtered.get("data") or records.get("data", [])
        cmp       = float(records.get("underlyingValue", 0))
        expiries  = records.get("expiryDates", [])
        expiry    = expiries[0] if expiries else ""

        if not data_list or cmp == 0:
            return None

        # Normalise: ensure each item has CE and PE keys
        cleaned = []
        total_ce_oi = 0
        total_pe_oi = 0

        for item in data_list:
            sp  = float(item.get("strikePrice", 0))
            ce  = item.get("CE", {}) or {}
            pe  = item.get("PE", {}) or {}
            ce_oi = int(ce.get("openInterest", 0) or 0)
            pe_oi = int(pe.get("openInterest", 0) or 0)
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi
            cleaned.append({"strikePrice": sp, "CE": ce, "PE": pe})

        # Find ATM strike (closest to CMP)
        strikes = [item["strikePrice"] for item in cleaned if item["strikePrice"] > 0]
        atm = min(strikes, key=lambda s: abs(s - cmp)) if strikes else cmp

        return {
            "atm"            : atm,
            "underlyingValue": cmp,
            "expiryDate"     : expiry,
            "data"           : cleaned,
            "totalCE_OI"     : total_ce_oi,
            "totalPE_OI"     : total_pe_oi,
        }
    except Exception as e:
        log.debug(f"  {ticker}: _parse_raw_chain failed â€” {e}")
        return None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 3: COMPUTE METRICS FROM OPTION CHAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def compute_pcr(chain: dict) -> dict:
    """
    Put/Call Ratio = Total Put OI / Total Call OI
    PCR > 1.2 â†’ bullish sentiment (more puts = hedging, contrarian bullish)
    PCR < 0.7 â†’ bearish pressure (call buyers dominating)
    PCR 0.7-1.2 â†’ neutral
    """
    total_ce_oi = int(chain.get("coiTotal", 0) or chain.get("totalCE_OI", 0) or sum(r.get("CE", {}).get("openInterest", 0) for r in chain.get("data", [])))
    total_pe_oi = int(chain.get("poiTotal", 0) or chain.get("totalPE_OI", 0) or sum(r.get("PE", {}).get("openInterest", 0) for r in chain.get("data", [])))

    pcr = round(total_pe_oi / total_ce_oi, 4) if total_ce_oi > 0 else 0.0

    # Sentiment label
    if pcr > 1.2:
        sentiment = "bullish"
    elif pcr < 0.7:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return {
        "total_ce_oi"   : total_ce_oi,
        "total_pe_oi"   : total_pe_oi,
        "pcr"           : pcr,
        "pcr_sentiment" : sentiment,
    }


def compute_max_pain(chain: dict) -> dict:
    """
    Max Pain = strike where total option buyer loss is maximised.
    i.e. the strike that causes the most pain to buyers of both calls and puts.

    Method:
      For each strike S:
        CE pain = sum over all strikes K < S of: CE_OI(K) Ã— (S - K)
        PE pain = sum over all strikes K > S of: PE_OI(K) Ã— (K - S)
        total_pain(S) = CE_pain + PE_pain
      Max pain strike = argmin(total_pain)

    Output:
      max_pain_strike  â€” strike with minimum total buyer pain
      cmp              â€” current market price (underlying)
      max_pain_distance_pct â€” |CMP - MaxPain| / CMP Ã— 100
      pinning_risk     â€” True if distance < 2% (price likely to pin at MP)
    """
    data = chain.get("data", [])
    cmp  = float(chain.get("underlyingValue", 0))

    if not data or cmp == 0:
        return {"max_pain_strike": 0, "cmp": cmp, "max_pain_distance_pct": 0, "pinning_risk": False}

    strikes    = []
    ce_oi_list = []
    pe_oi_list = []

    for item in data:
        sp    = float(item.get("strikePrice", 0))
        ce_oi = int(item.get("CE", {}).get("openInterest", 0) or 0)
        pe_oi = int(item.get("PE", {}).get("openInterest", 0) or 0)
        if sp > 0:
            strikes.append(sp)
            ce_oi_list.append(ce_oi)
            pe_oi_list.append(pe_oi)

    if not strikes:
        return {"max_pain_strike": 0, "cmp": cmp, "max_pain_distance_pct": 0, "pinning_risk": False}

    strikes    = np.array(strikes)
    ce_oi_arr  = np.array(ce_oi_list, dtype=float)
    pe_oi_arr  = np.array(pe_oi_list, dtype=float)

    total_pain = []
    for S in strikes:
        # CE writers lose when price > strike (CE buyers gain)
        # CE pain to buyers = OI Ã— max(0, S - K) for K < S
        ce_pain = np.sum(ce_oi_arr[strikes < S] * (S - strikes[strikes < S]))
        # PE pain to buyers = OI Ã— max(0, K - S) for K > S
        pe_pain = np.sum(pe_oi_arr[strikes > S] * (strikes[strikes > S] - S))
        total_pain.append(ce_pain + pe_pain)

    total_pain      = np.array(total_pain)
    max_pain_strike = float(strikes[np.argmin(total_pain)])
    distance_pct    = abs(cmp - max_pain_strike) / cmp * 100 if cmp > 0 else 0

    return {
        "max_pain_strike"       : max_pain_strike,
        "cmp"                   : cmp,
        "max_pain_distance_pct" : round(distance_pct, 3),
        # Within 2% = price likely to converge to max pain by expiry â†’ position risk
        "pinning_risk"          : distance_pct < 2.0,
    }


def compute_oi_signals(chain: dict) -> dict:
    """
    OI Buildup / Unwind detection.

    Signals:
      call_oi_buildup   â€” heavy call OI at strikes above CMP = resistance / bearish wall
      put_oi_buildup    â€” heavy put OI at strikes below CMP = support / bullish floor
      net_oi_bias       â€” (put_oi - call_oi) / total_oi â†’ positive = bullish, negative = bearish
      atm_ce_oi         â€” ATM call OI (sentiment near money)
      atm_pe_oi         â€” ATM put OI

    OI interpretation:
      put_oi >> call_oi  â†’ market hedging with puts = underlying resilient (bullish)
      call_oi >> put_oi  â†’ call sellers dominating = capped upside (bearish near-term)
    """
    data  = chain.get("data", [])
    cmp   = float(chain.get("underlyingValue", 0))
    atm   = float(chain.get("atm", cmp))

    if not data or cmp == 0:
        return {
            "call_oi_above_cmp" : 0,
            "put_oi_below_cmp"  : 0,
            "atm_ce_oi"         : 0,
            "atm_pe_oi"         : 0,
            "net_oi_bias"       : 0.0,
            "oi_signal"         : "neutral",
        }

    call_oi_above = 0
    put_oi_below  = 0
    atm_ce_oi     = 0
    atm_pe_oi     = 0

    for item in data:
        sp    = float(item.get("strikePrice", 0))
        ce_oi = int(item.get("CE", {}).get("openInterest", 0) or 0)
        pe_oi = int(item.get("PE", {}).get("openInterest", 0) or 0)

        if sp > cmp:
            call_oi_above += ce_oi
        elif sp < cmp:
            put_oi_below += pe_oi

        # ATM: within 0.5% of ATM strike
        if abs(sp - atm) / atm < 0.005 if atm else False:
            atm_ce_oi += ce_oi
            atm_pe_oi += pe_oi

    total_oi   = call_oi_above + put_oi_below
    net_bias   = (put_oi_below - call_oi_above) / total_oi if total_oi > 0 else 0.0

    # Signal
    if net_bias > 0.2:
        oi_signal = "bullish"    # puts dominate â†’ support strong
    elif net_bias < -0.2:
        oi_signal = "bearish"    # calls dominate â†’ resistance heavy
    else:
        oi_signal = "neutral"

    return {
        "call_oi_above_cmp" : call_oi_above,
        "put_oi_below_cmp"  : put_oi_below,
        "atm_ce_oi"         : atm_ce_oi,
        "atm_pe_oi"         : atm_pe_oi,
        "net_oi_bias"       : round(net_bias, 4),
        "oi_signal"         : oi_signal,
    }


def compute_iv_metrics(chain: dict, ticker: str) -> dict:
    """
    IV Percentile â€” where is today's ATM IV relative to its 52-week range?

    Method:
      1. Extract ATM IV from compiled chain (live)
      2. Fetch 90-day historical FnO data to get IV history
         â†’ nse.fetch_historical_fno_data() for OPTSTK
      3. IV Percentile = (current_iv - min_iv_90d) / (max_iv_90d - min_iv_90d) Ã— 100

    IV Percentile interpretation:
      > 80  â†’ IV very high (expensive options, vol likely to crush)
      50-80 â†’ elevated (pre-event pricing)
      < 30  â†’ IV low (cheap options, good for buying)

    For Rule D (pre-earnings): IV 50-80 is ideal â€” elevated but not at peak.
    IV > 90 = vol already priced in = enter with caution.
    """
    data  = chain.get("data", [])
    cmp   = float(chain.get("underlyingValue", 0))
    atm   = float(chain.get("atm", cmp))

    # Extract ATM IV from chain
    atm_iv = 0.0
    for item in data:
        sp = float(item.get("strikePrice", 0))
        if abs(sp - atm) / atm < 0.01 if atm else False:
            ce_iv = float(item.get("CE", {}).get("impliedVolatility", 0) or 0)
            pe_iv = float(item.get("PE", {}).get("impliedVolatility", 0) or 0)
            iv_vals = [v for v in [ce_iv, pe_iv] if v > 0]
            if iv_vals:
                atm_iv = float(np.mean(iv_vals))
            break

    # Fallback: average across all strikes
    if atm_iv == 0:
        all_ivs = []
        for item in data:
            for leg in ["CE", "PE"]:
                v = float(item.get(leg, {}).get("impliedVolatility", 0) or 0)
                if v > 0:
                    all_ivs.append(v)
        atm_iv = float(np.mean(all_ivs)) if all_ivs else 0.0

    # IV percentile placeholder â€” will be filled by historical data in main loop
    # (returned as 50 when history not available â€” neutral, won't bias scoring)
    return {
        "atm_iv"         : round(atm_iv, 4),
        "iv_percentile"  : 50.0,   # updated in main loop with history
        "iv_label"       : "moderate",
    }


def enrich_iv_percentile(ticker: str, current_iv: float, nse) -> float:
    """
    Fetches 90-day historical OPTSTK IV to compute percentile.
    Returns iv_percentile (0-100).

    Uses nse.fetch_historical_fno_data() with instrument=OPTSTK.
    Falls back to 50 (neutral) if data unavailable.
    """
    if current_iv <= 0:
        return 50.0

    try:
        to_dt   = TODAY_DATE
        from_dt = to_dt - timedelta(days=IV_LOOKBACK_DAYS)

        # Fetch ATM straddle IV from historical data
        # instrument=OPTSTK, option_type=CE (we just need the IV series)
        hist = nse.fetch_historical_fno_data(
            symbol       = ticker,
            instrument   = "OPTSTK",
            option_type  = "CE",
            from_date    = from_dt,
            to_date      = to_dt,
        )

        if not hist:
            return 50.0

        hist_df = pd.DataFrame(hist)

        # Column containing IV â€” NSE returns 'impliedVolatility' or 'iv'
        iv_col = next((c for c in hist_df.columns
                       if "implied" in c.lower() or c.lower() == "iv"), None)
        if iv_col is None:
            return 50.0

        hist_df[iv_col] = pd.to_numeric(hist_df[iv_col], errors="coerce")
        iv_series = hist_df[iv_col].dropna()
        iv_series = iv_series[iv_series > 0]

        if len(iv_series) < 5:
            return 50.0

        iv_min = iv_series.min()
        iv_max = iv_series.max()

        if iv_max == iv_min:
            return 50.0

        percentile = (current_iv - iv_min) / (iv_max - iv_min) * 100
        return float(np.clip(percentile, 0.0, 100.0))

    except Exception as e:
        log.debug(f"  {ticker}: IV history failed â€” {e}")
        return 50.0


def compute_rollover(ticker: str, nse) -> dict:
    """
    Rollover % = how much of current series OI has rolled into next series.

    Formula:
      rollover_pct = next_series_OI / (current_series_OI + next_series_OI) Ã— 100

    Method:
      Fetch historical FUTSTK data for last 30 days.
      Near expiry (last 5 days of series), OI migration to next series = rollover.
      High rollover (>70%) = strong carry-forward = momentum continuation signal.

    Returns:
      rollover_pct     â€” 0-100
      rollover_signal  â€” "strong"(>70) / "moderate"(40-70) / "weak"(<40)
    """
    try:
        to_dt   = TODAY_DATE
        from_dt = to_dt - timedelta(days=ROLLOVER_LOOKBACK_DAYS)

        import signal as _sig
        def _timeout_handler(s, f): raise TimeoutError("rollover fetch timeout")
        _sig.signal(_sig.SIGALRM, _timeout_handler) if hasattr(_sig, "SIGALRM") else None
        import socket; _old_timeout = socket.getdefaulttimeout(); socket.setdefaulttimeout(15)
        try:
            hist = nse.fetch_historical_fno_data(
                symbol     = ticker,
                instrument = "FUTSTK",
                from_date  = from_dt,
                to_date    = to_dt,
            )
        finally:
            socket.setdefaulttimeout(_old_timeout)

        if not hist:
            return {"rollover_pct": 50.0, "rollover_signal": "moderate", "current_oi": 0, "prev_oi": 0}

        hist_df = pd.DataFrame(hist)

        # Find OI column
        oi_col = next((c for c in hist_df.columns
                       if "openinterest" in c.lower().replace(" ", "") or c.lower() == "oi"), None)
        exp_col = next((c for c in hist_df.columns
                        if "expiry" in c.lower()), None)

        if oi_col is None or exp_col is None:
            return {"rollover_pct": 50.0, "rollover_signal": "moderate", "current_oi": 0, "prev_oi": 0}

        hist_df[oi_col] = pd.to_numeric(hist_df[oi_col], errors="coerce").fillna(0)
        hist_df[exp_col] = pd.to_datetime(hist_df[exp_col], errors="coerce")
        hist_df = hist_df.dropna(subset=[exp_col])

        # Two most recent expiries
        expiries = sorted(hist_df[exp_col].dropna().unique())
        if len(expiries) < 2:
            return {"rollover_pct": 50.0, "rollover_signal": "moderate", "current_oi": 0, "prev_oi": 0}

        curr_exp = expiries[-1]
        prev_exp = expiries[-2]

        curr_oi = hist_df[hist_df[exp_col] == curr_exp][oi_col].iloc[-1] if not hist_df[hist_df[exp_col] == curr_exp].empty else 0
        prev_oi = hist_df[hist_df[exp_col] == prev_exp][oi_col].iloc[-1] if not hist_df[hist_df[exp_col] == prev_exp].empty else 0

        total = curr_oi + prev_oi
        rollover_pct = float(curr_oi / total * 100) if total > 0 else 50.0

        signal = "strong" if rollover_pct > 70 else ("moderate" if rollover_pct >= 40 else "weak")

        return {
            "rollover_pct"   : round(rollover_pct, 2),
            "rollover_signal": signal,
            "current_oi"     : int(curr_oi),
            "prev_oi"        : int(prev_oi),
        }

    except Exception as e:
        log.debug(f"  {ticker}: rollover calc failed â€” {e}")
        return {"rollover_pct": 50.0, "rollover_signal": "moderate", "current_oi": 0, "prev_oi": 0}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN FETCH LOOP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def fetch_all_fno_data(nse, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Main loop: for each F&O ticker, fetch option chain + derive all metrics.
    Returns dict of DataFrames keyed by metric name.
    """
    log.info(f"=== Fetching FnO data for {len(tickers)} tickers ===")
    log.info(f"    Estimated time: {len(tickers) * RATE_LIMIT_SLEEP / 60:.1f} minutes")

    oi_records       = []
    pcr_records      = []
    iv_records       = []
    maxpain_records  = []
    rollover_records = []

    ok_count   = 0
    fail_count = 0

    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0:
            log.info(f"  Progress: {i}/{len(tickers)} | OK={ok_count} FAIL={fail_count}")

        chain = fetch_option_chain(nse, ticker)
        time.sleep(RATE_LIMIT_SLEEP)

        if chain is None:
            fail_count += 1
            log.debug(f"  {ticker}: no chain data")
            continue

        ok_count += 1
        cmp      = float(chain.get("underlyingValue", 0))
        expiry   = chain.get("expiryDate", "")

        # â”€â”€ PCR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pcr_data = compute_pcr(chain)
        pcr_records.append({"ticker": ticker, "cmp": cmp, "expiry": expiry,
                             "fetch_date": str(TODAY_DATE), **pcr_data})

        # â”€â”€ OI Signals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        oi_data = compute_oi_signals(chain)
        oi_records.append({"ticker": ticker, "cmp": cmp, "expiry": expiry,
                            "fetch_date": str(TODAY_DATE), **oi_data})

        # â”€â”€ Max Pain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        mp_data = compute_max_pain(chain)
        maxpain_records.append({"ticker": ticker, "expiry": expiry,
                                 "fetch_date": str(TODAY_DATE), **mp_data})

        # ── IV (basic from chain) ────────────────────────────────
        iv_data = compute_iv_metrics(chain, ticker)
        # COMPUTE_IV_PERCENTILE: True = real IV scoring (+~0.5s/ticker, accurate Layer 3 scoring)
        #                        False = iv_percentile=50 for all (neutral, wastes 6% score weight)
        # Set True for overnight/weekly runs, False for intraday speed.
        COMPUTE_IV_PERCENTILE = False
        if COMPUTE_IV_PERCENTILE:
            iv_data['iv_percentile'] = enrich_iv_percentile(ticker, iv_data['atm_iv'], nse)
        iv_label = (
            "high"     if iv_data["iv_percentile"] > 80 else
            "elevated" if iv_data["iv_percentile"] > 50 else
            "moderate" if iv_data["iv_percentile"] > 30 else
            "low"
        )
        iv_data["iv_label"] = iv_label
        iv_records.append({"ticker": ticker, "cmp": cmp, "expiry": expiry,
                            "fetch_date": str(TODAY_DATE), **iv_data})

        # â”€â”€ Rollover â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Note: rollover fetch adds ~0.5-1s per ticker
        # Batched separately to keep main loop snappy
        # (handled in step below)

    # â”€â”€ Rollover: separate loop (historical data calls) â”€â”€
    log.info(f"  Fetching rollover data for {ok_count} tickers ...")
    rollover_tickers = [r["ticker"] for r in pcr_records]  # only successful ones

    for i, ticker in enumerate(rollover_tickers, 1):
        if i % 20 == 0:
            log.info(f"  Rollover progress: {i}/{len(rollover_tickers)}")
        rv_data = compute_rollover(ticker, nse)
        rollover_records.append({"ticker": ticker, "fetch_date": str(TODAY_DATE), **rv_data})
        time.sleep(0.5)   # lighter call, shorter sleep

    log.info(f"  Done. OK={ok_count} | FAIL={fail_count} | Rollover={len(rollover_records)}")

    return {
        "oi"      : pd.DataFrame(oi_records),
        "pcr"     : pd.DataFrame(pcr_records),
        "iv"      : pd.DataFrame(iv_records),
        "maxpain" : pd.DataFrame(maxpain_records),
        "rollover": pd.DataFrame(rollover_records),
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SAVE ALL PARQUETS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def save_parquets(data: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Save all 5 parquets, return row counts."""
    file_map = {
        "oi"      : "fno_oi.parquet",
        "pcr"     : "fno_pcr.parquet",
        "iv"      : "fno_iv.parquet",
        "maxpain" : "fno_maxpain.parquet",
        "rollover": "fno_rollover.parquet",
    }
    counts = {}
    for key, filename in file_map.items():
        df = data.get(key, pd.DataFrame())
        if not df.empty:
            path = BASE_PATH / filename
            df.to_parquet(path, index=False)
            counts[key] = len(df)
            log.info(f"  âœ… {filename}: {len(df)} rows")
        else:
            counts[key] = 0
            log.warning(f"  âš ï¸  {filename}: empty â€” not saved")
    return counts


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# REPORT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def print_report(data: dict[str, pd.DataFrame], counts: dict[str, int], elapsed: float):
    print("\n" + "=" * 60)
    print("LAYER 0 FNO â€” DATA FETCH REPORT")
    print("=" * 60)
    print(f"  fno_oi.parquet       : {counts.get('oi',0):>5} tickers")
    print(f"  fno_pcr.parquet      : {counts.get('pcr',0):>5} tickers")
    print(f"  fno_iv.parquet       : {counts.get('iv',0):>5} tickers")
    print(f"  fno_maxpain.parquet  : {counts.get('maxpain',0):>5} tickers")
    print(f"  fno_rollover.parquet : {counts.get('rollover',0):>5} tickers")
    print(f"  Runtime              : {elapsed:.1f}s")
    print("=" * 60)

    # PCR extremes
    if not data["pcr"].empty:
        df_pcr = data["pcr"]
        bullish = df_pcr[df_pcr["pcr"] > 1.2].sort_values("pcr", ascending=False).head(10)
        bearish = df_pcr[df_pcr["pcr"] < 0.7].sort_values("pcr").head(10)
        if not bullish.empty:
            print("\nBULLISH PCR (>1.2) â€” put-heavy, contrarian long signal:")
            print(bullish[["ticker", "pcr", "total_ce_oi", "total_pe_oi"]].to_string(index=False))
        if not bearish.empty:
            print("\nBEARISH PCR (<0.7) â€” call-heavy, resistance building:")
            print(bearish[["ticker", "pcr", "total_ce_oi", "total_pe_oi"]].to_string(index=False))

    # OI signals
    if not data["oi"].empty:
        df_oi = data["oi"]
        bullish_oi = df_oi[df_oi["oi_signal"] == "bullish"].sort_values("net_oi_bias", ascending=False).head(10)
        if not bullish_oi.empty:
            print("\nBULLISH OI BIAS â€” put support dominates:")
            print(bullish_oi[["ticker", "net_oi_bias", "put_oi_below_cmp", "call_oi_above_cmp"]].to_string(index=False))

    # Pinning risks
    if not data["maxpain"].empty:
        df_mp = data["maxpain"]
        pinning = df_mp[df_mp["pinning_risk"]].sort_values("max_pain_distance_pct")
        if not pinning.empty:
            print(f"\nâš ï¸  PINNING RISK â€” {len(pinning)} tickers within 2% of Max Pain:")
            print(pinning[["ticker", "cmp", "max_pain_strike", "max_pain_distance_pct"]].head(10).to_string(index=False))

    # High IV
    if not data["iv"].empty:
        df_iv = data["iv"]
        high_iv = df_iv[df_iv["iv_percentile"] > 80].sort_values("iv_percentile", ascending=False).head(10)
        if not high_iv.empty:
            print("\nðŸ”¥ HIGH IV (>80th pct) â€” vol expensive, caution on pre-earnings entries:")
            print(high_iv[["ticker", "atm_iv", "iv_percentile", "iv_label"]].to_string(index=False))

    # Strong rollover
    if not data["rollover"].empty:
        df_rv = data["rollover"]
        strong_rv = df_rv[df_rv["rollover_signal"] == "strong"].sort_values("rollover_pct", ascending=False).head(10)
        if not strong_rv.empty:
            print("\nâœ… STRONG ROLLOVER (>70%) â€” momentum carry-forward:")
            print(strong_rv[["ticker", "rollover_pct", "current_oi", "prev_oi"]].to_string(index=False))

    print("\nâœ… layer0_fno complete.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    import time as time_module
    t0 = time_module.time()

    log.info("=" * 60)
    log.info(f"layer0_fno.py v1.0  |  {TODAY.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    nse = get_nse_client()

    # Step 1: Get F&O tickers
    tickers = get_fno_tickers(nse)
    if not tickers:
        log.error("No F&O tickers found. Exiting.")
        return

    # Step 2: Fetch all FnO data
    data = fetch_all_fno_data(nse, tickers)

    # Step 3: Save parquets
    counts = save_parquets(data)

    # Step 4: Report
    elapsed = time_module.time() - t0
    print_report(data, counts, elapsed)

    try:
        nse.exit()
    except Exception:
        pass


if __name__ == "__main__":
    main()






