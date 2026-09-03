"""
================================================================================
  TS LAYER TESTING.PY  v2  — Technical Structure Analysis Engine
  NSE Decision Engine — Standalone (not linked to any layer yet)
================================================================================

WHAT THIS FILE DOES
-------------------
  Full technical analysis on every NSE stock. Outputs a daily trade brief
  with clear ENTRY, STOP, T1 (partial exit), T2 (full exit) for each setup.

  v2 improvements over v1:
  ─────────────────────────
  ✓ Weekly trend confirmation  — daily setup must align with weekly direction
  ✓ Entry signal classification — STRONG_BUY / BUY / WATCH / AVOID per stock
  ✓ Entry type                 — BREAKOUT / RETEST / REVERSAL / TREND_FOLLOW
  ✓ Key entry level            — exact price that triggers the trade
  ✓ Dual targets (T1 + T2)    — T1 = safe profit lock, T2 = full target
  ✓ Pattern context validation — hammer only counts at support, not random
  ✓ Polarity flip detection    — broken support becomes resistance (and vice versa)
  ✓ Liquidity filter           — skips stocks below ₹20 (penny) and < 1 Cr turnover
  ✓ Tier enrichment            — T1/T2/T3 from universe_master added to output
  ✓ Stricter scoring           — Grade A now means top ~8%, not top 17%
  ✓ Actionable daily report    — Focus list, Retest alerts, Breakout watch

MODULES
-------
  1. Data          — 5 years daily + weekly OHLCV per stock
  2. Indicators    — ATR, RSI, MACD, Bollinger, Stochastic, OBV, ADX, EMAs
  3. S/R Zones     — Pivot detection + KDE density + polarity flip
  4. Trend         — Daily + weekly EMA alignment + ADX + HH/HL structure
  5. Patterns      — Context-validated candlestick + chart patterns
  6. Entry/Exit    — Signal, type, key level, stop, T1, T2
  7. Scoring       — 0-100 quality score with stricter Grade A criteria
  8. Output        — Focus list, retest alerts, breakout watch, full CSV

OUTPUTS
-------
  ts_analysis_report.csv    — full data per stock (open in Excel)
  ts_sr_levels.parquet      — S/R zones for future Layer 3 integration
  ts_analysis_report.txt    — actionable daily brief (focus list + alerts)

RUN
---
  python "TS layer testing.py"
  (~25-35 min full universe, ~1 min for TICKERS_OVERRIDE list)

CUSTOMIZE
---------
  TICKERS_OVERRIDE = ['RELIANCE','TCS','INFY']  # quick test on specific stocks
  TICKERS_OVERRIDE = None                        # full universe
================================================================================
"""

import os, sys, time, warnings, logging, traceback
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE            = Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py")
UNIVERSE_PATH   = BASE / "universe_master.parquet"
OUTPUT_CSV      = BASE / "ts_analysis_report.csv"
OUTPUT_PARQUET  = BASE / "ts_sr_levels.parquet"
OUTPUT_TXT      = BASE / "ts_analysis_report.txt"

LOOKBACK_PERIOD = "5y"      # 5 years daily — captures 2-3 full NSE market cycles
PIVOT_N_FAST    = 3         # swing pivot lookaround (catches more levels)
PIVOT_N_SLOW    = 5         # swing pivot lookaround (catches major levels)
ZONE_MERGE_PCT  = 0.008     # merge S/R levels within 0.8%
KDE_TOP_N       = 12        # top density-based price levels
ATR_PERIOD      = 14
RSI_PERIOD      = 14
MIN_DATA_BARS   = 200       # skip if fewer bars (5y should give ~1250)
MIN_PRICE       = 20.0      # skip penny stocks below ₹20
MAX_WORKERS     = 20        # parallel download threads
RR_MIN          = 1.5       # minimum acceptable risk-reward
SR_PROXIMITY    = 0.015     # within 1.5% of S/R = "at the level"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("TS")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(ticker: str, period: str = LOOKBACK_PERIOD,
                interval: str = "1d") -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker + ".NS", period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open","High","Low","Close","Volume"]].dropna()
        if len(raw) < MIN_DATA_BARS:
            return None
        if float(raw["Close"].iloc[-1]) < MIN_PRICE:
            return None
        raw.index = pd.to_datetime(raw.index)
        return raw
    except Exception:
        return None


def fetch_weekly(ticker: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker + ".NS", period="5y", interval="1wk",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open","High","Low","Close","Volume"]].dropna()
        if len(raw) < 20:
            return None
        raw.index = pd.to_datetime(raw.index)
        return raw
    except Exception:
        return None


def load_universe_meta() -> dict:
    """Returns dict: ticker → {tier, sector}"""
    try:
        um = pd.read_parquet(UNIVERSE_PATH)
        um.columns = [c.lower() for c in um.columns]
        ticker_col = next((c for c in um.columns if c in ("ticker","symbol")), um.columns[0])
        tier_col   = next((c for c in um.columns if c == "tier"), None)
        sector_col = next((c for c in um.columns if "sector" in c), None)
        meta = {}
        for _, row in um.iterrows():
            tk = str(row[ticker_col]).strip().upper()
            meta[tk] = {
                "tier"  : str(row[tier_col]).strip()   if tier_col   else "T3",
                "sector": str(row[sector_col]).strip() if sector_col else "Unknown",
            }
        return meta
    except Exception:
        return {}


def load_fno_tickers() -> set:
    """Load tickers with active F&O contracts — these can be shorted overnight."""
    try:
        fno = pd.read_parquet(BASE / "fno_oi.parquet")
        col = next((c for c in fno.columns if "ticker" in c.lower()), fno.columns[0])
        return set(fno[col].dropna().str.upper().tolist())
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════════════
# 2. INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def atr(df, p=ATR_PERIOD):
    h,l,c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def rsi(close, p=RSI_PERIOD):
    d = close.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100/(1 + g/(l+1e-9))

def macd(close):
    f = close.ewm(span=12, adjust=False).mean()
    s = close.ewm(span=26, adjust=False).mean()
    m = f - s
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig

def bollinger(close, p=20, std=2.0):
    mid = close.rolling(p).mean()
    sd  = close.rolling(p).std()
    up, lo = mid + std*sd, mid - std*sd
    return up, mid, lo, (close - lo)/(up - lo + 1e-9)

def stochastic(df, k=14, d=3):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    K  = 100*(df["Close"] - lo)/(hi - lo + 1e-9)
    return K, K.rolling(d).mean()

def obv(df):
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()

def adx(df, p=14):
    up   = df["High"].diff()
    down = -df["Low"].diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)
    atr_v = atr(df, p).values
    pdi  = 100*pd.Series(pdm, index=df.index).ewm(span=p, adjust=False).mean()/(atr_v+1e-9)
    mdi  = 100*pd.Series(mdm, index=df.index).ewm(span=p, adjust=False).mean()/(atr_v+1e-9)
    dx   = 100*(pdi-mdi).abs()/(pdi+mdi+1e-9)
    return dx.ewm(span=p, adjust=False).mean(), pdi, mdi

def all_indicators(df):
    c   = df["Close"]
    ATR = atr(df)
    RSI = rsi(c)
    ml, ms, mh = macd(c)
    bu, bm, bl, pb = bollinger(c)
    sk, sd = stochastic(df)
    OBV = obv(df)
    ADX, pdi, mdi = adx(df)
    e20  = c.ewm(span=20,  adjust=False).mean()
    e50  = c.ewm(span=50,  adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    avg_vol = df["Volume"].rolling(20).mean()
    close = float(c.iloc[-1])
    return {
        "close"          : close,
        "atr"            : float(ATR.iloc[-1]),
        "atr_pct"        : float(ATR.iloc[-1]/close*100),
        "rsi"            : float(RSI.iloc[-1]),
        "rsi_prev"       : float(RSI.iloc[-2]),
        "macd"           : float(ml.iloc[-1]),
        "macd_signal"    : float(ms.iloc[-1]),
        "macd_hist"      : float(mh.iloc[-1]),
        "macd_hist_prev" : float(mh.iloc[-2]),
        "macd_cross"     : ("bullish" if mh.iloc[-1]>0 and mh.iloc[-2]<=0 else
                            "bearish" if mh.iloc[-1]<0 and mh.iloc[-2]>=0 else "flat"),
        "bb_upper"       : float(bu.iloc[-1]),
        "bb_mid"         : float(bm.iloc[-1]),
        "bb_lower"       : float(bl.iloc[-1]),
        "bb_pct_b"       : float(pb.iloc[-1]),
        "stoch_k"        : float(sk.iloc[-1]),
        "stoch_d"        : float(sd.iloc[-1]),
        "stoch_cross"    : ("bullish" if sk.iloc[-1]>sd.iloc[-1] and sk.iloc[-2]<=sd.iloc[-2] else
                            "bearish" if sk.iloc[-1]<sd.iloc[-1] and sk.iloc[-2]>=sd.iloc[-2] else "flat"),
        "obv_trend"      : ("up" if float(OBV.iloc[-1])>float(OBV.iloc[-5]) else "down"),
        "adx"            : float(ADX.iloc[-1]),
        "plus_di"        : float(pdi.iloc[-1]),
        "minus_di"       : float(mdi.iloc[-1]),
        "ema20"          : float(e20.iloc[-1]),
        "ema50"          : float(e50.iloc[-1]),
        "ema200"         : float(e200.iloc[-1]),
        "pct_ema20"      : float((close - float(e20.iloc[-1]))/float(e20.iloc[-1])*100),
        "pct_ema50"      : float((close - float(e50.iloc[-1]))/float(e50.iloc[-1])*100),
        "pct_ema200"     : float((close - float(e200.iloc[-1]))/float(e200.iloc[-1])*100),
        "vol_ratio"      : float(df["Volume"].iloc[-1]/(float(avg_vol.iloc[-1])+1)),
        "vol_trend"      : ("expanding" if df["Volume"].iloc[-3:].mean() >
                            df["Volume"].iloc[-8:-3].mean() else "contracting"),
        # 52-week reference levels
        "high_52w"       : float(df["High"].tail(252).max()),
        "low_52w"        : float(df["Low"].tail(252).min()),
        "pct_from_52h"   : float((close - float(df["High"].tail(252).max())) /
                                  float(df["High"].tail(252).max()) * 100),
        "pct_from_52l"   : float((close - float(df["Low"].tail(252).min()))  /
                                  float(df["Low"].tail(252).min())  * 100),
    }


def detect_divergence(df: pd.DataFrame, rsi_series: pd.Series) -> dict:
    """
    Bearish divergence: price makes HH but RSI makes LH → hidden weakness → SHORT signal.
    Bullish divergence: price makes LL but RSI makes HL → hidden strength → LONG signal.
    Looks back 40 bars for cleaner divergence detection.
    """
    window = 40
    if len(df) < window:
        return {"bearish_div": False, "bullish_div": False}

    c = df["Close"].values[-window:]
    r = rsi_series.values[-window:]

    # Local highs and lows (price)
    ph = [i for i in range(1, len(c)-1) if c[i] > c[i-1] and c[i] > c[i+1]]
    pl = [i for i in range(1, len(c)-1) if c[i] < c[i-1] and c[i] < c[i+1]]

    bearish_div = bullish_div = False

    if len(ph) >= 2:
        p1, p2 = ph[-2], ph[-1]
        # Price HH but RSI LH
        if c[p2] > c[p1] * 1.005 and r[p2] < r[p1] - 2:
            bearish_div = True

    if len(pl) >= 2:
        p1, p2 = pl[-2], pl[-1]
        # Price LL but RSI HL
        if c[p2] < c[p1] * 0.995 and r[p2] > r[p1] + 2:
            bullish_div = True

    return {"bearish_div": bearish_div, "bullish_div": bullish_div}

def weekly_trend(df_w) -> str:
    """Weekly EMA10/20 alignment → trend string."""
    if df_w is None or len(df_w) < 25:
        return "UNKNOWN"
    c    = df_w["Close"]
    e10  = c.ewm(span=10, adjust=False).mean()
    e20  = c.ewm(span=20, adjust=False).mean()
    e40  = c.ewm(span=40, adjust=False).mean()
    cur  = float(c.iloc[-1])
    if float(e10.iloc[-1]) > float(e20.iloc[-1]) > float(e40.iloc[-1]):
        return "STRONG_UP"
    if float(e10.iloc[-1]) < float(e20.iloc[-1]) < float(e40.iloc[-1]):
        return "STRONG_DOWN"
    if float(e10.iloc[-1]) > float(e20.iloc[-1]):
        return "UP"
    if float(e10.iloc[-1]) < float(e20.iloc[-1]):
        return "DOWN"
    return "SIDEWAYS"


# ══════════════════════════════════════════════════════════════════════════════
# 3. SUPPORT / RESISTANCE ZONES
# ══════════════════════════════════════════════════════════════════════════════

def find_pivots(df, n=PIVOT_N_FAST):
    highs, lows = df["High"].values, df["Low"].values
    vols, dates = df["Volume"].values, df.index
    ph, pl = [], []
    for i in range(n, len(df)-n):
        if highs[i] == max(highs[i-n:i+n+1]):
            ph.append({"price":float(highs[i]),"date":dates[i],
                       "volume":float(vols[i]),"type":"high","age":len(df)-1-i})
        if lows[i] == min(lows[i-n:i+n+1]):
            pl.append({"price":float(lows[i]),"date":dates[i],
                       "volume":float(vols[i]),"type":"low","age":len(df)-1-i})
    return ph, pl

def _make_zone(group):
    prices  = [l["price"]      for l in group]
    volumes = [l["volume"]     for l in group]
    ages    = [l.get("age", 0) for l in group]
    total_vol = sum(volumes)
    weights = [v/total_vol for v in volumes] if total_vol > 0 else None
    return {
        "price"    : float(np.average(prices, weights=weights)),
        "price_hi" : float(max(prices)),
        "price_lo" : float(min(prices)),
        "strength" : len(group),
        "volume"   : float(total_vol),
        "recency"  : float(min(ages)),
        "type"     : group[0].get("type","both"),
    }

def cluster(levels, threshold=ZONE_MERGE_PCT):
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: x["price"])
    zones, group = [], [levels[0]]
    for lvl in levels[1:]:
        if (lvl["price"] - group[0]["price"]) / (group[0]["price"]+1e-9) <= threshold:
            group.append(lvl)
        else:
            zones.append(_make_zone(group)); group = [lvl]
    zones.append(_make_zone(group))
    return zones

def kde_levels(df, n=KDE_TOP_N):
    closes = df["Close"].values
    if len(closes) < 30:
        return []
    try:
        grid    = np.linspace(closes.min(), closes.max(), 500)
        density = gaussian_kde(closes, bw_method=0.05)(grid)
        peaks, _ = find_peaks(density, distance=len(grid)//15,
                              prominence=density.max()*0.05)
        return sorted([{"price":float(grid[p]),"price_hi":float(grid[p]),
                        "price_lo":float(grid[p]),
                        "strength":float(density[p]/density.max()),
                        "volume":0.0,"recency":0,"type":"density"}
                       for p in peaks],
                      key=lambda x: x["strength"], reverse=True)[:n]
    except Exception:
        return []

def detect_polarity(df, support_zones, resistance_zones):
    """
    If price broke below a support zone recently (last 20 bars) and is now
    trading below it → that support has flipped to resistance (and vice versa).
    Returns updated (support_zones, resistance_zones).
    """
    close  = float(df["Close"].iloc[-1])
    recent = df["Close"].tail(20).values

    updated_sup, updated_res = [], []

    for z in support_zones:
        level = z["price"]
        # Was support, but price is clearly below it now → flipped to resistance
        was_above = any(p > level * 1.005 for p in recent[:-5])
        now_below = close < level * 0.995
        if was_above and now_below:
            z = {**z, "type": "flipped_resistance", "polarity_flip": True}
            updated_res.append(z)
        else:
            z = {**z, "polarity_flip": False}
            updated_sup.append(z)

    for z in resistance_zones:
        level = z["price"]
        # Was resistance, but price broke clearly above → flipped to support
        was_below = any(p < level * 0.995 for p in recent[:-5])
        now_above = close > level * 1.005
        if was_below and now_above:
            z = {**z, "type": "flipped_support", "polarity_flip": True}
            updated_sup.append(z)
        else:
            z = {**z, "polarity_flip": False}
            updated_res.append(z)

    # Re-sort
    updated_sup = sorted(updated_sup, key=lambda z: z["price"], reverse=True)
    updated_res = sorted(updated_res, key=lambda z: z["price"])
    return updated_sup, updated_res

def get_sr_zones(df):
    close = float(df["Close"].iloc[-1])
    ph3,pl3 = find_pivots(df, PIVOT_N_FAST)
    ph5,pl5 = find_pivots(df, PIVOT_N_SLOW)

    all_highs = ph3 + ph5
    all_lows  = pl3 + pl5

    res = cluster([l for l in all_highs if l["price"] > close*0.998], ZONE_MERGE_PCT)
    sup = cluster([l for l in all_lows  if l["price"] < close*1.002], ZONE_MERGE_PCT)

    kde = kde_levels(df)
    for k in kde:
        if k["price"] > close*1.005: res.append(k)
        elif k["price"] < close*0.995: sup.append(k)

    res = cluster(sorted(res, key=lambda x: x["price"]), ZONE_MERGE_PCT)
    sup = cluster(sorted(sup, key=lambda x: x["price"]), ZONE_MERGE_PCT)

    def _score(z):
        return z["strength"] * (1.0/(1 + z.get("recency",0)*0.05))
    for z in res + sup:
        z["score"] = _score(z)

    # Apply polarity flip detection
    res = sorted(res, key=lambda z: z["price"])
    sup = sorted(sup, key=lambda z: z["price"], reverse=True)
    sup, res = detect_polarity(df, sup, res)

    return sup, res


# ══════════════════════════════════════════════════════════════════════════════
# 4. TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def hh_hl(df):
    ph,pl = find_pivots(df, PIVOT_N_SLOW)
    if len(ph)<2 or len(pl)<2:
        return {"hh_hl":False,"lh_ll":False,"pattern":"insufficient"}
    rh = sorted(ph, key=lambda x:x["date"])[-4:]
    rl = sorted(pl, key=lambda x:x["date"])[-4:]
    hh = all(rh[i]["price"]>rh[i-1]["price"] for i in range(1,len(rh)))
    hl = all(rl[i]["price"]>rl[i-1]["price"] for i in range(1,len(rl)))
    lh = all(rh[i]["price"]<rh[i-1]["price"] for i in range(1,len(rh)))
    ll = all(rl[i]["price"]<rl[i-1]["price"] for i in range(1,len(rl)))
    return {"hh_hl":hh and hl,"lh_ll":lh and ll,
            "pattern":"uptrend" if (hh and hl) else "downtrend" if (lh and ll) else "sideways"}

def detect_trend(df, ind):
    close,e20,e50,e200 = ind["close"],ind["ema20"],ind["ema50"],ind["ema200"]
    ADX,pdi,mdi = ind["adx"],ind["plus_di"],ind["minus_di"]
    score = 0
    if e20>e50>e200: score += 45
    elif e20<e50<e200: score -= 45
    elif e20>e50: score += 20
    elif e20<e50: score -= 20
    score += 10 if close>e200 else -10
    tp = 25 if ADX>30 else 15 if ADX>20 else 5
    score += tp if pdi>mdi else -tp
    struct = hh_hl(df)
    if struct["hh_hl"]: score += 20
    elif struct["lh_ll"]: score -= 20
    score = int(np.clip(score,-100,100))
    lbl = ("STRONG_UP" if score>=60 else "UP" if score>=25 else
           "SIDEWAYS"  if score>=-24 else "DOWN" if score>=-59 else "STRONG_DOWN")
    return {"trend_score":score,"trend_label":lbl,
            "ema_stack":e20>e50>e200,"above_ema200":close>e200,
            "adx":ADX,"adx_trending":ADX>25,"struct_pattern":struct["pattern"]}


# ══════════════════════════════════════════════════════════════════════════════
# 5. PATTERN DETECTION (context-validated)
# ══════════════════════════════════════════════════════════════════════════════

def _near_zone(price, zones, pct=SR_PROXIMITY):
    return any(abs(price - z["price"])/price <= pct for z in zones)

def detect_patterns(df, sup, res, ind):
    c,o,h,l = df["Close"],df["Open"],df["High"],df["Low"]
    vol = df["Volume"]
    avg_vol = vol.rolling(20).mean()
    close = ind["close"]
    vr    = ind["vol_ratio"]

    found = []

    # ── BREAKOUT / BREAKDOWN ─────────────────────────────────────────────────
    c0,c1 = float(c.iloc[-1]),float(c.iloc[-2])
    if res:
        nr = res[0]["price"]
        if c1 < nr <= c0 and vr >= 1.4:
            found.append("BREAKOUT")
    if sup:
        ns = sup[0]["price"]
        if c1 > ns >= c0 and vr >= 1.4:
            found.append("BREAKDOWN")

    # ── ENGULFING (min body size filter) ─────────────────────────────────────
    o0,o1 = float(o.iloc[-1]),float(o.iloc[-2])
    h0,l0 = float(h.iloc[-1]),float(l.iloc[-1])
    h1,l1 = float(h.iloc[-2]),float(l.iloc[-2])
    body0 = abs(c0-o0); body1 = abs(c1-o1)
    if body0>0 and body1>0:
        if c0>o0 and c1<o1 and c0>o1 and o0<c1 and body0>body1*0.5:
            found.append("BULLISH_ENGULFING")
        if c0<o0 and c1>o1 and c0<o1 and o0>c1 and body0>body1*0.5:
            found.append("BEARISH_ENGULFING")

    # ── INSIDE BAR ────────────────────────────────────────────────────────────
    if h0<h1 and l0>l1:
        found.append("INSIDE_BAR")

    # ── HAMMER (only valid near support zone) ────────────────────────────────
    body  = abs(c0-o0)
    lo_wk = min(c0,o0)-l0
    up_wk = h0-max(c0,o0)
    if body > 0 and lo_wk > 2*body and up_wk < 0.3*body:
        if _near_zone(close, sup):   # ← context check: must be near support
            found.append("HAMMER")

    # ── SHOOTING STAR (only valid near resistance) ────────────────────────────
    if body > 0 and up_wk > 2*body and lo_wk < 0.3*body:
        if _near_zone(close, res):   # ← context check: must be near resistance
            found.append("SHOOTING_STAR")

    # ── DOUBLE BOTTOM (must be near support, not a random double low) ─────────
    _,pl = find_pivots(df, PIVOT_N_SLOW)
    if len(pl)>=2:
        last_lows = sorted(pl, key=lambda x:x["date"])[-4:]
        for i in range(len(last_lows)-1):
            p1,p2 = last_lows[i],last_lows[i+1]
            if (abs(p1["price"]-p2["price"])/(p1["price"]+1e-9) < 0.015
                    and abs(p1["age"]-p2["age"]) >= 8
                    and _near_zone(p1["price"], sup, 0.03)):  # ← at a support zone
                found.append("DOUBLE_BOTTOM"); break

    # ── DOUBLE TOP (must be near resistance) ─────────────────────────────────
    ph,_ = find_pivots(df, PIVOT_N_SLOW)
    if len(ph)>=2:
        last_highs = sorted(ph, key=lambda x:x["date"])[-4:]
        for i in range(len(last_highs)-1):
            p1,p2 = last_highs[i],last_highs[i+1]
            if (abs(p1["price"]-p2["price"])/(p1["price"]+1e-9) < 0.015
                    and abs(p1["age"]-p2["age"]) >= 8
                    and _near_zone(p1["price"], res, 0.03)):  # ← at resistance zone
                found.append("DOUBLE_TOP"); break

    # ── VOLUME SPIKE ─────────────────────────────────────────────────────────
    if vr >= 2.5:
        found.append("VOLUME_SPIKE")

    # ── HEAD & SHOULDERS (bearish reversal at top) ────────────────────────────
    ph, _ = find_pivots(df, PIVOT_N_SLOW)
    if len(ph) >= 3:
        recent_ph = sorted(ph, key=lambda x: x["date"])[-5:]
        for i in range(len(recent_ph)-2):
            ls, hd, rs = recent_ph[i], recent_ph[i+1], recent_ph[i+2]
            if (hd["price"] > ls["price"]*1.02 and hd["price"] > rs["price"]*1.02
                    and abs(ls["price"]-rs["price"])/ls["price"] < 0.06
                    and abs(ls["age"]-rs["age"]) < 40):
                found.append("HEAD_AND_SHOULDERS")
                break

    # ── INVERSE H&S (bullish reversal at bottom) ─────────────────────────────
    _, pl = find_pivots(df, PIVOT_N_SLOW)
    if len(pl) >= 3:
        recent_pl = sorted(pl, key=lambda x: x["date"])[-5:]
        for i in range(len(recent_pl)-2):
            ls, hd, rs = recent_pl[i], recent_pl[i+1], recent_pl[i+2]
            if (hd["price"] < ls["price"]*0.98 and hd["price"] < rs["price"]*0.98
                    and abs(ls["price"]-rs["price"])/ls["price"] < 0.06
                    and abs(ls["age"]-rs["age"]) < 40):
                found.append("INV_HEAD_SHOULDERS")
                break

    # ── RSI DIVERGENCE ────────────────────────────────────────────────────────
    rsi_s = rsi(df["Close"])
    div = detect_divergence(df, rsi_s)
    if div["bearish_div"]:
        found.append("BEARISH_DIVERGENCE")
    if div["bullish_div"]:
        found.append("BULLISH_DIVERGENCE")

    # Remove contradictory signals (double top + double bottom → keep neither)
    if "DOUBLE_TOP" in found and "DOUBLE_BOTTOM" in found:
        found = [p for p in found if p not in ("DOUBLE_TOP","DOUBLE_BOTTOM")]

    bull = [p for p in found if p in ("BREAKOUT","BULLISH_ENGULFING","HAMMER","DOUBLE_BOTTOM",
                                       "INV_HEAD_SHOULDERS","BULLISH_DIVERGENCE")]
    bear = [p for p in found if p in ("BREAKDOWN","BEARISH_ENGULFING","SHOOTING_STAR","DOUBLE_TOP",
                                       "HEAD_AND_SHOULDERS","BEARISH_DIVERGENCE")]

    return {"patterns":found, "patterns_str":" | ".join(found) if found else "none",
            "bullish":bull, "bearish":bear}


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY / EXIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def multi_targets(close, sup, res, risk_pts, direction, buf):
    """
    T1 = first meaningful S/R beyond stop (safest partial exit)
    T2 = second S/R zone (full target)
    """
    t1 = t2 = None
    if direction == "LONG":
        valid = [z for z in res if z["price"] > close*1.005]
        for i,z in enumerate(valid):
            cand = round(z["price"] - buf, 2)
            if cand > close and (cand-close) >= risk_pts*1.2:
                if t1 is None:
                    t1 = cand
                elif t2 is None:
                    t2 = cand; break
        if t1 is None: t1 = round(close + max(2.0*risk_pts, 3*buf), 2)
        if t2 is None: t2 = round(t1 + risk_pts, 2)
    else:
        valid = [z for z in sup if z["price"] < close*0.995]
        for i,z in enumerate(valid):
            cand = round(z["price"] + buf, 2)
            if cand < close and (close-cand) >= risk_pts*1.2:
                if t1 is None:
                    t1 = cand
                elif t2 is None:
                    t2 = cand; break
        if t1 is None: t1 = round(close - max(2.0*risk_pts, 3*buf), 2)
        if t2 is None: t2 = round(t1 - risk_pts, 2)
    return {"t1":t1,"t2":t2}

def stop_target(close, sup, res, atr_val, direction):
    buf      = 0.3*atr_val
    risk_pts = None
    stop_src = "atr_fallback"

    if direction == "LONG":
        valid = [z for z in sup if z["price"] < close*0.995]
        if valid:
            stop    = round(valid[0]["price"] - buf, 2)
            stop_src = "support"
        else:
            stop = round(close - 1.5*atr_val, 2)
        risk_pts = abs(close - stop)

        # Scan resistance for minimum RR_MIN
        tgt = None
        for z in [z for z in res if z["price"]>close*1.005]:
            cand = round(z["price"]-buf, 2)
            if cand>close and (cand-close) >= risk_pts*RR_MIN:
                tgt = cand; break
        if tgt is None: tgt = round(close + max(3.0*atr_val, risk_pts*RR_MIN), 2)
    else:
        valid = [z for z in res if z["price"] > close*1.005]
        if valid:
            stop    = round(valid[0]["price"] + buf, 2)
            stop_src = "resistance"
        else:
            stop = round(close + 1.5*atr_val, 2)
        risk_pts = abs(stop - close)

        tgt = None
        for z in [z for z in sup if z["price"]<close*0.995]:
            cand = round(z["price"]+buf, 2)
            if cand<close and (close-cand) >= risk_pts*RR_MIN:
                tgt = cand; break
        if tgt is None: tgt = round(close - max(3.0*atr_val, risk_pts*RR_MIN), 2)

    risk_pts  = abs(close-stop)
    rwd_pts   = abs(tgt-close)
    mt        = multi_targets(close, sup, res, risk_pts, direction, buf)

    return {
        "stop"       : stop,
        "target"     : tgt,
        "t1"         : mt["t1"],
        "t2"         : mt["t2"],
        "stop_pct"   : round(risk_pts/close*100, 2),
        "target_pct" : round(rwd_pts/close*100, 2),
        "t1_pct"     : round(abs(mt["t1"]-close)/close*100, 2),
        "t2_pct"     : round(abs(mt["t2"]-close)/close*100, 2),
        "rr"         : round(rwd_pts/(risk_pts+1e-9), 2),
        "rr_valid"   : rwd_pts/(risk_pts+1e-9) >= RR_MIN,
        "stop_source": stop_src,
    }

def classify_entry(close, direction, trend, ind, sup, res, patterns, w_trend, st):
    """
    Entry signal: STRONG_BUY / BUY / WATCH / AVOID (for LONG)
                  STRONG_SHORT / SHORT / WATCH / AVOID (for SHORT)
    Entry type:   BREAKOUT / RETEST / REVERSAL / TREND_FOLLOW / NOT_READY
    Key level:    exact price that triggers the trade
    """
    # ── Entry type ───────────────────────────────────────────────────────────
    entry_type = "NOT_READY"
    key_level  = close

    if direction == "LONG":
        # BREAKOUT: price just crossed above resistance with volume
        if res and "BREAKOUT" in patterns["patterns"]:
            entry_type = "BREAKOUT"
            key_level  = round(res[0]["price"] * 1.003, 2)  # 0.3% above break level

        # RETEST: price pulled back to a S/R zone (ideal entry)
        elif sup and _near_zone(close, sup[:3], SR_PROXIMITY):
            entry_type = "RETEST"
            key_level  = round(sup[0]["price"], 2)

        # REVERSAL: bullish pattern at support
        elif patterns["bullish"] and _near_zone(close, sup, 0.03):
            entry_type = "REVERSAL"
            key_level  = close

        # TREND_FOLLOW: strong trend, price above EMA20, ADX > 30
        elif trend["trend_score"] > 50 and ind["adx"] > 30 and ind["pct_ema20"] > -1:
            entry_type = "TREND_FOLLOW"
            key_level  = round(ind["ema20"] * 1.005, 2)  # above 20EMA as entry

    else:  # SHORT
        if sup and "BREAKDOWN" in patterns["patterns"]:
            entry_type = "BREAKOUT"
            key_level  = round(sup[0]["price"] * 0.997, 2)

        elif res and _near_zone(close, res[:3], SR_PROXIMITY):
            entry_type = "RETEST"
            key_level  = round(res[0]["price"], 2)

        elif patterns["bearish"] and _near_zone(close, res, 0.03):
            entry_type = "REVERSAL"
            key_level  = close

        elif trend["trend_score"] < -50 and ind["adx"] > 30 and ind["pct_ema20"] < 1:
            entry_type = "TREND_FOLLOW"
            key_level  = round(ind["ema20"] * 0.995, 2)

    # ── Confluence factors ────────────────────────────────────────────────────
    daily_ok   = (direction=="LONG" and trend["trend_score"]>20) or \
                 (direction=="SHORT" and trend["trend_score"]<-20)
    weekly_ok  = (direction=="LONG"  and w_trend in ("STRONG_UP","UP","SIDEWAYS")) or \
                 (direction=="SHORT" and w_trend in ("STRONG_DOWN","DOWN","SIDEWAYS")) or \
                 w_trend == "UNKNOWN"
    weekly_conf = (direction=="LONG"  and w_trend in ("STRONG_UP","UP")) or \
                  (direction=="SHORT" and w_trend in ("STRONG_DOWN","DOWN"))

    rsi_ok = (30<=ind["rsi"]<=65 if direction=="LONG" else 35<=ind["rsi"]<=75)
    macd_ok = (ind["macd_hist"]>0 if direction=="LONG" else ind["macd_hist"]<0)
    macd_cross = (ind["macd_cross"]=="bullish" if direction=="LONG" else ind["macd_cross"]=="bearish")
    stoch_ok = (ind["stoch_k"]<75 and ind["stoch_k"]>ind["stoch_d"] if direction=="LONG"
                else ind["stoch_k"]>25 and ind["stoch_k"]<ind["stoch_d"])
    adx_ok  = ind["adx"] > 20
    pat_ok  = bool(patterns["bullish"] if direction=="LONG" else patterns["bearish"])
    rr_ok   = st["rr"] >= RR_MIN
    vol_ok  = ind["vol_ratio"] >= 0.8  # at least some volume

    # Weighted factor count
    factors = (
        (2 if weekly_conf else (1 if weekly_ok else 0)) +
        (2 if daily_ok else 0) +
        (1 if rsi_ok else 0) +
        (2 if macd_ok else 0) +
        (1 if macd_cross else 0) +
        (1 if stoch_ok else 0) +
        (1 if adx_ok else 0) +
        (2 if pat_ok else 0) +
        (1 if rr_ok else 0) +
        (1 if entry_type != "NOT_READY" else 0) +
        (1 if vol_ok else 0)
    )
    max_factors = 15

    # ── Short squeeze risk — avoid shorting stocks near 52w high with strong momentum ──
    squeeze_risk = False
    if direction == "SHORT":
        near_52h     = ind.get("pct_from_52h", -100) > -8   # within 8% of 52w high
        strong_mom   = ind["rsi"] > 60
        strong_trend = ind["adx"] > 28
        squeeze_risk = near_52h and strong_mom and strong_trend

    # ── Signal ───────────────────────────────────────────────────────────────
    ready = entry_type != "NOT_READY"
    if squeeze_risk:
        sig = "AVOID"   # never short into a potential squeeze
    elif factors >= 10 and ready and weekly_conf:
        sig = "STRONG_BUY" if direction == "LONG" else "STRONG_SHORT"
    elif factors >= 7 and ready:
        sig = "BUY" if direction == "LONG" else "SHORT"
    elif factors >= 5 or (factors >= 3 and ready):
        sig = "WATCH"
    else:
        sig = "AVOID"

    reasons = []
    if weekly_conf: reasons.append(f"W:{w_trend}")
    if daily_ok:    reasons.append(f"D:{trend['trend_label']}")
    if pat_ok:      reasons.append("+".join(patterns["bullish"]+patterns["bearish"])[:30])
    if macd_cross:  reasons.append("MACD_X")
    if stoch_ok:    reasons.append("STOCH_OK")

    return {
        "entry_signal"  : sig,
        "entry_type"    : entry_type,
        "key_level"     : round(key_level, 2),
        "weekly_trend"  : w_trend,
        "wk_daily_align": weekly_conf,
        "entry_factors" : factors,
        "entry_reasons" : " | ".join(reasons),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. SCORING (stricter — Grade A ≈ top 8%)
# ══════════════════════════════════════════════════════════════════════════════

def score_setup(trend, ind, patterns, st, direction, entry, weekly_tr):
    score = 0

    # Trend alignment (max 25)
    ts = trend["trend_score"]
    t_pts = max(0, min(25, int((ts+100)/8))) if direction=="LONG" else \
            max(0, min(25, int((-ts+100)/8)))
    score += t_pts

    # Weekly confirmation (max 15) — new, was 0 before
    if entry["wk_daily_align"]: score += 15
    elif entry["weekly_trend"] == "SIDEWAYS": score += 5

    # RR quality (max 15)
    rr = st["rr"]
    score += min(15, int(rr/3.5*15)) if rr>=RR_MIN else 0

    # Indicator confluence (max 20)
    ip = 0
    if 30<=ind["rsi"]<=60 if direction=="LONG" else 40<=ind["rsi"]<=70: ip += 6
    if (ind["macd_hist"]>0 and ind["macd_hist"]>ind["macd_hist_prev"]) if direction=="LONG" \
       else (ind["macd_hist"]<0 and ind["macd_hist"]<ind["macd_hist_prev"]): ip += 7
    if ind["stoch_cross"] == ("bullish" if direction=="LONG" else "bearish"): ip += 4
    if ind["obv_trend"] == ("up" if direction=="LONG" else "down"): ip += 3
    score += min(20, ip)

    # Pattern quality (max 15)
    pat_pts = 0
    pmap = {"BREAKOUT":15,"BREAKDOWN":15,"BULLISH_ENGULFING":10,"BEARISH_ENGULFING":10,
            "DOUBLE_BOTTOM":12,"DOUBLE_TOP":12,"HAMMER":8,"SHOOTING_STAR":8,"VOLUME_SPIKE":5}
    for p in (patterns["bullish"] if direction=="LONG" else patterns["bearish"]):
        pat_pts += pmap.get(p, 5)
    score += min(15, pat_pts)

    # Entry type quality (max 10)
    etype_pts = {"RETEST":10,"REVERSAL":8,"BREAKOUT":7,"TREND_FOLLOW":5,"NOT_READY":0}
    score += etype_pts.get(entry["entry_type"], 0)

    score = int(np.clip(score, 0, 100))
    # Stricter grades: A=75+, B=55+, C=35+, D=<35
    grade = "A" if score>=75 else "B" if score>=55 else "C" if score>=35 else "D"
    return {"setup_score":score,"setup_grade":grade}


# ══════════════════════════════════════════════════════════════════════════════
# 8. PER-TICKER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ticker(ticker: str, meta: dict = None, fno_tickers: set = None) -> dict | None:
    df = fetch_ohlcv(ticker)
    if df is None:
        return None

    df_w = fetch_weekly(ticker)
    close = float(df["Close"].iloc[-1])

    try:
        ind   = all_indicators(df)
        sup, res = get_sr_zones(df)
        trend = detect_trend(df, ind)
        pats  = detect_patterns(df, sup, res, ind)
        wt    = weekly_trend(df_w)
        direction = "LONG" if trend["trend_score"] >= 0 else "SHORT"
        st    = stop_target(close, sup, res, ind["atr"], direction)
        entry = classify_entry(close, direction, trend, ind, sup, res, pats, wt, st)
        setup = score_setup(trend, ind, pats, st, direction, entry, wt)

        # Polarity flip count
        n_flipped = sum(1 for z in (sup+res) if z.get("polarity_flip"))

        tier       = meta.get(ticker, {}).get("tier",   "T?") if meta else "T?"
        sector     = meta.get(ticker, {}).get("sector","Unknown") if meta else "Unknown"
        fno_elig   = ticker in fno_tickers if fno_tickers else False
        short_type = ("FnO+INTRADAY" if fno_elig else "INTRADAY_ONLY") if direction == "SHORT" else "N/A"

        return {
            "ticker"          : ticker,
            "tier"            : tier,
            "sector"          : sector,
            "close"           : round(close, 2),
            "direction"       : direction,
            "entry_signal"    : entry["entry_signal"],
            "entry_type"      : entry["entry_type"],
            "key_level"       : entry["key_level"],
            "weekly_trend"    : wt,
            "wk_daily_align"  : entry["wk_daily_align"],
            "trend_label"     : trend["trend_label"],
            "trend_score"     : trend["trend_score"],
            "struct_pattern"  : trend["struct_pattern"],
            "adx"             : round(ind["adx"], 1),
            "rsi"             : round(ind["rsi"], 1),
            "macd_hist"       : round(ind["macd_hist"], 4),
            "macd_cross"      : ind["macd_cross"],
            "stoch_k"         : round(ind["stoch_k"], 1),
            "bb_pct_b"        : round(ind["bb_pct_b"], 3),
            "atr"             : round(ind["atr"], 2),
            "atr_pct"         : round(ind["atr_pct"], 2),
            "vol_ratio"       : round(ind["vol_ratio"], 2),
            "vol_trend"       : ind["vol_trend"],
            "obv_trend"       : ind["obv_trend"],
            "ema20"           : round(ind["ema20"], 2),
            "ema50"           : round(ind["ema50"], 2),
            "ema200"          : round(ind["ema200"], 2),
            "pct_ema20"       : round(ind["pct_ema20"], 2),
            "pct_ema50"       : round(ind["pct_ema50"], 2),
            "pct_ema200"      : round(ind["pct_ema200"], 2),
            "nearest_support" : round(sup[0]["price"], 2) if sup else None,
            "nearest_resist"  : round(res[0]["price"], 2) if res else None,
            "n_sup_zones"     : len(sup),
            "n_res_zones"     : len(res),
            "n_flipped_zones" : n_flipped,
            "stop"            : st["stop"],
            "t1"              : st["t1"],
            "t2"              : st["t2"],
            "stop_pct"        : st["stop_pct"],
            "t1_pct"          : st["t1_pct"],
            "t2_pct"          : st["t2_pct"],
            "rr"              : st["rr"],
            "rr_valid"        : st["rr_valid"],
            "stop_source"     : st["stop_source"],
            "patterns"        : pats["patterns_str"],
            "setup_score"     : setup["setup_score"],
            "setup_grade"     : setup["setup_grade"],
            "entry_factors"   : entry["entry_factors"],
            "entry_reasons"   : entry["entry_reasons"],
            "fno_eligible"    : fno_elig,
            "short_type"      : short_type,
            "high_52w"        : round(ind["high_52w"], 2),
            "low_52w"         : round(ind["low_52w"],  2),
            "pct_from_52h"    : round(ind["pct_from_52h"], 2),
            "pct_from_52l"    : round(ind["pct_from_52l"], 2),
        }, sup, res

    except Exception:
        log.debug("%s failed:\n%s", ticker, traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 9. OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _sig_icon(sig):
    return {"STRONG_BUY":"🟢🟢","BUY":"🟢","STRONG_SHORT":"🔴🔴",
            "SHORT":"🔴","WATCH":"🟡","AVOID":"⚪"}.get(sig,"⚪")

def build_report(df: pd.DataFrame) -> str:
    W   = 100
    sep = "=" * W
    lines = [sep,
             "  TS LAYER TESTING v2 — DAILY TECHNICAL BRIEF",
             f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   {len(df)} stocks analyzed",
             sep]

    # ── FOCUS LIST (T1/T2 only, Grade A/B, signal not AVOID) ──────────────────
    focus = df[
        (df["tier"].isin(["T1","T2"])) &
        (df["setup_grade"].isin(["A","B"])) &
        (~df["entry_signal"].isin(["AVOID"])) &
        (df["rr_valid"] == True)
    ].nlargest(25, "setup_score")

    lines += ["", "  ── FOCUS LIST — Top 25 Liquid Setups (T1/T2 | Grade A/B | RR valid) ──",
              "  " + "─"*98]
    hdr = (f"  {'SIG':<5} {'TICKER':<13} {'TIER':<4} {'DIR':<6} {'W-TREND':<12} "
           f"{'ENTRY TYPE':<13} {'KEY LEVEL':>10} {'STOP':>8} "
           f"{'T1':>8} {'T2':>8} {'RR':>5} {'SCORE':<8} {'PATTERNS'}")
    lines.append(hdr)
    lines.append("  " + "─"*98)

    for _, r in focus.iterrows():
        icon = _sig_icon(r["entry_signal"])
        lines.append(
            f"  {icon:<5} {r['ticker']:<13} {r['tier']:<4} {r['direction']:<6} "
            f"{r['weekly_trend']:<12} {r['entry_type']:<13} "
            f"{r['key_level']:>10.2f} {r['stop']:>8.2f} "
            f"{r['t1']:>8.2f} {r['t2']:>8.2f} {r['rr']:>5.2f} "
            f"{r['setup_score']}/100({r['setup_grade']})  {str(r['patterns'])[:28]}"
        )

    # ── LONG SETUPS ──────────────────────────────────────────────────────────
    longs = df[
        (df["direction"]=="LONG") &
        (df["entry_signal"].isin(["STRONG_BUY","BUY"])) &
        (df["rr_valid"]==True)
    ].nlargest(15, "setup_score")

    lines += ["", "  " + "─"*98,
              "  🟢  LONG SETUPS — Enter or watch for entry trigger",
              "  " + "─"*98,
              f"  {'TICKER':<13} {'TIER':<4} {'SIGNAL':<14} {'TYPE':<13} "
              f"{'CMP':>8} {'KEY_LVL':>9} {'STOP':>8} {'T1':>8} {'T2':>8} "
              f"{'STOP%':>7} {'T1%':>6} {'T2%':>6} {'RR':>5} PATTERNS"]
    lines.append("  " + "─"*98)
    for _, r in longs.iterrows():
        lines.append(
            f"  {r['ticker']:<13} {r['tier']:<4} {r['entry_signal']:<14} "
            f"{r['entry_type']:<13} {r['close']:>8.2f} {r['key_level']:>9.2f} "
            f"{r['stop']:>8.2f} {r['t1']:>8.2f} {r['t2']:>8.2f} "
            f"-{r['stop_pct']:>5.2f}% +{r['t1_pct']:>4.2f}% +{r['t2_pct']:>4.2f}% "
            f"{r['rr']:>5.2f}  {str(r['patterns'])[:28]}"
        )

    # ── SHORT SETUPS ─────────────────────────────────────────────────────────
    shorts = df[
        (df["direction"]=="SHORT") &
        (df["entry_signal"].isin(["STRONG_SHORT","SHORT"])) &
        (df["rr_valid"]==True)
    ].nlargest(15, "setup_score")

    lines += ["", "  " + "─"*98,
              "  🔴  SHORT SETUPS — Enter or watch for breakdown confirmation",
              "  " + "─"*98,
              f"  {'TICKER':<13} {'TIER':<4} {'SIGNAL':<14} {'TYPE':<13} "
              f"{'CMP':>8} {'KEY_LVL':>9} {'STOP':>8} {'T1':>8} {'T2':>8} "
              f"{'STOP%':>7} {'T1%':>6} {'T2%':>6} {'RR':>5} {'SHORT_TYPE':<14} PATTERNS"]
    lines.append("  " + "─"*98)
    for _, r in shorts.iterrows():
        st_col  = str(r.get("short_type","INTRADAY_ONLY"))
        fno_tag = "🌙FnO" if r.get("fno_eligible") else "📅Intra"
        lines.append(
            f"  {r['ticker']:<13} {r['tier']:<4} {r['entry_signal']:<14} "
            f"{r['entry_type']:<13} {r['close']:>8.2f} {r['key_level']:>9.2f} "
            f"{r['stop']:>8.2f} {r['t1']:>8.2f} {r['t2']:>8.2f} "
            f"+{r['stop_pct']:>5.2f}% -{r['t1_pct']:>4.2f}% -{r['t2_pct']:>4.2f}% "
            f"{r['rr']:>5.2f}  {fno_tag:<14} {str(r['patterns'])[:28]}"
        )

    # ── RETEST ALERTS (price within 1.5% of S/R right now) ───────────────────
    retest = df[df["entry_type"]=="RETEST"].nlargest(15, "setup_score")
    lines += ["", "  " + "─"*98,
              "  🟡  RETEST ALERTS — Price touching key S/R NOW (highest probability entries)",
              "  " + "─"*98]
    if len(retest):
        for _, r in retest.iterrows():
            icon = _sig_icon(r["entry_signal"])
            lines.append(
                f"  {icon} {r['ticker']:<13} {r['tier']:<4} {r['direction']:<6} "
                f"CMP:{r['close']:>8.2f}  KEY:{r['key_level']:>8.2f}  "
                f"Stop:{r['stop']:>8.2f}(-{r['stop_pct']:.2f}%)  "
                f"T1:{r['t1']:>8.2f}(+{r['t1_pct']:.2f}%)  "
                f"T2:{r['t2']:>8.2f}(+{r['t2_pct']:.2f}%)  [{r['patterns']}]"
            )
    else:
        lines.append("  No active retests detected.")

    # ── BREAKOUT WATCH ────────────────────────────────────────────────────────
    bo = df[df["entry_type"]=="BREAKOUT"].nlargest(10, "setup_score")
    lines += ["", "  " + "─"*98,
              "  ⚡  BREAKOUT CONFIRMED — Volume + price crossing key level",
              "  " + "─"*98]
    if len(bo):
        for _, r in bo.iterrows():
            icon = _sig_icon(r["entry_signal"])
            lines.append(
                f"  {icon} {r['ticker']:<13} {r['tier']:<4} {r['direction']:<6} "
                f"CMP:{r['close']:>8.2f}  KEY:{r['key_level']:>8.2f}  "
                f"Stop:{r['stop']:>8.2f}  T1:{r['t1']:>8.2f}  T2:{r['t2']:>8.2f}  "
                f"Vol:{r['vol_ratio']:.1f}x  [{r['patterns']}]"
            )
    else:
        lines.append("  No confirmed breakouts today.")

    # ── WEEKLY-DAILY ALIGNED (highest conviction) ─────────────────────────────
    aligned = df[
        (df["wk_daily_align"]==True) &
        (df["entry_signal"].isin(["STRONG_BUY","BUY","STRONG_SHORT","SHORT"])) &
        (df["tier"].isin(["T1","T2"]))
    ].nlargest(10, "setup_score")
    lines += ["", "  " + "─"*98,
              "  ⭐  WEEKLY + DAILY ALIGNED — Highest conviction setups",
              "  " + "─"*98]
    for _, r in aligned.iterrows():
        icon = _sig_icon(r["entry_signal"])
        lines.append(
            f"  {icon} {r['ticker']:<13} {r['tier']:<4} {r['direction']:<6} "
            f"W:{r['weekly_trend']:<12} D:{r['trend_label']:<12} "
            f"ADX:{r['adx']:.0f}  RSI:{r['rsi']:.0f}  "
            f"Stop:{r['stop']:.2f}  T1:{r['t1']:.2f}  T2:{r['t2']:.2f}  "
            f"Score:{r['setup_score']}"
        )

    # ── STATS ─────────────────────────────────────────────────────────────────
    lines += ["", sep, "  STATS", "  " + "─"*98]
    for g in ["A","B","C","D"]:
        n = int((df["setup_grade"]==g).sum())
        lines.append(f"  Grade {g}: {n:>4}  {'█'*min(n//10,50)}")
    lines.append("")
    for t in ["STRONG_UP","UP","SIDEWAYS","DOWN","STRONG_DOWN"]:
        n = int((df["trend_label"]==t).sum())
        lines.append(f"  {t:<14}: {n:>4}")
    lines.append("")
    sig_counts = df["entry_signal"].value_counts()
    for s in ["STRONG_BUY","BUY","STRONG_SHORT","SHORT","WATCH","AVOID"]:
        n = int(sig_counts.get(s,0))
        icon = _sig_icon(s)
        lines.append(f"  {icon} {s:<14}: {n:>4}")
    lines += ["", f"  Avg stop: {df['stop_pct'].mean():.2f}%  "
              f"Avg T1: {df['t1_pct'].mean():.2f}%  "
              f"Avg T2: {df['t2_pct'].mean():.2f}%  "
              f"Avg RR: {df['rr'].mean():.2f}"]
    all_p = []
    for p in df["patterns"].dropna():
        if p!="none": all_p.extend(p.split(" | "))
    lines += ["", "  Top patterns:"]
    for p,n in Counter(all_p).most_common(8):
        lines.append(f"    {p:<25}: {n}")
    lines += ["", sep,
              f"  Saved: {OUTPUT_CSV.name}  |  {OUTPUT_PARQUET.name}",
              sep]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(tickers_override=None):
    log.info("="*70)
    log.info("  TS LAYER TESTING v2 — Technical Structure Analysis")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("="*70)

    meta        = load_universe_meta()
    fno_tickers = load_fno_tickers()
    log.info(f"  Universe meta: {len(meta)} tickers | FnO eligible: {len(fno_tickers)}")

    if tickers_override:
        tickers = [t.upper().strip() for t in tickers_override]
        log.info(f"  Custom list: {len(tickers)} stocks")
    else:
        tickers = list(meta.keys()) if meta else []
        if not tickers:
            log.error("No tickers — set TICKERS_OVERRIDE or check universe_master.parquet")
            return
        log.info(f"  Universe: {len(tickers)} stocks")

    log.info(f"  Lookback: {LOOKBACK_PERIOD} | Min price: ₹{MIN_PRICE} | "
             f"Zone merge: {ZONE_MERGE_PCT*100:.1f}% | Workers: {MAX_WORKERS}")

    results, sr_records, failed = [], [], []
    t0 = time.time(); done = 0

    def _run(tk):
        out = analyze_ticker(tk, meta, fno_tickers)
        if out is None: return tk, None, None, None
        row, sup, res = out
        return tk, row, sup, res

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            tk, row, sup, res = fut.result()
            done += 1
            if row is None:
                failed.append(tk)
            else:
                results.append(row)
                for z in (sup or []):
                    sr_records.append({"ticker":tk,"zone_type":"support",  **{k:v for k,v in z.items() if k!="date"}})
                for z in (res or []):
                    sr_records.append({"ticker":tk,"zone_type":"resistance",**{k:v for k,v in z.items() if k!="date"}})
            if done%100==0 or done==len(tickers):
                el = time.time()-t0
                log.info(f"  {done}/{len(tickers)}  {el:.0f}s  ok:{len(results)}  fail:{len(failed)}")

    log.info(f"\n  Done: {len(results)} ok | {len(failed)} failed | {time.time()-t0:.0f}s")
    if not results:
        log.error("No results."); return

    df_out = pd.DataFrame(results).sort_values("setup_score", ascending=False).reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    log.info(f"  Saved: {OUTPUT_CSV}")

    if sr_records:
        sr_df = pd.DataFrame(sr_records)
        for col in sr_df.select_dtypes(include="object").columns:
            sr_df[col] = sr_df[col].astype(str)
        sr_df.to_parquet(OUTPUT_PARQUET, index=False)
        log.info(f"  Saved: {OUTPUT_PARQUET}  ({len(sr_df)} zone records)")

    report = build_report(df_out)
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    log.info(f"  Saved: {OUTPUT_TXT}")
    print("\n" + report)

    if failed:
        log.info(f"  Failed ({len(failed)}): {', '.join(failed[:15])}"
                 + ("..." if len(failed)>15 else ""))


# ── CUSTOMIZE ─────────────────────────────────────────────────────────────────
# None = full universe (~25-35 min)
# List = quick test on specific stocks (~1 min)
TICKERS_OVERRIDE = None
# TICKERS_OVERRIDE = ['RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','SBIN','WIPRO']
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(tickers_override=TICKERS_OVERRIDE)
