# -*- coding: utf-8 -*-
# NSE DECISION ENGINE - LAYER 2: DAILY UNIVERSE FILTER
# Reads all Layer 1 parquets, applies regime + sector pre-read,
# then runs 3 gates: Liquidity -> Event blackout -> R2 grade gate
# Output: nse_layer2_candidates.parquet + console table
# Place in: D:\MBA\STOCK MARKET RESEARCH\NSE quants py\
# Run: python nse_layer2_daily_filter.py

import os
import logging
import warnings
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("layer2_run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("L2")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

PARQUETS = {
    "universe":    os.path.join(BASE, "universe_master.parquet"),
    "fractal":     os.path.join(BASE, "fractal_dna.parquet"),
    "edges":       os.path.join(BASE, "stable_edges.parquet"),
    "clusters":    os.path.join(BASE, "clusters.parquet"),
    "r2_grades":   os.path.join(BASE, "r2_grades.parquet"),
    "garch":       os.path.join(BASE, "garch_params.parquet"),
    "cusum":       os.path.join(BASE, "cusum_breaks.parquet"),
    "crisis":      os.path.join(BASE, "crisis_alpha.parquet"),
    "tier_beta":   os.path.join(BASE, "tier_beta.parquet"),
    "earnings":    os.path.join(BASE, "earnings_dna.parquet"),
    "spreads":     os.path.join(BASE, "spread_history.parquet"),
    "sector_mom":  os.path.join(BASE, "sector_momentum.parquet"),
}

OUTPUT = os.path.join(BASE, "nse_layer2_candidates.parquet")
REPORT = os.path.join(BASE, "layer2_report.txt")

# ─────────────────────────────────────────────
# TIER THRESHOLDS (daily volume in INR crore)
# ─────────────────────────────────────────────
TIER_VOL = {"T1": 5.0, "T2": 1.0, "T3": 0.0}

# ─────────────────────────────────────────────
# REGIME THRESHOLDS
# ─────────────────────────────────────────────
VIX_CRISIS   = 22.0   # India VIX above this -> Crisis
VIX_BULL     = 14.0   # India VIX below this -> Bull
# Between 14-22 = Sideways

# ─────────────────────────────────────────────
# SLIPPAGE TABLE (round trip %)
# ─────────────────────────────────────────────
SLIPPAGE = {"T1": 0.0015, "T2": 0.0025, "T3": 0.0050}

# ─────────────────────────────────────────────
# HELPER: safe parquet loader
# ─────────────────────────────────────────────
def load(name, key):
    path = PARQUETS[key]
    if not os.path.exists(path):
        log.warning("  MISSING: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    log.info("  %-20s  %6d rows", os.path.basename(path), len(df))
    return df


# ═══════════════════════════════════════════════════════
# STEP 0: LOAD ALL PARQUETS
# ═══════════════════════════════════════════════════════
def load_all():
    log.info("")
    log.info("=" * 60)
    log.info("  NSE DECISION ENGINE - LAYER 2 DAILY FILTER")
    log.info("  %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)
    log.info("")
    log.info("[STEP 0] Loading Layer 1 parquets ...")

    data = {}
    for name, key in [
        ("universe",   "universe"),
        ("fractal",    "fractal"),
        ("clusters",   "clusters"),
        ("r2_grades",  "r2_grades"),
        ("garch",      "garch"),
        ("crisis",     "crisis"),
        ("earnings",   "earnings"),
        ("spreads",    "spreads"),
        ("sector_mom", "sector_mom"),
    ]:
        data[name] = load(name, name)

    # Normalise ticker column
    for k in data:
        df = data[k]
        if df.empty:
            continue
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if "ticker" not in cols and "symbol" in cols:
            df = df.rename(columns={"symbol": "ticker"})
        data[k] = df

    return data


# ═══════════════════════════════════════════════════════
# STEP 1: PRE-READ — REGIME + SECTOR BIAS
# ═══════════════════════════════════════════════════════
def read_regime_and_sector(data):
    log.info("")
    log.info("[STEP 1] Pre-read: Regime + Sector bias ...")

    # --- Regime via India VIX (fallback chain) ---
    regime  = "Sideways"
    vix_val = None

    # Try 1: yfinance
    try:
        vix = yf.download("^INDIAVIX", period="5d", progress=False, auto_adjust=True)
        if not vix.empty:
            close = vix["Close"]
            if hasattr(close, "columns"):  # yfinance >=0.2 returns MultiIndex columns
                close = close.iloc[:, 0]
            vix_val = float(close.dropna().iloc[-1])
            if vix_val <= 0:
                vix_val = None
    except Exception as e:
        log.warning("  VIX yfinance failed: %s", e)

    # Try 2: NSE package
    if vix_val is None:
        try:
            import datetime as _dt
            from nse import NSE
            from pathlib import Path as _Path
            _dl = _Path(r"D:\MBA\STOCK MARKET RESEARCH\NSE quants py\nse_data")
            _dl.mkdir(exist_ok=True)
            _nse = NSE(_dl)
            _rows = _nse.fetch_historical_vix_data(
                from_date=_dt.date.today() - _dt.timedelta(days=7),
                to_date=_dt.date.today(),
            )
            _nse.exit()
            if _rows:
                vix_val = float(_rows[-1]["EOD_CLOSE_INDEX_VAL"])
                if vix_val <= 0:
                    vix_val = None
            log.info("  VIX via NSE package: %.2f", vix_val)
        except Exception as e:
            log.warning("  VIX NSE fallback failed: %s", e)

    # Try 3: safe default — Sideways
    if vix_val is None:
        vix_val = 18.0
        log.warning("  VIX all fetches failed -- hardcoded 18.0 (Sideways)")

    # Classify regime
    if vix_val > VIX_CRISIS:
        regime = "Crisis"
    elif vix_val < VIX_BULL:
        regime = "Bull"
    else:
        regime = "Sideways"

    log.info("  India VIX = %.2f  -->  Regime = %s", vix_val or 0.0, regime)

    # --- Sector bias from sector_momentum.parquet ---
    sector_rank = {}
    sm = data.get("sector_mom", pd.DataFrame())
    if not sm.empty:
        # Expect columns: sector, return_20d (or similar)
        ret_col = [c for c in sm.columns if "return" in c or "ret" in c or "20" in c]
        if ret_col:
            rc = ret_col[0]
            sm_sorted = sm.sort_values(rc, ascending=False).reset_index(drop=True)
            sm_sorted["rank"] = sm_sorted.index + 1
            sector_rank = dict(zip(sm_sorted.get("sector", sm_sorted.iloc[:, 0]), sm_sorted["rank"]))
            top3 = sm_sorted[sm_sorted.iloc[:, 0] != ''].iloc[:3]
            bot3 = sm_sorted[sm_sorted.iloc[:, 0] != ''].iloc[-3:]
            log.info("  Top sectors (20d): %s", list(top3.iloc[:, 0]))
            log.info("  Bot sectors (20d): %s", list(bot3.iloc[:, 0]))
        else:
            log.warning("  sector_momentum.parquet has no return column")
    else:
        log.warning("  sector_momentum.parquet missing or empty")

    return regime, vix_val, sector_rank


# ═══════════════════════════════════════════════════════
# STEP 2: FETCH TODAY'S PRICES + VOLUMES
# ═══════════════════════════════════════════════════════
def fetch_today(universe):
    log.info("")
    log.info("[STEP 2] Fetching today's price + volume for %d tickers ...", len(universe))

    tickers = list(universe["ticker"].dropna().unique())
    nse_tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers]

    # Download last 5 days (handles weekends/holidays)
    try:
        raw = yf.download(
            nse_tickers,
            period="5d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        log.error("  yfinance batch download failed: %s", e)
        return pd.DataFrame()

    records = []
    for t, nt in zip(tickers, nse_tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[nt] if nt in raw.columns.get_level_values(0) else None
            else:
                sub = raw
            if sub is None or sub.empty:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            last = sub.iloc[-1]
            close  = float(last["Close"])
            volume = float(last["Volume"]) if "Volume" in last else 0.0
            # Approx daily turnover in crore
            turnover_cr = (close * volume) / 1e7
            records.append({
                "ticker":       t,
                "close":        close,
                "volume":       volume,
                "turnover_cr":  turnover_cr,
                "price_date":   sub.index[-1].date(),
            })
        except Exception:
            continue

    prices = pd.DataFrame(records)
    log.info("  Fetched prices for %d / %d tickers", len(prices), len(tickers))
    return prices


# ═══════════════════════════════════════════════════════
# STEP 3: GATE 1 — LIQUIDITY
# ═══════════════════════════════════════════════════════
def gate_liquidity(df, regime):
    log.info("")
    log.info("[GATE 1] Liquidity filter ...")
    before = len(df)

    def passes(row):
        tier = row.get("tier", "T3")
        threshold = TIER_VOL.get(tier, 0.0)
        # In Crisis, tighten T2/T3 thresholds by 2x
        if regime == "Crisis" and tier != "T1":
            threshold *= 2.0
        return row.get("turnover_cr", 0.0) >= threshold and row.get("close", 0.0) >= 25.0

    mask = df.apply(passes, axis=1)
    df = df[mask].copy()
    log.info("  %d -> %d stocks (dropped %d illiquid)", before, len(df), before - len(df))
    return df


# ═══════════════════════════════════════════════════════
# STEP 4: GATE 2 — EVENT BLACKOUT
# ═══════════════════════════════════════════════════════
def gate_event_blackout(df, earnings_df):
    log.info("")
    log.info("[GATE 2] Event blackout filter ...")
    log.info("  Earnings DNA columns: %s", list(earnings_df.columns))
    if not earnings_df.empty:
        log.info("  Earnings DNA sample : %s", earnings_df.head(1).to_dict('records'))
    before = len(df)

    # Build set of tickers with upcoming results in next 7 days
    # earnings_dna has: ticker, next_result_date (if available)
    blackout_set = set()
    today = datetime.date.today()
    if not earnings_df.empty:
        date_col = [c for c in earnings_df.columns if "date" in c.lower() or "next" in c.lower()]
        if date_col:
            dc = date_col[0]
            try:
                earnings_df[dc] = pd.to_datetime(earnings_df[dc], errors="coerce")
                # Blackout = results in 0-2 days only (too close to trade)
                # Results in 3-7 days = Rule D pre-result window (let through)
                window = earnings_df[
                    (earnings_df[dc].dt.date >= today) &
                    (earnings_df[dc].dt.date <= today + datetime.timedelta(days=2))
                ]
                blackout_set = set(window["ticker"].dropna())
                # Also merge days_to_result into earnings_df for Gate 3 use
                if "days_to_result" in earnings_df.columns:
                    earnings_df["_days"] = earnings_df["days_to_result"]
            except Exception:
                pass

    # reliable_drift flag = H5 exception (these PASS even in result week)
    reliable_set = set()
    if "reliable_drift" in earnings_df.columns:
        reliable_set = set(earnings_df[earnings_df["reliable_drift"] == True]["ticker"])

    def passes_blackout(row):
        t = row["ticker"]
        if t in blackout_set and t not in reliable_set:
            return False
        return True

    mask = df.apply(passes_blackout, axis=1)
    df = df[mask].copy()
    df["in_earnings_window"] = df["ticker"].isin(blackout_set)
    df["reliable_drift"]     = df["ticker"].isin(reliable_set)
    # Merge days_to_result so Gate 3 can use it for D-grade exception
    if not earnings_df.empty and "days_to_result" in earnings_df.columns:
        dtr = earnings_df[["ticker","days_to_result"]].drop_duplicates("ticker")
        df = df.merge(dtr, on="ticker", how="left")
        df["days_to_result"] = df["days_to_result"].fillna(999).astype(int)
    else:
        df["days_to_result"] = 999
    log.info("  Blackout set: %d tickers (0-2d)  |  Reliable-drift exception: %d", len(blackout_set), len(reliable_set))
    log.info("  %d -> %d stocks (dropped %d in event blackout)", before, len(df), before - len(df))
    return df


# ═══════════════════════════════════════════════════════
# STEP 5: GATE 3 — R2 GRADE GATE
# ═══════════════════════════════════════════════════════
def gate_r2(df, r2_df, regime):
    log.info("")
    log.info("[GATE 3] R2 grade gate (regime=%s) ...", regime)
    before = len(df)

    if not r2_df.empty and "ticker" in r2_df.columns:
        grade_col = [c for c in r2_df.columns if "grade" in c.lower()]
        if grade_col:
            # ── Normalize ticker format on both sides ──
            r2_df = r2_df.copy()
            r2_df["ticker"] = r2_df["ticker"].str.replace(".NS", "", regex=False).str.strip().str.upper()
            df["ticker"]    = df["ticker"].str.replace(".NS", "", regex=False).str.strip().str.upper()

            r2_map = dict(zip(r2_df["ticker"], r2_df[grade_col[0]]))

            # ── Diagnostic ──
            log.info("  R2 map size: %d", len(r2_map))
            log.info("  Sample r2_grades tickers : %s", list(r2_map.keys())[:5])
            log.info("  Sample df tickers        : %s", list(df["ticker"].head(5)))

            df["r2_grade"] = df["ticker"].map(r2_map).fillna("D")
            matched = (df["r2_grade"] != "D").sum()
            log.info("  Matched %d / %d tickers to r2_grades", matched, len(df))
        else:
            log.warning("  No grade column found in r2_grades.parquet — columns: %s", list(r2_df.columns))
            df["r2_grade"] = "B"
    else:
        log.warning("  r2_grades.parquet empty or missing ticker column")
        df["r2_grade"] = "B"

    def passes_r2(row):
        grade = row.get("r2_grade", "D")
        if grade in ("A", "B"):
            return True
        if grade == "C" and regime in ("Bull", "Sideways"):
            return True
        return False

    mask = df.apply(passes_r2, axis=1)
    df_filtered = df[mask].copy()

    # Never fully bypass — if too few pass, take best-liquidity subset
    # (prelim_score not computed yet at this stage — turnover_cr is best proxy)
    min_slots = {"Bull": 100, "Sideways": 150, "Crisis": 30}.get(regime, 150)
    if len(df_filtered) < 50:
        sort_col = "turnover_cr" if "turnover_cr" in df.columns else df.columns[0]
        df_filtered = df.nlargest(min_slots, sort_col).copy()
        log.warning(
            "  Gate 3 yielded only %d A/B/C stocks — using top-%d by %s "
            "(r2_grade kept as metadata, no full bypass)", len(df[mask]), min_slots, sort_col
        )
    log.info("  %d -> %d stocks after R2 gate", before, len(df_filtered))
    return df_filtered

# ═══════════════════════════════════════════════════════
# STEP 6: CRISIS GATE — overlay crisis alpha
# ═══════════════════════════════════════════════════════
def apply_crisis_gate(df, crisis_df, regime):
    log.info("")
    log.info("[CRISIS GATE] Regime = %s ...", regime)

    crisis_set = set()
    if not crisis_df.empty and "ticker" in crisis_df.columns:
        crisis_set = set(crisis_df["ticker"].dropna())

    df["crisis_alpha"] = df["ticker"].isin(crisis_set)

    if regime == "Crisis":
        # In Crisis: only crisis alpha stocks proceed
        before = len(df)
        df = df[df["crisis_alpha"]].copy()
        log.info("  Crisis regime: kept %d crisis-alpha stocks (dropped %d)", len(df), before - len(df))
    else:
        log.info("  Non-crisis: all %d candidates pass crisis gate", len(df))

    return df


# ═══════════════════════════════════════════════════════
# STEP 7: ENRICH — attach signal metadata
# ═══════════════════════════════════════════════════════
def enrich(df, data, sector_rank, regime, vix_val):
    log.info("")
    log.info("[STEP 7] Enriching candidates with signal metadata ...")

    # Fractal DNA
    frac = data.get("fractal", pd.DataFrame())
    if not frac.empty and "ticker" in frac.columns:
        hurst_col = [c for c in frac.columns if "hurst" in c.lower() or c == "h"]
        type_col  = [c for c in frac.columns if "type" in c.lower() or "dna" in c.lower()]
        if hurst_col:
            df = df.merge(frac[["ticker"] + hurst_col[:1] + (type_col[:1] if type_col else [])],
                          on="ticker", how="left")

    # GARCH sigma
    garch = data.get("garch", pd.DataFrame())
    if not garch.empty and "ticker" in garch.columns:
        sigma_col = [c for c in garch.columns if "sigma" in c.lower() or "vol" in c.lower() or "omega" in c.lower()]
        if sigma_col:
            df = df.merge(garch[["ticker", sigma_col[0]]], on="ticker", how="left")
            df = df.rename(columns={sigma_col[0]: "garch_sigma"})

    # Cluster ID
    clusters = data.get("clusters", pd.DataFrame())
    if not clusters.empty and "ticker" in clusters.columns:
        clust_col = [c for c in clusters.columns if "cluster" in c.lower() or "community" in c.lower()]
        if clust_col:
            df = df.merge(clusters[["ticker", clust_col[0]]], on="ticker", how="left")

    # Spread / pairs signal
    spreads = data.get("spreads", pd.DataFrame())
    pair_tickers = set()
    if not spreads.empty:
        acol = [c for c in spreads.columns if "ticker_a" in c.lower() or "a" == c.lower()]
        bcol = [c for c in spreads.columns if "ticker_b" in c.lower() or "b" == c.lower()]
        if acol and bcol:
            pair_tickers = set(spreads[acol[0]].tolist() + spreads[bcol[0]].tolist())
    df["pairs_eligible"] = df["ticker"].isin(pair_tickers)

    # Sector — permanent fallback from sector_cache.json so Unknown never propagates
    _cache_path = os.path.join(BASE, "sector_cache.json")
    if os.path.exists(_cache_path):
        try:
            import json as _json
            with open(_cache_path, "r", encoding="utf-8") as _f:
                _sector_cache = _json.load(_f)
            if "sector" not in df.columns:
                df["sector"] = "Unknown"
            _unknown_mask = df["sector"].isna() | df["sector"].str.strip().str.lower().isin(["", "unknown"])
            df.loc[_unknown_mask, "sector"] = df.loc[_unknown_mask, "ticker"].map(_sector_cache)
            df["sector"] = df["sector"].fillna("Unknown")
            _fixed = int(_unknown_mask.sum())
            if _fixed:
                log.info("  Sector fallback: resolved %d Unknown tickers from sector_cache.json", _fixed)
        except Exception as _e:
            log.warning("  sector_cache.json fallback failed: %s", _e)

    # Sector rank (lower = better)
    if "sector" in df.columns and sector_rank:
        df["sector_rank"] = df["sector"].map(sector_rank).fillna(999)
    else:
        df["sector_rank"] = 999

    # Slippage
    df["slippage_rt"] = df["tier"].map(SLIPPAGE).fillna(0.005)

    # Regime + VIX stamp
    df["regime"]  = regime
    df["vix"]     = round(vix_val, 2) if vix_val else None
    df["run_date"] = datetime.date.today().isoformat()

    log.info("  Final candidate columns: %s", list(df.columns))
    return df


# ═══════════════════════════════════════════════════════
# STEP 8: SCORE + RANK
# ═══════════════════════════════════════════════════════
def score_and_rank(df):
    log.info("")
    log.info("[STEP 8] Scoring and ranking candidates ...")

    # Tier weight
    tier_w = {"T1": 1.0, "T2": 0.7, "T3": 0.4}
    df["tier_weight"] = df["tier"].map(tier_w).fillna(0.4)

    # Regime multiplier
    regime_mult = {"Crisis": 0.3, "Bull": 1.2, "Sideways": 1.0}
    rm = regime_mult.get(df["regime"].iloc[0] if len(df) > 0 else "Sideways", 1.0)

    # MR strength: based on Hurst (lower = stronger MR)
    hurst_col = [c for c in df.columns if "hurst" in c.lower() or c == "h"]
    if hurst_col:
        h = df[hurst_col[0]].fillna(0.5)
        df["mr_strength"] = ((0.5 - h).clip(lower=0) * 2).clip(0, 1)
    else:
        df["mr_strength"] = 0.0

    # Sector bias score (rank 1 = top sector = score 1.0)
    max_rank = df["sector_rank"].max() if "sector_rank" in df.columns else 1
    if max_rank > 0:
        df["sector_score"] = 1.0 - (df.get("sector_rank", 999) - 1) / max(max_rank, 1)
    else:
        df["sector_score"] = 0.5

    # Pairs bonus
    df["pairs_score"] = df["pairs_eligible"].astype(float) * 0.841  # IC from Layer 1.5

    # Preliminary score (ml_score comes from Layer 3 Rule F)
    # For now: use available signals
    # final_score = 0.4*ml + 0.3*MR_strength + 0.2*Granger + 0.1*tier
    # ml_score placeholder = 0 until Layer 3 runs
    df["ml_score"]     = 0.0  # Layer 3 will fill this
    df["granger_conviction"] = 0.0  # Layer 3 Rule B will fill this

    df["prelim_score"] = (
        0.4 * df["ml_score"] +
        0.3 * df["mr_strength"] +
        0.2 * df["granger_conviction"] +
        0.1 * df["tier_weight"] +
        0.2 * df["pairs_score"] +       # bonus for pairs signal
        0.1 * df["sector_score"]
    ) * rm

    # Crisis alpha bonus
    if "crisis_alpha" in df.columns:
        df.loc[df["crisis_alpha"], "prelim_score"] *= 0.8

    df = df.sort_values("prelim_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    log.info("  Scored %d candidates | Top score: %.4f", len(df), df["prelim_score"].max() if len(df) else 0)
    return df


# ═══════════════════════════════════════════════════════
# STEP 9: OUTPUT
# ═══════════════════════════════════════════════════════
def save_and_report(df, regime, vix_val):
    log.info("")
    log.info("[STEP 9] Saving output ...")

    # Save parquet
    df.to_parquet(OUTPUT, index=False)
    log.info("  Saved: %s  (%d rows)", OUTPUT, len(df))

    # Console + file report
    sep  = "=" * 72
    sep2 = "-" * 72

    lines = []
    lines.append(sep)
    lines.append("  NSE DECISION ENGINE - LAYER 2 DAILY FILTER REPORT")
    lines.append("  Date  : %s" % datetime.date.today().isoformat())
    lines.append("  Regime: %-12s   VIX: %.2f" % (regime, vix_val or 0.0))
    lines.append("  Candidates passed all gates: %d" % len(df))
    lines.append(sep)
    lines.append("")

    # Gate summary
    lines.append("  GATE SUMMARY")
    lines.append(sep2)

    # Tier breakdown
    if "tier" in df.columns:
        tc = df["tier"].value_counts()
        lines.append("  Tier breakdown:")
        for tier in ["T1", "T2", "T3"]:
            lines.append("    %-4s : %d" % (tier, tc.get(tier, 0)))
    lines.append("")

    # Grade breakdown
    if "r2_grade" in df.columns:
        gc = df["r2_grade"].value_counts()
        lines.append("  R2 grade breakdown:")
        for g in ["A", "B", "C", "D"]:
            lines.append("    %-2s : %d" % (g, gc.get(g, 0)))
    lines.append("")

    # Pairs eligible
    if "pairs_eligible" in df.columns:
        n_pairs = df["pairs_eligible"].sum()
        lines.append("  Pairs-eligible candidates: %d" % n_pairs)
    lines.append("")

    # Top 20
    lines.append("  TOP 20 CANDIDATES")
    lines.append(sep2)
    show_cols = ["rank", "ticker", "tier", "r2_grade", "prelim_score",
                 "mr_strength", "pairs_eligible", "crisis_alpha", "sector_rank", "turnover_cr"]
    show_cols = [c for c in show_cols if c in df.columns]
    top20 = df[show_cols].head(20)
    lines.append(top20.to_string(index=False, float_format=lambda x: "%.4f" % x))
    lines.append("")
    lines.append(sep)
    lines.append("  Output: nse_layer2_candidates.parquet")
    lines.append("  NEXT  : Run Layer 3 Rules Engine (A/C/D) on these candidates")
    lines.append(sep)

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    log.info("  Report saved: %s", REPORT)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    # 0. Load all parquets
    data = load_all()

    universe = data.get("universe", pd.DataFrame())
    if universe.empty:
        log.error("universe_master.parquet missing -- cannot run Layer 2")
        return

    # 1. Pre-read: regime + sector bias
    regime, vix_val, sector_rank = read_regime_and_sector(data)

    # 2. Fetch today's prices/volumes
    prices = fetch_today(universe)
    if prices.empty:
        log.error("Price fetch returned empty -- check internet / yfinance")
        return

    # Merge universe + prices
    df = universe.merge(prices, on="ticker", how="inner")
        
    log.info("")
    log.info("  Universe x prices join: %d stocks", len(df))

    # 3. Gate 1: Liquidity
    df = gate_liquidity(df, regime)
    if df.empty:
        log.error("All stocks filtered at liquidity gate")
        return

    # 4. Gate 2: Event blackout
    df = gate_event_blackout(df, data.get("earnings", pd.DataFrame()))

    # 5. Gate 3: R2 grade
    df = gate_r2(df, data.get("r2_grades", pd.DataFrame()), regime)
    if df.empty:
        log.warning("All stocks filtered at R2 grade gate -- running without grade filter")
        # Fallback: reload after liquidity only
        df = gate_liquidity(universe.merge(prices, on="ticker", how="inner"), regime)
        df["r2_grade"] = "B"

    # 6. Crisis gate
    df = apply_crisis_gate(df, data.get("crisis", pd.DataFrame()), regime)
    if df.empty:
        log.warning("Crisis gate emptied candidates -- crisis regime with no alpha stocks detected")
        return

    # 7. Enrich
    df = enrich(df, data, sector_rank, regime, vix_val)

    # 8. Score + rank
    df = score_and_rank(df)

    # 9. Save + report
    save_and_report(df, regime, vix_val)

    log.info("")
    log.info("  Layer 2 complete. %d candidates ready for Layer 3.", len(df))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
