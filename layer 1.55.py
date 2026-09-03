# -*- coding: utf-8 -*-
# NSE DECISION ENGINE - LAYER 1.5 v3 - Walk-Forward Validation Engine
# Signals (9 total):
#   Original 6 : Hurst, Granger, GARCH, CUSUM, Earnings Drift, Pairs
#   ML 3       : HAR-RV, XGBoost, LightGBM
#   Bonus 2    : Mean-Reversion Z-Score, Momentum Persistence
# NO PyTorch dependency — pure numpy / sklearn / xgboost / lightgbm
#
# Run    : python nse_layer1_5_v3.py
# Install: pip install arch scikit-learn xgboost lightgbm pandas numpy yfinance scipy python-dateutil

import os
import warnings
import logging
import traceback
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import multiprocessing as mp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.stats import spearmanr
from numpy.linalg import lstsq as np_lstsq
from arch import arch_model

# ML imports (no torch)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "layer1_5_v3_run.log"), mode="w"),
    ],
)
log = logging.getLogger("L1.5v3")

def pq(name):
    return os.path.join(PROJECT_DIR, name)

# ─────────────────────────────────────────────────────────────────────────────
# SLIPPAGE
# ─────────────────────────────────────────────────────────────────────────────
SLIPPAGE = {
    "T1": {"swing": 0.0015, "long": 0.0015, "intraday": 0.0008},
    "T2": {"swing": 0.0025, "long": 0.0025, "intraday": 0.0013},
    "T3": {"swing": 0.0050, "long": 0.0050, "intraday": 0.0025},
}

def get_slippage(tier, horizon):
    t = tier if tier in SLIPPAGE else "T3"
    h = horizon if horizon in SLIPPAGE["T1"] else "swing"
    return SLIPPAGE[t][h]

WF_CONFIGS = {
    "intraday": {"train_months": 6,  "test_months": 1},
    "swing":    {"train_months": 12, "test_months": 3},
    "long":     {"train_months": 24, "test_months": 6},
}

FORWARD_RETURN_DAYS = {"intraday": 1, "swing": 10, "long": 60}

# ─────────────────────────────────────────────────────────────────────────────
# FOLD GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_folds(wf_type, start_year=2010, end_year=2026):
    cfg = WF_CONFIGS[wf_type]
    folds, cursor, end = [], datetime(start_year, 1, 1), datetime(end_year, 1, 1)
    while True:
        tr_s = cursor
        tr_e = cursor + relativedelta(months=cfg["train_months"])
        te_s = tr_e
        te_e = te_s + relativedelta(months=cfg["test_months"])
        if te_e > end:
            break
        folds.append((tr_s, tr_e, te_s, te_e))
        cursor += relativedelta(months=cfg["test_months"])
    return folds

# ─────────────────────────────────────────────────────────────────────────────
# PRICE CACHE
# ─────────────────────────────────────────────────────────────────────────────
_PRICE_CACHE = {}

def fetch_prices(tickers, start, end, max_tickers=200):
    tickers = list(tickers)[:max_tickers]
    key = (tuple(sorted(tickers)), str(start.date()), str(end.date()))
    if key in _PRICE_CACHE:
        return _PRICE_CACHE[key]
    ns_tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers]
    try:
        raw = yf.download(
            ns_tickers, start=start - timedelta(days=5),
            end=end + timedelta(days=5),
            auto_adjust=True, progress=False, threads=True,
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices.columns = [c.replace(".NS", "") for c in prices.columns]
        prices = prices.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna(axis=1, how="all")
    except Exception as e:
        log.warning("Price fetch error: %s", e)
        prices = pd.DataFrame()
    _PRICE_CACHE[key] = prices
    return prices

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILS
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic(sig, fwd):
    df = pd.DataFrame({"s": sig, "f": fwd}).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    return spearmanr(df["s"], df["f"])

def simulate_pnl(sig, fwd, tiers, horizon):
    df = pd.DataFrame({"s": sig, "f": fwd, "t": tiers}).dropna(subset=["s", "f"])
    if len(df) < 10:
        return np.nan
    longs  = df[df["s"] >= df["s"].quantile(0.75)]
    shorts = df[df["s"] <= df["s"].quantile(0.25)]
    def net(rows, d):
        if len(rows) == 0:
            return 0.0
        return rows["f"].mean() * d - rows["t"].apply(lambda t: get_slippage(t, horizon)).mean()
    return net(longs, 1) + net(shorts, -1)

def make_row(signal, i, tr_s, te_s, ic, pval, pnl, n):
    return {
        "signal": signal, "fold": i,
        "train_start": tr_s, "test_start": te_s,
        "ic": ic, "pval": pval, "pnl": pnl, "n_stocks": n,
    }

def log_fold(name, i, total, ic, pnl, n):
    log.info("    %s fold %d/%d | IC=%.3f | PnL=%.4f | n=%d",
             name, i + 1, total, ic or 0, pnl or 0, n)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER  (shared by XGBoost / LightGBM)
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(ret, period):
    gain = ret.clip(lower=0).rolling(period).mean()
    loss = (-ret.clip(upper=0)).rolling(period).mean()
    rs   = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def build_features(px, lookbacks=(5, 10, 20, 60)):
    ret   = px.pct_change()
    feats = {}
    for lb in lookbacks:
        if len(px) < lb + 1:
            continue
        feats[f"mom_{lb}"]    = ret.rolling(lb).mean()
        feats[f"vol_{lb}"]    = ret.rolling(lb).std()
        feats[f"rsi_{lb}"]    = _rsi(ret, lb)
        feats[f"zscore_{lb}"] = (px - px.rolling(lb).mean()) / (px.rolling(lb).std() + 1e-9)
    feats["skew_20"] = ret.rolling(20).skew()
    feats["kurt_20"] = ret.rolling(20).kurt()
    # Extra: autocorrelation lag-1 and lag-5
    feats["ac1"]  = ret.rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) > 2 else np.nan, raw=False)
    feats["ac5"]  = ret.rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) > 6 else np.nan, raw=False)
    return pd.DataFrame(feats).dropna()

def build_ml_dataset(prices, tr_s, tr_e, te_s, te_e, fd, tier_map, min_rows=60):
    X_tr, y_tr = [], []
    X_te, te_tickers, te_fwd, te_tiers = [], [], [], []

    for col in prices.columns:
        tr_px = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), col].dropna()
        te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
        if len(tr_px) < min_rows or len(te_px) < fd + 1:
            continue
        tr_feat = build_features(tr_px)
        if len(tr_feat) < 20:
            continue
        tr_ret  = tr_px.pct_change(fd).shift(-fd).reindex(tr_feat.index).dropna()
        aligned = tr_feat.loc[tr_ret.index]
        if len(aligned) < 10:
            continue
        X_tr.append(aligned.values)
        y_tr.append(tr_ret.values)

        te_feat = build_features(te_px)
        if te_feat.empty:
            continue
        X_te.append(te_feat.iloc[-1].values)
        fwd = te_px.iloc[min(fd, len(te_px) - 1)] / te_px.iloc[0] - 1
        te_tickers.append(col)
        te_fwd.append(fwd)
        te_tiers.append(tier_map.get(col, "T3"))

    if not X_tr or not X_te:
        return None

    X_train = np.vstack(X_tr)
    y_train = np.concatenate(y_tr)
    X_test  = np.array(X_te)
    min_feat = min(X_train.shape[1], X_test.shape[1])
    return X_train[:, :min_feat], y_train, X_test[:, :min_feat], te_tickers, te_fwd, te_tiers


# =============================================================================
# ORIGINAL 6 SIGNALS
# =============================================================================

# ── [1] HURST ─────────────────────────────────────────────────────────────────
def hurst_exponent(ts):
    ts = np.array(ts)
    if len(ts) < 20:
        return np.nan
    lags = range(2, min(20, len(ts) // 2))
    rs_vals = []
    for lag in lags:
        chunks = [ts[i:i + lag] for i in range(0, len(ts) - lag, lag)]
        rs_c = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            dev = np.cumsum(chunk - np.mean(chunk))
            s   = np.std(chunk, ddof=1)
            if s > 0:
                rs_c.append(np.ptp(dev) / s)
        if rs_c:
            rs_vals.append(np.mean(rs_c))
    if len(rs_vals) < 2:
        return np.nan
    try:
        h, _ = np.polyfit(np.log(list(lags)[: len(rs_vals)]), np.log(rs_vals), 1)
        return h
    except Exception:
        return np.nan

def validate_hurst(universe, tier_map):
    log.info("  [1/11] HURST")
    folds   = generate_folds("swing")
    tickers = universe["ticker"].tolist()
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, tr_s, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["swing"]
        for col in prices.columns:
            tr_px = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), col].dropna()
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(tr_px) < 60 or len(te_px) < fd + 1:
                continue
            h = hurst_exponent(tr_px.values)
            if np.isnan(h):
                continue
            signals[col]  = h - 0.5
            fwd_rets[col] = te_px.iloc[-1] / te_px.iloc[0] - 1
            tiers[col]    = tier_map.get(col, "T3")
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "swing")
        results.append(make_row("hurst", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("hurst", i, len(folds), ic, pnl, len(signals))
    log.info("    hurst done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [2] GRANGER ───────────────────────────────────────────────────────────────
def validate_granger(universe, stable_edges, tier_map):
    log.info("  [2/11] GRANGER")
    folds = generate_folds("swing")
    cc = "Source"
    fc = "F_stat"
    threshold = stable_edges[fc].quantile(0.75)
    strong    = stable_edges[stable_edges[fc] >= threshold]
    conv_map  = dict(strong.groupby(cc)[fc].mean())
    tickers   = list(conv_map.keys())
    results   = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, te_s, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["swing"]
        for col in prices.columns:
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(te_px) < fd + 1 or col not in conv_map:
                continue
            signals[col]  = conv_map[col]
            fwd_rets[col] = te_px.iloc[-1] / te_px.iloc[0] - 1
            tiers[col]    = tier_map.get(col, "T3")
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "swing")
        results.append(make_row("granger", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("granger", i, len(folds), ic, pnl, len(signals))
    log.info("    granger done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [3] GARCH ─────────────────────────────────────────────────────────────────
def validate_garch(universe, garch_params, tier_map):
    log.info("  [3/11] GARCH")
    folds   = generate_folds("intraday")
    tc      = "ticker" if "ticker" in garch_params.columns else garch_params.columns[0]
    tickers = garch_params[tc].tolist()[:200]
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, tr_s, te_e, max_tickers=150)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["intraday"]
        for col in prices.columns:
            tr_px = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), col].dropna()
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(tr_px) < 60 or len(te_px) < fd + 1:
                continue
            try:
                ret = tr_px.pct_change().dropna() * 100
                res = arch_model(ret, vol="Garch", p=1, q=1, rescale=False).fit(
                    disp="off", show_warning=False
                )
                a = res.params.get("alpha[1]", np.nan)
                b = res.params.get("beta[1]", np.nan)
                p = a + b
                if np.isnan(p) or p >= 1.0:
                    continue
                signals[col]  = p
                fwd_rets[col] = te_px.iloc[fd] / te_px.iloc[0] - 1
                tiers[col]    = tier_map.get(col, "T3")
            except Exception:
                continue
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "intraday")
        results.append(make_row("garch", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 24 == 0:
            log_fold("garch", i, len(folds), ic, pnl, len(signals))
    log.info("    garch done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [4] CUSUM ─────────────────────────────────────────────────────────────────
def validate_cusum(universe, cusum_breaks, tier_map):
    log.info("  [4/11] CUSUM")
    folds = generate_folds("intraday")
    tc    = "ticker" if "ticker" in cusum_breaks.columns else cusum_breaks.columns[0]
    dc    = next(
        (c for c in cusum_breaks.columns if "date" in c.lower() and c != tc),
        cusum_breaks.columns[1],
    )
    cusum_breaks[dc] = pd.to_datetime(cusum_breaks[dc], errors="coerce")
    tickers = cusum_breaks[tc].unique().tolist()
    if not tickers:
        log.warning("    CUSUM: no tickers - skipping")
        return pd.DataFrame()
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        mask = (cusum_breaks[dc] >= tr_s) & (cusum_breaks[dc] <= tr_e)
        bc   = cusum_breaks[mask].groupby(tc).size()
        if bc.empty:
            continue
        prices = fetch_prices(bc.index.tolist(), te_s, te_e, max_tickers=150)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["intraday"]
        for col in prices.columns:
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(te_px) < fd + 1 or col not in bc.index:
                continue
            signals[col]  = bc[col]
            fwd_rets[col] = te_px.iloc[fd] / te_px.iloc[0] - 1
            tiers[col]    = tier_map.get(col, "T3")
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "intraday")
        results.append(make_row("cusum", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 24 == 0:
            log_fold("cusum", i, len(folds), ic, pnl, len(signals))
    log.info("    cusum done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [5] EARNINGS DRIFT ────────────────────────────────────────────────────────
def validate_earnings_drift(universe, earnings_dna, tier_map):
    log.info("  [5/11] EARNINGS DRIFT")
    folds = generate_folds("long")
    tc    = "ticker" if "ticker" in earnings_dna.columns else earnings_dna.columns[0]
    dc    = next(
        (c for c in earnings_dna.columns
         if any(w in c.lower() for w in ["drift", "return", "alpha", "excess"]) and c != tc),
        earnings_dna.columns[1] if len(earnings_dna.columns) > 1 else None,
    )
    tickers   = earnings_dna[tc].tolist()
    drift_map = dict(zip(earnings_dna[tc], earnings_dna[dc])) if dc else {}
    results   = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, te_s, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["long"]
        for col in prices.columns:
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(te_px) < fd + 1 or col not in drift_map:
                continue
            sv = drift_map[col]
            if pd.isna(sv):
                continue
            signals[col]  = sv
            fwd_rets[col] = te_px.iloc[fd] / te_px.iloc[0] - 1
            tiers[col]    = tier_map.get(col, "T2")
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "long")
        results.append(make_row("earnings_drift", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 5 == 0:
            log_fold("earnings", i, len(folds), ic, pnl, len(signals))
    log.info("    earnings done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [6] PAIRS ─────────────────────────────────────────────────────────────────
def validate_pairs(universe, spread_history, tier_map):
    log.info("  [6/11] PAIRS")
    folds = generate_folds("long")
    t1c = "Ticker_A"
    t2c = "Ticker_B"
    pairs = list(zip(spread_history[t1c], spread_history[t2c]))[:100]
    all_tickers = list(set(t for p in pairs for t in p))
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(all_tickers, tr_s, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["long"]
        for t1, t2 in pairs:
            if t1 not in prices.columns or t2 not in prices.columns:
                continue
            tr1 = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), t1].dropna()
            tr2 = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), t2].dropna()
            te1 = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t1].dropna()
            te2 = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), t2].dropna()
            ctr = tr1.index.intersection(tr2.index)
            cte = te1.index.intersection(te2.index)
            if len(ctr) < 120 or len(cte) < fd + 1:
                continue
            try:
                beta, _ = np.polyfit(tr2.loc[ctr], tr1.loc[ctr], 1)
            except Exception:
                continue
            spread     = te1.loc[cte] - beta * te2.loc[cte]
            z          = (spread.iloc[0] - spread.mean()) / (spread.std() + 1e-9)
            fwd_spread = spread.iloc[min(fd, len(spread) - 1)] - spread.iloc[0]
            key = t1 + "_" + t2
            signals[key]  = -z
            fwd_rets[key] = fwd_spread
            tiers[key]    = tier_map.get(t1, "T2")
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "long")
        results.append(make_row("pairs", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 5 == 0:
            log_fold("pairs", i, len(folds), ic, pnl, len(signals))
    log.info("    pairs done - %d folds", len(results))
    return pd.DataFrame(results)


# =============================================================================
# ML SIGNALS (no PyTorch)
# =============================================================================

# ── [7] HAR-RV ────────────────────────────────────────────────────────────────
# Heterogeneous AutoRegressive Realized Volatility
# Predicts next-period vol via daily/weekly/monthly RV components.
# Signal: low predicted vol → long (low-vol anomaly is strong on NSE).
def validate_har_rv(universe, tier_map):
    log.info("  [7/11] HAR-RV")
    folds   = generate_folds("swing")
    tickers = universe["ticker"].tolist()
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, tr_s, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["swing"]
        for col in prices.columns:
            tr_px = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), col].dropna()
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(tr_px) < 60 or len(te_px) < fd + 1:
                continue
            try:
                ret  = tr_px.pct_change().dropna()
                rv_d = ret ** 2
                rv_w = rv_d.rolling(5).mean()
                rv_m = rv_d.rolling(22).mean()
                df_h = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}).dropna()
                if len(df_h) < 30:
                    continue
                X = np.c_[np.ones(len(df_h) - 1), df_h[["rv_d", "rv_w", "rv_m"]].values[:-1]]
                y = df_h["rv_d"].values[1:]
                coef, *_ = np_lstsq(X, y, rcond=None)
                last     = df_h.iloc[-1]
                pred_rv  = coef[0] + coef[1] * last["rv_d"] + coef[2] * last["rv_w"] + coef[3] * last["rv_m"]
                signals[col]  = -pred_rv          # low vol → positive signal
                fwd_rets[col] = te_px.iloc[-1] / te_px.iloc[0] - 1
                tiers[col]    = tier_map.get(col, "T3")
            except Exception:
                continue
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "swing")
        results.append(make_row("har_rv", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("har_rv", i, len(folds), ic, pnl, len(signals))
    log.info("    har_rv done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [8] XGBOOST ───────────────────────────────────────────────────────────────
# Trains per-fold XGBoost on momentum / vol / RSI / z-score features.
# Learns non-linear combinations that tree models handle natively.
def validate_xgboost(universe, tier_map):
    log.info("  [8/11] XGBOOST")
    folds   = generate_folds("swing")
    tickers = universe["ticker"].tolist()[:300]
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, tr_s, te_e)
        if prices.empty:
            continue
        fd  = FORWARD_RETURN_DAYS["swing"]
        out = build_ml_dataset(prices, tr_s, tr_e, te_s, te_e, fd, tier_map)
        if out is None:
            continue
        X_train, y_train, X_test, te_tickers, te_fwd, te_tiers = out
        if len(X_train) < 30 or len(X_test) < 10:
            continue
        try:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds    = model.predict(X_test)
            signals  = dict(zip(te_tickers, preds))
            fwd_rets = dict(zip(te_tickers, te_fwd))
            tiers_d  = dict(zip(te_tickers, te_tiers))
        except Exception:
            continue
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers_d), "swing")
        results.append(make_row("xgboost", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("xgboost", i, len(folds), ic, pnl, len(signals))
    log.info("    xgboost done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [9] LIGHTGBM ──────────────────────────────────────────────────────────────
# Leaf-wise gradient boosting — faster than XGBoost on wide feature sets,
# often edges it out on financial tabular data.
def validate_lightgbm(universe, tier_map):
    log.info("  [9/11] LIGHTGBM")
    folds   = generate_folds("swing")
    tickers = universe["ticker"].tolist()[:300]
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, tr_s, te_e)
        if prices.empty:
            continue
        fd  = FORWARD_RETURN_DAYS["swing"]
        out = build_ml_dataset(prices, tr_s, tr_e, te_s, te_e, fd, tier_map)
        if out is None:
            continue
        X_train, y_train, X_test, te_tickers, te_fwd, te_tiers = out
        if len(X_train) < 30 or len(X_test) < 10:
            continue
        try:
            model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(X_train, y_train)
            preds    = model.predict(X_test)
            signals  = dict(zip(te_tickers, preds))
            fwd_rets = dict(zip(te_tickers, te_fwd))
            tiers_d  = dict(zip(te_tickers, te_tiers))
        except Exception:
            continue
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers_d), "swing")
        results.append(make_row("lightgbm", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("lightgbm", i, len(folds), ic, pnl, len(signals))
    log.info("    lightgbm done - %d folds", len(results))
    return pd.DataFrame(results)


# =============================================================================
# BONUS SIGNALS
# =============================================================================

# ── [10] MEAN-REVERSION Z-SCORE ───────────────────────────────────────────────
# Classic cross-sectional mean reversion: stocks with the most negative
# 20-day z-score relative to their own history tend to outperform short-term.
# Works well on NSE mid/small caps which overshoot on momentum.
# Signal: large negative z → expect bounce → positive forward return.
def validate_zscore_mr(universe, tier_map):
    log.info("  [10/11] ZSCORE MEAN-REVERSION")
    folds   = generate_folds("swing")
    tickers = universe["ticker"].tolist()
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        prices = fetch_prices(tickers, tr_s, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["swing"]
        for col in prices.columns:
            tr_px = prices.loc[(prices.index >= tr_s) & (prices.index <= tr_e), col].dropna()
            te_px = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(tr_px) < 60 or len(te_px) < fd + 1:
                continue
            try:
                # Fit rolling z-score parameters on train, apply at test start
                mu  = tr_px.rolling(20).mean().iloc[-1]
                sig = tr_px.rolling(20).std().iloc[-1]
                if sig < 1e-9:
                    continue
                z = (te_px.iloc[0] - mu) / sig
                signals[col]  = -z                # mean-revert: short high-z, long low-z
                fwd_rets[col] = te_px.iloc[min(fd, len(te_px) - 1)] / te_px.iloc[0] - 1
                tiers[col]    = tier_map.get(col, "T3")
            except Exception:
                continue
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "swing")
        results.append(make_row("zscore_mr", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("zscore_mr", i, len(folds), ic, pnl, len(signals))
    log.info("    zscore_mr done - %d folds", len(results))
    return pd.DataFrame(results)

# ── [11] MOMENTUM PERSISTENCE (12-1 minus 1-month reversal) ──────────────────
# Classic Jegadeesh-Titman style: 12-month momentum MINUS 1-month reversal.
# Skips the most recent month to avoid short-term reversal contamination.
# Well-documented on NSE; strongest in T1/T2 large-cap names.
# Signal: high 12-1 momentum score → long; low → short.
def validate_momentum_persistence(universe, tier_map):
    log.info("  [11/11] MOMENTUM PERSISTENCE (12-1)")
    folds   = generate_folds("swing")
    tickers = universe["ticker"].tolist()
    results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        # Need 13 months of history before test — extend lookback
        extended_start = tr_s - relativedelta(months=2)
        prices = fetch_prices(tickers, extended_start, te_e)
        if prices.empty:
            continue
        signals, fwd_rets, tiers = {}, {}, {}
        fd = FORWARD_RETURN_DAYS["swing"]
        for col in prices.columns:
            full_px = prices.loc[(prices.index >= extended_start) & (prices.index <= tr_e), col].dropna()
            te_px   = prices.loc[(prices.index >= te_s) & (prices.index <= te_e), col].dropna()
            if len(full_px) < 200 or len(te_px) < fd + 1:
                continue
            try:
                p_now  = full_px.iloc[-1]                    # end of train period
                p_1m   = full_px.iloc[-22] if len(full_px) >= 22 else full_px.iloc[0]   # 1 month ago
                p_12m  = full_px.iloc[-252] if len(full_px) >= 252 else full_px.iloc[0] # 12 months ago
                mom_12 = p_1m / p_12m - 1    # 12-to-1 month momentum (skip last month)
                rev_1  = p_now / p_1m - 1    # 1-month reversal term
                signal = mom_12 - rev_1       # combined: strong trend, recent pullback = buy
                signals[col]  = signal
                fwd_rets[col] = te_px.iloc[min(fd, len(te_px) - 1)] / te_px.iloc[0] - 1
                tiers[col]    = tier_map.get(col, "T3")
            except Exception:
                continue
        ic, pval = compute_ic(pd.Series(signals), pd.Series(fwd_rets))
        pnl = simulate_pnl(pd.Series(signals), pd.Series(fwd_rets), pd.Series(tiers), "swing")
        results.append(make_row("momentum_12_1", i, tr_s, te_s, ic, pval, pnl, len(signals)))
        if (i + 1) % 12 == 0:
            log_fold("momentum_12_1", i, len(folds), ic, pnl, len(signals))
    log.info("    momentum_12_1 done - %d folds", len(results))
    return pd.DataFrame(results)


# =============================================================================
# REPORT
# =============================================================================
def build_signal_report(all_df):
    rows = []
    for signal, grp in all_df.groupby("signal"):
        ic_v    = grp["ic"].dropna()
        pv      = grp["pnl"].dropna()
        mean_ic = ic_v.mean()
        ic_pos  = (ic_v > 0).mean()
        ic_t    = stats.ttest_1samp(ic_v, 0).statistic if len(ic_v) >= 5 else np.nan
        tot_pnl = pv.sum()
        pnl_pos = (pv > 0).mean()
        sharpe  = (pv.mean() / (pv.std() + 1e-9)) * np.sqrt(len(pv))
        ic_pass  = (not np.isnan(ic_t)) and abs(ic_t) > 1.5 and ic_pos > 0.55
        pnl_pass = pnl_pos > 0.55
        verdict  = "PASS" if (ic_pass and pnl_pass) else ("WEAK" if (ic_pass or pnl_pass) else "FAIL")
        rows.append({
            "signal":      signal,
            "n_folds":     len(grp),
            "mean_ic":     round(mean_ic, 4),
            "ic_tstat":    round(float(ic_t), 3) if not np.isnan(ic_t) else None,
            "ic_pct_pos":  round(ic_pos, 3),
            "total_pnl":   round(tot_pnl, 4),
            "pnl_pct_pos": round(pnl_pos, 3),
            "sharpe":      round(sharpe, 3),
            "verdict":     verdict,
        })
    return pd.DataFrame(rows)

def write_text_report(report_df, out_path):
    sep = "=" * 74
    lines = [
        sep,
        "  NSE DECISION ENGINE - LAYER 1.5 v3 WALK-FORWARD VALIDATION REPORT",
        "  Generated : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "  Signals   : 11 (Original 6 + HAR-RV + XGBoost + LightGBM + ZScore MR + Mom 12-1)",
        "  No PyTorch dependency",
        sep, "",
        "{:<22} {:>5} {:>8} {:>7} {:>6} {:>9} {:>6} {:>7}  {}".format(
            "Signal", "Folds", "MeanIC", "IC-t", "IC+%", "TotPnL", "PnL+%", "Sharpe", "Verdict"
        ),
        "-" * 74,
    ]
    for _, r in report_df.iterrows():
        lines.append(
            "{:<22} {:>5} {:>8.4f} {:>7} {:>6.1%} {:>9.4f} {:>6.1%} {:>7.3f}  {}".format(
                r["signal"], r["n_folds"], r["mean_ic"], str(r["ic_tstat"]),
                r["ic_pct_pos"], r["total_pnl"], r["pnl_pct_pos"], r["sharpe"], r["verdict"],
            )
        )
    passing = report_df[report_df["verdict"] == "PASS"]["signal"].tolist()
    weak    = report_df[report_df["verdict"] == "WEAK"]["signal"].tolist()
    failing = report_df[report_df["verdict"] == "FAIL"]["signal"].tolist()
    lines += [
        "",
        "PASS CRITERIA: IC t-stat > 1.5  AND  IC+ > 55%  AND  PnL+ > 55%", "",
        "LAYER 2 RECOMMENDATION:", "-" * 74,
        "  INCLUDE    : " + (", ".join(passing) if passing else "None"),
        "  USE CAUTION: " + (", ".join(weak)    if weak    else "None"),
        "  EXCLUDE    : " + (", ".join(failing) if failing else "None"),
        "", sep,
    ]
    txt = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("  NSE DECISION ENGINE - LAYER 1.5 v3 STARTING")
    log.info("  11 signals | No PyTorch")
    log.info("=" * 60)

    required = {
        "universe_master": "universe_master.parquet",
        "stable_edges":    "stable_edges.parquet",
        "garch_params":    "garch_params.parquet",
        "cusum_breaks":    "cusum_breaks.parquet",
        "earnings_dna":    "earnings_dna.parquet",
        "spread_history":  "spread_history.parquet",
    }
    data = {}
    for key, fname in required.items():
        path = pq(fname)
        if not os.path.exists(path):
            log.error("MISSING: %s - cannot continue", fname)
            return
        data[key] = pd.read_parquet(path)
        log.info("  %-35s %7d rows", fname, len(data[key]))

    universe       = data["universe_master"]
    stable_edges   = data["stable_edges"]
    garch_params   = data["garch_params"]
    cusum_breaks   = data["cusum_breaks"]
    earnings_dna   = data["earnings_dna"]
    spread_history = data["spread_history"]

    # Flexible column detection
    ticker_col = next(
        (c for c in universe.columns if c.lower() in ("ticker", "symbol", "stock")),
        universe.columns[0],
    )
    tier_col = next((c for c in universe.columns if c.lower() == "tier"), None)
    tier_map = dict(zip(universe[ticker_col], universe[tier_col])) if tier_col else {}
    if ticker_col != "ticker":
        universe = universe.rename(columns={ticker_col: "ticker"})
        log.info("  Renamed universe column %s -> ticker", ticker_col)

    signals_to_run = [
        # ── Original 6 ──────────────────────────────────────────────────────
        ("hurst",            lambda: validate_hurst(universe, tier_map)),
        ("granger",          lambda: validate_granger(universe, stable_edges, tier_map)),
        ("garch",            lambda: validate_garch(universe, garch_params, tier_map)),
        ("cusum",            lambda: validate_cusum(universe, cusum_breaks, tier_map)),
        ("earnings_drift",   lambda: validate_earnings_drift(universe, earnings_dna, tier_map)),
        ("pairs",            lambda: validate_pairs(universe, spread_history, tier_map)),
        # ── ML (no PyTorch) ─────────────────────────────────────────────────
        ("har_rv",           lambda: validate_har_rv(universe, tier_map)),
        ("xgboost",          lambda: validate_xgboost(universe, tier_map)),
        ("lightgbm",         lambda: validate_lightgbm(universe, tier_map)),
        # ── Bonus ───────────────────────────────────────────────────────────
        ("zscore_mr",        lambda: validate_zscore_mr(universe, tier_map)),
        ("momentum_12_1",    lambda: validate_momentum_persistence(universe, tier_map)),
    ]

    all_results = []
    for name, fn in signals_to_run:
        log.info("")
        log.info("-" * 55)
        try:
            df = fn()
            if df is not None and not df.empty:
                all_results.append(df)
                # Checkpoint after each signal
                pd.concat(all_results, ignore_index=True).to_parquet(
                    pq("layer1_5_v3_checkpoint.parquet"), index=False
                )
        except Exception as e:
            log.error("%s crashed: %s", name, e)
            traceback.print_exc()

    log.info("")
    log.info("-" * 55)
    if not all_results:
        log.error("No results produced. Exiting.")
        return

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_parquet(pq("layer1_5_v3_ic_results.parquet"), index=False)
    log.info("  saved: layer1_5_v3_ic_results.parquet")

    pnl_df = all_df[["signal", "fold", "test_start", "pnl"]].copy()
    pnl_df["cum_pnl"] = pnl_df.groupby("signal")["pnl"].cumsum()
    pnl_df.to_parquet(pq("layer1_5_v3_pnl_curves.parquet"), index=False)
    log.info("  saved: layer1_5_v3_pnl_curves.parquet")

    report_df = build_signal_report(all_df)
    report_df.to_parquet(pq("layer1_5_v3_signal_report.parquet"), index=False)
    log.info("  saved: layer1_5_v3_signal_report.parquet")

    write_text_report(report_df, pq("layer1_5_v3_validation_report.txt"))
    log.info("  saved: layer1_5_v3_validation_report.txt")

    log.info("")
    log.info("=" * 60)
    log.info("  LAYER 1.5 v3 COMPLETE")
    log.info("  Next: Layer 2 - Daily Signal Engine")
    log.info("=" * 60)


if __name__ == "__main__":
    mp.freeze_support()
    main()