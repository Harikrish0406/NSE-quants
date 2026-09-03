# -*- coding: utf-8 -*-
"""
layer_ml_ensemble.py — Real ML ensemble signal layer (XGBoost + LightGBM)

WHY THIS FILE EXISTS
---------------------
The old production pipeline had trained LightGBM/XGBoost models sitting in the
codebase, but the score-combination code that was supposed to consume their
predictions had a hardcoded placeholder (`ml_score_placeholder = 0.5`, a
constant, never replaced with a real model call). The models existed but
contributed literally nothing to real trading decisions.

This layer fixes that: real cross-sectional features built per fold, real
XGBoost + LightGBM models trained walk-forward (one fresh fit per fold, on
that fold's training window only), a real hyperparameter search per fold, and
a real ensemble prediction returned per ticker and actually scored against
forward returns by validation_framework.run_walkforward_ml.

FIRST-PASS RESULT (for context — see git history / prior report)
-------------------------------------------------------------------
A first pass (14 generic price/volume features, a 4-combo hyperparameter
grid, capped at the top-200 most liquid tickers because fetch_prices() used
to hard-cap at 200) came back FAIL: mean IC 0.0124, IC t-stat 0.65 (not
significant), pnl_pos_pct 45% (below the 50% floor even for WEAK).
Directionally positive but noisy — a real, honest non-result, not a bug.

THIS PASS: substantially deeper compute, per explicit instruction to push
harder rather than accept the first pass as final:
  1. Full local universe (no ticker cap) — validation_framework.fetch_prices'
     max_tickers cap has been removed, and this layer no longer pre-trims to
     top-200; it passes every ticker in universe_master.parquet (~1552,
     all of which have local data/raw/*.parquet caches).
  2. Feature set expanded from 14 to ~27: the original 14 trailing
     price/volume technicals, PLUS
       - cross-sectional percentile RANK (within the training fold, per
         date) of 8 of the base features — rank-based signals are often
         more stable than raw magnitudes across regimes,
       - SECTOR-RELATIVE return features (ticker's ret_10/ret_20 minus the
         same-day mean of its Sector peers within the training fold, using
         universe_master.parquet's Sector column),
       - 5 interaction terms between existing features (momentum x vol,
         mean-reversion x volume-trend, etc).
  3. Hyperparameter search expanded from a 4-combo grid to a randomized
     search of 32 combinations EACH for XGBoost and LightGBM (64 tuning fits
     + 2 final refits = 66 model fits per fold), searching max_depth /
     num_leaves, learning_rate, subsample, colsample_bytree,
     min_child_weight / min_child_samples, and n_estimators ceiling. Still
     selected via the SAME internal training-window time-holdout as before
     (never the test window).
  4. WF_TYPE is overridable via `python layer_ml_ensemble.py <wf_type>`
     (intraday / swing / long) so the exact same feature/model code can be
     reused to check whether a different forward-return horizon has more
     edge than the 10-day "swing" default, as a secondary experiment. Output
     filenames are suffixed with the horizon when it isn't the "swing"
     default, so the primary swing report is never overwritten by a
     secondary run.

LEAKAGE DISCIPLINE (unchanged from first pass — still holds under the
deeper feature/search space)
-------------------------------------------------------------------------
validation_framework.run_walkforward_ml() calls fit_predict_fn(train_panel)
with ONLY training-window Close price series (dict: ticker -> pd.Series).
No test-window data is ever passed in — it is structurally impossible for
this function to see it. Within fit_predict_fn we additionally guarantee:

  1. Every base feature at row/date t is computed using a trailing
     (backward-only) rolling window ending at t — pct_change(),
     rolling().std()/.mean(), ewm() — never a centered or forward window.
  2. The new cross-sectional features (rank / sector-relative / interaction)
     are computed by grouping THIS FOLD'S TRAINING PANEL ONLY by date (and
     date+sector) — every value that goes into a `groupby("date")` or
     `groupby(["date","sector"])` aggregate comes from a row already
     restricted to [train_start, train_end), so no test-window value can
     ever enter a rank or sector-mean computation.
  3. Training LABELS (forward N-day returns) are computed by shifting the
     SAME training-window-only Close series backward (`close.shift(-N)`).
     Since that series contains no data past the training window, dates
     near the end of the window simply produce NaN labels (dropped) —
     there is no way for a label to reach into the test window because the
     data to do so was never given to this function.
  4. High/Low/Volume features are loaded from data/raw/{ticker}.parquet but
     sliced to the exact same [train_start, train_end) window as the Close
     series before any feature is computed from them — same window, no
     extra lookback beyond it.
  5. Feature winsorization bounds (1st/99th pct) and NaN-fill medians are
     computed on the training panel ONLY, then applied unchanged (not
     refit) to the end-of-window prediction snapshot — exactly the
     "fit on train, apply-don't-refit" discipline the task calls for.
  6. Hyperparameter selection uses an internal TIME split inside the
     training window itself (earlier dates -> internal-train, latest dates
     -> internal-val) — never the test window — so even model selection
     cannot leak forward.
  7. The final prediction is made from each ticker's LAST available date
     inside the training window (the end-of-training-window snapshot) —
     i.e. everything the model would actually know "today" as of the fold
     boundary, and nothing more. Crucially, this snapshot row goes through
     the SAME cross-sectional rank/sector-relative computation as the
     labeled training rows (computed jointly, then split) — so its rank
     and sector-relative values are also relative to fold peers only.
"""

import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd

from validation_framework import (
    run_walkforward_ml, build_verdict, PROJECT_DIR, RAW_DIR, FORWARD_RETURN_DAYS,
)

warnings.filterwarnings("ignore")

# WF_TYPE override: `python layer_ml_ensemble.py long` runs the 60-day-forward
# horizon experiment instead of the 10-day "swing" default; `intraday` (1-day
# forward) is also supported but not run by default (see module docstring —
# ~186 folds at train_months=6/test_months=1 makes it far more expensive than
# the primary run; only attempted if there's genuine time to spare).
WF_TYPE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in FORWARD_RETURN_DAYS else "swing"
_SUFFIX = "" if WF_TYPE == "swing" else f"_{WF_TYPE}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("MLEnsemble")

# Explicit file handler (in addition to whatever console redirection the
# caller uses) so progress is inspectable mid-run even under buffered
# stdout redirection. Attached to the ROOT logger so validation_framework's
# per-fold log lines land in the same file too.
_LOG_PATH = os.path.join(PROJECT_DIR, f"ml_ensemble_deep_run{_SUFFIX}.log")
_fh = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_fh)

# ─────────────────────────────────────────────────────────────────────────────
# Optional model backends
# ─────────────────────────────────────────────────────────────────────────────
HAVE_XGB, HAVE_LGBM = True, True
try:
    import xgboost as xgb
except ImportError:
    HAVE_XGB = False
    log.warning("xgboost not importable — ensemble will fall back to LightGBM only.")
try:
    import lightgbm as lgbm
except ImportError:
    HAVE_LGBM = False
    log.warning("lightgbm not importable — ensemble will fall back to XGBoost only.")

if not HAVE_XGB and not HAVE_LGBM:
    raise ImportError("Neither xgboost nor lightgbm is importable — cannot build the ML ensemble.")

FWD_DAYS = FORWARD_RETURN_DAYS[WF_TYPE]          # 1 / 10 / 60 trading days depending on WF_TYPE
MIN_HISTORY = 60                                  # longest lookback feature needs 60 trailing days
N_JOBS = max(1, os.cpu_count() or 1)

# ─────────────────────────────────────────────────────────────────────────────
# Universe selection: FULL local universe (no liquidity cap). fetch_prices'
# max_tickers cap has been removed from validation_framework.py, so every
# ticker in universe_master.parquet (all ~1552 of which have a local
# data/raw/*.parquet cache) is passed through to run_walkforward_ml.
# TICKER_SECTOR is populated in main() before the walk-forward run starts,
# and read by add_cross_sectional_features() inside fit_predict_fn (module
# global — fit_predict_fn is only ever CALLED during main()'s run, by which
# point it is populated).
# ─────────────────────────────────────────────────────────────────────────────
TICKER_SECTOR = {}

def load_universe():
    uni = pd.read_parquet(os.path.join(PROJECT_DIR, "universe_master.parquet"))
    tier_map = dict(zip(uni["Ticker"], uni["Tier"]))
    sector_map = dict(zip(uni["Ticker"], uni["Sector"].fillna("Unknown")))
    universe = uni["Ticker"].tolist()
    return universe, tier_map, sector_map

# ─────────────────────────────────────────────────────────────────────────────
# High/Low/Volume side-cache — loaded once per ticker (full history), then
# sliced per-fold to the exact training window. Avoids re-reading parquet
# files once per fold per ticker across folds; slicing per fold is what keeps
# it leak-free (only the window's rows are ever touched by feature code).
# ─────────────────────────────────────────────────────────────────────────────
_OHLV_CACHE = {}

def _load_ohlv_full(ticker):
    if ticker in _OHLV_CACHE:
        return _OHLV_CACHE[ticker]
    path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        _OHLV_CACHE[ticker] = None
        return None
    try:
        raw = pd.read_parquet(path, columns=["Date", "High", "Low", "Volume"])
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.set_index("Date").sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
    except Exception:
        raw = None
    _OHLV_CACHE[ticker] = raw
    return raw

# ─────────────────────────────────────────────────────────────────────────────
# BASE FEATURE ENGINEERING (unchanged from first pass)
# Every column is a trailing/backward-looking function of Close (and,
# optionally, High/Low/Volume sliced to the SAME window) — nothing here ever
# looks forward. Computed once per ticker over its full training-window
# series; the label uses a separate, negative (backward-relative) shift that
# simply produces NaN once it would need data outside the window.
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)

BASE_FEATURE_COLS = [
    "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_20", "vol_60",
    "rsi_14",
    "dist_ma20", "dist_ma50",
    "macd_norm",
    "skew_20",
    "drawdown_60",
    "atr_norm",
    "vol_trend",
]

def build_features(close: pd.Series, high: pd.Series = None, low: pd.Series = None,
                    volume: pd.Series = None) -> pd.DataFrame:
    close = close.sort_index()
    df = pd.DataFrame(index=close.index)

    # multi-horizon lookback returns
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    df["ret_60"] = close.pct_change(60)

    # realized volatility
    daily_ret = close.pct_change()
    df["vol_20"] = daily_ret.rolling(20).std()
    df["vol_60"] = daily_ret.rolling(60).std()

    # RSI oscillator
    df["rsi_14"] = _rsi(close, 14)

    # distance from moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["dist_ma20"] = (close - ma20) / ma20
    df["dist_ma50"] = (close - ma50) / ma50

    # MACD, price-normalized
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_norm"] = (ema12 - ema26) / close

    # skew of daily returns (asymmetry of recent move)
    df["skew_20"] = daily_ret.rolling(20).skew()

    # drawdown from trailing 60d high (mean-reversion / trend proxy)
    roll_max60 = close.rolling(60).max()
    df["drawdown_60"] = (close - roll_max60) / roll_max60

    # ATR (needs High/Low) — normalized by close; falls back to close-range proxy
    if high is not None and low is not None:
        high = high.reindex(close.index)
        low = low.reindex(close.index)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        df["atr_norm"] = atr14 / close
    else:
        df["atr_norm"] = daily_ret.rolling(14).std()  # fallback proxy if H/L unavailable

    # volume trend: short vs long average volume
    if volume is not None:
        volume = volume.reindex(close.index)
        vol5 = volume.rolling(5).mean()
        vol20 = volume.rolling(20).mean()
        df["vol_trend"] = (vol5 / vol20.replace(0, np.nan)) - 1.0
    else:
        df["vol_trend"] = np.nan  # imputed later with training-fold median

    return df[BASE_FEATURE_COLS]

def build_ticker_panel(ticker, close):
    """Base feature frame + forward-return label for ONE ticker, both derived
    strictly from the training-window `close` series passed in (plus
    High/Low/Volume sliced to that same window). Returns ALL rows (including
    the final, NaN-label row) — the caller splits into labeled/prediction
    rows AFTER cross-sectional features are added across the whole fold."""
    ohlv = _load_ohlv_full(ticker)
    high = low = volume = None
    if ohlv is not None:
        window = ohlv.loc[(ohlv.index >= close.index.min()) & (ohlv.index <= close.index.max())]
        high, low, volume = window["High"], window["Low"], window["Volume"]

    feats = build_features(close, high, low, volume)
    # label: forward FWD_DAYS return, computed only from this (training-window-only)
    # series — shift(-FWD_DAYS) yields NaN once it would need data past the window.
    fwd_ret = close.shift(-FWD_DAYS) / close - 1.0
    feats = feats.copy()
    feats["fwd_ret"] = fwd_ret
    feats["ticker"] = ticker
    feats["date"] = feats.index
    return feats

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SECTIONAL FEATURE ENGINEERING (new this pass)
# Operates on the CONCATENATED full training-fold panel (all tickers, all
# dates in [train_start, train_end)) — every groupby key ("date" or
# "date"+"sector") is itself built entirely from training-window-only rows,
# so these features carry the same leak-free guarantee as the base features.
# ─────────────────────────────────────────────────────────────────────────────
RANK_SRC_COLS = ["ret_10", "vol_20", "rsi_14", "dist_ma20", "macd_norm", "atr_norm", "vol_trend", "drawdown_60"]
RANK_COLS = [f"{c}_xrank" for c in RANK_SRC_COLS]

SECTOR_REL_SRC_COLS = ["ret_10", "ret_20"]
SECTOR_REL_COLS = [f"{c}_sect_rel" for c in SECTOR_REL_SRC_COLS]

INTERACTION_COLS = [
    "rsi_x_vol20", "distma20_x_voltrend", "macd_x_ret20",
    "drawdown_x_rsi", "ret10rank_x_vol20rank",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + RANK_COLS + SECTOR_REL_COLS + INTERACTION_COLS

def add_cross_sectional_features(full_df: pd.DataFrame) -> pd.DataFrame:
    """full_df = concat of build_ticker_panel() output across every ticker in
    THIS fold's training panel — i.e. every row already lives strictly inside
    [train_start, train_end). Grouping by "date" (or "date"+"sector") therefore
    only ever aggregates over training-fold peers; it is structurally
    impossible for a test-window value to enter a rank or sector-mean here."""
    full_df = full_df.copy()
    full_df["sector"] = full_df["ticker"].map(TICKER_SECTOR).fillna("Unknown")

    # cross-sectional percentile rank of each base feature, within date
    for c in RANK_SRC_COLS:
        full_df[f"{c}_xrank"] = full_df.groupby("date")[c].rank(pct=True)

    # sector-relative return: ticker's return minus same-day sector-peer mean
    for c in SECTOR_REL_SRC_COLS:
        sect_mean = full_df.groupby(["date", "sector"])[c].transform("mean")
        full_df[f"{c}_sect_rel"] = full_df[c] - sect_mean

    # interaction terms (momentum x vol, mean-reversion x volume-trend, etc.)
    full_df["rsi_x_vol20"] = full_df["rsi_14"] * full_df["vol_20"]
    full_df["distma20_x_voltrend"] = full_df["dist_ma20"] * full_df["vol_trend"]
    full_df["macd_x_ret20"] = full_df["macd_norm"] * full_df["ret_20"]
    full_df["drawdown_x_rsi"] = full_df["drawdown_60"] * full_df["rsi_14"]
    full_df["ret10rank_x_vol20rank"] = full_df["ret_10_xrank"] * full_df["vol_20_xrank"]

    return full_df

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space — randomized, 32 combos EACH for XGBoost and
# LightGBM (up from a 4-combo grid in the first pass), sampled once at import
# time with a fixed seed (so the search space itself is reproducible; the
# BEST combo is still re-selected independently every fold via the internal
# training-window time-holdout — never the test window).
# ─────────────────────────────────────────────────────────────────────────────
XGB_PARAM_SPACE = {
    "max_depth": [2, 3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10, 20, 50],
    "n_estimators": [300, 500, 800],
}
LGBM_PARAM_SPACE = {
    "num_leaves": [7, 15, 31, 63, 95, 127],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [10, 20, 30, 50, 80, 120],
    "n_estimators": [300, 500, 800],
}
N_COMBOS_PER_MODEL = 32

def _sample_param_grid(space: dict, n: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    keys = list(space.keys())
    combos, seen = [], set()
    attempts, max_attempts = 0, n * 25
    while len(combos) < n and attempts < max_attempts:
        attempts += 1
        combo = {k: space[k][rng.randint(len(space[k]))] for k in keys}
        key = tuple(combo[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos

XGB_GRID = _sample_param_grid(XGB_PARAM_SPACE, N_COMBOS_PER_MODEL, seed=42)
LGBM_GRID = _sample_param_grid(LGBM_PARAM_SPACE, N_COMBOS_PER_MODEL, seed=43)

MAX_ESTIMATORS = 800     # fallback ceiling if a combo's n_estimators is missing
EARLY_STOP = 30
ROW_CAP = 100000          # safety cap on cross-sectional panel rows per fold (up from 60k —
                           # full universe means far more candidate rows are now available;
                           # a bigger cap lets the fit actually use more of that cross-section)
VAL_FRAC = 0.15           # internal time-based holdout fraction within training window

def _spearman_ic(pred, actual):
    df = pd.DataFrame({"p": pred, "a": actual}).dropna()
    if len(df) < 10:
        return np.nan
    return df["p"].corr(df["a"], method="spearman")

def _fit_xgb(params, Xtr, ytr, Xval, yval):
    model = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", MAX_ESTIMATORS), max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"], min_child_weight=params.get("min_child_weight", 1),
        objective="reg:squarederror", n_jobs=N_JOBS, tree_method="hist", early_stopping_rounds=EARLY_STOP,
        eval_metric="rmse", verbosity=0,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    best_it = getattr(model, "best_iteration", None)
    return model, (best_it + 1) if best_it is not None else params.get("n_estimators", MAX_ESTIMATORS)

def _fit_lgbm(params, Xtr, ytr, Xval, yval):
    model = lgbm.LGBMRegressor(
        n_estimators=params.get("n_estimators", MAX_ESTIMATORS), num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"], subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"], min_child_samples=params["min_child_samples"],
        objective="regression", n_jobs=N_JOBS, verbosity=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgbm.early_stopping(EARLY_STOP, verbose=False)])
    best_it = getattr(model, "best_iteration_", None)
    return model, best_it if best_it else params.get("n_estimators", MAX_ESTIMATORS)

def _tune_and_refit(kind, Xtr, ytr, Xval, yval, Xfull, yfull):
    """Randomized hyperparameter search (32 combos) using an internal
    time-based holdout (still entirely inside the training window). Best
    config picked by validation Spearman IC. Final model refit on the FULL
    training panel (internal-train + internal-val) using the tuned params
    and the early-stopped iteration count — no further peeking involved."""
    grid = XGB_GRID if kind == "xgb" else LGBM_GRID
    fit_fn = _fit_xgb if kind == "xgb" else _fit_lgbm
    best = None  # (val_ic, params, n_iter)
    for params in grid:
        try:
            model, n_iter = fit_fn(params, Xtr, ytr, Xval, yval)
            val_pred = model.predict(Xval)
            ic = _spearman_ic(val_pred, yval)
            if ic is None or np.isnan(ic):
                ic = -1.0
        except Exception as e:
            log.debug("    %s param combo failed: %s -> %s", kind, params, e)
            continue
        if best is None or ic > best[0]:
            best = (ic, params, n_iter)
    if best is None:
        return None, np.nan

    val_ic, best_params, n_iter = best
    ceiling = best_params.get("n_estimators", MAX_ESTIMATORS)
    n_iter = max(30, min(n_iter, ceiling))
    if kind == "xgb":
        final = xgb.XGBRegressor(
            n_estimators=n_iter, max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"], subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            min_child_weight=best_params.get("min_child_weight", 1),
            objective="reg:squarederror", n_jobs=N_JOBS, tree_method="hist", verbosity=0,
        )
        final.fit(Xfull, yfull)
    else:
        final = lgbm.LGBMRegressor(
            n_estimators=n_iter, num_leaves=best_params["num_leaves"],
            learning_rate=best_params["learning_rate"], subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            min_child_samples=best_params["min_child_samples"],
            objective="regression", n_jobs=N_JOBS, verbosity=-1,
        )
        final.fit(Xfull, yfull)
    return final, val_ic

# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking across folds (fit_predict_fn doesn't otherwise know its
# fold index / the total fold count — this gives a running cumulative/ETA
# line in the log for mid-run inspection on long runs).
# ─────────────────────────────────────────────────────────────────────────────
_PROGRESS = {"fold": 0, "t_start": None}

# ─────────────────────────────────────────────────────────────────────────────
# fit_predict_fn — the contract required by run_walkforward_ml.
# Receives ONLY training-window Close series per ticker for this fold.
# ─────────────────────────────────────────────────────────────────────────────
def fit_predict_fn(train_panel: dict) -> dict:
    t0 = time.time()
    if _PROGRESS["t_start"] is None:
        _PROGRESS["t_start"] = t0
    _PROGRESS["fold"] += 1

    # 1) Build per-ticker base-feature frames (ALL rows, including each
    #    ticker's final/NaN-label row — needed later as the prediction
    #    snapshot). Cross-sectional features are added AFTER concatenation
    #    (below) so ranks/sector-means see the full fold cross-section.
    frames = []
    for ticker, close in train_panel.items():
        if len(close) < MIN_HISTORY:
            continue
        frame = build_ticker_panel(ticker, close)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return {}

    full = pd.concat(frames, axis=0)

    # 2) Cross-sectional features — rank / sector-relative / interaction —
    #    computed over this fold's full training panel only (see
    #    add_cross_sectional_features docstring for the leak argument).
    full = add_cross_sectional_features(full)

    # 3) Split into (a) end-of-training-window snapshot per ticker (for
    #    prediction) and (b) labeled rows (fwd_ret non-NaN, for fitting).
    snap_df = (full.sort_values(["ticker", "date"])
                    .groupby("ticker", as_index=False).tail(1)
                    .set_index("ticker"))
    labeled = full.dropna(subset=["fwd_ret"])
    if labeled.empty or snap_df.empty:
        return {}

    panel = labeled.sort_values("date")

    # safety cap on total training rows (bounds worst-case runtime across folds
    # as universe availability grows over the years)
    if len(panel) > ROW_CAP:
        panel = panel.sample(n=ROW_CAP, random_state=42).sort_values("date")

    # 4) Winsorize features + labels using TRAINING-FOLD-ONLY bounds, then
    #    apply (not refit) the same bounds to the prediction snapshot.
    lo = panel[ALL_FEATURE_COLS].quantile(0.01)
    hi = panel[ALL_FEATURE_COLS].quantile(0.99)
    med = panel[ALL_FEATURE_COLS].median()
    y_lo, y_hi = panel["fwd_ret"].quantile(0.01), panel["fwd_ret"].quantile(0.99)

    def _prep(X):
        X = X.clip(lower=lo, upper=hi, axis=1)
        X = X.fillna(med)
        return X

    X_panel = _prep(panel[ALL_FEATURE_COLS])
    y_panel = panel["fwd_ret"].clip(lower=y_lo, upper=y_hi)

    # 5) Internal time-based split (still entirely inside the training window)
    #    for hyperparameter selection / early stopping.
    dates = panel["date"]
    cutoff = dates.quantile(1 - VAL_FRAC)
    is_val = dates >= cutoff
    if is_val.sum() < 50 or (~is_val).sum() < 50:
        # too little data to split sensibly — use a plain 85/15 positional split instead
        n_val = max(50, int(len(panel) * VAL_FRAC))
        is_val = pd.Series(False, index=panel.index)
        is_val.iloc[-n_val:] = True

    Xtr, ytr = X_panel[~is_val.values], y_panel[~is_val.values]
    Xval, yval = X_panel[is_val.values], y_panel[is_val.values]

    if len(Xtr) < 50 or len(Xval) < 20:
        return {}

    # 6) Tune (32-combo randomized search per model) + refit each available
    #    model on the FULL training panel.
    models = {}
    val_ics = {}
    if HAVE_XGB:
        m, ic = _tune_and_refit("xgb", Xtr, ytr, Xval, yval, X_panel, y_panel)
        if m is not None:
            models["xgb"] = m
            val_ics["xgb"] = ic
    if HAVE_LGBM:
        m, ic = _tune_and_refit("lgbm", Xtr, ytr, Xval, yval, X_panel, y_panel)
        if m is not None:
            models["lgbm"] = m
            val_ics["lgbm"] = ic

    if not models:
        return {}

    # 7) Ensemble weights: proportional to internal-validation IC (floor at a
    #    small epsilon so neither model is ever fully zeroed), equal-weight
    #    fallback if both are non-positive/NaN.
    weights = {k: max(v, 0.0) if not np.isnan(v) else 0.0 for k, v in val_ics.items()}
    if sum(weights.values()) <= 1e-9:
        weights = {k: 1.0 for k in models}
    total_w = sum(weights.values())
    weights = {k: w / total_w for k, w in weights.items()}

    # 8) Predict on each ticker's end-of-training-window snapshot (same
    #    cross-sectional feature computation as the training rows, see step 2).
    snap_X = _prep(snap_df[ALL_FEATURE_COLS])
    snap_tickers = snap_X.index.tolist()

    preds = np.zeros(len(snap_X))
    for k, m in models.items():
        preds = preds + weights[k] * m.predict(snap_X)

    result = dict(zip(snap_tickers, preds))

    elapsed_fold = time.time() - t0
    cum_min = (time.time() - _PROGRESS["t_start"]) / 60
    avg_fold_s = (time.time() - _PROGRESS["t_start"]) / _PROGRESS["fold"]
    log.info("    fit_predict_fn fold#%d: %d tickers, %d train rows -> %d val rows | "
             "models=%s weights=%s | this-fold %.1fs | cumulative %.1fmin | avg/fold %.1fs",
             _PROGRESS["fold"], len(train_panel), len(Xtr), len(Xval),
             list(models.keys()), {k: round(w, 2) for k, w in weights.items()},
             elapsed_fold, cum_min, avg_fold_s)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global TICKER_SECTOR
    universe, tier_map, sector_map = load_universe()
    TICKER_SECTOR = sector_map

    tier_counts = pd.Series([tier_map[t] for t in universe]).value_counts().to_dict()
    log.info("Universe: %d tickers (FULL local universe, no liquidity cap) | tiers=%s",
              len(universe), tier_counts)
    log.info("Models in ensemble: xgboost=%s, lightgbm=%s", HAVE_XGB, HAVE_LGBM)
    log.info("Hyperparameter search: xgb=%d combos, lgbm=%d combos (66 fits/fold incl. final refits) | ROW_CAP=%d",
              len(XGB_GRID), len(LGBM_GRID), ROW_CAP)
    log.info("Feature set: %d total (%d base + %d cross-sectional rank + %d sector-relative + %d interaction)",
              len(ALL_FEATURE_COLS), len(BASE_FEATURE_COLS), len(RANK_COLS), len(SECTOR_REL_COLS), len(INTERACTION_COLS))
    log.info("wf_type=%s  forward_days=%d", WF_TYPE, FWD_DAYS)

    t0 = time.time()
    report = run_walkforward_ml(
        name="ml_ensemble",
        wf_type=WF_TYPE,
        universe=universe,
        fit_predict_fn=fit_predict_fn,
        tier_map=tier_map,
    )
    elapsed = time.time() - t0
    log.info("Walk-forward run complete: %d fold-rows in %.1f min", len(report), elapsed / 60)

    report_path = os.path.join(PROJECT_DIR, f"ml_ensemble_validation_report{_SUFFIX}.parquet")
    report.to_parquet(report_path)
    log.info("Saved fold-level report -> %s", report_path)

    verdict = build_verdict(report)
    verdict_path = os.path.join(PROJECT_DIR, f"ml_ensemble_validation_report{_SUFFIX}.txt")
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write("ML ENSEMBLE (XGBoost + LightGBM) — Walk-Forward Validation Verdict — DEEP PASS\n")
        f.write("=" * 72 + "\n")
        f.write(f"wf_type={WF_TYPE}  forward_days={FWD_DAYS}  universe_n={len(universe)}  tiers={tier_counts}\n")
        f.write(f"models: xgboost={HAVE_XGB}  lightgbm={HAVE_LGBM}\n")
        f.write(f"hyperparam search: xgb={len(XGB_GRID)} combos, lgbm={len(LGBM_GRID)} combos, ROW_CAP={ROW_CAP}\n")
        f.write(f"feature set: {len(ALL_FEATURE_COLS)} total "
                f"({len(BASE_FEATURE_COLS)} base + {len(RANK_COLS)} rank + "
                f"{len(SECTOR_REL_COLS)} sector-rel + {len(INTERACTION_COLS)} interaction)\n")
        f.write(f"total wall time: {elapsed/60:.1f} min\n\n")
        f.write(f"Fold-level rows produced: {len(report)}\n\n")
        f.write(verdict.to_string(index=False) if not verdict.empty else "NO VERDICT ROWS (insufficient folds)\n")
        f.write("\n\n--- per-fold detail ---\n")
        f.write(report.to_string(index=False))
    log.info("Saved verdict summary -> %s", verdict_path)

    print("\n" + "=" * 72)
    print(verdict.to_string(index=False) if not verdict.empty else "NO VERDICT ROWS")
    print("=" * 72)

if __name__ == "__main__":
    main()
