"""
================================================================================
NSE LAYER 3 — RULES ENGINE (A/C/D/E)
Applies 4 quantitative trading rules to Layer 2 candidates
Input: nse_layer2_candidates.parquet
Output: nse_layer3_signals.parquet (ranked trade entry points + rules metadata)

FIXES in this version:
  1. REAL price history — batch yfinance download at start (not per-stock)
  2. Rule A — uses real GARCH on real returns, not white noise
  3. Rule C — uses spread_history.parquet properly, Hurst proxy as fallback
  4. Rule D — uses actual earnings_dna columns (recent_drift, direction)
  5. Rule E — real CUSUM on real momentum, not flat random array
  6. Batched download with fallback per-stock if batch misses a ticker
================================================================================
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
import yfinance as yf

from arch import arch_model
from scipy.stats import zscore

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_bool(val, default=False):
    try:
        return bool(val)
    except (TypeError, ValueError):
        return default


# ════════════════════════════════════════════════════════════════════════════════
# BATCH PRICE DOWNLOAD  ← replaces mock price series
# ════════════════════════════════════════════════════════════════════════════════

def batch_download_prices(tickers, period="1y"):
    """
    Download 1-year daily closes for all tickers in one yfinance call.
    Returns dict: { ticker -> np.array of closes (oldest first) }
    Fallback: per-stock download for any ticker missed in batch.
    """
    logger.info(f"  Batch downloading price history for {len(tickers)} tickers ...")
    nse_tickers = [t + ".NS" if not t.endswith(".NS") else t for t in tickers]
    ticker_map  = {nt: t for t, nt in zip(tickers, nse_tickers)}

    price_cache = {}

    try:
        raw = yf.download(
            nse_tickers,
            period=period,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        logger.warning(f"  Batch download failed: {e} — will fetch per stock")
        raw = None

    if raw is not None and not raw.empty:
        for nt, t in ticker_map.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    sub = raw[nt]["Close"] if nt in raw.columns.get_level_values(0) else None
                else:
                    sub = raw["Close"]
                if sub is not None and not sub.dropna().empty:
                    arr = sub.dropna().values
                    if len(arr) >= 20:
                        price_cache[t] = arr
            except Exception:
                pass

    # Per-stock fallback for misses
    missed = [t for t in tickers if t not in price_cache]
    if missed:
        logger.info(f"  Fetching {len(missed)} missed tickers individually ...")
        for t in missed:
            try:
                h = yf.download(t + ".NS", period=period, progress=False, auto_adjust=True)
                arr = h["Close"].dropna().values
                if len(arr) >= 20:
                    price_cache[t] = arr
            except Exception:
                pass

    logger.info(f"  Price cache built: {len(price_cache)} / {len(tickers)} tickers")
    return price_cache


# ════════════════════════════════════════════════════════════════════════════════
# RULE A: GARCH VOLATILITY BREAKOUT
# ════════════════════════════════════════════════════════════════════════════════

def rule_a_volatility_breakout(ticker, price_series, garch_params, current_close, current_volume):
    try:
        if len(price_series) < 20:
            return {'rule_a_fires': False, 'vol_zscore': 0.0, 'entry_score': 0.0,
                    'stop_loss': 0.0, 'target': 0.0, 'garch_sigma': 0.0}

        returns = np.diff(np.log(price_series)) * 100

        if garch_params and isinstance(garch_params, dict) and 'omega' in garch_params:
            sigma    = _safe_float(garch_params.get('sigma', np.std(returns)))
            mean_ret = _safe_float(garch_params.get('mean_ret', np.mean(returns)))
            if sigma <= 0:
                sigma = float(np.std(returns))
        else:
            try:
                fit = returns[-252:] if len(returns) >= 252 else returns
                model  = arch_model(fit, vol='Garch', p=1, q=1, mean='Constant')
                result = model.fit(disp='off', show_warning=False)
                sigma    = float(result.conditional_volatility.iloc[-1])
                mean_ret = float(result.params.get('mu', np.mean(returns)))
                if sigma <= 0 or np.isnan(sigma):
                    raise ValueError
            except Exception:
                sigma    = float(np.std(returns))
                mean_ret = float(np.mean(returns))

        # Normalize sigma if stored as percent (e.g. 2.3 means 2.3%)
        if sigma > 1.0:
            sigma = sigma / 100.0
        sigma = max(0.005, min(sigma, 0.15))

        upper_band  = mean_ret + 1.8 * sigma
        lower_band  = mean_ret - 0.5 * sigma
        target_band = mean_ret + 3.0 * sigma

        hist_last   = float(price_series[-1])
        current_ret = (current_close - hist_last) / hist_last * 100
        vol_zscore  = (current_ret - mean_ret) / sigma if sigma > 0 else 0.0

        fires = bool(current_ret > upper_band)
        band_range = target_band - upper_band
        entry_score = float(min(1.0, max(0.0, (current_ret - upper_band) / band_range))) \
                      if (fires and band_range > 0) else 0.0

        return {
            'rule_a_fires':    fires,
            'vol_zscore':      round(vol_zscore, 4),
            'upper_band':      round(upper_band, 4),
            'lower_band':      round(lower_band, 4),
            'target_band':     round(target_band, 4),
            'current_price':   current_close,
            'entry_score':     round(entry_score, 4),
            'stop_loss':       round(hist_last * (1 + lower_band / 100), 2),
            'target':          round(hist_last * (1 + target_band / 100), 2),
            'garch_sigma':     round(sigma, 4),
        }

    except Exception as e:
        logger.debug(f"Rule A error [{ticker}]: {e}")
        return {'rule_a_fires': False, 'vol_zscore': 0.0, 'entry_score': 0.0,
                'stop_loss': 0.0, 'target': 0.0, 'garch_sigma': 0.0}


# ════════════════════════════════════════════════════════════════════════════════
# RULE C: COINTEGRATED PAIRS MEAN REVERSION
# ════════════════════════════════════════════════════════════════════════════════

def rule_c_pairs_mr(ticker, price_series, spread_history_df, current_close, regime, pairs_eligible):
    try:
        if not _safe_bool(pairs_eligible) or regime == 'Crisis':
            return {'rule_c_fires': False, 'num_pairs': 0,
                    'spread_zscore': 0.0, 'pair_labels': [], 'entry_score': 0.0}

        spread_zscores = []
        pair_labels    = []

        # Use real spread history if available
        if spread_history_df is not None and not spread_history_df.empty:
            mask = pd.Series(False, index=spread_history_df.index)
            if 'ticker_a' in spread_history_df.columns:
                mask |= (spread_history_df['ticker_a'] == ticker)
            if 'ticker_b' in spread_history_df.columns:
                mask |= (spread_history_df['ticker_b'] == ticker)
            pair_rows = spread_history_df[mask].head(3)

            if not pair_rows.empty and 'spread_zscore' in pair_rows.columns:
                for _, pr in pair_rows.iterrows():
                    sz = _safe_float(pr.get('spread_zscore', 0))
                    spread_zscores.append(abs(sz))
                    ta = pr.get('ticker_a', ticker)
                    tb = pr.get('ticker_b', '?')
                    pair_labels.append(f"{ta}/{tb}")

        # Hurst proxy fallback
        if not spread_zscores and len(price_series) >= 20:
            ret20 = np.diff(np.log(price_series[-20:])) * 100
            if ret20.std() > 0:
                proxy_z = abs(ret20[-1] / ret20.std())
                spread_zscores.append(proxy_z)
                pair_labels.append(f"{ticker}/proxy")

        if not spread_zscores:
            return {'rule_c_fires': False, 'num_pairs': 0,
                    'spread_zscore': 0.0, 'pair_labels': [], 'entry_score': 0.0}

        mean_sz     = float(np.mean(spread_zscores))
        fires       = bool(mean_sz > 2.0)
        entry_score = float(min(1.0, max(0.0, (mean_sz - 2.0) / 2.0))) if fires else 0.0

        return {
            'rule_c_fires':     fires,
            'num_pairs':        len(spread_zscores),
            'spread_zscore':    round(mean_sz, 4),
            'pair_labels':      pair_labels,
            'entry_score':      round(entry_score, 4),
        }

    except Exception as e:
        logger.debug(f"Rule C error [{ticker}]: {e}")
        return {'rule_c_fires': False, 'num_pairs': 0,
                'spread_zscore': 0.0, 'pair_labels': [], 'entry_score': 0.0}


# ════════════════════════════════════════════════════════════════════════════════
# RULE D: EVENT-DRIVEN  (uses actual earnings_dna columns)
# ════════════════════════════════════════════════════════════════════════════════

def rule_d_event_driven(ticker, crisis_alpha_raw, recent_drift, direction):
    """
    Uses real earnings_dna columns: recent_drift (float), direction (UP/DOWN/FLAT).
    No days_to_result available — urgency fixed at neutral 0.3.
    Fires when analyst drift is strongly positive AND direction is UP.
    """
    try:
        crisis_alpha  = _safe_float(crisis_alpha_raw, default=0.0)
        recent_drift  = _safe_float(recent_drift, default=0.0)
        direction     = str(direction).strip().upper() if direction else 'FLAT'

        # Convert tiny decimal drift to % scale (0.005 → 0.5%)
        drift_pct = recent_drift * 100.0

        # Direction multiplier
        dir_mult = 1.0 if direction == 'UP' else (-0.5 if direction == 'DOWN' else 0.0)
        adjusted_drift = drift_pct * dir_mult

        # Sub-scores
        alpha_score = float(min(1.0, max(0.0, (crisis_alpha + 0.5) / 1.5)))
        drift_score = float(min(1.0, max(0.0, adjusted_drift / 2.0)))  # 2% = full score
        urgency_score = 0.3  # no date column — neutral

        event_score = 0.4 * alpha_score + 0.4 * drift_score + 0.2 * urgency_score
        fires       = bool(event_score > 0.35)
        entry_score = float(event_score) if fires else 0.0

        return {
            'rule_d_fires':      fires,
            'crisis_alpha':      round(crisis_alpha, 4),
            'earnings_drift':    round(adjusted_drift, 4),
            'days_to_result':    999,
            'event_score':       round(event_score, 4),
            'entry_score':       round(entry_score, 4),
        }

    except Exception as e:
        logger.debug(f"Rule D error [{ticker}]: {e}")
        return {'rule_d_fires': False, 'crisis_alpha': 0.0, 'earnings_drift': 0.0,
                'days_to_result': 999, 'event_score': 0.0, 'entry_score': 0.0}


# ════════════════════════════════════════════════════════════════════════════════
# RULE E: CUSUM MOMENTUM BREAKOUT
# ════════════════════════════════════════════════════════════════════════════════

def rule_e_cusum_momentum(ticker, price_series, regime, hurst=None, threshold=5.0):
    try:
        hurst = _safe_float(hurst, default=0.5)

        if regime == 'Crisis' or len(price_series) < 20:
            return {'rule_e_fires': False, 'cusum_value': 0.0,
                    'momentum': 0.0, 'entry_score': 0.0}

        returns = np.diff(np.log(price_series)) * 100
        ret20   = returns[-20:]

        if ret20.std() == 0:
            return {'rule_e_fires': False, 'cusum_value': 0.0,
                    'momentum': 0.0, 'entry_score': 0.0}

        ret_z     = zscore(ret20)
        cusum     = np.cumsum(ret_z)
        max_cusum = float(np.max(np.abs(cusum)))

        momentum = float(
            (price_series[-1] - price_series[-5]) / price_series[-5] * 100
        ) if len(price_series) >= 5 else 0.0

        fires = bool((max_cusum > threshold) and (momentum > 0.5))

        hurst_penalty = 0.7 if hurst < 0.45 else (0.85 if hurst < 0.5 else 1.0)
        entry_score   = float(min(1.0, max_cusum / (threshold * 2.0))) * hurst_penalty \
                        if fires else 0.0

        return {
            'rule_e_fires':  fires,
            'cusum_value':   round(max_cusum, 4),
            'momentum':      round(momentum, 4),
            'hurst':         round(hurst, 4),
            'entry_score':   round(entry_score, 4),
        }

    except Exception as e:
        logger.debug(f"Rule E error [{ticker}]: {e}")
        return {'rule_e_fires': False, 'cusum_value': 0.0,
                'momentum': 0.0, 'entry_score': 0.0}


# ════════════════════════════════════════════════════════════════════════════════
# AGGREGATOR
# ════════════════════════════════════════════════════════════════════════════════

def aggregate_rules_and_score(tier, rule_a_score, rule_c_score, rule_d_score,
                               rule_e_score, ml_score=0.5):
    tier_map   = {'T1': 1.0, 'T2': 0.8, 'T3': 0.6}
    tier_score = tier_map.get(str(tier), 0.6)

    scores      = [s for s in [rule_c_score, rule_e_score] if s > 0]
    mr_strength = float(np.mean(scores)) if scores else 0.0

    final_score = (
        0.35 * ml_score     +
        0.25 * mr_strength  +
        0.20 * rule_a_score +
        0.10 * rule_d_score +
        0.10 * tier_score
    )
    return round(float(final_score), 6)


# ════════════════════════════════════════════════════════════════════════════════
# DTYPE ENFORCEMENT
# ════════════════════════════════════════════════════════════════════════════════

def _enforce_dtypes(df):
    bool_cols  = ['rule_a_fires', 'rule_c_fires', 'rule_d_fires', 'rule_e_fires', 'pairs_eligible']
    float_cols = ['price_close', 'volume', 'turnover_cr',
                  'rule_a_vol_zscore', 'rule_a_entry_score', 'rule_a_stop_loss', 'rule_a_target',
                  'rule_a_garch_sigma', 'rule_a_upper_band', 'rule_a_lower_band',
                  'rule_c_spread_zscore', 'rule_c_entry_score',
                  'rule_d_crisis_alpha', 'rule_d_earnings_drift', 'rule_d_event_score',
                  'rule_d_entry_score',
                  'rule_e_cusum', 'rule_e_momentum', 'rule_e_entry_score', 'rule_e_hurst',
                  'vix', 'hurst', 'final_score', 'ml_score_placeholder']
    int_cols   = ['rule_c_pairs', 'rule_d_days_to_result', 'num_rules_firing', 'rank']
    str_cols   = ['ticker', 'companyname', 'tier', 'sector', 'regime', 'r2_grade', 'run_date']

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: bool(x) if x is not None else False).astype(bool)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    if 'crisis_alpha_flag' in df.columns:
        df['crisis_alpha_flag'] = df['crisis_alpha_flag'].map(
            lambda x: bool(x) if x is not None else False).astype(bool)
    return df


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def run_layer3_rules_engine(layer2_candidates_path, garch_params_path,
                             earnings_dna_path, spread_history_path, output_path):

    logger.info("")
    logger.info("=" * 80)
    logger.info("  NSE DECISION ENGINE - LAYER 3 RULES ENGINE")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # ── STEP 1: Load Layer 2 candidates ──────────────────────────────────────
    logger.info("\n[STEP 1] Loading Layer 2 candidates ...")
    try:
        candidates = pd.read_parquet(layer2_candidates_path)
        logger.info(f"  Loaded {len(candidates)} candidates")
    except FileNotFoundError:
        logger.error(f"  NOT FOUND: {layer2_candidates_path}")
        return None

    # ── STEP 2: Load ancillary data ───────────────────────────────────────────
    logger.info("\n[STEP 2] Loading ancillary data ...")

    garch_params_dict = {}
    try:
        garch_df = pd.read_parquet(garch_params_path)
        if 'ticker' in garch_df.columns:
            garch_params_dict = {row['ticker']: row.to_dict()
                                 for _, row in garch_df.iterrows()}
            logger.info(f"  GARCH params: {len(garch_params_dict)} tickers")
    except Exception as e:
        logger.warning(f"  GARCH params not loaded: {e}")

    earnings_dict = {}
    try:
        earnings_df = pd.read_parquet(earnings_dna_path)
        logger.info(f"  Earnings DNA columns: {list(earnings_df.columns)}")
        if 'ticker' in earnings_df.columns:
            earnings_dict = {row['ticker']: row.to_dict()
                             for _, row in earnings_df.iterrows()}
            logger.info(f"  Earnings DNA: {len(earnings_dict)} tickers")
    except Exception as e:
        logger.warning(f"  Earnings DNA not loaded: {e}")

    spread_history_df = None
    try:
        spread_history_df = pd.read_parquet(spread_history_path)
        logger.info(f"  Spread history: {len(spread_history_df)} pairs")
    except Exception as e:
        logger.warning(f"  Spread history not loaded: {e} — Rule C uses Hurst proxy")

    # ── STEP 3: Batch download real price history ─────────────────────────────
    logger.info("\n[STEP 3] Downloading real price history (batch) ...")
    tickers     = candidates['ticker'].tolist()
    price_cache = batch_download_prices(tickers, period="1y")
    missing_pct = (len(tickers) - len(price_cache)) / len(tickers) * 100
    logger.info(f"  Coverage: {len(price_cache)}/{len(tickers)} "
                f"({100-missing_pct:.1f}%)  |  Missing: {missing_pct:.1f}%")

    # ── STEP 4: Apply rules ───────────────────────────────────────────────────
    logger.info(f"\n[STEP 4] Applying Rules A/C/D/E to {len(candidates)} candidates ...")

    results      = []
    rule_a_count = rule_c_count = rule_d_count = rule_e_count = 0

    for idx, row in candidates.iterrows():
        ticker = str(row['ticker'])
        close  = _safe_float(row.get('close', 0))
        regime = str(row.get('regime', 'Sideways'))
        hurst  = _safe_float(row.get('hurst', 0.5))

        # Real price series — fallback to single-point placeholder only if missing
        price_series = price_cache.get(ticker)
        if price_series is None or len(price_series) < 20:
            # Minimal fallback: flat line, rules will return 0
            price_series = np.full(20, close)

        # ── Rule A ────────────────────────────────────────────────────────────
        rule_a = rule_a_volatility_breakout(
            ticker, price_series,
            garch_params_dict.get(ticker),
            close, _safe_float(row.get('volume', 0))
        )
        if rule_a.get('rule_a_fires'):
            rule_a_count += 1

        # ── Rule C ────────────────────────────────────────────────────────────
        rule_c = rule_c_pairs_mr(
            ticker, price_series, spread_history_df,
            close, regime, row.get('pairs_eligible', False)
        )
        if rule_c.get('rule_c_fires'):
            rule_c_count += 1

        # ── Rule D ────────────────────────────────────────────────────────────
        earnings_data  = earnings_dict.get(ticker, {})
        rule_d = rule_d_event_driven(
            ticker,
            row.get('crisis_alpha', 0),
            earnings_data.get('recent_drift', 0),
            earnings_data.get('direction', 'FLAT'),
        )
        if rule_d.get('rule_d_fires'):
            rule_d_count += 1

        # ── Rule E ────────────────────────────────────────────────────────────
        rule_e = rule_e_cusum_momentum(
            ticker, price_series, regime, hurst, threshold=5.0
        )
        if rule_e.get('rule_e_fires'):
            rule_e_count += 1

        # ── Aggregate ─────────────────────────────────────────────────────────
        final_score = aggregate_rules_and_score(
            tier         = row.get('tier', 'T3'),
            rule_a_score = _safe_float(rule_a.get('entry_score', 0)),
            rule_c_score = _safe_float(rule_c.get('entry_score', 0)),
            rule_d_score = _safe_float(rule_d.get('entry_score', 0)),
            rule_e_score = _safe_float(rule_e.get('entry_score', 0)),
            ml_score     = 0.5,
        )

        num_firing = int(sum([
            bool(rule_a.get('rule_a_fires', False)),
            bool(rule_c.get('rule_c_fires', False)),
            bool(rule_d.get('rule_d_fires', False)),
            bool(rule_e.get('rule_e_fires', False)),
        ]))

        signal = {
            'ticker':       ticker,
            'companyname':  str(row.get('companyname', '')),
            'tier':         str(row.get('tier', '')),
            'sector':       str(row.get('sector', '')),
            'price_close':  round(close, 2),
            'volume':       _safe_float(row.get('volume', 0)),
            'turnover_cr':  _safe_float(row.get('turnover_cr', 0)),

            'rule_a_fires':         bool(rule_a.get('rule_a_fires', False)),
            'rule_a_vol_zscore':    _safe_float(rule_a.get('vol_zscore', 0)),
            'rule_a_entry_score':   _safe_float(rule_a.get('entry_score', 0)),
            'rule_a_stop_loss':     _safe_float(rule_a.get('stop_loss', 0)),
            'rule_a_target':        _safe_float(rule_a.get('target', 0)),
            'rule_a_garch_sigma':   _safe_float(rule_a.get('garch_sigma', 0)),
            'rule_a_upper_band':    _safe_float(rule_a.get('upper_band', 0)),
            'rule_a_lower_band':    _safe_float(rule_a.get('lower_band', 0)),

            'rule_c_fires':         bool(rule_c.get('rule_c_fires', False)),
            'rule_c_pairs':         int(rule_c.get('num_pairs', 0)),
            'rule_c_spread_zscore': _safe_float(rule_c.get('spread_zscore', 0)),
            'rule_c_entry_score':   _safe_float(rule_c.get('entry_score', 0)),

            'rule_d_fires':          bool(rule_d.get('rule_d_fires', False)),
            'rule_d_crisis_alpha':   _safe_float(rule_d.get('crisis_alpha', 0)),
            'rule_d_earnings_drift': _safe_float(rule_d.get('earnings_drift', 0)),
            'rule_d_days_to_result': int(min(_safe_float(rule_d.get('days_to_result', 999)), 999)),
            'rule_d_event_score':    _safe_float(rule_d.get('event_score', 0)),
            'rule_d_entry_score':    _safe_float(rule_d.get('entry_score', 0)),

            'rule_e_fires':         bool(rule_e.get('rule_e_fires', False)),
            'rule_e_cusum':         _safe_float(rule_e.get('cusum_value', 0)),
            'rule_e_momentum':      _safe_float(rule_e.get('momentum', 0)),
            'rule_e_entry_score':   _safe_float(rule_e.get('entry_score', 0)),
            'rule_e_hurst':         _safe_float(rule_e.get('hurst', hurst)),

            'regime':           regime,
            'vix':              _safe_float(row.get('vix', 0)),
            'r2_grade':         str(row.get('r2_grade', 'D')),
            'hurst':            hurst,
            'pairs_eligible':   bool(row.get('pairs_eligible', False)),
            'crisis_alpha_flag':bool(row.get('crisis_alpha', False)),

            'num_rules_firing':     num_firing,
            'final_score':          final_score,
            'ml_score_placeholder': 0.5,
            'run_date':             datetime.now().strftime('%Y-%m-%d'),
        }
        results.append(signal)

    # ── STEP 5: Rank + save ───────────────────────────────────────────────────
    logger.info(f"\n[STEP 5] Ranking and saving {len(results)} signals ...")
    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values('final_score', ascending=False).reset_index(drop=True)
    output_df['rank'] = (output_df.index + 1).astype('int32')
    output_df = _enforce_dtypes(output_df)

    try:
        output_df.to_parquet(output_path, index=False)
        logger.info(f"  Saved: {output_path}  ({len(output_df)} rows, {output_df.shape[1]} cols)")
    except Exception as e:
        logger.error(f"  Parquet save failed: {e}")
        csv_path = output_path.replace('.parquet', '_emergency.csv')
        output_df.to_csv(csv_path, index=False)
        logger.warning(f"  Emergency CSV: {csv_path}")

    _print_report(output_df, rule_a_count, rule_c_count, rule_d_count, rule_e_count)
    return output_df


# ════════════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════════════

def _print_report(df, ra, rc, rd, re):
    n = len(df)
    logger.info("\n" + "=" * 80)
    logger.info("  NSE DECISION ENGINE — LAYER 3 RULES REPORT")
    logger.info(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"  Signals: {n}")
    logger.info("=" * 80)
    logger.info(f"\n  RULE FIRING SUMMARY")
    logger.info(f"  {'─'*50}")
    logger.info(f"  Rule A  (Vol Breakout)   : {ra:5d}  ({ra/n*100:.1f}%)")
    logger.info(f"  Rule C  (Pairs MR)       : {rc:5d}  ({rc/n*100:.1f}%)")
    logger.info(f"  Rule D  (Event-driven)   : {rd:5d}  ({rd/n*100:.1f}%)")
    logger.info(f"  Rule E  (CUSUM Momentum) : {re:5d}  ({re/n*100:.1f}%)")
    logger.info(f"\n  MULTI-RULE SYNERGY")
    logger.info(f"  {'─'*50}")
    for k, label in [(0,'0 rules (watch)'),(1,'1 rule (weak)'),(2,'2 rules (confirmed)'),
                     (3,'3 rules (strong)'),(4,'4 rules (very strong)')]:
        cnt = (df['num_rules_firing'] == k).sum()
        logger.info(f"  {label:<28}: {cnt:5d}")
    logger.info(f"\n  TOP 25 SIGNALS (by final_score)")
    logger.info(f"  {'─'*50}")
    for _, s in df.head(25).iterrows():
        rules_str = "".join([
            'A' if s['rule_a_fires'] else '.',
            'C' if s['rule_c_fires'] else '.',
            'D' if s['rule_d_fires'] else '.',
            'E' if s['rule_e_fires'] else '.',
        ])
        logger.info(f"  {int(s['rank']):3d}  {s['ticker']:14s}  {s['tier']}  "
                    f"{s['sector'][:18]:18s}  ₹{s['price_close']:8.2f}  "
                    f"[{rules_str}]  Score:{s['final_score']:.4f}")
    logger.info("\n" + "=" * 80)
    logger.info("  NEXT: Run Layer 4 Portfolio Constructor")
    logger.info("=" * 80 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    BASE = r'D:\MBA\STOCK MARKET RESEARCH\NSE quants py'

    signals_df = run_layer3_rules_engine(
        layer2_candidates_path = rf'{BASE}\nse_layer2_candidates.parquet',
        garch_params_path      = rf'{BASE}\garch_params.parquet',
        earnings_dna_path      = rf'{BASE}\earnings_dna.parquet',
        spread_history_path    = rf'{BASE}\spread_history.parquet',
        output_path            = rf'{BASE}\nse_layer3_signals.parquet',
    )

    if signals_df is not None:
        report_path = rf'{BASE}\nse_layer3_signals_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Layer 3 Signals Report\nGenerated: {datetime.now()}\n")
            f.write(f"Total: {len(signals_df)} signals\n\nTop 25:\n")
            f.write(signals_df.head(25).to_string())
        logger.info(f"Text report saved: {report_path}")