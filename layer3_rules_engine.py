"""
================================================================================
NSE LAYER 3 â€” RULES ENGINE (A/C/D/E)
Applies 4 quantitative trading rules to Layer 2 candidates
Input: layer2_candidates.parquet (1,239 stocks)
Output: layer3_signals.parquet (ranked trade entry points + rules metadata)
================================================================================

ARCHITECTURE:
  Rule A (Vol Breakout)  â†’ GARCH volatility + price breaks
  Rule C (Pairs MR)      â†’ Cointegrated pairs mean reversion
  Rule D (Event)         â†’ Earnings surprise + analyst upgrade detection
  Rule E (CUSUM Mom)     â†’ Momentum breakout with change-point detection

Each rule fires independently. Final score: 0.4Ã—ML + 0.3Ã—MR_strength + 0.2Ã—Granger + 0.1Ã—tier
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
import json

# Quantitative packages
from arch import arch_model
from statsmodels.tsa.stattools import coint
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore, norm, rankdata
from scipy.optimize import minimize_scalar
import networkx as nx

warnings.filterwarnings('ignore')

import yfinance as yf
from layer3_fno_patch import load_fno_data, rule_fno_boost, aggregate_rules_and_score_v2
from smc_bridge import build_ohlc_cache, compute_smc_score

def batch_download_prices(tickers, period="1y"):
    nse_tickers = [t + ".NS" if not t.endswith(".NS") else t for t in tickers]
    ticker_map  = {nt: t for t, nt in zip(tickers, nse_tickers)}
    price_cache = {}
    try:
        raw = yf.download(nse_tickers, period=period, progress=False, auto_adjust=True, group_by="ticker")
    except Exception as e:
        logger.warning(f"Batch download failed: {e}")
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
    logger.info(f"  Price cache: {len(price_cache)}/{len(tickers)} tickers")
    return price_cache

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOGGING SETUP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RULE A: GARCH VOLATILITY BREAKOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def rule_a_volatility_breakout(ticker, price_series, garch_params, current_close, current_volume):
    """
    Rule A: Entry when price breaks ABOVE garch_sigma Ã— 1.8 confidence band.
    
    LOGIC:
    - Fit GARCH(1,1) to returns
    - Compute conditional volatility forecast
    - Entry: close > mean + 1.8Ïƒ (upper band)
    - Stop: close < mean - 0.5Ïƒ (lower band)
    - Target: +3Ïƒ move
    
    Returns:
    {
        'rule_a_fires': bool,
        'vol_zscore': float,
        'upper_band': float,
        'current_price': float,
        'entry_score': float (0-1)
    }
    """
    try:
        if len(price_series) < 20:
            return {'rule_a_fires': False, 'vol_zscore': 0, 'entry_score': 0}
        
        returns = np.diff(np.log(price_series)) * 100  # log returns in %
        
        # Read GARCH params or fit fresh
        if garch_params is not None and 'omega' in garch_params:
            # normalize sigma from % to decimal if needed
            sigma = garch_params.get('sigma', np.std(returns))
            mean_ret = garch_params.get('mean_ret', np.mean(returns))
        else:
            # Fit fresh GARCH(1,1)
            try:
                model = arch_model(returns[-252:] if len(returns) >= 252 else returns, vol='Garch', p=1, q=1)
                result = model.fit(disp='off', show_warning=False)
                sigma = result.conditional_volatility.iloc[-1]
                mean_ret = np.mean(returns)
            except:
                sigma = np.std(returns)
                mean_ret = np.mean(returns)
        
        # Bollinger bands
        upper_band = mean_ret + 1.8 * sigma
        lower_band = mean_ret - 0.5 * sigma
        target_band = mean_ret + 3.0 * sigma
        
        # Current z-score
        if len(price_series) < 2:
            return {'rule_a_fires': False, 'entry_score': 0, 'vol_zscore': 0, 'upper_band': 0, 'lower_band': 0, 'target_band': 0, 'current_price': current_close}
        current_ret = (current_close - price_series[-1]) / price_series[-1] * 100
        vol_zscore = (current_ret - mean_ret) / sigma if sigma > 0 else 0
        
        # Long: price breaks above upper band
        fires = vol_zscore > 1.5
        entry_score = min(1.0, max(0, (current_ret - upper_band) / (target_band - upper_band))) if fires else 0

        # Short: price breaks below lower band (downside vol breakout)
        short_target_band = mean_ret - 3.0 * sigma
        short_fires = vol_zscore < -1.5
        short_entry_score = min(1.0, max(0, (lower_band - current_ret) / (lower_band - short_target_band + 1e-9))) if short_fires else 0

        return {
            'rule_a_fires':       fires,
            'rule_a_short_fires': short_fires,
            'vol_zscore':         vol_zscore,
            'upper_band':         upper_band,
            'lower_band':         lower_band,
            'target_band':        target_band,
            'current_price':      current_close,
            'entry_score':        entry_score,
            'rule_a_short_score': short_entry_score,
            'stop_loss':  price_series[-1] * (1 + lower_band / 100),
            'target':     price_series[-1] * (1 + target_band / 100),
        }
    except Exception as e:
        logger.debug(f"Rule A error for {ticker}: {str(e)}")
        return {'rule_a_fires': False, 'rule_a_short_fires': False,
                'vol_zscore': 0, 'entry_score': 0, 'rule_a_short_score': 0}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RULE C: COINTEGRATED PAIRS MEAN REVERSION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def rule_c_pairs_mr(ticker, price_series, candidate_pairs, current_close, regime, pairs_eligible, price_cache={}):
    """
    Rule C: Mean reversion on cointegrated pairs.
    
    LOGIC:
    - Find cointegrated pairs from pairs_eligible list
    - Spread = log(price_a) - Î²*log(price_b)
    - Entry: spread > 2Ïƒ (pair overspread)
    - Exit: spread returns to mean
    
    Returns:
    {
        'rule_c_fires': bool,
        'num_pairs': int,
        'spread_zscore': float,
        'pair_labels': list,
        'entry_score': float
    }
    """
    try:
        if not pairs_eligible or regime == 'Crisis':
            return {
                'rule_c_fires': False,
                'num_pairs': 0,
                'spread_zscore': 0,
                'pair_labels': [],
                'entry_score': 0
            }
        
        valid_pairs = []
        
        for pair_str in (candidate_pairs or [])[:3]:  # Check top 3 pairs
            try:
                ticker_a, ticker_b = pair_str.split(' / ')
                if ticker_a == ticker:
                    # We are the first leg; partner is ticker_b
                    valid_pairs.append((ticker_a, ticker_b))
                elif ticker_b == ticker:
                    # We are the second leg; partner is ticker_a
                    valid_pairs.append((ticker_b, ticker_a))
            except:
                pass
        
        if not valid_pairs:
            return {
                'rule_c_fires': False,
                'num_pairs': 0,
                'spread_zscore': 0,
                'pair_labels': [],
                'entry_score': 0
            }
        
        spread_zscores = []
        for ta, tb in valid_pairs:
            try:
                prices_b = price_cache.get(tb)
                if prices_b is None or len(prices_b) < 20 or len(price_series) < 20:
                    continue
                min_len = min(len(price_series), len(prices_b))
                log_spread = np.log(price_series[-min_len:] + 1e-10) - np.log(prices_b[-min_len:] + 1e-10)
                if len(log_spread) < 20:
                    continue
                spread_mean = np.mean(log_spread)
                spread_std = np.std(log_spread) + 1e-10
                z = (log_spread[-1] - spread_mean) / spread_std
                spread_zscores.append(z)  # keep sign: negative = stock is cheap leg
            except:
                pass
        
        if not spread_zscores:
            return {
                'rule_c_fires': False,
                'num_pairs': 0,
                'spread_zscore': 0,
                'pair_labels': [],
                'entry_score': 0
            }
        
        mean_spread_zscore = np.mean(spread_zscores)

        # Long: stock is undervalued vs partner (spread below -2σ → expect reversion up)
        fires       = mean_spread_zscore < -2.0
        entry_score = min(1.0, max(0, (-mean_spread_zscore - 2.0) / 2.0)) if fires else 0

        # Short: stock is overvalued vs partner (spread above +2σ → expect reversion down)
        short_fires       = mean_spread_zscore > 2.0
        short_entry_score = min(1.0, max(0, (mean_spread_zscore - 2.0) / 2.0)) if short_fires else 0

        return {
            'rule_c_fires':       fires,
            'rule_c_short_fires': short_fires,
            'num_pairs':          len(valid_pairs),
            'spread_zscore':      mean_spread_zscore,
            'pair_labels':        [f"{ta}/{tb}" for ta, tb in valid_pairs],
            'entry_score':        entry_score,
            'rule_c_short_score': short_entry_score,
        }
    except Exception as e:
        logger.debug(f"Rule C error for {ticker}: {str(e)}")
        return {
            'rule_c_fires': False, 'rule_c_short_fires': False,
            'num_pairs': 0, 'spread_zscore': 0,
            'pair_labels': [], 'entry_score': 0, 'rule_c_short_score': 0,
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RULE D: EVENT-DRIVEN (EARNINGS SURPRISE + ANALYST UPGRADES)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def rule_d_event_driven(ticker, price_series, crisis_alpha, days_to_result):
    """
    Rule D: Event-driven signals from earnings drift + analyst sentiment.
    
    LOGIC:
    - crisis_alpha > 0  â†’ stock shows alpha in crisis (earnings quality high)
    - recent_earnings_drift > 0  â†’ analyst expectations rising
    - days_to_result < 5  â†’ earnings imminent (vol expansion expected)
    
    Returns:
    {
        'rule_d_fires': bool,
        'crisis_alpha': float,
        'earnings_drift': float,
        'entry_score': float
    }
    """
    try:
        # Compute event scores
        alpha_score = float(crisis_alpha) if crisis_alpha is not None else 0.0  # already 0-1 percentile rank

        # Proximity score: how close is the result date?
        # 3-5 days = sweet spot (pre-result momentum window)
        # 0-2 days = too close (blackout), score drops
        # >5 days or 999 = no urgency
        if days_to_result is not None and days_to_result != 999:
            if 3 <= days_to_result <= 5:
                proximity_score = 1.0   # ideal window
            elif days_to_result == 6 or days_to_result == 7:
                proximity_score = 0.6   # close window
            elif 0 <= days_to_result <= 2:
                proximity_score = 0.0   # blackout â€” do not trade
            elif days_to_result < 0:
                proximity_score = 0.2   # just passed
            else:
                proximity_score = 0.1   # >7 days, weak signal
        else:
            proximity_score = 0.0

        # Composite event score
        event_score = 0.5 * alpha_score + 0.5 * proximity_score

        # Fire if strong alpha AND result is 3-7 days out (not in blackout)
        fires = alpha_score > 0.5 and proximity_score >= 0.6 and event_score > 0.4
        entry_score = event_score if fires else 0
        
        return {
            'rule_d_fires': fires,
            'crisis_alpha': crisis_alpha if crisis_alpha else 0,
            'proximity_score': proximity_score,
            'days_to_result': days_to_result if days_to_result else 999,
            'event_score': event_score,
            'entry_score': entry_score,
        }
    except Exception as e:
        logger.debug(f"Rule D error for {ticker}: {str(e)}")
        return {
            'rule_d_fires': False,
            'crisis_alpha': 0,
            'earnings_drift': 0,
            'days_to_result': 999,
            'event_score': 0,
            'entry_score': 0
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RULE E: CUSUM MOMENTUM BREAKOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def rule_e_cusum_momentum(ticker, price_series, regime, hurst=None, threshold=5.0):
    """
    Rule E: Page's CUSUM change-point detection on upside momentum.

    Fixes over old version:
    - Proper one-sided Page's CUSUM (resets to 0 on negative evidence)
    - 60-day baseline for mu/sigma (not 20-day zscore which amplifies noise)
    - Upside only — abs() was also catching downtrends
    - Regime-aware thresholds: Sideways=7.0, Bull=4.5
    - 10-day momentum at 1.5% (Sideways) / 0.5% (Bull) — 5-day 0.5% was noise
    - Trend filter: price must be above 20-day MA
    - Persistence: CUSUM must have been building for >=3 consecutive days
    - Hurst gate in Sideways: requires Hurst > 0.52 (stock must show trending DNA)
    """
    try:
        if regime == 'Crisis' or len(price_series) < 60:
            return {'rule_e_fires': False, 'cusum_value': 0, 'momentum': 0, 'entry_score': 0}

        # Regime-aware threshold — Sideways needs much stronger signal
        regime_threshold = {'Bull': 4.5, 'Sideways': 7.0}.get(regime, threshold)

        returns = np.diff(np.log(price_series)) * 100  # log returns in %

        # Use 60-day baseline for stable mu/sigma — avoids 20-day noise amplification
        baseline = returns[-60:]
        mu    = np.mean(baseline)
        sigma = np.std(baseline) + 1e-9

        # Proper Page's one-sided CUSUM (upside only — we are long-only)
        # S[t] = max(0, S[t-1] + (r[t] - mu)/sigma - k)
        # k = 0.5 = slack allowance before evidence accumulates
        k = 0.5
        cusum_pos = np.zeros(len(returns))
        for i in range(1, len(returns)):
            cusum_pos[i] = max(0.0, cusum_pos[i - 1] + (returns[i] - mu) / sigma - k)

        current_cusum = cusum_pos[-1]

        # Persistence: CUSUM must have been building for >=3 consecutive days
        # Prevents single-day spikes from firing
        persistent = (
            len(cusum_pos) >= 4 and
            cusum_pos[-1] > cusum_pos[-2] and
            cusum_pos[-2] > cusum_pos[-3] and
            cusum_pos[-3] > 0
        )

        # Trend filter: price must be above 20-day MA (confirming uptrend)
        ma20      = np.mean(price_series[-20:])
        above_ma  = price_series[-1] > ma20

        # 10-day momentum with regime-aware threshold
        momentum_threshold = 1.5 if regime == 'Sideways' else 0.5
        momentum = (
            (price_series[-1] - price_series[-10]) / price_series[-10] * 100
            if len(price_series) >= 10 else 0
        )

        # In Sideways, require Hurst > 0.52 — stock must show trending DNA, not mean-reversion
        hurst_ok = True
        if regime == 'Sideways':
            hurst_ok = (hurst is not None and hurst > 0.52)

        fires = (
            current_cusum > regime_threshold and
            momentum > momentum_threshold and
            above_ma and
            persistent and
            hurst_ok
        )

        # ── Downside CUSUM for SHORT signal ──────────────────────────────────
        # Mirror of upside: S_neg[t] = max(0, S_neg[t-1] - (r[t]-mu)/sigma - k)
        cusum_neg = np.zeros(len(returns))
        for i in range(1, len(returns)):
            cusum_neg[i] = max(0.0, cusum_neg[i - 1] - (returns[i] - mu) / sigma - k)

        current_cusum_neg = cusum_neg[-1]

        persistent_short = (
            len(cusum_neg) >= 4 and
            cusum_neg[-1] > cusum_neg[-2] and
            cusum_neg[-2] > cusum_neg[-3] and
            cusum_neg[-3] > 0
        )
        below_ma      = price_series[-1] < ma20
        momentum_neg  = -momentum  # flip sign: large positive neg_momentum = stock falling hard

        short_fires = (
            current_cusum_neg > regime_threshold and
            momentum_neg > momentum_threshold and
            below_ma and
            persistent_short and
            hurst_ok
        )

        hurst_penalty    = 0.7 if (hurst and hurst < 0.5) else 1.0
        entry_score      = min(1.0, current_cusum     / (regime_threshold * 2.0)) * hurst_penalty if fires       else 0
        short_entry_score = min(1.0, current_cusum_neg / (regime_threshold * 2.0)) * hurst_penalty if short_fires else 0

        return {
            'rule_e_fires':        fires,
            'rule_e_short_fires':  short_fires,
            'cusum_value':         current_cusum,
            'momentum':            momentum,
            'hurst':               hurst if hurst else 0.5,
            'entry_score':         entry_score,
            'rule_e_short_score':  short_entry_score,
            'momentum_threshold':  momentum_threshold,
        }
    except Exception as e:
        logger.debug(f"Rule E error for {ticker}: {str(e)}")
        return {
            'rule_e_fires': False, 'rule_e_short_fires': False,
            'cusum_value': 0, 'momentum': 0, 'entry_score': 0,
            'rule_e_short_score': 0,
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AGGREGATOR: Combine all rules + ML scoring
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def run_layer3_rules_engine(layer2_candidates_path, garch_params_path, earnings_dna_path, output_path):
    logger.info("=" * 80)
    logger.info("  NSE DECISION ENGINE - LAYER 3 RULES ENGINE")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    logger.info("[STEP 1] Loading Layer 2 candidates ...")
    try:
        candidates = pd.read_parquet(layer2_candidates_path)
        logger.info(f"  Loaded {len(candidates)} candidates")
    except FileNotFoundError:
        logger.error(f"  ERROR: {layer2_candidates_path} not found")
        return None

    logger.info("[STEP 2] Loading ancillary data ...")
    garch_params_dict = {}
    try:
        garch_df = pd.read_parquet(garch_params_path)
        garch_df.columns = garch_df.columns.str.lower()
        garch_params_dict = dict(zip(garch_df['ticker'], garch_df.to_dict('records')))
        logger.info(f"  GARCH params: {len(garch_params_dict)} tickers")
    except:
        logger.warning("  Could not load GARCH params")

    # Load crisis_alpha and normalise to 0-1 scale
    crisis_alpha_dict = {}
    try:
        ca_path = earnings_dna_path.replace('earnings_dna.parquet', 'crisis_alpha.parquet')
        ca_df = pd.read_parquet(ca_path)
        # Columns: Ticker, CrisisReturnAvg (values ~0.001 range)
        # Normalise: rank-based percentile so scores spread across 0-1
        ca_df.columns = ca_df.columns.str.lower()
        ca_df = ca_df.rename(columns={'ticker': 'ticker', 'crisisreturnavg': 'crisis_raw'})
        ca_df['crisis_alpha'] = (ca_df['crisis_raw'].rank(pct=True)).clip(0, 1)
        crisis_alpha_dict = dict(zip(ca_df['ticker'], ca_df['crisis_alpha']))
        logger.info(f"  Crisis alpha: {len(crisis_alpha_dict)} tickers loaded and normalised")
    except Exception as e:
        logger.warning(f"  Could not load crisis_alpha: {e}")

    # Load FnO data (5 parquets from layer0_fno.py)
    fno_data = load_fno_data(earnings_dna_path)

    earnings_dict = {}
    try:
        earnings_df = pd.read_parquet(earnings_dna_path)
        earnings_dict = dict(zip(earnings_df['ticker'], earnings_df.to_dict('records')))
        logger.info(f"  Earnings DNA: {len(earnings_dict)} tickers")
    except:
        logger.warning("  Could not load earnings DNA")

    # Load sector momentum (Rule A replacement)
    sector_mom_map = {}
    try:
        sector_mom_path = earnings_dna_path.replace('earnings_dna.parquet', 'sector_momentum.parquet')
        sec_df = pd.read_parquet(sector_mom_path)
        sec_df = sec_df[sec_df['Sector'].str.strip() != '']
        for _, srow in sec_df.iterrows():
            sector_mom_map[srow['Sector'].strip()] = {
                'sector_mom_rank'   : int(srow['Rank']),
                'sector_mom_return' : float(srow['Rolling20dReturn']),
            }
        logger.info(f"  Sector momentum: {len(sector_mom_map)} sectors loaded")
    except Exception as e:
        logger.warning(f"  Could not load sector_momentum: {e}")

    spreads_path = earnings_dna_path.replace('earnings_dna.parquet','spread_history.parquet')
    ticker_pairs_lookup = {}
    try:
        spreads_df = pd.read_parquet(spreads_path)
        spreads_df.columns = spreads_df.columns.str.lower()
        acol = [c for c in spreads_df.columns if 'ticker_a' in c or c=='a'][0]
        bcol = [c for c in spreads_df.columns if 'ticker_b' in c or c=='b'][0]
        for _, r in spreads_df.iterrows():
            ta,tb = str(r[acol]).strip(), str(r[bcol]).strip()
            ticker_pairs_lookup.setdefault(ta,[]).append(ta+' / '+tb)
            ticker_pairs_lookup.setdefault(tb,[]).append(ta+' / '+tb)
        logger.info(f"  Spread history: {len(spreads_df)} pairs, {len(ticker_pairs_lookup)} tickers indexed")
    except Exception as e:
        logger.warning(f"  Could not load spread_history: {e}")

    logger.info("[STEP 3] Applying rules A/C/D/E ...")
    tickers = candidates['ticker'].tolist()
    logger.info(f'  Downloading prices for {len(tickers)} tickers ...')
    price_cache = batch_download_prices(tickers, period='1y')

    logger.info(f'[STEP 3b] Building SMC OHLC cache for {len(tickers)} tickers ...')
    ohlc_cache = build_ohlc_cache(tickers, period='1y')

    results = []
    rule_a_count = rule_c_count = rule_d_count = rule_e_count = smc_count = 0

    for idx, row in candidates.iterrows():
        ticker = row['ticker']
        price_series = price_cache.get(ticker, np.full(20, row['close']))

        rule_a = rule_a_volatility_breakout(ticker, price_series, garch_params_dict.get(ticker), row['close'], row.get('volume', 0))
        rule_a_count += rule_a.get('rule_a_fires', False)

        candidate_pairs = ticker_pairs_lookup.get(ticker, [])
        rule_c = rule_c_pairs_mr(ticker, price_series, candidate_pairs, row['close'], row.get('regime', 'Sideways'), row.get('pairs_eligible', False), price_cache)
        rule_c_count += rule_c.get('rule_c_fires', False)

        earnings_data = earnings_dict.get(ticker, {})
        rule_d = rule_d_event_driven(ticker, price_series, crisis_alpha_dict.get(ticker, 0), earnings_data.get('days_to_result', 999))
        rule_d_count += rule_d.get('rule_d_fires', False)

        rule_e = rule_e_cusum_momentum(ticker, price_series, row.get('regime', 'Sideways'), row.get('hurst', 0.5), threshold=5.0)
        rule_e_count += rule_e.get('rule_e_fires', False)

        # ── FnO boost (must be computed before short aggregation) ───────────
        fno_boost = rule_fno_boost(
            ticker,
            fno_data,
            earnings_data.get("days_to_result", 999)
        )

        # ── SHORT SIGNAL AGGREGATION ──────────────────────────
        short_a = rule_a.get('rule_a_short_fires', False)
        short_c = rule_c.get('rule_c_short_fires', False)
        short_e = rule_e.get('rule_e_short_fires', False)
        fno_ok  = fno_boost.get('fno_eligible', False)
        # Cash market: any stock can be shorted intraday
        intraday_short_ok = short_a or short_c or short_e
        # F&O overnight: only fno_eligible stocks
        fno_short_ok = intraday_short_ok and fno_ok
        short_score_total = float(np.mean([
            rule_a.get('rule_a_short_score', 0),
            rule_c.get('rule_c_short_score', 0),
            rule_e.get('rule_e_short_score', 0),
        ]))
        num_short_rules = int(sum([bool(short_a), bool(short_c), bool(short_e)]))
        # Earnings blackout: no shorts within 2 days of results
        days_to_res = earnings_data.get('days_to_result', 999)
        if days_to_res <= 2:
            intraday_short_ok = False
            fno_short_ok      = False
            short_score_total = 0.0

        # ── Rule S: SMC — multiplier signal ──────────────────────────────────
        smc_result = compute_smc_score(ticker, ohlc_cache.get(ticker))
        if smc_result.get('smc_rule_fires'):
            smc_count += 1

        _long_fires = (rule_a.get('rule_a_fires', False) or
                       rule_c.get('rule_c_fires', False) or
                       rule_e.get('rule_e_fires', False))

        # Pure short: no long rule fired — score using short entry scores so L4 doesn't kill it
        if not _long_fires and intraday_short_ok:
            _a_score = rule_a.get('rule_a_short_score', 0)
            _c_score = rule_c.get('rule_c_short_score', 0)
            _e_score = rule_e.get('rule_e_short_score', 0)
        else:
            _a_score = rule_a.get('entry_score', 0)
            _c_score = rule_c.get('entry_score', 0)
            _e_score = rule_e.get('entry_score', 0)

        final_score = aggregate_rules_and_score_v2({
            'tier'              : row.get('tier', 'T3'),
            'rule_a_entry_score': _a_score,
            'rule_c_entry_score': _c_score,
            'rule_d_entry_score': rule_d.get('entry_score', 0),
            'rule_e_entry_score': _e_score,
            'smc_conviction'    : smc_result.get('smc_conviction', 0.0),
            'smc_signal'        : smc_result.get('smc_signal', 0),
            'fno_oi_score'      : fno_boost.get('oi_score', 0.0),
            'fno_iv_score'      : fno_boost.get('iv_score', 0.0),
            'fno_iv_percentile' : fno_boost.get('iv_percentile', 50.0),
            'fno_pinning_risk'  : fno_boost.get('pinning_risk', False),
            'sector_mom_rank'   : row.get('sector_mom_rank', 7),
            'fno_eligible'      : fno_boost.get('fno_eligible', False),
        }, 0.5)

        # Realized 20-day volatility (decimal form) — used by Layer 4 for stop/target sizing
        # Replaces the 0.02 hardcoded default that was making all stops uniform at 4%
        try:
            _rets = np.diff(np.log(price_series[-21:])) * 100  # last 20 returns in %
            _realized_sigma = float(np.std(_rets)) / 100.0     # convert back to decimal
            _realized_sigma = float(np.clip(_realized_sigma, 0.005, 0.15))
        except Exception:
            _realized_sigma = 0.02

        results.append({
            'ticker': ticker,
            'companyname': row.get('companyname', ''),
            'tier': row.get('tier', ''),
            'sector': row.get('sector', ''),
            'sector_mom_rank'   : sector_mom_map.get(str(row.get('sector','')).strip(), {}).get('sector_mom_rank',   7),
            'sector_mom_return' : sector_mom_map.get(str(row.get('sector','')).strip(), {}).get('sector_mom_return', 0.0),
            'price_close': row['close'],
            'volume': row.get('volume', 0),
            'turnover_cr': row.get('turnover_cr', 0),
            'garch_sigma': _realized_sigma,
            'rule_a_fires': rule_a.get('rule_a_fires', False),
            'rule_a_vol_zscore': rule_a.get('vol_zscore', 0),
            'rule_a_entry_score': rule_a.get('entry_score', 0),
            'rule_a_stop_loss': rule_a.get('stop_loss', 0),
            'rule_a_target': rule_a.get('target', 0),
            'rule_c_fires': rule_c.get('rule_c_fires', False),
            'rule_c_pairs': rule_c.get('num_pairs', 0),
            'rule_c_spread_zscore': rule_c.get('spread_zscore', 0),
            'rule_c_entry_score': rule_c.get('entry_score', 0),
            'rule_d_fires': rule_d.get('rule_d_fires', False),
            'rule_d_crisis_alpha': rule_d.get('crisis_alpha', 0),
            'rule_d_proximity_score': rule_d.get('proximity_score', 0),
            'rule_d_days_to_result': rule_d.get('days_to_result', 999),
            'rule_d_entry_score': rule_d.get('entry_score', 0),
            'rule_e_fires': rule_e.get('rule_e_fires', False),
            'rule_e_cusum': rule_e.get('cusum_value', 0),
            'rule_e_momentum': rule_e.get('momentum', 0),
            'rule_e_entry_score': rule_e.get('entry_score', 0),
            'regime': row.get('regime', 'Sideways'),
            'vix': row.get('vix', 0),
            'r2_grade': row.get('r2_grade', 'D'),
            'hurst': row.get('hurst', 0.5),
            'pairs_eligible': row.get('pairs_eligible', False),
            'crisis_alpha': row.get('crisis_alpha', False),
            'num_rules_firing': int(sum([bool(rule_a.get('rule_a_fires', False)), bool(rule_c.get('rule_c_fires', False)), bool(rule_d.get('rule_d_fires', False)), bool(rule_e.get('rule_e_fires', False))])),
            'final_score': float(final_score) if final_score is not None else 0.0,
            'fno_oi_signal'      : fno_boost.get('oi_signal', 'neutral'),
            'fno_pcr'           : fno_boost.get('pcr', 0.0),
            'fno_iv_percentile' : fno_boost.get('iv_percentile', 50.0),
            'fno_pinning_risk'  : fno_boost.get('pinning_risk', False),
            'fno_rollover_pct'  : fno_boost.get('rollover_pct', 50.0),
            'fno_oi_score'      : fno_boost.get('oi_score', 0.0),
            'fno_iv_score'      : fno_boost.get('iv_score', 0.0),
            'fno_eligible'      : fno_boost.get('fno_eligible', False),
            'rule_a_short_fires': bool(short_a),
            'rule_a_short_score': float(rule_a.get('rule_a_short_score', 0)),
            'rule_c_short_fires': bool(short_c),
            'rule_c_short_score': float(rule_c.get('rule_c_short_score', 0)),
            'rule_e_short_fires': bool(short_e),
            'rule_e_short_score': float(rule_e.get('rule_e_short_score', 0)),
            'short_score': short_score_total,
            'num_short_rules': num_short_rules,
            'intraday_short_eligible': bool(intraday_short_ok),
            'fno_short_eligible': bool(fno_short_ok),
            # LONG takes priority when any long rule fires; SHORT only when no long rules fire but short rules do
            'direction': 'LONG' if (rule_a.get('rule_a_fires',False) or rule_c.get('rule_c_fires',False) or rule_e.get('rule_e_fires',False)) else 'SHORT' if intraday_short_ok else 'LONG',
            'ml_score_placeholder': 0.5,
            'smc_conviction':  smc_result.get('smc_conviction', 0.0),
            'smc_entry_score': smc_result.get('smc_entry_score', 0.0),
            'smc_signal':      smc_result.get('smc_signal', 0),
            'smc_bull':        smc_result.get('smc_bull', False),
            'smc_bear':        smc_result.get('smc_bear', False),
            'smc_wyckoff':     smc_result.get('wyckoff_phase', 'Undefined'),
            'smc_ers':         smc_result.get('ers', 0.0),
            'smc_rule_fires':  smc_result.get('smc_rule_fires', False),
            'run_date': datetime.now().strftime('%Y-%m-%d'),
        })

    logger.info(f"[STEP 4] Ranking {len(results)} signals ...")
    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values('final_score', ascending=False).reset_index(drop=True)
    output_df['rank'] = output_df.index + 1

    logger.info(f"[STEP 5] Saving output ...")
    output_df['rule_d_crisis_alpha'] = pd.to_numeric(output_df['rule_d_crisis_alpha'], errors='coerce').fillna(0.0).astype(float)
    for col in ['rule_a_stop_loss','rule_a_target','rule_d_crisis_alpha','rule_d_proximity_score']:
        output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0.0).astype(float)
    for col in ['rule_a_fires','rule_c_fires','rule_d_fires','rule_e_fires']:
        output_df[col] = output_df[col].map(lambda x: bool(x) if x is not None else False).astype(bool)
    for col in ['final_score','rule_a_vol_zscore','rule_a_entry_score','rule_c_spread_zscore','rule_c_entry_score','rule_d_entry_score','rule_e_cusum','rule_e_momentum','rule_e_entry_score']:
        output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0.0).astype(float)
    for col in ['smc_conviction', 'smc_ers', 'smc_entry_score']:
        if col in output_df.columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0.0).astype('float64')
    for col in ['smc_signal']:
        if col in output_df.columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0).astype('int32')
    for col in ['smc_bull', 'smc_bear', 'smc_rule_fires']:
        if col in output_df.columns:
            output_df[col] = output_df[col].map(lambda x: bool(x) if x is not None else False).astype(bool)

    output_df.to_parquet(output_path, index=False)
    logger.info(f"  Saved: {output_path}  ({len(output_df)} rows)")

    logger.info(f"  Rule A: {rule_a_count} | Rule C: {rule_c_count} | Rule D: {rule_d_count} | Rule E: {rule_e_count} | SMC: {smc_count}")
    return output_df


if __name__ == '__main__':
    LAYER2_PATH       = r'D:\MBA\STOCK MARKET RESEARCH\NSE quants py\nse_layer2_candidates.parquet'
    GARCH_PARAMS_PATH = r'D:\MBA\STOCK MARKET RESEARCH\NSE quants py\garch_params.parquet'
    EARNINGS_DNA_PATH = r'D:\MBA\STOCK MARKET RESEARCH\NSE quants py\earnings_dna.parquet'
    OUTPUT_PATH       = r'D:\MBA\STOCK MARKET RESEARCH\NSE quants py\nse_layer3_signals.parquet'

    signals_df = run_layer3_rules_engine(
        layer2_candidates_path=LAYER2_PATH,
        garch_params_path=GARCH_PARAMS_PATH,
        earnings_dna_path=EARNINGS_DNA_PATH,
        output_path=OUTPUT_PATH
    )
















