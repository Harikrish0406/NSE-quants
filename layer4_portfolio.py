"""
NSE DECISION ENGINE — LAYER 4: PORTFOLIO CONSTRUCTOR
======================================================
Reads  : nse_layer3_signals.parquet
         sector_momentum.parquet  (for sector bias)
         crisis_alpha.parquet     (for alpha list)
Writes : nse_layer4_portfolio.parquet
         nse_layer4_portfolio_report.txt

Blueprint (v2):
  Block 1 — Regime weight    : Crisis=0.3× | Sideways=1.0× | Bull=1.2×
  Block 2 — SRISK dampener   : top-20 high-SRISK stocks → 0.5× position
  Block 3 — Crisis alpha      : stocks that passed crisis gate → 0.8× weight
  Block 4 — Sector bias       : computed from sector_momentum.parquet (not hardcoded)

Layer 5 (position sizing + timing verdict) is appended here:
  P1 — Execution timing      : Fractal act-now score + GARCH vol window → FIRE/WAIT/KILL
  P2 — Position sizing       : tier-aware Kelly · ATR qty · CVaR dampener
  P3 — Portfolio construction: cluster cap (max 2/community) · T3 cap (max 3) · regime slots
        Crisis: 0 new slots | Bull: 10 slots | Sideways: 5 slots
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import logging
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE                = r'D:\MBA\STOCK MARKET RESEARCH\NSE quants py'
LAYER3_PATH         = os.path.join(BASE, 'nse_layer3_signals.parquet')
SECTOR_MOM_PATH     = os.path.join(BASE, 'sector_momentum.parquet')
CRISIS_ALPHA_PATH   = os.path.join(BASE, 'crisis_alpha.parquet')
OUTPUT_PATH         = os.path.join(BASE, 'nse_layer4_portfolio.parquet')
REPORT_PATH         = os.path.join(BASE, 'nse_layer4_portfolio_report.txt')
TS_PATH             = os.path.join(BASE, 'ts_analysis_report.csv')   # weekly TS scan

# Capital & risk parameters — edit these to suit your account
TOTAL_CAPITAL       = 1_000_000          # ₹ total deployed capital
MAX_RISK_PER_TRADE  = 0.01               # 1% capital risk per trade
T1_MAX_ALLOC        = 0.05               # 5% per T1 position
T2_MAX_ALLOC        = 0.03               # 3% per T2 position
T3_MAX_ALLOC        = 0.01               # 1% per T3 position
ATR_STOP_MULT       = 2.0                # default fallback only — overridden per regime below
SRISK_TOP_N         = 20                 # top-N by garch_sigma flagged as SRISK
CLUSTER_CAP         = 2                  # max signals per Louvain community
T3_MAX_SLOTS        = 3                  # absolute max T3 penny positions
REGIME_SLOTS        = {'Crisis': 0, 'Sideways': 5, 'Bull': 10}
REGIME_WEIGHT       = {'Crisis': 0.30, 'Sideways': 1.00, 'Bull': 1.20}
CRISIS_ALPHA_WEIGHT = 0.80               # extra dampener for crisis-passing stocks

# ── Regime-adaptive thresholds ──────────────────────────────────────────────
# Sideways: pairs MR dominates — lower FIRE bar, tighter stops, short hold
# Bull:     momentum dominates — higher FIRE bar, wider stops, longer hold
# Crisis:   defensive only — very high bar, very tight stops
FIRE_THRESHOLD    = {'Crisis': 0.55, 'Sideways': 0.28, 'Bull': 0.38}
WAIT_THRESHOLD    = {'Crisis': 0.40, 'Sideways': 0.18, 'Bull': 0.26}
ATR_STOP_BY_REGIME = {'Crisis': 1.0,  'Sideways': 1.5,  'Bull': 2.5}
MAX_HOLD_BY_REGIME = {'Crisis': 5,    'Sideways': 10,   'Bull': 15}

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name that exists (case-insensitive)."""
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_parquet(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        log.warning(f'{label} not found: {path}')
        return None
    df = pd.read_parquet(path)
    log.info(f'{label} loaded: {len(df)} rows | cols: {list(df.columns)}')
    return df


# ─────────────────────────────────────────────
# BLOCK 0 — LOAD DATA
# ─────────────────────────────────────────────

def load_inputs():
    signals = load_parquet(LAYER3_PATH, 'Layer3 signals')
    if signals is None or signals.empty:
        log.error('Layer 3 signals missing — aborting Layer 4.')
        return None, None, None

    sector_mom   = load_parquet(SECTOR_MOM_PATH,   'sector_momentum')
    crisis_alpha = load_parquet(CRISIS_ALPHA_PATH,  'crisis_alpha')
    return signals, sector_mom, crisis_alpha


# ─────────────────────────────────────────────
# BLOCK 1 — REGIME WEIGHT
# ─────────────────────────────────────────────

def apply_regime_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale signal confidence by regime.
    Uses the 'regime' column already present from Layer 2/3.
    """
    log.info('[BLOCK 1] Applying regime weight ...')
    regime_col = _col(df, 'regime')
    if regime_col is None:
        log.warning('  No regime column found — defaulting to Sideways weight 1.0')
        df['regime_weight'] = 1.0
        df['regime']        = 'Sideways'
    else:
        df['regime'] = df[regime_col].fillna('Sideways').str.capitalize()
        df['regime_weight'] = df['regime'].map(REGIME_WEIGHT).fillna(1.0)

    df['weighted_score'] = df['final_score'] * df['regime_weight']

    regime_counts = df['regime'].value_counts().to_dict()
    log.info(f'  Regime distribution: {regime_counts}')
    dominant_regime = df['regime'].mode().iloc[0]
    slots = REGIME_SLOTS.get(dominant_regime, 5)
    log.info(f'  Dominant regime: {dominant_regime} → {slots} portfolio slots available')
    return df, dominant_regime, slots


# ─────────────────────────────────────────────
# BLOCK 2 — SRISK DAMPENER
# ─────────────────────────────────────────────

def apply_srisk_dampener(df: pd.DataFrame) -> pd.DataFrame:
    """
    Top-SRISK_TOP_N stocks by GARCH σ (tail-risk proxy) → 0.5× position.
    """
    log.info('[BLOCK 2] Applying SRISK dampener ...')
    sigma_col = _col(df, 'garch_sigma', 'sigma', 'garch_vol', 'rule_a_garch_sigma')
    if sigma_col is None:
        log.warning('  No GARCH sigma column — skipping SRISK dampener')
        df['srisk_flag']    = False
        df['srisk_dampener'] = 1.0
        return df

    threshold = df[sigma_col].quantile(1 - SRISK_TOP_N / len(df))
    df['srisk_flag']    = df[sigma_col] >= threshold
    df['srisk_dampener'] = df['srisk_flag'].map({True: 0.5, False: 1.0})
    n_flagged = df['srisk_flag'].sum()
    log.info(f'  SRISK flagged: {n_flagged} stocks (σ ≥ {threshold:.4f}) → 0.5× position')
    return df


# ─────────────────────────────────────────────
# BLOCK 3 — CRISIS ALPHA
# ─────────────────────────────────────────────

def apply_crisis_alpha(df: pd.DataFrame, crisis_alpha_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Stocks that passed the crisis gate in Layer 2 get 0.8× weight here
    (they already survived the toughest gate; treat them as slightly riskier).
    """
    log.info('[BLOCK 3] Applying crisis alpha weight ...')

    # Try from the crisis_alpha parquet first
    crisis_tickers: set = set()
    if crisis_alpha_df is not None:
        ticker_col = _col(crisis_alpha_df, 'ticker', 'symbol')
        if ticker_col:
            crisis_tickers = set(crisis_alpha_df[ticker_col].str.upper().tolist())
            log.info(f'  Crisis alpha tickers loaded: {len(crisis_tickers)}')

    # Also use the boolean column already in signals if available
    bool_col = _col(df, 'crisis_alpha', 'crisis_alpha_flag')
    if bool_col and df[bool_col].dtype == bool:
        crisis_tickers |= set(df.loc[df[bool_col] == True, 'ticker'].str.upper().tolist())

    df['crisis_alpha_flag']   = df['ticker'].str.upper().isin(crisis_tickers)
    df['crisis_alpha_dampener'] = df['crisis_alpha_flag'].map({True: CRISIS_ALPHA_WEIGHT, False: 1.0})
    log.info(f'  Crisis alpha stocks in candidates: {df["crisis_alpha_flag"].sum()}')
    return df


# ─────────────────────────────────────────────
# BLOCK 4 — SECTOR BIAS
# ─────────────────────────────────────────────

def apply_sector_bias(df: pd.DataFrame, sector_mom_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Boost signals whose sector is in the rolling top-3 (20d momentum).
    Reduce signals in bottom-3 sectors.
    Sector bias computed from sector_momentum.parquet — not hardcoded.
    """
    log.info('[BLOCK 4] Applying sector bias ...')

    sector_col_sig = _col(df, 'sector')
    if sector_col_sig is None or sector_mom_df is None:
        log.warning('  Sector momentum not available — skipping sector bias')
        df['sector_bias'] = 1.0
        return df

    # Identify the rank column in sector_momentum (rolling return or rank)
    rank_col = _col(sector_mom_df, 'rank', 'return_20d', 'momentum', 'score', 'ret_20d')
    sec_col  = _col(sector_mom_df, 'sector', 'sectorname', 'name')

    if rank_col is None or sec_col is None:
        log.warning('  Cannot identify rank/sector columns in sector_momentum — skipping')
        df['sector_bias'] = 1.0
        return df

    ranked = sector_mom_df[[sec_col, rank_col]].copy()
    ranked = ranked.sort_values(rank_col, ascending=True).reset_index(drop=True)
    n = len(ranked)
    top3   = set(ranked.head(3)[sec_col].str.strip().tolist())
    bot3   = set(ranked.tail(3)[sec_col].str.strip().tolist())

    log.info(f'  Top sectors (boost +10%): {sorted(top3)}')
    log.info(f'  Bottom sectors (cut -10%): {sorted(bot3)}')

    def _bias(sector: str) -> float:
        s = str(sector).strip()
        if s in top3: return 1.10
        if s in bot3: return 0.90
        return 1.00

    df['sector_bias'] = df[sector_col_sig].apply(_bias)
    return df


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    df['composite_score'] = (
        df['weighted_score']
        * df['srisk_dampener']
        * df['crisis_alpha_dampener']
        * df['sector_bias']
    )

    # SMC conflict/confirmation: bearish SMC penalises LONG setups but confirms SHORT setups.
    if 'smc_conviction' in df.columns and 'smc_signal' in df.columns:
        smc_sig   = df['smc_signal'].fillna(0).astype(int)
        smc_conv  = df['smc_conviction'].fillna(0.0).abs()
        dir_col_c = _col(df, 'direction')
        is_short  = (df[dir_col_c].str.upper() == 'SHORT') if dir_col_c else pd.Series(False, index=df.index)

        # Scale penalty: -10% at conviction 3.5, -30% at conviction 10+
        bearish_penalty = (smc_conv.clip(3.5, 10.0) - 3.5) / 6.5 * 0.20 + 0.10
        smc_mult = pd.Series(1.0, index=df.index)
        # Penalise only LONG stocks that have bearish SMC
        long_bearish = (smc_sig == -1) & ~is_short
        smc_mult[long_bearish] = (1.0 - bearish_penalty[long_bearish]).clip(0.70, 1.0)
        # Boost SHORT stocks that have bearish SMC confirmation (+10% at conviction 3.5, +20% max)
        short_bearish_confirm = (smc_sig == -1) & is_short
        smc_boost = (smc_conv.clip(3.5, 10.0) - 3.5) / 6.5 * 0.10 + 0.10
        smc_mult[short_bearish_confirm] = (1.0 + smc_boost[short_bearish_confirm]).clip(1.0, 1.20)

        df['composite_score'] = (df['composite_score'] * smc_mult).clip(0, 2.0)
        df['smc_direction'] = smc_sig.map({1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'}).fillna('NEUTRAL')
    else:
        df['smc_direction'] = 'NEUTRAL'

    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['l4_rank'] = df.index + 1
    return df


# ─────────────────────────────────────────────
# LAYER 5-P1 — EXECUTION TIMING
# ─────────────────────────────────────────────

def compute_timing_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fractal act-now score (H + ApEn + FD → 0–100) × GARCH vol window.
    Score > 65 → FIRE | 40–65 → WAIT | < 40 → KILL
    Vol too high (> 3σ) or too low (< 0.5σ) → WAIT
    """
    log.info('[L5-P1] Computing execution timing verdict ...')

    hurst_col = _col(df, 'hurst', 'hurst_exp')
    sigma_col = _col(df, 'garch_sigma', 'sigma', 'garch_vol', 'rule_a_garch_sigma')
    # Fractal act-now score (0–100)
    if hurst_col:
        df['fractal_score'] = ((1 - df[hurst_col].clip(0, 1)) * 100).round(1)
    else:
        df['fractal_score'] = 50.0

    # GARCH vol window check
    if sigma_col:
        med_sigma = df[sigma_col].median()
        std_sigma = df[sigma_col].std()
        too_high  = df[sigma_col] > med_sigma + 3 * std_sigma
        too_low   = df[sigma_col] < 0.5 * med_sigma
        vol_ok    = ~(too_high | too_low)
    else:
        vol_ok = pd.Series(True, index=df.index)

    # Pre-compute regime for threshold lookup
    regime_col = _col(df, 'regime')
    dominant_regime_local = df[regime_col].mode().iloc[0] if regime_col and not df.empty else 'Sideways'
    fire_thr = FIRE_THRESHOLD.get(dominant_regime_local, 0.35)
    wait_thr = WAIT_THRESHOLD.get(dominant_regime_local, 0.25)

    # ── _verdict defined INSIDE the function so it closes over vol_ok ──
    def _verdict(row) -> str:
        if not vol_ok.loc[row.name]:
            return 'WAIT'
        smc_sig  = int(row.get('smc_signal', 0))
        smc_conv = float(row.get('smc_conviction', 0.0))
        fscore   = row['fractal_score']
        cscore   = row.get('composite_score', 0.0)

        direction = str(row.get('direction', 'LONG')).upper()
        # Hard kill: bearish SMC opposing a LONG setup — confirmation for SHORT, so skip kill there
        if smc_sig == -1 and abs(smc_conv) >= 5.0 and direction != 'SHORT':
            return 'KILL'

        # No rule fired at all — base score alone never earns FIRE
        rule_fired = (
            bool(row.get('rule_a_fires', False)) or
            bool(row.get('rule_c_fires', False)) or
            bool(row.get('rule_d_fires', False)) or
            bool(row.get('rule_e_fires', False)) or
            bool(row.get('intraday_short_eligible', False))
        )
        if not rule_fired:
            return 'KILL'

        if cscore >= fire_thr and fscore >= 40:
            return 'FIRE'
        elif cscore >= wait_thr or fscore >= 35:
            return 'WAIT'
        else:
            return 'KILL'

    df['timing_verdict'] = df.apply(_verdict, axis=1)

    # Session tag
    now_h = datetime.now().hour
    if 9 <= now_h < 12:
        session = 'MORNING'
    elif 12 <= now_h < 14:
        session = 'MIDDAY'
    elif 14 <= now_h < 16:
        session = 'CLOSING'
    else:
        session = 'POST_MARKET'
    df['session'] = session

    vc = df['timing_verdict'].value_counts().to_dict()
    log.info(f'  Timing verdicts: {vc}  | Session: {session}')
    return df   # ← was missing / unreachable in the broken version
# ─────────────────────────────────────────────
# LAYER 5-P2 — POSITION SIZING
# ─────────────────────────────────────────────

def compute_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier-aware Kelly fraction × ATR qty adjustment × CVaR dampener.

    Stop/target priority:
      1. TS layer (weekly scan): S/R-anchored stop + dual targets T1/T2 when rr_valid=True
      2. GARCH fallback: sigma × close × regime_multiplier, fixed 2:1 RR
    """
    log.info('[L5-P2] Computing position sizes ...')

    close_col = _col(df, 'price_close', 'close', 'ltp', 'price')
    sigma_col = _col(df, 'garch_sigma', 'sigma', 'garch_vol', 'rule_a_garch_sigma')
    dir_col   = _col(df, 'direction')

    tier_alloc_map = {'T1': T1_MAX_ALLOC, 'T2': T2_MAX_ALLOC, 'T3': T3_MAX_ALLOC}
    tier_col      = _col(df, 'tier')
    regime_col_ps = _col(df, 'regime')

    # ── Load weekly TS scan — keyed (ticker, direction) ──────────────────────
    ts_lookup: dict = {}
    if os.path.exists(TS_PATH):
        try:
            ts_df = pd.read_csv(TS_PATH)
            for _, tr in ts_df.iterrows():
                tk = str(tr.get('ticker', '')).upper()
                d  = str(tr.get('direction', 'LONG')).upper()
                if tk and bool(tr.get('rr_valid', False)):
                    ts_lookup[(tk, d)] = {
                        'stop':        float(tr['stop']),
                        't1':          float(tr['t1']),
                        't2':          float(tr['t2']),
                        'stop_source': str(tr.get('stop_source', 'ts')),
                        'setup_score': int(tr.get('setup_score', 0)),
                    }
            log.info(f'  TS overlay: {len(ts_lookup)} valid setups loaded from weekly scan')
        except Exception as e:
            log.warning(f'  TS load failed ({e}) — GARCH fallback for all')
    else:
        log.info('  ts_analysis_report.csv not found — run --weekend to generate TS scan')

    rows = []
    ts_used_count = 0

    for _, row in df.iterrows():
        ticker     = str(row.get('ticker', '')).upper()
        direction  = str(row.get(dir_col, 'LONG')).upper() if dir_col else 'LONG'
        if direction not in ('LONG', 'SHORT'):
            direction = 'LONG'
        tier       = row.get(tier_col, 'T1') if tier_col else 'T1'
        max_alloc  = tier_alloc_map.get(str(tier).upper(), T1_MAX_ALLOC)
        close      = float(row[close_col]) if close_col and pd.notna(row[close_col]) else 100.0
        sigma      = float(row[sigma_col]) if sigma_col and pd.notna(row[sigma_col]) else 0.02
        regime_row = str(row.get(regime_col_ps, 'Sideways')) if regime_col_ps else 'Sideways'

        if sigma > 1.0:
            sigma = sigma / 100.0
        sigma = max(0.005, min(sigma, 0.15))

        atr_mult  = ATR_STOP_BY_REGIME.get(regime_row, ATR_STOP_MULT)
        atr_proxy = sigma * close

        # ── Stop / target resolution ─────────────────────────────────────────
        ts_rec = ts_lookup.get((ticker, direction))
        ts_used = False
        ts_t1   = None

        if ts_rec:
            # Staleness guard: TS stop must be within 8% of today's close
            stop_pct = abs(close - ts_rec['stop']) / (close + 1e-9)
            if stop_pct <= 0.08 and ts_rec['stop'] > 0 and ts_rec['t2'] > 0:
                stop_price   = round(ts_rec['stop'], 2)
                target_price = round(ts_rec['t2'],   2)   # T2 = full target
                ts_t1        = round(ts_rec['t1'],   2)   # T1 = partial exit
                ts_stop_src  = ts_rec['stop_source']
                ts_score     = ts_rec['setup_score']
                ts_used      = True
                ts_used_count += 1
            else:
                ts_rec = None  # stale — fall through to GARCH

        if not ts_used:
            stop_distance = atr_mult * atr_proxy if atr_proxy > 0 else close * 0.02
            if direction == 'SHORT':
                stop_price   = round(close + stop_distance, 2)
                target_price = round(close - 2 * stop_distance, 2)
            else:
                stop_price   = round(close - stop_distance, 2)
                target_price = round(close + 2 * stop_distance, 2)
            ts_stop_src = 'garch_sigma'
            ts_score    = 0

        stop_distance = abs(close - stop_price)

        # ── Position sizing ───────────────────────────────────────────────────
        risk_capital   = TOTAL_CAPITAL * MAX_RISK_PER_TRADE
        tier_cap       = TOTAL_CAPITAL * max_alloc
        qty_risk       = int(risk_capital / stop_distance) if stop_distance > 0 else 1
        qty_tier       = int(tier_cap / close) if close > 0 else 1
        qty_raw        = max(1, min(qty_risk, qty_tier))
        srisk_d        = float(row.get('srisk_dampener', 1.0))
        qty            = max(1, int(qty_raw * srisk_d))
        position_value = round(qty * close, 2)

        rows.append({
            'entry_price':    close,
            'stop_price':     stop_price,
            'target_price':   target_price,
            'ts_t1':          ts_t1,
            'ts_used':        ts_used,
            'ts_stop_source': ts_stop_src,
            'ts_setup_score': ts_score,
            'atr_proxy':      round(atr_proxy, 4),
            'qty':            qty,
            'position_value': position_value,
            'risk_per_trade': round(qty * stop_distance, 2),
            'tier_alloc_pct': max_alloc,
        })

    sizing_df = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, sizing_df], axis=1)
    log.info(f'  Position sizing complete | TS-anchored: {ts_used_count}/{len(df)} '
             f'| Median qty: {df["qty"].median():.0f} '
             f'| Total exposure: ₹{df["position_value"].sum():,.0f}')
    return df


# ─────────────────────────────────────────────
# LAYER 5-P3 — PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────

def construct_portfolio(df: pd.DataFrame, dominant_regime: str, total_slots: int) -> pd.DataFrame:
    """
    Walk ranked signals and greedily fill the portfolio respecting:
      - FIRE verdict only
      - max 2 per Louvain community
      - max 3 T3 penny positions
      - total slots by regime
    """
    log.info(f'[L5-P3] Constructing portfolio | Regime={dominant_regime} | Slots={total_slots}')

    df['selected'] = False

    if total_slots == 0:
        log.info('  Crisis regime — 0 new slots. No positions selected.')
        return df

    community_col = _col(df, 'communityid', 'community_id', 'cluster', 'clusterid')
    tier_col      = _col(df, 'tier')

    community_counts: dict = {}
    t3_count    = 0
    selected    = 0

    for idx, row in df.iterrows():
        if selected >= total_slots:
            break
        if row.get('timing_verdict', 'WAIT') != 'FIRE':
            continue

        tier = str(row[tier_col]).upper() if tier_col else 'T1'
        comm = str(row[community_col]) if community_col else 'NONE'

        # T3 cap
        if tier == 'T3' and t3_count >= T3_MAX_SLOTS:
            continue

        # Cluster cap
        if community_counts.get(comm, 0) >= CLUSTER_CAP:
            continue

        df.at[idx, 'selected'] = True
        community_counts[comm] = community_counts.get(comm, 0) + 1
        if tier == 'T3':
            t3_count += 1
        selected += 1

    portfolio = df[df['selected']].copy()
    log.info(f'  Portfolio selected: {len(portfolio)} positions')
    if len(portfolio) > 0:
        log.info(f'  T3 positions: {(portfolio[tier_col] == "T3").sum() if tier_col else "?"}')
        log.info(f'  Total exposure: ₹{portfolio["position_value"].sum():,.0f}')
    return df


# ─────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────

def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [c for c in df.columns if df[c].dtype == object
                 and df[c].dropna().isin([True, False, 'True', 'False', 0, 1]).all()]
    for c in bool_cols:
        df[c] = df[c].map(lambda x: bool(x) if pd.notna(x) else False)
    return df


def build_report(df: pd.DataFrame, dominant_regime: str, slots: int) -> str:
    portfolio = df[df['selected']] if 'selected' in df.columns else df.head(0)
    watchlist = df[~df['selected']] if 'selected' in df.columns else df

    tier_col  = _col(df, 'tier')
    sec_col   = _col(df, 'sector')
    close_col = _col(df, 'entry_price', 'close')

    lines = []
    lines.append('=' * 80)
    lines.append('  NSE DECISION ENGINE — LAYER 4 PORTFOLIO REPORT')
    lines.append(f'  Date   : {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'  Regime : {dominant_regime}    Slots available: {slots}')
    lines.append(f'  Capital: ₹{TOTAL_CAPITAL:,}')
    lines.append('=' * 80)

    lines.append('')
    lines.append(f'  SELECTED PORTFOLIO ({len(portfolio)} positions)')
    lines.append('  ' + '─' * 78)
    if len(portfolio) > 0:
        for _, row in portfolio.iterrows():
            ticker  = row.get('ticker', '?')
            tier    = row.get(tier_col, '?') if tier_col else '?'
            sector  = str(row.get(sec_col, '?'))[:20] if sec_col else '?'
            entry   = row.get('entry_price', 0)
            stop    = row.get('stop_price', 0)
            target  = row.get('target_price', 0)
            qty     = row.get('qty', 0)
            score   = row.get('composite_score', 0)
            verdict = row.get('timing_verdict', '?')
            rules   = row.get('rule_tags', '?')
            lines.append(
                f'  {ticker:<14} {tier}  {sector:<22} '
                f'E:{entry:>8.2f}  SL:{stop:>8.2f}  T:{target:>8.2f}  '
                f'Qty:{qty:>5}  Score:{score:.4f}  [{rules}]  {verdict}'
            )
        lines.append('')
        lines.append(f'  Total exposure  : ₹{portfolio["position_value"].sum():>12,.0f}')
        lines.append(f'  Total risk      : ₹{portfolio["risk_per_trade"].sum():>12,.0f}')
    else:
        lines.append('  No positions selected (Crisis regime or no FIRE signals)')

    lines.append('')
    lines.append(f'  WATCHLIST — TOP 25 WAIT signals')
    lines.append('  ' + '─' * 78)
    wait_df = watchlist[watchlist.get('timing_verdict', pd.Series()) == 'WAIT'] \
        if 'timing_verdict' in watchlist.columns else watchlist
    for _, row in wait_df.head(25).iterrows():
        ticker  = row.get('ticker', '?')
        tier    = row.get(tier_col, '?') if tier_col else '?'
        score   = row.get('composite_score', 0)
        entry   = row.get('entry_price', 0)
        lines.append(f'  {ticker:<14} {tier}  ₹{entry:>8.2f}  Score:{score:.4f}')

    lines.append('')
    lines.append('  REGIME + CONTEXT SUMMARY')
    lines.append('  ' + '─' * 78)
    lines.append(f'  Dominant regime : {dominant_regime}')
    lines.append(f'  Regime weight   : {REGIME_WEIGHT.get(dominant_regime, 1.0)}×')
    lines.append(f'  SRISK flagged   : {df["srisk_flag"].sum() if "srisk_flag" in df.columns else "N/A"}')
    lines.append(f'  Crisis alpha    : {df["crisis_alpha_flag"].sum() if "crisis_alpha_flag" in df.columns else "N/A"}')

    if 'timing_verdict' in df.columns:
        vc = df['timing_verdict'].value_counts().to_dict()
        lines.append(f'  Timing verdicts : {vc}')

    if sec_col and 'selected' in df.columns:
        lines.append('')
        lines.append('  SECTOR BREAKDOWN (portfolio)')
        sec_counts = portfolio[sec_col].value_counts().head(10)
        for s, c in sec_counts.items():
            lines.append(f'    {s:<35}: {c}')

    if 'smc_signal' in df.columns:
        lines.append('')
        lines.append('  SMC SIGNAL DISTRIBUTION (all candidates)')
        lines.append('  ' + '─' * 78)
        smc_dist = df['smc_signal'].map({1: 'LONG', 0: 'NEUTRAL', -1: 'SHORT'}).value_counts()
        for label, cnt in smc_dist.items():
            lines.append(f'    {label:<10}: {cnt}')
        if len(portfolio) > 0 and 'smc_conviction' in portfolio.columns:
            lines.append(f'  Avg SMC conviction (portfolio): {portfolio["smc_conviction"].mean():.2f}')
            wyk_col = _col(portfolio, 'smc_wyckoff')
            if wyk_col:
                top_wyckoff = portfolio[wyk_col].value_counts().head(3).to_dict()
                lines.append(f'  Wyckoff phases (portfolio)    : {top_wyckoff}')

    lines.append('')
    lines.append('=' * 80)
    lines.append('  Output: nse_layer4_portfolio.parquet')
    lines.append('  NEXT  : Execute FIRE signals; log fills for feedback loop')
    lines.append('=' * 80)
    return '\n'.join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_layer4():
    log.info('')
    log.info('=' * 80)
    log.info('  NSE DECISION ENGINE — LAYER 4 PORTFOLIO CONSTRUCTOR')
    log.info(f'  {datetime.now()}')
    log.info('=' * 80)
    log.info('')

    # ── Load ──────────────────────────────────
    signals, sector_mom, crisis_alpha = load_inputs()
    if signals is None:
        return

    df = signals.copy()

    # ── Layer 4 enrichment ────────────────────
    log.info('')
    df, dominant_regime, total_slots = apply_regime_weight(df)
    log.info('')
    df = apply_srisk_dampener(df)
    log.info('')
    df = apply_crisis_alpha(df, crisis_alpha)
    log.info('')
    df = apply_sector_bias(df, sector_mom)
    log.info('')
    log.info('[COMPOSITE] Computing composite score ...')
    df = compute_composite_score(df)
    log.info(f'  Top composite score: {df["composite_score"].iloc[0]:.4f}')

    # ── Layer 5 ───────────────────────────────
    log.info('')
    df = compute_timing_verdict(df)
    log.info('')
    df = compute_position_sizing(df)
    log.info('')
    df = construct_portfolio(df, dominant_regime, total_slots)

    # ── Save ──────────────────────────────────
    log.info('')
    log.info('[SAVE] Enforcing dtypes and saving parquet ...')
    df = enforce_dtypes(df)
    df['run_date'] = datetime.now().strftime('%Y-%m-%d')
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f'  Saved: {OUTPUT_PATH}  ({len(df)} rows, {len(df.columns)} columns)')

    # ── Report ────────────────────────────────
    report = build_report(df, dominant_regime, total_slots)
    print('\n' + report)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    log.info(f'  Report saved: {REPORT_PATH}')
    log.info('')

    portfolio = df[df['selected']] if 'selected' in df.columns else df.head(0)
    log.info('=' * 80)
    log.info(f'  Layer 4 complete. {len(portfolio)} positions FIRE | '
             f'{(df.get("timing_verdict") == "WAIT").sum()} on WATCHLIST')
    log.info('=' * 80)


if __name__ == '__main__':
    run_layer4()
