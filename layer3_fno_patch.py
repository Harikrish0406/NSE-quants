# ============================================================
# layer3_fno_patch.py  —  FnO Signal Integration for Layer 3
# Version: 1.0  |  2026-05-20
#
# INSTRUCTIONS — apply this patch to layer3_rules_engine.py:
#
# 1. Add import at top (after existing imports):
#       from layer3_fno_patch import load_fno_data, rule_fno_boost
#
# 2. In run_layer3_rules_engine(), after loading crisis_alpha (STEP 2),
#    add:
#       fno_data = load_fno_data(earnings_dna_path)
#
# 3. In the main ticker loop (after rule_e), add:
#       fno_boost = rule_fno_boost(ticker, fno_data, earnings_data.get('days_to_result', 999))
#
# 4. Pass fno_boost into aggregate_rules_and_score():
#       final_score = aggregate_rules_and_score({
#           ...existing keys...,
#           'fno_oi_score': fno_boost.get('oi_score', 0),
#           'fno_iv_score': fno_boost.get('iv_score', 0),
#       }, 0.5)
#
# 5. Replace aggregate_rules_and_score() with the updated version below.
#
# 6. Add fno fields to results.append({...}):
#       'fno_oi_signal'    : fno_boost.get('oi_signal', 'neutral'),
#       'fno_pcr'          : fno_boost.get('pcr', 0.0),
#       'fno_iv_percentile': fno_boost.get('iv_percentile', 50.0),
#       'fno_pinning_risk' : fno_boost.get('pinning_risk', False),
#       'fno_rollover_pct' : fno_boost.get('rollover_pct', 50.0),
#       'fno_oi_score'     : fno_boost.get('oi_score', 0.0),
#       'fno_iv_score'     : fno_boost.get('iv_score', 0.0),
# ============================================================

import pandas as pd
import numpy as np
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# LOAD ALL FNO PARQUETS
# ══════════════════════════════════════════════════════════════
def load_fno_data(earnings_dna_path: str) -> dict:
    """
    Loads all 5 FnO parquets into lookup dicts keyed by ticker.
    Safe — returns empty dicts if parquets are missing (first run).

    Call once at the top of run_layer3_rules_engine() and pass the
    result dict into rule_fno_boost() for each ticker.
    """
    base = Path(earnings_dna_path).parent

    fno_data = {
        "oi"      : {},
        "pcr"     : {},
        "iv"      : {},
        "maxpain" : {},
        "rollover": {},
    }

    file_map = {
        "oi"      : "fno_oi.parquet",
        "pcr"     : "fno_pcr.parquet",
        "iv"      : "fno_iv.parquet",
        "maxpain" : "fno_maxpain.parquet",
        "rollover": "fno_rollover.parquet",
    }

    for key, filename in file_map.items():
        path = base / filename
        try:
            df = pd.read_parquet(path)
            if "ticker" in df.columns:
                fno_data[key] = df.set_index("ticker").to_dict("index")
                log.info(f"  FnO {key}: {len(fno_data[key])} tickers loaded")
            else:
                log.warning(f"  FnO {key}: no 'ticker' column in {filename}")
        except FileNotFoundError:
            log.warning(f"  FnO {key}: {filename} not found — run layer0_fno.py first")
        except Exception as e:
            log.warning(f"  FnO {key}: load failed — {e}")

    return fno_data


# ══════════════════════════════════════════════════════════════
# FNO BOOST SIGNAL FUNCTION
# ══════════════════════════════════════════════════════════════
def rule_fno_boost(ticker: str, fno_data: dict, days_to_result: int = 999) -> dict:
    """
    Computes two FnO-derived score boosts for a ticker:

      oi_score  (0-1) → Rule E boost via OI momentum
      iv_score  (0-1) → Rule D quality gate via IV percentile

    ── OI Score (Rule E boost) ────────────────────────────────
    Inputs: PCR, net_oi_bias, rollover_pct, pinning_risk

    Logic:
      PCR > 1.2           → +0.3 (put-heavy = bullish sentiment)
      PCR 0.7-1.2         → +0.0 (neutral)
      PCR < 0.7           → -0.2 (call-heavy = bearish pressure)
      net_oi_bias > 0.2   → +0.2 (put support dominates)
      net_oi_bias < -0.2  → -0.1 (call wall resistance)
      rollover > 70%      → +0.2 (strong carry-forward)
      rollover 40-70%     → +0.1 (moderate carry)
      pinning_risk=True   → -0.3 (price likely to stall at max pain)

    Final oi_score clipped to [0, 1].

    ── IV Score (Rule D quality gate) ────────────────────────
    Inputs: iv_percentile, days_to_result

    Logic (pre-earnings IV):
      iv_percentile 50-80  → ideal pre-earnings window → +0.2 boost
      iv_percentile > 80   → vol already priced in → cap at 0.6 (caution)
      iv_percentile < 30   → low vol, no event pricing → neutral (0.0)
      No earnings within 7 days → iv_score = 0 (IV signal only for Rule D context)

    Final iv_score clipped to [0, 1].

    Returns dict with all raw signals + computed scores.
    Gracefully returns neutral scores if FnO data missing for ticker.
    """

    # ── Defaults (neutral, no bias) ──────────────────────
    result = {
        "pcr"           : 1.0,
        "pcr_sentiment" : "neutral",
        "net_oi_bias"   : 0.0,
        "oi_signal"     : "neutral",
        "atm_iv"        : 0.0,
        "iv_percentile" : 50.0,
        "iv_label"      : "moderate",
        "pinning_risk"  : False,
        "max_pain_distance_pct": 0.0,
        "rollover_pct"  : 50.0,
        "rollover_signal": "moderate",
        "oi_score"      : 0.0,
        "iv_score"      : 0.0,
        "fno_eligible"  : False,
    }

    # Check if ticker has any FnO data at all
    has_fno = any(ticker in fno_data.get(k, {}) for k in ["pcr", "oi", "iv"])
    if not has_fno:
        return result   # not an F&O stock — neutral, won't affect score

    result["fno_eligible"] = True

    # ── Pull raw data ─────────────────────────────────────
    pcr_row      = fno_data.get("pcr", {}).get(ticker, {})
    oi_row       = fno_data.get("oi", {}).get(ticker, {})
    iv_row       = fno_data.get("iv", {}).get(ticker, {})
    mp_row       = fno_data.get("maxpain", {}).get(ticker, {})
    rv_row       = fno_data.get("rollover", {}).get(ticker, {})

    pcr            = float(pcr_row.get("pcr", 1.0))
    pcr_sentiment  = str(pcr_row.get("pcr_sentiment", "neutral"))
    net_oi_bias    = float(oi_row.get("net_oi_bias", 0.0))
    oi_signal      = str(oi_row.get("oi_signal", "neutral"))
    atm_iv         = float(iv_row.get("atm_iv", 0.0))
    iv_percentile  = float(iv_row.get("iv_percentile", 50.0))
    iv_label       = str(iv_row.get("iv_label", "moderate"))
    pinning_risk   = bool(mp_row.get("pinning_risk", False))
    mp_dist        = float(mp_row.get("max_pain_distance_pct", 0.0))
    rollover_pct   = float(rv_row.get("rollover_pct", 50.0))
    rollover_signal= str(rv_row.get("rollover_signal", "moderate"))

    result.update({
        "pcr"                  : pcr,
        "pcr_sentiment"        : pcr_sentiment,
        "net_oi_bias"          : net_oi_bias,
        "oi_signal"            : oi_signal,
        "atm_iv"               : atm_iv,
        "iv_percentile"        : iv_percentile,
        "iv_label"             : iv_label,
        "pinning_risk"         : pinning_risk,
        "max_pain_distance_pct": mp_dist,
        "rollover_pct"         : rollover_pct,
        "rollover_signal"      : rollover_signal,
    })

    # ══════════════════════════════════════════════════════
    # OI SCORE — Rule E momentum quality gate
    # ══════════════════════════════════════════════════════
    oi_score = 0.5   # neutral baseline for F&O eligible stocks

    # PCR component
    if pcr > 1.2:
        oi_score += 0.3     # put-heavy → hedging → resilient underlying
    elif pcr < 0.7:
        oi_score -= 0.2     # call-heavy → resistance → momentum at risk

    # Net OI bias component
    if net_oi_bias > 0.2:
        oi_score += 0.2     # put support dominates → floor strong
    elif net_oi_bias < -0.2:
        oi_score -= 0.1     # call wall dominant → cap on upside

    # Rollover component
    if rollover_pct > 70:
        oi_score += 0.2     # strong carry-forward → trend continuation
    elif rollover_pct >= 40:
        oi_score += 0.1     # moderate rollover → trend intact

    # Pinning risk penalty
    if pinning_risk:
        oi_score -= 0.3     # price expected to stay near max pain → no directional move

    oi_score = float(np.clip(oi_score, 0.0, 1.0))

    # ══════════════════════════════════════════════════════
    # IV SCORE — Rule D pre-earnings quality gate
    # Only relevant when earnings are within 7 days
    # ══════════════════════════════════════════════════════
    iv_score = 0.0   # default: IV signal doesn't affect non-event stocks

    if days_to_result is not None and days_to_result != 999 and days_to_result <= 7:
        # Pre-earnings window: IV percentile tells us if vol is being priced in
        if 50 <= iv_percentile <= 80:
            # Sweet spot: IV elevated but not extreme
            # Options are pricing the event = stock likely to move
            iv_score = 0.2 + (iv_percentile - 50) / 150   # 0.2 → 0.4 range

        elif iv_percentile > 80:
            # Vol already fully priced in — entry risky, IV crush likely
            # Cap the rule_d entry_score downstream
            iv_score = -0.1   # slight penalty (caution flag)

        elif iv_percentile < 30:
            # Low IV = market not pricing any event = weaker signal
            iv_score = 0.0

        else:
            # 30-50: mild elevation, slight boost
            iv_score = 0.1

    iv_score = float(np.clip(iv_score, -0.2, 0.5))

    result["oi_score"] = oi_score
    result["iv_score"] = iv_score

    return result


# ══════════════════════════════════════════════════════════════
# UPDATED aggregate_rules_and_score()
# Replace the existing function in layer3_rules_engine.py
# with this version.
# ══════════════════════════════════════════════════════════════
def aggregate_rules_and_score_v2(row, ml_score=0.5):
    """
    Scoring formula — Rules A / C / D / E + FnO.
    SMC is a MULTIPLIER on the quant-rules base score, not an additive term.

    Multiplier ranges (applied after base score is computed) — GATED HARD
    2026-08-31 because SMC is an unvalidated structure overlay that fired on
    ~40-50% of candidates at the old ±3.5 bar:
      conviction below SMC_GATE (6.0): ×1.00           (no effect at all)
      Bullish SMC, conviction ≥ gate : ×1.00 → ×1.15  (ramps from 0 at gate to +15% at conv 10)
      Bearish SMC, conviction ≥ gate : ×1.00 → ×0.75  (ramps from 0 at gate to -25% at conv 10)

    This guarantees SMC alone cannot trigger a FIRE — it only amplifies or
    suppresses signals already produced by the quant rules, and now only when
    the SMC read is genuinely strong.

    Non-F&O base weights (sum ≈ 0.95 + tier_base):
      0.42 × Rule C  (pairs MR, IC=0.737)
      0.28 × Rule E  (CUSUM momentum)
      0.20 × zscore  (MR proxy — max of C/E)
      0.05 × sector  (20d sector momentum rank)
      + tier_base

    F&O base weights:
      0.38 × Rule C
      0.20 × Rule E
      0.13 × zscore
      0.12 × OI score
      0.06 × IV score
      0.06 × sector
      + tier_base
    """
    rule_c_score = float(row.get("rule_c_entry_score", 0))
    rule_e_score = float(row.get("rule_e_entry_score", 0))
    rule_d_score = float(row.get("rule_d_entry_score", 0))
    oi_score     = float(row.get("fno_oi_score",       0))
    iv_score     = float(row.get("fno_iv_score",       0))
    iv_pct       = float(row.get("fno_iv_percentile",  50.0))
    pinning      = bool(row.get("fno_pinning_risk",    False))

    # Rule D cap when vol already priced in
    if iv_pct > 80:
        rule_d_score = min(rule_d_score, 0.65)

    # Pinning risk: dampen Rule E momentum when near max pain
    if pinning:
        rule_e_score *= 0.6

    pairs_score    = rule_c_score
    momentum_score = rule_e_score
    zscore_score   = max(rule_c_score, rule_e_score)

    # Minimal floor — tier gives tiny quality nudge, not a free FIRE pass
    # Reduced from 0.10/0.08/0.04: stocks now need real signal to reach 0.35 FIRE threshold
    tier_base = {"T1": 0.05, "T2": 0.03, "T3": 0.01}
    base = tier_base.get(str(row.get("tier", "T3")), 0.01)

    iv_contribution = float(np.clip(iv_score, -0.2, 0.5))

    sector_rank  = float(row.get('sector_mom_rank', 7))
    sector_score = float(np.clip(1.0 - (sector_rank - 1) / 11.0, 0.0, 1.0))

    fno_eligible = bool(row.get("fno_eligible", False))

    # ── Base score from quant rules only (SMC excluded from additive part) ──
    # sector_score removed from both formulas — Layer 4 already applies ±10% sector_bias
    # on composite_score, so including it here was double-counting.
    # Freed weight redistributed to pairs_score (highest IC=0.737).
    if fno_eligible:
        base_score = (
            0.44 * pairs_score     +
            0.20 * momentum_score  +
            0.13 * zscore_score    +
            0.12 * oi_score        +
            0.06 * iv_contribution +
            float(base)
        )
    else:
        base_score = (
            0.47 * pairs_score     +
            0.28 * momentum_score  +
            0.20 * zscore_score    +
            float(base)
        )

    # ── SMC as multiplier: amplify or suppress the quant signal ─────────────
    # Gated HARD (2026-08-31): SMC is an unvalidated structure overlay (not in
    # the layer1_5 walk-forward report) that fired on ~40-50% of candidates at
    # the old ±3.5 signal bar and swung the score ±30-40%. Now it only moves
    # anything once conviction clears SMC_GATE, ramps from 0 at the gate to
    # full at conviction 10, and the magnitude is cut. SMC still can't FIRE
    # a stock on its own — it only scales a base score the quant rules built.
    SMC_GATE        = 6.0     # |conviction| below this → multiplier is exactly 1.0
    SMC_BOOST_MAX   = 0.15    # was 0.30
    SMC_PENALTY_MAX = 0.25    # was 0.40

    conviction = float(row.get("smc_conviction", 0.0))
    smc_sig    = int(row.get("smc_signal",       0))
    abs_conv   = min(abs(conviction), 10.0)

    if smc_sig != 0 and abs_conv >= SMC_GATE:
        strength = (abs_conv - SMC_GATE) / (10.0 - SMC_GATE)   # 0 at gate → 1 at conv 10
        if smc_sig == 1:
            smc_mult = 1.0 + strength * SMC_BOOST_MAX
        else:  # smc_sig == -1
            smc_mult = 1.0 - strength * SMC_PENALTY_MAX
    else:
        smc_mult = 1.0  # gated out — SMC read not strong enough to move the score

    final_score = base_score * smc_mult
    return float(np.clip(final_score, 0.0, 1.5))

# ══════════════════════════════════════════════════════════════
# SELF-TEST (run this file directly to verify)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("layer3_fno_patch.py — self test")
    print("=" * 60)

    # Simulate a bullish FnO scenario
    mock_fno_data = {
        "pcr"     : {"TESTSTOCK": {"pcr": 1.35, "pcr_sentiment": "bullish", "total_ce_oi": 1000, "total_pe_oi": 1350}},
        "oi"      : {"TESTSTOCK": {"net_oi_bias": 0.25, "oi_signal": "bullish", "call_oi_above_cmp": 800, "put_oi_below_cmp": 1200, "atm_ce_oi": 100, "atm_pe_oi": 120}},
        "iv"      : {"TESTSTOCK": {"atm_iv": 32.5, "iv_percentile": 65.0, "iv_label": "elevated"}},
        "maxpain" : {"TESTSTOCK": {"max_pain_strike": 1020.0, "cmp": 1050.0, "max_pain_distance_pct": 2.86, "pinning_risk": False}},
        "rollover": {"TESTSTOCK": {"rollover_pct": 74.0, "rollover_signal": "strong", "current_oi": 50000, "prev_oi": 18000}},
    }

    # Test 1: pre-earnings window (5 days out)
    boost = rule_fno_boost("TESTSTOCK", mock_fno_data, days_to_result=5)
    print(f"\nTest 1 — TESTSTOCK (5d to earnings):")
    print(f"  PCR={boost['pcr']} ({boost['pcr_sentiment']})")
    print(f"  OI signal={boost['oi_signal']}  net_bias={boost['net_oi_bias']}")
    print(f"  IV={boost['atm_iv']}%  percentile={boost['iv_percentile']}")
    print(f"  Rollover={boost['rollover_pct']}% ({boost['rollover_signal']})")
    print(f"  Pinning={boost['pinning_risk']}")
    print(f"  ➤ OI score : {boost['oi_score']:.3f}  (expected ~0.80)")
    print(f"  ➤ IV score : {boost['iv_score']:.3f}  (expected ~0.23)")

    # Test 2: no earnings
    boost2 = rule_fno_boost("TESTSTOCK", mock_fno_data, days_to_result=999)
    print(f"\nTest 2 — TESTSTOCK (no earnings):")
    print(f"  ➤ OI score : {boost2['oi_score']:.3f}  (should be same as Test 1)")
    print(f"  ➤ IV score : {boost2['iv_score']:.3f}  (expected 0.0 — no event)")

    # Test 3: pinning risk + high IV
    mock_fno_data["maxpain"]["TESTSTOCK"]["pinning_risk"] = True
    mock_fno_data["iv"]["TESTSTOCK"]["iv_percentile"] = 88.0
    boost3 = rule_fno_boost("TESTSTOCK", mock_fno_data, days_to_result=4)
    print(f"\nTest 3 — TESTSTOCK (pinning + high IV):")
    print(f"  ➤ OI score : {boost3['oi_score']:.3f}  (should be lower due to pinning)")
    print(f"  ➤ IV score : {boost3['iv_score']:.3f}  (should be negative — caution)")

    # Test 4: aggregate scoring
    test_row = {
        "tier"                : "T1",
        "rule_c_entry_score"  : 0.70,
        "rule_d_entry_score"  : 0.65,
        "rule_e_entry_score"  : 0.55,
        "fno_oi_score"        : boost["oi_score"],
        "fno_iv_score"        : boost["iv_score"],
        "fno_iv_percentile"   : 65.0,
        "fno_pinning_risk"    : False,
    }
    score = aggregate_rules_and_score_v2(test_row)
    print(f"\nTest 4 — aggregate_rules_and_score_v2:")
    print(f"  Inputs: C={test_row['rule_c_entry_score']} E={test_row['rule_e_entry_score']} OI={test_row['fno_oi_score']:.3f} IV={test_row['fno_iv_score']:.3f}")
    print(f"  ➤ Final score: {score:.4f}  (should be > 0.35 FIRE threshold)")

    print("\n✅ Self-test complete.")