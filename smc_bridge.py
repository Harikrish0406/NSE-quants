"""
smc_bridge.py — SMC Pipeline Bridge for NSE Layer Stack
========================================================
Loads D:\\MBA\\STOCK MARKET RESEARCH\\price analytics\\1.py via importlib
(the filename starts with a digit so it cannot be imported normally).

Provides:
  build_ohlc_cache(tickers, period)  → {ticker: ohlc_df}  (full OHLC download)
  compute_smc_score(ticker, ohlc_df) → scoring dict used by Layer 3 & 4

Used by: layer3_rules_engine.py, layer4_portfolio.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Path to the SMC pipeline module ───────────────────────────────────────────
_SMC_PATH = r"D:\MBA\STOCK MARKET RESEARCH\price analytics\1.py"

# ── Module-level singleton so we only exec the module once ────────────────────
_smc_mod = None


def _get_smc():
    global _smc_mod
    if _smc_mod is None:
        if not os.path.exists(_SMC_PATH):
            log.error(f"SMC module not found: {_SMC_PATH}")
            return None
        try:
            spec = importlib.util.spec_from_file_location("smc_v3", _SMC_PATH)
            mod  = importlib.util.module_from_spec(spec)
            # Suppress the output_dir creation side-effect during import
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec.loader.exec_module(mod)
            _smc_mod = mod
            log.info("SMC module loaded from price analytics/1.py")
        except Exception as e:
            log.error(f"Failed to load SMC module: {e}")
            return None
    return _smc_mod


# ─────────────────────────────────────────────────────────────────────────────
# OHLC DOWNLOAD  (full Open/High/Low/Close/Volume for SMC computation)
# ─────────────────────────────────────────────────────────────────────────────

def build_ohlc_cache(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """
    Download full OHLC+Volume for all tickers in a single batch call.
    Returns {ticker_without_NS: ohlc_df}.
    Requires >= 60 bars to be included (SMC needs ~60 bars to initialise).
    """
    nse_tickers = [t + ".NS" if not t.endswith(".NS") else t for t in tickers]
    ticker_map  = {nt: t for t, nt in zip(tickers, nse_tickers)}
    ohlc_cache: dict[str, pd.DataFrame] = {}

    log.info(f"  SMC OHLC batch download: {len(nse_tickers)} tickers ...")
    try:
        raw = yf.download(
            nse_tickers,
            period=period,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        log.warning(f"  SMC OHLC batch download failed: {e}")
        raw = None

    if raw is not None and not raw.empty:
        for nt, t in ticker_map.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if nt not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[nt]
                else:
                    sub = raw
                needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub.columns]
                if len(needed) < 5:
                    continue
                sub = sub[needed].dropna()
                if len(sub) >= 60:
                    ohlc_cache[t] = sub
            except Exception:
                pass

    # Per-stock fallback for misses
    missed = [t for t in tickers if t not in ohlc_cache]
    if missed:
        log.info(f"  SMC OHLC fallback: fetching {len(missed)} missed tickers individually ...")
        for t in missed:
            try:
                h = yf.download(
                    t + ".NS", period=period, progress=False, auto_adjust=True
                )
                needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in h.columns]
                if len(needed) < 5:
                    continue
                h = h[needed].dropna()
                if len(h) >= 60:
                    ohlc_cache[t] = h
            except Exception:
                pass

    log.info(f"  SMC OHLC cache: {len(ohlc_cache)}/{len(tickers)} tickers")
    return ohlc_cache


# ─────────────────────────────────────────────────────────────────────────────
# SMC SCORE FOR ONE TICKER
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_SMC = {
    "smc_conviction":  0.0,
    "smc_signal":      0,
    "smc_bull":        False,
    "smc_bear":        False,
    "wyckoff_phase":   "Undefined",
    "ers":             0.0,
    "smc_rule_fires":  False,
    "smc_entry_score": 0.0,
}


def compute_smc_score(ticker: str, ohlc_df: pd.DataFrame | None) -> dict:
    """
    Run the full SMC pipeline on OHLC data and return a compact scoring dict.

    Fields returned:
      smc_conviction  float   Weighted conviction score (negative = bearish)
      smc_signal      int     1=LONG, -1=SHORT, 0=FLAT
      smc_bull        bool    smc_signal == 1
      smc_bear        bool    smc_signal == -1
      wyckoff_phase   str     Accumulation / Markup / Distribution / Markdown / ...
      ers             float   Equilibrium Rebalancing Score (0-10)
      smc_rule_fires  bool    |conviction| >= 70% of LONG_THRESHOLD

    Returns the empty dict if data is insufficient or the module fails to load.
    """
    if ohlc_df is None or len(ohlc_df) < 60:
        return _EMPTY_SMC.copy()

    smc = _get_smc()
    if smc is None:
        return _EMPTY_SMC.copy()

    try:
        df         = smc.compute_all_smc(ohlc_df.copy())
        conviction = float(smc.compute_conviction_score(df).iloc[-1])
        signal     = int(smc.simple_smc_signal(df).iloc[-1])
        wyckoff    = str(df["wyckoff_phase"].iloc[-1]) if "wyckoff_phase" in df.columns else "Undefined"
        ers        = float(df["ers"].iloc[-1])          if "ers"          in df.columns else 0.0

        threshold  = getattr(smc, "LONG_THRESHOLD", 3.5)
        fires      = abs(conviction) >= threshold * 0.70   # 70% of threshold

        # entry_score: conviction mapped to 0-1 for long signals only
        # negative conviction (bearish) → 0.0, so bearish SMC contributes nothing
        entry_score = float(np.clip(conviction / 10.0, 0.0, 1.0))

        return {
            "smc_conviction":  round(conviction, 3),
            "smc_signal":      signal,
            "smc_bull":        signal == 1,
            "smc_bear":        signal == -1,
            "wyckoff_phase":   wyckoff,
            "ers":             round(ers, 2),
            "smc_rule_fires":  bool(fires),
            "smc_entry_score": round(entry_score, 4),
        }

    except Exception as e:
        log.debug(f"SMC score error [{ticker}]: {e}")
        return _EMPTY_SMC.copy()
