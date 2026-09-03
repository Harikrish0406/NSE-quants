# NSE Quant Stack

Layered NSE equity selection system. Two independent codebases live here:

## 1. Production pipeline (top level)

Run by `layer5_orchestrator.py` / `layer6_dashboard.py`.

| File | Role |
|---|---|
| `layer0_nse_data.py` | NSE data pull (prices, corp actions, deals, announcements) |
| `layer0_fno.py` | F&O data pull (OI, PCR, IV, maxpain, rollover) |
| `layer1_heavy_compute.py` | Heavy compute (GARCH, clusters, crisis alpha, sector momentum) |
| `layer1_daily.py` | Daily lightweight compute |
| `layer1_5_validation.py` | Walk-forward signal validation |
| `layer2_daily_filter.py` | Daily universe filter + regime/VIX detection |
| `layer3_rules_engine.py` | Rules A/C/D/E + SMC decision layer |
| `layer3_fno_patch.py` | Live scoring formula (`aggregate_rules_and_score_v2`) + FnO terms |
| `layer4_portfolio.py` | Portfolio construction, TS-anchored stops/targets, sizing |
| `layer5_orchestrator.py` | Orchestrates L2 -> L3 -> L4 |
| `layer6_dashboard.py` | Dashboard + one-command pipeline runner |
| `TS layer testing.py` | Weekly technical-structure scan (S/R zones, entry/exit) |
| `f1_paper_tracker.py` | Paper-trade logger + exit updater |
| `smc_bridge.py` | Loader for the external SMC pipeline (`price analytics/1.py`) |

Supporting / one-off: `apply_option_c.py`, `patch_sectors.py`, `nse_monitor.py`,
`nse_interactive_walkthrough.py`, `layer 1.55.py`, `layer 3 .py`,
`layer6_dashboard_backup.py`.

### Run

```
python layer6_dashboard.py --all       # weekday (~12-15 min)
python layer6_dashboard.py --weekend   # full weekend run incl. TS scan
python f1_paper_tracker.py             # log + update + report
```

## 2. `claude quants/` — leak-free replica

Standalone rebuild focused on removing lookahead bugs and validating every signal
before it is weighted.

| File | Role |
|---|---|
| `validation_framework.py` | Leak-free walk-forward framework |
| `layer_swing_signals.py` | Swing signals (mean-reversion, momentum, vol breakout) |
| `layer_ml_ensemble.py` | XGBoost / LightGBM walk-forward ensemble |
| `layer_altdata_signals.py` | Alt-data signals (bulk/block deals, earnings drift, F&O positioning) |
| `layer_combiner.py` | Sharpe/IC-weighted signal combiner, FIRE/WAIT/KILL |
| `combiner_backtest.py` | Backtest of the combiner as an assembled strategy |
| `live_scoring.py` | Live signal scoring |
| `paper_tracker.py` | Independent paper-trade ledger, ATR-based stops |
| `refresh_cache.py` | Incremental yfinance price-cache refresh |
| `live_run.py` | Daily runner (refresh -> score -> combine -> track) |
| `orchestrator.py` / `run_all.py` | Sequential runners |
| `layer0_nse_data.py`, `layer0_fno.py`, `layer1_*.py` | Data layers copied from production (path constant changed) |

## Notes

- Source only. Data caches, parquet/csv outputs, logs and `__pycache__` are gitignored.
- The SMC pipeline (`smc_bridge.py` dependency) lives outside this repo at
  `price analytics/1.py`.
