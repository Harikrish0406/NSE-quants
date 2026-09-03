# NSE Quant Stack

Layered daily decision engine for NSE equities: data fetch -> structural/statistical
signals -> scoring -> position-sized portfolio -> paper tracking.

**Full explanation: [HOW_IT_WORKS.md](HOW_IT_WORKS.md)**

## Run

```
python layer6_dashboard.py --all        # full pipeline, auto weekday/weekend (~12-25 min / ~60-90 min)
python layer6_dashboard.py --weekend    # force the heavy weekend rebuild
python layer6_dashboard.py --dash       # rebuild dashboard from last outputs, no pipeline
python f1_paper_tracker.py daily        # log + update + report paper trades
```

If the progress bar renders as scrolling lines: `$env:NSE_FORCE_RICH=1` first.

## Files

| File | Role |
|---|---|
| `layer0_nse_data.py` | NSE data pull — earnings, deals, corp actions, announcements, institutional score |
| `layer0_fno.py` | F&O option-chain pull — OI, PCR, IV, maxpain, rollover |
| `layer1_heavy_compute.py` | Weekend rebuild — universe, GARCH, Hurst, Granger network, clusters, pairs, crisis alpha |
| `layer1_daily.py` | Weekday incremental — sector momentum, spread Z, CUSUM |
| `layer1_5_validation.py` | Walk-forward IC/PnL validation of 11 signals (informational) |
| `layer2_daily_filter.py` | Regime detect + liquidity / event / R2 / crisis gates + rank |
| `layer3_rules_engine.py` | Rules A/C/D/E + F&O boost + SMC |
| `layer3_fno_patch.py` | F&O boost signal + the live `aggregate_rules_and_score_v2` scoring formula |
| `layer4_portfolio.py` | Regime/risk weighting -> FIRE/WAIT/KILL -> stops/targets/sizing -> slots |
| `layer5_orchestrator.py` | Older standalone subprocess runner (superseded by layer6) |
| `layer6_dashboard.py` | Main entry point — runs the pipeline, builds HTML + CLI report |
| `TS layer testing.py` | Technical-structure scan — S/R zones, entry/exit, stop/target levels |
| `f1_paper_tracker.py` | Paper-trade ledger, exit updater, readiness-gate report |
| `smc_bridge.py` | Loader for the external SMC pipeline (`price analytics/1.py`) |
| `patch_sectors.py` | Cache-backed sector enrichment |
| `nse_monitor.py` | Live terminal monitor for Layer 1.5 runs |
| `apply_option_c.py` | Stale one-off code patcher — do not run |
| `layer 1.55.py`, `layer 3 .py`, `layer6_dashboard_backup.py` | Older snapshots, kept for reference |

## Notes

- Source only. Data caches, parquet/csv outputs, logs and `__pycache__` are gitignored.
- The SMC pipeline (`smc_bridge.py` dependency) lives outside this repo at `price analytics/1.py`.
