# How the NSE Quant Stack works

A layered daily decision engine for NSE equities. It ingests market + F&O +
corporate data, runs structural/statistical signals, scores every candidate,
builds a position-sized portfolio of FIRE trades, and paper-tracks them.

One command runs the whole thing:

```
python layer6_dashboard.py --all      # auto-detects weekday vs weekend
```

---

## 1. The layer chain

```
Layer 0   data fetch        NSE prices/deals/actions/announcements + F&O chain
Layer 1   heavy compute     (weekend) universe rebuild, GARCH, Hurst, Granger
                            network, clusters, pairs, crisis alpha, earnings drift
Layer 1d  daily compute     (weekday) sector momentum, spread Z, CUSUM increment
Layer 1.5 validation        (weekend) 11-signal walk-forward IC/PnL check — informational
Layer 2   daily filter      regime detect -> liquidity / event / R2 / crisis gates -> rank
Layer 3   rules engine      Rules A/C/D/E + F&O boost + SMC multiplier -> final_score
Layer 4   portfolio         regime/risk weighting -> FIRE/WAIT/KILL -> sizing -> slots
TS layer  structure scan    S/R zones, entry/exit classification, stop/target levels
                            |
paper tracker               logs FIRE trades, updates vs live price, reports
layer 6   dashboard         runs the pipeline, builds HTML + CLI report, EOD log
```

**Weekday run** (`WEEKDAY_PIPELINE`, ~12–25 min):
`Layer0-NSE -> Layer0-FnO -> PatchSectors -> Layer1-Daily -> Layer2 -> Layer3 -> Layer4 -> Layer-TS`

**Weekend run** (`WEEKEND_PIPELINE`, ~60–90 min): same, but inserts
`Layer1-Heavy` after Layer0-FnO and `Layer1-Val` after Layer4.

---

## 2. Run commands

| Command | What it does |
|---|---|
| `python layer6_dashboard.py --all` | Full pipeline, auto weekday/weekend. Default with no flag. |
| `python layer6_dashboard.py --weekend` | Force the heavy weekend rebuild on any day. |
| `python layer6_dashboard.py --weekly` | Layer1-Val + Layer-TS only. |
| `python layer6_dashboard.py --dash` | Rebuild the HTML dashboard from the last parquet outputs, no pipeline run. |
| `python layer6_dashboard.py --monitor` | Live price monitor for open paper trades. |
| `python layer6_dashboard.py --select` | Interactive layer picker. |
| `python f1_paper_tracker.py daily` | Log new FIRE trades + update open ones + print the report. Also runs automatically at the end of `--all`. |

`--all` also shells out to `f1_paper_tracker.py log` then `update` at the end, so
the dashboard run is what actually drives paper tracking.

Set `NSE_FORCE_RICH=1` if the progress UI renders as scrolling lines (the terminal
misreports as non-TTY):

```
$env:NSE_FORCE_RICH=1; python layer6_dashboard.py --all
```

---

## 3. Layer by layer

### Layer 0 — data fetch

**`layer0_nse_data.py`** — uses the `nse` package (`curl_cffi` under the hood, to
get past NSE's TLS fingerprinting; raw `requests` reliably fails).
- Board meetings -> `earnings_calendar.parquet` / `earnings_dna.parquet` (soonest result per ticker, `days_to_result`).
- Bulk / block deals -> `bulk_deals.parquet` / `block_deals.parquet` (with `value_cr`).
- Corporate actions (div/split/bonus/rights) -> `corporate_actions.parquet`.
- 7-day announcements, regex-flagged (`is_result`, `is_order_win`, `is_adverse`, …) -> `announcements.parquet`.
- `build_score_summary()` — additive institutional score per ticker, clipped to [-1, 1]:
  `+0.30` block-buy >=₹5cr/10d, `+0.20` bulk-buy >=₹1cr/5d, `+0.10` corp action ex-date <=10d,
  `-0.30` adverse announcement, `+0.10` result in 3–5d, `-0.15` result in 0–2d (blackout),
  `-0.10` ex-div in 1–2d -> `score_summary.parquet`.
- Run after 4 PM IST. ~60–90 s.

**`layer0_fno.py`** — option-chain signals for ~200 F&O stocks.
- PCR (>1.2 bullish, <0.7 bearish), max-pain + `pinning_risk` (CMP within 2% of max-pain strike),
  net OI bias, ATM IV + 90-day percentile, rollover ratio (>70% strong).
- Outputs `fno_oi/pcr/iv/maxpain/rollover.parquet`. ~90–120 s.
- **Known gap:** `COMPUTE_IV_PERCENTILE = False` is hardcoded, so `iv_percentile` is
  always the neutral 50.0 in production — the IV band logic downstream never differentiates.

### Layer 1 — heavy compute (`layer1_heavy_compute.py`, weekend only, ~40 min)

Rebuilds the universe and every structural signal from scratch:
- **Universe** — NSE EQ list, full OHLCV via `yfinance`, >=3 yr history, liquidity tiers
  (T1 >=₹5cr, T2 >=₹1cr, T3 below) -> `universe_master.parquet`.
- **Fractal DNA** — Hurst exponent, sample entropy, DFA; type = MR / Trend / Random -> `fractal_dna.parquet`.
- **Granger network** — within-sector pairwise Granger causality, per-sector process pools,
  checkpointed/resumable; a pair survives only if F>=8.0 and p<=0.05 across the full window
  and both halves -> `stable_edges.parquet`.
- **Clusters** — Louvain communities on the Granger graph -> `clusters.parquet`.
- **R2 grades** — 5-fold time-series regression of next-day return on lagged z-score, MR tickers only -> `r2_grades.parquet`.
- **GARCH(1,1)** params per ticker -> `garch_params.parquet`.
- **CUSUM + crisis alpha** — full-history regime breaks + avg return during 2016/2018/2020 crisis windows -> `cusum_breaks.parquet`, `crisis_alpha.parquet`.
- **Tier beta** — T3 tickers vs their sector's T1 leader -> `tier_beta.parquet`.
- **Earnings drift** — historical earnings-like events (vol z >3, |ret| >3%), avg D1->D10 drift, reliability flag -> `earnings_dna_drift.parquet`.
- **Spreads** — cross-cluster pairs, ADF cointegration (p<=0.05), OU half-life 0–252 d -> `spread_history.parquet`.

### Layer 1 daily (`layer1_daily.py`, weekday, ~60 s)

Recomputes only what changes day to day — does **not** touch the weekend heavy outputs:
- 20-day rolling sector return -> `sector_momentum.parquet`.
- `Z_now` column refresh on the existing `spread_history.parquet`.
- CUSUM breaks for the last 5 trading days, merged into `cusum_breaks.parquet` (`CUSUM_THRESHOLD = 5.0`).

### Layer 1.5 validation (`layer1_5_validation.py`, weekend, ~70 min–2 h)

Walk-forward IC/PnL for 11 signals (Hurst, Granger, GARCH, CUSUM, earnings drift, pairs,
HAR-RV, XGBoost, LightGBM, z-score MR, momentum 12-1). Rolls 2010->2026.
PASS = `|IC t-stat| > 1.5 AND IC%positive > 55% AND PnL%positive > 55%`; WEAK if one holds; else FAIL.
Writes `layer1_5_v3_*` outputs. **Informational only** — it does not automatically re-weight
Layers 3/4. Use it to sanity-check which signals still have edge.

### Layer 2 — daily filter (`layer2_daily_filter.py`)

- **Regime** — India VIX via `^INDIAVIX` -> `nse` package -> hardcoded 18.0 fallback.
  `VIX_CRISIS = 22`, `VIX_BULL = 14`, else Sideways.
- **Gate 1 liquidity** — tier turnover thresholds (doubled for T2/T3 in Crisis), close >=₹25.
- **Gate 2 event blackout** — blocks tickers with results in 0–2 days unless `reliable_drift`.
- **Gate 3 R2 grade** — A/B always pass; C passes in Bull/Sideways only; falls back to top-N by
  turnover if fewer than 50 survive (grade kept as metadata, never a full bypass).
- **Crisis gate** — in Crisis regime only `crisis_alpha`-flagged tickers proceed.
- **Enrich** — merges Hurst/type, GARCH sigma, cluster ID, pairs eligibility, sector, slippage, regime stamp.
- **Rank** — `prelim_score` orders candidates into Layer 3 (it is not carried forward as a live scoring term).
- Output: `nse_layer2_candidates.parquet` + `layer2_report.txt`.

### Layer 3 — rules engine (`layer3_rules_engine.py` + `layer3_fno_patch.py`)

Four rules run per candidate, long and short:

| Rule | Signal | Fires when |
|---|---|---|
| **A** | GARCH volatility breakout | return z-score vs GARCH sigma exceeds 1.5σ |
| **C** | Pairs mean reversion | mean spread z across up to 3 pairs < -2 (long) / > 2 (short); disabled in Crisis |
| **D** | Event-driven | `alpha_score > 0.5 AND proximity_score >= 0.6 AND event_score > 0.4` (result 3–5 days out) |
| **E** | CUSUM momentum | regime-aware CUSUM threshold (Bull 4.5 / Sideways 7.0), >=3 rising days, price > 20-MA; Sideways also needs Hurst > 0.52 |

Plus:
- **F&O boost** — `oi_score` from PCR / net OI / rollover / pinning penalty; `iv_score` only
  within 7 days of earnings.
- **SMC multiplier** — Smart Money Concepts read from the external `price analytics/1.py`
  via `smc_bridge.py`. **Gated hard (2026-08-31):** no effect below conviction 6.0, then
  ramps to at most +15% (bullish) / -25% (bearish) at conviction 10. SMC can never create a FIRE on its own.

**The scoring formula** — `aggregate_rules_and_score_v2` in `layer3_fno_patch.py`:

```
pairs   = rule_c_entry_score
mom     = rule_e_entry_score           (× 0.6 if pinning_risk)
zscore  = max(pairs, mom)
tier_base = {T1: 0.05, T2: 0.03, T3: 0.01}

non-F&O:  base = 0.47·pairs + 0.28·mom + 0.20·zscore + tier_base
F&O:      base = 0.44·pairs + 0.20·mom + 0.13·zscore + 0.12·oi_score
                 + 0.06·clip(iv_score,-0.2,0.5) + tier_base

final_score = clip(base · smc_mult, 0, 1.5)
```

**Rule A and Rule D contribute zero weight to `final_score`** — they are computed and
can gate/cap things, but neither `rule_a_entry_score` nor `rule_d_entry_score` appears in
the sum. This is why every closed paper trade so far is Rule C or E. Known gap, not yet fixed.

Output: `nse_layer3_signals.parquet`, ranked by `final_score`.

### Layer 4 — portfolio construction (`layer4_portfolio.py`)

1. **Regime weight** — Crisis 0.30× / Sideways 1.00× / Bull 1.20× on `final_score`.
2. **SRISK dampener** — top-20 by GARCH sigma get 0.5× size.
3. **Crisis alpha dampener** — Layer 2 crisis-gate survivors get 0.8×.
4. **Sector bias** — top-3 sectors by 20d momentum ×1.10, bottom-3 ×0.90.
   `composite_score = weighted_score × srisk × crisis_alpha × sector_bias`, then an SMC
   confirm/conflict multiplier (bearish SMC penalises LONG, boosts SHORT).
5. **Timing verdict** — `act_now = (1 - Hurst)·100`; vol sanity check; hard KILL if bearish
   SMC (conv >=5) opposes a LONG, or if no rule fired at all.
   `FIRE_THRESHOLD = {Crisis: 0.55, Sideways: 0.28, Bull: 0.38}`.
6. **Position sizing** — stop/target priority:
   1. Weekly TS scan (`ts_analysis_report.csv`), keyed by `(ticker, direction)`, used only if
      `rr_valid` and the stored stop is within 8% of today's close — gives S/R-anchored stop
      + T1 (partial) / T2 (full).
   2. GARCH-sigma fallback, direction-aware, 2:1 R:R, regime-scaled ATR multiplier
      (`{Crisis: 1.0, Sideways: 1.5, Bull: 2.5}`).
   `qty = min(risk_capital / stop_distance, tier_cap / close)`, scaled by SRISK dampener.
   `TOTAL_CAPITAL = ₹1,000,000`, `MAX_RISK_PER_TRADE = 1%`.
7. **Portfolio build** — greedy walk of FIRE signals, `CLUSTER_CAP = 2` per Louvain community,
   `T3_MAX_SLOTS = 3`, `REGIME_SLOTS = {Crisis: 0, Sideways: 5, Bull: 10}`.

Output: `nse_layer4_portfolio.parquet` + `nse_layer4_portfolio_report.txt`.

### TS layer (`TS layer testing.py`)

Standalone technical-structure scan, now run every pipeline run. Layer 4 reads its output
for stop/target anchoring.
- 5-year daily + weekly OHLCV, full indicator set (ATR, RSI, MACD, Bollinger, Stoch, OBV, ADX, EMAs).
- S/R zones — pivots + KDE + zone clustering + polarity-flip detection.
- Trend — daily/weekly EMA alignment, ADX, HH/HL structure.
- Patterns — candlestick/chart patterns counted **only when they occur at a real S/R zone**.
- `classify_entry()` -> STRONG_BUY / BUY / WATCH / AVOID + entry type (BREAKOUT / RETEST / REVERSAL / TREND_FOLLOW).
- `stop_target()` / `multi_targets()` -> T1 / T2, gated by `RR_MIN = 1.5`.
- `score_setup()` -> 0–100 quality score.
- Outputs: `ts_analysis_report.csv`, `ts_sr_levels.parquet`, `ts_analysis_report.txt`. Full universe ~25–35 min.

### Paper tracker (`f1_paper_tracker.py`)

CLI: `log` / `update` / `report` / `daily`.
- `check_cooldown()` — no re-entry into a ticker for 3 days after a LOSS.
- `check_already_open()` — no duplicate open positions.
- `cmd_log()` — reads FIRE verdicts from `nse_layer4_portfolio.parquet`, applies both guards.
- `cmd_update()` — Day-0 checks SL only (no same-day win); direction-aware SL/target hit
  (SL-and-target same candle = LOSS); `MAX_HOLD` by regime `{Crisis: 5, Sideways: 10, Bull: 15}`.
- One CSV holds two trade universes: `pipeline_ver` tags each trade `v1` (pre-2026-06-03) or
  `v2` (TS-anchored stops + SHORT support). `paper_trades_v2.csv` / `.xlsx` is the clean v2-only view.
- `cmd_report()` — win rate, avg win/loss, R:R, expectancy, per-rule + per-regime breakdown,
  composite-score-vs-outcome IC, and a **live-trading readiness gate: >=30 closed trades AND
  win rate >=50% AND avg R:R >=1.5.**

### Layer 6 dashboard (`layer6_dashboard.py`)

The main entry point. Runs the pipeline (subprocess per layer, weighted `rich` progress bar
via `LAYER_TIME_ESTIMATES`), then builds:
- `nse_dashboard_{date}.html` — FIRE table, WAIT watchlist, Layer 3 top-10, rules-fired
  summary, SMC overview, sector momentum, open/closed paper trades, TS setups, 14-day history.
- A matching CLI report.
- `nse_layer5_eod_report_{date}.txt` + one row appended to `nse_run_log.csv`.

---

## 4. Data flow

```
Layer0-NSE  ┐
Layer0-FnO  ┤ (weekday: also PatchSectors, Layer1-Daily)
            ┘ (weekend: also Layer1-Heavy)
     │  garch_params, earnings_dna, crisis_alpha, sector_momentum,
     │  spread_history, clusters, fno_*, universe_master
     ▼
Layer2  regime + gates + rank  ->  nse_layer2_candidates.parquet
     ▼
Layer3  rules A/C/D/E + F&O + SMC  ->  nse_layer3_signals.parquet (ranked by final_score)
     ▼
Layer4  regime/risk weighting -> FIRE/WAIT/KILL -> sizing -> slots
     │      (reads yesterday's ts_analysis_report.csv for stops/targets)
     ▼
nse_layer4_portfolio.parquet
     ▼
f1_paper_tracker.py  log -> update -> report   (paper_trades.csv / _v2.csv)
     ▼
layer6_dashboard.py  HTML + CLI report + EOD report + run log

TS layer runs alongside each pipeline (writes ts_analysis_report.csv for the NEXT run's Layer 4).
```

---

## 5. Support scripts

| File | Purpose |
|---|---|
| `smc_bridge.py` | Loads the external SMC pipeline (`price analytics/1.py`, filename starts with a digit so it needs `importlib`). `compute_smc_score()` returns conviction / signal / Wyckoff phase — used by Layer 3 and Layer 4. |
| `patch_sectors.py` | Idempotent, cache-backed sector enrichment for `universe_master.parquet` via Yahoo `.info["sector"]`; also regenerates `sector_momentum.parquet`. Runs daily (cache makes it fast). |
| `nse_monitor.py` | Second-terminal live monitor for a Layer 1.5 run — tails the log, redraws IC/PnL every 15 s. |
| `apply_option_c.py` | One-off historical code patcher. Stale — its search patterns no longer match `layer1_heavy_compute.py`. Do not run. |
| `layer5_orchestrator.py` | Earlier standalone subprocess runner (Layer2->3->4 only). Superseded in practice by `layer6_dashboard.py`'s built-in orchestration, still runnable. |

**Superseded / backup files (kept, not run):** `layer 1.55.py` (older `layer1_5_validation.py`
snapshot — still referenced in the weekend list, so it currently runs alongside the current
file), `layer 3 .py` (pre-F&O/SMC rules engine — dead), `layer6_dashboard_backup.py`
(pre-console-redesign snapshot — dead).

---

## 6. Output files

| File | Written by | Contents |
|---|---|---|
| `universe_master.parquet` | Layer 1 Heavy | ticker list, tiers, sector |
| `fractal_dna.parquet` | Layer 1 Heavy | Hurst / entropy / DFA / type |
| `stable_edges.parquet` | Layer 1 Heavy | persistent Granger edges |
| `clusters.parquet` | Layer 1 Heavy | Louvain community per ticker |
| `garch_params.parquet` | Layer 1 Heavy | GARCH(1,1) params + sigma |
| `crisis_alpha.parquet` | Layer 1 Heavy | positive-crisis-return tickers |
| `spread_history.parquet` | Layer 1 Heavy / Daily | cointegrated pairs, half-life, `Z_now` |
| `sector_momentum.parquet` | Layer 1 Daily / patch_sectors | 20d sector return + rank |
| `cusum_breaks.parquet` | Layer 1 Heavy / Daily | regime-change breaks |
| `earnings_dna*.parquet` | Layer 0 / Layer 1 Heavy | upcoming results, historical drift |
| `fno_*.parquet` | Layer 0 FnO | OI / PCR / IV / maxpain / rollover |
| `score_summary.parquet` | Layer 0 NSE | institutional score per ticker |
| `nse_layer2_candidates.parquet` | Layer 2 | filtered + enriched + ranked candidates |
| `nse_layer3_signals.parquet` | Layer 3 | per-ticker rule scores + `final_score` |
| `nse_layer4_portfolio.parquet` | Layer 4 | FIRE/WAIT/KILL + entry/stop/target/qty |
| `ts_analysis_report.csv` / `ts_sr_levels.parquet` | TS layer | S/R zones, entry classification, stops |
| `paper_trades.csv` / `paper_trades_v2.csv` | paper tracker | trade ledger |
| `nse_dashboard_{date}.html` | Layer 6 | full visual dashboard |
| `nse_layer5_eod_report_{date}.txt` / `nse_run_log.csv` | Layer 6 | daily summary + run history |

---

## 7. Known gaps

1. **Rules A and D carry zero weight** in `final_score` (`layer3_fno_patch.py`). They fire but cannot move the score — every closed trade is Rule C/E.
2. **`COMPUTE_IV_PERCENTILE = False`** is hardcoded in `layer0_fno.py`, so `iv_percentile` is always 50.0 and the IV band logic never differentiates.
3. **`layer 1.55.py` still runs every weekend** alongside `layer1_5_validation.py` — two near-duplicate validation runs. Confirm whether that's intentional.
4. **Price-cache staleness** — caches only re-fetch tickers *missing* from local cache, never stale-but-present ones. Caught once (2-months stale). No automatic staleness check.
5. **IC ceiling** — historical IC ≈ 0.12, which bounds win rate. Target > 0.25. Next levers: filter FIRE to TS `STRONG_BUY`/`STRONG_SHORT`, reconsider Rule E in Sideways regime.
6. **T1 partial exits** not logged — `ts_t1` is written but the tracker is all-or-nothing (SL / T2 / MAX_HOLD).
7. **Layer 1.5 validation is not wired back** into Layer 3/4 weights — signal weights in the scoring formula are fixed by hand.

---

## 8. External dependency

`smc_bridge.py` loads `D:\MBA\STOCK MARKET RESEARCH\price analytics\1.py` — a standalone
Smart Money Concepts / Wyckoff / ICT pipeline (order blocks, FVG, liquidity sweeps,
premium/discount zones, market structure shift). It lives **outside this repo**. Only
`compute_all_smc` / `compute_conviction_score` / `simple_smc_signal` are used here
(`LONG_THRESHOLD = 3.5`, `SHORT_THRESHOLD = -3.5`).
