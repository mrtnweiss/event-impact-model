# Event Impact Model — Research Memo

## Summary
This repo is a small, reproducible event-to-signal pipeline built around SEC filing timestamps (8-K / 10-Q / 10-K). The focus is on correct event alignment (timezone + NYSE sessions), leakage-aware modeling, and a minimal but realistic costed backtest. Outputs are deterministic artifacts and a generated report.

## Hypothesis
SEC filing events contain information that can be mapped to a tradable timestamp and can help predict short-horizon post-event returns.

## Data
- Universe: SEC company tickers snapshot (config-driven, size-limited for the MVP)
- Events: EDGAR submissions metadata (`acceptanceDateTime`, UTC)
- Prices: Stooq daily OHLCV (benchmark: SPY)

## Event timestamping and trade-date mapping
- Source timestamp: `acceptanceDateTime` (UTC) → converted to US/Eastern.
- Session bucket:
  - premarket (< 09:30 ET)
  - intraday (09:30–16:00 ET)
  - afterhours (>= 16:00 ET)
- Effective trade date (daily MVP, conservative):
  - premarket → same trading day open
  - intraday/afterhours → next trading day open
- NYSE calendar logic handles weekends/holidays and DST edge cases.

## Event study
- Returns: close-to-close daily returns.
- Benchmark: market model vs SPY (OLS per ticker).
- Estimation window: [-120, -21] trading days.
- Event window: [-10, +5] trading days.
- Outputs:
  - AR by tau
  - CAR / CAAR
  - Pre-trend check: CAR[-10, -2] should be close to 0
  - Subgroup tests (form, session bucket) with BH-FDR (q-values)

## Predictive modeling
- Label (MVP): CAR[+1, +5].
- Features (strictly pre-event):
  - short/medium momentum (1/5/20d)
  - volatility (20d)
  - liquidity proxy (20d dollar volume)
  - calendar features (day-of-week, month)
  - event metadata (form, session bucket)
- Models:
  - baselines: Ridge (regression), Logistic (sign)
  - main: LightGBM regression
- CV: walk-forward with purge/embargo-style guardrails.

## Backtest and execution assumptions
- Signal: cross-sectional ranking of OOS predictions per formation date.
- Portfolio: long top-q / short bottom-q.
- Execution: configurable delay (default next trading day), fixed holding horizon.
- Costs: proportional to turnover (bps).
- Constraints: max active names, per-name cap, gross target; optional portfolio vol targeting.
- Primary goal: a mechanically correct backtest, not a production execution simulator.

## Robustness checks
- Parameter sensitivity grids (diagnostic only; no “best row” selection).
- Sanity checks:
  - label shuffles / within-day shuffles (should degrade signal)
  - subsample splits (stability across regimes / buckets)

## Main artifacts
- Report: `reports/REPORT.md`
- Figures: `reports/figures/*.png`
- Robustness tables: `reports/robustness_*.csv`

## Limitations / known risks
- Universe snapshot may imply survivorship bias (documented).
- Daily data compresses intraday structure and execution (no intraday fills).
- Price coverage gaps (Stooq missing tickers) drop events and reduce sample size.
- SEC metadata is not a full “earnings calendar”; filings are sparse and heterogeneous.
- Costs and slippage are simplified; capacity/impact is not modeled.

## Next steps
- Improve universe definition (historical membership snapshots).
- Add richer “as-of” features and stricter availability timing.
- Add intraday execution model (open/close, spread proxy, delayed fills).
- Add more event families (earnings, guidance, macro events) with verified timestamps.
