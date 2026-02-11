from __future__ import annotations

from pathlib import Path

import pandas as pd

from event_impact_model.backtest.engine import BacktestConfig, run_backtest
from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("backtest_engine")


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    # predictions (prefer LGBM)
    oos_lgbm = processed / "oos_predictions_lgbm.parquet"
    if oos_lgbm.exists():
        oos = pd.read_parquet(oos_lgbm).copy()
        pred_col = "y_pred_lgbm"
        log.info("Using LGBM OOS predictions.")
    else:
        oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
        pred_col = "y_pred_ridge"
        log.warning("Using ridge OOS predictions (no LGBM file found).")

    prices = pd.read_parquet(processed / "prices.parquet").copy()

    cfg = BacktestConfig(
        q=0.2,
        delay_days=1,
        horizon_days=5,
        cost_bps=2.0,
        max_positions=20,
        name_cap=0.10,
        gross_target=1.0,
        vol_target_ann=0.10,  # set None to disable
        vol_lookback=60,
    )

    daily, summary = run_backtest(prices=prices, oos=oos, pred_col=pred_col, cfg=cfg)

    out_daily = reports / "backtest_engine_daily.csv"
    out_summary = reports / "backtest_engine_summary.csv"
    daily.to_csv(out_daily, index=False)
    summary.to_csv(out_summary, index=False)

    log.info(f"Saved: {out_daily} rows={len(daily):,}")
    log.info(f"Saved: {out_summary}")

    s = summary.iloc[0].to_dict()
    log.info(
        f"Sharpe gross={s['sharpe_gross']:.2f} net={s['sharpe_net']:.2f} | "
        f"MaxDD gross={100 * s['maxdd_gross']:.2f}% net={100 * s['maxdd_net']:.2f}% | "
        f"avg_active={s['avg_active']:.1f} mean_turnover={s['mean_turnover']:.3f} mean_cost={s['mean_cost']:.6f}"
    )


if __name__ == "__main__":
    main()
