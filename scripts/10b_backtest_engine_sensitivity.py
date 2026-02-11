from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from event_impact_model.backtest.engine import BacktestConfig, run_backtest
from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("backtest_engine_sensitivity")


def _load_oos(processed: Path) -> tuple[pd.DataFrame, str]:
    oos_lgbm = processed / "oos_predictions_lgbm.parquet"
    if oos_lgbm.exists():
        oos = pd.read_parquet(oos_lgbm).copy()
        pred_col = "y_pred_lgbm"
        log.info("Using LGBM OOS predictions.")
        return oos, pred_col

    oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
    pred_col = "y_pred_ridge"
    log.warning("Using ridge OOS predictions (no LGBM file found).")
    return oos, pred_col


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    px = pd.read_parquet(processed / "prices.parquet").copy()
    oos, pred_col = _load_oos(processed)

    # Base config (keep constraints consistent across grid)
    base = BacktestConfig(
        q=0.2,
        delay_days=1,
        horizon_days=5,
        cost_bps=2.0,
        max_positions=20,
        name_cap=0.10,
        gross_target=1.0,
        vol_target_ann=None,  # grid focuses on core knobs; VT is validated in scripts/10
        vol_lookback=60,
        min_positions=2,
    )

    log.info(
        f"Default config: delay={base.delay_days} horizon={base.horizon_days} q={base.q} cost_bps={base.cost_bps}"
    )

    delays = [0, 1]
    horizons = [3, 5, 10]
    qs = [0.1, 0.2, 0.3]
    costs = [1.0, 2.0, 5.0, 10.0]

    rows: list[dict] = []
    for delay in delays:
        for horizon in horizons:
            for q in qs:
                for cost_bps in costs:
                    cfg = replace(
                        base, delay_days=delay, horizon_days=horizon, q=q, cost_bps=cost_bps
                    )

                    daily, summary = run_backtest(prices=px, oos=oos, pred_col=pred_col, cfg=cfg)
                    s = summary.iloc[0].to_dict()

                    rows.append(
                        {
                            "delay_days": delay,
                            "horizon_days": horizon,
                            "q": q,
                            "cost_bps": cost_bps,
                            "days": int(s["days"]),
                            "sharpe_gross": float(s["sharpe_gross"]),
                            "sharpe_net": float(s["sharpe_net"]),
                            "maxdd_net": float(s["maxdd_net"]),
                            "mean_turnover": float(s["mean_turnover"]),
                            "mean_cost": float(s["mean_cost"]),
                            "avg_active": float(s["avg_active"]),
                        }
                    )

    grid = pd.DataFrame(rows)
    out = reports / "backtest_engine_sensitivity.csv"
    grid.to_csv(out, index=False)
    log.info(f"Saved: {out} rows={len(grid):,}")

    # Guardrail (wichtig fürs Memo/Review)
    log.info("Sensitivity grid is for robustness; we do NOT select the best row for trading.")

    # Print top 10 by net Sharpe (purely diagnostic)
    top = grid.sort_values("sharpe_net", ascending=False).head(10)
    for r in top.itertuples(index=False):
        log.info(
            f"TOP | delay={r.delay_days} horizon={r.horizon_days} q={r.q} cost={r.cost_bps} "
            f"sharpe_net={r.sharpe_net:.2f} maxdd_net={r.maxdd_net:.2%} avg_active={r.avg_active:.1f}"
        )


if __name__ == "__main__":
    main()
