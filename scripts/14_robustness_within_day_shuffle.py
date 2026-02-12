from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.backtest.engine import BacktestConfig, run_backtest
from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("robustness_within_day_shuffle")


def _load_oos(processed: Path) -> tuple[pd.DataFrame, str]:
    oos_lgbm = processed / "oos_predictions_lgbm.parquet"
    if oos_lgbm.exists():
        oos = pd.read_parquet(oos_lgbm).copy()
        return oos, "y_pred_lgbm"
    oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
    return oos, "y_pred_ridge"


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    prices = pd.read_parquet(processed / "prices.parquet").copy()
    oos, pred_col = _load_oos(processed)

    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.dropna(subset=["ticker", "date", pred_col]).copy()

    cfg = BacktestConfig(
        q=0.2,
        delay_days=1,
        horizon_days=5,
        cost_bps=2.0,
        max_positions=20,
        name_cap=0.10,
        gross_target=1.0,
        vol_target_ann=None,
        vol_lookback=60,
        min_positions=2,
    )

    rng = np.random.default_rng(42)

    base_daily, base_sum = run_backtest(prices=prices, oos=oos, pred_col=pred_col, cfg=cfg)
    base = base_sum.iloc[0].to_dict()
    base["variant"] = "base_oos_pred"

    o2 = oos.copy()
    o2["pred_within_day_shuffle"] = o2[pred_col].astype(float)

    for d, idx in o2.groupby(o2["date"].dt.date).groups.items():
        vals = o2.loc[idx, "pred_within_day_shuffle"].to_numpy()
        o2.loc[idx, "pred_within_day_shuffle"] = rng.permutation(vals)

    shuf_daily, shuf_sum = run_backtest(
        prices=prices, oos=o2, pred_col="pred_within_day_shuffle", cfg=cfg
    )
    shuf = shuf_sum.iloc[0].to_dict()
    shuf["variant"] = "within_day_shuffle"

    out = pd.DataFrame([base, shuf])
    keep = [
        "variant",
        "days",
        "sharpe_gross",
        "sharpe_net",
        "maxdd_gross",
        "maxdd_net",
        "mean_turnover",
        "mean_cost",
        "avg_active",
        "avg_long",
        "avg_short",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    out_path = reports / "robustness_within_day_shuffle.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
