from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.backtest.engine import BacktestConfig, run_backtest
from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("robustness_sanity")


def _load_oos(processed: Path) -> tuple[pd.DataFrame, str]:
    oos_lgbm = processed / "oos_predictions_lgbm.parquet"
    if oos_lgbm.exists():
        oos = pd.read_parquet(oos_lgbm).copy()
        return oos, "y_pred_lgbm"
    oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
    return oos, "y_pred_ridge"


def _run_variant(
    prices: pd.DataFrame,
    oos: pd.DataFrame,
    pred_col: str,
    cfg: BacktestConfig,
) -> dict:
    daily, summary = run_backtest(prices=prices, oos=oos, pred_col=pred_col, cfg=cfg)
    s = summary.iloc[0].to_dict()
    s["days"] = int(s["days"])
    return s


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

    rows: list[dict] = []

    base = _run_variant(prices, oos, pred_col=pred_col, cfg=cfg)
    base["variant"] = "base_oos_pred"
    rows.append(base)

    oos_rand_pred = oos.copy()
    oos_rand_pred["pred_random"] = rng.standard_normal(len(oos_rand_pred))
    s = _run_variant(prices, oos_rand_pred, pred_col="pred_random", cfg=cfg)
    s["variant"] = "random_predictions"
    rows.append(s)

    oos_rand_label = oos.copy()
    if "y_true" in oos_rand_label.columns:
        y = pd.to_numeric(oos_rand_label["y_true"], errors="coerce")
        oos_rand_label["pred_shuffled_labels"] = rng.permutation(y.fillna(0.0).to_numpy())
    else:
        oos_rand_label["pred_shuffled_labels"] = rng.permutation(
            pd.to_numeric(oos_rand_label[pred_col], errors="coerce").fillna(0.0).to_numpy()
        )
    s = _run_variant(prices, oos_rand_label, pred_col="pred_shuffled_labels", cfg=cfg)
    s["variant"] = "shuffled_labels_proxy"
    rows.append(s)

    out = pd.DataFrame(rows)
    out = out[
        [
            "variant",
            "days",
            "sharpe_gross",
            "sharpe_net",
            "maxdd_gross",
            "maxdd_net",
            "mean_turnover",
            "mean_cost",
            "avg_active",
        ]
        + [c for c in ["avg_long", "avg_short"] if c in out.columns]
    ]

    out_path = reports / "robustness_sanity.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
