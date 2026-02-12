from __future__ import annotations

from pathlib import Path

import pandas as pd

from event_impact_model.backtest.engine import BacktestConfig, run_backtest
from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("robustness_subsamples")


def _load_oos(processed: Path) -> tuple[pd.DataFrame, str]:
    oos_lgbm = processed / "oos_predictions_lgbm.parquet"
    if oos_lgbm.exists():
        oos = pd.read_parquet(oos_lgbm).copy()
        return oos, "y_pred_lgbm"
    oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
    return oos, "y_pred_ridge"


def _run_one(prices: pd.DataFrame, oos: pd.DataFrame, pred_col: str, cfg: BacktestConfig) -> dict:
    daily, summary = run_backtest(prices=prices, oos=oos, pred_col=pred_col, cfg=cfg)
    s = summary.iloc[0].to_dict()
    s["days"] = int(s["days"])
    return s


def _subset_or_empty(oos: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    sub = oos.loc[mask].copy()
    return sub


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    prices = pd.read_parquet(processed / "prices.parquet").copy()
    oos, pred_col = _load_oos(processed)

    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.dropna(subset=["ticker", "date", pred_col]).copy()
    oos["year"] = oos["date"].dt.year.astype(int)

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

    rows: list[dict] = []

    base = _run_one(prices, oos, pred_col=pred_col, cfg=cfg)
    base.update({"group": "ALL", "split": "ALL"})
    rows.append(base)

    for y in sorted(oos["year"].unique().tolist()):
        sub = _subset_or_empty(oos, oos["year"] == y)
        if len(sub) < 200:
            continue
        s = _run_one(prices, sub, pred_col=pred_col, cfg=cfg)
        s.update({"group": "year", "split": str(y)})
        rows.append(s)

    if "form" in oos.columns:
        for v, sub in oos.groupby("form"):
            if len(sub) < 200:
                continue
            s = _run_one(prices, sub, pred_col=pred_col, cfg=cfg)
            s.update({"group": "form", "split": str(v)})
            rows.append(s)

    if "session_bucket" in oos.columns:
        for v, sub in oos.groupby("session_bucket"):
            if len(sub) < 200:
                continue
            s = _run_one(prices, sub, pred_col=pred_col, cfg=cfg)
            s.update({"group": "session_bucket", "split": str(v)})
            rows.append(s)

    out = pd.DataFrame(rows)

    keep = [
        "group",
        "split",
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
    out = out[keep].sort_values(["group", "split"])

    out_path = reports / "robustness_subsamples.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
