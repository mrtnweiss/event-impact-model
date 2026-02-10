# scripts/08b_backtest_oos_eventlevel.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("backtest_oos_eventlevel")


def summarize_returns(x: pd.Series) -> dict:
    x = x.dropna().astype(float)
    n = len(x)
    mu = x.mean() if n else np.nan
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = mu / (sd / np.sqrt(n)) if n > 1 and sd and sd > 0 else np.nan
    sharpe = (mu / sd) * np.sqrt(252) if n > 1 and sd and sd > 0 else np.nan
    hit = float((x > 0).mean()) if n else np.nan
    return {
        "n": n,
        "mean": float(mu),
        "std": float(sd),
        "t_stat": float(t),
        "sharpe": float(sharpe),
        "hit": hit,
    }


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.dropna(subset=["y_pred_ridge", "y_true"])

    # Parameters
    q = 0.2  # top/bottom quantile

    # Overall event-level L/S
    oos = oos.sort_values("y_pred_ridge", ascending=False).reset_index(drop=True)
    n = len(oos)
    k = max(1, int(np.floor(q * n)))

    long = oos.head(k).copy()
    short = oos.tail(k).copy()

    # realized proxy: y_true (CAR[+1,+5])
    long_ret = long["y_true"]
    short_ret = short["y_true"]
    ls_ret = long_ret.reset_index(drop=True) - short_ret.reset_index(drop=True)

    overall = summarize_returns(ls_ret)
    log.info(
        f"Event-level LS (top/bottom {q:.0%}) | n_pairs={overall['n']:,} mean={overall['mean']:.6f} "
        f"t≈{overall['t_stat']:.2f} sharpe≈{overall['sharpe']:.2f} hit={overall['hit']:.3f}"
    )

    # Monthly evaluation (more interpretable for sparse events)
    oos["ym"] = oos["date"].dt.to_period("M").astype(str)
    monthly_rows = []
    for ym, g in oos.groupby("ym"):
        g = g.sort_values("y_pred_ridge", ascending=False).reset_index(drop=True)
        n_m = len(g)
        if n_m < 20:
            continue
        k_m = max(1, int(np.floor(q * n_m)))
        long_m = g.head(k_m)["y_true"].reset_index(drop=True)
        short_m = g.tail(k_m)["y_true"].reset_index(drop=True)
        ls_m = long_m - short_m

        s = summarize_returns(ls_m)
        monthly_rows.append({"ym": ym, **s})

    monthly = pd.DataFrame(monthly_rows).sort_values("ym")
    out1 = reports / "backtest_oos_eventlevel_overall.csv"
    out2 = reports / "backtest_oos_eventlevel_monthly.csv"

    pd.DataFrame([overall]).to_csv(out1, index=False)
    monthly.to_csv(out2, index=False)

    log.info(f"Saved: {out1}")
    log.info(f"Saved: {out2} (rows={len(monthly):,})")

    if not monthly.empty:
        log.info(
            "Monthly mean of means (rough): "
            f"{monthly['mean'].mean():.6f} over {len(monthly)} months"
        )


if __name__ == "__main__":
    main()
