# scripts/08_backtest_oos.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("backtest_oos")


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.dropna(subset=["y_pred_ridge", "y_true"])

    # Parameters
    q = 0.2          # top/bottom quantile
    hold_days = 5    # aligns with y_true = CAR[+1,+5]

    # group by formation day
    daily = []
    for d, g in oos.groupby(oos["date"].dt.date):
        g = g.sort_values("y_pred_ridge", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 10:
            continue

        k = max(1, int(np.floor(q * n)))
        long = g.head(k)
        short = g.tail(k)

        # Event-level realized returns proxy: use y_true
        long_ret = float(long["y_true"].mean())
        short_ret = float(short["y_true"].mean())
        ls_ret = long_ret - short_ret

        daily.append({
            "date": pd.Timestamp(d),
            "n_events": n,
            "n_long": len(long),
            "n_short": len(short),
            "mean_pred_long": float(long["y_pred_ridge"].mean()),
            "mean_pred_short": float(short["y_pred_ridge"].mean()),
            "ret_long": long_ret,
            "ret_short": short_ret,
            "ret_ls": ls_ret,
        })

    bt = pd.DataFrame(daily).sort_values("date")
    if bt.empty:
        raise RuntimeError("Backtest produced no days (too few OOS events per day).")

    # Metrics (treat each formation day as one observation)
    mu = bt["ret_ls"].mean()
    sd = bt["ret_ls"].std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(252) if sd and sd > 0 else np.nan

    hit = float((bt["ret_ls"] > 0).mean())

    log.info(f"Backtest days: {len(bt):,}")
    log.info(f"Mean daily LS return (event proxy): {mu:.6f}")
    log.info(f"Std daily LS return: {sd:.6f}")
    log.info(f"Sharpe (ann.): {sharpe:.2f}")
    log.info(f"Hit-rate: {hit:.3f}")

    out = reports / "backtest_oos_daily.csv"
    bt.to_csv(out, index=False)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
