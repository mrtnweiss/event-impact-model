from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("backtest_costed")


def compute_fwd_return_cc(px: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Forward close-to-close return: r_{t->t+h} = close[t+h]/close[t] - 1
    Uses row-shift per ticker, so it assumes the DataFrame is sorted by trading days.
    """
    px = px.sort_values(["ticker", "date"]).copy()
    g = px.groupby("ticker", group_keys=False)
    px["close_fwd"] = g["close"].shift(-horizon_days)
    px["ret_fwd_cc"] = (px["close_fwd"] / px["close"]) - 1.0
    return px


def sharpe_ann(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    if len(x) < 2:
        return float("nan")
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    return float((mu / sd) * np.sqrt(252)) if sd > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def run_daily_quantile_ls(
    df: pd.DataFrame,
    pred_col: str,
    q: float,
    cost_bps: float,
) -> pd.DataFrame:
    """
    Event-day formation:
      - each date: long top-q by prediction, short bottom-q
      - realized return: ret_fwd_cc (already forward horizon)
      - costs: 2 legs * cost_bps per notional (flat, per formation)
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date

    rows = []
    for date, g in d.groupby("date"):
        g = g.dropna(subset=[pred_col, "ret_fwd_cc"]).sort_values(pred_col, ascending=False)
        n = len(g)
        if n < 10:
            continue

        k = max(1, int(np.floor(q * n)))
        long = g.head(k)
        short = g.tail(k)

        long_ret = float(long["ret_fwd_cc"].mean())
        short_ret = float(short["ret_fwd_cc"].mean())
        gross = long_ret - short_ret

        # Flat costs per formation day (approx): 2 legs
        cost = 2.0 * (cost_bps * 1e-4)
        net = gross - cost

        rows.append(
            {
                "date": pd.Timestamp(date),
                "n": n,
                "k": k,
                "gross_ls": gross,
                "net_ls": net,
                "cost": cost,
            }
        )

    bt = pd.DataFrame(rows).sort_values("date")
    if bt.empty:
        raise RuntimeError("No formation days produced (too few events per day).")

    bt["equity_gross"] = (1.0 + bt["gross_ls"]).cumprod()
    bt["equity_net"] = (1.0 + bt["net_ls"]).cumprod()
    return bt


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    # Prefer LGBM predictions; fallback to ridge
    oos_lgbm = processed / "oos_predictions_lgbm.parquet"
    if oos_lgbm.exists():
        oos = pd.read_parquet(oos_lgbm).copy()
        pred_col = "y_pred_lgbm"
        log.info("Using LGBM OOS predictions.")
    else:
        oos = pd.read_parquet(processed / "oos_predictions.parquet").copy()
        pred_col = "y_pred_ridge"
        log.warning("Using ridge OOS predictions (no LGBM file found).")

    px = pd.read_parquet(processed / "prices.parquet").copy()

    oos["date"] = pd.to_datetime(oos["date"])
    px["date"] = pd.to_datetime(px["date"])
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"])

    # Parameters (MVP)
    horizon_days = 5  # holding horizon
    q = 0.2  # top/bottom quantile
    cost_bps = 2.0  # per leg (bps)

    px2 = compute_fwd_return_cc(px, horizon_days=horizon_days)

    df = oos.merge(
        px2[["ticker", "date", "ret_fwd_cc"]],
        on=["ticker", "date"],
        how="left",
    )

    bt = run_daily_quantile_ls(df, pred_col=pred_col, q=q, cost_bps=cost_bps)

    out = reports / "backtest_costed_daily.csv"
    bt.to_csv(out, index=False)

    log.info(f"Saved: {out} rows={len(bt):,}")
    log.info(f"Gross Sharpe: {sharpe_ann(bt['gross_ls']):.2f}")
    log.info(f"Net Sharpe:   {sharpe_ann(bt['net_ls']):.2f}")
    log.info(f"MaxDD gross:  {max_drawdown(bt['equity_gross']):.2%}")
    log.info(f"MaxDD net:    {max_drawdown(bt['equity_net']):.2%}")


if __name__ == "__main__":
    main()
