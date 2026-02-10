from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("backtest_costed")


def compute_trade_return_cc(px: pd.DataFrame, delay_days: int, horizon_days: int) -> pd.DataFrame:
    """
    Trade return with execution delay and holding horizon, close-to-close:

      entry at t+delay_days close
      exit  at entry + horizon_days close

    return = close[t+delay+h]/close[t+delay] - 1

    Uses row shifts per ticker (trading-day index).
    """
    px = px.sort_values(["ticker", "date"]).copy()
    g = px.groupby("ticker", group_keys=False)

    entry_close = g["close"].shift(-delay_days)
    exit_close = g["close"].shift(-(delay_days + horizon_days))

    px["ret_trade_cc"] = (exit_close / entry_close) - 1.0
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


def run_eventlevel_quantile_ls(
    df: pd.DataFrame,
    pred_col: str,
    q: float,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Event-level L/S:
      - sort all OOS events by prediction
      - long top-q events, short bottom-q events (paired by rank)
      - realized return: ret_trade_cc for each event
      - costs: apply 2 legs * cost_bps per "pair" (flat approx)

    Returns:
      - overall summary (1 row)
      - monthly summary (one row per month)
    """
    d = df.dropna(subset=[pred_col, "ret_trade_cc"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    if d.empty:
        raise RuntimeError("No events with predictions + forward returns.")

    d = d.sort_values(pred_col, ascending=False).reset_index(drop=True)
    n = len(d)
    k = max(1, int(np.floor(q * n)))

    long = d.head(k).copy()
    short = d.tail(k).copy().reset_index(drop=True)
    long = long.reset_index(drop=True)

    # pair returns
    ls = long["ret_trade_cc"] - short["ret_trade_cc"]
    cost = 2.0 * (cost_bps * 1e-4)
    ls_net = ls - cost

    overall = pd.DataFrame(
        [{
            "n_pairs": int(len(ls)),
            "mean_gross": float(ls.mean()),
            "std_gross": float(ls.std(ddof=1)) if len(ls) > 1 else float("nan"),
            "sharpe_gross": float((ls.mean() / ls.std(ddof=1)) * np.sqrt(252)) if len(ls) > 1 and ls.std(ddof=1) > 0 else float("nan"),
            "mean_net": float(ls_net.mean()),
            "std_net": float(ls_net.std(ddof=1)) if len(ls_net) > 1 else float("nan"),
            "sharpe_net": float((ls_net.mean() / ls_net.std(ddof=1)) * np.sqrt(252)) if len(ls_net) > 1 and ls_net.std(ddof=1) > 0 else float("nan"),
            "hit_rate": float((ls_net > 0).mean()),
            "cost_per_pair": float(cost),
        }]
    )

    # monthly: do the same inside each month (more interpretable with sparse daily formation)
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    monthly_rows = []
    for ym, g in d.groupby("ym"):
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        n_m = len(g)
        if n_m < 30:
            continue
        k_m = max(1, int(np.floor(q * n_m)))
        long_m = g.head(k_m)["ret_trade_cc"].reset_index(drop=True)
        short_m = g.tail(k_m)["ret_trade_cc"].reset_index(drop=True)
        ls_m = long_m - short_m
        ls_m_net = ls_m - cost

        monthly_rows.append(
            {
                "ym": ym,
                "n_pairs": int(len(ls_m)),
                "mean_gross": float(ls_m.mean()),
                "mean_net": float(ls_m_net.mean()),
                "hit_rate": float((ls_m_net > 0).mean()),
            }
        )

    monthly = pd.DataFrame(monthly_rows).sort_values("ym")
    return overall, monthly


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
    delay_days = 1       # trade at t+1 close (MVP)
    horizon_days = 5     # hold 5 trading days
    q = 0.2
    cost_bps = 2.0       # per leg

    px2 = compute_trade_return_cc(px, delay_days=delay_days, horizon_days=horizon_days)

    df = oos.merge(
        px2[["ticker", "date", "ret_trade_cc"]],
        on=["ticker", "date"],
        how="left",
    )


    overall, monthly = run_eventlevel_quantile_ls(df, pred_col=pred_col, q=q, cost_bps=cost_bps)

    out1 = reports / "backtest_costed_eventlevel_overall.csv"
    out2 = reports / "backtest_costed_eventlevel_monthly.csv"
    overall.to_csv(out1, index=False)
    monthly.to_csv(out2, index=False)

    log.info(f"Saved: {out1}")
    log.info(f"Saved: {out2} rows={len(monthly):,}")

    r = overall.iloc[0].to_dict()
    log.info(
        f"Event-level LS (top/bot {q:.0%}) n_pairs={int(r['n_pairs']):,} "
        f"mean_net={r['mean_net']:.6f} sharpe_net≈{r['sharpe_net']:.2f} hit={r['hit_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
