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
    """
    px = px.sort_values(["ticker", "date"]).copy()
    g = px.groupby("ticker", group_keys=False)

    entry_close = g["close"].shift(-delay_days)
    exit_close = g["close"].shift(-(delay_days + horizon_days))

    px["ret_trade_cc"] = (exit_close / entry_close) - 1.0
    return px


def run_eventlevel_quantile_ls(
    df: pd.DataFrame,
    pred_col: str,
    q: float,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Event-level L/S pairing + Monthly basket with turnover-based costs.
    Requires columns: ticker, date, pred_col, ret_trade_cc
    """
    d = df.dropna(subset=[pred_col, "ret_trade_cc"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    if d.empty:
        raise RuntimeError("No events with predictions + trade returns (ret_trade_cc).")

    # -------------------
    # Event-level pairing
    # -------------------
    d = d.sort_values(pred_col, ascending=False).reset_index(drop=True)
    n = len(d)
    k = max(1, int(np.floor(q * n)))

    long = d.head(k).reset_index(drop=True)
    short = d.tail(k).reset_index(drop=True)

    ls = long["ret_trade_cc"] - short["ret_trade_cc"]

    # flat per-pair cost proxy: 2 legs
    cost_pair = 2.0 * (cost_bps * 1e-4)
    ls_net = ls - cost_pair

    def _sharpe252(x: pd.Series) -> float:
        x = x.dropna().astype(float)
        if len(x) < 2:
            return float("nan")
        mu = float(x.mean())
        sd = float(x.std(ddof=1))
        return float((mu / sd) * np.sqrt(252)) if sd > 0 else float("nan")

    overall = pd.DataFrame(
        [
            {
                "n_pairs": int(len(ls)),
                "mean_gross": float(ls.mean()),
                "std_gross": float(ls.std(ddof=1)) if len(ls) > 1 else float("nan"),
                "sharpe_gross": _sharpe252(ls),
                "mean_net": float(ls_net.mean()),
                "std_net": float(ls_net.std(ddof=1)) if len(ls_net) > 1 else float("nan"),
                "sharpe_net": _sharpe252(ls_net),
                "hit_rate": float((ls_net > 0).mean()),
                "cost_per_pair": float(cost_pair),
            }
        ]
    )

    # -------------------
    # Monthly baskets + turnover cost
    # -------------------
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    monthly_rows: list[dict] = []

    prev_long: set[str] | None = None
    prev_short: set[str] | None = None

    for ym, g in d.groupby("ym"):
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        n_m = len(g)
        if n_m < 30:
            continue

        k_m = max(1, int(np.floor(q * n_m)))

        long_g = g.head(k_m).copy()
        short_g = g.tail(k_m).copy()

        gross_ls = float(long_g["ret_trade_cc"].mean() - short_g["ret_trade_cc"].mean())

        long_set = set(long_g["ticker"].astype(str))
        short_set = set(short_g["ticker"].astype(str))

        if prev_long is None:
            turnover = 1.0
        else:
            long_overlap = len(long_set & prev_long) / max(1, len(long_set))
            short_overlap = len(short_set & prev_short) / max(1, len(short_set))
            turnover = 0.5 * ((1.0 - long_overlap) + (1.0 - short_overlap))

        cost = 2.0 * turnover * (cost_bps * 1e-4)
        net_ls = gross_ls - cost

        monthly_rows.append(
            {
                "ym": ym,
                "n_events": int(n_m),
                "k": int(k_m),
                "gross_ls": gross_ls,
                "net_ls": net_ls,
                "turnover": float(turnover),
                "cost": float(cost),
            }
        )

        prev_long = long_set
        prev_short = short_set

    monthly = pd.DataFrame(monthly_rows).sort_values("ym")
    return overall, monthly


def run_sensitivity_grid(
    base_df: pd.DataFrame,  # columns: ticker, date, pred_col
    pred_col: str,
    px: pd.DataFrame,
    delays: list[int],
    horizons: list[int],
    qs: list[float],
    costs_bps: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []

    for delay_days in delays:
        for horizon_days in horizons:
            px2 = compute_trade_return_cc(px, delay_days=delay_days, horizon_days=horizon_days)
            d = base_df.merge(
                px2[["ticker", "date", "ret_trade_cc"]], on=["ticker", "date"], how="left"
            )

            for q in qs:
                for cost_bps in costs_bps:
                    overall, monthly = run_eventlevel_quantile_ls(
                        d, pred_col=pred_col, q=q, cost_bps=cost_bps
                    )

                    # monthly sharpe (ann)
                    months = int(len(monthly))
                    sharpe_m = float("nan")
                    if months >= 6:
                        x = monthly["net_ls"].astype(float)
                        mu = float(x.mean())
                        sd = float(x.std(ddof=1))
                        sharpe_m = float((mu / sd) * np.sqrt(12)) if sd > 0 else float("nan")

                    r = overall.iloc[0].to_dict()
                    rows.append(
                        {
                            "delay_days": delay_days,
                            "horizon_days": horizon_days,
                            "q": q,
                            "cost_bps": cost_bps,
                            "n_pairs": int(r["n_pairs"]),
                            "event_mean_net": float(r["mean_net"]),
                            "event_hit": float(r["hit_rate"]),
                            "monthly_sharpe_ann": sharpe_m,
                            "months": months,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

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

    # Base DF for merges
    base_df = oos[["ticker", "date", pred_col]].copy()

    # Main config (single run)
    delay_days = 1
    horizon_days = 5
    q = 0.2
    cost_bps = 2.0

    log.info(f"Default config: delay={delay_days} horizon={horizon_days} q={q} cost_bps={cost_bps}")

    px2 = compute_trade_return_cc(px, delay_days=delay_days, horizon_days=horizon_days)
    df = base_df.merge(px2[["ticker", "date", "ret_trade_cc"]], on=["ticker", "date"], how="left")

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

    if not monthly.empty and len(monthly) >= 6:
        x = monthly["net_ls"].astype(float)
        mu = float(x.mean())
        sd = float(x.std(ddof=1))
        sharpe_m = (mu / sd) * np.sqrt(12) if sd > 0 else float("nan")
        log.info(f"Monthly net Sharpe (ann.): {sharpe_m:.2f} | months={len(monthly):,}")

    # Sensitivity grid
    grid = run_sensitivity_grid(
        base_df=base_df,
        pred_col=pred_col,
        px=px,
        delays=[0, 1],
        horizons=[3, 5, 10],
        qs=[0.1, 0.2, 0.3],
        costs_bps=[1.0, 2.0, 5.0, 10.0],
    )
    out3 = reports / "backtest_costed_sensitivity.csv"
    grid.to_csv(out3, index=False)
    log.info(f"Saved: {out3} rows={len(grid):,}")

    log.info("Sensitivity grid is for robustness; we do NOT select the best row for trading.")
    top = grid.sort_values("monthly_sharpe_ann", ascending=False).head(10)
    for r in top.itertuples(index=False):
        log.info(
            f"TOP | delay={r.delay_days} horizon={r.horizon_days} q={r.q} cost={r.cost_bps} "
            f"monthlySharpe={r.monthly_sharpe_ann:.2f} months={r.months}"
        )


if __name__ == "__main__":
    main()
