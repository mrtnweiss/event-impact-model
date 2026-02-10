# scripts/04_event_study_mvp.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir, write_parquet
from event_impact_model.utils.log import get_logger

log = get_logger("event_study_mvp")


def compute_returns(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["ticker", "date"]).copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"])
    px["ret"] = px.groupby("ticker")["close"].pct_change()
    return px.dropna(subset=["ret"])


def add_event_index_dates(ev: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """
    Since trade_date is already NYSE-trading-day-aware, we align by exact match:
      trade_date_aligned = trade_date  if a return observation exists on that date.
    Otherwise drop the event.

    This avoids silent forward-shifts to "next available print".
    """
    ev = ev.copy()
    ev["trade_date"] = pd.to_datetime(ev["trade_date"]).dt.date

    # build a fast membership set per ticker
    dates_by_ticker: dict[str, set] = {}
    for t, g in rets.groupby("ticker"):
        dates_by_ticker[t] = set(g["date"].unique())

    aligned = []
    missing = 0
    for row in ev.itertuples(index=False):
        t = row.ticker
        d = row.trade_date
        s = dates_by_ticker.get(t)
        if s is None or d not in s:
            aligned.append(None)
            missing += 1
        else:
            aligned.append(d)

    ev["trade_date_aligned"] = aligned
    ev = ev.dropna(subset=["trade_date_aligned"]).copy()

    log.info(f"Alignment: dropped {missing:,} events without price return on trade_date")
    return ev


def market_model_ar_for_event(
    rets: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    mkt_ticker: str,
    est_pre: int,
    est_gap: int,
    tau_min: int,
    tau_max: int,
    min_est_obs: int = 60,
) -> dict | None:
    """
    OLS market model on estimation window then AR over tau_min..tau_max (inclusive).
    """
    g = rets[rets["ticker"] == ticker].copy()
    m = (
        rets[rets["ticker"] == mkt_ticker][["date", "ret"]]
        .rename(columns={"ret": "mkt_ret"})
        .copy()
    )

    if g.empty or m.empty:
        return None

    df = g.merge(m, on="date", how="inner").sort_values("date").reset_index(drop=True)

    dates = df["date"].tolist()
    try:
        e_idx = dates.index(event_date)
    except ValueError:
        return None

    est_start = e_idx - est_pre
    est_end = e_idx - est_gap  # exclusive
    if est_start < 0 or est_end <= est_start:
        return None

    est = df.iloc[est_start:est_end].dropna(subset=["ret", "mkt_ret"])
    if len(est) < min_est_obs:
        return None

    X = np.column_stack([np.ones(len(est)), est["mkt_ret"].to_numpy()])
    y = est["ret"].to_numpy()
    try:
        alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    except Exception:
        return None

    w_start = e_idx + tau_min
    w_end = e_idx + tau_max + 1
    if w_start < 0 or w_end > len(df):
        return None

    evw = df.iloc[w_start:w_end].copy()
    taus = np.arange(tau_min, tau_max + 1)
    evw["tau"] = taus
    evw["ar"] = evw["ret"] - (alpha + beta * evw["mkt_ret"])

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "n_est": int(len(est)),
        "ar_by_tau": dict(zip(evw["tau"].astype(int), evw["ar"].astype(float), strict=True)),
    }


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(processed)
    ensure_dir(reports)

    px = pd.read_parquet(processed / "prices.parquet")
    ev = pd.read_parquet(processed / "events.parquet")

    px["date"] = pd.to_datetime(px["date"]).dt.date
    ev["accepted_utc"] = pd.to_datetime(ev["accepted_utc"], utc=True, errors="coerce")
    ev["ticker"] = ev["ticker"].astype(str).str.upper().str.strip()
    px["ticker"] = px["ticker"].astype(str).str.upper().str.strip()

    # Parameters
    mkt = "SPY"
    est_pre = 120
    est_gap = 21
    tau_min = -10
    tau_max = 5
    min_est_obs = 60

    rets = compute_returns(px)

    tick_px = set(rets["ticker"].unique())
    tick_ev = set(ev["ticker"].unique())
    common = sorted(list((tick_px & tick_ev) - {mkt}))
    if mkt not in tick_px:
        raise ValueError("Benchmark SPY not present in prices returns. Ensure SPY is downloaded.")

    log.info(f"Tickers with both prices & events (excluding SPY): {len(common)}")

    # keep session_bucket from events input
    keep_cols = ["event_id", "ticker", "form", "accepted_utc", "trade_date", "session_bucket"]
    ev = ev[keep_cols].copy()
    ev = ev[ev["ticker"].isin(common)].copy()

    ev = add_event_index_dates(ev, rets)
    log.info(f"Events after aligning trade dates to trading days: {len(ev):,}")

    out_rows = []
    ar_panel = []

    for row in ev.itertuples(index=False):
        res = market_model_ar_for_event(
            rets=rets,
            ticker=row.ticker,
            event_date=row.trade_date_aligned,
            mkt_ticker=mkt,
            est_pre=est_pre,
            est_gap=est_gap,
            tau_min=tau_min,
            tau_max=tau_max,
            min_est_obs=min_est_obs,
        )
        if res is None:
            continue

        out_rows.append(
            {
                "event_id": row.event_id,
                "ticker": row.ticker,
                "form": row.form,
                "session_bucket": row.session_bucket,
                "accepted_utc": row.accepted_utc,
                "trade_date": row.trade_date,
                "trade_date_aligned": row.trade_date_aligned,
                "alpha": res["alpha"],
                "beta": res["beta"],
                "n_est": res["n_est"],
            }
        )
        for tau, ar in res["ar_by_tau"].items():
            ar_panel.append({"event_id": row.event_id, "tau": int(tau), "ar": float(ar)})

    events_out = pd.DataFrame(out_rows)
    ar_out = pd.DataFrame(ar_panel)

    if events_out.empty:
        raise RuntimeError(
            "No events produced (likely insufficient history for estimation window)."
        )

    # CAAR over tau range
    caar = ar_out.groupby("tau")["ar"].mean().reset_index().rename(columns={"ar": "caar"})
    caar["n_events"] = ar_out.groupby("tau")["ar"].count().values
    caar.to_csv(reports / "event_study_summary.csv", index=False)

    # Convenience CAR main window for quick log
    main_taus = list(range(-1, 6))
    car_main = ar_out[ar_out["tau"].isin(main_taus)].groupby("event_id")["ar"].sum()
    events_out = events_out.merge(car_main.rename("car_main_-1_5"), on="event_id", how="left")

    write_parquet(events_out, processed / "event_study_events.parquet")
    write_parquet(ar_out, processed / "event_study_ar.parquet")

    mean_car = events_out["car_main_-1_5"].mean()
    sd = events_out["car_main_-1_5"].std(ddof=1)
    t_stat = (
        mean_car / (sd / np.sqrt(len(events_out))) if sd and len(events_out) > 1 else float("nan")
    )

    log.info(f"Saved: data/processed/event_study_events.parquet rows={len(events_out):,}")
    log.info(f"Saved: data/processed/event_study_ar.parquet rows={len(ar_out):,}")
    log.info("Saved: reports/event_study_summary.csv")
    log.info(f"CAR[-1,5] mean={mean_car:.6f} t-stat≈{t_stat:.2f} n_events={len(events_out):,}")


if __name__ == "__main__":
    main()
