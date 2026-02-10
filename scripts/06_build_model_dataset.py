# scripts/06_build_model_dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir, write_parquet
from event_impact_model.utils.log import get_logger

log = get_logger("build_model_dataset")


def compute_daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["ticker", "date"]).copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["volume"] = pd.to_numeric(px.get("volume", np.nan), errors="coerce")
    px = px.dropna(subset=["close"])
    px["ret"] = px.groupby("ticker")["close"].pct_change()
    px["dollar_vol"] = px["close"] * px["volume"]
    return px


def rolling_features(px: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-event features computed from returns up to t-1.
    """
    g = px.groupby("ticker", group_keys=False)

    # momentum as cumulative returns (t-k..t-1)
    px["mom_1d"] = g["ret"].shift(1)
    px["mom_5d"] = g["ret"].shift(1).rolling(5).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    px["mom_20d"] = g["ret"].shift(1).rolling(20).apply(lambda x: np.prod(1 + x) - 1, raw=False)

    # vol as std of returns (t-20..t-1)
    px["vol_20d"] = g["ret"].shift(1).rolling(20).std()

    # liquidity proxy
    px["dollar_vol_20d"] = g["dollar_vol"].shift(1).rolling(20).mean()

    return px


def main() -> None:
    processed = Path("data/processed")
    ensure_dir(processed)

    px = pd.read_parquet(processed / "prices.parquet")
    ar = pd.read_parquet(processed / "event_study_ar.parquet")
    ev = pd.read_parquet(processed / "event_study_events.parquet")

    px["date"] = pd.to_datetime(px["date"]).dt.date
    ev["trade_date_aligned"] = pd.to_datetime(ev["trade_date_aligned"]).dt.date

    # Build labels: CAR[+1,+5]
    label_taus = [1, 2, 3, 4, 5]
    y = ar[ar["tau"].isin(label_taus)].groupby("event_id")["ar"].sum().rename("y_car_p1_p5")

    ev = ev.merge(y, on="event_id", how="inner")
    ev = ev.dropna(subset=["y_car_p1_p5"]).copy()

    # Features from prices on the aligned trade date
    px2 = compute_daily_returns(px)
    px2 = rolling_features(px2)

    feat_cols = ["mom_1d", "mom_5d", "mom_20d", "vol_20d", "dollar_vol_20d"]
    keep_cols = ["ticker", "date"] + feat_cols
    feats = px2[keep_cols].copy()

    # Join on (ticker, trade_date_aligned)
    df = ev.merge(
        feats,
        left_on=["ticker", "trade_date_aligned"],
        right_on=["ticker", "date"],
        how="left",
    )

    # Event meta features
    dt = pd.to_datetime(df["trade_date_aligned"])
    df["dow"] = dt.dt.dayofweek.astype("int64")
    df["month"] = dt.dt.month.astype("int64")

    # Keep model columns
    out_cols = [
        "event_id",
        "ticker",
        "form",
        "session_bucket",
        "accepted_utc",
        "trade_date_aligned",
        "y_car_p1_p5",
        "mom_1d",
        "mom_5d",
        "mom_20d",
        "vol_20d",
        "dollar_vol_20d",
        "dow",
        "month",
    ]
    df = df[out_cols].copy()

    # Drop rows with missing features (need enough history)
    before = len(df)
    df = df.dropna(
        subset=["y_car_p1_p5", "mom_1d", "mom_5d", "mom_20d", "vol_20d", "dollar_vol_20d"]
    )
    after = len(df)

    out = processed / "model_dataset.parquet"
    write_parquet(df, out)

    log.info(
        f"Saved model dataset: {out} rows={after:,} (dropped {before - after:,} due to missing features)"
    )
    log.info(f"Date range: {df['trade_date_aligned'].min()} .. {df['trade_date_aligned'].max()}")
    log.info(
        f"Tickers: {df['ticker'].nunique()} | forms: {df['form'].nunique()} | buckets: {df['session_bucket'].nunique()}"
    )


if __name__ == "__main__":
    main()
