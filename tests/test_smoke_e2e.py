import pandas as pd
from pathlib import Path

def test_smoke_can_read_artifacts_and_compute_basic_stats():
    # Minimal: ensure the repository can load and process sample fixtures
    prices = pd.read_parquet("tests/fixtures/prices_sample.parquet")
    events = pd.read_parquet("tests/fixtures/events_sample.parquet")

    assert {"ticker", "date", "close"}.issubset(prices.columns)
    assert {"event_id", "ticker", "trade_date"}.issubset(events.columns)

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["ret"] = prices.groupby("ticker")["close"].pct_change()

    assert prices["ret"].notna().sum() > 0
