import subprocess
import sys

import pandas as pd


def test_smoke_can_read_fixtures_and_compute_returns():
    prices = pd.read_parquet("tests/fixtures/prices_sample.parquet")
    events = pd.read_parquet("tests/fixtures/events_sample.parquet")

    assert {"ticker", "date", "close"}.issubset(prices.columns)
    assert {"event_id", "ticker", "trade_date"}.issubset(events.columns)

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["ret"] = prices.groupby("ticker")["close"].pct_change()

    assert prices["ret"].notna().sum() > 0


def test_smoke_report_build_runs_offline():
    proc = subprocess.run(
        [sys.executable, "scripts/11_build_report.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
