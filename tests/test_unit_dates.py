import pandas as pd

from event_impact_model.utils.calendars import effective_trade_date
from event_impact_model.utils.dates import session_bucket


def test_session_bucket():
    assert session_bucket(pd.Timestamp("2024-01-01 08:00", tz="US/Eastern")) == "premarket"
    assert session_bucket(pd.Timestamp("2024-01-01 10:00", tz="US/Eastern")) == "intraday"
    assert session_bucket(pd.Timestamp("2024-01-01 16:30", tz="US/Eastern")) == "afterhours"


def test_effective_trade_date_afterhours_friday_goes_to_monday():
    # Fri afterhours -> next trading day (usually Monday)
    ts = pd.Timestamp("2024-01-05 16:30", tz="US/Eastern")  # Friday
    d = effective_trade_date(ts, "afterhours")
    # Monday 2024-01-08
    assert d.date().isoformat() == "2024-01-08"


def test_effective_trade_date_on_holiday_goes_to_next_trading_day():
    # US Independence Day 2024-07-04 (NYSE closed)
    ts = pd.Timestamp("2024-07-04 08:00", tz="US/Eastern")
    d = effective_trade_date(ts, "premarket")
    assert d.date().isoformat() == "2024-07-05"


def test_effective_trade_date_premarket_trading_day_same_day():
    ts = pd.Timestamp("2024-01-03 08:00", tz="US/Eastern")  # Wed
    d = effective_trade_date(ts, "premarket")
    assert d.date().isoformat() == "2024-01-03"


def test_effective_trade_date_intraday_next_trading_day():
    ts = pd.Timestamp("2024-01-03 10:00", tz="US/Eastern")
    d = effective_trade_date(ts, "intraday")
    assert d.date().isoformat() == "2024-01-04"


def test_effective_trade_date_dst_boundary_does_not_crash():
    # DST start 2024-03-10 (Sunday). Premarket on Sunday -> next trading day Monday.
    ts = pd.Timestamp("2024-03-10 08:00", tz="US/Eastern")
    d = effective_trade_date(ts, "premarket")
    assert d.date().isoformat() == "2024-03-11"
