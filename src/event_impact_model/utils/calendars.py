from __future__ import annotations

import functools

import pandas as pd
import pandas_market_calendars as mcal


@functools.lru_cache(maxsize=4)
def _nyse_calendar():
    return mcal.get_calendar("NYSE")


def _ensure_et(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Ensure timestamp is timezone-aware US/Eastern.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        # treat naive as ET (explicitly)
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    return ts


def is_trading_day(date_et: pd.Timestamp) -> bool:
    """
    date_et: Timestamp (tz-naive or aware). Only date part used (ET).
    """
    d = pd.Timestamp(pd.Timestamp(date_et).date())
    cal = _nyse_calendar()
    sched = cal.schedule(start_date=d, end_date=d)
    return len(sched) > 0


def same_or_next_trading_day(date_et: pd.Timestamp, horizon_days: int = 14) -> pd.Timestamp:
    """
    Return same day if it is a NYSE trading day, otherwise the next NYSE trading day.
    Output is tz-naive Timestamp at midnight (date).
    """
    ts_et = _ensure_et(date_et)
    d0 = pd.Timestamp(ts_et.date())

    cal = _nyse_calendar()
    sched = cal.schedule(start_date=d0, end_date=d0 + pd.Timedelta(days=horizon_days))
    if sched.empty:
        raise RuntimeError("NYSE schedule empty; increase horizon_days or check calendar install.")

    days = sched.index.tz_localize(None)  # tz-naive dates at midnight
    for d in days:
        if d.date() >= d0.date():
            return pd.Timestamp(d.date())

    raise RuntimeError("No same-or-next trading day found within horizon.")


def next_trading_day(date_et: pd.Timestamp, horizon_days: int = 14) -> pd.Timestamp:
    """
    Return the next NYSE trading day strictly after the given date (ET).
    Output is tz-naive Timestamp at midnight (date).
    """
    ts_et = _ensure_et(date_et)
    d0 = pd.Timestamp(ts_et.date())

    cal = _nyse_calendar()
    sched = cal.schedule(start_date=d0, end_date=d0 + pd.Timedelta(days=horizon_days))
    if sched.empty:
        raise RuntimeError("NYSE schedule empty; increase horizon_days or check calendar install.")

    days = sched.index.tz_localize(None)
    for d in days:
        if d.date() > d0.date():
            return pd.Timestamp(d.date())

    raise RuntimeError("No next trading day found within horizon.")


def effective_trade_date(accepted_et: pd.Timestamp, bucket: str) -> pd.Timestamp:
    """
    Conservative 'tradable next open' mapping for Daily MVP:

    - premarket  -> same (or next) NYSE trading day
    - intraday   -> next NYSE trading day
    - afterhours -> next NYSE trading day

    Returns tz-naive Timestamp at midnight representing the trading date.
    """
    b = str(bucket).lower().strip()

    base = same_or_next_trading_day(accepted_et)

    if b == "premarket":
        return base

    if b in {"intraday", "afterhours"}:
        return next_trading_day(base)

    # Unknown bucket => be conservative
    return next_trading_day(base)
