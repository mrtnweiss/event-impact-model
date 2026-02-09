from datetime import time
import pandas as pd
import pytz

ET = pytz.timezone("US/Eastern")

def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def to_et(ts: pd.Timestamp) -> pd.Timestamp:
    ts = to_utc(ts)
    return ts.tz_convert(ET)

def session_bucket(et_ts: pd.Timestamp) -> str:
    t = et_ts.timetz()
    if t < time(9, 30):
        return "premarket"
    if t >= time(16, 0):
        return "afterhours"
    return "intraday"
