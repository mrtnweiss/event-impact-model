import pandas as pd
from event_impact_model.utils.dates import session_bucket

def test_session_bucket():
    assert session_bucket(pd.Timestamp("2024-01-01 08:00", tz="US/Eastern")) == "premarket"
    assert session_bucket(pd.Timestamp("2024-01-01 10:00", tz="US/Eastern")) == "intraday"
    assert session_bucket(pd.Timestamp("2024-01-01 16:30", tz="US/Eastern")) == "afterhours"
