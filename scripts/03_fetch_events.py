# scripts/03_fetch_events.py
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from event_impact_model.utils.dates import session_bucket, to_et
from event_impact_model.utils.io import ensure_dir, write_parquet
from event_impact_model.utils.log import get_logger

log = get_logger("fetch_events")

SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


def load_latest_universe_csv(raw_dir: Path) -> pd.DataFrame:
    candidates = sorted(raw_dir.glob("universe_sec_*.csv")) + sorted(raw_dir.glob("sp500_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No universe snapshot found in data/raw. Run scripts/01_fetch_universe_sec.py first."
        )
    latest = candidates[-1]
    log.info(f"Using universe file: {latest}")
    return pd.read_csv(latest)


def stable_event_id(ticker: str, form: str, accepted_utc: pd.Timestamp, accession: str) -> str:
    s = f"{ticker}|{form}|{accepted_utc.isoformat()}|{accession}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


_DIGITS14 = re.compile(r"^\d{14}$")


def parse_acceptance_datetime_utc(val) -> pd.Timestamp | None:
    """
    Robustly parse SEC acceptanceDateTime:
    - usually "YYYYMMDDhhmmss" (14 digits, UTC)
    - sometimes may be missing/invalid
    Returns pd.Timestamp(tz='UTC') or None.
    """
    if val is None:
        return None

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # Common case: 14-digit string
    if _DIGITS14.match(s):
        try:
            return pd.Timestamp(
                f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}",
                tz="UTC",
            )
        except Exception:
            return None

    # Fallback: try pandas parse (assume UTC if tz-naive)
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        # ensure Timestamp
        return pd.Timestamp(ts)
    except Exception:
        return None


def mvp_trade_date(accepted_et: pd.Timestamp, bucket: str) -> pd.Timestamp:
    """
    MVP daily mapping (conservative):
      - premarket -> same calendar date
      - intraday/afterhours -> next calendar date
    Later we replace with a real NYSE trading calendar.
    """
    if bucket == "premarket":
        return pd.Timestamp(accepted_et.date())
    return pd.Timestamp((accepted_et + pd.Timedelta(days=1)).date())


def fetch_submissions_json(
    cik_10: str,
    cache_dir: Path,
    session: requests.Session,
) -> dict | None:
    cache_path = cache_dir / f"submissions_{cik_10}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = SEC_SUBMISSIONS.format(cik=cik_10)
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None

    cache_path.write_text(r.text, encoding="utf-8")
    try:
        return r.json()
    except Exception:
        return None


def main(config_path: str = "configs/base.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    raw_dir = Path(cfg["paths"]["raw"])
    processed_dir = Path(cfg["paths"]["processed"])
    cache_dir = Path(cfg["paths"]["cache"]) / "sec"
    ensure_dir(processed_dir)
    ensure_dir(cache_dir)

    uni = load_latest_universe_csv(raw_dir)

    if "ticker" not in uni.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column.")
    if "cik" not in uni.columns:
        raise ValueError(
            "Universe CSV must contain a 'cik' column. "
            "Use scripts/01_fetch_universe_sec.py (SEC universe) for this pipeline."
        )

    uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()
    uni["cik"] = uni["cik"].astype(str).str.zfill(10)

    forms = set(cfg["events"]["forms"])
    max_events_per_ticker = int(cfg["events"].get("max_events_per_ticker", 2000))

    user_agent = cfg["events"]["user_agent"]
    if "@" not in user_agent or " " not in user_agent:
        log.warning("SEC user_agent should be like 'Name email@domain.com' to avoid blocks.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/plain,*/*;q=0.8",
        }
    )

    events: list[dict] = []
    failed: list[str] = []
    bad_acceptance = 0
    total_filings_seen = 0

    for row in tqdm(
        uni.itertuples(index=False), total=len(uni), desc="Downloading EDGAR submissions"
    ):
        ticker = row.ticker
        cik_10 = row.cik

        data = fetch_submissions_json(cik_10, cache_dir, session)
        if not data:
            failed.append(ticker)
            time.sleep(0.15)
            continue

        recent = (data.get("filings", {}) or {}).get("recent", {}) or {}
        forms_list = recent.get("form", []) or []
        acc_list = recent.get("accessionNumber", []) or []
        acc_dt_list = recent.get("acceptanceDateTime", []) or []

        n = min(len(forms_list), len(acc_list), len(acc_dt_list), max_events_per_ticker)

        for i in range(n):
            total_filings_seen += 1
            form = forms_list[i]
            if form not in forms:
                continue

            accession = str(acc_list[i])
            accepted_utc = parse_acceptance_datetime_utc(acc_dt_list[i])
            if accepted_utc is None:
                bad_acceptance += 1
                continue

            accepted_et = to_et(accepted_utc)
            bucket = session_bucket(accepted_et)

            events.append(
                {
                    "event_id": stable_event_id(ticker, form, accepted_utc, accession),
                    "ticker": ticker,
                    "cik": cik_10,
                    "form": form,
                    "accession": accession,
                    "accepted_utc": accepted_utc,
                    "accepted_et": accepted_et,
                    "session_bucket": bucket,
                    "trade_date": mvp_trade_date(accepted_et, bucket).date(),
                }
            )

        time.sleep(0.15)

    ev = pd.DataFrame(events)
    if not ev.empty:
        ev = ev.drop_duplicates(subset=["event_id"]).sort_values(["ticker", "accepted_utc"])

    out = processed_dir / "events.parquet"
    write_parquet(ev, out)

    log.info(
        f"Saved events: {out} rows={len(ev):,} tickers={ev['ticker'].nunique() if not ev.empty else 0}"
    )
    log.info(
        f"Bad acceptanceDateTime skipped: {bad_acceptance} (out of {total_filings_seen} filings scanned)"
    )
    if failed:
        log.info(f"Failed tickers (first 30): {failed[:30]} (total {len(failed)})")


if __name__ == "__main__":
    main()
