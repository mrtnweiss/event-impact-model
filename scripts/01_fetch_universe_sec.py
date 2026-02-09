import datetime as dt
from pathlib import Path

import pandas as pd
import requests
import yaml

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("fetch_universe_sec")

SEC_TICKER_CIK = "https://www.sec.gov/files/company_tickers.json"


def normalize_ticker(t: str) -> str:
    # keep BRK-B style (SEC uses BRK-B), good for many providers
    return t.strip().upper()


def main(config_path: str = "configs/base.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw"])
    ensure_dir(raw_dir)

    max_tickers = int(cfg["universe"].get("max_tickers", 200))

    headers = {
        "User-Agent": cfg["events"]["user_agent"],  # reuse SEC-compliant UA
        "Accept-Encoding": "gzip, deflate",
    }

    r = requests.get(SEC_TICKER_CIK, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.rename(columns={"ticker": "ticker", "title": "name", "cik_str": "cik"})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)

    # simple deterministic subset for MVP: sort by ticker
    df = df.sort_values("ticker").head(max_tickers).reset_index(drop=True)

    df["source"] = "sec_company_tickers"
    df["asof_date"] = dt.date.today().isoformat()

    out = raw_dir / f"universe_sec_{df['asof_date'].iloc[0]}.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved universe snapshot: {out} ({len(df)} tickers)")


if __name__ == "__main__":
    main()
