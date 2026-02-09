# scripts/02_fetch_prices.py
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from event_impact_model.utils.io import ensure_dir, write_parquet
from event_impact_model.utils.log import get_logger

log = get_logger("fetch_prices")


def load_latest_universe_csv(raw_dir: Path) -> pd.DataFrame:
    """
    Loads the most recent universe snapshot from data/raw.

    Supports:
      - universe_sec_YYYY-MM-DD.csv (SEC universe)
      - sp500_YYYY-MM-DD.csv (older Wikipedia version)
    """
    candidates = sorted(raw_dir.glob("universe_sec_*.csv")) + sorted(raw_dir.glob("sp500_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No universe snapshot found in data/raw. Run scripts/01_fetch_universe_sec.py first."
        )
    latest = candidates[-1]
    log.info(f"Using universe file: {latest}")
    return pd.read_csv(latest)


def normalize_ticker_for_stooq(t: str) -> str:
    """
    Stooq US equities are commonly requested as {ticker}.us.
    Some tickers with '-' might not exist on Stooq; we handle missing gracefully.
    """
    return t.strip().upper()


def stooq_daily_csv_url_us(ticker: str) -> str:
    """
    Stooq daily CSV endpoint. Example:
      https://stooq.com/q/d/l/?s=spy.us&i=d
    """
    return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"


def fetch_stooq_daily_csv(
    ticker: str, cache_dir: Path, session: requests.Session
) -> pd.DataFrame | None:
    """
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume (Stooq standard),
    or None if no data.
    """
    cache_path = cache_dir / f"{ticker}.csv"

    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            return df if not df.empty else None
        except Exception:
            # corrupted cache file
            cache_path.unlink(missing_ok=True)

    url = stooq_daily_csv_url_us(ticker)
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None

    txt = r.text.strip()
    if not txt or "No data" in txt:
        return None

    cache_path.write_text(txt, encoding="utf-8")
    try:
        df = pd.read_csv(cache_path)
        return df if not df.empty else None
    except Exception:
        return None


def standardize_stooq_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Converts Stooq CSV -> standardized schema:
      ticker, date, open, high, low, close, volume
    """
    cols = {c.lower(): c for c in raw.columns}
    if "date" not in cols:
        return None

    df = raw.copy()
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Stooq uses 'Date' etc; now lowercase
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ticker"] = ticker
    keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["close"])
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    return df if not df.empty else None


def main(config_path: str = "configs/base.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    raw_dir = Path(cfg["paths"]["raw"])
    processed_dir = Path(cfg["paths"]["processed"])
    cache_dir = Path(cfg["paths"]["cache"]) / "stooq"

    ensure_dir(processed_dir)
    ensure_dir(cache_dir)

    uni = load_latest_universe_csv(raw_dir)

    if "ticker" not in uni.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column.")

    tickers = [normalize_ticker_for_stooq(t) for t in uni["ticker"].astype(str).tolist()]
    tickers = [t for t in tickers if t]  # drop empties

    benchmark = str(cfg["prices"].get("benchmark", "SPY")).strip().upper()
    if benchmark and benchmark not in tickers:
        tickers = [benchmark] + tickers

    start = pd.to_datetime(cfg["prices"]["start"]).date()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36",
            "Accept": "text/csv,text/plain,*/*;q=0.8",
        }
    )

    rows: list[pd.DataFrame] = []
    missing: list[str] = []

    for t in tqdm(tickers, desc="Downloading Stooq daily prices"):
        raw = fetch_stooq_daily_csv(t, cache_dir, session)
        if raw is None:
            missing.append(t)
            continue

        df = standardize_stooq_df(raw, t)
        if df is None or df.empty:
            missing.append(t)
            continue

        df = df[df["date"] >= start].copy()
        if df.empty:
            missing.append(t)
            continue

        rows.append(df)

        # be polite
        time.sleep(0.15)

    if rows:
        prices = pd.concat(rows, ignore_index=True)
        prices = prices.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    else:
        prices = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])

    out = processed_dir / "prices.parquet"
    write_parquet(prices, out)

    log.info(
        f"Saved prices: {out} rows={len(prices):,} tickers={prices['ticker'].nunique() if not prices.empty else 0}"
    )
    if missing:
        log.info(f"Missing tickers (first 30): {missing[:30]} (total {len(missing)})")


if __name__ == "__main__":
    main()
