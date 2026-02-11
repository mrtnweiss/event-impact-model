from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    q: float = 0.2
    delay_days: int = 1
    horizon_days: int = 5
    cost_bps: float = 2.0  # per notional traded (bps * turnover)
    max_positions: int = 20
    name_cap: float = 0.10  # max abs weight per name
    gross_target: float = 1.0  # target gross exposure before vol targeting
    vol_target_ann: float | None = None  # e.g. 0.10 for 10% annualized; None disables
    vol_lookback: int = 60
    min_positions: int = 2


def _to_trade_calendar(px: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.to_datetime(px["date"]).sort_values().unique()
    return pd.DatetimeIndex(dates)


def _compute_close_to_close_returns(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["ticker", "date"]).copy()
    px["date"] = pd.to_datetime(px["date"])
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"])
    px["ret_cc"] = px.groupby("ticker")["close"].pct_change()
    return px.dropna(subset=["ret_cc"])


def _map_to_trade_date(dates: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    # map to next available trade date (left)
    idx = dates.searchsorted(pd.Timestamp(d), side="left")
    if idx >= len(dates):
        return None
    return pd.Timestamp(dates[idx])


def _shift_trade_date(dates: pd.DatetimeIndex, d: pd.Timestamp, k: int) -> pd.Timestamp | None:
    # shift by k trading days on the global calendar
    idx = dates.searchsorted(pd.Timestamp(d), side="left")
    idx2 = idx + int(k)
    if idx2 < 0 or idx2 >= len(dates):
        return None
    return pd.Timestamp(dates[idx2])


def _make_signal_trades(
    oos: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    cfg: BacktestConfig,
    pred_col: str,
) -> pd.DataFrame:
    """
    Build trades from OOS predictions:
      - formation date: oos['date']
      - enter: formation mapped to trading day, then +delay_days
      - exit: enter + horizon_days
      - cross-sectional selection each formation day: long top-q, short bottom-q

    NOTE: SEC filings are sparse; do NOT require >=10 events per day.
    """
    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.dropna(subset=[pred_col, "ticker", "date"]).copy()

    trades: list[dict] = []

    for d, g in oos.groupby(oos["date"].dt.date):
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        n = len(g)

        # Minimal requirement: need at least 2 names for L/S,
        # and cfg.min_positions overall (you can set cfg.min_positions=2..5).
        if n < max(2, cfg.min_positions):
            continue

        # choose k so that long and short do not overlap
        k = int(np.floor(cfg.q * n))
        k = max(1, k)
        k = min(k, n // 2)  # ensure 2*k <= n
        if k < 1:
            continue

        long = g.head(k)
        short = g.tail(k)

        formation_ts = pd.Timestamp(d)
        t0 = _map_to_trade_date(trade_dates, formation_ts)
        if t0 is None:
            continue
        entry = _shift_trade_date(trade_dates, t0, cfg.delay_days)
        exit_ = _shift_trade_date(trade_dates, t0, cfg.delay_days + cfg.horizon_days)
        if entry is None or exit_ is None:
            continue

        for _, r in long.iterrows():
            trades.append({"ticker": str(r["ticker"]), "entry": entry, "exit": exit_, "side": 1.0})
        for _, r in short.iterrows():
            trades.append({"ticker": str(r["ticker"]), "entry": entry, "exit": exit_, "side": -1.0})

    tr = pd.DataFrame(trades)
    if tr.empty:
        raise RuntimeError("No trades created (check OOS dates and price calendar coverage).")
    return tr


def run_backtest(
    prices: pd.DataFrame,
    oos: pd.DataFrame,
    pred_col: str,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily holdings engine:
      - create trades (entry/exit) from OOS predictions
      - build daily target weights from all active trades
      - apply caps + max_positions
      - compute turnover, costs, net returns, equity curve
      - optional vol targeting
    Returns:
      daily dataframe, summary dataframe
    """
    pxr = _compute_close_to_close_returns(prices)
    trade_dates = _to_trade_calendar(pxr)

    trades = _make_signal_trades(oos=oos, trade_dates=trade_dates, cfg=cfg, pred_col=pred_col)
    trades = trades.sort_values(["entry", "exit", "ticker"]).reset_index(drop=True)

    # daily returns pivot for fast lookup
    rets = (
        pxr.pivot(index="date", columns="ticker", values="ret_cc").sort_index().reindex(trade_dates)
    )

    # Build daily weights by maintaining active trades
    active: dict[str, float] = {}
    # index trades by entry and exit date for add/remove
    by_entry = {}
    by_exit = {}
    for r in trades.itertuples(index=False):
        by_entry.setdefault(r.entry, []).append((r.ticker, float(r.side)))
        by_exit.setdefault(r.exit, []).append((r.ticker, float(r.side)))

    rows: list[dict] = []
    w_prev = pd.Series(dtype=float)

    # for vol targeting
    strat_rets: list[float] = []

    for dt in trade_dates:
        # remove exits first (positions are active on [entry, exit) i.e. exit day is out)
        if dt in by_exit:
            for t, _s in by_exit[dt]:
                active.pop(t, None)

        # add entries
        if dt in by_entry:
            for t, s in by_entry[dt]:
                # if multiple trades in same name overlap, they overwrite; MVP: keep last signal
                active[t] = s

        # build raw weights: equal weight per active trade side
        if not active:
            w = pd.Series(dtype=float)
        else:
            a = pd.Series(active, dtype=float)

            # enforce max_positions by strongest absolute signals (here: just keep first N deterministically)
            if len(a) > cfg.max_positions:
                a = a.sort_values(ascending=False)  # longs first; deterministic
                a = a.head(cfg.max_positions)

            # equal weight within active, then scale to gross_target
            w = a / max(1, len(a))
            # gross normalize
            gross = float(w.abs().sum())
            if gross > 0:
                w = w * (cfg.gross_target / gross)

            # per-name cap
            if cfg.name_cap is not None:
                w = w.clip(lower=-cfg.name_cap, upper=cfg.name_cap)
                # renormalize gross again after capping
                gross2 = float(w.abs().sum())
                if gross2 > 0:
                    w = w * (cfg.gross_target / gross2)

        # compute gross return
        if w.empty:
            r_gross = 0.0
        else:
            r_vec = rets.loc[dt, w.index].fillna(0.0)
            r_gross = float((w * r_vec).sum())

        # turnover & costs
        if w_prev.empty and w.empty:
            turnover = 0.0
        elif w_prev.empty:
            turnover = float(w.abs().sum())
        elif w.empty:
            turnover = float(w_prev.abs().sum())
        else:
            # standard definition: sum |Δw|
            turnover = float((w.reindex(w_prev.index, fill_value=0.0) - w_prev).abs().sum())

        cost = turnover * (cfg.cost_bps * 1e-4)
        r_net = r_gross - cost

        # optional vol targeting (apply scaling multiplicatively on weights via return scaling proxy)
        # MVP: scale net return by target_vol / realized_vol (rolling) to mimic leverage control
        lev = 1.0
        if cfg.vol_target_ann is not None:
            strat_rets.append(r_net)
            if len(strat_rets) >= cfg.vol_lookback:
                window = np.array(strat_rets[-cfg.vol_lookback :], dtype=float)
                vol = float(np.std(window, ddof=1)) * np.sqrt(252) if len(window) > 1 else 0.0
                if vol > 0:
                    lev = float(cfg.vol_target_ann / vol)
                    # clamp leverage to avoid explosions
                    lev = float(np.clip(lev, 0.0, 3.0))
        r_net_vt = r_net * lev
        r_gross_vt = r_gross * lev
        cost_vt = cost * lev  # approximate: costs scale with leverage

        rows.append(
            {
                "date": dt,
                "n_active": int(len(w)),
                "gross_exposure": float(w.abs().sum()) if not w.empty else 0.0,
                "turnover": float(turnover),
                "cost": float(cost_vt),
                "lev": float(lev),
                "ret_gross": float(r_gross_vt),
                "ret_net": float(r_net_vt),
            }
        )

        w_prev = w

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["equity_gross"] = (1.0 + daily["ret_gross"]).cumprod()
    daily["equity_net"] = (1.0 + daily["ret_net"]).cumprod()

    def _sharpe(x: pd.Series) -> float:
        x = x.astype(float)
        if x.std(ddof=1) <= 0 or len(x) < 2:
            return float("nan")
        return float((x.mean() / x.std(ddof=1)) * np.sqrt(252))

    def _maxdd(eq: pd.Series) -> float:
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        return float(dd.min())

    summary = pd.DataFrame(
        [
            {
                "days": int(len(daily)),
                "sharpe_gross": _sharpe(daily["ret_gross"]),
                "sharpe_net": _sharpe(daily["ret_net"]),
                "maxdd_gross": _maxdd(daily["equity_gross"]),
                "maxdd_net": _maxdd(daily["equity_net"]),
                "mean_turnover": float(daily["turnover"].mean()),
                "mean_cost": float(daily["cost"].mean()),
                "avg_active": float(daily["n_active"].mean()),
            }
        ]
    )

    return daily, summary
