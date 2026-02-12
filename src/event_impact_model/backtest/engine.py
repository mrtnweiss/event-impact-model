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
    max_positions: int = 20  # total active names (both sides)
    name_cap: float = 0.10  # max abs weight per name
    gross_target: float = 1.0  # target gross exposure before vol targeting
    vol_target_ann: float | None = None  # e.g. 0.10 for 10% annualized; None disables
    vol_lookback: int = 60
    min_positions: int = 2  # minimum names per formation day to trade


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
    idx = dates.searchsorted(pd.Timestamp(d), side="left")
    if idx >= len(dates):
        return None
    return pd.Timestamp(dates[idx])


def _shift_trade_date(dates: pd.DatetimeIndex, d: pd.Timestamp, k: int) -> pd.Timestamp | None:
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
    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.dropna(subset=[pred_col, "ticker", "date"]).copy()

    trades: list[dict] = []

    for d, g in oos.groupby(oos["date"].dt.date):
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        n = len(g)

        if n < max(2, cfg.min_positions):
            continue

        k = int(np.floor(cfg.q * n))
        k = max(1, k)
        k = min(k, n // 2)
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


def _cap_and_renormalize_gross(w: pd.Series, gross_target: float, name_cap: float) -> pd.Series:
    if w.empty:
        return w

    w = w.clip(lower=-name_cap, upper=name_cap)

    gross = float(w.abs().sum())
    if gross > 0:
        w = w * (gross_target / gross)
    return w


def _limit_positions_side_balanced(a: pd.Series, max_positions: int) -> pd.Series:
    if a.empty or len(a) <= max_positions:
        return a

    longs = a[a > 0].sort_index()
    shorts = a[a < 0].sort_index()

    half = max_positions // 2
    n_long = min(len(longs), half)
    n_short = min(len(shorts), max_positions - n_long)

    if n_long < half:
        n_short = min(len(shorts), max_positions - n_long)
    if n_short < (max_positions - half):
        n_long = min(len(longs), max_positions - n_short)

    keep = pd.concat([longs.head(n_long), shorts.head(n_short)])
    return keep


def run_backtest(
    prices: pd.DataFrame,
    oos: pd.DataFrame,
    pred_col: str,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pxr = _compute_close_to_close_returns(prices)
    trade_dates = _to_trade_calendar(pxr)

    trades = _make_signal_trades(oos=oos, trade_dates=trade_dates, cfg=cfg, pred_col=pred_col)
    trades = trades.sort_values(["entry", "exit", "ticker"]).reset_index(drop=True)

    rets = (
        pxr.pivot(index="date", columns="ticker", values="ret_cc").sort_index().reindex(trade_dates)
    )

    active: dict[str, float] = {}
    by_entry: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    by_exit: dict[pd.Timestamp, list[tuple[str, float]]] = {}

    for r in trades.itertuples(index=False):
        by_entry.setdefault(r.entry, []).append((r.ticker, float(r.side)))
        by_exit.setdefault(r.exit, []).append((r.ticker, float(r.side)))

    rows: list[dict] = []
    w_prev = pd.Series(dtype=float)
    strat_rets: list[float] = []

    for dt in trade_dates:
        if dt in by_exit:
            for t, _s in by_exit[dt]:
                active.pop(t, None)

        if dt in by_entry:
            for t, s in by_entry[dt]:
                active[t] = s

        if not active:
            w = pd.Series(dtype=float)
            n_long = 0
            n_short = 0
        else:
            a = pd.Series(active, dtype=float)

            if cfg.max_positions is not None and len(a) > cfg.max_positions:
                a = _limit_positions_side_balanced(a, cfg.max_positions)

            n_long = int((a > 0).sum())
            n_short = int((a < 0).sum())

            w = a / max(1, len(a))

            gross = float(w.abs().sum())
            if gross > 0:
                w = w * (cfg.gross_target / gross)

            if cfg.name_cap is not None:
                w = _cap_and_renormalize_gross(w, cfg.gross_target, cfg.name_cap)

        if w.empty:
            r_gross = 0.0
        else:
            r_vec = rets.loc[dt, w.index].fillna(0.0)
            r_gross = float((w * r_vec).sum())

        if w_prev.empty and w.empty:
            turnover = 0.0
        else:
            idx = w.index.union(w_prev.index)
            turnover = float(
                (w.reindex(idx, fill_value=0.0) - w_prev.reindex(idx, fill_value=0.0)).abs().sum()
            )

        cost = turnover * (cfg.cost_bps * 1e-4)
        r_net = r_gross - cost

        lev = 1.0
        if cfg.vol_target_ann is not None:
            strat_rets.append(r_net)
            if len(strat_rets) >= cfg.vol_lookback:
                window = np.array(strat_rets[-cfg.vol_lookback :], dtype=float)
                vol = float(np.std(window, ddof=1)) * np.sqrt(252) if len(window) > 1 else 0.0
                if vol > 0:
                    lev = float(cfg.vol_target_ann / vol)
                    lev = float(np.clip(lev, 0.0, 3.0))

        rows.append(
            {
                "date": dt,
                "n_active": int(len(w)),
                "n_long": int(n_long),
                "n_short": int(n_short),
                "gross_exposure": float(w.abs().sum()) if not w.empty else 0.0,
                "turnover": float(turnover),
                "cost": float(cost * lev),
                "lev": float(lev),
                "ret_gross": float(r_gross * lev),
                "ret_net": float((r_net) * lev),
            }
        )

        w_prev = w

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["equity_gross"] = (1.0 + daily["ret_gross"]).cumprod()
    daily["equity_net"] = (1.0 + daily["ret_net"]).cumprod()

    def _sharpe(x: pd.Series) -> float:
        x = x.astype(float)
        if len(x) < 2:
            return float("nan")
        sd = float(x.std(ddof=1))
        if sd <= 0:
            return float("nan")
        return float((x.mean() / sd) * np.sqrt(252))

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
                "avg_long": float(daily["n_long"].mean()),
                "avg_short": float(daily["n_short"].mean()),
            }
        ]
    )

    return daily, summary
