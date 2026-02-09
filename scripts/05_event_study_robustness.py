# scripts/05_event_study_robustness.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("event_study_robustness")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (q-values).
    Returns array of same shape with monotone BH q-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def t_stat_mean(x: np.ndarray) -> tuple[float, float]:
    """
    Mean and approximate t-stat (normal approx).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0:
        return mu, float("nan")
    t = mu / (sd / np.sqrt(n))
    return mu, t


def bootstrap_pvalue_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> float:
    """
    Two-sided bootstrap p-value for mean=0.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    # two-sided: proportion of |boot| >= |mu|
    p = float((np.abs(boots) >= abs(mu)).mean())
    return p


def car_from_ar(ar_df: pd.DataFrame, taus: list[int]) -> pd.Series:
    """
    Compute CAR over a set of taus for each event_id.
    """
    sub = ar_df[ar_df["tau"].isin(taus)].copy()
    return sub.groupby("event_id")["ar"].sum()


def main() -> None:
    processed = Path("data/processed")
    reports = Path("reports")
    ensure_dir(reports)

    # Inputs from Script 04
    ev = pd.read_parquet(processed / "event_study_events.parquet")
    ar = pd.read_parquet(processed / "event_study_ar.parquet")

    # Windows
    pretrend_taus = list(range(-10, -1))  # -10..-2
    car_main_taus = list(range(-1, 6))    # -1..+5

    # Compute per-event CARs
    car_pre = car_from_ar(ar, pretrend_taus).rename("car_pretrend")
    car_main = car_from_ar(ar, car_main_taus).rename("car_main")

    ev = ev.merge(car_pre, on="event_id", how="left").merge(car_main, on="event_id", how="left")
    ev = ev.dropna(subset=["car_pretrend", "car_main"]).copy()

    log.info(f"Events with both pretrend and main CAR: {len(ev):,}")

    # Overall tests
    overall = []
    for label, col in [("CAR_pretrend[-10,-2]", "car_pretrend"), ("CAR_main[-1,5]", "car_main")]:
        mu, t = t_stat_mean(ev[col].to_numpy())
        p = bootstrap_pvalue_mean(ev[col].to_numpy(), n_boot=2000, seed=42)
        overall.append({
            "group": "ALL",
            "split": "ALL",
            "metric": label,
            "n": len(ev),
            "mean": mu,
            "t_stat": t,
            "boot_p": p,
        })

    # Subgroup tests (form, session_bucket)
    tests = []
    for split_col in ["form", "session_bucket"]:
        for split_val, g in ev.groupby(split_col):
            if len(g) < 100:
                continue
            for label, col in [("CAR_pretrend[-10,-2]", "car_pretrend"), ("CAR_main[-1,5]", "car_main")]:
                mu, t = t_stat_mean(g[col].to_numpy())
                p = bootstrap_pvalue_mean(g[col].to_numpy(), n_boot=2000, seed=42)
                tests.append({
                    "group": split_col,
                    "split": str(split_val),
                    "metric": label,
                    "n": len(g),
                    "mean": mu,
                    "t_stat": t,
                    "boot_p": p,
                })

    df_tests = pd.DataFrame(overall + tests)

    # FDR across subgroup tests (exclude ALL/ALL rows)
    mask = ~((df_tests["group"] == "ALL") & (df_tests["split"] == "ALL"))
    pvals = df_tests.loc[mask, "boot_p"].to_numpy()
    if len(pvals) > 0:
        df_tests.loc[mask, "bh_q"] = bh_fdr(pvals)
    else:
        df_tests["bh_q"] = np.nan

    out = reports / "event_study_robustness.csv"
    df_tests.to_csv(out, index=False)
    log.info(f"Saved: {out}")

    # Print top findings (smallest p)
    if mask.any():
        top = df_tests.loc[mask].sort_values("boot_p").head(10)
        log.info("Top subgroup results by bootstrap p-value:")
        for r in top.itertuples(index=False):
            log.info(f"{r.group}={r.split} | {r.metric} | n={r.n} mean={r.mean:.6f} t={r.t_stat:.2f} p={r.boot_p:.4f} q={getattr(r,'bh_q',np.nan):.4f}")


if __name__ == "__main__":
    main()
