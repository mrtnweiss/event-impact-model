from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from event_impact_model.utils.io import ensure_dir
from event_impact_model.utils.log import get_logger

log = get_logger("build_report")


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning(f"Missing: {path}")
        return None
    return pd.read_csv(path)


def plot_caar(event_summary_csv: Path, out_png: Path) -> None:
    df = _read_csv(event_summary_csv)
    if df is None or df.empty or not {"tau", "caar"}.issubset(df.columns):
        log.warning("Skipping CAAR plot (missing columns).")
        return

    df = df.sort_values("tau")
    plt.figure()
    plt.plot(df["tau"], df["caar"])
    plt.axhline(0.0)
    plt.xlabel("Tau (trading days)")
    plt.ylabel("CAAR")
    plt.title("CAAR over event window")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    log.info(f"Saved: {out_png}")


def plot_equity(daily_csv: Path, out_png: Path, title: str) -> None:
    df = _read_csv(daily_csv)
    if df is None or df.empty or "date" not in df.columns:
        return

    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in ["equity_gross", "equity_net"] if c in df.columns]
    if not cols:
        log.warning(f"Skipping equity plot (no equity columns) for {daily_csv}")
        return

    plt.figure()
    for c in cols:
        plt.plot(df["date"], df[c], label=c)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    log.info(f"Saved: {out_png}")


def load_cv_metrics(processed: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    ridge = processed / "cv_metrics.csv"
    lgbm = processed / "cv_metrics_lgbm.csv"

    d1 = _read_csv(ridge)
    if d1 is not None and not d1.empty:
        d1["model_file"] = "ridge/logit"
        rows.append(d1)

    d2 = _read_csv(lgbm)
    if d2 is not None and not d2.empty:
        d2["model_file"] = "lgbm"
        rows.append(d2)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "model" not in df.columns:
        df["model"] = df.get("model_file", "unknown")

    keep = [
        c
        for c in ["model", "mse", "r2", "pearson", "spearman", "hit_rate", "accuracy"]
        if c in df.columns
    ]
    out = (
        df[keep]
        .groupby("model", dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("model")
    )
    return out


def _engine_extras(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {}

    out: dict[str, float] = {}

    if "gross_exposure" in daily.columns:
        out["avg_gross_exposure"] = float(
            pd.to_numeric(daily["gross_exposure"], errors="coerce").mean()
        )

    if "lev" in daily.columns:
        out["avg_lev"] = float(pd.to_numeric(daily["lev"], errors="coerce").mean())

    if "ret_gross" in daily.columns and "cost" in daily.columns:
        rg = pd.to_numeric(daily["ret_gross"], errors="coerce").fillna(0.0)
        c = pd.to_numeric(daily["cost"], errors="coerce").fillna(0.0)
        denom = float(rg.abs().sum())
        out["cost_share_of_gross_abs"] = float(c.sum() / denom) if denom > 0 else float("nan")

    return out


def load_backtest_summaries(reports: Path) -> pd.DataFrame:
    variants = [
        ("engine", reports / "backtest_engine_summary.csv", reports / "backtest_engine_daily.csv"),
        (
            "engine_vt10",
            reports / "backtest_engine_summary_vt10.csv",
            reports / "backtest_engine_daily_vt10.csv",
        ),
        ("costed_eventlevel", reports / "backtest_costed_eventlevel_overall.csv", None),
    ]

    rows: list[dict] = []
    for name, sum_path, daily_path in variants:
        df = _read_csv(sum_path)
        if df is None or df.empty:
            continue

        r = df.iloc[0].to_dict()
        r["variant"] = name

        if daily_path is not None:
            d = _read_csv(Path(daily_path))
            if d is not None and not d.empty:
                r.update(_engine_extras(d))

        rows.append(r)

    return pd.DataFrame(rows)


def load_robustness_sanity(reports: Path) -> pd.DataFrame:
    p = reports / "robustness_sanity.csv"
    df = _read_csv(p)
    return df if df is not None else pd.DataFrame()


def load_robustness_subsamples(reports: Path) -> pd.DataFrame:
    p = reports / "robustness_subsamples.csv"
    df = _read_csv(p)
    return df if df is not None else pd.DataFrame()


def load_robustness_within_day_shuffle(reports: Path) -> pd.DataFrame:
    p = reports / "robustness_within_day_shuffle.csv"
    df = _read_csv(p)
    return df if df is not None else pd.DataFrame()


def write_markdown_report(
    out_md: Path,
    cv_summary: pd.DataFrame,
    bt_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    subsamples: pd.DataFrame,
    within_day_shuffle: pd.DataFrame,
    figs_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Event Impact Model — Report\n")
    lines.append("Autogenerated from the current pipeline artifacts.\n")

    lines.append("## Figures\n")
    caar_png = figs_dir / "caar.png"
    if caar_png.exists():
        lines.append(f"### CAAR\n\n![](figures/{caar_png.name})\n")

    eq_png = figs_dir / "equity_engine.png"
    if eq_png.exists():
        lines.append(f"### Equity Curve (Engine)\n\n![](figures/{eq_png.name})\n")

    eq_vt_png = figs_dir / "equity_engine_vt10.png"
    if eq_vt_png.exists():
        lines.append(
            f"### Equity Curve (Engine, vol target 10%)\n\n![](figures/{eq_vt_png.name})\n"
        )

    lines.append("## Cross-Validation Summary\n")
    if cv_summary.empty:
        lines.append("_No CV metrics found._\n")
    else:
        lines.append(cv_summary.to_markdown(index=False))
        lines.append("")

    lines.append("## Backtest Summary\n")
    if bt_summary.empty:
        lines.append("_No backtest summaries found._\n")
    else:
        cols = [
            c
            for c in [
                "variant",
                "sharpe_gross",
                "sharpe_net",
                "maxdd_gross",
                "maxdd_net",
                "avg_gross_exposure",
                "avg_lev",
                "mean_turnover",
                "mean_cost",
                "cost_share_of_gross_abs",
                "avg_active",
                "avg_long",
                "avg_short",
                "n_pairs",
                "mean_net",
                "hit_rate",
            ]
            if c in bt_summary.columns
        ]
        view = bt_summary[cols] if cols else bt_summary
        lines.append(view.to_markdown(index=False))
        lines.append("")

    lines.append("## Robustness (Sanity Checks)\n")
    if robustness.empty:
        lines.append("_No robustness sanity table found._\n")
    else:
        lines.append(robustness.to_markdown(index=False))
        lines.append("")

    lines.append("## Robustness (Subsamples)\n")
    if subsamples.empty:
        lines.append("_No subsample table found._\n")
    else:
        lines.append(subsamples.to_markdown(index=False))
        lines.append("")

    lines.append("## Robustness (Within-day Shuffle)\n")
    if within_day_shuffle.empty:
        lines.append("_No within-day shuffle table found._\n")
    else:
        lines.append(within_day_shuffle.to_markdown(index=False))
        lines.append("")

    lines.append("## Notes\n")
    lines.append("- Sensitivity grids are used for robustness checks, not for parameter optimization.\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved: {out_md}")


def main() -> None:
    root = Path(".")
    processed = root / "data" / "processed"
    reports = root / "reports"
    figs = reports / "figures"

    ensure_dir(reports)
    ensure_dir(figs)

    plot_caar(reports / "event_study_summary.csv", figs / "caar.png")
    plot_equity(
        reports / "backtest_engine_daily.csv",
        figs / "equity_engine.png",
        "Equity (Backtest Engine)",
    )
    plot_equity(
        reports / "backtest_engine_daily_vt10.csv",
        figs / "equity_engine_vt10.png",
        "Equity (Backtest Engine, Vol Target 10%)",
    )

    cv = load_cv_metrics(processed)
    cv_summary = summarize_cv(cv)

    bt_summary = load_backtest_summaries(reports)
    robustness = load_robustness_sanity(reports)
    subsamples = load_robustness_subsamples(reports)
    within_day_shuffle = load_robustness_within_day_shuffle(reports)

    write_markdown_report(
        reports / "REPORT.md",
        cv_summary=cv_summary,
        bt_summary=bt_summary,
        robustness=robustness,
        subsamples=subsamples,
        within_day_shuffle=within_day_shuffle,
        figs_dir=figs,
    )


if __name__ == "__main__":
    main()
