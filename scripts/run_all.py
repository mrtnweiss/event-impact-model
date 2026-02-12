from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    steps = [
        ["python", str(root / "scripts" / "01_fetch_universe_sec.py")],
        ["python", str(root / "scripts" / "02_fetch_prices.py")],
        ["python", str(root / "scripts" / "03_fetch_events.py")],
        ["python", str(root / "scripts" / "04_event_study_mvp.py")],
        ["python", str(root / "scripts" / "05_event_study_robustness.py")],
        ["python", str(root / "scripts" / "06_build_model_dataset.py")],
        ["python", str(root / "scripts" / "07_train_cv_baselines.py")],
        ["python", str(root / "scripts" / "07b_train_cv_lgbm.py")],
        ["python", str(root / "scripts" / "09_backtest_costed.py")],
        ["python", str(root / "scripts" / "10_backtest_engine.py")],
        ["python", str(root / "scripts" / "12_robustness_sanity.py")],
        ["python", str(root / "scripts" / "13_robustness_subsamples.py")],
        ["python", str(root / "scripts" / "14_robustness_within_day_shuffle.py")],
        ["python", str(root / "scripts" / "11_build_report.py")],
    ]

    for s in steps:
        run(s)

    print("OK: pipeline completed")


if __name__ == "__main__":
    sys.exit(main())
