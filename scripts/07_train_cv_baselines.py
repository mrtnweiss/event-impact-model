# scripts/07_train_cv_baselines.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from event_impact_model.cv.purged import PurgedWalkForwardSplitter
from event_impact_model.utils.io import ensure_dir, write_parquet
from event_impact_model.utils.log import get_logger

log = get_logger("train_cv_baselines")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # correlations (IC)
    pr = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    sr = spearmanr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan

    # sign accuracy
    hit = accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
    return {"mse": mse, "r2": r2, "pearson": pr, "spearman": sr, "hit_rate": hit}


def main() -> None:
    processed = Path("data/processed")
    ensure_dir(processed)

    df = pd.read_parquet(processed / "model_dataset.parquet").copy()
    df["trade_date_aligned"] = pd.to_datetime(df["trade_date_aligned"])

    # Target
    y = df["y_car_p1_p5"].to_numpy()

    # Features
    num_cols = ["mom_1d", "mom_5d", "mom_20d", "vol_20d", "dollar_vol_20d", "dow", "month"]
    cat_cols = ["form", "session_bucket"]

    X = df[num_cols + cat_cols].copy()

    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    ridge_model = Pipeline(
        [
            ("pre", preproc),
            ("model", Ridge(alpha=10.0, random_state=42)),
        ]
    )

    # We'll also train a logistic baseline on sign(y)
    logit_model = Pipeline(
        [
            ("pre", preproc),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )

    # CV settings (purged + embargo)
    splitter = PurgedWalkForwardSplitter(
        n_splits=5,
        label_horizon_days=7,  # approx for CAR[+1,+5]
        embargo_days=5,
    )

    splits = splitter.split(df["trade_date_aligned"])

    oos_rows = []
    fold_metrics = []

    for fold, train_mask, test_mask, (d0, d1) in splits:
        X_train, y_train = X.loc[train_mask], y[train_mask]
        X_test, y_test = X.loc[test_mask], y[test_mask]

        if len(y_train) < 200 or len(y_test) < 50:
            log.warning(
                f"Fold {fold}: too small train/test (train={len(y_train)}, test={len(y_test)}), skipping"
            )
            continue

        # Ridge
        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        m.update(
            {
                "fold": fold,
                "model": "ridge",
                "test_start": str(d0),
                "test_end": str(d1),
                "n_test": len(y_test),
            }
        )
        fold_metrics.append(m)

        # Store OOS predictions
        test_idx = df.index[test_mask]
        for idx_i, yp in zip(test_idx, y_pred, strict=True):
            oos_rows.append(
                {
                    "event_id": df.loc[idx_i, "event_id"],
                    "ticker": df.loc[idx_i, "ticker"],
                    "date": df.loc[idx_i, "trade_date_aligned"].date(),
                    "y_true": float(df.loc[idx_i, "y_car_p1_p5"]),
                    "y_pred_ridge": float(yp),
                    "fold": int(fold),
                }
            )

        # Logistic on sign
        yb_train = (y_train > 0).astype(int)
        yb_test = (y_test > 0).astype(int)
        logit_model.fit(X_train, yb_train)
        proba = logit_model.predict_proba(X_test)[:, 1]
        pred_class = (proba > 0.5).astype(int)
        acc = accuracy_score(yb_test, pred_class)

        fold_metrics.append(
            {
                "fold": fold,
                "model": "logit_sign",
                "test_start": str(d0),
                "test_end": str(d1),
                "n_test": len(y_test),
                "accuracy": float(acc),
                "pos_rate_pred": float(proba.mean()),
                "pos_rate_true": float(yb_test.mean()),
            }
        )

        log.info(
            f"Fold {fold} [{d0}..{d1}] ridge: spearman={m['spearman']:.3f} pearson={m['pearson']:.3f} hit={m['hit_rate']:.3f} mse={m['mse']:.6f}"
        )

    oos = pd.DataFrame(oos_rows).sort_values(["date", "event_id"])
    metrics = pd.DataFrame(fold_metrics)

    write_parquet(oos, processed / "oos_predictions.parquet")
    metrics.to_csv(processed / "cv_metrics.csv", index=False)

    log.info(f"Saved: data/processed/oos_predictions.parquet rows={len(oos):,}")
    log.info(f"Saved: data/processed/cv_metrics.csv rows={len(metrics):,}")

    # Aggregate metrics for ridge
    ridge_metrics = metrics[metrics["model"] == "ridge"].copy()
    if not ridge_metrics.empty:
        log.info("Ridge CV mean metrics:")
        for col in ["mse", "r2", "pearson", "spearman", "hit_rate"]:
            log.info(f"  {col}: {ridge_metrics[col].mean():.4f}")

    logit_metrics = metrics[metrics["model"] == "logit_sign"].copy()
    if not logit_metrics.empty:
        log.info(f"Logit sign mean accuracy: {logit_metrics['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
