from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor

from event_impact_model.cv.purged import PurgedWalkForwardSplitter
from event_impact_model.utils.io import ensure_dir, write_parquet
from event_impact_model.utils.log import get_logger

log = get_logger("train_cv_lgbm")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    sr = spearmanr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    hit = accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
    return {"mse": mse, "r2": r2, "pearson": pr, "spearman": sr, "hit_rate": hit}


def main() -> None:
    processed = Path("data/processed")
    ensure_dir(processed)

    df = pd.read_parquet(processed / "model_dataset.parquet").copy()
    df["trade_date_aligned"] = pd.to_datetime(df["trade_date_aligned"])

    y = df["y_car_p1_p5"].to_numpy()

    num_cols = ["mom_1d", "mom_5d", "mom_20d", "vol_20d", "dollar_vol_20d", "dow", "month"]
    cat_cols = ["form", "session_bucket"]
    X = df[num_cols + cat_cols].copy()

    # Trees don't need scaling; just one-hot categoricals
    preproc = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )

    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1,
    )


    pipe = Pipeline([("pre", preproc), ("model", model)])

    splitter = PurgedWalkForwardSplitter(n_splits=5, label_horizon_days=7, embargo_days=5)
    splits = splitter.split(df["trade_date_aligned"])

    oos_rows = []
    fold_metrics = []

    for fold, train_mask, test_mask, (d0, d1) in splits:
        X_train, y_train = X.loc[train_mask], y[train_mask]
        X_test, y_test = X.loc[test_mask], y[test_mask]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        m.update({"fold": fold, "model": "lgbm", "test_start": d0, "test_end": d1, "n_test": len(y_test)})
        fold_metrics.append(m)

        test_idx = df.index[test_mask]
        for idx_i, yp in zip(test_idx, y_pred, strict=True):
            oos_rows.append(
                {
                    "event_id": df.loc[idx_i, "event_id"],
                    "ticker": df.loc[idx_i, "ticker"],
                    "date": df.loc[idx_i, "trade_date_aligned"].date(),
                    "y_true": float(df.loc[idx_i, "y_car_p1_p5"]),
                    "y_pred_lgbm": float(yp),
                    "fold": int(fold),
                }
            )

        log.info(
            f"Fold {fold} [{d0}..{d1}] lgbm: spearman={m['spearman']:.3f} "
            f"pearson={m['pearson']:.3f} hit={m['hit_rate']:.3f} mse={m['mse']:.6f}"
        )

    oos = pd.DataFrame(oos_rows).sort_values(["date", "event_id"])
    metrics = pd.DataFrame(fold_metrics)

    write_parquet(oos, processed / "oos_predictions_lgbm.parquet")
    metrics.to_csv(processed / "cv_metrics_lgbm.csv", index=False)

    log.info(f"Saved: data/processed/oos_predictions_lgbm.parquet rows={len(oos):,}")
    log.info(f"Saved: data/processed/cv_metrics_lgbm.csv rows={len(metrics):,}")

    if not metrics.empty:
        log.info("LGBM CV mean metrics:")
        for col in ["mse", "r2", "pearson", "spearman", "hit_rate"]:
            log.info(f"  {col}: {metrics[col].mean():.4f}")


if __name__ == "__main__":
    main()
