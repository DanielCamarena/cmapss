from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .common import MODELS_DIR, TrainData, build_tabular_data, save_json


def train_rf(data: TrainData | None = None, seed: int = 42) -> pd.DataFrame:
    if data is None:
        data = build_tabular_data()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(data.x_train, data.y_train)
    pred = model.predict(data.x_valid).astype(np.float32)

    out = data.valid_meta.copy()
    out["y_true"] = data.y_valid
    out["y_pred"] = pred
    out["model_name"] = "rf"

    joblib.dump(model, MODELS_DIR / "rf_model.joblib")
    save_json(
        MODELS_DIR / "rf_metadata.json",
        {
            "model_name": "rf",
            "seed": seed,
            "feature_cols": data.feature_cols,
            "n_train": int(len(data.x_train)),
            "n_valid": int(len(data.x_valid)),
        },
    )
    out.to_parquet(Path("out/predictive_layer/02_valid_predictions_rf.parquet"), index=False)
    return out


if __name__ == "__main__":
    train_rf()


