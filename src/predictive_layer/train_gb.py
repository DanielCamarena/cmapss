from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .common import MODELS_DIR, TrainData, build_tabular_data, save_json


def train_gb(data: TrainData | None = None, seed: int = 42) -> pd.DataFrame:
    if data is None:
        data = build_tabular_data()

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        random_state=seed,
        loss="squared_error",
    )
    model.fit(data.x_train, data.y_train)
    pred = model.predict(data.x_valid).astype(np.float32)

    out = data.valid_meta.copy()
    out["y_true"] = data.y_valid
    out["y_pred"] = pred
    out["model_name"] = "gb"

    joblib.dump(model, MODELS_DIR / "gb_model.joblib")
    save_json(
        MODELS_DIR / "gb_metadata.json",
        {
            "model_name": "gb",
            "seed": seed,
            "feature_cols": data.feature_cols,
            "n_train": int(len(data.x_train)),
            "n_valid": int(len(data.x_valid)),
        },
    )
    out.to_parquet(Path("out/predictive_layer/02_valid_predictions_gb.parquet"), index=False)
    return out


if __name__ == "__main__":
    train_gb()


