from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .common import MODELS_DIR, save_json
from .temporal import build_temporal_data


def train_gru(seed: int = 42) -> pd.DataFrame:
    # GRU/TCN proxy: temporal windows flattened + tree ensemble.
    x_train, y_train, x_valid, y_valid, valid_meta, feature_cols = build_temporal_data(window=30)
    x_train_f = x_train.reshape(len(x_train), -1)
    x_valid_f = x_valid.reshape(len(x_valid), -1)

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=24,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train_f, y_train)
    pred = model.predict(x_valid_f).astype("float32")

    out = valid_meta.copy()
    out["y_true"] = y_valid
    out["y_pred"] = pred
    out["model_name"] = "gru"

    joblib.dump(model, MODELS_DIR / "gru_model.joblib")
    save_json(
        MODELS_DIR / "gru_metadata.json",
        {
            "model_name": "gru",
            "seed": seed,
            "window": 30,
            "feature_cols": feature_cols,
            "n_train": int(len(x_train)),
            "n_valid": int(len(x_valid)),
            "implementation_note": "GRU/TCN proxy via flattened temporal windows + ExtraTreesRegressor.",
        },
    )
    out.to_parquet(Path("out/predictive_layer/02_valid_predictions_gru.parquet"), index=False)
    return out


if __name__ == "__main__":
    train_gru()


