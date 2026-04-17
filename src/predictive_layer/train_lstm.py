from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .common import MODELS_DIR, save_json
from .temporal import build_temporal_data


def train_lstm(seed: int = 42) -> pd.DataFrame:
    # LSTM proxy: temporal windows flattened + nonlinear regressor.
    x_train, y_train, x_valid, y_valid, valid_meta, feature_cols = build_temporal_data(window=30)
    x_train_f = x_train.reshape(len(x_train), -1)
    x_valid_f = x_valid.reshape(len(x_valid), -1)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=40,
        random_state=seed,
    )
    model.fit(x_train_f, y_train)
    pred = model.predict(x_valid_f).astype("float32")

    out = valid_meta.copy()
    out["y_true"] = y_valid
    out["y_pred"] = pred
    out["model_name"] = "lstm"

    joblib.dump(model, MODELS_DIR / "lstm_model.joblib")
    save_json(
        MODELS_DIR / "lstm_metadata.json",
        {
            "model_name": "lstm",
            "seed": seed,
            "window": 30,
            "feature_cols": feature_cols,
            "n_train": int(len(x_train)),
            "n_valid": int(len(x_valid)),
            "implementation_note": "LSTM proxy via flattened temporal windows + MLPRegressor.",
        },
    )
    out.to_parquet(Path("out/predictive_layer/02_valid_predictions_lstm.parquet"), index=False)
    return out


if __name__ == "__main__":
    train_lstm()


