from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
OUT_PREDICTIVE_LAYER = ROOT / "out" / "predictive_layer"
MODELS_DIR = OUT_PREDICTIVE_LAYER / "models"
OUT_PROCESSED = ROOT / "out" / "processed"
OUT_EDA = ROOT / "out" / "eda"


def ensure_dirs() -> None:
    OUT_PREDICTIVE_LAYER.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def read_selected_features() -> Tuple[List[str], List[str]]:
    path = OUT_EDA / "05_preprocessing_config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    selected = cfg["selected_features"]
    selected_z = [f"{c}_z" for c in selected]
    return selected, selected_z


def load_train_valid_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(OUT_PROCESSED / "train_processed.parquet")
    valid_df = pd.read_parquet(OUT_PROCESSED / "valid_processed.parquet")
    test_df = pd.read_parquet(OUT_PROCESSED / "test_processed.parquet")
    return train_df, valid_df, test_df


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def metrics_frame(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def by_dataset_metrics(df: pd.DataFrame, pred_col: str = "y_pred") -> pd.DataFrame:
    rows = []
    for ds, d in df.groupby("dataset"):
        m = metrics_frame(d["y_true"].values, d[pred_col].values)
        rows.append({"dataset": ds, **m, "count": int(len(d))})
    return pd.DataFrame(rows)


def by_rul_band_metrics(df: pd.DataFrame, pred_col: str = "y_pred") -> pd.DataFrame:
    d = df.copy()
    d["band"] = np.where(d["y_true"] <= 20, "0-20", np.where(d["y_true"] <= 60, "21-60", ">60"))
    rows = []
    for band, b in d.groupby("band"):
        m = metrics_frame(b["y_true"].values, b[pred_col].values)
        rows.append({"band": band, **m, "count": int(len(b))})
    order = {"0-20": 0, "21-60": 1, ">60": 2}
    return pd.DataFrame(rows).sort_values(by="band", key=lambda s: s.map(order))


def evaluate_latency(predict_fn, x: np.ndarray, n_runs: int = 5) -> Dict[str, float]:
    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(x)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(timings)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p95": float(np.quantile(arr, 0.95)),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
    }


def write_decisions_and_contract() -> None:
    (OUT_PREDICTIVE_LAYER / "01_modeling_decisions_frozen.txt").write_text(
        "\n".join(
            [
                "Modeling decisions frozen",
                "=========================",
                "Primary target: target_rul_capped",
                "Secondary target: target_rul_linear",
                "Cap value: 130",
                "Split: by unit_id within each dataset; no temporal leakage.",
                "Model family comparison: RF, GB, LSTM, GRU.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (OUT_PREDICTIVE_LAYER / "01_model_comparison_protocol.txt").write_text(
        "\n".join(
            [
                "Comparison protocol",
                "===================",
                "1) Train all candidates on train_processed",
                "2) Evaluate on valid_processed",
                "3) Compare RMSE/MAE global + by dataset + by RUL band",
                "4) Add latency metrics",
                "5) Select champion by RMSE global with stability tie-breaker",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_contract_files() -> None:
    input_schema = {
        "schema_version": "v1",
        "fields": {
            "dataset_id": "str",
            "unit_id": "int>=1",
            "cycle": "int>=1",
            "op_settings": "list[float] len=3",
            "sensors": "list[float] len=21",
            "source": "manual|csv|api",
        },
    }
    output_schema = {
        "schema_version": "v1",
        "fields": {
            "rul_pred": "float>=0",
            "confidence_band": {"low": "float>=0", "high": "float>=0"},
            "model_version": "str",
            "timestamp": "iso8601",
            "service_status": "ok|degraded|fallback",
        },
    }
    error_contract = {
        "schema_version": "v1",
        "errors": [
            {"code": "VALIDATION_ERROR", "status": "degraded"},
            {"code": "MODEL_ERROR", "status": "fallback"},
            {"code": "UNKNOWN_ERROR", "status": "fallback"},
        ],
    }
    examples = {
        "valid_input_example": {
            "dataset_id": "FD001",
            "unit_id": 1,
            "cycle": 50,
            "op_settings": [0.1, 0.0, -0.1],
            "sensors": [0.0] * 21,
            "source": "manual",
        },
        "valid_output_example": {
            "rul_pred": 72.5,
            "confidence_band": {"low": 62.1, "high": 82.9},
            "model_version": "predictive_layer-champion-v1",
            "timestamp": "2026-04-17T00:00:00Z",
            "service_status": "ok",
        },
    }
    save_json(OUT_PREDICTIVE_LAYER / "05_input_schema_v1.json", input_schema)
    save_json(OUT_PREDICTIVE_LAYER / "05_output_schema_v1.json", output_schema)
    save_json(OUT_PREDICTIVE_LAYER / "05_error_contract_v1.json", error_contract)
    save_json(OUT_PREDICTIVE_LAYER / "05_contract_examples.json", examples)


@dataclass
class TrainData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    valid_meta: pd.DataFrame
    feature_cols: List[str]


def build_tabular_data(target_col: str = "target_rul_capped") -> TrainData:
    train_df, valid_df, _ = load_train_valid_test()
    _, feature_cols_z = read_selected_features()
    x_train = train_df[feature_cols_z].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    x_valid = valid_df[feature_cols_z].values.astype(np.float32)
    y_valid = valid_df[target_col].values.astype(np.float32)
    valid_meta = valid_df[["dataset", "unit_id", "cycle"]].copy()
    return TrainData(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        valid_meta=valid_meta,
        feature_cols=feature_cols_z,
    )


