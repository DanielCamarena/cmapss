from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from .common import ROOT


OUT_PREDICTIVE_LAYER = ROOT / "out" / "predictive_layer"
MODELS_DIR = OUT_PREDICTIVE_LAYER / "models"


def _load_config() -> dict:
    champion = json.loads((OUT_PREDICTIVE_LAYER / "champion.json").read_text(encoding="utf-8"))
    normalizer = json.loads((OUT_PREDICTIVE_LAYER / "normalizer.json").read_text(encoding="utf-8"))
    calib = json.loads((OUT_PREDICTIVE_LAYER / "04_confidence_band_policy.json").read_text(encoding="utf-8"))
    return {"champion": champion["champion"], "normalizer": normalizer, "calib": calib}


def _to_feature_vector(payload: Dict[str, Any], feature_order: List[str], normalizer: dict) -> np.ndarray:
    raw = {
        "op_setting_1": float(payload["op_settings"][0]),
        "op_setting_2": float(payload["op_settings"][1]),
        "op_setting_3": float(payload["op_settings"][2]),
    }
    for i in range(1, 22):
        raw[f"sensor_{i}"] = float(payload["sensors"][i - 1])

    z = []
    for f in feature_order:
        mean = normalizer["means"][f]
        std = normalizer["stds"][f] if normalizer["stds"][f] != 0 else 1.0
        z.append((raw[f] - mean) / std)
    return np.array(z, dtype=np.float32)


def _predict_with_model(champion: str, x_vec: np.ndarray) -> float:
    if champion in {"rf", "gb"}:
        model = joblib.load(MODELS_DIR / f"{champion}_model.joblib")
        return float(model.predict(x_vec.reshape(1, -1))[0])

    # Temporal proxies are trained on flattened windows of len=30.
    if champion in {"lstm", "gru"}:
        model = joblib.load(MODELS_DIR / f"{champion}_model.joblib")
        seq_flat = np.tile(x_vec, 30).reshape(1, -1).astype(np.float32)
        return float(model.predict(seq_flat)[0])

    model = joblib.load(MODELS_DIR / "gb_model.joblib")
    return float(model.predict(x_vec.reshape(1, -1))[0])


def predict_rul(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_config()
    normalizer = cfg["normalizer"]
    feature_order = normalizer["feature_order"]
    x_vec = _to_feature_vector(payload, feature_order, normalizer)
    rul_pred = max(0.0, _predict_with_model(cfg["champion"], x_vec))

    residual_std = float(cfg["calib"]["residual_std"])
    width = max(6.0, min(30.0, residual_std * 1.6))
    low = max(0.0, rul_pred - width)
    high = max(low, rul_pred + width)

    return {
        "rul_pred": round(rul_pred, 2),
        "confidence_band": {"low": round(low, 2), "high": round(high, 2)},
        "model_version": f"predictive_layer-{cfg['champion']}-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service_status": "ok",
    }


