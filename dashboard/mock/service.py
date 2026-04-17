import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent
SCENARIOS_PATH = BASE_DIR / "scenarios.json"


class ValidationError(Exception):
    pass


def _load_scenarios() -> Dict[str, Any]:
    with SCENARIOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _risk_from_rul(rul_pred: float, critical_max: float, warning_max: float) -> str:
    if rul_pred <= critical_max:
        return "critical"
    if rul_pred <= warning_max:
        return "warning"
    return "healthy"


def _recommendation_from_risk(risk_level: str) -> str:
    if risk_level == "critical":
        return "Program inspection in the next maintenance window and reduce mission load."
    if risk_level == "warning":
        return "Increase monitoring frequency and schedule preventive inspection soon."
    return "Maintain normal operation and keep routine monitoring."


def _validate_payload(payload: Dict[str, Any], scenarios: Dict[str, Any]) -> None:
    required = ["dataset_id", "unit_id", "cycle", "op_settings", "sensors", "source"]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    dataset_id = payload["dataset_id"]
    if dataset_id not in scenarios["dataset_unit_limits"]:
        raise ValidationError(f"Invalid dataset_id: {dataset_id}")

    unit_id = payload["unit_id"]
    if not isinstance(unit_id, int) or unit_id < 1:
        raise ValidationError("unit_id must be an integer >= 1")

    max_unit = scenarios["dataset_unit_limits"][dataset_id]
    if unit_id > max_unit:
        raise ValidationError(
            f"Invalid unit_id for {dataset_id}. unit_id must be between 1 and {max_unit}"
        )

    cycle = payload["cycle"]
    if not isinstance(cycle, int) or cycle < 1 or cycle > 400:
        raise ValidationError("cycle must be an integer in range [1, 400]")

    op_settings = payload["op_settings"]
    if not isinstance(op_settings, list) or len(op_settings) != 3:
        raise ValidationError("op_settings must be a list of 3 numbers")
    if any((not isinstance(v, (int, float)) or v < -5.0 or v > 5.0) for v in op_settings):
        raise ValidationError("op_settings values must be in range [-5.0, 5.0]")

    sensors = payload["sensors"]
    if not isinstance(sensors, list) or len(sensors) != 21:
        raise ValidationError("sensors must be a list of 21 numbers")
    if any((not isinstance(v, (int, float)) or v < -500.0 or v > 500.0) for v in sensors):
        raise ValidationError("sensor values must be in range [-500.0, 500.0]")

    source = payload["source"]
    if source not in ("manual", "csv"):
        raise ValidationError("source must be either 'manual' or 'csv'")


def _simulate_rul(payload: Dict[str, Any]) -> float:
    dataset_penalty = {"FD001": 0.0, "FD002": 8.0, "FD003": 4.0, "FD004": 12.0}[payload["dataset_id"]]
    cycle = payload["cycle"]
    sensors = payload["sensors"]
    settings = payload["op_settings"]

    sensor_load = sum(abs(v) for v in sensors[:7]) / 7.0
    op_load = sum(abs(v) for v in settings) / 3.0

    # Mock equation: decreases with cycle, adds degradation pressure from first sensors.
    raw = 150.0 - (0.52 * cycle) - (0.04 * sensor_load) - (1.5 * op_load) - dataset_penalty
    noise = random.uniform(-3.0, 3.0)
    return max(0.0, raw + noise)


def _simulate_latency(scenarios: Dict[str, Any]) -> None:
    latency_ms = random.randint(scenarios["latency_ms"]["min"], scenarios["latency_ms"]["max"])
    time.sleep(latency_ms / 1000.0)


def _confidence_band(rul_pred: float) -> Dict[str, float]:
    width = 6.0 + min(18.0, rul_pred * 0.15)
    low = max(0.0, rul_pred - width)
    high = max(low, rul_pred + width)
    return {"low": round(low, 2), "high": round(high, 2)}


def generate_history(cycle: int, rul_pred: float, length: int = 30) -> List[Dict[str, float]]:
    length = max(10, min(length, cycle))
    start_cycle = cycle - length + 1
    points: List[Dict[str, float]] = []
    current = max(rul_pred + (length * 0.95), rul_pred + 5.0)

    for c in range(start_cycle, cycle + 1):
        decrement = 0.7 + abs(random.uniform(-0.25, 0.35))
        current = max(rul_pred, current - decrement)
        points.append({"cycle": c, "rul_est": round(current, 2)})

    # Ensure last point is exactly current prediction.
    points[-1]["rul_est"] = round(rul_pred, 2)
    return points


def predict(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = _load_scenarios()
    _validate_payload(input_payload, scenarios)
    _simulate_latency(scenarios)

    rul_pred = round(_simulate_rul(input_payload), 2)
    thresholds = scenarios["risk_thresholds"]
    risk_level = _risk_from_rul(
        rul_pred=rul_pred,
        critical_max=thresholds["critical_max"],
        warning_max=thresholds["warning_max"],
    )
    confidence_band = _confidence_band(rul_pred)
    recommendation_text = _recommendation_from_risk(risk_level)

    return {
        "rul_pred": rul_pred,
        "risk_level": risk_level,
        "confidence_band": confidence_band,
        "recommendation_text": recommendation_text,
        "model_version": "mock-rul-v1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
