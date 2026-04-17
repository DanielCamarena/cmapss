from __future__ import annotations

import importlib
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from dashboard.mock.service import ValidationError, generate_history, predict


def _risk_score_from_rul(rul_pred: float) -> float:
    # Simple fallback scoring: lower RUL => higher risk.
    return round(max(0.0, min(100.0, (1.0 - (rul_pred / 130.0)) * 100.0)), 2)


def _priority_from_level(risk_level: str) -> str:
    if risk_level == "critical":
        return "urgent"
    if risk_level == "warning":
        return "high"
    return "low"


def _fallback_recommendation(risk_level: str) -> Dict[str, Any]:
    if risk_level == "critical":
        return {
            "recommendation_text": "Program immediate inspection and reduce mission load until verification.",
            "recommendation_alternatives": [
                "Execute borescope inspection in the next maintenance window.",
                "Apply conservative derating profile while diagnostics are pending.",
            ],
        }
    if risk_level == "warning":
        return {
            "recommendation_text": "Schedule preventive inspection soon and increase monitoring frequency.",
            "recommendation_alternatives": [
                "Shorten maintenance interval for this unit.",
                "Trigger additional data capture on next operational cycles.",
            ],
        }
    return {
        "recommendation_text": "Keep normal operation and continue routine health monitoring.",
        "recommendation_alternatives": [
            "Maintain standard inspection cadence.",
            "Re-evaluate risk on next scheduled data update.",
        ],
    }


def _run_mock_provider(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_output = predict(payload)
    history: List[Dict[str, float]] = generate_history(
        cycle=payload["cycle"], rul_pred=model_output["rul_pred"], length=30
    )
    risk_level = model_output["risk_level"]
    risk_score = _risk_score_from_rul(model_output["rul_pred"])
    rec = _fallback_recommendation(risk_level)

    return {
        **model_output,
        "risk_score": risk_score,
        "recommendation_priority": _priority_from_level(risk_level),
        "recommendation_text": rec["recommendation_text"],
        "recommendation_alternatives": rec["recommendation_alternatives"],
        "rationale": [
            "Fallback mock provider used (agent layer unavailable or failed).",
            f"RUL-based risk policy applied for risk_level={risk_level}.",
        ],
        "dashboard_note": "Dashboard is running with mock backend adapter.",
        "audit_record_id": f"FB-{uuid4().hex[:10].upper()}",
        "evidence_summary": "No external multimodal evidence attached.",
        "service_status": "fallback",
        "history": history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_prediction_with_adapter(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Preferred path: real/advanced agent layer orchestration if available.
    try:
        module = importlib.import_module("src.agent_layer.orchestrator")
        orchestrate_prediction = getattr(module, "orchestrate_prediction")
        return orchestrate_prediction(payload)
    except ValidationError:
        raise
    except Exception:
        # Any import/runtime failure in layer 2 should not break dashboard.
        return _run_mock_provider(payload)
