from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from dashboard.mock.service import ValidationError, generate_history

from .multimodal_extractor import extract_evidence_stub, summarize_evidence
from .recommender import build_recommendation
from .risk_engine import compute_risk_decision
from .tools import tool_dashboard_explainer, tool_read_model_output, tool_read_policy


AUDIT_DIR = Path(__file__).resolve().parents[2] / "out" / "agent_layer"
AUDIT_PATH = AUDIT_DIR / "audit_log.jsonl"


def _write_audit(record: Dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def orchestrate_prediction(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    model_output = tool_read_model_output(input_payload)
    history: List[Dict[str, float]] = generate_history(
        cycle=input_payload["cycle"], rul_pred=model_output["rul_pred"], length=30
    )

    policy = tool_read_policy()
    risk_decision = compute_risk_decision(
        rul_pred=model_output["rul_pred"],
        confidence_band=model_output["confidence_band"],
        history=history,
        thresholds=policy["risk_thresholds"],
    )
    recommendation = build_recommendation(
        risk_level=risk_decision["risk_level"],
        risk_score=risk_decision["risk_score"],
        uncertainty_score=risk_decision["uncertainty_score"],
    )
    evidence = extract_evidence_stub()

    audit_record_id = f"AR-{uuid4().hex[:10].upper()}"
    dashboard_note = tool_dashboard_explainer(
        risk_level=risk_decision["risk_level"], risk_score=risk_decision["risk_score"]
    )

    output = {
        **model_output,
        "risk_level": risk_decision["risk_level"],
        "risk_score": risk_decision["risk_score"],
        "recommendation_priority": recommendation["recommendation_priority"],
        "recommendation_text": recommendation["recommendation_text"],
        "recommendation_alternatives": recommendation["recommendation_alternatives"],
        "rationale": risk_decision["rationale"],
        "dashboard_note": dashboard_note,
        "audit_record_id": audit_record_id,
        "evidence_summary": summarize_evidence(evidence["evidence_items"]),
        "service_status": "ok",
    }

    _write_audit(
        {
            "audit_record_id": audit_record_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_payload": {
                "dataset_id": input_payload["dataset_id"],
                "unit_id": input_payload["unit_id"],
                "cycle": input_payload["cycle"],
                "source": input_payload["source"],
            },
            "output": {
                "rul_pred": output["rul_pred"],
                "risk_level": output["risk_level"],
                "risk_score": output["risk_score"],
                "recommendation_priority": output["recommendation_priority"],
            },
        }
    )
    output["history"] = history
    return output
