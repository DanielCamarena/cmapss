from __future__ import annotations

from typing import Any, Dict, List


def extract_evidence_stub() -> Dict[str, Any]:
    # Placeholder for future PDF/image/table extraction integration.
    return {
        "evidence_items": [],
        "evidence_confidence": 0.0,
        "mode": "stub",
        "notes": "Multimodal extraction not enabled in current mock integration.",
    }


def summarize_evidence(evidence_items: List[Dict[str, Any]]) -> str:
    if not evidence_items:
        return "No external multimodal evidence attached."
    return f"{len(evidence_items)} evidence item(s) linked to this decision."
