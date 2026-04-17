from __future__ import annotations

from typing import Any, Dict, List


def _priority_from_risk(risk_level: str) -> str:
    if risk_level == "critical":
        return "urgent"
    if risk_level == "warning":
        return "high"
    return "low"


def build_recommendation(
    risk_level: str,
    risk_score: float,
    uncertainty_score: float,
) -> Dict[str, Any]:
    alternatives: List[str]
    if risk_level == "critical":
        main = "Program immediate inspection and reduce mission load until verification."
        alternatives = [
            "Execute borescope inspection in the next maintenance window.",
            "Apply conservative derating profile while diagnostics are pending.",
        ]
    elif risk_level == "warning":
        main = "Schedule preventive inspection soon and increase monitoring frequency."
        alternatives = [
            "Shorten maintenance interval for this unit.",
            "Trigger additional data capture on next operational cycles.",
        ]
    else:
        main = "Keep normal operation and continue routine health monitoring."
        alternatives = [
            "Maintain standard inspection cadence.",
            "Re-evaluate risk on next scheduled data update.",
        ]

    if uncertainty_score >= 0.6:
        alternatives.insert(0, "Collect additional evidence before high-impact action.")

    return {
        "recommendation_text": main,
        "recommendation_priority": _priority_from_risk(risk_level),
        "recommendation_alternatives": alternatives,
        "recommendation_summary": f"{risk_level.upper()} risk ({risk_score:.1f}/100).",
    }
