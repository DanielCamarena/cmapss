from __future__ import annotations

from typing import Any, Dict, List


def _band_width(confidence_band: Dict[str, float]) -> float:
    return max(0.0, confidence_band["high"] - confidence_band["low"])


def _norm_rul_risk(rul_pred: float) -> float:
    # 0 risk when RUL is high, 1 risk when near zero.
    return max(0.0, min(1.0, 1.0 - (rul_pred / 130.0)))


def _norm_uncertainty_risk(confidence_band: Dict[str, float]) -> float:
    width = _band_width(confidence_band)
    return max(0.0, min(1.0, width / 45.0))


def _norm_trend_risk(history: List[Dict[str, float]]) -> float:
    if len(history) < 2:
        return 0.0
    first = history[0]["rul_est"]
    last = history[-1]["rul_est"]
    decline = max(0.0, first - last)
    return max(0.0, min(1.0, decline / 35.0))


def _map_score_to_level(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 50:
        return "warning"
    return "healthy"


def compute_risk_decision(
    rul_pred: float,
    confidence_band: Dict[str, float],
    history: List[Dict[str, float]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    rationale: List[str] = []
    uncertainty = _norm_uncertainty_risk(confidence_band)
    rul_risk = _norm_rul_risk(rul_pred)
    trend_risk = _norm_trend_risk(history)

    weighted = (0.6 * rul_risk) + (0.25 * uncertainty) + (0.15 * trend_risk)
    risk_score = round(max(0.0, min(100.0, weighted * 100.0)), 2)
    risk_level = _map_score_to_level(risk_score)

    # Hard-stop consistency with PHM business rule.
    if rul_pred <= thresholds["critical_max"]:
        risk_level = "critical"
        risk_score = max(85.0, risk_score)
        rationale.append(
            f"Hard-stop rule activated: rul_pred <= {thresholds['critical_max']}."
        )
    elif rul_pred <= thresholds["warning_max"] and risk_level == "healthy":
        risk_level = "warning"
        risk_score = max(55.0, risk_score)
        rationale.append(
            f"Boundary rule activated: rul_pred <= {thresholds['warning_max']}."
        )

    if uncertainty >= 0.6:
        rationale.append("High uncertainty detected from confidence band width.")
    if trend_risk >= 0.55:
        rationale.append("Fast degradation trend detected in recent cycles.")

    rationale.append(
        f"Composite score from RUL risk ({rul_risk:.2f}), uncertainty ({uncertainty:.2f}), trend ({trend_risk:.2f})."
    )

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "rationale": rationale,
        "uncertainty_score": round(uncertainty, 3),
    }
