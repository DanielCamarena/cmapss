from __future__ import annotations

from typing import Any, Dict

from .llm_client import LLMClientError, is_llm_enabled, interpret_comparison


def _deterministic_interpretation(comparison: Dict[str, Any]) -> Dict[str, str]:
    delta_rul = float(comparison.get("delta_rul", 0.0))
    delta_risk = float(comparison.get("delta_risk_score", 0.0))
    base_level = str(comparison.get("baseline_risk_level", "")).lower()
    scenario_level = str(comparison.get("scenario_risk_level", "")).lower()

    level_order = {"healthy": 0, "warning": 1, "critical": 2}
    base_rank = level_order.get(base_level, 1)
    scenario_rank = level_order.get(scenario_level, 1)

    favorable = delta_rul >= 3.0 and delta_risk <= -1.0 and scenario_rank <= base_rank
    unfavorable = delta_rul <= -3.0 and delta_risk >= 1.0 and scenario_rank >= base_rank

    if favorable:
        return {
            "impact_label": "favorable",
            "comparison_interpretation": (
                f"Scenario improves outlook: RUL changes by {delta_rul:+.2f} cycles and "
                f"risk score changes by {delta_risk:+.2f}."
            ),
            "operator_guidance": "Use this scenario as preferred operating profile and keep routine monitoring.",
        }

    if unfavorable:
        return {
            "impact_label": "unfavorable",
            "comparison_interpretation": (
                f"Scenario worsens outlook: RUL changes by {delta_rul:+.2f} cycles and "
                f"risk score changes by {delta_risk:+.2f}."
            ),
            "operator_guidance": "Avoid this scenario and schedule preventive inspection before applying similar conditions.",
        }

    return {
        "impact_label": "mixed",
        "comparison_interpretation": (
            f"Scenario impact is mixed: RUL changes by {delta_rul:+.2f} cycles and "
            f"risk score changes by {delta_risk:+.2f}."
        ),
        "operator_guidance": "Review trade-offs with engineering and run another constrained scenario before decision.",
    }


def build_comparison_interpretation(
    scenario_prompt: str,
    comparison: Dict[str, Any],
    assistant_mode: str,
) -> Dict[str, str]:
    deterministic = _deterministic_interpretation(comparison)

    if assistant_mode != "llm_enabled" or not is_llm_enabled():
        return deterministic

    try:
        llm_out = interpret_comparison(scenario_prompt=scenario_prompt, comparison=comparison)
        llm_model_used = llm_out.get("llm_model_used")
        impact = str(llm_out.get("impact_label", deterministic["impact_label"])).lower()
        if impact not in {"favorable", "mixed", "unfavorable"}:
            impact = deterministic["impact_label"]

        interpretation = str(
            llm_out.get("comparison_interpretation", deterministic["comparison_interpretation"])
        ).strip()
        guidance = str(llm_out.get("operator_guidance", deterministic["operator_guidance"])).strip()

        return {
            "impact_label": impact,
            "comparison_interpretation": interpretation or deterministic["comparison_interpretation"],
            "operator_guidance": guidance or deterministic["operator_guidance"],
            "llm_model_used": llm_model_used,
        }
    except LLMClientError:
        return deterministic
