from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _default_constraints() -> Dict[str, Any]:
    return {
        "cycle_min": 1,
        "cycle_max": 400,
        "cycle_delta_max": 80,
        "op_setting_min": -5.0,
        "op_setting_max": 5.0,
        "op_setting_delta_max": 2.0,
        "sensor_min": -500.0,
        "sensor_max": 500.0,
        "sensor_delta_max": 40.0,
        "allow_id_changes": False,
    }


def _merge_constraints(user_constraints: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = _default_constraints()
    if user_constraints:
        cfg.update(user_constraints)
    return cfg


def _parse_signed_number(text: str) -> float | None:
    m = re.search(r"([+-]?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return float(m.group(1))


def _apply_cycle_rule(
    prompt_l: str,
    cycle: int,
    cfg: Dict[str, Any],
    assumptions: List[str],
) -> Tuple[int, bool]:
    changed = False
    next_cycle = cycle

    if "increase cycle" in prompt_l or "more cycles" in prompt_l or "later cycle" in prompt_l:
        n = _parse_signed_number(prompt_l) or 20.0
        delta = _clamp(abs(n), 1, cfg["cycle_delta_max"])
        next_cycle = int(_clamp(cycle + delta, cfg["cycle_min"], cfg["cycle_max"]))
        assumptions.append(f"Applied cycle increase by +{int(delta)} from prompt.")
        changed = True
    elif "decrease cycle" in prompt_l or "fewer cycles" in prompt_l or "earlier cycle" in prompt_l:
        n = _parse_signed_number(prompt_l) or 20.0
        delta = _clamp(abs(n), 1, cfg["cycle_delta_max"])
        next_cycle = int(_clamp(cycle - delta, cfg["cycle_min"], cfg["cycle_max"]))
        assumptions.append(f"Applied cycle decrease by -{int(delta)} from prompt.")
        changed = True
    elif any(k in prompt_l for k in ["degradation", "worse", "critical", "high risk"]):
        delta = 25
        next_cycle = int(_clamp(cycle + delta, cfg["cycle_min"], cfg["cycle_max"]))
        assumptions.append("No explicit cycle value found; used +25 cycles for degradation scenario.")
        changed = True

    return next_cycle, changed


def _apply_operational_rules(
    prompt_l: str,
    op_settings: List[float],
    sensors: List[float],
    cfg: Dict[str, Any],
    assumptions: List[str],
) -> Tuple[List[float], List[float], bool]:
    changed = False
    ops = list(op_settings)
    sens = list(sensors)

    if any(k in prompt_l for k in ["high load", "higher load", "stress", "harsh", "aggressive"]):
        ops[0] = _clamp(ops[0] + 0.6, cfg["op_setting_min"], cfg["op_setting_max"])
        ops[2] = _clamp(ops[2] + 0.4, cfg["op_setting_min"], cfg["op_setting_max"])
        sens[0] = _clamp(sens[0] + 8.0, cfg["sensor_min"], cfg["sensor_max"])
        sens[1] = _clamp(sens[1] - 5.0, cfg["sensor_min"], cfg["sensor_max"])
        assumptions.append("Applied high-load scenario defaults on op_settings and first sensors.")
        changed = True

    if any(k in prompt_l for k in ["reduce load", "lower load", "conservative", "safe mode"]):
        ops[0] = _clamp(ops[0] - 0.5, cfg["op_setting_min"], cfg["op_setting_max"])
        ops[2] = _clamp(ops[2] - 0.3, cfg["op_setting_min"], cfg["op_setting_max"])
        sens[0] = _clamp(sens[0] - 4.0, cfg["sensor_min"], cfg["sensor_max"])
        assumptions.append("Applied low-load scenario defaults.")
        changed = True

    # Direct command patterns: sensor_5 +10, op_setting_2 -0.3
    for m in re.finditer(r"(sensor_(\d{1,2})|op_setting_(\d))\s*([+-]\s*\d+(?:\.\d+)?)", prompt_l):
        field = m.group(1)
        idx = int(m.group(2) or m.group(3))
        raw_delta = float(m.group(4).replace(" ", ""))
        if field.startswith("sensor_") and 1 <= idx <= 21:
            delta = _clamp(raw_delta, -cfg["sensor_delta_max"], cfg["sensor_delta_max"])
            sens[idx - 1] = _clamp(sens[idx - 1] + delta, cfg["sensor_min"], cfg["sensor_max"])
            assumptions.append(f"Applied direct command: {field} {delta:+.2f}.")
            changed = True
        if field.startswith("op_setting_") and 1 <= idx <= 3:
            delta = _clamp(raw_delta, -cfg["op_setting_delta_max"], cfg["op_setting_delta_max"])
            ops[idx - 1] = _clamp(ops[idx - 1] + delta, cfg["op_setting_min"], cfg["op_setting_max"])
            assumptions.append(f"Applied direct command: {field} {delta:+.2f}.")
            changed = True

    return ops, sens, changed


def _diff_payload(base_payload: Dict[str, Any], proposed_payload: Dict[str, Any]) -> List[str]:
    changes: List[str] = []
    if base_payload["cycle"] != proposed_payload["cycle"]:
        changes.append(f"cycle: {base_payload['cycle']} -> {proposed_payload['cycle']}")

    for i, (b, p) in enumerate(zip(base_payload["op_settings"], proposed_payload["op_settings"]), start=1):
        if abs(float(b) - float(p)) > 1e-9:
            changes.append(f"op_setting_{i}: {float(b):.3f} -> {float(p):.3f}")

    for i, (b, p) in enumerate(zip(base_payload["sensors"], proposed_payload["sensors"]), start=1):
        if abs(float(b) - float(p)) > 1e-9:
            changes.append(f"sensor_{i}: {float(b):.3f} -> {float(p):.3f}")

    return changes


def propose_scenario(
    scenario_prompt: str,
    base_payload: Dict[str, Any],
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not scenario_prompt or not scenario_prompt.strip():
        raise ValueError("scenario_prompt must be a non-empty string")

    cfg = _merge_constraints(constraints)
    prompt_l = scenario_prompt.lower()
    assumptions: List[str] = []
    safety_notes: List[str] = []

    proposed = {
        "dataset_id": base_payload["dataset_id"],
        "unit_id": int(base_payload["unit_id"]),
        "cycle": int(base_payload["cycle"]),
        "op_settings": [float(v) for v in base_payload["op_settings"]],
        "sensors": [float(v) for v in base_payload["sensors"]],
        "source": "api",
    }

    # ID fields are locked by default.
    if not cfg.get("allow_id_changes", False):
        safety_notes.append("dataset_id and unit_id are locked by policy.")

    next_cycle, cycle_changed = _apply_cycle_rule(
        prompt_l, proposed["cycle"], cfg, assumptions
    )
    proposed["cycle"] = next_cycle

    next_ops, next_sens, op_changed = _apply_operational_rules(
        prompt_l, proposed["op_settings"], proposed["sensors"], cfg, assumptions
    )
    proposed["op_settings"] = next_ops
    proposed["sensors"] = next_sens

    if not cycle_changed and not op_changed:
        safety_notes.append("Prompt was ambiguous; no change applied.")
        assumptions.append("No deterministic scenario rule matched the prompt.")

    changes = _diff_payload(base_payload, proposed)
    if not changes:
        safety_notes.append("No payload fields changed.")

    return {
        "proposed_payload": proposed,
        "change_summary": changes,
        "assumptions": assumptions,
        "safety_notes": safety_notes,
        "service_status": "ok",
        "assistant_mode": "rules_only",
    }


def compare_decisions(base_result: Dict[str, Any], scenario_result: Dict[str, Any]) -> Dict[str, Any]:
    delta_rul = float(scenario_result["rul_pred"]) - float(base_result["rul_pred"])
    delta_risk = float(scenario_result["risk_score"]) - float(base_result["risk_score"])
    return {
        "baseline_rul": float(base_result["rul_pred"]),
        "scenario_rul": float(scenario_result["rul_pred"]),
        "delta_rul": round(delta_rul, 3),
        "baseline_risk_score": float(base_result["risk_score"]),
        "scenario_risk_score": float(scenario_result["risk_score"]),
        "delta_risk_score": round(delta_risk, 3),
        "baseline_risk_level": base_result["risk_level"],
        "scenario_risk_level": scenario_result["risk_level"],
        "baseline_recommendation": base_result.get("recommendation_text", ""),
        "scenario_recommendation": scenario_result.get("recommendation_text", ""),
    }

