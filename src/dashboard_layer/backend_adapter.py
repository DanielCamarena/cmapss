from __future__ import annotations

import importlib
from typing import Any, Dict

from .errors import ServiceUnavailableError, ValidationError


def run_prediction_with_adapter(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Preferred path: agent layer orchestration.
    try:
        try:
            module = importlib.import_module("src.agent_layer.orchestrator")
        except ModuleNotFoundError:
            module = importlib.import_module("agent_layer.orchestrator")
        orchestrate_prediction = getattr(module, "orchestrate_prediction")
        return orchestrate_prediction(payload)
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        raise ServiceUnavailableError(f"Agent layer unavailable: {e}") from e


def run_scenario_with_adapter(
    scenario_prompt: str,
    base_payload: Dict[str, Any],
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        try:
            module = importlib.import_module("src.agent_layer.orchestrator")
        except ModuleNotFoundError:
            module = importlib.import_module("agent_layer.orchestrator")
        orchestrate_scenario = getattr(module, "orchestrate_scenario")
        return orchestrate_scenario(
            scenario_prompt=scenario_prompt,
            base_payload=base_payload,
            constraints=constraints,
        )
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        raise ServiceUnavailableError(f"Scenario assistant unavailable: {e}") from e
