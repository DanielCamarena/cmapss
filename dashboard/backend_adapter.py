from __future__ import annotations

import importlib
from typing import Any, Dict

from dashboard.errors import ServiceUnavailableError, ValidationError


def run_prediction_with_adapter(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Preferred path: real/advanced agent layer orchestration.
    try:
        try:
            module = importlib.import_module("src.agent_layer.orchestrator")
        except ModuleNotFoundError:
            module = importlib.import_module("agent_layer.orchestrator")
        orchestrate_prediction = getattr(module, "orchestrate_prediction")
        return orchestrate_prediction(payload)
    except ValueError as e:
        # Map agent-layer validation errors to dashboard validation flow.
        raise ValidationError(str(e))
    except Exception as e:
        raise ServiceUnavailableError(f"Agent layer unavailable: {e}") from e
