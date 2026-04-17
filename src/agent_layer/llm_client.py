from __future__ import annotations

import json
import os
from typing import Any, Dict, List


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview"]


class LLMClientError(RuntimeError):
    """Raised when LLM integration fails."""


def is_llm_enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def current_mode_label() -> str:
    return "llm_enabled" if is_llm_enabled() else "rules_only"


def _candidate_models() -> List[str]:
    selected = os.getenv("GEMINI_MODEL", "").strip()
    models: List[str] = []
    if selected:
        models.append(selected)
    for model in FALLBACK_MODELS:
        if model not in models:
            models.append(model)
    return models


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _create_genai_client() -> Any:
    try:
        from google import genai
    except Exception as e:
        raise LLMClientError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from e

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    raise LLMClientError(
        "Missing Gemini API credentials. Set GEMINI_API_KEY or GOOGLE_API_KEY. "
        "For Vertex AI, configure GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, "
        "and GOOGLE_CLOUD_LOCATION."
    )


def _extract_text_from_response(response: Any) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts:
            maybe_text = str(getattr(parts[0], "text", "") or "").strip()
            if maybe_text:
                return maybe_text

    raise LLMClientError("Empty text response from Gemini.")


def _gemini_json_call(prompt: str, timeout_seconds: int = 8) -> tuple[Dict[str, Any], str]:
    del timeout_seconds  # SDK handles transport details.
    client = _create_genai_client()

    last_error = ""
    for model in _candidate_models():
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = _extract_text_from_response(response)
            return json.loads(_strip_code_fence(text)), model
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["404", "not found", "not available", "unsupported"]):
                last_error = f"model '{model}' unavailable"
                continue
            raise LLMClientError(f"Gemini request failed on model '{model}': {e}") from e

    configured = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    raise LLMClientError(
        "Gemini model unavailable. "
        f"Checked configured model '{configured}' and fallbacks {FALLBACK_MODELS}. "
        f"Last error: {last_error}. Update GEMINI_MODEL to a currently available model."
    )


def propose_scenario_patch(
    scenario_prompt: str,
    base_payload: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""
You are a PHM scenario assistant. Return ONLY JSON.
Goal: convert user what-if text into bounded numeric deltas.

User prompt:
{scenario_prompt}

Base payload:
{json.dumps(base_payload, ensure_ascii=True)}

Constraints:
{json.dumps(constraints, ensure_ascii=True)}

Output schema:
{{
  "cycle_delta": int,
  "op_setting_deltas": {{"1": float, "2": float, "3": float}},
  "sensor_deltas": {{"1": float, "2": float, ..., "21": float}},
  "assumptions": [str],
  "safety_notes": [str]
}}

Rules:
- If uncertain, keep deltas at 0.
- Never propose id changes.
- Keep values small and realistic.
""".strip()
    payload, model_used = _gemini_json_call(prompt)
    if isinstance(payload, dict):
        payload["llm_model_used"] = model_used
    return payload


def interpret_comparison(
    scenario_prompt: str,
    comparison: Dict[str, Any],
) -> Dict[str, str]:
    prompt = f"""
You are a maintenance decision explainer. Return ONLY JSON.

Scenario prompt:
{scenario_prompt}

Comparison payload:
{json.dumps(comparison, ensure_ascii=True)}

Output schema:
{{
  "impact_label": "favorable|mixed|unfavorable",
  "comparison_interpretation": "short paragraph for operator",
  "operator_guidance": "one practical next step"
}}

Keep it concise, operational, and safety-oriented.
""".strip()
    payload, model_used = _gemini_json_call(prompt)
    if isinstance(payload, dict):
        payload["llm_model_used"] = model_used
    return payload
