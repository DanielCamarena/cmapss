from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from agent_layer.orchestrator import orchestrate_prediction, orchestrate_scenario


ROOT = Path(__file__).resolve().parents[1]
OUT_AGENT = ROOT / "out" / "agent_layer"
OUT_DASH_LAYER = ROOT / "out" / "dashboard_layer"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _phase_1_contracts() -> None:
    input_contract = {
        "schema_version": "v1",
        "description": "Dashboard to agent layer raw ingress contract.",
        "fields": {
            "dataset_id": "str",
            "unit_id": "int>=1",
            "cycle": "int>=1",
            "op_settings": "list[float] len=3",
            "sensors": "list[float] len=21",
            "source": "manual|csv|api",
            "history": "optional[list[dict]]",
        },
    }
    output_contract = {
        "schema_version": "v1",
        "description": "Agent layer decision output for dashboard.",
        "fields": {
            "rul_pred": "float>=0",
            "confidence_band": {"low": "float>=0", "high": "float>=0"},
            "risk_level": "healthy|warning|critical",
            "risk_score": "float in [0,100]",
            "recommendation_text": "str",
            "recommendation_priority": "low|high|urgent",
            "recommendation_alternatives": "list[str]",
            "rationale": "list[str]",
            "assistant_mode": "optional[rules_only|llm_enabled]",
            "audit_record_id": "str",
            "service_status": "ok|degraded|fallback",
            "timestamp": "iso8601",
        },
    }
    scenario_contract = {
        "schema_version": "v1",
        "description": "Scenario assistant contract (Contract D).",
        "input": {
            "scenario_prompt": "str",
            "base_payload": "contract A payload",
            "constraints": "optional[dict]",
        },
        "output": {
            "proposed_payload": "contract A payload",
            "change_summary": "list[str]",
            "assumptions": "list[str]",
            "safety_notes": "list[str]",
            "service_status": "ok|degraded|fallback",
            "assistant_mode": "rules_only|llm_enabled",
            "baseline_result": "optional[contract C decision]",
            "scenario_result": "optional[contract C decision]",
            "comparison": "optional[dict]",
        },
    }
    examples = {
        "valid_input": {
            "dataset_id": "FD001",
            "unit_id": 1,
            "cycle": 75,
            "op_settings": [0.2, -0.1, 0.0],
            "sensors": [0.0] * 21,
            "source": "manual",
        },
        "invalid_input_missing_fields": {"dataset_id": "FD001"},
    }

    _write_json(OUT_AGENT / "01_input_contract_v1.json", input_contract)
    _write_json(OUT_AGENT / "01_output_contract_v1.json", output_contract)
    _write_json(OUT_AGENT / "01_scenario_contract_v1.json", scenario_contract)
    _write_json(OUT_AGENT / "01_contract_examples.json", examples)
    _write_text(
        OUT_AGENT / "01_policy_rules.txt",
        "\n".join(
            [
                "Policy rules",
                "============",
                "1) Never hide uncertainty from confidence_band.",
                "2) Never emit high-impact recommendation without rationale.",
                "3) If tools or LLM fail, fallback to deterministic rules.",
                "4) Preserve audit trail for each decision.",
                "5) Scenario assistant must output explicit payload diff.",
            ]
        ),
    )
    rec_catalog = pd.DataFrame(
        [
            {"risk_level": "healthy", "priority": "low", "action": "monitor_only"},
            {"risk_level": "warning", "priority": "high", "action": "schedule_inspection"},
            {"risk_level": "critical", "priority": "urgent", "action": "immediate_inspection"},
        ]
    )
    rec_catalog.to_csv(OUT_AGENT / "01_recommendation_catalog.csv", index=False)


def _phase_2_risk_files() -> None:
    _write_text(
        OUT_AGENT / "02_risk_scoring_design.txt",
        "\n".join(
            [
                "Risk scoring design",
                "===================",
                "Score = 0.6 * rul_risk + 0.25 * uncertainty + 0.15 * trend",
                "Hard-stop mapping by RUL:",
                "- critical if rul_pred <= 20",
                "- warning if 20 < rul_pred <= 60",
                "- healthy if rul_pred > 60",
                "Score thresholds:",
                "- warning at score >= 50",
                "- critical at score >= 80",
            ]
        ),
    )
    _write_json(
        OUT_AGENT / "02_thresholds_config.json",
        {
            "critical_max": 20.0,
            "warning_max": 60.0,
            "score_warning_min": 50.0,
            "score_critical_min": 80.0,
        },
    )


def _phase_3_toolchain_file() -> None:
    _write_text(
        OUT_AGENT / "03_toolchain_contracts.txt",
        "\n".join(
            [
                "Toolchain contracts",
                "===================",
                "Pipeline: ingest -> validate -> get_predictive_output -> score -> recommend -> audit",
                "Primary model provider: src.predictive_layer.inference_service.predict_rul",
                "Fallback behavior: return degraded service status (no synthetic prediction).",
                "Scenario pipeline: prompt -> propose_payload -> validate -> rerun -> compare",
                "Policy source priority:",
                "1) out/agent_layer/02_thresholds_config.json",
                "2) internal defaults",
            ]
        ),
    )


def _phase_4_tool_use_and_modes() -> None:
    _write_text(
        OUT_AGENT / "04_tool_use_policy.txt",
        "\n".join(
            [
                "Tool Use and Assistant Mode Policy",
                "==================================",
                "Primary scenario path: deterministic parser + bounded payload edits.",
                "If GEMINI_API_KEY is configured: enable llm_enabled mode for scenario enrichment.",
                "If LLM request fails: fallback to rules_only and keep service_status=fallback.",
                "dataset_id and unit_id remain locked unless constraints explicitly allow changes.",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "04_llm_usage_scope.txt",
        "\n".join(
            [
                "LLM usage scope",
                "===============",
                "1) Scenario prompt enrichment (optional).",
                "2) Scenario comparison interpretation (optional).",
                "3) Core risk scoring remains deterministic.",
            ]
        ),
    )


def _phase_5_recommendation_files() -> None:
    _write_text(
        OUT_AGENT / "05_recommendation_templates.txt",
        "\n".join(
            [
                "Recommendation templates",
                "========================",
                "critical: Program immediate inspection and reduce mission load.",
                "warning: Schedule preventive inspection and increase monitoring frequency.",
                "healthy: Keep normal operation and routine monitoring.",
            ]
        ),
    )
    pd.DataFrame(
        [
            {"risk_level": "critical", "recommendation_priority": "urgent"},
            {"risk_level": "warning", "recommendation_priority": "high"},
            {"risk_level": "healthy", "recommendation_priority": "low"},
        ]
    ).to_csv(OUT_AGENT / "05_priority_matrix.csv", index=False)


def _phase_6_dashboard_mapping_files() -> None:
    _write_text(
        OUT_AGENT / "06_dashboard_mapping.txt",
        "\n".join(
            [
                "Dashboard mapping",
                "=================",
                "Resumen tab: rul_pred, risk_level, risk_score, confidence_band",
                "Detalle tab: input payload + rationale + model_version + service_status",
                "Historico tab: history trend by cycle/rul_est",
                "Recomendaciones tab: recommendation_text, alternatives, priority, evidence_summary, audit_record_id",
                "Escenarios tab: scenario_prompt, change_summary, baseline vs scenario comparison",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "06_explainability_checklist.txt",
        "\n".join(
            [
                "[x] rationale list is present",
                "[x] uncertainty is reflected via confidence_band",
                "[x] recommendation has explicit priority",
                "[x] audit_record_id present in output",
                "[x] service_status present in output",
            ]
        ),
    )
    _write_text(
        OUT_DASH_LAYER / "contract_integration_checklist.txt",
        "\n".join(
            [
                "Dashboard contract integration checklist",
                "=======================================",
                "[x] backend_adapter calls agent_layer.orchestrate_prediction",
                "[x] degraded path exists if layer 2 fails",
                "[x] tabs render risk/recommendation/rationale/audit data",
                "[x] scenario assistant contract available for dashboard integration",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "06_scenario_assistant_flow.txt",
        "\n".join(
            [
                "Scenario assistant flow",
                "=======================",
                "1) User writes scenario prompt in dashboard.",
                "2) Agent layer proposes payload changes with guardrails.",
                "3) User confirms scenario payload.",
                "4) Baseline and scenario predictions are executed.",
                "5) Dashboard displays delta in RUL/risk/recommendation.",
            ]
        ),
    )
    _write_json(
        OUT_AGENT / "06_comparison_payload_schema.json",
        {
            "baseline_result": "contract C payload",
            "scenario_result": "contract C payload",
            "comparison": {
                "delta_rul": "float",
                "delta_risk_score": "float",
                "baseline_risk_level": "str",
                "scenario_risk_level": "str",
            },
        },
    )


def _phase_7_llm_files() -> None:
    _write_text(
        OUT_AGENT / "07_engine_decision_record.txt",
        "\n".join(
            [
                "Engine decision record",
                "======================",
                "Base decision engine: deterministic rules + score.",
                "Optional LLM: text enhancement only (recommendation wording, rationale summary).",
                "Optional LLM: scenario prompt understanding with contract D guardrails.",
                "Business risk classification remains deterministic.",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "07_llm_integration_policy.txt",
        "\n".join(
            [
                "LLM integration policy",
                "======================",
                "1) Use GEMINI_API_KEY via environment variable only.",
                "2) Timeout and retry limits must be configured.",
                "3) If LLM fails, return deterministic output and service_status=fallback.",
                "4) Never block dashboard response waiting for non-critical LLM output.",
                "5) Scenario proposals must pass payload validation before rerun.",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "07_nonfunctional_requirements.txt",
        "\n".join(
            [
                "Non-functional requirements",
                "===========================",
                "- P95 decision latency under 2 seconds for dashboard interactions.",
                "- Full audit trace per prediction.",
                "- Contract stability across minor releases.",
                "- Scenario assistant response under 3 seconds in rules mode.",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "07_scenario_assistant_policy.txt",
        "\n".join(
            [
                "Scenario assistant policy",
                "=========================",
                "1) Use deterministic rules by default (assistant_mode=rules_only).",
                "2) Do not mutate dataset_id/unit_id unless explicitly allowed by constraints.",
                "3) Clamp all numeric changes to configured ranges.",
                "4) Always return change_summary and safety_notes.",
                "5) If LLM parsing fails, fall back to rules mode.",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "07_prompt_templates_scenarios.txt",
        "\n".join(
            [
                "Scenario prompt templates",
                "=========================",
                "Template 1: 'Increase cycle by 30 and apply high load profile.'",
                "Template 2: 'Reduce load conservatively and rerun prognosis.'",
                "Template 3: 'Apply sensor_5 +12 and op_setting_2 -0.4, compare with baseline.'",
            ]
        ),
    )


def _phase_8_test_files() -> None:
    _write_text(
        OUT_AGENT / "08_test_matrix.txt",
        "\n".join(
            [
                "Test matrix",
                "===========",
                "Nominal cases: healthy, warning, critical",
                "Adverse cases: missing fields, out-of-range values, provider unavailable",
                "Integration cases: dashboard adapter path + degraded path",
                "Scenario cases: ambiguous prompt, high-load prompt, direct sensor command prompt",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "08_failure_modes.txt",
        "\n".join(
            [
                "Failure modes",
                "=============",
                "1) predictive layer unavailable -> service degraded",
                "2) invalid input -> validation error",
                "3) recommendation generation failure -> degraded recommendation fallback",
                "4) scenario prompt ambiguous -> no-change proposal with safety notes",
            ]
        ),
    )
    _write_text(
        OUT_AGENT / "08_acceptance_criteria.txt",
        "\n".join(
            [
                "Acceptance criteria",
                "===================",
                "[ ] Agent layer produces contract-compliant output.",
                "[ ] Risk hard-stops follow RUL thresholds 20/60.",
                "[ ] Dashboard renders decision fields without crash.",
                "[ ] Degraded mode preserves operability.",
                "[ ] Scenario assistant returns valid proposed_payload and comparison.",
            ]
        ),
    )


def _smoke_test() -> None:
    sample_csv = ROOT / "dashboard" / "mock" / "sample_input.csv"
    df = pd.read_csv(sample_csv)
    row = df.iloc[0]
    payload = {
        "dataset_id": str(row["dataset_id"]),
        "unit_id": int(row["unit_id"]),
        "cycle": int(row["cycle"]),
        "op_settings": [float(row[f"op_setting_{i}"]) for i in range(1, 4)],
        "sensors": [float(row[f"sensor_{i}"]) for i in range(1, 22)],
        "source": "csv",
    }
    out = orchestrate_prediction(payload)
    _write_text(
        OUT_AGENT / "03_smoke_orchestrator.txt",
        "Agent orchestrator smoke output\n=============================\n"
        + json.dumps(
            {
                "risk_level": out["risk_level"],
                "risk_score": out["risk_score"],
                "recommendation_priority": out["recommendation_priority"],
                "service_status": out["service_status"],
                "model_version": out["model_version"],
            },
            indent=2,
        ),
    )
    scenario_out = orchestrate_scenario(
        scenario_prompt="Increase cycle by 25 and apply high load profile.",
        base_payload=payload,
    )
    _write_text(
        OUT_AGENT / "03_smoke_scenario_assistant.txt",
        "Scenario assistant smoke output\n==============================\n"
        + json.dumps(
            {
                "assistant_mode": scenario_out["assistant_mode"],
                "service_status": scenario_out["service_status"],
                "changes": scenario_out["change_summary"][:5],
                "delta_rul": scenario_out["comparison"]["delta_rul"],
                "delta_risk_score": scenario_out["comparison"]["delta_risk_score"],
            },
            indent=2,
        ),
    )


def main() -> None:
    OUT_AGENT.mkdir(parents=True, exist_ok=True)
    OUT_DASH_LAYER.mkdir(parents=True, exist_ok=True)

    _phase_1_contracts()
    _phase_2_risk_files()
    _phase_3_toolchain_file()
    _phase_4_tool_use_and_modes()
    _phase_5_recommendation_files()
    _phase_6_dashboard_mapping_files()
    _phase_7_llm_files()
    _phase_8_test_files()
    _smoke_test()

    print("Plan 4 execution completed.")
    print(f"Artifacts: {OUT_AGENT}")


if __name__ == "__main__":
    main()
