from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_RESEARCH = ROOT / "out" / "research"
OUT_EDA = ROOT / "out" / "eda"
DATA_README = ROOT / "data" / "readme.txt"


def ensure_dirs() -> None:
    OUT_RESEARCH.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, fallback: str = "") -> str:
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def load_eda_context() -> dict:
    target_def = read_text(OUT_EDA / "04_target_definition.txt", "Target definition unavailable.")
    findings = read_text(OUT_EDA / "06_findings_summary.md", "Findings unavailable.")
    risks = read_text(OUT_EDA / "06_risks_and_actions.md", "Risks unavailable.")
    preproc_cfg_path = OUT_EDA / "05_preprocessing_config.json"
    preproc_cfg = {}
    if preproc_cfg_path.exists():
        preproc_cfg = json.loads(preproc_cfg_path.read_text(encoding="utf-8"))
    return {
        "target_definition": target_def,
        "findings": findings,
        "risks": risks,
        "preprocessing": preproc_cfg,
        "data_readme": read_text(DATA_README, "Dataset readme unavailable."),
    }


def phase0_template() -> None:
    content = dedent(
        """
        FICHA DE REFERENCIA - PLANTILLA
        ===============================
        1) Fuente / Documento:
        2) Problema que aborda:
        3) Supuestos principales:
        4) Variables o señales clave:
        5) Metodo/modelo:
        6) Metricas de evaluacion:
        7) Limitaciones:
        8) Relevancia para capa RUL del proyecto:
        9) Decisiones que sugiere para implementacion:
        """
    )
    write_text(OUT_RESEARCH / "00_template_ficha_referencia.txt", content)


def phase1_doc_cards(ctx: dict) -> None:
    nasa = dedent(
        f"""
        FICHA - NASA_CMAPSS
        ===================
        Documento: doc/NASA_CMAPSS.pdf
        Problema: benchmark para pronostico de vida util remanente (RUL) en motores turbofan.
        Supuestos:
        - Motores de una misma flota con variaciones iniciales.
        - Diferentes condiciones operativas segun subset FD001-FD004.
        - Trayectorias train hasta falla, test truncadas antes de falla.
        Variables clave:
        - unit_id, cycle, 3 settings operativas, 21 sensores.
        Metricas sugeridas:
        - RMSE/MAE de RUL por subset y global.
        Limitaciones:
        - Dataset simulado; requiere cautela para extrapolacion industrial directa.
        Implicancias para este proyecto:
        - Justifica enfoque por series temporales multivariadas.
        - Requiere validacion por dataset para controlar heterogeneidad.

        Extracto de contexto del proyecto:
        {ctx["target_definition"]}
        """
    )
    write_text(OUT_RESEARCH / "01_ficha_nasa_cmapss.txt", nasa)

    damage = dedent(
        """
        FICHA - DAMAGE PROPAGATION MODELING
        ===================================
        Documento: doc/Damage Propagation Modeling.pdf
        Problema: modelar evolucion de dano/degradacion hasta falla.
        Aporte principal:
        - Marco para relacionar degradacion progresiva y estimacion de vida remanente.
        Implicancias:
        - Conveniente incorporar restricciones fisicas o reglas de monotonia en RUL.
        - Riesgo de sobreajuste si solo se usan modelos black-box sin validacion por regimen.
        Decision sugerida:
        - Iniciar con baseline data-driven y reservar etapa de mejora physics-guided.
        """
    )
    write_text(OUT_RESEARCH / "01_ficha_damage_propagation.txt", damage)

    ramasso = dedent(
        """
        FICHA - RAMASSO2014
        ===================
        Documento: doc/Ramasso2014.pdf
        Problema: enfoques PHM para prognostico, diagnostico y manejo de incertidumbre.
        Aporte principal:
        - Comparacion de estrategias de inferencia para salud/degradacion.
        Implicancias:
        - Importante reportar incertidumbre (bandas de confianza) junto con RUL.
        - Recomendable separar capa de prediccion y capa de decision/riesgo.
        Decision sugerida:
        - Adoptar arquitectura por capas con contratos claros (RUL -> riesgo -> dashboard).
        """
    )
    write_text(OUT_RESEARCH / "01_ficha_ramasso2014.txt", ramasso)


def phase2_methodology(ctx: dict) -> None:
    rows = [
        {
            "model_family": "Gradient Boosting (tabular baseline)",
            "input_representation": "features agregadas por ventana y estado actual",
            "strengths": "rapido, interpretable relativo, buen baseline",
            "weaknesses": "pierde dinamica temporal fina",
            "complexity": "low",
            "priority": 1,
        },
        {
            "model_family": "LSTM/GRU",
            "input_representation": "secuencias por unit_id",
            "strengths": "captura dependencia temporal",
            "weaknesses": "mas costo de entrenamiento",
            "complexity": "medium",
            "priority": 2,
        },
        {
            "model_family": "TCN",
            "input_representation": "secuencias convolucionales temporales",
            "strengths": "estable y eficiente para series",
            "weaknesses": "ajuste de hiperparametros",
            "complexity": "medium",
            "priority": 3,
        },
        {
            "model_family": "Physics-guided hybrid",
            "input_representation": "secuencia + restricciones",
            "strengths": "mejor coherencia fisica",
            "weaknesses": "mayor complejidad de implementacion",
            "complexity": "high",
            "priority": 4,
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_RESEARCH / "02_model_matrix.csv", index=False)

    method = dedent(
        f"""
        METODOLOGIA PROPUESTA (CAPA 1)
        ==============================
        Estrategia seleccionada:
        - Baseline principal: Gradient Boosting sobre features tabulares derivadas de secuencias.
        - Baseline temporal secundario: LSTM/GRU para capturar dinamica por ciclo.
        - Mejora futura: enfoque hibrido physics-guided.

        Target:
        - Primario: RUL capped (cap=130) segun salida de plan 1.
        - Secundario: RUL linear para analisis.

        Criterio de seleccion de modelo:
        1) Error (RMSE/MAE)
        2) Estabilidad por subsets FD001-FD004
        3) Latencia de inferencia para uso en dashboard
        4) Facilidad de mantenimiento/versionado

        Contexto de hallazgos EDA:
        {ctx["findings"]}
        """
    )
    write_text(OUT_RESEARCH / "02_metodologia_propuesta.txt", method)


def phase3_architecture_contract(ctx: dict) -> None:
    api_contract = dedent(
        """
        API CONTRACT - RUL LAYER
        ========================
        Input payload:
        - dataset_id: str
        - unit_id: int
        - cycle: int
        - op_settings: list[float] length=3
        - sensors: list[float] length=21
        - source: manual|csv|api

        Output payload:
        - rul_pred: float
        - confidence_band: {low: float, high: float}
        - model_version: str
        - timestamp: iso8601
        - service_status: ok|degraded|fallback
        - optional_diagnostics: object

        Contract rules:
        - Output siempre presente aun en modo fallback.
        - Si error recuperable: retornar fallback con mensaje diagnostico.
        """
    )
    write_text(OUT_RESEARCH / "03_api_contract_rul_layer.txt", api_contract)

    pipeline = dedent(
        """
        INFERENCE PIPELINE - RUL LAYER
        ==============================
        1) Input validation
           - tipos/rangos, campos requeridos
        2) Feature transformation
           - z-score con estadisticos de entrenamiento
           - armado de ventana secuencial (si aplica)
        3) Model inference
           - baseline tabular o temporal
        4) Postprocessing
           - clip de RUL >= 0
           - banda de confianza
        5) Output packaging
           - contrato estandar para capa 2/capa 3
        """
    )
    write_text(OUT_RESEARCH / "03_inference_pipeline.txt", pipeline)

    error_policy = dedent(
        """
        ERROR POLICY - RUL LAYER
        ========================
        - ValidationError: input invalido -> mensaje explicito + status=degraded
        - ModelError: falla de inferencia -> fallback deterministico + status=fallback
        - UnknownError: log completo + respuesta segura para no romper dashboard
        - Todo evento debe quedar trazado con timestamp y request id.
        """
    )
    write_text(OUT_RESEARCH / "03_error_policy.txt", error_policy)


def phase4_eval() -> None:
    protocol = dedent(
        """
        EVALUATION PROTOCOL - RUL
        =========================
        Metrics:
        - RMSE global
        - MAE global
        - RMSE/MAE por subset FD001-FD004
        - Error por bandas de RUL (bajo, medio, alto)

        Validation strategy:
        - Split por unit_id (sin mezcla de ciclos entre train/valid)
        - No leakage temporal
        - Analisis por subset y combinado

        Robustness checks:
        - Ruido aditivo en sensores
        - Drift de condiciones operativas
        - Sensores faltantes simulados
        """
    )
    write_text(OUT_RESEARCH / "04_eval_protocol.txt", protocol)

    acceptance = dedent(
        """
        ACCEPTANCE CRITERIA - RUL RELEASE
        =================================
        - RMSE y MAE mejor o igual al baseline previo.
        - Sin degradacion severa en ningun subset FD.
        - Latencia de inferencia compatible con dashboard interactivo.
        - Contrato de salida estable y validado.
        """
    )
    write_text(OUT_RESEARCH / "04_acceptance_criteria.txt", acceptance)


def phase5_integration() -> None:
    backlog = dedent(
        """
        INTEGRATION BACKLOG
        ===================
        MVP:
        - Endpoint local de inferencia
        - Integracion en dashboard con payload mock-compatible

        V1:
        - Batch scoring por multiples motores
        - Logging de predicciones y estado de servicio

        V2:
        - Reentrenamiento/calibracion periodica
        - Monitoreo de drift y alertas operativas
        """
    )
    write_text(OUT_RESEARCH / "05_integration_backlog.txt", backlog)

    release = dedent(
        """
        RELEASE PLAN - RUL LAYER
        ========================
        Hito 1: baseline tabular + contrato estable.
        Hito 2: baseline temporal + comparativo.
        Hito 3: hardening de errores y fallback.
        Hito 4: integracion total con capa 2 y dashboard.
        """
    )
    write_text(OUT_RESEARCH / "05_release_plan.txt", release)


def phase6_closure(ctx: dict) -> None:
    master = dedent(
        f"""
        MASTER PLAN - RUL LAYER
        =======================
        Resumen:
        - Plan 1 (EDA) completo y usado como base para estrategia de modelado.
        - Arquitectura por contrato definida para acople con capas 2 y 3.
        - Baselines priorizados: tabular + secuencial.

        Decisiones consolidadas:
        - Target primario: RUL capped (130)
        - Contrato de salida estable para dashboard/agent layer
        - Validacion por unit_id y analisis por subset FD

        Insumos EDA clave:
        {ctx["target_definition"]}
        """
    )
    write_text(OUT_RESEARCH / "06_master_plan_rul_layer.txt", master)

    risks_mit = dedent(
        f"""
        RISKS AND MITIGATIONS - RUL LAYER
        =================================
        Riesgo 1: Heterogeneidad entre subsets FD.
        Mitigacion: reporte por subset + entrenamiento combinado y por dominio.

        Riesgo 2: Sensores ruidosos/no informativos.
        Mitigacion: feature selection y robust normalization.

        Riesgo 3: Leakage temporal.
        Mitigacion: split estricto por unit_id y preservacion de orden por ciclo.

        Riesgo 4: Inestabilidad operacional.
        Mitigacion: error policy + fallback + logging.

        Contexto de riesgo heredado de plan 1:
        {ctx["risks"]}
        """
    )
    write_text(OUT_RESEARCH / "06_risks_mitigations.txt", risks_mit)


def main() -> None:
    ensure_dirs()
    ctx = load_eda_context()
    phase0_template()
    phase1_doc_cards(ctx)
    phase2_methodology(ctx)
    phase3_architecture_contract(ctx)
    phase4_eval()
    phase5_integration()
    phase6_closure(ctx)
    print("Plan 2 (research + RUL layer design) completed.")
    print(f"Outputs: {OUT_RESEARCH}")


if __name__ == "__main__":
    main()
