from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.backend_adapter import run_prediction_with_adapter
from dashboard.components import render_history_chart, render_kpis, render_result_detail
from dashboard.errors import ServiceUnavailableError, ValidationError


DATASET_UNIT_LIMITS = {
    "FD001": 100,
    "FD002": 260,
    "FD003": 100,
    "FD004": 249,
}

REQUIRED_CSV_COLUMNS = (
    ["dataset_id", "unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def init_state() -> None:
    if "app_status" not in st.session_state:
        st.session_state["app_status"] = "sin_datos"
    if "last_payload" not in st.session_state:
        st.session_state["last_payload"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = ""
    if "autodemo_loaded" not in st.session_state:
        st.session_state["autodemo_loaded"] = False


def default_demo_payload() -> Dict[str, Any]:
    sensors = [0.0] * 21
    sensors[0] = 18.5
    sensors[1] = -7.2
    sensors[2] = 10.1
    sensors[3] = 5.6
    return {
        "dataset_id": "FD001",
        "unit_id": 1,
        "cycle": 75,
        "op_settings": [0.2, -0.1, 0.0],
        "sensors": sensors,
        "source": "manual",
    }


def payload_from_manual() -> Dict[str, Any]:
    st.sidebar.subheader("Manual Input")
    dataset_id = st.sidebar.selectbox("Dataset", list(DATASET_UNIT_LIMITS.keys()), index=0)
    max_unit = DATASET_UNIT_LIMITS[dataset_id]
    unit_id = st.sidebar.slider("Unit ID", min_value=1, max_value=max_unit, value=1, step=1)
    cycle = st.sidebar.slider("Cycle", min_value=1, max_value=400, value=60, step=1)

    st.sidebar.markdown("Operational Settings")
    op_settings = [
        st.sidebar.slider(f"op_setting_{i}", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        for i in range(1, 4)
    ]

    sensors: List[float] = []
    with st.sidebar.expander("Sensors (1..21)", expanded=False):
        for i in range(1, 22):
            value = st.number_input(
                f"sensor_{i}", min_value=-500.0, max_value=500.0, value=0.0, step=0.5
            )
            sensors.append(float(value))

    return {
        "dataset_id": dataset_id,
        "unit_id": int(unit_id),
        "cycle": int(cycle),
        "op_settings": [float(v) for v in op_settings],
        "sensors": sensors,
        "source": "manual",
    }


def validate_csv_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)


def payload_from_csv(df: pd.DataFrame) -> Dict[str, Any]:
    row_idx = st.sidebar.number_input(
        "CSV row index",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        step=1,
    )
    row = df.iloc[int(row_idx)]
    return {
        "dataset_id": str(row["dataset_id"]),
        "unit_id": int(row["unit_id"]),
        "cycle": int(row["cycle"]),
        "op_settings": [float(row[f"op_setting_{i}"]) for i in range(1, 4)],
        "sensors": [float(row[f"sensor_{i}"]) for i in range(1, 22)],
        "source": "csv",
    }


def infer_dataset_from_filename(filename: str) -> str | None:
    upper = filename.upper()
    for ds in DATASET_UNIT_LIMITS:
        if ds in upper:
            return ds
    return None


def parse_nasa_txt(uploaded_file: Any, dataset_id: str) -> pd.DataFrame:
    txt_df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, engine="python")
    if txt_df.shape[1] < 26:
        raise ValueError("NASA TXT file must contain at least 26 columns.")
    txt_df = txt_df.iloc[:, :26].copy()
    parsed = pd.DataFrame(index=txt_df.index)
    parsed["dataset_id"] = dataset_id
    parsed["unit_id"] = txt_df.iloc[:, 0].astype(int)
    parsed["cycle"] = txt_df.iloc[:, 1].astype(int)
    for i in range(1, 4):
        parsed[f"op_setting_{i}"] = txt_df.iloc[:, 1 + i].astype(float)
    for i in range(1, 22):
        parsed[f"sensor_{i}"] = txt_df.iloc[:, 4 + i].astype(float)
    return parsed


def run_prediction(payload: Dict[str, Any]) -> None:
    try:
        st.session_state["app_status"] = "loading"
        with st.spinner("Running prediction pipeline..."):
            result = run_prediction_with_adapter(payload)
        st.session_state["last_payload"] = payload
        st.session_state["last_result"] = result
        st.session_state["history"] = result.get("history", [])
        st.session_state["app_status"] = result.get("service_status", "ok")
        st.session_state["last_error"] = ""
    except ValidationError as e:
        st.session_state["app_status"] = "error_validacion"
        st.session_state["last_error"] = str(e)
        st.session_state["last_result"] = None
        st.session_state["history"] = []
    except ServiceUnavailableError as e:
        st.session_state["app_status"] = "degraded"
        st.session_state["last_error"] = str(e)
        st.session_state["last_result"] = None
        st.session_state["history"] = []


def render_status_banner() -> None:
    status = st.session_state["app_status"]
    if status == "sin_datos":
        st.info("State: sin_datos. Provide input and run prediction.")
    elif status == "loading":
        st.warning("State: loading. Pipeline is processing input.")
    elif status == "error_validacion":
        st.error(f"State: error_validacion. {st.session_state['last_error']}")
    elif status in {"degraded", "fallback"}:
        st.warning(f"State: {status}. {st.session_state['last_error']}")
    elif status == "ok":
        st.success("State: ok. Prediction ready.")


def main() -> None:
    st.set_page_config(page_title="PHM Dashboard Mock - C-MAPSS", layout="wide")
    init_state()

    st.title("PHM Dashboard Mock - NASA C-MAPSS")
    st.caption(
        "Academic PoC over NASA C-MAPSS benchmark. This view simulates data/model/agent layers."
    )
    render_status_banner()

    if st.session_state["last_result"] is None and not st.session_state["autodemo_loaded"]:
        run_prediction(default_demo_payload())
        st.session_state["autodemo_loaded"] = True
        st.info("Demo scenario auto-loaded so tabs show simulated content.")

    input_mode = st.sidebar.radio("Input Mode", options=["Manual", "CSV"], index=0)
    payload: Dict[str, Any] | None = None

    if input_mode == "Manual":
        if st.sidebar.button("Load Demo Scenario", use_container_width=True):
            run_prediction(default_demo_payload())
        payload = payload_from_manual()
    else:
        st.sidebar.subheader("CSV/TXT Input")
        sample_path = Path(__file__).resolve().parent / "mock" / "sample_input.csv"
        if sample_path.exists():
            st.sidebar.download_button(
                label="Download sample CSV",
                data=sample_path.read_bytes(),
                file_name="sample_input.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.sidebar.caption("You can upload `sample_input.csv` or NASA raw `train/test_*.txt`.")
        uploaded = st.sidebar.file_uploader("Upload CSV or NASA TXT", type=["csv", "txt"])
        if uploaded is None:
            st.session_state["app_status"] = "sin_datos"
            st.sidebar.info("Waiting for file. Required columns are documented below.")
            with st.sidebar.expander("Expected CSV columns", expanded=False):
                st.write(REQUIRED_CSV_COLUMNS)
        else:
            name = uploaded.name.lower()
            try:
                if name.endswith(".txt"):
                    inferred_ds = infer_dataset_from_filename(uploaded.name) or "FD001"
                    selected_ds = st.sidebar.selectbox(
                        "Dataset for TXT file",
                        list(DATASET_UNIT_LIMITS.keys()),
                        index=list(DATASET_UNIT_LIMITS.keys()).index(inferred_ds),
                    )
                    df = parse_nasa_txt(uploaded, selected_ds)
                    st.sidebar.success(
                        f"NASA TXT parsed as {selected_ds}. Rows available: {len(df)}"
                    )
                else:
                    df = pd.read_csv(uploaded)
            except Exception as e:
                st.session_state["app_status"] = "error_validacion"
                st.session_state["last_error"] = f"File parse error: {e}"
                st.sidebar.error(st.session_state["last_error"])
                df = None

        if uploaded is not None and df is not None:
            valid, missing = validate_csv_columns(df)
            if not valid:
                st.session_state["app_status"] = "error_validacion"
                st.session_state["last_error"] = f"Missing required CSV columns: {missing}"
                st.sidebar.error(st.session_state["last_error"])
                st.sidebar.caption(f"Expected columns: {REQUIRED_CSV_COLUMNS}")
            else:
                payload = payload_from_csv(df)

    if st.sidebar.button("Run Prediction", use_container_width=True):
        if payload is None:
            st.session_state["app_status"] = "sin_datos"
        else:
            run_prediction(payload)

    tabs = st.tabs(["Resumen", "Detalle", "Historico", "Recomendaciones"])
    result = st.session_state["last_result"]
    current_payload = st.session_state["last_payload"]
    history = st.session_state["history"]

    with tabs[0]:
        st.subheader("Resumen")
        if result is None:
            st.info("No prediction yet.")
        else:
            render_kpis(result)

    with tabs[1]:
        st.subheader("Detalle")
        if result is None or current_payload is None:
            st.info("No detail available.")
        else:
            render_result_detail(current_payload, result)

    with tabs[2]:
        st.subheader("Historico")
        if not history:
            st.info("No trend available.")
        else:
            render_history_chart(history)

    with tabs[3]:
        st.subheader("Recomendaciones")
        if result is None:
            st.info("No recommendation available.")
        else:
            st.markdown(f"**Risk level:** `{result['risk_level']}`")
            st.markdown(
                f"**Priority:** `{result.get('recommendation_priority', 'N/A')}` | "
                f"**Risk score:** `{result.get('risk_score', 0):.1f}/100`"
            )
            st.write(result["recommendation_text"])
            if result.get("recommendation_alternatives"):
                st.markdown("**Alternative actions:**")
                for alt in result["recommendation_alternatives"]:
                    st.write(f"- {alt}")
            if result.get("dashboard_note"):
                st.caption(result["dashboard_note"])
            if result.get("evidence_summary"):
                st.caption(f"Evidence: {result['evidence_summary']}")
            if result.get("audit_record_id"):
                st.caption(f"Audit record: {result['audit_record_id']}")
            st.caption(
                "Operational guidance is mock-only and intended for UI/flow validation."
            )


if __name__ == "__main__":
    main()
