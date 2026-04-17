from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard_layer.backend_adapter import (
    run_prediction_with_adapter,
    run_scenario_with_adapter,
)
from dashboard_layer.components import (
    render_fleet_descriptive_charts,
    render_history_mini_chart,
    render_kpis,
    render_multimodel_comparison_chart,
    render_risk_band_count_chart,
    render_scenario_comparison_chart,
    render_top_risk_units_chart,
    render_unit_rul_history_chart,
)
from dashboard_layer.errors import ServiceUnavailableError, ValidationError


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
    if "last_scenario" not in st.session_state:
        st.session_state["last_scenario"] = None
    if "analysis_df" not in st.session_state:
        st.session_state["analysis_df"] = None


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


def build_friendly_csv_error(df: pd.DataFrame, missing: List[str]) -> str:
    sensor_cols_present = [c for c in df.columns if c.startswith("sensor_")]
    missing_sensors = [c for c in missing if c.startswith("sensor_")]
    missing_core = [c for c in missing if not c.startswith("sensor_")]
    msg_parts = [
        "Invalid CSV format for this dashboard.",
        "Required minimum schema: dataset_id, unit_id, cycle, op_setting_1..3, and sensor_1..sensor_21.",
    ]
    if missing_core:
        msg_parts.append(f"Missing core fields: {missing_core}.")
    if missing_sensors:
        msg_parts.append(
            f"Missing sensors: {len(missing_sensors)} out of 21. "
            f"Detected in file: {len(sensor_cols_present)}."
        )
    msg_parts.append("Tip: check exact column names (no extra spaces or naming variations).")
    return " ".join(msg_parts)


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
        st.session_state["last_scenario"] = None
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


def run_scenario(scenario_prompt: str, base_payload: Dict[str, Any]) -> None:
    try:
        with st.spinner("Running scenario assistant..."):
            out = run_scenario_with_adapter(
                scenario_prompt=scenario_prompt,
                base_payload=base_payload,
            )
        st.session_state["last_scenario"] = out
        st.session_state["last_error"] = ""
    except ValidationError as e:
        st.session_state["last_error"] = str(e)
    except ServiceUnavailableError as e:
        st.session_state["last_error"] = str(e)


def render_status_banner() -> None:
    status = st.session_state["app_status"]
    if status == "sin_datos":
        if st.session_state.get("last_result") is not None:
            st.info(
                "State: no_data. Provide new input and run prediction. "
                "Showing last loaded prediction."
            )
        else:
            st.info("State: no_data. Provide input and run prediction.")
    elif status == "loading":
        st.warning("State: loading. Pipeline is processing input.")
    elif status == "error_validacion":
        st.error(f"State: validation_error. {st.session_state['last_error']}")
    elif status == "degraded":
        st.warning(f"State: degraded. {st.session_state['last_error']}")
    elif status == "ok":
        st.success("State: ok. Prediction ready.")


def build_next_step_text(result: Dict[str, Any]) -> str:
    risk = str(result.get("risk_level", "")).lower()
    rul = float(result.get("rul_pred", 0.0))
    if risk == "critical":
        return f"Immediate action: trigger priority inspection. Estimated RUL {rul:.1f} cycles."
    if risk == "warning":
        return f"Recommended action: schedule inspection in the next maintenance window. Estimated RUL {rul:.1f} cycles."
    return f"Recommended action: continue operation with monitoring. Estimated RUL {rul:.1f} cycles."


def parse_model_version(model_version: str) -> Dict[str, str]:
    # Expected pattern: predictive_layer-rf-v1
    parts = str(model_version).split("-")
    parsed = {
        "layer": parts[0] if len(parts) > 0 else "unknown",
        "family": parts[1] if len(parts) > 1 else "unknown",
        "revision": parts[2] if len(parts) > 2 else "unknown",
    }
    family_map = {
        "rf": "Random Forest",
        "gb": "Gradient Boosting",
        "lstm": "LSTM proxy (flattened window + MLP)",
        "gru": "GRU proxy (flattened window + ExtraTrees)",
    }
    parsed["family_label"] = family_map.get(parsed["family"], parsed["family"])
    return parsed


def load_model_metadata(model_family: str) -> Dict[str, Any] | None:
    meta_path = ROOT_DIR / "out" / "predictive_layer" / "models" / f"{model_family}_metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_prediction_explanation(result: Dict[str, Any]) -> str:
    rul = float(result.get("rul_pred", 0.0))
    low = float(result.get("confidence_band", {}).get("low", 0.0))
    high = float(result.get("confidence_band", {}).get("high", 0.0))
    width = max(0.0, high - low)
    return (
        f"Estimated RUL: {rul:.2f} cycles. "
        f"Confidence band: [{low:.2f}, {high:.2f}] (width {width:.2f} cycles). "
        f"Risk: {result.get('risk_level', 'N/A')} with score {float(result.get('risk_score', 0.0)):.1f}/100."
    )


def _predictive_assets_paths() -> Dict[str, Path]:
    out_pred = ROOT_DIR / "out" / "predictive_layer"
    return {
        "out_pred": out_pred,
        "normalizer": out_pred / "normalizer.json",
        "champion": out_pred / "champion.json",
        "calib": out_pred / "04_confidence_band_policy.json",
        "models": out_pred / "models",
        "metrics_global": out_pred / "03_metrics_global_by_model.csv",
    }


def _load_predictive_assets() -> Dict[str, Any]:
    paths = _predictive_assets_paths()
    normalizer = json.loads(paths["normalizer"].read_text(encoding="utf-8"))
    champion = json.loads(paths["champion"].read_text(encoding="utf-8")).get("champion", "rf")
    calib = json.loads(paths["calib"].read_text(encoding="utf-8"))
    models = {
        "rf": joblib.load(paths["models"] / "rf_model.joblib"),
        "gb": joblib.load(paths["models"] / "gb_model.joblib"),
        "lstm": joblib.load(paths["models"] / "lstm_model.joblib"),
        "gru": joblib.load(paths["models"] / "gru_model.joblib"),
    }
    return {
        "normalizer": normalizer,
        "champion": champion,
        "calib": calib,
        "models": models,
        "paths": paths,
    }


def _payload_from_df_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "dataset_id": str(row["dataset_id"]),
        "unit_id": int(row["unit_id"]),
        "cycle": int(row["cycle"]),
        "op_settings": [float(row[f"op_setting_{i}"]) for i in range(1, 4)],
        "sensors": [float(row[f"sensor_{i}"]) for i in range(1, 22)],
        "source": "csv",
    }


def _to_feature_vector_from_payload(payload: Dict[str, Any], normalizer: Dict[str, Any]) -> np.ndarray:
    raw = {
        "op_setting_1": float(payload["op_settings"][0]),
        "op_setting_2": float(payload["op_settings"][1]),
        "op_setting_3": float(payload["op_settings"][2]),
    }
    for i in range(1, 22):
        raw[f"sensor_{i}"] = float(payload["sensors"][i - 1])

    z = []
    for f in normalizer["feature_order"]:
        mean = float(normalizer["means"][f])
        std = float(normalizer["stds"][f]) if float(normalizer["stds"][f]) != 0 else 1.0
        z.append((float(raw[f]) - mean) / std)
    return np.array(z, dtype=np.float32)


def _predict_rul_for_model(model_name: str, payload: Dict[str, Any], assets: Dict[str, Any]) -> float:
    x_vec = _to_feature_vector_from_payload(payload, assets["normalizer"])
    model = assets["models"][model_name]
    if model_name in {"rf", "gb"}:
        return float(model.predict(x_vec.reshape(1, -1))[0])
    seq_flat = np.tile(x_vec, 30).reshape(1, -1).astype(np.float32)
    return float(model.predict(seq_flat)[0])


def _confidence_band_for_rul(rul_pred: float, assets: Dict[str, Any]) -> Dict[str, float]:
    residual_std = float(assets["calib"]["residual_std"])
    width = max(6.0, min(30.0, residual_std * 1.6))
    low = max(0.0, rul_pred - width)
    high = max(low, rul_pred + width)
    return {"low": round(low, 2), "high": round(high, 2)}


def _predict_with_risk(payload: Dict[str, Any], model_name: str, assets: Dict[str, Any]) -> Dict[str, Any]:
    from agent_layer.risk_engine import compute_risk_decision
    from agent_layer.tools import generate_history, tool_read_policy

    rul_pred = max(0.0, _predict_rul_for_model(model_name, payload, assets))
    band = _confidence_band_for_rul(rul_pred, assets)
    history = generate_history(cycle=int(payload["cycle"]), rul_pred=rul_pred, length=30)
    risk = compute_risk_decision(
        rul_pred=rul_pred,
        confidence_band=band,
        history=history,
        thresholds=tool_read_policy(),
    )
    return {
        "rul_pred": round(rul_pred, 2),
        "confidence_band": band,
        "risk_level": risk["risk_level"],
        "risk_score": float(risk["risk_score"]),
    }


def _safe_load_metrics_global() -> pd.DataFrame | None:
    path = _predictive_assets_paths()["metrics_global"]
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="PHM Dashboard - NASA C-MAPSS", layout="wide")
    init_state()

    st.title("PHM Dashboard - NASA C-MAPSS")
    st.caption("Final dashboard layer integrated with predictive and agent layers.")
    render_status_banner()

    input_mode = st.sidebar.radio("Input Mode", options=["Manual", "CSV"], index=0)
    payload: Dict[str, Any] | None = None

    if input_mode == "Manual":
        if st.sidebar.button("Load Demo Scenario", use_container_width=True):
            run_prediction(default_demo_payload())
        st.sidebar.caption("Demo intentionally uses only a few non-zero sensors.")
        payload = payload_from_manual()
    else:
        st.sidebar.subheader("CSV/TXT Input")
        sample_path = ROOT_DIR / "dashboard" / "mock" / "sample_input.csv"
        if sample_path.exists():
            st.sidebar.download_button(
                label="Download sample CSV",
                data=sample_path.read_bytes(),
                file_name="sample_input.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.sidebar.caption("Upload sample CSV or NASA raw train/test TXT.")
        uploaded = st.sidebar.file_uploader("Upload CSV or NASA TXT", type=["csv", "txt"])
        if uploaded is None:
            st.session_state["app_status"] = "sin_datos"
            st.session_state["analysis_df"] = None
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
                    st.session_state["last_error"] = build_friendly_csv_error(df, missing)
                    st.sidebar.error(st.session_state["last_error"])
                    st.sidebar.caption(f"Expected columns: {REQUIRED_CSV_COLUMNS}")
                else:
                    payload = payload_from_csv(df)
                    st.session_state["analysis_df"] = df.copy()

    if st.sidebar.button("Run Prediction", use_container_width=True):
        if payload is None:
            st.session_state["app_status"] = "sin_datos"
        else:
            run_prediction(payload)

    tabs = st.tabs(["Summary", "Analysis", "Scenarios", "Technical Audit"])
    result = st.session_state["last_result"]
    current_payload = st.session_state["last_payload"]
    history = st.session_state["history"]

    with tabs[0]:
        st.subheader("Summary")
        if result is None:
            st.info("No prediction yet.")
        else:
            if st.session_state.get("app_status") == "sin_datos":
                st.caption("Last loaded prediction (not from a new execution).")
            render_kpis(result)

            st.markdown("---")
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("**Immediate action (operator)**")
                st.write(build_next_step_text(result))
                st.markdown(
                    f"**Priority:** `{result.get('recommendation_priority', 'N/A')}` | "
                    f"**Risk:** `{result.get('risk_level', 'N/A')}` | "
                    f"**Risk score:** `{result.get('risk_score', 0):.1f}/100`"
                )
                if result.get("recommendation_text"):
                    st.caption(result["recommendation_text"])
                if result.get("recommendation_alternatives"):
                    st.markdown("**Quick alternatives:**")
                    for alt in result["recommendation_alternatives"][:2]:
                        st.write(f"- {alt}")

                with st.expander("View alternatives and evidence", expanded=False):
                    if result.get("recommendation_alternatives"):
                        st.markdown("**Alternative actions:**")
                        for alt in result["recommendation_alternatives"]:
                            st.write(f"- {alt}")
                    if result.get("dashboard_note"):
                        st.caption(result["dashboard_note"])
                    if result.get("rationale"):
                        st.markdown("**Rationale:**")
                        for idx, item in enumerate(result["rationale"], start=1):
                            st.write(f"{idx}. {item}")

            with col_right:
                st.markdown("**Fast degradation trend**")
                if history:
                    render_history_mini_chart(history)
                else:
                    st.info("No trend available yet.")

    with tabs[1]:
        st.subheader("Analysis")
        analysis_df = st.session_state.get("analysis_df")
        selected_payload = current_payload if current_payload is not None else payload
        assets = None
        try:
            assets = _load_predictive_assets()
        except Exception as e:
            st.warning(f"Predictive assets unavailable for advanced analysis: {e}")

        st.markdown("**Selected unit: full history**")
        if analysis_df is None or selected_payload is None or assets is None:
            st.info("Upload CSV/TXT and run a prediction to enable this block.")
        else:
            unit_id = int(selected_payload["unit_id"])
            unit_df = analysis_df[analysis_df["unit_id"].astype(int) == unit_id].copy()
            unit_df = unit_df.sort_values("cycle")
            if unit_df.empty:
                st.info("No rows found for the selected unit in the loaded dataset.")
            else:
                unit_rows = []
                for _, row in unit_df.iterrows():
                    p = _payload_from_df_row(row)
                    out = _predict_with_risk(p, assets["champion"], assets)
                    unit_rows.append(
                        {
                            "cycle": int(p["cycle"]),
                            "rul_pred": float(out["rul_pred"]),
                            "risk_level": out["risk_level"],
                            "risk_score": float(out["risk_score"]),
                        }
                    )
                unit_hist_df = pd.DataFrame(unit_rows)
                render_unit_rul_history_chart(unit_hist_df)
                st.dataframe(unit_hist_df.tail(10), use_container_width=True, hide_index=True)

            st.markdown("---")
        st.markdown("**Multi-model comparison (selected case)**")
        if selected_payload is None or assets is None:
            st.info("No selected case for multi-model comparison.")
        else:
            rows = []
            for model_name in ["rf", "gb", "lstm", "gru"]:
                pred = max(0.0, _predict_rul_for_model(model_name, selected_payload, assets))
                rows.append({"model": model_name, "rul_pred": round(pred, 2)})
            multi_df = pd.DataFrame(rows)
            champion_name = str(assets["champion"])
            spread = float(multi_df["rul_pred"].max() - multi_df["rul_pred"].min())
            st.caption(
                f"Current champion: `{champion_name}` | Spread across models: `{spread:.2f}` cycles."
            )
            render_multimodel_comparison_chart(multi_df)
            st.dataframe(multi_df.sort_values("rul_pred"), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Offline metric by model (validation)**")
        metrics_df = _safe_load_metrics_global()
        if metrics_df is None:
            st.info("No offline metrics file found.")
        else:
            st.dataframe(metrics_df.sort_values("rmse"), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Top-N highest-risk units (champion)**")
        if analysis_df is None or assets is None:
            st.info("Upload CSV/TXT to compute per-unit risk ranking.")
        else:
            n_top = st.slider("Top N", min_value=3, max_value=30, value=10, step=1)
            latest = (
                analysis_df.sort_values("cycle")
                .groupby("unit_id", as_index=False, sort=False)
                .tail(1)
                .reset_index(drop=True)
            )
            fleet_rows = []
            for _, row in latest.iterrows():
                p = _payload_from_df_row(row)
                out = _predict_with_risk(p, assets["champion"], assets)
                fleet_rows.append(
                    {
                        "unit_id": int(p["unit_id"]),
                        "dataset_id": str(p["dataset_id"]),
                        "last_cycle": int(p["cycle"]),
                        "rul_pred": float(out["rul_pred"]),
                        "risk_level": out["risk_level"],
                        "risk_score": float(out["risk_score"]),
                    }
                )
            fleet_df = pd.DataFrame(fleet_rows).sort_values("risk_score", ascending=False)
            top_df = fleet_df.head(int(n_top)).copy()
            render_top_risk_units_chart(top_df)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.markdown("**Risk band count (fleet)**")
            band_df = (
                fleet_df.groupby("risk_level", as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            render_risk_band_count_chart(band_df)
            st.dataframe(
                band_df.sort_values("count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("**Fleet descriptive analysis (CSV/TXT loaded)**")
        if analysis_df is None:
            st.info("Upload a CSV/TXT file to enable multi-unit/cycle descriptive views.")
        else:
            render_fleet_descriptive_charts(analysis_df)

    with tabs[2]:
        st.subheader("Scenarios")
        if current_payload is None:
            st.info("Run a baseline prediction first, then test what-if scenarios.")
        else:
            default_prompt = "Increase cycle by 25 and apply high load profile."
            scenario_prompt = st.text_area(
                "Describe scenario (what-if)",
                value=default_prompt,
                height=100,
            )
            st.caption(
                "Tip: in rules mode, prompts work best with patterns like "
                "`increase cycle by 20`, `high load`, `reduce load`, "
                "`sensor_5 +10`, `op_setting_2 -0.3`."
            )
            if st.button("Run Scenario Assistant", use_container_width=True):
                run_scenario(scenario_prompt, current_payload)

            scenario_out = st.session_state.get("last_scenario")
            if scenario_out is not None:
                mode_label = scenario_out.get("assistant_mode", "rules_only")
                service_label = scenario_out.get("service_status", "ok")
                st.markdown(f"**Assistant mode:** `{mode_label}` | **Service status:** `{service_label}`")
                if scenario_out.get("llm_model_used"):
                    st.caption(f"LLM model used: {scenario_out.get('llm_model_used')}")

                st.markdown("**Change summary**")
                if scenario_out.get("change_summary"):
                    for item in scenario_out["change_summary"]:
                        st.write(f"- {item}")
                else:
                    st.write("- No changes applied.")

                st.markdown("**Assumptions**")
                for item in scenario_out.get("assumptions", []):
                    st.write(f"- {item}")

                st.markdown("**Safety notes**")
                for item in scenario_out.get("safety_notes", []):
                    st.write(f"- {item}")

                comp = scenario_out.get("comparison", {})
                if comp:
                    st.markdown("**Comparison interpretation**")
                    st.write(
                        scenario_out.get(
                            "comparison_interpretation",
                            "No interpretation available.",
                        )
                    )
                    st.markdown(
                        f"**Operator guidance:** {scenario_out.get('operator_guidance', 'Not available')}"
                    )
                    st.caption(f"Impact label: {scenario_out.get('impact_label', 'mixed')}")

                    st.markdown("**Baseline vs Scenario**")
                    render_scenario_comparison_chart(comp)
                    with st.expander("View comparison JSON", expanded=False):
                        st.json(comp)

                if scenario_out.get("scenario_result"):
                    st.markdown("**Scenario decision output**")
                    st.json(scenario_out["scenario_result"])

    with tabs[3]:
        st.subheader("Technical Audit")
        if result is None or current_payload is None:
            st.info("No technical data available.")
        else:
            st.markdown("**Traceability**")
            st.write(f"Audit record: `{result.get('audit_record_id', 'N/A')}`")
            st.write(f"Service status: `{result.get('service_status', 'N/A')}`")
            st.write(f"Model version: `{result.get('model_version', 'N/A')}`")
            st.write(f"Timestamp: `{result.get('timestamp', 'N/A')}`")

            st.markdown("**Prediction sheet**")
            st.write(build_prediction_explanation(result))

            st.markdown("**Model sheet**")
            model_version = str(result.get("model_version", "unknown"))
            model_info = parse_model_version(model_version)
            st.write(f"- Layer: `{model_info['layer']}`")
            st.write(f"- Family: `{model_info['family']}` ({model_info['family_label']})")
            st.write(f"- Revision: `{model_info['revision']}`")

            metadata = load_model_metadata(model_info["family"])
            if metadata is None:
                st.caption("No model metadata file found for this model family.")
            else:
                st.write(f"- Training seed: `{metadata.get('seed', 'N/A')}`")
                st.write(
                    f"- Train/valid samples: `{metadata.get('n_train', 'N/A')}` / "
                    f"`{metadata.get('n_valid', 'N/A')}`"
                )
                if metadata.get("window") is not None:
                    st.write(f"- Temporal window: `{metadata.get('window')}`")
                st.write(f"- Number of features: `{len(metadata.get('feature_cols', []))}`")
                if metadata.get("implementation_note"):
                    st.caption(f"Implementation note: {metadata['implementation_note']}")

            with st.expander("View input context (JSON)", expanded=False):
                st.json(
                    {
                        "dataset_id": current_payload["dataset_id"],
                        "unit_id": current_payload["unit_id"],
                        "cycle": current_payload["cycle"],
                        "source": current_payload["source"],
                        "op_settings": current_payload["op_settings"],
                        "sensors_preview": current_payload["sensors"][:6],
                    }
                )

            with st.expander("View full technical output (JSON)", expanded=False):
                st.json(result)


if __name__ == "__main__":
    main()
