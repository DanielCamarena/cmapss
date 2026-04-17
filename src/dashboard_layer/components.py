from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


RISK_COLORS = {
    "healthy": "#2E8B57",
    "warning": "#E8A317",
    "critical": "#C62828",
}


def risk_badge(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#6B7280")
    return (
        f"<span style='background:{color};color:white;padding:0.25rem 0.7rem;"
        f"border-radius:999px;font-weight:700'>{risk_level.upper()}</span>"
    )


def render_kpis(result: Dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RUL Estimate (cycles)", f"{result['rul_pred']:.2f}")
    col2.metric(
        "Confidence Band",
        f"{result['confidence_band']['low']:.1f} - {result['confidence_band']['high']:.1f}",
    )
    col3.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
    col4.metric("Risk Score", f"{result.get('risk_score', 0):.1f}/100")


def render_history_chart(history: List[Dict[str, float]]) -> None:
    df = pd.DataFrame(history)
    fig = px.line(
        df,
        x="cycle",
        y="rul_est",
        title="RUL Trend by Cycle",
        markers=True,
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_fleet_descriptive_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No fleet data available.")
        return

    work = df.copy()
    for col in ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    units_count = int(work["unit_id"].nunique()) if "unit_id" in work.columns else 0
    rows_count = int(len(work))
    max_cycle = (
        int(work["cycle"].max()) if "cycle" in work.columns and work["cycle"].notna().any() else 0
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", rows_count)
    col2.metric("Units", units_count)
    col3.metric("Max Cycle", max_cycle)

    if {"unit_id", "cycle"}.issubset(work.columns):
        unit_cycles = (
            work.groupby("unit_id", dropna=True)["cycle"].max().reset_index().sort_values("cycle")
        )
        if len(unit_cycles) > 20:
            unit_cycles = unit_cycles.tail(20)
        fig_units = px.bar(
            unit_cycles,
            x="unit_id",
            y="cycle",
            title="Max Cycle by Unit (Top recent units)",
        )
        fig_units.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_units, use_container_width=True)

    if "cycle" in work.columns:
        fig_hist = px.histogram(
            work.dropna(subset=["cycle"]),
            x="cycle",
            nbins=30,
            title="Cycle Distribution (fleet)",
        )
        fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    op_cols = [c for c in ["op_setting_1", "op_setting_2", "op_setting_3"] if c in work.columns]
    if op_cols:
        fig_ops = make_subplots(
            rows=1,
            cols=len(op_cols),
            subplot_titles=op_cols,
            shared_yaxes=False,
        )
        for idx, col in enumerate(op_cols, start=1):
            series = work[col].dropna()
            fig_ops.add_trace(
                go.Box(
                    y=series,
                    name=col,
                    boxpoints="outliers",
                    marker=dict(size=4),
                ),
                row=1,
                col=idx,
            )
            fig_ops.update_yaxes(title_text="value", row=1, col=idx)

        fig_ops.update_layout(
            title="Operational Settings Distribution (independent axes)",
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_ops, use_container_width=True)
        st.caption(
            "Each op_setting uses its own y-axis scale to preserve visibility across different magnitudes."
        )

def render_history_mini_chart(history: List[Dict[str, float]]) -> None:
    df = pd.DataFrame(history)
    fig = px.line(
        df,
        x="cycle",
        y="rul_est",
        title="Quick Trend",
        markers=False,
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_unit_rul_history_chart(unit_df: pd.DataFrame) -> None:
    if unit_df.empty or "cycle" not in unit_df.columns or "rul_pred" not in unit_df.columns:
        st.info("No unit prediction history available.")
        return
    fig = px.line(
        unit_df.sort_values("cycle"),
        x="cycle",
        y="rul_pred",
        title="Selected Unit: Predicted RUL by Cycle",
        markers=True,
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_multimodel_comparison_chart(pred_df: pd.DataFrame) -> None:
    if pred_df.empty:
        st.info("No multi-model comparison available.")
        return
    fig = px.bar(
        pred_df,
        x="model",
        y="rul_pred",
        color="model",
        title="Multi-model RUL comparison (selected case)",
        text="rul_pred",
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_top_risk_units_chart(top_df: pd.DataFrame) -> None:
    if top_df.empty:
        st.info("No Top-N risk units available.")
        return
    fig = px.bar(
        top_df,
        x="unit_id",
        y="risk_score",
        color="risk_level",
        title="Top-N Units by Risk Score (champion model)",
        text="risk_score",
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_risk_band_count_chart(count_df: pd.DataFrame) -> None:
    if count_df.empty:
        st.info("No risk-band count data available.")
        return
    order = ["healthy", "warning", "critical"]
    plot_df = count_df.copy()
    plot_df["risk_level"] = pd.Categorical(plot_df["risk_level"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("risk_level")
    fig = px.bar(
        plot_df,
        x="risk_level",
        y="count",
        color="risk_level",
        category_orders={"risk_level": order},
        title="Fleet Risk Band Count",
        text="count",
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_scenario_comparison_chart(comparison: Dict[str, Any]) -> None:
    required = [
        "baseline_rul",
        "scenario_rul",
        "baseline_risk_score",
        "scenario_risk_score",
    ]
    if not all(k in comparison for k in required):
        st.info("Comparison chart not available: missing comparison fields.")
        return

    df = pd.DataFrame(
        [
            {"metric": "RUL", "case": "Baseline", "value": float(comparison["baseline_rul"])},
            {"metric": "RUL", "case": "Scenario", "value": float(comparison["scenario_rul"])},
            {
                "metric": "Risk Score",
                "case": "Baseline",
                "value": float(comparison["baseline_risk_score"]),
            },
            {
                "metric": "Risk Score",
                "case": "Scenario",
                "value": float(comparison["scenario_risk_score"]),
            },
        ]
    )

    fig = px.bar(
        df,
        x="metric",
        y="value",
        color="case",
        barmode="group",
        title="Baseline vs Scenario (RUL and Risk Score)",
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Delta RUL", f"{float(comparison.get('delta_rul', 0.0)):+.2f}")
    col2.metric("Delta Risk Score", f"{float(comparison.get('delta_risk_score', 0.0)):+.2f}")


def render_result_detail(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    st.subheader("Prediction Context")
    st.json(
        {
            "dataset_id": payload["dataset_id"],
            "unit_id": payload["unit_id"],
            "cycle": payload["cycle"],
            "source": payload["source"],
            "op_settings": payload["op_settings"],
            "sensors_preview": payload["sensors"][:6],
        }
    )
    st.subheader("Decision Output")
    st.json(result)
    if result.get("rationale"):
        st.subheader("Decision Rationale")
        for idx, item in enumerate(result["rationale"], start=1):
            st.write(f"{idx}. {item}")
    st.caption(f"Audit record id: {result.get('audit_record_id', 'N/A')}")
    st.caption("Risk policy: critical <= 20, warning <= 60, healthy > 60 cycles.")
