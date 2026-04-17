from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st


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

