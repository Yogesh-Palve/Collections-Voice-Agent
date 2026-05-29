"""
Streamlit dashboard — Forward Flow calls, follow-ups, export.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import streamlit as st

import data_logger
from config import get_settings

st.set_page_config(page_title="Forward Flow Dashboard", layout="wide")
st.title("Forward Flow Review — Dashboard")

settings = get_settings()
logger = data_logger.DataLogger(settings)

today = date.today()
week_start = today - timedelta(days=today.weekday())
week_end = week_start + timedelta(days=6)

calls = logger.get_calls_in_range(week_start, week_end)
pending = logger.get_pending_followups(today)

if not calls:
    st.info("No Forward Flow calls recorded this week yet.")
else:
    df = pd.DataFrame(calls)

    def outcome_color(outcome: str) -> str:
        """Background color by call outcome."""
        if outcome == "Committed":
            return "background-color: #c6efce"
        if outcome == "Escalated":
            return "background-color: #ffc7ce"
        if outcome == "No Answer":
            return "background-color: #ffeb9c"
        return ""

    display_cols = [
        "call_date",
        "call_time",
        "be_name",
        "be_id",
        "be_zone",
        "forward_flow_target_pct",
        "forward_flow_actual_pct",
        "call_outcome",
        "commitment_date",
    ]
    show = df[[c for c in display_cols if c in df.columns]].copy()

    st.subheader("Calls This Week")
    styled = show.style.map(
        lambda v: outcome_color(v) if isinstance(v, str) else "",
        subset=["call_outcome"] if "call_outcome" in show.columns else None,
    )
    st.dataframe(styled, use_container_width=True)

    st.subheader("BE Forward Flow — Target vs Actual")
    if "be_name" in df.columns:
        chart_df = df.groupby("be_name", as_index=False).agg(
            forward_flow_target_pct=("forward_flow_target_pct", "first"),
            forward_flow_actual_pct=("forward_flow_actual_pct", "mean"),
        )
        st.bar_chart(
            chart_df.set_index("be_name")[["forward_flow_target_pct", "forward_flow_actual_pct"]]
        )

st.subheader("Pending Follow-ups Today")
if pending:
    st.dataframe(pd.DataFrame(pending), use_container_width=True)
else:
    st.success("No commitment follow-ups due today.")

col1, col2 = st.columns(2)
with col1:
    month = st.text_input("Export month (YYYY-MM)", value=today.strftime("%Y-%m"))
    if st.button("Export monthly summary to Excel"):
        path = logger.export_summary_report(month)
        st.success(f"Summary written to {path}")

with col2:
    if calls and st.button("Export current week view to Excel"):
        buffer = BytesIO()
        pd.DataFrame(calls).to_excel(buffer, index=False, sheet_name="WeekExport")
        buffer.seek(0)
        st.download_button(
            label="Download Excel",
            data=buffer,
            file_name=f"forward_flow_week_{week_start}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
