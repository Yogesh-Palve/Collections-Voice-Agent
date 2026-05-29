"""
Per-call cost metrics dashboard — reads logs/*.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import get_settings

st.set_page_config(page_title="Call Metrics", layout="wide")
st.title("Call Metrics & Cost Tracking")

settings = get_settings()
log_dir = settings.log_path
files = sorted(log_dir.glob("*.json"))
records = []

for f in files:
    if f.name.endswith("_state.json"):
        continue
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if "session_id" in data and "total_prompt_tokens" in data:
            records.append(data)
    except (json.JSONDecodeError, OSError):
        continue

if not records:
    st.info("No call metrics yet. Complete a web voice call to generate logs/*.json")
    st.stop()

df = pd.DataFrame(records)
df["start_time"] = pd.to_datetime(df.get("start_time"), errors="coerce")

st.subheader("All calls")
display_cols = [
    c
    for c in [
        "session_id",
        "start_time",
        "duration_seconds",
        "be_id",
        "call_type",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tts_characters",
        "total_stt_seconds",
        "total_estimated_cost_usd",
        "interruption_count",
        "resolution_status",
    ]
    if c in df.columns
]
st.dataframe(df[display_cols], use_container_width=True)

st.subheader("Cost breakdown (USD)")
cost_cols = [
    "estimated_llm_cost_usd",
    "estimated_tts_cost_usd",
    "estimated_stt_cost_usd",
    "total_estimated_cost_usd",
]
present = [c for c in cost_cols if c in df.columns]
if present:
    st.bar_chart(df[present].fillna(0).sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM", f"${df.get('estimated_llm_cost_usd', pd.Series([0])).sum():.4f}")
    col2.metric("TTS", f"${df.get('estimated_tts_cost_usd', pd.Series([0])).sum():.4f}")
    col3.metric("STT", f"${df.get('estimated_stt_cost_usd', pd.Series([0])).sum():.4f}")
    col4.metric("Total", f"${df.get('total_estimated_cost_usd', pd.Series([0])).sum():.4f}")

st.subheader("Provider usage")
if "stt_provider_used" in df.columns:
    st.write(df["stt_provider_used"].value_counts())
if "tts_provider_used" in df.columns:
    st.write(df["tts_provider_used"].value_counts())

st.subheader("Engagement")
if "engagement_turns" in df.columns:
    st.line_chart(df.set_index("start_time")["engagement_turns"])

selected = st.selectbox("View transcript", df["session_id"].tolist())
row = df[df["session_id"] == selected].iloc[0]
transcript = row.get("transcript", [])
if transcript:
    for entry in transcript:
        st.markdown(f"**{entry.get('role', '?')}** ({entry.get('timestamp', '')}): {entry.get('content', '')}")

if st.button("Download metrics CSV"):
    st.download_button(
        "Download",
        df.to_csv(index=False).encode("utf-8"),
        file_name="call_metrics.csv",
        mime="text/csv",
    )
