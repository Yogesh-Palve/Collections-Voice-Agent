"""
Streamlit demo UI — microphone-free text mode for quick pipeline testing.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from config import get_settings
from flows.forward_flow_review import ForwardFlowReviewFlow
from llm import LLMClient
from models.forward_flow import ForwardFlowTriggerRequest
from utils import estimate_costs, save_session_metrics
import data_logger

st.set_page_config(page_title="Voice Agent Demo", layout="wide")
st.title("Collections Voice Agent — Demo")

settings = get_settings()
tab_chat, tab_forward = st.tabs(["General chat", "Forward Flow Review"])

with tab_chat:
    st.caption("Text-only LLM test (no STT/TTS). Add GROQ_API_KEY to .env.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_msg = st.chat_input("Type as Business Executive (Hindi/English)")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        client = LLMClient(settings)
        resp = client.get_response(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": resp.text})
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

with tab_forward:
    st.subheader("Forward Flow Review (text)")
    col1, col2 = st.columns(2)
    with col1:
        be_name = st.text_input("BE Name", "Ramesh Kumar")
        be_id = st.text_input("BE ID", "BE-2041")
        be_zone = st.text_input("Zone", "Pune West")
    with col2:
        target = st.number_input("Target %", value=8.0)
        actual = st.number_input("Actual %", value=10.0)

    if "ff_flow" not in st.session_state:
        st.session_state.ff_flow = None

    if st.button("Start Forward Flow Call"):
        flow = ForwardFlowReviewFlow(llm=LLMClient(settings))
        flow.start_session(
            ForwardFlowTriggerRequest(
                be_id=be_id,
                be_name=be_name,
                be_phone="+910000000000",
                be_zone=be_zone,
                target_pct=target,
                actual_pct=actual,
            )
        )
        st.session_state.ff_flow = flow
        st.session_state.ff_opening = flow.process_assistant_opening()
        st.success(f"Session: {flow.state.session_id}")

    if st.session_state.get("ff_flow"):
        st.info(st.session_state.get("ff_opening", ""))
        be_turn = st.text_input("BE says")
        if st.button("Send turn") and be_turn:
            reply, ext = st.session_state.ff_flow.process_be_turn(be_turn)
            st.write("**Priya:**", reply)
            st.json(ext.model_dump())
        if st.button("End call & save"):
            record = st.session_state.ff_flow.finalize()
            data_logger.log_forward_flow_call(record.to_row_dict())
            st.success("Saved to Excel + SQLite")
            st.json(record.model_dump())
