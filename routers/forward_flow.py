"""
Forward-flow outbound trigger and session management API.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import data_logger
from flows.forward_flow_review import ForwardFlowReviewFlow
from models.forward_flow import ForwardFlowTriggerRequest
from session_store import (
    get_flow,
    get_pipeline,
    register_flow,
    register_pipeline,
    remove_flow,
    remove_pipeline,
    save_flow_state,
)
from telephony import trigger_outbound_call
from config import get_settings
from voice_pipeline import VoicePipeline
from llm import LLMClient

router = APIRouter(prefix="/api/calls", tags=["calls"])


class ForwardFlowTurnRequest(BaseModel):
    """Process one BE utterance in an active session."""

    session_id: str
    message: str


class EndCallRequest(BaseModel):
    """End session and persist forward-flow record."""

    session_id: str


@router.post("/trigger-forward-flow")
def trigger_forward_flow(req: ForwardFlowTriggerRequest) -> dict[str, Any]:
    """
    Start outbound forward-flow review: session + optional Twilio dial.
    """
    llm = LLMClient(get_settings())
    flow = ForwardFlowReviewFlow(llm=llm)
    state = flow.start_session(req)
    register_flow(flow)

    pipeline = VoicePipeline(
        session_id=state.session_id,
        llm=llm,
        forward_flow=flow,
        call_type="forward-flow",
        be_id=req.be_id,
    )
    pipeline.metrics["forward_flow_target_pct"] = req.target_pct
    pipeline.metrics["forward_flow_actual_pct"] = req.actual_pct
    register_pipeline(pipeline)

    opening = flow.process_assistant_opening()

    call_sid = trigger_outbound_call(req.be_phone, state.session_id)

    return {
        "session_id": state.session_id,
        "flow": "ForwardFlowReviewFlow",
        "opening_message": opening,
        "twilio_call_sid": call_sid,
        "status": "started",
    }


@router.post("/forward-flow/turn")
def forward_flow_turn(req: ForwardFlowTurnRequest) -> dict[str, Any]:
    """Process BE message and return bot reply + extraction."""
    flow = get_flow(req.session_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Session not found")

    reply, extraction = flow.process_be_turn(req.message)
    save_flow_state(req.session_id, get_settings().log_path)

    return {
        "session_id": req.session_id,
        "reply": reply,
        "current_step": flow.state.current_step,
        "extraction": extraction.model_dump(),
        "escalated": flow.state.escalated_to_human,
    }


@router.post("/forward-flow/end")
def end_forward_flow(req: EndCallRequest) -> dict[str, Any]:
    """Finalize call, dual-write record, clear session."""
    flow = get_flow(req.session_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Session not found")

    record = flow.finalize()
    data_logger.log_forward_flow_call(record.to_row_dict())
    remove_flow(req.session_id)

    pipeline = get_pipeline(req.session_id)
    if pipeline:
        pipeline.end_session(resolution_status="resolved" if not record.escalated_to_human else "escalated")
        remove_pipeline(req.session_id)

    return {
        "session_id": req.session_id,
        "record": record.model_dump(),
        "status": "completed",
    }
