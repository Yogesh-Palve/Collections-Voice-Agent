"""
Call session lifecycle: start, end, metrics.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import data_logger
from config import get_settings
from flows.forward_flow_review import ForwardFlowReviewFlow
from llm import LLMClient
from session_store import (
    get_flow,
    get_pipeline,
    register_flow,
    register_pipeline,
    remove_flow,
    remove_pipeline,
    save_flow_state,
)
from utils import load_conversation_state, save_session_metrics
from voice_pipeline import VoicePipeline

router = APIRouter(prefix="/api/call", tags=["call"])


class CallStartRequest(BaseModel):
    """Initialize a voice call session."""

    be_id: str = ""
    call_type: str = Field(default="general", description="DPD|bounce|HNI|cross-sell|general|forward-flow")
    resume_session_id: Optional[str] = None


class CallEndRequest(BaseModel):
    """End call and persist metrics."""

    session_id: str
    call_dropped: bool = False
    resolution_status: str = "follow-up-needed"


@router.post("/start")
def start_call(req: CallStartRequest) -> dict[str, Any]:
    """Create voice pipeline session; optionally resume dropped call."""
    settings = get_settings()
    resumed = False
    session_id = req.resume_session_id or str(uuid4())

    if req.resume_session_id:
        state = load_conversation_state(req.resume_session_id, settings.log_path)
        if state:
            resumed = True

    llm = LLMClient(settings)
    forward_flow = None
    if req.call_type == "forward-flow":
        forward_flow = ForwardFlowReviewFlow(llm=llm)
        # Minimal state if not resuming full flow
        if not resumed:
            from models.forward_flow import ForwardFlowTriggerRequest

            forward_flow.start_session(
                ForwardFlowTriggerRequest(
                    be_id=req.be_id or "UNKNOWN",
                    be_name="",
                    be_phone="",
                    be_zone="",
                    target_pct=0,
                    actual_pct=0,
                )
            )
        register_flow(forward_flow)

    pipeline = VoicePipeline(
        session_id=session_id,
        settings=settings,
        llm=llm,
        forward_flow=forward_flow,
        call_type=req.call_type,
        be_id=req.be_id,
    )
    if resumed:
        pipeline.metrics["resumed"] = True

    register_pipeline(pipeline)
    text, audio_b64 = _optional_opening(pipeline)

    return {
        "session_id": session_id,
        "resumed": resumed,
        "opening_text": text,
        "has_audio": bool(audio_b64),
        "audio_base64": audio_b64,
    }


def _optional_opening(pipeline: VoicePipeline) -> tuple[str, str]:
    """Generate opening; return base64 audio if synthesis succeeds."""
    import base64

    try:
        text, audio = pipeline.opening_audio()
        return text, base64.b64encode(audio).decode("ascii") if audio else ""
    except Exception:
        return "", ""


@router.post("/end")
def end_call(req: CallEndRequest) -> dict[str, Any]:
    """End session, save transcript + metrics, forward-flow record if applicable."""
    pipeline = get_pipeline(req.session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Session not found")

    metrics = pipeline.end_session(
        call_dropped=req.call_dropped,
        resolution_status=req.resolution_status,
    )

    flow = get_flow(req.session_id)
    if flow:
        record = flow.finalize()
        data_logger.log_forward_flow_call(record.to_row_dict())
        remove_flow(req.session_id)

    save_flow_state(req.session_id, get_settings().log_path)
    remove_pipeline(req.session_id)

    return {"session_id": req.session_id, "metrics": metrics, "status": "ended"}


@router.get("/metrics/{session_id}")
def get_metrics(session_id: str) -> dict[str, Any]:
    """Retrieve call metrics JSON for a session."""
    pipeline = get_pipeline(session_id)
    if pipeline:
        return pipeline.metrics

    from pathlib import Path

    path = get_settings().log_path / f"{session_id}.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail="Metrics not found")
