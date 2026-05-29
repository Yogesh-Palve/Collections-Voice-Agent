"""In-memory session registry with optional resume from logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from flows.forward_flow_review import ForwardFlowReviewFlow
from logging_utils import setup_error_logger
from voice_pipeline import VoicePipeline

logger = setup_error_logger(__name__)

_active_flows: dict[str, ForwardFlowReviewFlow] = {}
_active_pipelines: dict[str, VoicePipeline] = {}


def register_flow(flow: ForwardFlowReviewFlow) -> str:
    """Store active forward-flow and return session_id."""
    _active_flows[flow.state.session_id] = flow
    return flow.state.session_id


def get_flow(session_id: str) -> Optional[ForwardFlowReviewFlow]:
    """Retrieve active forward-flow by session id."""
    return _active_flows.get(session_id)


def remove_flow(session_id: str) -> None:
    """Remove completed forward-flow session."""
    _active_flows.pop(session_id, None)


def register_pipeline(pipeline: VoicePipeline) -> str:
    """Store active voice pipeline."""
    _active_pipelines[pipeline.session_id] = pipeline
    return pipeline.session_id


def get_pipeline(session_id: str) -> Optional[VoicePipeline]:
    """Get voice pipeline by session id."""
    return _active_pipelines.get(session_id)


def remove_pipeline(session_id: str) -> None:
    """Remove voice pipeline session."""
    _active_pipelines.pop(session_id, None)


def save_flow_state(session_id: str, log_dir: Path) -> None:
    """Persist minimal flow state for call-drop resume."""
    flow = get_flow(session_id)
    pipeline = get_pipeline(session_id)
    payload: dict[str, Any] | None = None

    if flow:
        payload = {
            "type": "forward_flow",
            "session_id": flow.state.session_id,
            "be_id": flow.state.be_id,
            "be_name": flow.state.be_name,
            "be_phone": flow.state.be_phone,
            "be_zone": flow.state.be_zone,
            "target_pct": flow.state.target_pct,
            "actual_pct": flow.state.actual_pct,
            "current_step": flow.state.current_step,
            "escalated_to_human": flow.state.escalated_to_human,
            "conversation": flow.state.conversation,
            "extracted": flow.state.extracted,
            "call_outcome": flow.state.call_outcome,
        }
    elif pipeline:
        payload = {
            "type": "voice",
            "session_id": pipeline.session_id,
            "metrics": pipeline.metrics,
            "conversation": pipeline.conversation,
            "call_type": pipeline.call_type,
            "be_id": pipeline.be_id,
        }

    if not payload:
        return

    path = log_dir / f"{session_id}_state.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to save flow state %s: %s", session_id, exc)


def load_flow_state(session_id: str, log_dir: Path) -> Optional[dict[str, Any]]:
    """Load saved flow state JSON if present."""
    path = log_dir / f"{session_id}_state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load flow state %s: %s", session_id, exc)
        return None
