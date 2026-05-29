"""Forward Flow Review call models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ForwardFlowTriggerRequest(BaseModel):
    """Payload for POST /api/calls/trigger-forward-flow."""

    be_id: str
    be_name: str
    be_phone: str
    be_zone: str
    target_pct: float
    actual_pct: float


class ForwardFlowExtraction(BaseModel):
    """Structured fields parsed from BE natural speech."""

    be_name_confirmed: Optional[bool] = None
    be_was_aware: Optional[Literal["Yes", "No"]] = None
    be_can_reduce: Optional[Literal["Yes", "No", "Unsure"]] = None
    be_strategy: Optional[str] = None
    commitment_date_raw: Optional[str] = None
    commitment_date_parsed: Optional[str] = None
    strategy_mentioned: Optional[str] = None
    sentiment: Literal["cooperative", "hesitant", "resistant"] = "cooperative"
    escalate: bool = False
    no_answer: bool = False
    notes: Optional[str] = None


class ForwardFlowCallRecord(BaseModel):
    """Dual-write record for Excel + SQLite."""

    session_id: str
    call_date: str
    call_time: str
    be_name: str
    be_id: str
    be_zone: str
    forward_flow_target_pct: float
    forward_flow_actual_pct: float
    be_was_aware: str = ""
    be_can_reduce: str = ""
    be_strategy: str = ""
    commitment_date: Optional[str] = None
    escalated_to_human: bool = False
    call_outcome: Literal["Committed", "Escalated", "No Answer", "In Progress"] = "In Progress"
    notes: str = ""

    def to_row_dict(self) -> dict:
        """Flat dict for spreadsheet / DB insert."""
        return self.model_dump()
