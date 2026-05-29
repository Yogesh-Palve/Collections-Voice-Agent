"""
ForwardFlowReviewFlow — outbound forward-flow target review with strict steps 1–6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from llm import LLMClient
from models.forward_flow import ForwardFlowCallRecord, ForwardFlowExtraction
from models.forward_flow import ForwardFlowTriggerRequest


FLOW_NAME = "ForwardFlowReviewFlow"
TOTAL_STEPS = 6


@dataclass
class ForwardFlowSessionState:
    """In-memory session state for an active forward-flow call."""

    session_id: str
    be_id: str
    be_name: str
    be_phone: str
    be_zone: str
    target_pct: float
    actual_pct: float
    current_step: int = 1
    escalated_to_human: bool = False
    conversation: list[dict[str, str]] = field(default_factory=list)
    extracted: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    call_outcome: str = "In Progress"

    def to_llm_context(self) -> dict[str, Any]:
        """Context dict for system prompt formatting."""
        return {
            "be_name": self.be_name,
            "be_id": self.be_id,
            "be_zone": self.be_zone,
            "target_pct": self.target_pct,
            "actual_pct": self.actual_pct,
            "current_step": self.current_step,
            "escalated_to_human": self.escalated_to_human,
        }


class ForwardFlowReviewFlow:
    """
    State machine for forward-flow review calls.

    Coordinates LLM replies, extraction after each BE turn, and final record build.
    """

    STEP_INSTRUCTIONS = {
        1: "Greet and verify you are speaking with the correct BE. Ask them to confirm their name.",
        2: "State forward flow target vs actual for their zone. Ask if they were aware of the increase.",
        3: "Ask if they believe they can bring forward flow down. If they cannot, ask reason and prepare escalation.",
        4: "Ask what strategy they will use to reduce forward flow. Acknowledge their plan.",
        5: "Ask for a specific commitment date to reach the target %. Confirm the date back to them.",
        6: "Thank them, recap strategy and commitment date, ask if they need anything else, and close.",
    }

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        """Initialize flow with optional shared LLM client."""
        self.llm = llm or LLMClient()

    @classmethod
    def from_trigger(cls, req: ForwardFlowTriggerRequest, llm: Optional[LLMClient] = None) -> ForwardFlowReviewFlow:
        """Create flow instance from outbound trigger payload."""
        flow = cls(llm=llm)
        flow.state = ForwardFlowSessionState(
            session_id=str(uuid4()),
            be_id=req.be_id,
            be_name=req.be_name,
            be_phone=req.be_phone,
            be_zone=req.be_zone,
            target_pct=req.target_pct,
            actual_pct=req.actual_pct,
        )
        return flow

    def start_session(self, req: ForwardFlowTriggerRequest) -> ForwardFlowSessionState:
        """Initialize session from trigger request."""
        self.state = ForwardFlowSessionState(
            session_id=str(uuid4()),
            be_id=req.be_id,
            be_name=req.be_name,
            be_phone=req.be_phone,
            be_zone=req.be_zone,
            target_pct=req.target_pct,
            actual_pct=req.actual_pct,
        )
        return self.state

    def opening_message(self) -> str:
        """Scripted opening for step 1 (also usable before streaming LLM)."""
        agent = self.llm.settings.agent_name
        company = self.llm.settings.company_name
        return (
            f"Hello, am I speaking with {self.state.be_name}? "
            f"This is {agent} calling from the Collections team at {company}."
        )

    def process_be_turn(self, be_message: str) -> tuple[str, ForwardFlowExtraction]:
        """
        Process BE utterance: extract fields, advance step, generate bot reply.

        Returns:
            Tuple of (assistant_reply, extraction).
        """
        if be_message.strip():
            self.state.conversation.append({"role": "user", "content": be_message})

        extraction_result = self.llm.extract_from_be_message(
            be_message,
            current_step=self.state.current_step,
            context=self.state.to_llm_context(),
        )
        extraction = extraction_result.data
        self._merge_extraction(extraction)
        self._advance_step(extraction)

        step_instruction = self.STEP_INSTRUCTIONS.get(
            self.state.current_step, self.STEP_INSTRUCTIONS[6]
        )
        turn_prompt = self.llm.build_user_turn_prompt(
            flow_name=FLOW_NAME,
            current_step=self.state.current_step,
            total_steps=TOTAL_STEPS,
            step_instruction=step_instruction,
            be_last_message=be_message,
        )

        history = list(self.state.conversation)
        history.append({"role": "user", "content": turn_prompt})

        response = self.llm.get_response(history, context=self.state.to_llm_context())
        reply = response.text
        self.state.conversation.append({"role": "assistant", "content": reply})
        return reply, extraction

    def process_assistant_opening(self) -> str:
        """First bot message without prior BE input."""
        opening = self.opening_message()
        self.state.conversation.append({"role": "assistant", "content": opening})
        return opening

    def build_call_record(self) -> ForwardFlowCallRecord:
        """Build structured record from accumulated extraction and session."""
        now = datetime.now()
        ex = self.state.extracted
        outcome = self.state.call_outcome
        if self.state.escalated_to_human:
            outcome = "Escalated"
        elif ex.get("commitment_date_parsed") and ex.get("be_can_reduce") == "Yes":
            outcome = "Committed"
        elif ex.get("no_answer"):
            outcome = "No Answer"

        strategy = ex.get("be_strategy") or ex.get("strategy_mentioned") or ""

        return ForwardFlowCallRecord(
            session_id=self.state.session_id,
            call_date=now.strftime("%Y-%m-%d"),
            call_time=now.strftime("%I:%M %p").lstrip("0"),
            be_name=self.state.be_name,
            be_id=self.state.be_id,
            be_zone=self.state.be_zone,
            forward_flow_target_pct=self.state.target_pct,
            forward_flow_actual_pct=self.state.actual_pct,
            be_was_aware=ex.get("be_was_aware") or "",
            be_can_reduce=ex.get("be_can_reduce") or "",
            be_strategy=strategy,
            commitment_date=ex.get("commitment_date_parsed"),
            escalated_to_human=self.state.escalated_to_human,
            call_outcome=outcome,  # type: ignore[arg-type]
            notes=ex.get("notes") or "",
        )

    def finalize(self) -> ForwardFlowCallRecord:
        """Mark session complete and return record for persistence."""
        self.state.completed = True
        if self.state.current_step >= TOTAL_STEPS and self.state.call_outcome == "In Progress":
            self.state.call_outcome = "Committed"
        return self.build_call_record()

    def _merge_extraction(self, extraction: ForwardFlowExtraction) -> None:
        """Merge non-null extraction fields into session state."""
        data = extraction.model_dump(exclude_none=True)
        for key, value in data.items():
            if value is not None and value != "":
                self.state.extracted[key] = value
        if extraction.escalate:
            self.state.escalated_to_human = True
            self.state.call_outcome = "Escalated"
        if extraction.no_answer:
            self.state.call_outcome = "No Answer"

    def _advance_step(self, extraction: ForwardFlowExtraction) -> None:
        """Advance flow step based on step number and extraction."""
        step = self.state.current_step

        if step == 1 and extraction.be_name_confirmed is True:
            self.state.current_step = 2
        elif step == 2 and extraction.be_was_aware in ("Yes", "No"):
            self.state.current_step = 3
        elif step == 3:
            if extraction.be_can_reduce == "Yes":
                self.state.current_step = 4
            elif extraction.be_can_reduce in ("No", "Unsure") or extraction.escalate:
                self.state.escalated_to_human = True
                self.state.call_outcome = "Escalated"
                self.state.current_step = 6
        elif step == 4 and (extraction.be_strategy or extraction.strategy_mentioned):
            self.state.current_step = 5
        elif step == 5 and extraction.commitment_date_parsed:
            self.state.current_step = 6
        elif step == 6:
            self.state.completed = True
