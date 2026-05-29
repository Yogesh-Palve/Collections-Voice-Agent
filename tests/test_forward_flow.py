"""Tests for ForwardFlowReviewFlow state machine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flows.forward_flow_review import FLOW_NAME, ForwardFlowReviewFlow, TOTAL_STEPS
from models.forward_flow import ForwardFlowExtraction, ForwardFlowTriggerRequest


@pytest.fixture
def mock_llm():
    """LLM with deterministic responses."""
    llm = MagicMock()
    llm.settings.agent_name = "Priya"
    llm.settings.company_name = "ABC Finance"
    llm.get_response.return_value = MagicMock(
        text="Thank you. What strategy will you use?",
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        model="llama3-70b-8192",
    )
    llm.build_user_turn_prompt.return_value = "turn prompt"
    return llm


def test_flow_name_and_steps():
    """Flow constants match spec."""
    assert FLOW_NAME == "ForwardFlowReviewFlow"
    assert TOTAL_STEPS == 6


def test_opening_message(mock_llm):
    """Opening script includes BE name and agent."""
    flow = ForwardFlowReviewFlow(llm=mock_llm)
    req = ForwardFlowTriggerRequest(
        be_id="BE-2041",
        be_name="Ramesh Kumar",
        be_phone="+919999999999",
        be_zone="Pune West",
        target_pct=8,
        actual_pct=10,
    )
    flow.start_session(req)
    opening = flow.opening_message()
    assert "Ramesh Kumar" in opening
    assert "Priya" in opening


def test_step_advance_on_extraction(mock_llm):
    """Step advances after identity confirmed."""
    from llm import ExtractionResult

    flow = ForwardFlowReviewFlow(llm=mock_llm)
    req = ForwardFlowTriggerRequest(
        be_id="BE-2041",
        be_name="Ramesh Kumar",
        be_phone="+919999999999",
        be_zone="Pune West",
        target_pct=8,
        actual_pct=10,
    )
    flow.start_session(req)

    mock_llm.extract_from_be_message.return_value = ExtractionResult(
        data=ForwardFlowExtraction(be_name_confirmed=True, sentiment="cooperative"),
    )

    reply, ext = flow.process_be_turn("Haan, Ramesh bol raha hoon.")
    assert flow.state.current_step >= 2
    assert reply


def test_escalation_on_refusal(mock_llm):
    """No/Unsure on step 3 triggers escalation."""
    from llm import ExtractionResult

    flow = ForwardFlowReviewFlow(llm=mock_llm)
    req = ForwardFlowTriggerRequest(
        be_id="BE-99",
        be_name="Test BE",
        be_phone="+910000000000",
        be_zone="Zone A",
        target_pct=8,
        actual_pct=15,
    )
    flow.start_session(req)
    flow.state.current_step = 3

    mock_llm.extract_from_be_message.return_value = ExtractionResult(
        data=ForwardFlowExtraction(be_can_reduce="No", escalate=True, sentiment="resistant"),
    )

    flow.process_be_turn("Nahi, possible nahi hai.")
    assert flow.state.escalated_to_human is True
    assert flow.state.call_outcome == "Escalated"


def test_build_call_record(mock_llm):
    """Finalize produces record with required fields."""
    flow = ForwardFlowReviewFlow(llm=mock_llm)
    req = ForwardFlowTriggerRequest(
        be_id="BE-2041",
        be_name="Ramesh Kumar",
        be_phone="+919999999999",
        be_zone="Pune West",
        target_pct=8,
        actual_pct=10,
    )
    flow.start_session(req)
    flow.state.extracted = {
        "be_was_aware": "Yes",
        "be_can_reduce": "Yes",
        "be_strategy": "High EMI payers first",
        "commitment_date_parsed": "2026-06-05",
    }
    record = flow.build_call_record()
    assert record.be_id == "BE-2041"
    assert record.forward_flow_actual_pct == 10
    assert record.session_id
