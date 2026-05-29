"""Tests for llm.py — mocked Groq and prompt loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm import LLMClient, TokenUsage


@pytest.fixture
def mock_groq_completion():
    """Patch Groq chat completions."""
    with patch("llm.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="Namaste, main Priya bol rahi hoon."))]
        completion.usage = MagicMock(prompt_tokens=50, completion_tokens=20)

        extraction_completion = MagicMock()
        extraction_completion.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"be_was_aware": "Yes", "sentiment": "cooperative", "escalate": false}'
                )
            )
        ]
        extraction_completion.usage = MagicMock(prompt_tokens=30, completion_tokens=15)

        mock_client.chat.completions.create.side_effect = [completion, extraction_completion]
        yield mock_client


def test_prompts_load():
    """System and user prompts exist and format."""
    client = LLMClient()
    assert "collections" in client.system_prompt_template.lower() or "Collections" in client.system_prompt_template
    assert "{agent_name}" in client.system_prompt_template or "{agent_name}" in client.build_system_prompt()


def test_get_response_returns_text(mock_groq_completion):
    """get_response returns assistant text and token usage."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        client = LLMClient()
        client._client = mock_groq_completion

        resp = client.get_response(
            [{"role": "user", "content": "Haan, Ramesh bol raha hoon."}],
            context={"be_name": "Ramesh Kumar", "be_id": "BE-2041"},
        )
        assert "Priya" in resp.text or len(resp.text) > 0
        assert resp.usage.prompt_tokens == 50
        assert resp.usage.completion_tokens == 20


def test_extract_from_be_message():
    """Extraction returns structured ForwardFlowExtraction."""
    mock_client = MagicMock()
    extraction_completion = MagicMock()
    extraction_completion.choices = [
        MagicMock(
            message=MagicMock(
                content='{"be_was_aware": "Yes", "sentiment": "cooperative", "escalate": false}'
            )
        )
    ]
    extraction_completion.usage = MagicMock(prompt_tokens=30, completion_tokens=15)
    mock_client.chat.completions.create.return_value = extraction_completion

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        client = LLMClient()
        client._client = mock_client

        result = client.extract_from_be_message(
            "Haan mujhe pata tha, main reduce kar sakta hoon.",
            current_step=3,
        )
        assert result.data.be_was_aware == "Yes"
        assert result.data.sentiment == "cooperative"


def test_trim_history():
    """History trimming keeps last N turns."""
    client = LLMClient()
    client.max_turns = 2
    history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    trimmed = client._trim_history(history)
    assert len(trimmed) == 4


def test_estimate_cost():
    """Cost estimation uses settings rates."""
    client = LLMClient()
    cost = client.estimate_cost_usd(TokenUsage(prompt_tokens=1000, completion_tokens=500))
    assert cost > 0
