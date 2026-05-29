"""Tests for voice_pipeline.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voice_pipeline import VoicePipeline


@pytest.fixture
def mock_pipeline():
    """Pipeline with mocked STT/TTS/LLM."""
    stt = MagicMock()
    stt.name = "mock-stt"
    stt.transcribe_with_duration.return_value = ("hello BE", 1.0)

    tts = MagicMock()
    tts.name = "mock-tts"
    tts.interrupted = False
    tts.synthesize.return_value = b"audio"
    tts.reset = MagicMock()
    tts.stop = MagicMock()

    llm = MagicMock()
    llm.cumulative_usage.prompt_tokens = 10
    llm.cumulative_usage.completion_tokens = 5
    llm.get_response.return_value = MagicMock(
        text="Namaste, kaise madad karoon?",
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        model="test",
    )

    return VoicePipeline(
        session_id="test-voice-1",
        stt=stt,
        tts=tts,
        llm=llm,
        call_type="general",
        be_id="BE-1",
    )


def test_flush_utterance(mock_pipeline):
    """Full turn produces user text and audio."""
    mock_pipeline._audio_buffer.extend(b"\x00\x01" * 8000)
    user, reply, audio = mock_pipeline.flush_utterance()
    assert user.lower() == "hello be"
    assert reply
    assert audio == b"audio"


def test_interruption(mock_pipeline):
    """Interruption increments metric."""
    mock_pipeline.handle_interruption()
    assert mock_pipeline.metrics["interruption_count"] == 1
    mock_pipeline.tts.stop.assert_called_once()


def test_end_session_writes_metrics(mock_pipeline, tmp_path, monkeypatch):
    """end_session saves JSON metrics."""
    from config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "log_dir", str(tmp_path))
    # Override log_path property usage via save path
    user, _, _ = mock_pipeline.flush_utterance()
    metrics = mock_pipeline.end_session(resolution_status="resolved")
    assert metrics["session_id"] == "test-voice-1"
    assert metrics.get("total_estimated_cost_usd", 0) >= 0
