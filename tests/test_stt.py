"""Tests for stt.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stt import DeepgramSTT, ElevenLabsSTT, get_stt_provider, _pcm_to_wav


def test_pcm_to_wav():
    """WAV wrapper produces RIFF header."""
    wav = _pcm_to_wav(b"\x00\x01" * 100, 16000)
    assert wav[:4] == b"RIFF"


def test_deepgram_empty_on_no_audio():
    """Empty chunk returns empty string."""
    stt = DeepgramSTT()
    stt._client = None
    assert stt.transcribe(b"") == ""


@patch("stt.httpx.Client")
def test_elevenlabs_stt_mock(mock_client_cls):
    """ElevenLabs STT parses text from JSON response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "namaste"}
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    from config import get_settings

    settings = get_settings()
    settings.elevenlabs_api_key = "test-key"
    stt = ElevenLabsSTT(settings)
    pcm = b"\x00\x00" * 8000  # 0.5s at 16kHz
    text = stt.transcribe(pcm, sample_rate=16000)
    assert text == "namaste"


def test_get_stt_provider_factory():
    """Factory returns provider for known names."""
    p = get_stt_provider("elevenlabs")
    assert p.name == "elevenlabs"
