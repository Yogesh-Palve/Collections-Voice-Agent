"""Tests for tts.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tts import ElevenLabsTTS, SarvamTTS, get_tts_provider


@patch("tts.httpx.Client")
def test_elevenlabs_tts_interrupt(mock_client_cls):
    """Interrupted flag stops synthesis."""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    stream_resp = MagicMock()
    stream_resp.iter_bytes.return_value = [b"audio"]
    stream_resp.raise_for_status = MagicMock()
    mock_client.stream.return_value.__enter__.return_value = stream_resp
    mock_client_cls.return_value = mock_client

    from config import get_settings

    settings = get_settings()
    settings.elevenlabs_api_key = "key"
    settings.elevenlabs_voice_id = "voice"
    tts = ElevenLabsTTS(settings)
    tts.interrupted = True
    assert tts.synthesize("hello") == b""


def test_tts_provider_stop():
    """stop() sets interrupted."""
    tts = get_tts_provider("elevenlabs")
    tts.reset()
    assert tts.interrupted is False
    tts.stop()
    assert tts.interrupted is True


@patch("tts.httpx.Client")
def test_sarvam_tts_base64(mock_client_cls):
    """Sarvam returns decoded base64 audio."""
    import base64

    audio = base64.b64encode(b"mp3bytes").decode()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"audios": [audio]}
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    from config import get_settings

    settings = get_settings()
    settings.sarvam_api_key = "key"
    tts = SarvamTTS(settings)
    result = tts.synthesize("नमस्ते")
    assert result == b"mp3bytes"
