"""Sarvam TTS request payload tests."""

from __future__ import annotations

from tts import SarvamTTS
from config import get_settings


def test_sarvam_v3_payload_omits_pitch_and_loudness():
    """Bulbul V3 must not send pitch or loudness."""
    settings = get_settings()
    settings.sarvam_model = "bulbul:v3"
    tts = SarvamTTS(settings)
    payload = tts._build_request_payload("namaste")
    assert "pitch" not in payload
    assert "loudness" not in payload
    assert payload["model"] == "bulbul:v3"
    assert payload["inputs"] == ["namaste"]


def test_sarvam_v1_payload_includes_legacy_params():
    """Older bulbul:v1 model keeps pitch/loudness."""
    settings = get_settings()
    settings.sarvam_model = "bulbul:v1"
    tts = SarvamTTS(settings)
    payload = tts._build_request_payload("hello")
    assert "pitch" in payload
    assert "loudness" in payload
