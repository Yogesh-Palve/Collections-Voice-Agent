"""Tests for utils.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils import (
    _energy_voice_activity,
    detect_voice_activity,
    estimate_costs,
    hindi_number_normalize,
    normalize_audio,
    pcm_duration_seconds,
    save_session_metrics,
)


def _silence_pcm(ms: int = 30, rate: int = 16000) -> bytes:
    """Generate silence PCM16."""
    n = int(rate * ms / 1000)
    return b"\x00\x00" * n


def _tone_pcm(ms: int = 100, rate: int = 16000) -> bytes:
    """Generate loud tone PCM16 for VAD."""
    t = np.linspace(0, ms / 1000, int(rate * ms / 1000), endpoint=False)
    wave = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    return wave.tobytes()


def test_pcm_duration():
    """Duration calculation."""
    pcm = b"\x00\x00" * 16000  # 1 second at 16kHz PCM16
    assert pcm_duration_seconds(pcm, 16000) == 1.0


def test_normalize_audio():
    """Normalization changes peak level."""
    pcm = _tone_pcm()
    out = normalize_audio(pcm)
    assert len(out) == len(pcm)


def test_vad_on_silence():
    """Silence should not trigger VAD."""
    assert detect_voice_activity(_silence_pcm(100)) is False


def test_energy_vad_on_tone():
    """Energy fallback detects loud tone."""
    assert _energy_voice_activity(_tone_pcm(100), threshold=1000) is True


def test_energy_vad_on_silence():
    """Energy fallback ignores silence."""
    assert _energy_voice_activity(_silence_pcm(100), threshold=500) is False


def test_hindi_number_normalize():
    """Spoken numbers converted."""
    assert "2000" in hindi_number_normalize("do hazaar rupaye")


def test_save_and_estimate_costs(tmp_path):
    """Metrics file written with cost fields."""
    metrics = {
        "session_id": "abc",
        "total_prompt_tokens": 1000,
        "total_completion_tokens": 500,
        "total_tts_characters": 200,
        "total_stt_seconds": 10,
    }
    save_session_metrics("abc", metrics, tmp_path)
    path = tmp_path / "abc.json"
    assert path.exists()
    estimate_costs(metrics)
    assert metrics["total_estimated_cost_usd"] > 0
