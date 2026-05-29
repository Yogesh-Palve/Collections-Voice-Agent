"""
Step 1 scaffold verification: config loads, directories exist, requirements parse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_structure_exists():
    """Required directories and config files are present."""
    expected = [
        ".env.example",
        "requirements.txt",
        "config.py",
        "logs",
        "prompts",
    ]
    for name in expected:
        path = PROJECT_ROOT / name
        assert path.exists(), f"Missing: {name}"


def test_settings_load_with_defaults():
    """Settings module loads without requiring real API keys."""
    from config import get_settings

    settings = get_settings()
    assert settings.stt_provider in ("deepgram", "elevenlabs", "cartesia", "hume")
    assert settings.tts_provider in ("elevenlabs", "sarvam")
    assert settings.max_context_turns >= 1
    assert settings.log_path.is_dir()


def test_requirements_file_lists_core_deps():
    """requirements.txt includes packages from the spec."""
    req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("fastapi", "groq", "deepgram-sdk", "elevenlabs", "webrtcvad", "twilio", "streamlit"):
        assert package in req
