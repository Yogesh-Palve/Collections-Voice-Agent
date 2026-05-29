"""
Speech-to-text abstraction — Deepgram, ElevenLabs, Cartesia, Hume (switchable via .env).
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from config import Settings, get_settings
from logging_utils import setup_error_logger

logger = setup_error_logger(__name__)


class STTProvider(ABC):
    """Abstract STT provider."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """
        Transcribe a PCM16 mono audio buffer.

        Returns:
            Transcript text or empty string on failure.
        """

    def transcribe_with_duration(
        self, audio_chunk: bytes, *, sample_rate: int = 16000
    ) -> tuple[str, float]:
        """
        Transcribe and return estimated audio duration in seconds.

        Returns:
            (transcript, seconds)
        """
        seconds = len(audio_chunk) / (2 * sample_rate) if audio_chunk else 0.0
        return self.transcribe(audio_chunk, sample_rate=sample_rate), seconds


class DeepgramSTT(STTProvider):
    """Deepgram nova-2 (batch chunk; supports hi-en multilingual)."""

    name = "deepgram"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize Deepgram client."""
        self.settings = settings or get_settings()
        self._client = None
        if self.settings.deepgram_api_key:
            try:
                from deepgram import DeepgramClient

                self._client = DeepgramClient(self.settings.deepgram_api_key)
            except Exception as exc:
                logger.error("Deepgram init failed: %s", exc)

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Transcribe via Deepgram prerecorded API."""
        if not self._client or not audio_chunk:
            return ""
        try:
            payload = {
                "buffer": audio_chunk,
                "mimetype": "audio/raw",
            }
            options = {
                "model": self.settings.deepgram_model,
                "language": self.settings.deepgram_language,
                "smart_format": True,
                "punctuate": True,
            }
            response = self._client.listen.rest.v("1").transcribe_file(
                payload, options
            )
            channels = response.results.channels
            if channels and channels[0].alternatives:
                return (channels[0].alternatives[0].transcript or "").strip()
            return ""
        except Exception as exc:
            logger.error("Deepgram transcribe failed: %s", exc)
            return ""


class ElevenLabsSTT(STTProvider):
    """ElevenLabs speech-to-text fallback."""

    name = "elevenlabs"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Store API key."""
        self.settings = settings or get_settings()

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Transcribe via ElevenLabs STT REST API."""
        if not self.settings.elevenlabs_api_key or not audio_chunk:
            return ""
        try:
            wav = _pcm_to_wav(audio_chunk, sample_rate)
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={"xi-api-key": self.settings.elevenlabs_api_key},
                    files={"file": ("audio.wav", wav, "audio/wav")},
                    data={"model_id": "scribe_v1"},
                )
                resp.raise_for_status()
                data = resp.json()
                return (data.get("text") or data.get("transcript") or "").strip()
        except Exception as exc:
            logger.error("ElevenLabs STT failed: %s", exc)
            return ""


class CartesiaSTT(STTProvider):
    """Cartesia STT via HTTP (optional provider)."""

    name = "cartesia"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Transcribe using Cartesia API if configured."""
        if not self.settings.cartesia_api_key or not audio_chunk:
            return ""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.cartesia.ai/stt",
                    headers={
                        "X-API-Key": self.settings.cartesia_api_key,
                        "Cartesia-Version": "2024-06-10",
                    },
                    files={"file": ("audio.wav", _pcm_to_wav(audio_chunk, sample_rate), "audio/wav")},
                )
                if resp.status_code == 404:
                    logger.error("Cartesia STT endpoint unavailable")
                    return ""
                resp.raise_for_status()
                return (resp.json().get("text") or "").strip()
        except Exception as exc:
            logger.error("Cartesia STT failed: %s", exc)
            return ""


class HumeSTT(STTProvider):
    """Hume AI STT fallback."""

    name = "hume"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Transcribe via Hume if API key present."""
        if not self.settings.hume_api_key or not audio_chunk:
            return ""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.hume.ai/v0/batch/jobs",
                    headers={"X-Hume-Api-Key": self.settings.hume_api_key},
                    files={"file": ("audio.wav", _pcm_to_wav(audio_chunk, sample_rate), "audio/wav")},
                )
                # Batch-only on Hume; return empty for realtime chunk path
                if resp.status_code >= 400:
                    return ""
                return (resp.json().get("text") or "").strip()
        except Exception as exc:
            logger.error("Hume STT failed: %s", exc)
            return ""


class FallbackSTT(STTProvider):
    """Primary with automatic fallback chain."""

    name = "fallback"

    def __init__(self, primary: STTProvider, fallback: STTProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Try primary, then fallback."""
        text = self.primary.transcribe(audio_chunk, sample_rate=sample_rate)
        if text:
            return text
        return self.fallback.transcribe(audio_chunk, sample_rate=sample_rate)


def get_stt_provider(name: Optional[str] = None, settings: Optional[Settings] = None) -> STTProvider:
    """
    Factory for STT provider controlled by STT_PROVIDER env var.

    deepgram uses ElevenLabs as fallback when primary returns empty.
    """
    settings = settings or get_settings()
    key = (name or settings.stt_provider).lower().strip()

    providers = {
        "deepgram": DeepgramSTT(settings),
        "elevenlabs": ElevenLabsSTT(settings),
        "cartesia": CartesiaSTT(settings),
        "hume": HumeSTT(settings),
    }
    primary = providers.get(key, DeepgramSTT(settings))
    if key == "deepgram":
        return FallbackSTT(primary, ElevenLabsSTT(settings))
    return primary


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono in a minimal WAV container."""
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
