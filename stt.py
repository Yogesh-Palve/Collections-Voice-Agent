"""
Speech-to-text abstraction — Deepgram, ElevenLabs, Cartesia, Hume (switchable via .env).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import httpx

from config import Settings, get_settings
from logging_utils import setup_error_logger
from utils import pcm_to_wav, prepare_pcm_for_stt

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
        from utils import pcm_duration_seconds

        seconds = pcm_duration_seconds(audio_chunk, sample_rate) if audio_chunk else 0.0
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
        """Transcribe via Deepgram prerecorded API (WAV container)."""
        if not self._client or not audio_chunk:
            return ""

        prepared, rate = prepare_pcm_for_stt(
            audio_chunk,
            target_rate=self.settings.sample_rate,
            source_rate=sample_rate,
            min_ms=self.settings.min_stt_audio_ms,
        )
        if not prepared:
            return ""

        wav = pcm_to_wav(prepared, rate)
        try:
            payload = {
                "buffer": wav,
                "mimetype": "audio/wav",
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
    """ElevenLabs speech-to-text fallback (requires STT-enabled API key)."""

    name = "elevenlabs"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Store API key."""
        self.settings = settings or get_settings()

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Transcribe via ElevenLabs STT REST API."""
        if not self.settings.elevenlabs_api_key or not audio_chunk:
            return ""

        prepared, rate = prepare_pcm_for_stt(
            audio_chunk,
            target_rate=self.settings.sample_rate,
            source_rate=sample_rate,
            min_ms=self.settings.min_stt_audio_ms,
        )
        if not prepared:
            return ""

        wav = pcm_to_wav(prepared, rate)
        try:
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
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.error(
                    "ElevenLabs STT unauthorized (401). "
                    "Your key may be TTS-only — set ELEVENLABS_STT_FALLBACK=false "
                    "or use a key with Speech-to-Text access."
                )
            else:
                logger.error("ElevenLabs STT failed: %s", exc)
            return ""
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
        prepared, rate = prepare_pcm_for_stt(
            audio_chunk,
            target_rate=self.settings.sample_rate,
            source_rate=sample_rate,
            min_ms=self.settings.min_stt_audio_ms,
        )
        if not prepared:
            return ""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.cartesia.ai/stt",
                    headers={
                        "X-API-Key": self.settings.cartesia_api_key,
                        "Cartesia-Version": "2024-06-10",
                    },
                    files={
                        "file": (
                            "audio.wav",
                            pcm_to_wav(prepared, rate),
                            "audio/wav",
                        )
                    },
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
        prepared, rate = prepare_pcm_for_stt(
            audio_chunk,
            target_rate=self.settings.sample_rate,
            source_rate=sample_rate,
            min_ms=self.settings.min_stt_audio_ms,
        )
        if not prepared:
            return ""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.hume.ai/v0/batch/jobs",
                    headers={"X-Hume-Api-Key": self.settings.hume_api_key},
                    files={
                        "file": (
                            "audio.wav",
                            pcm_to_wav(prepared, rate),
                            "audio/wav",
                        )
                    },
                )
                if resp.status_code >= 400:
                    return ""
                return (resp.json().get("text") or "").strip()
        except Exception as exc:
            logger.error("Hume STT failed: %s", exc)
            return ""


class FallbackSTT(STTProvider):
    """Primary with optional ElevenLabs fallback."""

    name = "fallback"

    def __init__(self, primary: STTProvider, fallback: Optional[STTProvider]) -> None:
        self.primary = primary
        self.fallback = fallback

    def transcribe(self, audio_chunk: bytes, *, sample_rate: int = 16000) -> str:
        """Try primary, then optional fallback."""
        text = self.primary.transcribe(audio_chunk, sample_rate=sample_rate)
        if text or not self.fallback:
            return text
        return self.fallback.transcribe(audio_chunk, sample_rate=sample_rate)


def get_stt_provider(name: Optional[str] = None, settings: Optional[Settings] = None) -> STTProvider:
    """
    Factory for STT provider controlled by STT_PROVIDER env var.

    Deepgram optionally falls back to ElevenLabs STT when ELEVENLABS_STT_FALLBACK=true.
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
        fallback = None
        if settings.elevenlabs_stt_fallback and settings.elevenlabs_api_key:
            fallback = ElevenLabsSTT(settings)
        return FallbackSTT(primary, fallback)

    return primary


# Backwards-compatible alias used in tests
_pcm_to_wav = pcm_to_wav
