"""
Text-to-speech abstraction — ElevenLabs, Sarvam (switchable via .env).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Optional

import httpx

from config import Settings, get_settings
from logging_utils import setup_error_logger

logger = setup_error_logger(__name__)


class TTSProvider(ABC):
    """Abstract TTS provider with interruption support."""

    name: str = "base"
    interrupted: bool = False

    def stop(self) -> None:
        """Signal in-flight synthesis to stop."""
        self.interrupted = True

    def reset(self) -> None:
        """Clear interruption flag before new utterance."""
        self.interrupted = False

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return complete audio bytes (typically MP3)."""

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream audio chunks; default wraps full synthesize.

        Yields:
            Audio byte chunks.
        """
        if self.interrupted:
            return
        audio = self.synthesize(text)
        if audio and not self.interrupted:
            yield audio


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs low-latency streaming TTS."""

    name = "elevenlabs"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def synthesize(self, text: str) -> bytes:
        """Synthesize via ElevenLabs REST streaming endpoint."""
        if not text.strip() or not self.settings.elevenlabs_api_key:
            return b""
        if not self.settings.elevenlabs_voice_id:
            logger.error("ELEVENLABS_VOICE_ID not set")
            return b""
        try:
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech/"
                f"{self.settings.elevenlabs_voice_id}/stream"
            )
            payload = {
                "text": text,
                "model_id": self.settings.elevenlabs_tts_model,
            }
            chunks: list[bytes] = []
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    url,
                    headers={"xi-api-key": self.settings.elevenlabs_api_key},
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if self.interrupted:
                            break
                        chunks.append(chunk)
            return b"".join(chunks)
        except Exception as exc:
            logger.error("ElevenLabs TTS failed: %s", exc)
            return b""

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """Yield chunks as they arrive from ElevenLabs."""
        if not text.strip() or not self.settings.elevenlabs_api_key:
            return
        if self.interrupted:
            return
        try:
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech/"
                f"{self.settings.elevenlabs_voice_id}/stream"
            )
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    url,
                    headers={"xi-api-key": self.settings.elevenlabs_api_key},
                    json={
                        "text": text,
                        "model_id": self.settings.elevenlabs_tts_model,
                    },
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if self.interrupted:
                            break
                        if chunk:
                            yield chunk
        except Exception as exc:
            logger.error("ElevenLabs TTS stream failed: %s", exc)


class SarvamTTS(TTSProvider):
    """Sarvam AI TTS for Hindi / Indian languages."""

    name = "sarvam"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def synthesize(self, text: str) -> bytes:
        """Synthesize via Sarvam bulbul TTS API."""
        if not text.strip() or not self.settings.sarvam_api_key:
            return b""
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={
                        "api-subscription-key": self.settings.sarvam_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": [text],
                        "target_language_code": self.settings.sarvam_language,
                        "speaker": self.settings.sarvam_speaker,
                        "pitch": 0,
                        "pace": 1.0,
                        "loudness": 1.0,
                        "speech_sample_rate": 22050,
                        "enable_preprocessing": True,
                        "model": "bulbul:v1",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                audios = data.get("audios") or []
                if not audios:
                    return b""
                import base64

                return base64.b64decode(audios[0])
        except Exception as exc:
            logger.error("Sarvam TTS failed: %s", exc)
            return b""


def get_tts_provider(name: Optional[str] = None, settings: Optional[Settings] = None) -> TTSProvider:
    """Factory controlled by TTS_PROVIDER env var."""
    settings = settings or get_settings()
    key = (name or settings.tts_provider).lower().strip()
    if key == "sarvam":
        return SarvamTTS(settings)
    return ElevenLabsTTS(settings)
