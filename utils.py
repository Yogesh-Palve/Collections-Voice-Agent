"""
Audio processing, VAD, metrics persistence, Hindi number normalization.
"""

from __future__ import annotations

import json
import io
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import Settings, get_settings
from logging_utils import setup_error_logger

logger = setup_error_logger(__name__)

# webrtcvad imports pkg_resources at load time; ensure setuptools is importable first
try:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated",
        category=UserWarning,
        module=r"webrtcvad",
    )
    import setuptools  # noqa: F401
except ImportError:
    pass

_webrtc_vad_warned = False

# Hindi / Hinglish number words → integer
_HINDI_NUMBERS = {
    "zero": 0, "ek": 1, "one": 1, "do": 2, "two": 2, "teen": 3, "three": 3,
    "char": 4, "four": 4, "paanch": 5, "panch": 5, "five": 5,
    "chhe": 6, "six": 6, "saat": 7, "seven": 7, "aath": 8, "eight": 8,
    "nau": 9, "nine": 9, "das": 10, "ten": 10,
    "hazaar": 1000, "hazar": 1000, "thousand": 1000,
    "lakh": 100000, "lac": 100000, "crore": 10000000,
}


def _energy_voice_activity(
    audio_chunk: bytes,
    *,
    threshold: float = 400.0,
) -> bool:
    """
    RMS energy fallback when WebRTC VAD is unavailable.

    Args:
        audio_chunk: PCM16 mono bytes.
        threshold: RMS above this counts as speech.
    """
    samples = np.frombuffer(audio_chunk, dtype=np.int16)
    if samples.size == 0:
        return False
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    return rms >= threshold


def _webrtc_voice_activity(
    audio_chunk: bytes,
    *,
    sample_rate: int,
    frame_ms: int,
    aggressiveness: int,
) -> bool:
    """WebRTC VAD over fixed frames."""
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sample_rate * frame_ms / 1000) * 2
    for i in range(0, len(audio_chunk) - frame_len + 1, frame_len):
        frame = audio_chunk[i : i + frame_len]
        if len(frame) < frame_len:
            break
        if vad.is_speech(frame, sample_rate):
            return True
    return False


def detect_voice_activity(
    audio_chunk: bytes,
    *,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    energy_threshold: Optional[float] = None,
) -> bool:
    """
    Return True if any frame in the buffer contains speech (WebRTC VAD).

    Falls back to RMS energy detection if webrtcvad cannot load (e.g. missing
    pkg_resources / setuptools on Python 3.12+).

    Args:
        audio_chunk: PCM16 mono bytes.
        sample_rate: Must be 8000, 16000, 32000, or 48000.
        frame_ms: 10, 20, or 30.
        aggressiveness: 0-3.
        energy_threshold: RMS threshold for fallback VAD.
    """
    global _webrtc_vad_warned
    if not audio_chunk:
        return False

    settings = get_settings()
    threshold = energy_threshold if energy_threshold is not None else settings.vad_energy_threshold

    try:
        return _webrtc_voice_activity(
            audio_chunk,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            aggressiveness=aggressiveness,
        )
    except Exception as exc:
        if not _webrtc_vad_warned:
            logger.warning(
                "WebRTC VAD unavailable (%s); using energy-based VAD. "
                "Install setuptools: pip install setuptools",
                exc,
            )
            _webrtc_vad_warned = True
        return _energy_voice_activity(audio_chunk, threshold=threshold)


def normalize_audio(audio_chunk: bytes, target_db: float = -20.0) -> bytes:
    """
    Peak-normalize PCM16 mono audio toward target dBFS.

    Args:
        audio_chunk: PCM16 bytes.
        target_db: Target peak level in dBFS.
    """
    if not audio_chunk:
        return audio_chunk
    try:
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        peak = np.max(np.abs(samples))
        if peak < 1:
            return audio_chunk
        target_peak = 32767 * (10 ** (target_db / 20))
        gain = target_peak / peak
        normalized = np.clip(samples * gain, -32768, 32767).astype(np.int16)
        return normalized.tobytes()
    except Exception as exc:
        logger.error("normalize_audio failed: %s", exc)
        return audio_chunk


def pcm_duration_seconds(audio_chunk: bytes, sample_rate: int = 16000) -> float:
    """Duration of PCM16 mono buffer in seconds."""
    if not audio_chunk:
        return 0.0
    return len(audio_chunk) / (2 * sample_rate)


def hindi_number_normalize(text: str) -> str:
    """
    Replace spoken Hindi/English number phrases with digits where possible.

    Example: "do hazaar" → "2000"
    """
    if not text:
        return text
    lowered = text.lower()

    def replace_phrase(match: re.Match) -> str:
        parts = match.group(0).lower().split()
        total = 0
        current = 0
        for p in parts:
            if p in _HINDI_NUMBERS:
                val = _HINDI_NUMBERS[p]
                if val >= 1000:
                    current = max(current, 1) * val
                    total += current
                    current = 0
                else:
                    current += val
        total += current
        return str(total) if total else match.group(0)

    pattern = r"\b(?:(?:{})\s*)+\b".format("|".join(re.escape(k) for k in _HINDI_NUMBERS))
    return re.sub(pattern, replace_phrase, lowered, flags=re.IGNORECASE)


def save_session_metrics(session_id: str, metrics_dict: dict[str, Any], log_dir: Optional[Path] = None) -> Path:
    """
    Persist per-call metrics JSON to logs/{session_id}.json.

    Args:
        session_id: Call session UUID.
        metrics_dict: Metrics payload.
        log_dir: Override log directory.

    Returns:
        Path to written file.
    """
    settings = get_settings()
    directory = log_dir or settings.log_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.json"
    try:
        path.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError as exc:
        logger.error("save_session_metrics failed: %s", exc)
    return path


def load_conversation_state(session_id: str, log_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """
    Load saved conversation/metrics state for call resume.

    Args:
        session_id: Session id.
        log_dir: Override log directory.
    """
    settings = get_settings()
    directory = log_dir or settings.log_path
    for name in (f"{session_id}_state.json", f"{session_id}.json"):
        path = directory / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("load_conversation_state failed: %s", exc)
    return None


def append_transcript_entry(
    metrics: dict[str, Any],
    role: str,
    content: str,
) -> None:
    """Append a transcript line with UTC timestamp."""
    metrics.setdefault("transcript", []).append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def estimate_costs(metrics: dict[str, Any], settings: Optional[Settings] = None) -> dict[str, float]:
    """Compute estimated USD costs from usage counters in metrics."""
    settings = settings or get_settings()
    prompt = metrics.get("total_prompt_tokens", 0)
    completion = metrics.get("total_completion_tokens", 0)
    tts_chars = metrics.get("total_tts_characters", 0)
    stt_seconds = metrics.get("total_stt_seconds", 0)

    llm = (prompt / 1000) * settings.cost_per_1k_prompt_tokens + (
        completion / 1000
    ) * settings.cost_per_1k_completion_tokens
    tts = tts_chars * settings.cost_per_tts_character
    stt = stt_seconds * settings.cost_per_stt_second
    total = llm + tts + stt

    metrics["estimated_llm_cost_usd"] = round(llm, 6)
    metrics["estimated_tts_cost_usd"] = round(tts, 6)
    metrics["estimated_stt_cost_usd"] = round(stt, 6)
    metrics["total_estimated_cost_usd"] = round(total, 6)
    return {
        "llm": llm,
        "tts": tts,
        "stt": stt,
        "total": total,
    }


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Convert Twilio mulaw 8kHz to PCM16 (for STT upsampling path)."""
    import audioop

    return audioop.ulaw2lin(mulaw_bytes, 2)


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Simple linear resample PCM16 mono."""
    if from_rate == to_rate or not pcm:
        return pcm
    import audioop

    pcm, _ = audioop.ratecv(pcm, 2, 1, from_rate, to_rate, None)
    return pcm


def prepare_pcm_for_stt(
    pcm: bytes,
    *,
    target_rate: int = 16000,
    source_rate: Optional[int] = None,
    min_ms: int = 300,
) -> tuple[Optional[bytes], int]:
    """
    Normalize PCM for STT APIs: even length, resample, minimum duration.

    Returns:
        (pcm_bytes, sample_rate) or (None, 0) if audio is too short/invalid.
    """
    if not pcm:
        return None, 0

    if len(pcm) % 2 != 0:
        pcm = pcm[:-1]
    if not pcm:
        return None, 0

    src_rate = source_rate or target_rate
    if src_rate != target_rate:
        pcm = resample_pcm16(pcm, src_rate, target_rate)

    duration_ms = pcm_duration_seconds(pcm, target_rate) * 1000
    if duration_ms < min_ms:
        logger.warning(
            "STT audio too short: %.0fms (min %dms); skipping transcription",
            duration_ms,
            min_ms,
        )
        return None, 0

    return pcm, target_rate


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono in a WAV container for STT APIs."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
