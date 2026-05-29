"""
Voice pipeline orchestrator: VAD → STT → LLM → TTS with metrics and interruption.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from config import Settings, get_settings
from flows.forward_flow_review import ForwardFlowReviewFlow
from llm import LLMClient
from logging_utils import setup_error_logger
from stt import STTProvider, get_stt_provider
from tts import TTSProvider, get_tts_provider
from utils import (
    append_transcript_entry,
    detect_voice_activity,
    estimate_costs,
    hindi_number_normalize,
    normalize_audio,
    pcm_duration_seconds,
    save_session_metrics,
)

logger = setup_error_logger(__name__)


class VoicePipeline:
    """
    End-to-end voice session for web or telephony.

    Supports general collections dialogue and ForwardFlowReviewFlow.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        settings: Optional[Settings] = None,
        stt: Optional[STTProvider] = None,
        tts: Optional[TTSProvider] = None,
        llm: Optional[LLMClient] = None,
        forward_flow: Optional[ForwardFlowReviewFlow] = None,
        call_type: str = "general",
        be_id: str = "",
    ) -> None:
        """Initialize providers and metrics."""
        self.settings = settings or get_settings()
        self.session_id = session_id or str(uuid4())
        self.stt = stt or get_stt_provider(settings=self.settings)
        self.tts = tts or get_tts_provider(settings=self.settings)
        self.llm = llm or LLMClient(self.settings)
        self.forward_flow = forward_flow
        self.call_type = call_type
        self.be_id = be_id

        self._audio_buffer = bytearray()
        self._agent_speaking = False
        self._retry_count = 0
        self.max_retries = 3

        now = datetime.now(timezone.utc)
        self.metrics: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": now.isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "be_id": be_id,
            "call_type": call_type,
            "stt_provider_used": self.stt.name,
            "tts_provider_used": self.tts.name,
            "llm_model_used": self.settings.groq_model,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tts_characters": 0,
            "total_stt_seconds": 0,
            "interruption_count": 0,
            "call_dropped": False,
            "resumed": False,
            "transcript": [],
            "resolution_status": "follow-up-needed",
            "engagement_turns": 0,
        }
        self.conversation: list[dict[str, str]] = []

    def ingest_audio(self, chunk: bytes) -> bool:
        """
        Buffer audio after VAD check.

        Returns:
            True if speech detected in chunk.
        """
        if self._agent_speaking and detect_voice_activity(
            chunk,
            sample_rate=self.settings.sample_rate,
            frame_ms=self.settings.vad_frame_ms,
            aggressiveness=self.settings.vad_aggressiveness,
        ):
            self.handle_interruption()
        normalized = normalize_audio(chunk)
        if detect_voice_activity(
            normalized,
            sample_rate=self.settings.sample_rate,
            frame_ms=self.settings.vad_frame_ms,
            aggressiveness=self.settings.vad_aggressiveness,
        ):
            self._audio_buffer.extend(normalized)
            return True
        return False

    def handle_interruption(self) -> None:
        """Stop TTS when user speaks over agent."""
        self.tts.stop()
        self._agent_speaking = False
        self.metrics["interruption_count"] += 1

    def flush_utterance(
        self,
        on_filler: Optional[Callable[[bytes], None]] = None,
    ) -> tuple[str, str, bytes]:
        """
        Process buffered speech through STT → LLM → TTS.

        Args:
            on_filler: Optional callback with filler audio while LLM runs.

        Returns:
            (user_text, assistant_text, assistant_audio_bytes)
        """
        pcm = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if not pcm:
            return "", "", b""

        start = time.perf_counter()
        user_text, stt_seconds = self._transcribe_with_retry(pcm)
        user_text = hindi_number_normalize(user_text)

        if user_text:
            append_transcript_entry(self.metrics, "user", user_text)
            self.metrics["engagement_turns"] += 1

        filler_audio = b""
        if on_filler and user_text:
            filler_audio = self._play_filler()

        assistant_text, assistant_audio = self._generate_response(user_text)

        if filler_audio and on_filler:
            on_filler(filler_audio)

        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > self.settings.max_response_latency_ms:
            logger.error(
                "Latency budget exceeded: %.0fms > %dms",
                elapsed_ms,
                self.settings.max_response_latency_ms,
            )

        return user_text, assistant_text, assistant_audio

    def _transcribe_with_retry(self, pcm: bytes) -> tuple[str, float]:
        """STT with retry on failure."""
        seconds = pcm_duration_seconds(pcm, self.settings.sample_rate)
        for attempt in range(self.max_retries):
            try:
                text, dur = self.stt.transcribe_with_duration(
                    pcm, sample_rate=self.settings.sample_rate
                )
                if text:
                    self.metrics["total_stt_seconds"] += dur or seconds
                    self._retry_count = 0
                    return text, dur or seconds
            except Exception as exc:
                logger.error("STT attempt %d failed: %s", attempt + 1, exc)
            time.sleep(0.1 * (attempt + 1))
        self.metrics["total_stt_seconds"] += seconds
        return "", seconds

    def _generate_response(self, user_text: str) -> tuple[str, bytes]:
        """LLM + TTS for one turn."""
        if not user_text:
            return "", b""

        if self.forward_flow:
            reply, _ = self.forward_flow.process_be_turn(user_text)
        else:
            self.conversation.append({"role": "user", "content": user_text})
            resp = self.llm.get_response(self.conversation)
            reply = resp.text
            self.metrics["total_prompt_tokens"] += resp.usage.prompt_tokens
            self.metrics["total_completion_tokens"] += resp.usage.completion_tokens
            self.conversation.append({"role": "assistant", "content": reply})

        append_transcript_entry(self.metrics, "assistant", reply)
        self.metrics["total_tts_characters"] += len(reply)

        self.tts.reset()
        self._agent_speaking = True
        audio = self._synthesize_with_retry(reply)
        self._agent_speaking = False
        return reply, audio

    def _synthesize_with_retry(self, text: str) -> bytes:
        """TTS with retry; respects interruption flag."""
        for attempt in range(self.max_retries):
            if self.tts.interrupted:
                return b""
            try:
                audio = self.tts.synthesize(text)
                if audio:
                    return audio
            except Exception as exc:
                logger.error("TTS attempt %d failed: %s", attempt + 1, exc)
            time.sleep(0.1 * (attempt + 1))
        return b""

    def _play_filler(self) -> bytes:
        """Short filler phrase while LLM thinks."""
        phrases = self.settings.filler_phrase_list
        if not phrases:
            return b""
        phrase = random.choice(phrases)
        self.tts.reset()
        return self.tts.synthesize(phrase)

    def opening_audio(self) -> tuple[str, bytes]:
        """First agent turn (forward flow or generic greeting)."""
        if self.forward_flow:
            text = self.forward_flow.process_assistant_opening()
        else:
            agent = self.settings.agent_name
            company = self.settings.company_name
            text = (
                f"Namaste, main {agent} bol rahi hoon {company} se. "
                "Kripya apna naam aur employee ID batayein."
            )
            self.conversation.append({"role": "assistant", "content": text})

        append_transcript_entry(self.metrics, "assistant", text)
        self.metrics["total_tts_characters"] += len(text)
        self.tts.reset()
        return text, self.tts.synthesize(text)

    def end_session(
        self,
        *,
        call_dropped: bool = False,
        resolution_status: str = "follow-up-needed",
    ) -> dict[str, Any]:
        """Finalize metrics and persist to logs."""
        end = datetime.now(timezone.utc)
        self.metrics["end_time"] = end.isoformat()
        start = datetime.fromisoformat(self.metrics["start_time"].replace("Z", "+00:00"))
        self.metrics["duration_seconds"] = (end - start).total_seconds()
        self.metrics["call_dropped"] = call_dropped
        self.metrics["resolution_status"] = resolution_status

        if self.forward_flow:
            if self.forward_flow.state.escalated_to_human:
                self.metrics["resolution_status"] = "escalated"
            elif self.forward_flow.state.completed:
                self.metrics["resolution_status"] = "resolved"

        self.metrics["total_prompt_tokens"] += self.llm.cumulative_usage.prompt_tokens
        self.metrics["total_completion_tokens"] += self.llm.cumulative_usage.completion_tokens
        estimate_costs(self.metrics, self.settings)
        save_session_metrics(self.session_id, self.metrics)
        return self.metrics
