"""
Central configuration loaded from environment variables.

All provider switches and tunables are read here so modules never hardcode keys.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root before settings are instantiated
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings backed by .env / environment."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Providers
    stt_provider: str = Field(default="deepgram", alias="STT_PROVIDER")
    tts_provider: str = Field(default="elevenlabs", alias="TTS_PROVIDER")
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")

    # STT
    deepgram_api_key: str = Field(default="", alias="DEEPGRAM_API_KEY")
    deepgram_model: str = Field(default="nova-2", alias="DEEPGRAM_MODEL")
    deepgram_language: str = Field(default="hi", alias="DEEPGRAM_LANGUAGE")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_stt_fallback: bool = Field(default=False, alias="ELEVENLABS_STT_FALLBACK")
    cartesia_api_key: str = Field(default="", alias="CARTESIA_API_KEY")
    hume_api_key: str = Field(default="", alias="HUME_API_KEY")

    # TTS
    elevenlabs_voice_id: str = Field(default="", alias="ELEVENLABS_VOICE_ID")
    elevenlabs_tts_model: str = Field(
        default="eleven_flash_v2_5", alias="ELEVENLABS_TTS_MODEL"
    )
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    sarvam_speaker: str = Field(default="shubh", alias="SARVAM_SPEAKER")
    sarvam_language: str = Field(default="hi-IN", alias="SARVAM_LANGUAGE")
    sarvam_model: str = Field(default="bulbul:v3", alias="SARVAM_MODEL")
    sarvam_pace: float = Field(default=1.0, alias="SARVAM_PACE")
    sarvam_sample_rate: int = Field(default=22050, alias="SARVAM_SAMPLE_RATE")

    # LLM
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", alias="GROQ_MODEL")
    groq_extraction_model: str = Field(
        default="llama3-8b-8192", alias="GROQ_EXTRACTION_MODEL"
    )
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")

    data_dir: str = Field(default="data", alias="DATA_DIR")

    # Twilio
    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(default="", alias="TWILIO_PHONE_NUMBER")
    twilio_webhook_base_url: str = Field(default="", alias="TWILIO_WEBHOOK_BASE_URL")

    # App
    company_name: str = Field(default="ABC Finance Collections", alias="COMPANY_NAME")
    agent_name: str = Field(default="Priya", alias="AGENT_NAME")
    max_response_latency_ms: int = Field(default=8000, alias="MAX_RESPONSE_LATENCY_MS")
    max_context_turns: int = Field(default=10, alias="MAX_CONTEXT_TURNS")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    # Audio / VAD
    vad_aggressiveness: int = Field(default=2, alias="VAD_AGGRESSIVENESS")
    vad_frame_ms: int = Field(default=30, alias="VAD_FRAME_MS")
    vad_energy_threshold: float = Field(default=400.0, alias="VAD_ENERGY_THRESHOLD")
    min_stt_audio_ms: int = Field(default=300, alias="MIN_STT_AUDIO_MS")
    sample_rate: int = Field(default=16000, alias="SAMPLE_RATE")

    # Cost estimates (USD)
    cost_per_1k_prompt_tokens: float = Field(
        default=0.00059, alias="COST_PER_1K_PROMPT_TOKENS"
    )
    cost_per_1k_completion_tokens: float = Field(
        default=0.00079, alias="COST_PER_1K_COMPLETION_TOKENS"
    )
    cost_per_tts_character: float = Field(default=0.00003, alias="COST_PER_TTS_CHARACTER")
    cost_per_stt_second: float = Field(default=0.0043, alias="COST_PER_STT_SECOND")

    filler_phrases: str = Field(
        default="Hmm, ek second., Ji, sun rahi hoon., One moment please.",
        alias="FILLER_PHRASES",
    )

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    @property
    def project_root(self) -> Path:
        """Absolute path to project root."""
        return _PROJECT_ROOT

    @property
    def log_path(self) -> Path:
        """Absolute path to metrics/log directory."""
        path = self.project_root / self.log_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def prompts_dir(self) -> Path:
        """Directory containing system_prompt.txt and user_prompt.txt."""
        return self.project_root / "prompts"

    @property
    def data_path(self) -> Path:
        """Directory for SQLite, Excel, and persistent call records."""
        path = self.project_root / self.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def filler_phrase_list(self) -> list[str]:
        """Short phrases played while the LLM is thinking."""
        return [p.strip() for p in self.filler_phrases.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
