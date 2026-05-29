"""
LLM client for Groq (LLaMA 3 70B) with token tracking and structured extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from groq import Groq

from config import Settings, get_settings
from logging_utils import setup_error_logger
from models.forward_flow import ForwardFlowExtraction

logger = setup_error_logger(__name__)


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Conversational reply plus usage metadata."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""


@dataclass
class ExtractionResult:
    """Parsed extraction JSON and its token usage."""

    data: ForwardFlowExtraction
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_json: dict = field(default_factory=dict)


class LLMClient:
    """
    Groq-backed LLM with prompts loaded from prompts/ and history trimming.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Load prompts and initialize Groq client."""
        self.settings = settings or get_settings()
        self.model = self.settings.groq_model
        self.extraction_model = getattr(
            self.settings, "groq_extraction_model", self.settings.groq_model
        )
        self.temperature = self.settings.llm_temperature
        self.max_turns = self.settings.max_context_turns

        prompts_dir = self.settings.prompts_dir
        self.system_prompt_template = self._read_prompt(prompts_dir / "system_prompt.txt")
        self.user_prompt_template = self._read_prompt(prompts_dir / "user_prompt.txt")
        self.extraction_prompt = self._read_prompt(prompts_dir / "extraction_prompt.txt")

        self._client: Optional[Groq] = None
        if self.settings.groq_api_key:
            self._client = Groq(api_key=self.settings.groq_api_key)

        self.cumulative_usage = TokenUsage()

    @staticmethod
    def _read_prompt(path: Path) -> str:
        """Read a prompt file; return empty string if missing."""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("Failed to read prompt %s: %s", path, exc)
            return ""

    def build_system_prompt(self, context: Optional[dict[str, Any]] = None) -> str:
        """Fill system prompt template with session/flow context."""
        ctx = context or {}
        defaults = {
            "agent_name": self.settings.agent_name,
            "company_name": self.settings.company_name,
            "be_name": ctx.get("be_name", ""),
            "be_id": ctx.get("be_id", ""),
            "be_zone": ctx.get("be_zone", ""),
            "target_pct": ctx.get("target_pct", ""),
            "actual_pct": ctx.get("actual_pct", ""),
            "current_step": ctx.get("current_step", ""),
        }
        try:
            return self.system_prompt_template.format(**defaults)
        except KeyError:
            return self.system_prompt_template

    def build_user_turn_prompt(
        self,
        *,
        flow_name: str,
        current_step: int,
        total_steps: int,
        step_instruction: str,
        be_last_message: str,
    ) -> str:
        """Build per-turn user instruction from template."""
        return self.user_prompt_template.format(
            flow_name=flow_name,
            current_step=current_step,
            total_steps=total_steps,
            step_instruction=step_instruction,
            be_last_message=be_last_message or "(call start)",
            agent_name=self.settings.agent_name,
        )

    def get_response(
        self,
        conversation_history: list[dict[str, str]],
        context: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate assistant reply from conversation history.

        Args:
            conversation_history: List of {"role": "user"|"assistant", "content": "..."}.
            context: Optional flow/session values for system prompt formatting.

        Returns:
            LLMResponse with text and token usage.
        """
        if not self._client:
            logger.error("Groq client not configured (missing GROQ_API_KEY)")
            return LLMResponse(
                text="I'm sorry, I'm unable to connect right now. Please try again shortly.",
                model=self.model,
            )

        messages = [{"role": "system", "content": self.build_system_prompt(context)}]
        messages.extend(self._trim_history(conversation_history))

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=256,
            )
            text = (completion.choices[0].message.content or "").strip()
            usage = TokenUsage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
            )
            self._accumulate_usage(usage)
            return LLMResponse(text=text, usage=usage, model=self.model)
        except Exception as exc:
            logger.error("LLM get_response failed: %s", exc)
            return LLMResponse(
                text="Sorry, there was a brief technical issue. Could you repeat that?",
                model=self.model,
            )

    def extract_from_be_message(
        self,
        be_message: str,
        *,
        current_step: int,
        context: Optional[dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Lightweight second pass: parse BE speech into structured JSON.

        Args:
            be_message: Latest Business Executive utterance.
            current_step: Forward flow step (1-6).
            context: Optional known BE/flow fields.

        Returns:
            ExtractionResult with ForwardFlowExtraction model.
        """
        empty = ForwardFlowExtraction()
        if not self._client or not be_message.strip():
            return ExtractionResult(data=empty)

        today = date.today().isoformat()
        ctx = context or {}
        user_payload = json.dumps(
            {
                "today": today,
                "current_step": current_step,
                "be_message": be_message,
                "context": ctx,
            },
            ensure_ascii=False,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.extraction_model,
                messages=[
                    {"role": "system", "content": self.extraction_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = (completion.choices[0].message.content or "{}").strip()
            parsed = self._parse_json(raw)
            usage = TokenUsage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
            )
            self._accumulate_usage(usage)
            data = ForwardFlowExtraction.model_validate(parsed)
            return ExtractionResult(data=data, usage=usage, raw_json=parsed)
        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            return ExtractionResult(data=empty)

    def estimate_cost_usd(self, usage: Optional[TokenUsage] = None) -> float:
        """Estimate USD cost for given or cumulative token usage."""
        u = usage or self.cumulative_usage
        s = self.settings
        prompt_cost = (u.prompt_tokens / 1000) * s.cost_per_1k_prompt_tokens
        completion_cost = (u.completion_tokens / 1000) * s.cost_per_1k_completion_tokens
        return round(prompt_cost + completion_cost, 6)

    def reset_usage(self) -> None:
        """Reset per-session cumulative token counters."""
        self.cumulative_usage = TokenUsage()

    def _trim_history(
        self, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Keep last N user/assistant turns to limit tokens."""
        if len(history) <= self.max_turns * 2:
            return history
        return history[-(self.max_turns * 2) :]

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        """Add usage to session cumulative totals."""
        self.cumulative_usage.prompt_tokens += usage.prompt_tokens
        self.cumulative_usage.completion_tokens += usage.completion_tokens

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from model output, tolerating minor formatting issues."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}
