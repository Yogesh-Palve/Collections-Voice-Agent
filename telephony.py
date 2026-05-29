"""
Twilio helpers for outbound forward-flow calls.
"""

from __future__ import annotations

from typing import Optional

from config import Settings, get_settings
from logging_utils import setup_error_logger

logger = setup_error_logger(__name__)


def trigger_outbound_call(
    be_phone: str,
    session_id: str,
    settings: Optional[Settings] = None,
) -> Optional[str]:
    """
    Place outbound Twilio call with Media Stream URL for session.

    Args:
        be_phone: E.164 phone number.
        session_id: Active session id for webhook context.

    Returns:
        Twilio Call SID if created, else None when not configured.
    """
    settings = settings or get_settings()
    if not all(
        [
            settings.twilio_account_sid,
            settings.twilio_auth_token,
            settings.twilio_phone_number,
            settings.twilio_webhook_base_url,
        ]
    ):
        logger.error("Twilio not fully configured; skipping outbound dial")
        return None

    try:
        from twilio.rest import Client

        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        webhook = (
            f"{settings.twilio_webhook_base_url.rstrip('/')}"
            f"/twilio/voice?session_id={session_id}"
        )
        call = client.calls.create(
            to=be_phone,
            from_=settings.twilio_phone_number,
            url=webhook,
            method="POST",
        )
        return call.sid
    except Exception as exc:
        logger.error("Twilio outbound call failed: %s", exc)
        return None
