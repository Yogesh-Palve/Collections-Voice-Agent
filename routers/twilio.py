"""
Twilio voice webhooks and Media Streams WebSocket.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from config import get_settings
from logging_utils import setup_error_logger
from session_store import get_pipeline, register_pipeline
from utils import mulaw_to_pcm16, resample_pcm16
from voice_pipeline import VoicePipeline

logger = setup_error_logger(__name__)
router = APIRouter(prefix="/twilio", tags=["twilio"])


@router.post("/voice")
async def twilio_voice(request: Request, session_id: Optional[str] = None) -> Response:
    """
    Incoming/outbound call TwiML — connect to Media Stream.

    Query param session_id links to pre-created VoicePipeline session.
    """
    settings = get_settings()
    base = settings.twilio_webhook_base_url.rstrip("/")
    stream_url = base.replace("https://", "wss://").replace("http://", "ws://")
    sid = session_id or ""
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}/twilio/media">
      <Parameter name="session_id" value="{sid}" />
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/media")
async def twilio_media(websocket: WebSocket) -> None:
    """
    Twilio Media Streams: mulaw 8kHz → pipeline → mulaw back.

    Uses same VoicePipeline as web interface.
    """
    await websocket.accept()
    settings = get_settings()
    pipeline: Optional[VoicePipeline] = None
    stream_sid: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                start = msg.get("start", {})
                stream_sid = start.get("streamSid")
                params = {p["name"]: p["value"] for p in start.get("customParameters", [])}
                sid = params.get("session_id") or start.get("callSid", "")
                pipeline = get_pipeline(sid)
                if not pipeline:
                    pipeline = VoicePipeline(session_id=sid or None, settings=settings)
                    register_pipeline(pipeline)
                    text, audio = pipeline.opening_audio()
                    if audio:
                        await _send_mulaw(websocket, stream_sid, audio)

            elif event == "media" and pipeline and stream_sid:
                payload = base64.b64decode(msg["media"]["payload"])
                pcm8 = mulaw_to_pcm16(payload)
                pcm16 = resample_pcm16(pcm8, 8000, settings.sample_rate)
                if pipeline.ingest_audio(pcm16):
                    pass
                # End of utterance heuristic: silence not tracked on twilio simply;
                # process on 'stop' or after mark — for demo, process each media batch with VAD end via timer
            elif event == "stop" and pipeline and stream_sid:
                user_text, reply, audio = pipeline.flush_utterance()
                if audio:
                    await _send_mulaw(websocket, stream_sid, audio)

    except WebSocketDisconnect:
        if pipeline:
            pipeline.end_session(call_dropped=True)
    except Exception as exc:
        logger.error("Twilio media stream error: %s", exc)


async def _send_mulaw(websocket: WebSocket, stream_sid: str, audio_mp3: bytes) -> None:
    """Send audio back to Twilio (expects mulaw payload in media event)."""
    # Twilio expects mulaw 8k; if MP3, skip conversion in minimal impl — send empty on mp3
    # Production: decode mp3 → pcm → mulaw. Here base64 pcm placeholder if already pcm16
    import audioop

    try:
        if audio_mp3[:3] == b"ID3" or audio_mp3[:2] == b"\xff\xfb":
            return  # MP3 not converted in stub; use web client for full audio
        pcm = audio_mp3
        pcm8, _ = audioop.ratecv(pcm, 2, 1, 16000, 8000, None)
        mulaw = audioop.lin2ulaw(pcm8, 2)
        payload = base64.b64encode(mulaw).decode("ascii")
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
            )
        )
    except Exception as exc:
        logger.error("Send mulaw failed: %s", exc)
