"""
FastAPI application: web UI, WebSocket audio pipeline, Twilio, REST APIs.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from logging_utils import setup_error_logger
from routers.calls import router as calls_router
from routers.forward_flow import router as forward_flow_router
from routers.twilio import router as twilio_router
from session_store import get_pipeline, register_pipeline, save_flow_state
from voice_pipeline import VoicePipeline

logger = setup_error_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Collections Voice Agent",
    description="STT → LLM → TTS pipeline for collections / BE communication",
    version="1.0.0",
)

app.include_router(forward_flow_router)
app.include_router(calls_router)
app.include_router(twilio_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    """Serve web voice interface."""
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(html)
    return FileResponse(__file__)  # fallback


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket) -> None:
    """
    Real-time bidirectional audio over WebSocket.

    Client → server JSON:
      {"type":"start","session_id":"...","call_type":"general","be_id":"..."}
      {"type":"audio","data":"<base64 pcm16 16kHz>"}
      {"type":"end_utterance"}
      {"type":"interrupt"}
      {"type":"end_call"}

    Server → client JSON:
      {"type":"transcript","role":"user|assistant","text":"..."}
      {"type":"audio","data":"<base64>","mime":"audio/mpeg"}
      {"type":"metrics","data":{...}}
      {"type":"error","message":"..."}
    """
    await websocket.accept()
    settings = get_settings()
    pipeline: Optional[VoicePipeline] = None

    async def send_json(payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "start":
                sid = msg.get("session_id")
                pipeline = get_pipeline(sid) if sid else None
                if not pipeline:
                    pipeline = VoicePipeline(
                        session_id=sid,
                        settings=settings,
                        call_type=msg.get("call_type", "general"),
                        be_id=msg.get("be_id", ""),
                    )
                    if msg.get("sample_rate"):
                        pipeline.set_input_sample_rate(int(msg["sample_rate"]))
                    register_pipeline(pipeline)
                text, audio = pipeline.opening_audio()
                await send_json({"type": "transcript", "role": "assistant", "text": text})
                if audio:
                    await send_json(
                        {
                            "type": "audio",
                            "data": base64.b64encode(audio).decode("ascii"),
                            "mime": "audio/mpeg",
                        }
                    )
                await send_json({"type": "session", "session_id": pipeline.session_id})

            elif mtype == "audio" and pipeline:
                chunk = base64.b64decode(msg.get("data", ""))
                if msg.get("sample_rate"):
                    pipeline.set_input_sample_rate(int(msg["sample_rate"]))
                pipeline.ingest_audio(chunk, buffer_always=True)

            elif mtype == "end_utterance" and pipeline:
                loop = asyncio.get_event_loop()

                def run_pipeline() -> tuple[str, str, bytes]:
                    return pipeline.flush_utterance()

                user_text, reply, audio = await loop.run_in_executor(None, run_pipeline)
                if user_text:
                    await send_json({"type": "transcript", "role": "user", "text": user_text})
                if reply:
                    await send_json({"type": "transcript", "role": "assistant", "text": reply})
                if audio:
                    await send_json(
                        {
                            "type": "audio",
                            "data": base64.b64encode(audio).decode("ascii"),
                            "mime": "audio/mpeg",
                        }
                    )

            elif mtype == "interrupt" and pipeline:
                pipeline.handle_interruption()
                await send_json({"type": "interrupted"})

            elif mtype == "end_call" and pipeline:
                metrics = pipeline.end_session(
                    call_dropped=msg.get("call_dropped", False),
                    resolution_status=msg.get("resolution_status", "follow-up-needed"),
                )
                save_flow_state(pipeline.session_id, settings.log_path)
                await send_json({"type": "metrics", "data": metrics})
                break

    except WebSocketDisconnect:
        if pipeline:
            pipeline.end_session(call_dropped=True)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
