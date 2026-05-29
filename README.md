# Collections Voice Agent

Production-oriented voice pipeline for collections teams and Business Executives (BEs): **STT → LLM → TTS**, with web (FastAPI/WebSocket) and telephony (Twilio).

## Setup

```powershell
cd collections-voice-agent
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Add API keys: GROQ, DEEPGRAM, ELEVENLABS, etc.
```

## Run

| Service | Command |
|---------|---------|
| API + Web UI | `uvicorn main:app --reload` → http://localhost:8000 |
| Streamlit demo | `streamlit run app.py` |
| Multi-page (dashboard + metrics) | `streamlit run streamlit_app.py` |

## Architecture

```
User/BE (phone/web) → VAD → STT → LLM → TTS → user
```

- **STT**: `STT_PROVIDER=deepgram` (fallback: ElevenLabs)
- **LLM**: Groq `llama3-70b-8192` + extraction `llama3-8b-8192`
- **TTS**: `TTS_PROVIDER=elevenlabs` or `sarvam`

## Key APIs

- `POST /api/call/start` — start session
- `POST /api/call/end` — end + metrics → `logs/{session_id}.json`
- `GET /api/call/metrics/{session_id}`
- `WebSocket /ws/audio` — real-time voice
- `POST /api/calls/trigger-forward-flow` — outbound forward-flow call
- `POST /twilio/voice` — Twilio TwiML (set `TWILIO_WEBHOOK_BASE_URL`)

## Forward Flow data

Dual-write: `data/collections_log.xlsx` + `data/collections.db`

## Tests

```powershell
pytest tests/ -v
```

## Project layout

See repo root: `stt.py`, `tts.py`, `llm.py`, `utils.py`, `voice_pipeline.py`, `main.py`, `app.py`, `streamlit_app.py`, `pages/`, `flows/`, `data_logger.py`.
