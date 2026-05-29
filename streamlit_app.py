"""
Streamlit entry — run: streamlit run streamlit_app.py
Multi-page app includes pages/dashboard.py for Forward Flow metrics.
"""

import streamlit as st

st.set_page_config(
    page_title="Collections Voice Agent",
    page_icon="📞",
    layout="wide",
)

st.title("Collections Voice Agent")
st.markdown(
    """
**Sidebar pages:**
- **Dashboard** — Forward Flow calls, follow-ups, charts
- **Metrics** — per-call token/STT/TTS costs from `logs/*.json`

**API (FastAPI):** `uvicorn main:app --reload` → [Web UI](http://localhost:8000)

**Text demo:** `streamlit run app.py`

**Trigger outbound forward-flow:**
```bash
curl -X POST http://localhost:8000/api/calls/trigger-forward-flow -H "Content-Type: application/json" -d "{\"be_id\":\"BE-2041\",\"be_name\":\"Ramesh Kumar\",\"be_phone\":\"+919876543210\",\"be_zone\":\"Pune West\",\"target_pct\":8,\"actual_pct\":10}"
```
"""
)
