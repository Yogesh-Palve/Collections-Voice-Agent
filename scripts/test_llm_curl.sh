#!/usr/bin/env bash
# Quick manual test — start server first: uvicorn main:app --reload

BASE="${BASE_URL:-http://localhost:8000}"

echo "=== Trigger forward-flow outbound session ==="
curl -s -X POST "$BASE/api/calls/trigger-forward-flow" \
  -H "Content-Type: application/json" \
  -d '{
    "be_id": "BE-2041",
    "be_name": "Ramesh Kumar",
    "be_phone": "+919876543210",
    "be_zone": "Pune West",
    "target_pct": 8,
    "actual_pct": 10
  }' | python -m json.tool
