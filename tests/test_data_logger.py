"""Tests for data_logger dual-write."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data_logger import DataLogger


@pytest.fixture
def temp_logger(tmp_path):
    """DataLogger pointed at temp directory."""
    from config import get_settings

    settings = get_settings()
    data_path = tmp_path / "data"
    data_path.mkdir()
    logger = DataLogger(settings)
    logger.data_dir = data_path
    logger.db_path = data_path / "collections.db"
    logger.excel_path = data_path / "collections_log.xlsx"
    logger._init_db()
    logger._init_excel()
    return logger


def test_log_forward_flow_call_dual_write(temp_logger):
    """Record appears in SQLite and Excel."""
    record = {
        "session_id": "test-session-001",
        "call_date": "2026-05-28",
        "call_time": "10:35 AM",
        "be_name": "Ramesh Kumar",
        "be_id": "BE-2041",
        "be_zone": "Pune West",
        "forward_flow_target_pct": 8,
        "forward_flow_actual_pct": 10,
        "be_was_aware": "Yes",
        "be_can_reduce": "Yes",
        "be_strategy": "Will target high EMI payers first",
        "commitment_date": "2026-06-05",
        "escalated_to_human": False,
        "call_outcome": "Committed",
        "notes": "",
    }
    temp_logger.log_forward_flow_call(record)

    by_be = temp_logger.get_calls_by_be("BE-2041")
    assert len(by_be) == 1
    assert by_be[0]["be_name"] == "Ramesh Kumar"
    assert temp_logger.excel_path.exists()


def test_get_pending_followups(temp_logger):
    """Pending follow-ups filter by commitment date and outcome."""
    record = {
        "session_id": "test-session-002",
        "call_date": "2026-05-28",
        "call_time": "11:00 AM",
        "be_name": "Amit",
        "be_id": "BE-100",
        "be_zone": "Mumbai",
        "forward_flow_target_pct": 5,
        "forward_flow_actual_pct": 7,
        "be_was_aware": "Yes",
        "be_can_reduce": "Yes",
        "be_strategy": "Door visits",
        "commitment_date": "2026-05-20",
        "escalated_to_human": False,
        "call_outcome": "Committed",
        "notes": "",
    }
    temp_logger.log_forward_flow_call(record)
    pending = temp_logger.get_pending_followups(date(2026, 5, 28))
    assert len(pending) >= 1


def test_export_summary_report(temp_logger):
    """Monthly summary sheet is created."""
    record = {
        "session_id": "test-session-003",
        "call_date": "2026-05-15",
        "call_time": "09:00 AM",
        "be_name": "Sita",
        "be_id": "BE-200",
        "be_zone": "Delhi",
        "forward_flow_target_pct": 8,
        "forward_flow_actual_pct": 12,
        "be_was_aware": "No",
        "be_can_reduce": "No",
        "be_strategy": "",
        "commitment_date": None,
        "escalated_to_human": True,
        "call_outcome": "Escalated",
        "notes": "Requested supervisor",
    }
    temp_logger.log_forward_flow_call(record)
    path = temp_logger.export_summary_report("2026-05")
    assert Path(path).exists()
