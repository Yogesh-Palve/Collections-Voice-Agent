"""Central error logging to logs/errors.log."""

from __future__ import annotations

import logging
from pathlib import Path

from config import get_settings


def setup_error_logger(name: str = "collections_voice_agent") -> logging.Logger:
    """Configure module logger with file handler for errors."""
    settings = get_settings()
    log_dir = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    error_file = log_dir / "errors.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(error_file, encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
