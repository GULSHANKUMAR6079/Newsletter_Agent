"""
app/utils/logging_utils.py
──────────────────────────
Centralised logger factory for the Blog Writing Agent.

Usage:
    from app.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Router decided: %s", mode)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    return handler


@lru_cache(maxsize=None)
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with the given name.
    Uses lru_cache so the same logger is always returned for the same name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_build_handler())
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_global_level(level: int) -> None:
    """Adjust the global logging level at runtime (e.g. from debug toggle)."""
    logging.getLogger().setLevel(level)
