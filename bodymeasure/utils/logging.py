"""Logging setup. Use `get_logger(__name__)` everywhere instead of `print`."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_configured = False


def configure(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root = logging.getLogger("bodymeasure")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure()
    if not name.startswith("bodymeasure"):
        name = f"bodymeasure.{name.split('.', 1)[-1]}"
    return logging.getLogger(name)
