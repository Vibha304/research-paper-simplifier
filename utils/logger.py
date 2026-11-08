"""Application-wide logging setup (console + rotating file)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "rps"
_configured = False


def setup_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure the ``rps`` logger hierarchy once per process."""
    global _configured
    if _configured:
        return

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the application logger, e.g. ``rps.services.pdf``."""
    short = name.split(".", 1)[-1] if name.startswith(("services", "backend", "frontend")) else name
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")
