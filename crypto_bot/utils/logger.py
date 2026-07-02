from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "fields") and isinstance(record.fields, dict):
            payload.update(record.fields)
        return json.dumps(payload, default=str)


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "runtime/logs",
    as_json: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter: logging.Formatter
    if as_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    try:
        file_handler = RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=5_000_000, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError) as _file_err:
        # Dosya yazma yetkisi yoksa yalnızca konsola yaz
        stream_handler.stream.write(
            f"[WARN] Log file unavailable ({_file_err}), using console only.\n"
        )

    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, event: str, message: str, **fields: Any) -> None:
    if fields:
        fields_str = " | " + " ".join(f"{k}={v}" for k, v in fields.items())
    else:
        fields_str = ""
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        f"[{event}] {message}{fields_str}",
        extra={"event": event, "fields": fields},
    )
