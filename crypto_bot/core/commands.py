from __future__ import annotations

"""Dosya tabanlı kontrol komutları: dashboard yazar, bot okur.

Bot ve dashboard ayrı süreçlerdir (docker'da ayrı konteynerler); paylaştıkları
tek şey runtime/ dizinidir. Her komut ayrı bir JSON dosyasıdır; atomik yazılır
(tmp + rename) ve bot işledikten sonra silinir.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_ACTIONS = {"close_position", "close_all"}


def write_command(commands_dir: str | Path, action: str, **payload: Any) -> dict[str, Any]:
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unknown command: {action}")
    directory = Path(commands_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    command = {"id": uuid.uuid4().hex[:12], "action": action, "created_at": created.isoformat(), **payload}
    name = f"{created.strftime('%Y%m%dT%H%M%S%f')}_{command['id']}.json"
    tmp = directory / f".{name}.tmp"
    tmp.write_text(json.dumps(command), encoding="utf-8")
    os.replace(tmp, directory / name)
    return command


def pending_commands(commands_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(commands_dir)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def pop_commands(commands_dir: str | Path) -> list[dict[str, Any]]:
    """Bekleyen komutları eski→yeni sırayla döndürür ve dosyalarını siler."""
    directory = Path(commands_dir)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            command = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            command = None
        path.unlink(missing_ok=True)
        if isinstance(command, dict) and command.get("action") in ALLOWED_ACTIONS:
            out.append(command)
    return out
