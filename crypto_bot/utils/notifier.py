from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import requests


@dataclass
class TelegramNotifier:
    enabled: bool
    token: Optional[str] = None
    chat_id: Optional[str] = None
    _offset: int = 0

    def send(self, message: str) -> None:
        if not self.enabled or not self.token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            requests.post(url, json=payload, timeout=10)
        except requests.RequestException:
            pass

    def poll_commands(self, handler: Callable[[str], str]) -> None:
        if not self.enabled or not self.token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"timeout": 1, "offset": self._offset + 1}
        try:
            response = requests.get(url, params=params, timeout=10)
            payload = response.json()
        except Exception:
            return

        for update in payload.get("result", []):
            self._offset = max(self._offset, int(update.get("update_id", 0)))
            message = update.get("message", {})
            text = (message.get("text") or "").strip()
            chat_id = str(message.get("chat", {}).get("id", ""))
            if chat_id != str(self.chat_id):
                continue
            if text.startswith("/"):
                reply = handler(text)
                if reply:
                    self.send(reply)
