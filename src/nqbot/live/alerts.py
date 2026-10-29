"""Optional alert webhooks (Slack or Telegram). Secrets come from env vars only.

    Slack:    NQBOT_SLACK_WEBHOOK=https://hooks.slack.com/services/...
    Telegram: NQBOT_TELEGRAM_TOKEN=...  NQBOT_TELEGRAM_CHAT=...

Alerting never raises: a failed webhook is logged and trading continues.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections.abc import Callable
from typing import Any

log = logging.getLogger("nqbot.alerts")

Sender = Callable[[str, bytes, dict[str, str]], None]


def _http_post(url: str, body: bytes, headers: dict[str, str]) -> None:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 (fixed https endpoints)
        resp.read()


class Alerter:
    def __init__(self, kind: str = "none", env: dict[str, str] | None = None, sender: Sender | None = None) -> None:
        self.kind = kind
        self.env = dict(os.environ if env is None else env)
        self.sender = sender or _http_post
        self.sent: list[str] = []
        self.failures = 0

    def _target(self, text: str) -> tuple[str, dict[str, Any]] | None:
        if self.kind == "slack":
            url = self.env.get("NQBOT_SLACK_WEBHOOK")
            return (url, {"text": text}) if url else None
        if self.kind == "telegram":
            token, chat = self.env.get("NQBOT_TELEGRAM_TOKEN"), self.env.get("NQBOT_TELEGRAM_CHAT")
            if token and chat:
                return f"https://api.telegram.org/bot{token}/sendMessage", {"chat_id": chat, "text": text}
        return None

    def send(self, level: str, text: str) -> None:
        msg = f"[nqbot {level.upper()}] {text}"
        self.sent.append(msg)
        log.log(logging.WARNING if level in ("halt", "error") else logging.INFO, msg)
        target = self._target(msg)
        if target is None:
            return
        url, payload = target
        try:
            self.sender(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        except Exception as exc:  # noqa: BLE001 - alerts must never crash trading
            self.failures += 1
            log.error("alert webhook failed: %s", exc)
