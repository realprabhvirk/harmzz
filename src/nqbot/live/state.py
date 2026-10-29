"""Crash-safe state persistence (atomic JSON) and an append-only trade log."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.trades_path = self.dir / "trades.jsonl"
        self.events_path = self.dir / "events.jsonl"

    def load(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        with open(self.state_path, encoding="utf-8") as fh:
            return json.load(fh)

    def save(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, default=_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    def append_trade(self, trade: dict[str, Any]) -> None:
        self._append(self.trades_path, trade)

    def log_event(self, event: dict[str, Any]) -> None:
        self._append(self.events_path, event)

    def trades(self) -> list[dict[str, Any]]:
        if not self.trades_path.exists():
            return []
        return [json.loads(x) for x in self.trades_path.read_text().splitlines() if x.strip()]

    @staticmethod
    def _append(path: Path, rec: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=_default) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
