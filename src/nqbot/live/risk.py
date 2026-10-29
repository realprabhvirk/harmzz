"""Live risk limits: daily loss, max position, max orders, KILL file, flat window, stale data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nqbot.config import RiskConfig, hhmm
from nqbot.sessions import in_window

STALE_OK = 0
STALE_NO_ENTRIES = 1
STALE_FLATTEN = 2


@dataclass
class RiskManager:
    cfg: RiskConfig
    orders_today: int = 0
    rejects_today: int = 0
    session_id: int = -1

    def new_session(self, session_id: int) -> None:
        if session_id != self.session_id:
            self.session_id = session_id
            self.orders_today = 0
            self.rejects_today = 0

    def kill_switch(self) -> bool:
        return Path(self.cfg.kill_file).exists()

    def in_flat_window(self, et_mod: int) -> bool:
        return in_window(et_mod, hhmm(self.cfg.flat_window_start), hhmm(self.cfg.flat_window_end))

    def daily_loss_breached(self, realized: float, unrealized: float) -> bool:
        return self.cfg.max_daily_loss > 0 and realized + unrealized <= -self.cfg.max_daily_loss

    def check_new_orders(self, n_orders: int, position_after: int) -> str | None:
        """Reason string if the order batch is not allowed, else None."""
        if self.kill_switch():
            return "kill file present"
        if abs(position_after) > self.cfg.max_position:
            return f"position {position_after} would exceed max_position {self.cfg.max_position}"
        if self.orders_today + n_orders > self.cfg.max_orders_per_day:
            return f"max_orders_per_day {self.cfg.max_orders_per_day} reached"
        return None

    def record_orders(self, n: int) -> None:
        self.orders_today += n

    def stale_level(self, seconds_since_bar: float, in_session: bool, has_position: bool) -> int:
        if not in_session:
            return STALE_OK
        if has_position and seconds_since_bar >= self.cfg.stale_flatten_seconds:
            return STALE_FLATTEN
        if seconds_since_bar >= self.cfg.stale_seconds:
            return STALE_NO_ENTRIES
        return STALE_OK
