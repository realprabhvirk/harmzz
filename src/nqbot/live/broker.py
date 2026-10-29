"""Broker interface and the SimBroker used for replay.

SimBroker fills orders with the SAME functions the backtester uses
(``nqbot.core.market_fill`` / ``bracket_check``), so a replay through the live
runner reproduces backtest trades exactly.

Order timing semantics: an order submitted at time T executes at the open of
the first bar whose label is >= T (for SimBroker: the next processed bar).
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from nqbot.core import (
    EXIT_END_OF_DATA,
    EXIT_NONE,
    EXIT_STOP,
    EXIT_TARGET,
    P_ENTRY_DELAY,
    bracket_check,
    market_fill,
)
from nqbot.live.feed import Bar


class BrokerError(RuntimeError):
    pass


class OrderRejected(BrokerError):
    pass


@dataclass
class Fill:
    kind: str  # "entry" | "exit"
    side: int  # +1 buy, -1 sell
    qty: int
    price: float
    theo: float
    reason: int  # exit reason for exits, EXIT_NONE for entries
    label_min: float  # utc minute of the bar (sim) / of the fill time (live)
    order_id: str


@dataclass
class WorkingOrder:
    order_id: str
    role: str  # "entry" | "stop" | "target" | "flatten"
    side: int
    qty: int
    price: float = float("nan")


@dataclass
class BrokerEvent:
    kind: str  # "reject" | "cancel" | "expired"
    order_id: str
    role: str
    message: str = ""


class Broker(ABC):
    """Synchronous broker interface used by the live runner."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def position(self) -> int:
        """Signed position in contracts (the broker's best current knowledge)."""

    @abstractmethod
    def working_orders(self) -> list[WorkingOrder]: ...

    @abstractmethod
    def place_bracket(self, direction: int, qty: int, stop: float, target: float, submit_min: float) -> str:
        """Market entry + OCO (stop, limit target). Raises OrderRejected if the
        protective stop cannot be confirmed working."""

    @abstractmethod
    def place_protection(self, side: int, qty: int, stop: float, target: float | None) -> str:
        """Standalone protective stop (+ optional OCO target) for an existing position.
        ``side`` is the side of the closing order. Raises OrderRejected on failure."""

    @abstractmethod
    def resize_protection(self, qty: int) -> None: ...

    @abstractmethod
    def flatten(self, reason: int, submit_min: float) -> None:
        """Cancel all working orders and close any position at market."""

    @abstractmethod
    def cancel_all(self) -> None: ...

    @abstractmethod
    def cancel_entry(self, order_id: str) -> None:
        """Cancel the unfilled remainder of an entry order."""

    @abstractmethod
    def poll(self) -> tuple[list[Fill], list[BrokerEvent]]:
        """New fills and order events since the last call."""

    @abstractmethod
    def last_price(self) -> float | None: ...


_ids = itertools.count(1)


@dataclass
class _Bracket:
    order_id: str
    direction: int
    qty: int
    stop: float
    target: float
    submit_min: float
    filled: bool = False


@dataclass
class SimBroker(Broker):
    """Bar-replay broker. Call ``process_bar`` for each bar, ``end_of_data`` at the end."""

    params: np.ndarray
    connected: bool = False
    _pos: int = 0
    _bracket: _Bracket | None = None
    _protect: tuple[str, int, float, float] | None = None  # (id, side, stop, target)
    _flatten: tuple[str, int, float] | None = None  # (id, reason, submit_min)
    _fills: list[Fill] = field(default_factory=list)
    _events: list[BrokerEvent] = field(default_factory=list)
    _last: float | None = None
    reject_next_bracket: bool = False  # test hook

    def connect(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def position(self) -> int:
        return self._pos

    def last_price(self) -> float | None:
        return self._last

    def working_orders(self) -> list[WorkingOrder]:
        out: list[WorkingOrder] = []
        b = self._bracket
        if b is not None:
            if not b.filled:
                out.append(WorkingOrder(b.order_id, "entry", b.direction, b.qty))
            out.append(WorkingOrder(b.order_id + ":stop", "stop", -b.direction, b.qty, b.stop))
            out.append(WorkingOrder(b.order_id + ":target", "target", -b.direction, b.qty, b.target))
        if self._protect is not None:
            pid, side, stop, tgt = self._protect
            out.append(WorkingOrder(pid + ":stop", "stop", side, abs(self._pos), stop))
            if np.isfinite(tgt):
                out.append(WorkingOrder(pid + ":target", "target", side, abs(self._pos), tgt))
        if self._flatten is not None:
            out.append(WorkingOrder(self._flatten[0], "flatten", -int(np.sign(self._pos)), abs(self._pos)))
        return out

    def place_bracket(self, direction: int, qty: int, stop: float, target: float, submit_min: float) -> str:
        if self.reject_next_bracket:
            self.reject_next_bracket = False
            raise OrderRejected("simulated reject")
        if self._bracket is not None or self._pos != 0:
            raise BrokerError("bracket already working or position open")
        oid = f"sim{next(_ids)}"
        self._bracket = _Bracket(oid, direction, qty, stop, target, submit_min)
        return oid

    def place_protection(self, side: int, qty: int, stop: float, target: float | None) -> str:
        oid = f"sim{next(_ids)}"
        self._protect = (oid, side, stop, float("nan") if target is None else target)
        return oid

    def resize_protection(self, qty: int) -> None:
        if self._bracket is not None:
            self._bracket.qty = qty

    def flatten(self, reason: int, submit_min: float) -> None:
        if self._bracket is not None and not self._bracket.filled:
            self._events.append(BrokerEvent("cancel", self._bracket.order_id, "entry"))
            self._bracket = None
        if self._pos != 0:
            self._flatten = (f"sim{next(_ids)}", reason, submit_min)

    def cancel_entry(self, order_id: str) -> None:
        b = self._bracket
        if b is not None and b.order_id == order_id and not b.filled:
            self._bracket = None
            self._events.append(BrokerEvent("cancel", order_id, "entry"))

    def cancel_all(self) -> None:
        self._bracket = None
        self._protect = None
        self._flatten = None

    def poll(self) -> tuple[list[Fill], list[BrokerEvent]]:
        f, e = self._fills, self._events
        self._fills, self._events = [], []
        return f, e

    # ----- simulation -----
    def _exit(self, theo: float, fill: float, reason: int, label_min: float, oid: str) -> None:
        side = -int(np.sign(self._pos))
        self._fills.append(Fill("exit", side, abs(self._pos), fill, theo, reason, label_min, oid))
        self._pos = 0
        self._bracket = None
        self._protect = None
        self._flatten = None

    def process_bar(self, bar: Bar) -> None:
        p = self.params
        o, h, low = bar.open, bar.high, bar.low
        if self._flatten is not None and self._pos != 0:
            oid, reason, _ = self._flatten
            side = -int(np.sign(self._pos))
            self._exit(o, market_fill(o, side, p), reason, bar.label_min, oid)
        b = self._bracket
        if b is not None and not b.filled:
            if bar.label_min - b.submit_min > p[P_ENTRY_DELAY]:
                self._events.append(BrokerEvent("expired", b.order_id, "entry"))
                self._bracket = None
            else:
                fill = market_fill(o, b.direction, p)
                b.filled = True
                self._pos = b.direction * b.qty
                self._fills.append(Fill("entry", b.direction, b.qty, fill, o, EXIT_NONE, bar.label_min, b.order_id))
        if self._pos != 0:
            if self._bracket is not None:
                stop, tgt, oid = self._bracket.stop, self._bracket.target, self._bracket.order_id
            elif self._protect is not None:
                oid, _, stop, tgt = self._protect
                if not np.isfinite(tgt):
                    tgt = np.inf if self._pos > 0 else -np.inf
            else:
                stop = tgt = float("nan")
                oid = ""
            if np.isfinite(stop):
                code, theo, fill = bracket_check(int(np.sign(self._pos)), stop, tgt, o, h, low, p)
                if code in (EXIT_STOP, EXIT_TARGET):
                    self._exit(theo, fill, code, bar.label_min, f"{oid}:{'stop' if code == EXIT_STOP else 'target'}")
        self._last = bar.close

    def end_of_data(self, bar: Bar) -> None:
        if self._pos != 0:
            side = -int(np.sign(self._pos))
            self._exit(bar.close, market_fill(bar.close, side, self.params), EXIT_END_OF_DATA, bar.label_min, "eod")
        self.cancel_all()
