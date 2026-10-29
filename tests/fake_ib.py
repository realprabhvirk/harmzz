"""A fake ib_insync/ib_async IB object: just enough API surface for IBBroker.

Simulates: order acks, market auto-fills, manual (partial) fills, rejects,
bracket/OCA sibling cancellation, disconnects and failing reconnects, stale
ticks, and delayed position reports. Time only moves through ``sleep``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

T0 = dt.datetime(2024, 3, 12, 15, 0, tzinfo=dt.timezone.utc)


class Event:
    def __init__(self) -> None:
        self.handlers: list[Callable[..., None]] = []

    def __iadd__(self, h: Callable[..., None]) -> "Event":
        self.handlers.append(h)
        return self

    def emit(self, *args: Any) -> None:
        for h in self.handlers:
            h(*args)


@dataclass
class Contract:
    symbol: str = ""
    lastTradeDateOrContractMonth: str = ""
    exchange: str = ""
    currency: str = "USD"
    conId: int = 0


@dataclass
class Order:
    action: str
    totalQuantity: float
    orderType: str
    lmtPrice: float = float("nan")
    auxPrice: float = float("nan")
    orderId: int = 0
    parentId: int = 0
    transmit: bool = True
    tif: str = ""
    outsideRth: bool = False
    ocaGroup: str = ""
    ocaType: int = 0
    orderRef: str = ""


def Future(**kw: Any) -> Contract:  # noqa: N802 - mimics ib_insync
    return Contract(**kw)


def MarketOrder(action: str, qty: float) -> Order:  # noqa: N802
    return Order(action, qty, "MKT")


def LimitOrder(action: str, qty: float, price: float) -> Order:  # noqa: N802
    return Order(action, qty, "LMT", lmtPrice=price)


def StopOrder(action: str, qty: float, price: float) -> Order:  # noqa: N802
    return Order(action, qty, "STP", auxPrice=price)


api = SimpleNamespace(Future=Future, MarketOrder=MarketOrder, LimitOrder=LimitOrder, StopOrder=StopOrder)


@dataclass
class OrderStatus:
    status: str = "PendingSubmit"
    filled: float = 0.0
    remaining: float = 0.0


@dataclass
class Trade:
    contract: Contract
    order: Order
    orderStatus: OrderStatus
    fills: list[Any] = field(default_factory=list)


@dataclass
class Position:
    account: str
    contract: Contract
    position: float
    avgCost: float = 0.0


class FakeIB:
    def __init__(self, price: float = 18000.0) -> None:
        self.now = T0
        self.connected = False
        self.fail_connects = 0
        self.connect_calls = 0
        self._next_id = 100
        self.client = SimpleNamespace(getReqId=self._req_id)
        self.errorEvent = Event()
        self.trades_: list[Trade] = []
        self._by_id: dict[int, Trade] = {}
        self.ticker = SimpleNamespace(last=price, time=T0)
        self.reported_position = 0.0
        self.position_delay = 0.0
        self.ack_delay = 0.0
        self.auto_fill_market = True
        self.reject_types: set[str] = set()  # orderTypes to reject
        self.never_ack_types: set[str] = set()
        self._scheduled: list[tuple[dt.datetime, Callable[[], None]]] = []
        self._held: list[Trade] = []
        self._exec = 0
        self.accounts = ["DU1234567"]
        self.contract = Contract("MNQ", "202406", "CME", "USD", conId=42)

    # ----- connection
    def connect(self, host: str, port: int, clientId: int = 0, timeout: float = 0) -> None:  # noqa: N803
        self.connect_calls += 1
        if self.fail_connects > 0:
            self.fail_connects -= 1
            raise ConnectionRefusedError("fake: TWS not reachable")
        self.connected = True

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:  # noqa: N802
        return list(self.accounts)

    def qualifyContracts(self, c: Contract) -> list[Contract]:  # noqa: N802
        c.conId = 42
        return [c]

    def reqMktData(self, *a: Any) -> Any:  # noqa: N802
        return self.ticker

    # ----- time
    def sleep(self, secs: float = 0.0) -> None:
        self.now += dt.timedelta(seconds=secs)
        self._run_due()

    def clock(self) -> float:
        return self.now.timestamp()

    def _schedule(self, delay: float, fn: Callable[[], None]) -> None:
        if delay <= 0:
            fn()
        else:
            self._scheduled.append((self.now + dt.timedelta(seconds=delay), fn))

    def _run_due(self) -> None:
        due = [x for x in self._scheduled if x[0] <= self.now]
        self._scheduled = [x for x in self._scheduled if x[0] > self.now]
        for _, fn in sorted(due, key=lambda x: x[0]):
            fn()

    def set_price(self, px: float, stale_s: float = 0.0) -> None:
        self.ticker.last = px
        self.ticker.time = self.now - dt.timedelta(seconds=stale_s)

    # ----- orders
    def _req_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def placeOrder(self, contract: Contract, order: Order) -> Trade:  # noqa: N802
        if not self.connected:
            raise ConnectionError("not connected")
        if order.orderId == 0:
            order.orderId = self._req_id()
        if order.orderId in self._by_id:  # modification
            t = self._by_id[order.orderId]
            t.order = order
            t.orderStatus.remaining = order.totalQuantity - t.orderStatus.filled
            return t
        t = Trade(contract, order, OrderStatus(remaining=order.totalQuantity))
        self.trades_.append(t)
        self._by_id[order.orderId] = t
        if not order.transmit:
            self._held.append(t)  # IB holds untransmitted orders until the group is transmitted
            return t
        group_root = order.parentId or order.orderId
        held = [h for h in self._held if h.order.orderId == group_root or h.order.parentId == group_root]
        self._held = [h for h in self._held if h not in held]
        for h in held:
            self._activate(h)
        self._activate(t)
        return t

    def _activate(self, t: Trade) -> None:
        order, contract = t.order, t.contract
        if order.orderType in self.reject_types:
            def rej() -> None:
                t.orderStatus.status = "Inactive"
                self.errorEvent.emit(order.orderId, 201, "Order rejected - reason: fake", contract)
                # IB cancels the rest of a bracket when a child is rejected
                for o in self.trades_:
                    if o.order.parentId == order.parentId and order.parentId and o.orderStatus.status in ("PendingSubmit", "Submitted", "PreSubmitted"):
                        o.orderStatus.status = "Cancelled"
                    if o.order.orderId == order.parentId and o.orderStatus.status != "Filled":
                        o.orderStatus.status = "Cancelled"
            self._schedule(self.ack_delay, rej)
            return
        if order.orderType in self.never_ack_types:
            return

        def ack() -> None:
            if t.orderStatus.status == "PendingSubmit":
                held = order.parentId and self._by_id.get(order.parentId) and self._by_id[order.parentId].orderStatus.status != "Filled"
                t.orderStatus.status = "PreSubmitted" if held else "Submitted"
            if order.orderType == "MKT" and self.auto_fill_market:
                self.fill(order.orderId, order.totalQuantity - t.orderStatus.filled, self.ticker.last)
        self._schedule(self.ack_delay, ack)

    def cancelOrder(self, order: Order) -> None:  # noqa: N802
        t = self._by_id.get(order.orderId)
        if t and t.orderStatus.status not in ("Filled", "Cancelled", "Inactive"):
            t.orderStatus.status = "Cancelled"

    def trades(self) -> list[Trade]:
        return list(self.trades_)

    def openTrades(self) -> list[Trade]:  # noqa: N802
        return [t for t in self.trades_ if t.orderStatus.status in ("PendingSubmit", "PreSubmitted", "Submitted")]

    def positions(self) -> list[Position]:
        if self.reported_position == 0:
            return []
        return [Position(self.accounts[0], self.contract, self.reported_position)]

    # ----- simulation controls
    def fill(self, order_id: int, qty: float, price: float) -> None:
        t = self._by_id[order_id]
        self._exec += 1
        ex = SimpleNamespace(execId=f"e{self._exec}", shares=qty, price=price, side="BOT" if t.order.action == "BUY" else "SLD")
        t.fills.append(SimpleNamespace(execution=ex, time=self.now, contract=t.contract))
        t.orderStatus.filled += qty
        t.orderStatus.remaining = t.order.totalQuantity - t.orderStatus.filled
        t.orderStatus.status = "Filled" if t.orderStatus.remaining <= 0 else "Submitted"
        signed = qty if t.order.action == "BUY" else -qty
        self._schedule(self.position_delay, lambda: self._report(signed))
        if t.orderStatus.status == "Filled":
            # activate bracket children; cancel OCA siblings
            for o in self.trades_:
                if o.order.parentId == order_id and o.orderStatus.status == "PreSubmitted":
                    o.orderStatus.status = "Submitted"
            if t.order.ocaGroup:
                for o in self.trades_:
                    if o is not t and o.order.ocaGroup == t.order.ocaGroup and o.orderStatus.status in ("PendingSubmit", "PreSubmitted", "Submitted"):
                        o.orderStatus.status = "Cancelled"
        elif t.order.ocaGroup:
            # partial fill of an OCA member reduces siblings (ocaType 1 cancels on full fill only here)
            pass

    def _report(self, signed: float) -> None:
        self.reported_position += signed

    def order(self, role: str) -> Trade:
        """Most recent order with this role (from orderRef)."""
        for t in reversed(self.trades_):
            if f":{role}:" in t.order.orderRef:
                return t
        raise KeyError(role)
