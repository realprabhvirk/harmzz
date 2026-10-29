"""Interactive Brokers adapter (ib_async / ib_insync API).

NOT verified against a real TWS/Gateway in this repo: it is tested against
``tests/fake_ib.py``, which mimics the parts of the API used here. Paper-trade
it before trusting it.

Design:
- Every order carries ``orderRef = "<prefix>:<role>:<group>"`` so roles survive restarts.
- Entries are brackets: market parent (DAY) + GTC limit target + GTC stop,
  children linked to the parent (IB treats bracket children as OCO) and given
  an explicit OCA group. Stops/targets use outsideRth so they work overnight.
- Fills are detected by polling ``ib.trades()`` executions (robust to missed
  events). Position is tracked from our own fills, because IB position updates
  can lag; a persistent mismatch with ``ib.positions()`` is adopted and reported.
- ``place_bracket`` / ``place_protection`` wait for the stop to be acknowledged
  and raise ``OrderRejected`` otherwise.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from nqbot.config import LiveConfig
from nqbot.core import EXIT_NONE, EXIT_RISK, EXIT_STOP, EXIT_TARGET
from nqbot.live.broker import Broker, BrokerError, BrokerEvent, Fill, OrderRejected, WorkingOrder

WORKING = {"PendingSubmit", "ApiPending", "PreSubmitted", "Submitted"}
ACKED = {"PreSubmitted", "Submitted"}
DEAD = {"Cancelled", "ApiCancelled", "Inactive"}


class IBBroker(Broker):
    def __init__(
        self,
        ib: Any,
        api: Any,
        cfg: LiveConfig,
        symbol: str,
        contract_month: str,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.time,
        prefix: str = "nqbot",
        stale_tick_s: float = 30.0,
        position_mismatch_s: float = 20.0,
    ) -> None:
        self.ib = ib
        self.api = api
        self.cfg = cfg
        self.symbol = symbol
        self.contract_month = contract_month
        self.sleep = sleep or ib.sleep
        self.clock = clock
        self.prefix = prefix
        self.stale_tick_s = stale_tick_s
        self.position_mismatch_s = position_mismatch_s
        self.contract: Any = None
        self._ticker: Any = None
        self._pos = 0
        self._seen_exec: set[str] = set()
        self._reported: set[int] = set()
        self._our_cancels: set[int] = set()
        self._theo: dict[int, float] = {}
        self._flatten_reason = EXIT_RISK
        self._fills: list[Fill] = []
        self._events: list[BrokerEvent] = []
        self._errors: dict[int, str] = {}
        self._mismatch_since: float | None = None
        self._group = 0
        self._handlers_attached = False

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        if not self.ib.isConnected():
            self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, timeout=10)
        if not self._handlers_attached:
            self.ib.errorEvent += self._on_error
            self._handlers_attached = True
        if self.contract is None:
            c = self.api.Future(
                symbol=self.symbol, lastTradeDateOrContractMonth=self.contract_month,
                exchange=self.cfg.exchange, currency="USD",
            )
            q = self.ib.qualifyContracts(c)
            if not q:
                raise BrokerError(f"could not qualify {self.symbol} {self.contract_month}")
            self.contract = q[0]
        self._ticker = self.ib.reqMktData(self.contract, "", False, False)
        # executions already known to the client belong to the past; position is
        # taken from the broker on (re)connect
        for t in self._our_trades():
            for f in t.fills:
                self._seen_exec.add(f.execution.execId)
        self._pos = self.broker_position()
        self._mismatch_since = None

    def is_connected(self) -> bool:
        return bool(self.ib.isConnected())

    def accounts(self) -> list[str]:
        return list(self.ib.managedAccounts())

    def _on_error(self, req_id: int, code: int, msg: str, *args: Any) -> None:
        self._errors[req_id] = f"{code}: {msg}"

    def _require(self) -> None:
        if not self.is_connected() or self.contract is None:
            raise BrokerError("not connected")

    # --------------------------------------------------------------- helpers
    def _ref(self, role: str, group: str) -> str:
        return f"{self.prefix}:{role}:{group}"

    def _role(self, trade: Any) -> str | None:
        parts = str(getattr(trade.order, "orderRef", "") or "").split(":")
        if len(parts) >= 2 and parts[0] == self.prefix:
            return parts[1]
        return None

    def _ours(self, trade: Any) -> bool:
        if self._role(trade) is None:
            return False
        con = getattr(trade.contract, "conId", None)
        return self.contract is None or con in (None, 0) or con == getattr(self.contract, "conId", con)

    def _our_trades(self) -> list[Any]:
        return [t for t in self.ib.trades() if self._ours(t)]

    def _new_group(self) -> str:
        self._group += 1
        return f"{int(self.clock())}-{self._group}"

    def _order_id(self) -> int:
        return int(self.ib.client.getReqId())

    def _side(self, action: str) -> int:
        return 1 if action.upper() in ("BUY", "BOT") else -1

    def broker_position(self) -> int:
        tot = 0.0
        for p in self.ib.positions():
            c = p.contract
            if getattr(c, "conId", None) == getattr(self.contract, "conId", None) or (
                getattr(c, "symbol", None) == self.symbol
                and str(getattr(c, "lastTradeDateOrContractMonth", "")).startswith(self.contract_month)
            ):
                tot += float(p.position)
        return int(round(tot))

    # ----------------------------------------------------------- scanning
    def _scan(self) -> None:
        for t in self._our_trades():
            role = self._role(t)
            oid = int(t.order.orderId)
            for f in t.fills:
                ex = f.execution
                if ex.execId in self._seen_exec:
                    continue
                self._seen_exec.add(ex.execId)
                side = self._side(ex.side)
                qty = int(round(float(ex.shares)))
                self._pos += side * qty
                price = float(ex.price)
                label = float(math.floor(_ts(f.time).timestamp() / 60))
                if role == "entry":
                    theo = self._theo.get(oid, price)
                    self._fills.append(Fill("entry", side, qty, price, theo, EXIT_NONE, label, str(oid)))
                else:
                    if role == "stop":
                        reason, theo = EXIT_STOP, float(getattr(t.order, "auxPrice", price))
                    elif role == "target":
                        reason, theo = EXIT_TARGET, float(getattr(t.order, "lmtPrice", price))
                    else:
                        reason, theo = self._flatten_reason, self._theo.get(oid, price)
                    self._fills.append(Fill("exit", side, qty, price, theo, reason, label, str(oid)))
            status = t.orderStatus.status
            if status in DEAD and oid not in self._reported:
                self._reported.add(oid)
                filled = float(t.orderStatus.filled or 0)
                if oid in self._our_cancels:
                    kind = "cancel"
                elif status == "Cancelled" and role in ("stop", "target") and filled == 0 and self._sibling_filled(t):
                    kind = "cancel"  # OCO sibling filled
                else:
                    kind = "reject"
                self._events.append(BrokerEvent(kind, str(oid), role or "?", self._errors.get(oid, status)))
        # delayed/mismatched position reports: tolerate briefly, then adopt
        bp = self.broker_position()
        if bp != self._pos:
            now = self.clock()
            if self._mismatch_since is None:
                self._mismatch_since = now
            elif now - self._mismatch_since >= self.position_mismatch_s:
                self._events.append(BrokerEvent("position_adopted", "", "position", f"local {self._pos} -> broker {bp}"))
                self._pos = bp
                self._mismatch_since = None
        else:
            self._mismatch_since = None

    def _sibling_filled(self, trade: Any) -> bool:
        grp = self._role_group(trade)
        for t in self._our_trades():
            if t is not trade and self._role_group(t) == grp and float(t.orderStatus.filled or 0) > 0:
                return True
        return False

    @staticmethod
    def _role_group(trade: Any) -> str:
        parts = str(trade.order.orderRef).split(":")
        return parts[2] if len(parts) > 2 else ""

    # ------------------------------------------------------- Broker API
    def position(self) -> int:
        if self.is_connected():
            self._scan()
        return self._pos

    def poll(self) -> tuple[list[Fill], list[BrokerEvent]]:
        if self.is_connected():
            self._scan()
        f, e = self._fills, self._events
        self._fills, self._events = [], []
        return f, e

    def working_orders(self) -> list[WorkingOrder]:
        out = []
        for t in self._our_trades():
            if t.orderStatus.status not in WORKING:
                continue
            o = t.order
            role = self._role(t) or "?"
            price = float(getattr(o, "auxPrice", np.nan)) if role == "stop" else float(getattr(o, "lmtPrice", np.nan))
            remaining = float(o.totalQuantity) - float(t.orderStatus.filled or 0)
            out.append(WorkingOrder(str(o.orderId), role, self._side(o.action), int(round(remaining)), price))
        return out

    def last_price(self) -> float | None:
        tk = self._ticker
        if tk is None:
            return None
        px = getattr(tk, "last", None)
        ts = getattr(tk, "time", None)
        if px is None or not np.isfinite(px) or px <= 0 or ts is None:
            return None
        if self.clock() - _ts(ts).timestamp() > self.stale_tick_s:
            return None  # stale tick
        return float(px)

    def _wait(self, cond: Callable[[], bool], timeout: float) -> bool:
        deadline = self.clock() + timeout
        while True:
            if cond():
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(0.1)

    def _status(self, trade: Any) -> str:
        return str(trade.orderStatus.status)

    def place_bracket(self, direction: int, qty: int, stop: float, target: float, submit_min: float) -> str:
        self._require()
        action, opp = ("BUY", "SELL") if direction > 0 else ("SELL", "BUY")
        grp = self._new_group()
        parent = self.api.MarketOrder(action, qty)
        parent.orderId = self._order_id()
        parent.transmit = False
        parent.tif = "DAY"
        parent.orderRef = self._ref("entry", grp)
        tp = self.api.LimitOrder(opp, qty, target)
        sl = self.api.StopOrder(opp, qty, stop)
        for child, role in ((tp, "target"), (sl, "stop")):
            child.orderId = self._order_id()
            child.parentId = parent.orderId
            child.tif = "GTC"
            child.outsideRth = True
            child.ocaGroup = f"{self.prefix}-oca-{grp}"
            child.ocaType = 1
            child.orderRef = self._ref(role, grp)
            child.transmit = False
        sl.transmit = True  # last child transmits the whole bracket
        last = self.last_price()
        self._theo[parent.orderId] = last if last is not None else float("nan")
        trades = [self.ib.placeOrder(self.contract, o) for o in (parent, tp, sl)]
        t_parent, t_tp, t_sl = trades
        ok = self._wait(
            lambda: self._status(t_sl) in ACKED | {"Filled"} or self._status(t_sl) in DEAD or self._status(t_parent) in DEAD,
            self.cfg.order_confirm_timeout_s,
        )
        if not ok or self._status(t_sl) not in ACKED | {"Filled"}:
            msg = self._errors.get(sl.orderId) or self._errors.get(parent.orderId) or self._status(t_sl)
            # never leave anything half-placed
            for o in (parent, tp, sl):
                self._our_cancels.add(o.orderId)
                self.ib.cancelOrder(o)
            raise OrderRejected(f"bracket not confirmed: {msg}")
        return str(parent.orderId)

    def place_protection(self, side: int, qty: int, stop: float, target: float | None) -> str:
        self._require()
        action = "BUY" if side > 0 else "SELL"
        grp = self._new_group()
        orders = []
        sl = self.api.StopOrder(action, qty, stop)
        orders.append((sl, "stop"))
        if target is not None and np.isfinite(target):
            orders.append((self.api.LimitOrder(action, qty, target), "target"))
        for o, role in orders:
            o.orderId = self._order_id()
            o.tif = "GTC"
            o.outsideRth = True
            o.ocaGroup = f"{self.prefix}-oca-{grp}"
            o.ocaType = 1
            o.orderRef = self._ref(role, grp)
            o.transmit = True
        trades = [self.ib.placeOrder(self.contract, o) for o, _ in orders]
        t_sl = trades[0]
        ok = self._wait(lambda: self._status(t_sl) in ACKED | DEAD | {"Filled"}, self.cfg.order_confirm_timeout_s)
        if not ok or self._status(t_sl) not in ACKED | {"Filled"}:
            for o, _ in orders:
                self._our_cancels.add(o.orderId)
                self.ib.cancelOrder(o)
            raise OrderRejected(f"protective stop not confirmed: {self._errors.get(sl.orderId, self._status(t_sl))}")
        return str(sl.orderId)

    def resize_protection(self, qty: int) -> None:
        self._require()
        for t in self._our_trades():
            if self._role(t) in ("stop", "target") and t.orderStatus.status in WORKING:
                t.order.totalQuantity = qty
                self.ib.placeOrder(self.contract, t.order)

    def cancel_entry(self, order_id: str) -> None:
        for t in self._our_trades():
            if str(t.order.orderId) == order_id and t.orderStatus.status in WORKING:
                self._our_cancels.add(int(t.order.orderId))
                self.ib.cancelOrder(t.order)

    def cancel_all(self) -> None:
        if not self.is_connected():
            return
        for t in self._our_trades():
            if t.orderStatus.status in WORKING:
                self._our_cancels.add(int(t.order.orderId))
                self.ib.cancelOrder(t.order)

    def flatten(self, reason: int, submit_min: float) -> None:
        self._require()
        self._flatten_reason = reason
        self.cancel_all()
        pos = self.position()
        if pos == 0:
            return
        action = "SELL" if pos > 0 else "BUY"
        mo = self.api.MarketOrder(action, abs(pos))
        mo.orderId = self._order_id()
        mo.tif = "DAY"
        mo.orderRef = self._ref("flatten", self._new_group())
        mo.transmit = True
        last = self.last_price()
        self._theo[mo.orderId] = last if last is not None else float("nan")
        t = self.ib.placeOrder(self.contract, mo)
        filled = self._wait(lambda: self._status(t) == "Filled" or self._status(t) in DEAD, self.cfg.order_confirm_timeout_s)
        if not filled or self._status(t) != "Filled":
            self._events.append(BrokerEvent("flatten_unconfirmed", str(mo.orderId), "flatten", self._status(t)))


def _ts(x: Any) -> dt.datetime:
    if isinstance(x, dt.datetime):
        return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc)
