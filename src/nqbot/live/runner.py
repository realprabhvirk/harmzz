"""Live runner. Uses the exact strategy code of the backtester.

Per bar:
  on_bar_open(label)  new-session bookkeeping; flatten a position that survived a
                      data gap past the flat deadline (same rule as the engine)
  <broker executes>   SimBroker.process_bar in replay; IB in real time
  on_bar_close(bar)   book fills, enforce "never without a resting stop", risk
                      checks, features -> strategy signals -> core.decide(),
                      submit orders, persist state

Every strategy decision goes through ``nqbot.core``; nothing here re-implements
strategy logic.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from nqbot.config import AppConfig, hhmm
from nqbot.core import (
    EXIT_NAMES,
    EXIT_NONE,
    EXIT_RISK,
    N_STATE,
    PEND_ENTRY,
    PEND_FLATTEN,
    S_ENTRY_FILL,
    S_ENTRY_MIN,
    S_ENTRY_THEO,
    S_PEND,
    S_PEND_DIR,
    S_PEND_REASON,
    S_PEND_STOP,
    S_PEND_TGT,
    S_POS,
    S_POS_SESSION,
    S_QTY,
    S_SESSION,
    S_STOP,
    S_TGT,
    S_COOLDOWN_UNTIL,
    S_DAY_PNL,
    decide,
    flatten_reason_at_open,
    new_state,
    on_entry,
    on_exit,
    pack_params,
    roll_session,
)
from nqbot.features import features_from_arrays
from nqbot.instruments import get_instrument
from nqbot.live.alerts import Alerter
from nqbot.live.broker import Broker, BrokerError, Fill, OrderRejected
from nqbot.live.feed import Bar
from nqbot.live.risk import STALE_FLATTEN, STALE_NO_ENTRIES, STALE_OK, RiskManager
from nqbot.live.state import StateStore
from nqbot.sessions import bar_time_fields
from nqbot.strategies import get_strategy

log = logging.getLogger("nqbot.runner")

TIME_FIELDS = ("utc_min", "et_mod", "session_id", "in_session", "sess_el", "flat_mod", "entry_end_mod")


class RingBuffer:
    """Fixed-length history with contiguous numpy views (amortised O(1) append)."""

    def __init__(self, n: int, fields: dict[str, Any]) -> None:
        self.n = n
        self.cap = 2 * n
        self.arr = {k: np.empty(self.cap, dtype=d) for k, d in fields.items()}
        self.start = 0
        self.end = 0

    def __len__(self) -> int:
        return self.end - self.start

    def append(self, vals: dict[str, Any]) -> None:
        if self.end == self.cap:
            m = self.end - self.start
            for a in self.arr.values():
                a[:m] = a[self.start : self.end]
            self.start, self.end = 0, m
        for k, a in self.arr.items():
            a[self.end] = vals[k]
        self.end += 1
        if self.end - self.start > self.n:
            self.start += 1

    def view(self, k: str) -> np.ndarray:
        return self.arr[k][self.start : self.end]

    def last(self, k: str) -> Any:
        return self.arr[k][self.end - 1]


class LiveRunner:
    def __init__(
        self,
        cfg: AppConfig,
        broker: Broker,
        store: StateStore,
        alerter: Alerter | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.store = store
        self.alerter = alerter or Alerter("none")
        self.clock = clock
        self.sleep = sleep
        self.p = pack_params(cfg.strategy, cfg.costs)
        self.strategy = get_strategy(cfg.strategy)
        inst = get_instrument(cfg.costs.instrument)
        self.tick = inst.tick_size
        self.point_value = inst.point_value
        self.risk = RiskManager(cfg.risk)
        self.st = new_state()
        self.halted = False
        self.halt_reason = ""
        self.flattening = False
        self.pending: dict[str, Any] | None = None  # working entry bracket
        self.open_trade: dict[str, Any] | None = None
        self.trades: list[dict[str, Any]] = []
        self.last_bar_wall: float | None = None
        self.last_label: float | None = None
        self.last_close: float | None = None
        self.stale_level = STALE_OK
        self._saved_key: str | None = None
        self._cid = -1
        self._contract: str | None = None
        self._entry_start = hhmm(cfg.strategy.entry_start)
        self._flat_time = hhmm(cfg.strategy.flat_time)
        self._entry_end = hhmm(cfg.strategy.entry_end)
        fields: dict[str, Any] = {k: np.float64 for k in ("open", "high", "low", "close", "volume", "spread")}
        fields["cid"] = np.int64
        for k in TIME_FIELDS:
            fields[k] = bool if k == "in_session" else np.int64
        self.buf = RingBuffer(cfg.live.buffer_bars, fields)

    # ------------------------------------------------------------------ utils
    def _tf(self, label_min: float) -> dict[str, Any]:
        s = self.cfg.strategy
        return bar_time_fields(int(label_min), s.session, self._flat_time, self._entry_end, s.trade_holidays)

    def _event(self, event: str, **kw: Any) -> None:
        rec = {"wall": self.clock(), "label_min": self.last_label, "event": event, **kw}
        self.store.log_event(rec)
        log.info("%s %s", event, kw)

    def _now_min(self) -> float:
        return (self.last_label + 1.0) if self.last_label is not None else float(int(self.clock() // 60))

    @property
    def position(self) -> int:
        return int(self.st[S_POS] * self.st[S_QTY])

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "st": [float(x) for x in self.st],
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "pending": self.pending,
            "open_trade": self.open_trade,
            "orders_today": self.risk.orders_today,
            "rejects_today": self.risk.rejects_today,
            "risk_session": self.risk.session_id,
            "last_label": self.last_label,
            "n_trades": len(self.trades),
        }

    def persist(self, force: bool = False) -> None:
        """Atomic save, skipped when nothing but the bar clock changed."""
        snap = self.snapshot()
        key = json.dumps({k: v for k, v in snap.items() if k != "last_label"}, sort_keys=True, default=str)
        if force or key != self._saved_key:
            self.store.save(snap)
            self._saved_key = key

    def restore(self) -> bool:
        s = self.store.load()
        if not s:
            return False
        st = np.asarray(s["st"], dtype=np.float64)
        if len(st) != N_STATE:
            raise ValueError("state file from an incompatible version")
        self.st = st
        self.halted = bool(s.get("halted", False))
        self.halt_reason = s.get("halt_reason", "")
        self.pending = s.get("pending")
        self.open_trade = s.get("open_trade")
        self.risk.orders_today = int(s.get("orders_today", 0))
        self.risk.rejects_today = int(s.get("rejects_today", 0))
        self.risk.session_id = int(s.get("risk_session", -1))
        self.last_label = s.get("last_label")
        self.trades = self.store.trades()
        return True

    def clear_halt(self) -> None:
        self.halted = False
        self.halt_reason = ""
        self.persist()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        restored = self.restore()
        self.broker.connect()
        self._event("start", restored=restored, halted=self.halted)
        self.reconcile()
        self.persist(force=True)

    def seed(self, bars: list[Bar]) -> None:
        """Warm the feature buffer with history (no decisions, no orders)."""
        for b in bars:
            if b.volume > 0:
                self._append(b)

    def _append(self, bar: Bar) -> None:
        prev_close = self.buf.last("close") if len(self.buf) else float("nan")
        spread = 0.0
        if bar.contract != self._contract:
            if self._contract is not None:
                self._cid += 1
                spread = bar.roll_spread if np.isfinite(bar.roll_spread) else bar.open - prev_close
            else:
                self._cid = 0
            self._contract = bar.contract
        vals: dict[str, Any] = {
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
            "volume": bar.volume, "spread": spread, "cid": self._cid,
        }
        vals.update(self._tf(bar.label_min))
        self.buf.append(vals)

    # --------------------------------------------------------------- per bar
    def on_bar_open(self, label_min: float) -> None:
        tf = self._tf(label_min)
        roll_session(self.st, tf["session_id"])
        self.risk.new_session(tf["session_id"])
        if self.flattening or self.st[S_POS] == 0:
            return
        reason = flatten_reason_at_open(self.st, tf["et_mod"], tf["session_id"], tf["flat_mod"])
        if reason != EXIT_NONE:
            self._flatten(reason, label_min)

    def on_bar_close(self, bar: Bar) -> None:
        if bar.volume <= 0:  # same rule as backtest ingestion
            return
        self.last_bar_wall = self.clock()
        self.last_label = bar.label_min
        self.last_close = bar.close
        self.process_broker()
        self._append(bar)
        tf = {k: self.buf.last(k) for k in TIME_FIELDS}

        if not self.halted:
            self._risk_checks(bar, tf)
        if not self.halted:
            self.ensure_protected()
        if self.halted or self.flattening or self.pending is not None:
            self.persist()
            return

        sig_dir, sig_stop, sig_tgt = 0, float("nan"), float("nan")
        dmod = int(tf["et_mod"]) + 1
        if self.st[S_POS] == 0 and self._entry_start <= dmod <= int(tf["entry_end_mod"]):
            # Outside this window decide() ignores the signal, so skipping is exact.
            sig_dir, sig_stop, sig_tgt = self._signal()
        if self.stale_level >= STALE_NO_ENTRIES:
            sig_dir = 0
        decide(
            self.st, self.p, bar.label_min, int(tf["et_mod"]), int(tf["sess_el"]), int(tf["session_id"]),
            int(tf["entry_end_mod"]), int(tf["flat_mod"]), bool(bar.roll_guard), sig_dir, sig_stop, sig_tgt,
        )
        pend = int(self.st[S_PEND])
        if pend == PEND_ENTRY:
            self._submit_entry()
        elif pend == PEND_FLATTEN:
            self._flatten(int(self.st[S_PEND_REASON]), self._now_min())
        self.persist()

    def _signal(self) -> tuple[int, float, float]:
        b = self.buf
        feat = features_from_arrays(
            b.view("open"), b.view("high"), b.view("low"), b.view("close"), b.view("volume"),
            b.view("cid"), b.view("spread"), {k: b.view(k) for k in TIME_FIELDS},
            self.cfg.strategy, np.zeros(len(b), dtype=bool),
        )
        sig = self.strategy.signals(feat, self.cfg.strategy, self.tick)
        return sig.at(len(b) - 1)

    def on_end_of_data(self, bar: Bar) -> None:
        """Replay only: close at the last bar's close, like the backtester."""
        end = getattr(self.broker, "end_of_data", None)
        if end is not None:
            end(bar)
        self.process_broker()
        self.persist()

    # ----------------------------------------------------------- order flow
    def _submit_entry(self) -> None:
        d = int(self.st[S_PEND_DIR])
        qty = int(self.cfg.strategy.qty)
        stop, tgt = float(self.st[S_PEND_STOP]), float(self.st[S_PEND_TGT])
        why = self.risk.check_new_orders(3, d * qty)
        if why:
            self._event("entry_blocked", reason=why)
            if self.risk.kill_switch():
                self.halt("kill file present")
            return
        submit = self._now_min()
        try:
            oid = self.broker.place_bracket(d, qty, stop, tgt, submit)
        except OrderRejected as exc:
            self.risk.rejects_today += 1
            self._event("entry_rejected", error=str(exc))
            self.alerter.send("error", f"entry rejected: {exc}")
            # the broker may have filled the parent before the stop was refused
            if self.broker.position() != 0:
                self.halt(f"protective stop rejected after entry: {exc}")
            elif self.risk.rejects_today >= self.cfg.live.max_rejects_per_day:
                self.halt("too many rejected orders")
            return
        self.risk.record_orders(3)
        self.pending = {
            "order_id": oid, "dir": d, "qty": qty, "stop": stop, "target": tgt,
            "submit_min": submit, "wall": self.clock(),
        }
        self._event("entry_submitted", **self.pending)

    def _flatten(self, reason: int, submit_min: float) -> None:
        if self.broker.position() == 0 and self.pending is None:
            return
        self.flattening = True
        self.risk.record_orders(1)
        self._event("flatten", reason=EXIT_NAMES.get(reason, str(reason)))
        self.broker.flatten(reason, submit_min)
        if self.pending is not None and self.broker.position() == 0:
            self.pending = None

    def process_broker(self) -> None:
        fills, events = self.broker.poll()
        for f in fills:
            self._book(f)
        for e in events:
            self._event(f"order_{e.kind}", order_id=e.order_id, role=e.role, message=e.message)
            if e.role == "entry" and self.pending is not None and e.order_id == self.pending["order_id"]:
                self.pending = None
            if e.kind == "reject":
                self.risk.rejects_today += 1
                self.alerter.send("error", f"order rejected ({e.role}): {e.message}")
                if self.risk.rejects_today >= self.cfg.live.max_rejects_per_day:
                    self.halt("too many rejected orders")
            elif e.kind == "flatten_unconfirmed":
                self.halt(f"flatten order not confirmed: {e.message}")
            elif e.kind == "position_adopted":
                self.alerter.send("warn", f"position re-synced from broker: {e.message}")
                self._adopt_broker_position()
        if self.broker.position() == 0:
            self.flattening = False

    def _book(self, f: Fill) -> None:
        st = self.st
        if f.kind == "entry":
            pend = self.pending or {}
            stop = float(pend.get("stop", np.nan))
            tgt = float(pend.get("target", np.nan))
            if st[S_POS] == 0:
                on_entry(st, f.side, float(f.qty), f.theo, f.price, stop, tgt, f.label_min)
                self.open_trade = {"entry_min": f.label_min, "dir": f.side, "stop": stop, "target": tgt}
            elif st[S_POS] == f.side:  # partial fill continuation: average in
                q0, q1 = st[S_QTY], float(f.qty)
                st[S_ENTRY_FILL] = (st[S_ENTRY_FILL] * q0 + f.price * q1) / (q0 + q1)
                st[S_ENTRY_THEO] = (st[S_ENTRY_THEO] * q0 + f.theo * q1) / (q0 + q1)
                st[S_QTY] = q0 + q1
            else:
                self.halt("entry fill against an open position")
                return
            filled_qty = int(st[S_QTY])
            if self.pending is not None and filled_qty >= int(self.pending["qty"]):
                self.pending = None
            self._event("fill", kind="entry", side=f.side, qty=f.qty, price=f.price)
            self.alerter.send("fill", f"entry {'BUY' if f.side > 0 else 'SELL'} {f.qty} @ {f.price}")
            return
        # exit
        if st[S_POS] == 0:
            self._event("unexpected_exit_fill", side=f.side, qty=f.qty, price=f.price)
            self.alerter.send("error", f"exit fill with no position: {f}")
            return
        ot = self.open_trade or {}
        entry = {
            "entry_min": float(st[S_ENTRY_MIN]), "dir": int(st[S_POS]),
            "entry_theo": float(st[S_ENTRY_THEO]), "entry_fill": float(st[S_ENTRY_FILL]),
            "stop": float(ot.get("stop", st[S_STOP])), "target": float(ot.get("target", st[S_TGT])),
        }
        qty_pos = int(st[S_QTY])
        q = min(int(f.qty), qty_pos)
        if q < qty_pos:  # partial exit: book q contracts, keep the rest open
            tmp = st.copy()
            tmp[S_QTY] = q
            gross, slip, comm, net = on_exit(tmp, f.theo, f.price, f.reason, f.label_min, self.p)
            st[S_DAY_PNL] = tmp[S_DAY_PNL]
            st[S_COOLDOWN_UNTIL] = tmp[S_COOLDOWN_UNTIL]
            st[S_QTY] = qty_pos - q
        else:
            gross, slip, comm, net = on_exit(st, f.theo, f.price, f.reason, f.label_min, self.p)
            self.open_trade = None
        trade = {
            **entry, "qty": q, "exit_min": f.label_min, "exit_theo": f.theo, "exit_fill": f.price,
            "reason": int(f.reason), "exit_reason": EXIT_NAMES.get(int(f.reason), "unknown"),
            "gross": gross, "slippage": slip, "commission": comm, "net": net,
        }
        self.trades.append(trade)
        self.store.append_trade(trade)
        self._event("fill", kind="exit", side=f.side, qty=f.qty, price=f.price, reason=trade["exit_reason"])
        self.alerter.send("fill", f"exit {trade['exit_reason']} {q} @ {f.price} net {net:.2f}")

    # ---------------------------------------------------------- protection
    def ensure_protected(self) -> None:
        """Invariant: a non-zero position always has a resting stop for its full size."""
        pos = self.broker.position()
        if pos == 0 or self.flattening:
            return
        side = -int(np.sign(pos))
        stops = [o for o in self.broker.working_orders() if o.role == "stop" and o.side == side]
        covered = sum(o.qty for o in stops)
        if stops and covered == abs(pos):
            return
        try:
            if stops:
                self.broker.resize_protection(abs(pos))
                self._event("protection_resized", qty=abs(pos))
                return
            stop = float(self.st[S_STOP]) if self.st[S_POS] == np.sign(pos) else float("nan")
            last = self.broker.last_price() or self.last_close
            if last is not None and np.isfinite(stop) and (stop - last) * np.sign(pos) >= 0:
                stop = float("nan")  # already through: a new stop would be on the wrong side
            if not np.isfinite(stop):
                if last is None:
                    raise OrderRejected("no price available for an emergency stop")
                em = self.cfg.live.emergency_stop_points
                stop = round((last - np.sign(pos) * em) / self.tick) * self.tick
            tgt = float(self.st[S_TGT]) if self.st[S_POS] == np.sign(pos) else float("nan")
            self.broker.place_protection(side, abs(pos), stop, tgt if np.isfinite(tgt) else None)
            if self.st[S_POS] == np.sign(pos):
                self.st[S_STOP] = stop
            self._event("protection_placed", stop=stop, qty=abs(pos))
            self.alerter.send("warn", f"placed protective stop {stop} for position {pos}")
        except (OrderRejected, BrokerError) as exc:
            self.halt(f"cannot place protective stop: {exc}")

    # ------------------------------------------------------------- risk
    def _risk_checks(self, bar: Bar, tf: dict[str, Any]) -> None:
        if self.risk.kill_switch():
            self.halt("kill file present")
            return
        pos = self.broker.position()
        unreal = 0.0
        if self.st[S_POS] != 0:
            unreal = (bar.close - self.st[S_ENTRY_FILL]) * self.st[S_POS] * self.st[S_QTY] * self.point_value
        if self.risk.daily_loss_breached(float(self.st[S_DAY_PNL]), unreal):
            self.halt(f"daily loss limit: realised {self.st[S_DAY_PNL]:.2f} + unrealised {unreal:.2f}")
            return
        if abs(pos) > self.cfg.risk.max_position:
            self.halt(f"position {pos} exceeds max_position")
            return
        if pos != 0 and self.risk.in_flat_window(int(tf["et_mod"]) + 1):
            self._flatten(EXIT_RISK, self._now_min())

    def check_entry_timeout(self, now: float | None = None) -> None:
        """Cancel the unfilled remainder of an entry (partial fills) after entry_timeout_s."""
        now = self.clock() if now is None else now
        p = self.pending
        if p is not None and now - float(p.get("wall", now)) >= self.cfg.live.entry_timeout_s:
            self._event("entry_timeout", order_id=p["order_id"], filled=self.position)
            self.broker.cancel_entry(p["order_id"])
            self.process_broker()
            self.pending = None
            self.ensure_protected()
            self.persist()

    def _adopt_broker_position(self) -> None:
        bp = self.broker.position()
        if bp == self.position:
            return
        if bp == 0:
            last = self.broker.last_price() or self.last_close or float(self.st[S_ENTRY_FILL])
            self._book(Fill("exit", -int(np.sign(self.position)), abs(self.position), last, last, EXIT_RISK, self._now_min(), "adopt"))
        elif self.st[S_POS] == np.sign(bp):
            self.st[S_QTY] = abs(bp)
        else:
            self.halt(f"broker position {bp} conflicts with local {self.position}")

    def check_stale(self, now: float | None = None) -> int:
        """Call periodically between bars (wall clock)."""
        now = self.clock() if now is None else now
        if self.last_bar_wall is None or self.last_label is None:
            return STALE_OK
        tf = self._tf(int(now // 60))
        level = self.risk.stale_level(now - self.last_bar_wall, bool(tf["in_session"]), self.broker.position() != 0)
        if level != self.stale_level:
            self._event("stale_level", level=level)
            if level >= STALE_NO_ENTRIES:
                self.alerter.send("warn", f"market data stale for {now - self.last_bar_wall:.0f}s")
        self.stale_level = level
        if level == STALE_FLATTEN and not self.halted:
            self.halt("market data stale while in a position")
        return level

    def halt(self, reason: str) -> None:
        if self.halted:
            return
        self.halted = True
        self.halt_reason = reason
        self._event("halt", reason=reason)
        self.alerter.send("halt", f"HALTED: {reason}")
        try:
            if self.broker.is_connected():
                if self.broker.position() != 0 or self.pending is not None:
                    self._flatten(EXIT_RISK, self._now_min())
                else:
                    self.broker.cancel_all()
        except BrokerError as exc:  # pragma: no cover - logged, operator must intervene
            self.alerter.send("halt", f"flatten during halt failed: {exc}")
        self.persist()

    # ----------------------------------------------------- reconciliation
    def reconcile(self) -> None:
        """Startup / post-reconnect sync of local state with the broker."""
        bp = self.broker.position()
        orders = self.broker.working_orders()
        sp = self.position
        entry_orders = [o for o in orders if o.role == "entry"]
        if entry_orders and bp == 0:
            # an entry still working after a restart is stale: never fill it blind
            self.broker.cancel_all()
            self._event("reconcile_cancel_stale_entry", n=len(entry_orders))
            orders = []
        self.pending = None
        if bp == 0:
            if orders:
                self.broker.cancel_all()
                self._event("reconcile_cancel_orphans", n=len(orders))
                self.alerter.send("warn", f"cancelled {len(orders)} orphan order(s)")
            if sp != 0:
                last = self.broker.last_price() or float(self.st[S_ENTRY_FILL])
                self._event("reconcile_position_gone", local=sp, price=last)
                self.alerter.send("warn", f"local position {sp} not at broker; booked closed at ~{last}")
                f = Fill("exit", -int(np.sign(sp)), abs(sp), last, last, EXIT_RISK, self._now_min(), "reconcile")
                self._book(f)
            self.flattening = False
            return
        if abs(bp) > self.cfg.risk.max_position:
            self.halt(f"broker position {bp} exceeds max_position")
            return
        if sp == 0 or np.sign(sp) != np.sign(bp):
            self._event("reconcile_unknown_position", local=sp, broker=bp)
            self.halt(f"broker position {bp} does not match local {sp}")
            return
        if sp != bp:
            self._event("reconcile_qty_adopted", local=sp, broker=bp)
            self.alerter.send("warn", f"position size {sp} -> broker {bp}")
            self.st[S_QTY] = abs(bp)
        if self.st[S_POS_SESSION] < 0:
            self.st[S_POS_SESSION] = self.st[S_SESSION]
        self.ensure_protected()

    def ensure_connected(self) -> bool:
        """Reconnect with exponential backoff, then re-sync. Returns connection state."""
        if self.broker.is_connected():
            return True
        lc = self.cfg.live
        delay = lc.reconnect_initial_s
        self.alerter.send("warn", "broker disconnected; reconnecting")
        for attempt in range(1, lc.reconnect_max_attempts + 1):
            self.sleep(delay)
            try:
                self.broker.connect()
            except Exception as exc:  # noqa: BLE001 - any connect failure is retried
                self._event("reconnect_failed", attempt=attempt, error=str(exc))
                delay = min(delay * 2, lc.reconnect_max_s)
                continue
            if self.broker.is_connected():
                self._event("reconnected", attempt=attempt)
                self.alerter.send("info", f"reconnected after {attempt} attempt(s)")
                self.process_broker()
                self.reconcile()
                self.persist()
                return True
            delay = min(delay * 2, lc.reconnect_max_s)
        self.halted = True
        self.halt_reason = "could not reconnect to broker"
        self._event("halt", reason=self.halt_reason)
        self.alerter.send("halt", "HALTED: could not reconnect; broker-side stops remain working")
        self.persist()
        return False

    # ----------------------------------------------------------- replay
    def run_replay(self, bars: list[Bar]) -> list[dict[str, Any]]:
        sim = self.broker
        for bar in bars:
            if bar.volume <= 0:
                continue
            self.on_bar_open(bar.label_min)
            sim.process_bar(bar)  # type: ignore[attr-defined]
            self.on_bar_close(bar)
        if bars:
            self.on_end_of_data(bars[-1])
        return self.trades
