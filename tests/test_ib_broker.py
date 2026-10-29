from __future__ import annotations

import pytest

from nqbot.config import LiveConfig
from nqbot.core import EXIT_RISK, EXIT_STOP, EXIT_TARGET
from nqbot.live.broker import BrokerError, OrderRejected
from nqbot.live.ib_broker import IBBroker
from tests.fake_ib import FakeIB, api


def make(price: float = 18000.0, **kw) -> tuple[FakeIB, IBBroker]:
    ib = FakeIB(price)
    cfg = LiveConfig(order_confirm_timeout_s=2.0)
    br = IBBroker(ib, api, cfg, "MNQ", "202406", sleep=ib.sleep, clock=ib.clock, **kw)
    br.connect()
    return ib, br


def test_bracket_structure_and_entry_fill() -> None:
    ib, br = make()
    oid = br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    parent, tp, sl = ib.order("entry"), ib.order("target"), ib.order("stop")
    assert str(parent.order.orderId) == oid
    assert parent.order.orderType == "MKT" and parent.order.action == "BUY" and parent.order.tif == "DAY"
    for child, typ in ((tp, "LMT"), (sl, "STP")):
        assert child.order.parentId == parent.order.orderId
        assert child.order.action == "SELL" and child.order.orderType == typ
        assert child.order.tif == "GTC" and child.order.outsideRth
        assert child.order.ocaGroup and child.order.ocaGroup == tp.order.ocaGroup
    assert sl.order.auxPrice == 17950.0 and tp.order.lmtPrice == 18100.0
    assert not parent.order.transmit and not tp.order.transmit and sl.order.transmit
    fills, _ = br.poll()
    assert [(f.kind, f.side, f.qty, f.price) for f in fills] == [("entry", 1, 1, 18000.0)]
    assert br.position() == 1
    roles = sorted(o.role for o in br.working_orders())
    assert roles == ["stop", "target"]


def test_stop_fill_cancels_target_via_oca() -> None:
    ib, br = make()
    br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    br.poll()
    ib.fill(ib.order("stop").order.orderId, 1, 17949.5)
    fills, events = br.poll()
    assert len(fills) == 1 and fills[0].kind == "exit" and fills[0].reason == EXIT_STOP
    assert fills[0].theo == 17950.0 and fills[0].price == 17949.5
    assert ib.order("target").orderStatus.status == "Cancelled"
    assert [e.kind for e in events] == ["cancel"]  # OCO sibling, not a reject
    assert br.position() == 0 and br.working_orders() == []


def test_target_fill() -> None:
    ib, br = make()
    br.place_bracket(-1, 1, 18050.0, 17900.0, 0.0)
    br.poll()
    ib.fill(ib.order("target").order.orderId, 1, 17900.0)
    fills, _ = br.poll()
    assert fills[0].reason == EXIT_TARGET and fills[0].side == 1
    assert br.position() == 0


def test_partial_entry_then_resize_and_cancel_remainder() -> None:
    ib, br = make()
    ib.auto_fill_market = False
    oid = br.place_bracket(1, 2, 17950.0, 18100.0, 0.0)
    ib.fill(int(oid), 1, 18000.25)
    fills, _ = br.poll()
    assert fills[0].qty == 1 and br.position() == 1
    br.cancel_entry(oid)
    _, events = br.poll()
    assert [e.kind for e in events] == ["cancel"]
    br.resize_protection(1)
    stops = [o for o in br.working_orders() if o.role == "stop"]
    assert stops[0].qty == 1


def test_stop_rejected_raises_and_cleans_up() -> None:
    ib, br = make()
    ib.reject_types = {"STP"}
    with pytest.raises(OrderRejected, match="201"):
        br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    assert br.working_orders() == []
    _, events = br.poll()
    assert any(e.role == "stop" for e in events)


def test_unacknowledged_stop_times_out() -> None:
    ib, br = make()
    ib.never_ack_types = {"STP"}
    with pytest.raises(OrderRejected):
        br.place_protection(-1, 1, 17950.0, None)
    assert ib.order("stop").orderStatus.status == "Cancelled"


def test_protection_with_target_is_oca() -> None:
    ib, br = make()
    br.place_protection(-1, 1, 17950.0, 18100.0)
    sl, tp = ib.order("stop"), ib.order("target")
    assert sl.order.ocaGroup == tp.order.ocaGroup and sl.order.ocaType == 1
    assert sl.order.parentId == 0


def test_flatten_cancels_and_closes() -> None:
    ib, br = make()
    br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    br.poll()
    ib.set_price(18020.0)
    br.flatten(EXIT_RISK, 0.0)
    fills, events = br.poll()
    assert fills[-1].kind == "exit" and fills[-1].reason == EXIT_RISK and fills[-1].price == 18020.0
    assert br.position() == 0
    assert ib.order("stop").orderStatus.status == "Cancelled"
    assert {e.kind for e in events} == {"cancel"}


def test_flatten_unconfirmed_event() -> None:
    ib, br = make()
    br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    br.poll()
    ib.auto_fill_market = False
    br.flatten(EXIT_RISK, 0.0)
    _, events = br.poll()
    assert any(e.kind == "flatten_unconfirmed" for e in events)


def test_delayed_position_updates_are_tolerated_then_adopted() -> None:
    ib, br = make()
    ib.position_delay = 5.0
    br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    assert br.position() == 1  # from our own fills, not the lagging report
    assert br.broker_position() == 0
    ib.sleep(6)
    assert br.broker_position() == 1
    _, events = br.poll()
    assert not any(e.kind == "position_adopted" for e in events)
    # a mismatch that persists is adopted and reported
    ib.reported_position = 3
    br.poll()
    ib.sleep(25)
    _, events = br.poll()
    assert any(e.kind == "position_adopted" for e in events)
    assert br.position() == 3


def test_stale_ticks() -> None:
    ib, br = make()
    assert br.last_price() == 18000.0
    ib.set_price(18001.0, stale_s=120)
    assert br.last_price() is None
    ib.set_price(float("nan"))
    assert br.last_price() is None


def test_disconnect_and_reconnect() -> None:
    ib, br = make()
    br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    br.poll()
    ib.disconnect()
    assert not br.is_connected()
    with pytest.raises(BrokerError):
        br.place_bracket(1, 1, 1.0, 2.0, 0.0)
    assert br.poll() == ([], [])
    ib.fail_connects = 1
    with pytest.raises(ConnectionRefusedError):
        br.connect()
    br.connect()
    assert br.is_connected() and br.position() == 1
    br.cancel_all()


def test_roles_survive_restart() -> None:
    ib, br = make()
    br.place_bracket(1, 1, 17950.0, 18100.0, 0.0)
    br2 = IBBroker(ib, api, LiveConfig(), "MNQ", "202406", sleep=ib.sleep, clock=ib.clock)
    br2.connect()
    assert br2.position() == 1
    assert sorted(o.role for o in br2.working_orders()) == ["stop", "target"]
    assert br2.poll()[0] == []  # old executions are not re-reported
    assert br2.accounts() == ["DU1234567"]


def test_unqualified_contract() -> None:
    ib = FakeIB()
    ib.qualifyContracts = lambda c: []  # type: ignore[method-assign]
    br = IBBroker(ib, api, LiveConfig(), "MNQ", "202406", sleep=ib.sleep, clock=ib.clock)
    with pytest.raises(BrokerError):
        br.connect()
