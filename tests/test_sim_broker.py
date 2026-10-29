from __future__ import annotations

import pytest

from nqbot.config import AppConfig
from nqbot.core import EXIT_END_OF_DATA, EXIT_FLAT_TIME, EXIT_STOP, EXIT_TARGET, pack_params
from nqbot.live.broker import BrokerError, OrderRejected, SimBroker
from nqbot.live.feed import Bar


def bar(m: float, o: float, h: float, low: float, c: float) -> Bar:
    return Bar(m, o, h, low, c, 10.0, "NQM4")


def sim() -> SimBroker:
    cfg = AppConfig()
    cfg.costs.instrument = "NQ"
    s = SimBroker(pack_params(cfg.strategy, cfg.costs))
    s.connect()
    return s


def test_bracket_lifecycle_and_working_orders() -> None:
    s = sim()
    oid = s.place_bracket(1, 1, 95.0, 105.0, 100.0)
    roles = [o.role for o in s.working_orders()]
    assert roles == ["entry", "stop", "target"]
    with pytest.raises(BrokerError):
        s.place_bracket(1, 1, 95.0, 105.0, 100.0)
    s.process_bar(bar(100, 100, 101, 99, 100))
    f, _ = s.poll()
    assert f[0].kind == "entry" and f[0].price == 100.25 and s.position() == 1
    s.process_bar(bar(101, 100, 106, 99, 104))
    f, _ = s.poll()
    assert f[0].reason == EXIT_TARGET and f[0].price == 105.0 and s.position() == 0
    assert s.working_orders() == [] and s.last_price() == 104 and oid.startswith("sim")


def test_expiry_cancel_and_reject() -> None:
    s = sim()
    s.place_bracket(1, 1, 95.0, 105.0, 100.0)
    s.process_bar(bar(110, 100, 101, 99, 100))  # data gap: too late
    _, e = s.poll()
    assert e[0].kind == "expired" and s.position() == 0
    oid = s.place_bracket(-1, 1, 105.0, 95.0, 120.0)
    s.cancel_entry(oid)
    _, e = s.poll()
    assert e[0].kind == "cancel"
    oid = s.place_bracket(-1, 1, 105.0, 95.0, 130.0)
    s.flatten(EXIT_FLAT_TIME, 130.0)  # before the fill: cancels the entry
    _, e = s.poll()
    assert e[0].kind == "cancel" and s.working_orders() == []
    s.reject_next_bracket = True
    with pytest.raises(OrderRejected):
        s.place_bracket(1, 1, 95.0, 105.0, 140.0)


def test_protection_flatten_and_end_of_data() -> None:
    s = sim()
    s._pos = 1  # position adopted from elsewhere
    s.place_protection(-1, 1, 95.0, None)
    assert [o.role for o in s.working_orders()] == ["stop"]
    s.process_bar(bar(1, 100, 120, 99, 110))  # no target: nothing happens
    assert s.position() == 1
    s.process_bar(bar(2, 96, 97, 94, 95))
    f, _ = s.poll()
    assert f[0].reason == EXIT_STOP and f[0].price == 94.75
    s._pos = -1
    s.place_protection(1, 1, 105.0, 90.0)
    assert {o.role for o in s.working_orders()} == {"stop", "target"}
    s.flatten(EXIT_FLAT_TIME, 3.0)
    assert any(o.role == "flatten" for o in s.working_orders())
    s.process_bar(bar(3, 100, 100, 100, 100))
    f, _ = s.poll()
    assert f[0].reason == EXIT_FLAT_TIME and f[0].price == 100.25
    s._pos = 1
    s.resize_protection(1)
    s.end_of_data(bar(4, 100, 100, 100, 101))
    f, _ = s.poll()
    assert f[0].reason == EXIT_END_OF_DATA and f[0].price == 100.75
    s.cancel_all()
    assert s.is_connected()
