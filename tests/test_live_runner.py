from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nqbot.config import AppConfig, LiveConfig
from nqbot.core import S_POS, S_STOP, pack_params
from nqbot.data import prepare_bars, synthetic_bars
from nqbot.engine import run_backtest
from nqbot.live.alerts import Alerter
from nqbot.live.broker import SimBroker
from nqbot.live.feed import Bar, IBBarFeed, MinuteAggregator, bars_from_frame
from nqbot.live.ib_broker import IBBroker
from nqbot.live.loop import calendar_roll_guard, run_live
from nqbot.live.risk import STALE_FLATTEN, STALE_NO_ENTRIES, STALE_OK, RiskManager
from nqbot.live.runner import LiveRunner, RingBuffer
from nqbot.live.state import StateStore
from nqbot.sessions import bar_time_fields, session_arrays
from tests.fake_ib import FakeIB, api
from tests.helpers import bars_from, day_rows, nq_cfg

COLS = ["dir", "qty", "entry_fill", "entry_theo", "exit_fill", "exit_theo", "stop", "target", "reason", "gross", "slippage", "commission", "net"]


def live_cfg(**strategy: object) -> AppConfig:
    cfg = AppConfig()
    cfg.risk.max_daily_loss = 0.0
    cfg.risk.max_orders_per_day = 10**6
    cfg.risk.kill_file = "/nonexistent/KILL"
    for k, v in strategy.items():
        setattr(cfg.strategy, k, v)
    cfg.risk.max_position = cfg.strategy.qty
    return cfg


def sim_runner(cfg: AppConfig, tmp: Path, broker: SimBroker | None = None) -> LiveRunner:
    r = LiveRunner(cfg, broker or SimBroker(pack_params(cfg.strategy, cfg.costs)), StateStore(tmp))
    r.start()
    return r


def compare(df_engine: pd.DataFrame, cfg: AppConfig, live_trades: list[dict]) -> None:
    bt = run_backtest(df_engine, cfg).trades
    lt = pd.DataFrame(live_trades)
    assert len(bt) == len(lt) and len(bt) > 0
    utc = df_engine.index.as_unit("ns").asi8 // 60_000_000_000
    a = bt[COLS].astype(float).reset_index(drop=True)
    b = lt[COLS].astype(float).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert (lt["entry_min"].to_numpy() == utc[bt["entry_idx"]]).all()
    assert (lt["exit_min"].to_numpy() == utc[bt["exit_idx"]]).all()


# --------------------------------------------------------------------- parity
@pytest.mark.parametrize(
    "variant,session,extra",
    [
        ("vwap_fade", "rth", {}),
        ("vwap_pullback", "rth", {"pb_min_slope_atr": 0.1}),
        ("vwap_fade", "globex", {"qty": 2, "cooldown_minutes": 0, "max_trades_per_day": 6}),
        ("vwap_fade", "rth", {"max_daily_loss": 40.0, "band_k": 1.0}),
    ],
)
def test_parity_replay_equals_backtest(tmp_path: Path, variant: str, session: str, extra: dict) -> None:
    raw = synthetic_bars(start="2024-02-26", days=16, seed=13, mode="globex", zero_volume_frac=0.01)
    df = prepare_bars(raw)  # backtest drops zero-volume bars at ingestion
    cfg = live_cfg(**{"variant": variant, "session": session, "band_k": 1.5, **extra})
    runner = sim_runner(cfg, tmp_path)
    feed = bars_from_frame(prepare_bars(raw, drop_zero_volume=False))  # live sees them and skips them
    assert any(b.volume == 0 for b in feed)
    trades = runner.run_replay(feed)
    assert {t["exit_reason"] for t in trades} >= {"stop", "target"}
    compare(df, cfg, trades)
    assert len(StateStore(tmp_path).trades()) == len(trades)


def test_parity_through_roll_and_gaps(tmp_path: Path) -> None:
    rows = day_rows("2024-03-12")
    for m in (5, 6, 7):
        del rows[f"2024-03-12 11:0{m}"]
    rows.update(day_rows("2024-03-13", price=150.0))
    for k in list(rows)[::17]:
        o = rows[k][0]
        rows[k] = [o, o + 3.0, o - 3.0, o + (1.0 if hash(k) % 2 else -1.0), 25.0]
    df = bars_from(rows, {"2024-03-13 09:30": "NQM5"})
    df.loc[df.index[df.contract == "NQM5"][0], "roll_spread"] = 50.0
    cfg = live_cfg(band_k=1.0, fade_trend_filter=False, atr_len=5)
    cfg.costs.instrument = "NQ"
    runner = sim_runner(cfg, tmp_path)
    trades = runner.run_replay(bars_from_frame(df))
    bt = run_backtest(df, cfg).trades
    assert len(bt) >= 2 and len(trades) == len(bt)
    compare(df, cfg, trades)


def test_crash_recovery_continues_identically(tmp_path: Path) -> None:
    df = prepare_bars(synthetic_bars(start="2024-04-01", days=12, seed=5, mode="rth"))
    cfg = live_cfg(band_k=1.5)
    bars = bars_from_frame(df)
    sim = SimBroker(pack_params(cfg.strategy, cfg.costs))
    a = sim_runner(cfg, tmp_path, sim)
    # find a cut point while a position is open, to exercise reconciliation of a live position
    cut = None
    for i, bar in enumerate(bars):
        a.on_bar_open(bar.label_min)
        sim.process_bar(bar)
        a.on_bar_close(bar)
        if i > len(bars) // 2 and a.st[S_POS] != 0:
            cut = i
            break
    assert cut is not None
    # "crash": a new process with the same state dir and the same (exchange-side) broker
    b = LiveRunner(cfg, sim, StateStore(tmp_path))
    b.start()
    assert b.st[S_POS] == a.st[S_POS] and b.halted is False
    b.seed(bars[max(0, cut + 1 - cfg.live.buffer_bars) : cut + 1])
    for bar in bars[cut + 1 :]:
        b.on_bar_open(bar.label_min)
        sim.process_bar(bar)
        b.on_bar_close(bar)
    b.on_end_of_data(bars[-1])
    compare(df, cfg, b.trades)


# --------------------------------------------------------------- risk limits
def _day_df() -> pd.DataFrame:
    rows = day_rows("2024-03-12")
    rows["2024-03-12 10:00"] = [100.0, 100.0, 100.0, 100.0, 10]
    return bars_from(rows)


class ScriptedStrategy:
    """Long signal on chosen bar labels (by ET HH:MM); used to drive the runner."""

    name = "scripted"

    def __init__(self, at: set[str]) -> None:
        self.at = at

    def signals(self, feat, cfg, tick):  # type: ignore[no-untyped-def]
        from nqbot.strategies.base import Signals

        n = len(feat)
        mod = feat.time["et_mod"]
        d = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if f"{mod[i] // 60:02d}:{mod[i] % 60:02d}" in self.at:
                d[i] = 1
        c = feat.close
        return Signals(d, np.where(d != 0, c - 5, np.nan), np.where(d != 0, c + 5, np.nan))


def scripted_runner(tmp: Path, at: set[str], **risk: object) -> tuple[LiveRunner, SimBroker]:
    cfg = live_cfg(cooldown_minutes=0, max_trades_per_day=10)
    cfg.costs.instrument = "NQ"
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    sim = SimBroker(pack_params(cfg.strategy, cfg.costs))
    r = sim_runner(cfg, tmp, sim)
    r.strategy = ScriptedStrategy(at)  # type: ignore[assignment]
    return r, sim


def test_daily_loss_limit_halts_and_flattens(tmp_path: Path) -> None:
    rows = day_rows("2024-03-12")
    rows["2024-03-12 10:03"] = [100.0, 100.0, 97.0, 97.0, 10]  # -3 pts * 20 = -60 unrealised
    r, _ = scripted_runner(tmp_path, {"10:00"}, max_daily_loss=50.0)
    r.run_replay(bars_from_frame(bars_from(rows)))
    assert r.halted and "daily loss" in r.halt_reason
    assert r.trades[0]["exit_reason"] == "risk" and r.position == 0
    assert len(r.trades) == 1


def test_max_orders_blocks_second_entry(tmp_path: Path) -> None:
    rows = day_rows("2024-03-12")
    r, _ = scripted_runner(tmp_path, {"10:00", "11:00"}, max_orders_per_day=4)
    r.run_replay(bars_from_frame(bars_from(rows)))
    assert len(r.trades) == 1  # bracket (3) + flatten (1) = 4 orders, then blocked


def test_kill_file(tmp_path: Path) -> None:
    kill = tmp_path / "KILL"
    r, sim = scripted_runner(tmp_path, {"10:00"}, kill_file=str(kill))
    bars = bars_from_frame(_day_df())
    for bar in bars:
        r.on_bar_open(bar.label_min)
        sim.process_bar(bar)
        r.on_bar_close(bar)
        if r.position != 0 and not kill.exists():
            kill.write_text("stop")
    assert r.halted and r.halt_reason == "kill file present"
    assert r.position == 0 and r.trades[-1]["exit_reason"] == "risk"
    assert r.risk.check_new_orders(1, 1) == "kill file present"


def test_flat_window_forces_flat(tmp_path: Path) -> None:
    r, _ = scripted_runner(tmp_path, {"10:00"}, flat_window_start="10:30", flat_window_end="11:00")
    r.run_replay(bars_from_frame(_day_df()))
    t = r.trades[0]
    assert t["exit_reason"] == "risk"
    assert pd.Timestamp(int(t["exit_min"]) * 60, unit="s", tz="UTC").tz_convert("America/New_York").strftime("%H:%M") == "10:30"


def test_max_position_halt(tmp_path: Path) -> None:
    r, sim = scripted_runner(tmp_path, set())
    sim._pos = 5  # something outside the bot changed the position
    bar = bars_from_frame(_day_df())[40]
    r.on_bar_open(bar.label_min)
    r.on_bar_close(bar)
    assert r.halted and "max_position" in r.halt_reason


def test_entry_reject_and_reject_limit(tmp_path: Path) -> None:
    r, sim = scripted_runner(tmp_path, {"10:00", "10:01", "10:02"})
    r.cfg.live.max_rejects_per_day = 2
    bars = bars_from_frame(_day_df())
    for bar in bars:
        sim.reject_next_bracket = True
        r.on_bar_open(bar.label_min)
        sim.process_bar(bar)
        r.on_bar_close(bar)
    assert r.halted and r.halt_reason == "too many rejected orders"
    assert r.trades == []


def test_stale_data(tmp_path: Path) -> None:
    clock = SimpleNamespace(t=0.0)
    cfg = live_cfg()
    r = LiveRunner(cfg, SimBroker(pack_params(cfg.strategy, cfg.costs)), StateStore(tmp_path), clock=lambda: clock.t)
    r.start()
    bar = bars_from_frame(_day_df())[60]
    clock.t = bar.ts.timestamp() + 60
    r.on_bar_open(bar.label_min)
    r.on_bar_close(bar)
    assert r.check_stale(clock.t + 30) == STALE_OK
    assert r.check_stale(clock.t + cfg.risk.stale_seconds + 1) == STALE_NO_ENTRIES
    r.broker._pos = 1  # type: ignore[attr-defined]
    assert r.check_stale(clock.t + cfg.risk.stale_flatten_seconds + 1) == STALE_FLATTEN
    assert r.halted


def test_risk_manager_units() -> None:
    from nqbot.config import RiskConfig

    rm = RiskManager(RiskConfig(max_daily_loss=100, max_position=1, max_orders_per_day=3, kill_file="/nonexistent"))
    assert rm.daily_loss_breached(-60, -40) and not rm.daily_loss_breached(-60, -39)
    assert rm.check_new_orders(3, 1) is None
    assert "max_position" in (rm.check_new_orders(3, 2) or "")
    rm.record_orders(3)
    assert "max_orders" in (rm.check_new_orders(1, 1) or "")
    rm.new_session(5)
    assert rm.orders_today == 0
    assert rm.in_flat_window(16 * 60) and not rm.in_flat_window(15 * 60)
    assert rm.stale_level(1e9, False, True) == STALE_OK


# ------------------------------------------------------ IB-backed reconciliation
def ib_runner(tmp: Path, ib: FakeIB | None = None, state: dict | None = None) -> tuple[LiveRunner, FakeIB, IBBroker, Alerter]:
    ib = ib or FakeIB()
    cfg = live_cfg()
    cfg.live = LiveConfig(order_confirm_timeout_s=1.0, reconnect_initial_s=1.0, reconnect_max_s=4.0, reconnect_max_attempts=4)
    br = IBBroker(ib, api, cfg.live, "MNQ", "202406", sleep=ib.sleep, clock=ib.clock)
    store = StateStore(tmp)
    if state is not None:
        store.save(state)
    alerter = Alerter("none")
    r = LiveRunner(cfg, br, store, alerter, clock=ib.clock, sleep=ib.sleep)
    return r, ib, br, alerter


def _external_long(ib: FakeIB, qty: int = 1) -> None:
    """A position opened outside this process (e.g. before a crash), no stop."""
    ib.connected = True
    t = ib.placeOrder(ib.contract, api.MarketOrder("BUY", qty))
    t.order.orderRef = "nqbot:entry:old"


def _state_with_long(tmp: Path, stop: float = 17950.0) -> dict:
    cfg = live_cfg()
    r = LiveRunner(cfg, SimBroker(pack_params(cfg.strategy, cfg.costs)), StateStore(tmp / "seed"))
    r.st[S_POS] = 1
    r.st[1] = 1  # qty
    r.st[3] = 18000.0  # entry fill
    r.st[2] = 18000.0
    r.st[S_STOP] = stop
    r.st[5] = 18100.0
    return r.snapshot()


def test_reconcile_places_missing_stop(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib)
    r, ib, br, alerter = ib_runner(tmp_path, ib, _state_with_long(tmp_path))
    r.start()
    assert not r.halted
    stops = [o for o in br.working_orders() if o.role == "stop"]
    assert len(stops) == 1 and stops[0].price == 17950.0 and stops[0].qty == 1
    assert any("protective stop" in m for m in alerter.sent)


def test_reconcile_emergency_stop_when_saved_stop_is_through(tmp_path: Path) -> None:
    ib = FakeIB(price=17900.0)
    _external_long(ib)
    r, ib, br, _ = ib_runner(tmp_path, ib, _state_with_long(tmp_path, stop=17950.0))
    r.start()
    stops = [o for o in br.working_orders() if o.role == "stop"]
    assert stops[0].price == 17900.0 - r.cfg.live.emergency_stop_points


def test_reconcile_stop_rejected_flattens_and_halts(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib)
    ib.reject_types = {"STP"}
    r, ib, br, alerter = ib_runner(tmp_path, ib, _state_with_long(tmp_path))
    r.start()
    assert r.halted and "protective stop" in r.halt_reason
    assert br.position() == 0
    assert any("HALTED" in m for m in alerter.sent)
    # halt persists across restarts until cleared
    r2, *_ = ib_runner(tmp_path, ib)
    r2.start()
    assert r2.halted
    r2.clear_halt()
    assert not r2.halted


def test_reconcile_unknown_position_halts(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib)
    r, ib, br, _ = ib_runner(tmp_path, ib)
    r.start()
    assert r.halted and "does not match" in r.halt_reason
    assert br.position() == 0


def test_reconcile_cancels_orphans_and_books_vanished_position(tmp_path: Path) -> None:
    ib = FakeIB()
    ib.connected = True
    t = ib.placeOrder(ib.contract, api.StopOrder("SELL", 1, 17000.0))
    t.order.orderRef = "nqbot:stop:old"
    r, ib, br, alerter = ib_runner(tmp_path, ib, _state_with_long(tmp_path))
    r.start()
    assert t.orderStatus.status == "Cancelled"
    assert r.position == 0 and r.trades[-1]["exit_reason"] == "risk"
    assert any("orphan" in m for m in alerter.sent) and any("not at broker" in m for m in alerter.sent)


def test_reconcile_adopts_qty_and_cancels_stale_entry(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib, qty=1)
    _external_long(ib, qty=1)
    state = _state_with_long(tmp_path)
    r, ib, br, _ = ib_runner(tmp_path, ib, state)
    r.cfg.risk.max_position = 2
    r.start()
    assert r.position == 2 and not r.halted
    assert sum(o.qty for o in br.working_orders() if o.role == "stop") == 2
    # a working entry with no position after restart is cancelled
    ib2 = FakeIB()
    ib2.connected = True
    ib2.auto_fill_market = False
    e = ib2.placeOrder(ib2.contract, api.MarketOrder("BUY", 1))
    e.order.orderRef = "nqbot:entry:x"
    r2, *_ = ib_runner(tmp_path / "b", ib2)
    r2.start()
    assert e.orderStatus.status == "Cancelled"


def test_reconcile_position_over_limit(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib, qty=3)
    r, ib, br, _ = ib_runner(tmp_path, ib, _state_with_long(tmp_path))
    r.start()
    assert r.halted and "max_position" in r.halt_reason


def test_reconnect_with_backoff_then_resync(tmp_path: Path) -> None:
    r, ib, br, alerter = ib_runner(tmp_path)
    r.start()
    sleeps: list[float] = []
    r.sleep = lambda s: (sleeps.append(s), ib.sleep(s))  # type: ignore[assignment]
    ib.disconnect()
    ib.fail_connects = 2
    assert r.ensure_connected()
    assert sleeps == [1.0, 2.0, 4.0]
    assert any("reconnected" in m for m in alerter.sent)
    ib.disconnect()
    ib.fail_connects = 99
    sleeps.clear()
    assert not r.ensure_connected()
    assert sleeps == [1.0, 2.0, 4.0, 4.0]
    assert r.halted and "reconnect" in r.halt_reason


def test_ib_entry_partial_fill_timeout(tmp_path: Path) -> None:
    r, ib, br, _ = ib_runner(tmp_path)
    r.cfg.strategy.qty = 2
    r.cfg.risk.max_position = 2
    r.start()
    ib.auto_fill_market = False
    r.st[11] = 1  # emulate decide(): pending entry
    r.st[12] = 1
    r.st[13] = 17950.0
    r.st[14] = 18100.0
    r._submit_entry()
    oid = int(r.pending["order_id"])  # type: ignore[index]
    ib.fill(oid, 1, 18000.0)
    r.process_broker()
    assert r.position == 1 and r.pending is not None
    ib.sleep(r.cfg.live.entry_timeout_s + 1)
    r.check_entry_timeout()
    assert r.pending is None and r.position == 1
    stops = [o for o in br.working_orders() if o.role == "stop"]
    assert sum(s.qty for s in stops) == 1


def test_ib_flatten_unconfirmed_halts(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib)
    r, ib, br, _ = ib_runner(tmp_path, ib, _state_with_long(tmp_path))
    r.start()
    assert not r.halted
    ib.auto_fill_market = False
    r._flatten(3, 0.0)
    r.process_broker()
    assert r.halted and "not confirmed" in r.halt_reason


def test_position_adopted_event(tmp_path: Path) -> None:
    ib = FakeIB()
    _external_long(ib)
    r, ib, br, alerter = ib_runner(tmp_path, ib, _state_with_long(tmp_path))
    r.start()
    ib.reported_position = 0  # position vanished at the broker
    br.poll()
    ib.sleep(30)
    r.process_broker()
    assert r.position == 0 and any("re-synced" in m for m in alerter.sent)


# --------------------------------------------------------------- live loop
class FakeFeed:
    def __init__(self, bars: list[Bar]) -> None:
        self.bars = bars
        self.i = 0

    def history(self) -> list[Bar]:
        return self.bars[:30]

    def poll(self, now: dt.datetime) -> list[Bar]:
        out = [b for b in self.bars[30:] if (b.label_min + 1) * 60 <= now.timestamp()]
        out = out[self.i :]
        self.i += len(out)
        return out


def test_run_live_loop_with_fake_ib(tmp_path: Path) -> None:
    r, ib, br, _ = ib_runner(tmp_path)
    bars = bars_from_frame(_day_df())[:120]
    ib.now = bars[30].ts.to_pydatetime()
    feed = FakeFeed(bars)
    run_live(r, feed, ib.sleep, now_fn=lambda: ib.now, poll_s=30.0, max_iterations=60)
    assert r.last_label is not None and r.last_label > bars[30].label_min
    assert not r.halted


def test_calendar_roll_guard() -> None:
    g = calendar_roll_guard("202406")
    before = Bar(pd.Timestamp("2024-06-12 15:00", tz="UTC").value // 60_000_000_000, 1, 1, 1, 1, 1, "x")
    after = Bar(pd.Timestamp("2024-06-13 15:00", tz="UTC").value // 60_000_000_000, 1, 1, 1, 1, 1, "x")
    assert not g(before) and g(after)


# ------------------------------------------------------------- pieces
def test_bar_time_fields_match_session_arrays() -> None:
    idx = pd.date_range("2024-03-08 12:00", "2024-03-12 00:00", freq="7min", tz="UTC").append(
        pd.date_range("2024-11-01", "2024-11-05", freq="11min", tz="UTC")
    ).append(pd.date_range("2024-11-28 13:00", "2024-11-29 20:00", freq="13min", tz="UTC"))
    for mode in ("rth", "globex"):
        for th in (False, True):
            arr = session_arrays(idx, mode, trade_holidays=th)
            for i in range(0, len(idx), 3):
                tf = bar_time_fields(int(arr["utc_min"][i]), mode, trade_holidays=th)
                for k, v in tf.items():
                    assert v == arr[k][i], (k, idx[i], mode)
    with pytest.raises(ValueError):
        bar_time_fields(0, "asia")


def test_ring_buffer() -> None:
    rb = RingBuffer(3, {"x": np.float64})
    for i in range(10):
        rb.append({"x": float(i)})
        assert list(rb.view("x")) == [float(j) for j in range(max(0, i - 2), i + 1)]
    assert rb.last("x") == 9.0 and len(rb) == 3


def test_minute_aggregator_and_ib_feed() -> None:
    t0 = dt.datetime(2024, 3, 12, 15, 0, tzinfo=dt.timezone.utc)
    agg = MinuteAggregator("MNQM4", grace_s=3)
    out = []
    for k in range(12):
        out += agg.add(t0 + dt.timedelta(seconds=5 * k), 100 + k, 101 + k, 99 + k, 100.5 + k, 2)
    assert out == []
    out += agg.add(t0 + dt.timedelta(seconds=60), 200, 200, 200, 200, 1)
    b = out[0]
    assert (b.open, b.high, b.low, b.close, b.volume) == (100, 112, 99, 111.5, 24)
    assert b.label_min == t0.timestamp() / 60
    assert agg.flush_if_due(t0 + dt.timedelta(seconds=62)) == []
    assert len(agg.flush_if_due(t0 + dt.timedelta(seconds=124))) == 1
    agg.add(t0 + dt.timedelta(seconds=180), 1, 1, 1, 1, 0)
    assert agg.flush_if_due(t0 + dt.timedelta(seconds=300)) == []  # zero volume dropped

    rt: list = []
    hist = [SimpleNamespace(date=t0, open=1.0, high=2.0, low=0.5, close=1.5, volume=10),
            SimpleNamespace(date=t0 + dt.timedelta(minutes=1), open=1.0, high=2.0, low=0.5, close=1.5, volume=0)]
    ib = SimpleNamespace(reqRealTimeBars=lambda *a: rt, reqHistoricalData=lambda *a, **k: hist)
    feed = IBBarFeed(ib, object(), "MNQM4")
    assert len(feed.history()) == 1
    for k in range(13):
        rt.append(SimpleNamespace(time=t0 + dt.timedelta(seconds=5 * k), open_=1.0, high=2.0, low=0.5, close=1.5, volume=3))
    bars = feed.poll(t0 + dt.timedelta(seconds=61))
    assert len(bars) == 1 and bars[0].volume == 36 and feed.last_update is not None


def test_state_store(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "x")
    assert s.load() is None and s.trades() == []
    s.save({"a": np.float64(1.5), "b": np.array([1, 2]), "c": float("nan")})
    assert s.load()["a"] == 1.5 and s.load()["b"] == [1, 2]
    s.append_trade({"net": 1.0})
    s.log_event({"e": "x", "obj": object()})
    assert s.trades() == [{"net": 1.0}]
    assert not (tmp_path / "x" / "state.tmp").exists()


def test_restore_rejects_incompatible_state(tmp_path: Path) -> None:
    StateStore(tmp_path).save({"st": [0.0, 1.0]})
    cfg = live_cfg()
    r = LiveRunner(cfg, SimBroker(pack_params(cfg.strategy, cfg.costs)), StateStore(tmp_path))
    with pytest.raises(ValueError):
        r.restore()


def test_alerts() -> None:
    sent: list = []
    a = Alerter("slack", env={"NQBOT_SLACK_WEBHOOK": "https://hooks.example/x"}, sender=lambda u, b, h: sent.append((u, b)))
    a.send("halt", "boom")
    assert sent[0][0] == "https://hooks.example/x" and b"boom" in sent[0][1]
    t = Alerter("telegram", env={"NQBOT_TELEGRAM_TOKEN": "T", "NQBOT_TELEGRAM_CHAT": "C"}, sender=lambda u, b, h: sent.append((u, b)))
    t.send("fill", "x")
    assert sent[1][0] == "https://api.telegram.org/botT/sendMessage" and b'"chat_id": "C"' in sent[1][1]

    def boom(*a):  # type: ignore[no-untyped-def]
        raise OSError("network down")

    f = Alerter("slack", env={"NQBOT_SLACK_WEBHOOK": "u"}, sender=boom)
    f.send("halt", "still fine")
    assert f.failures == 1
    assert Alerter("slack", env={})._target("x") is None
    assert Alerter("none").sent == []


# ------------------------------------------------------------- CLI gating
def test_live_gate() -> None:
    from nqbot.cli.live import LIVE_ENV, LIVE_ENV_VALUE, GateError, check_mode_gate, confirmation_phrase

    cfg = AppConfig()
    ok_env = {LIVE_ENV: LIVE_ENV_VALUE}
    phrase = confirmation_phrase("U999", cfg)
    assert phrase == "LIVE U999 MNQ 1"
    check_mode_gate("paper", False, "DU1", cfg, {}, input)
    with pytest.raises(GateError):
        check_mode_gate("paper", False, "U999", cfg, {}, input)  # real account in paper mode
    with pytest.raises(GateError):
        check_mode_gate("paper", True, "DU1", cfg, {}, input)
    with pytest.raises(GateError):
        check_mode_gate("live", True, "U999", cfg, {}, lambda _: phrase)  # no env var
    with pytest.raises(GateError):
        check_mode_gate("live", False, "U999", cfg, ok_env, lambda _: phrase)  # no flag
    with pytest.raises(GateError):
        check_mode_gate("live", True, "U999", cfg, ok_env, lambda _: "yes")  # wrong phrase
    check_mode_gate("live", True, "U999", cfg, ok_env, lambda _: phrase)
    with pytest.raises(GateError):
        check_mode_gate("demo", False, "DU1", cfg, {}, input)


def test_live_cli_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from nqbot.cli.live import main

    assert main(["replay", "--synthetic-days", "8", "--state-dir", str(tmp_path / "s")]) == 0
    assert "parity OK" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["replay", "--synthetic-days", "8", "--state-dir", str(tmp_path / "s")])
