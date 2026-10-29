from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqbot.causality import check_engine_causality
from nqbot.config import AppConfig
from nqbot.engine import run_backtest
from tests.helpers import bars_from, day_rows, nq_cfg, run, signals_at

DAY = "2024-03-12"
SLIP = 5.0  # 1 tick NQ
COMM = 2.25


def _one_trade(rows: dict, entry: tuple[int, float, float] = (1, 90.0, 110.0), at: str = "09:44", **kw):
    df = bars_from(rows)
    res = run(df, signals_at(df, {f"{DAY} {at}": entry}), nq_cfg(**kw))
    return df, res


def test_fill_at_next_open_and_costs() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 09:45"] = [101.0, 101.0, 101.0, 101.0, 10]
    rows[f"{DAY} 09:50"] = [101.0, 110.25, 101.0, 110.0, 10]  # trades 1 tick through target
    df, res = _one_trade(rows)
    t = res.trades.iloc[0]
    assert t.entry_time == pd.Timestamp(f"{DAY} 09:45", tz="America/New_York")
    assert t.entry_theo == 101.0 and t.entry_fill == 101.25
    assert t.exit_reason == "target" and t.exit_theo == 110.0 and t.exit_fill == 110.0
    assert t.gross == pytest.approx((110 - 101) * 20)
    assert t.slippage == pytest.approx(SLIP)  # limit exit pays none
    assert t.commission == pytest.approx(2 * COMM)
    assert t.net == pytest.approx(t.gross - SLIP - 2 * COMM)


def test_target_needs_trade_through() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 09:50"] = [100.0, 110.0, 100.0, 105.0, 10]  # touches target exactly
    _, res = _one_trade(rows)
    assert res.trades.iloc[0].exit_reason != "target"
    cfg = nq_cfg()
    cfg.costs.target_through_ticks = 0
    df = bars_from(rows)
    res2 = run(df, signals_at(df, {f"{DAY} 09:44": (1, 90.0, 110.0)}), cfg)
    assert res2.trades.iloc[0].exit_reason == "target"


def test_gap_through_stop_fills_at_open() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 10:00"] = [85.0, 86.0, 84.0, 85.0, 10]  # opens below the 90 stop
    _, res = _one_trade(rows)
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop"
    assert t.exit_theo == 85.0 and t.exit_fill == 84.75  # worse than the stop, plus slippage
    assert t.slippage == pytest.approx(2 * SLIP)


def test_stop_and_target_same_bar_stop_wins() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 10:00"] = [100.0, 115.0, 85.0, 100.0, 10]
    _, res = _one_trade(rows)
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_theo == 90.0 and t.exit_fill == 89.75


def test_short_mirror() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 10:00"] = [100.0, 100.0, 89.0, 95.0, 10]
    _, res = _one_trade(rows, entry=(-1, 110.0, 90.0))
    t = res.trades.iloc[0]
    assert t.dir == -1 and t.entry_fill == 99.75
    assert t.exit_reason == "target" and t.exit_fill == 90.0
    rows[f"{DAY} 10:00"] = [111.0, 112.0, 89.0, 95.0, 10]  # gap above short stop
    _, res = _one_trade(rows, entry=(-1, 110.0, 90.0))
    assert res.trades.iloc[0].exit_fill == 111.25


def test_entry_bar_is_checked() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 09:45"] = [100.0, 100.0, 89.0, 95.0, 10]
    _, res = _one_trade(rows)
    t = res.trades.iloc[0]
    assert t.entry_idx == t.exit_idx and t.exit_reason == "stop"
    rows[f"{DAY} 09:45"] = [88.0, 89.0, 87.0, 88.0, 10]  # entry opens already through the stop
    _, res = _one_trade(rows)
    t = res.trades.iloc[0]
    assert t.entry_fill == 88.25 and t.exit_fill == 87.75 and t.exit_reason == "stop"


def test_time_flatten_at_next_open() -> None:
    rows = day_rows(DAY)
    rows[f"{DAY} 15:55"] = [103.0, 104.0, 102.0, 103.0, 10]
    _, res = _one_trade(rows, entry=(1, 90.0, 120.0))
    t = res.trades.iloc[0]
    assert t.exit_reason == "flat_time"
    assert t.exit_time == pd.Timestamp(f"{DAY} 15:55", tz="America/New_York")
    assert t.exit_theo == 103.0 and t.exit_fill == 102.75
    assert (res.position.iloc[-5:] == 0).all()


def test_end_of_data_flat() -> None:
    rows = {k: v for k, v in day_rows(DAY).items() if k < f"{DAY} 10:30"}
    _, res = _one_trade(rows)
    t = res.trades.iloc[0]
    assert t.exit_reason == "end_of_data" and t.exit_idx == len(rows) - 1
    assert t.exit_fill == 99.75
    assert res.position.iloc[-1] == 0


def test_roll_flat_in_old_contract() -> None:
    rows = day_rows(DAY)
    contracts = {f"{DAY} 11:00": "NQM5"}  # contract switches at 11:00 (synthetic edge case)
    for k in rows:
        if k >= f"{DAY} 11:00":
            rows[k] = [150.0, 150.0, 150.0, 150.0, 10]
    df = bars_from(rows, contracts)
    df.loc[df.index[df.contract == "NQM5"][0], "roll_spread"] = 50.0
    res = run(df, signals_at(df, {f"{DAY} 09:44": (1, 90.0, 120.0)}), nq_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "roll"
    assert t.contract == "NQZ4"
    assert df.contract.iloc[t.exit_idx] == "NQZ4"
    assert t.exit_time == pd.Timestamp(f"{DAY} 10:59", tz="America/New_York")
    # no new entries under the guard either
    res2 = run(df, signals_at(df, {f"{DAY} 10:58": (1, 90.0, 120.0)}), nq_cfg())
    assert len(res2.trades) == 0


def test_gating_rules() -> None:
    rows = day_rows(DAY)
    df = bars_from(rows)
    # before warm-up / entry window and after entry_end: ignored
    sig = signals_at(df, {f"{DAY} 09:40": (1, 90.0, 110.0), f"{DAY} 15:31": (1, 90.0, 110.0)})
    assert len(run(df, sig, nq_cfg()).trades) == 0
    # last allowed decision is 15:29 (fills 15:30)
    sig = signals_at(df, {f"{DAY} 15:29": (1, 90.0, 110.0)})
    assert len(run(df, sig, nq_cfg()).trades) == 1
    # warm-up measured from session start
    sig = signals_at(df, {f"{DAY} 09:44": (1, 90.0, 110.0)})
    assert len(run(df, sig, nq_cfg(warmup_minutes=30, entry_start="09:30")).trades) == 0
    assert len(run(df, sig, nq_cfg(warmup_minutes=15, entry_start="09:30")).trades) == 1
    # invalid levels
    sig = signals_at(df, {f"{DAY} 10:00": (1, np.nan, 110.0)})
    assert len(run(df, sig, nq_cfg()).trades) == 0


def _stopped_day(n_signals: int, **kw):
    rows = day_rows(DAY)
    entries = {}
    for i in range(n_signals):
        t = pd.Timestamp(f"{DAY} 10:00") + pd.Timedelta(minutes=10 * i)
        stop_bar = (t + pd.Timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
        rows[stop_bar] = [100.0, 100.0, 89.0, 95.0, 10]
        entries[t.strftime("%Y-%m-%d %H:%M")] = (1, 90.0, 110.0)
        for j in range(3, 10):
            k = (t + pd.Timedelta(minutes=j)).strftime("%Y-%m-%d %H:%M")
            entries[k] = (1, 90.0, 110.0)
    df = bars_from(rows)
    return run(df, signals_at(df, entries), nq_cfg(**kw))


def test_max_trades_per_day_and_one_position() -> None:
    res = _stopped_day(5, max_trades_per_day=2)
    assert len(res.trades) == 2
    assert res.position.abs().max() == 1
    res = _stopped_day(5, max_trades_per_day=0)
    assert len(res.trades) == 0


def test_cooldown_after_stop() -> None:
    no_cd = _stopped_day(1, max_trades_per_day=10, cooldown_minutes=0)
    cd = _stopped_day(1, max_trades_per_day=10, cooldown_minutes=30)
    assert len(no_cd.trades) > 1
    assert len(cd.trades) == 1
    t0 = no_cd.trades.iloc[0]
    # stop at 10:02, next signal decided at 10:03 -> fills at 10:04
    assert no_cd.trades.iloc[1].entry_idx == t0.exit_idx + 2


def test_daily_loss_limit_blocks_entries() -> None:
    res = _stopped_day(3, max_trades_per_day=10, max_daily_loss=100.0, cooldown_minutes=0)
    # first stop loses ~ 10pts*20 = 200 > 100 -> no further entries that day
    assert len(res.trades) == 1


def test_session_carry_is_flattened() -> None:
    # early close day (half day 13:15) with flat_time after the close: calendar flattens at 13:10
    day = "2024-11-29"
    rows = day_rows(day, end="13:15")
    df = bars_from(rows)
    res = run(df, signals_at(df, {f"{day} 10:00": (1, 90.0, 120.0)}), nq_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason == "flat_time"
    assert t.exit_time == pd.Timestamp(f"{day} 13:10", tz="America/New_York")


def test_globex_session_carry_flatten() -> None:
    # bars only up to 11:00, then the next session: position must not survive into it
    rows = {k: v for k, v in day_rows(DAY).items() if k < f"{DAY} 11:00"}
    rows.update(day_rows("2024-03-13"))
    df = bars_from(rows)
    res = run(df, signals_at(df, {f"{DAY} 10:00": (1, 90.0, 120.0)}), nq_cfg())
    t = res.trades.iloc[0]
    assert t.exit_reason in ("session", "flat_time")
    assert t.exit_time.tz_convert("America/New_York").date().isoformat() == "2024-03-13"


def test_holiday_no_entries() -> None:
    day = "2024-01-15"  # MLK day
    df = bars_from(day_rows(day, end="13:00"))
    sig = signals_at(df, {f"{day} 10:00": (1, 90.0, 120.0)})
    assert len(run(df, sig, nq_cfg()).trades) == 0
    assert len(run(df, sig, nq_cfg(trade_holidays=True)).trades) == 1


def test_reconciliation_identities(synth_globex: pd.DataFrame) -> None:
    for variant in ("vwap_fade", "vwap_pullback"):
        cfg = AppConfig()
        cfg.strategy.variant = variant
        cfg.strategy.band_k = 1.5
        cfg.costs.slippage_ticks = 2.0
        res = run_backtest(synth_globex, cfg)
        tr = res.trades
        assert len(tr) > 20
        # sum of trade P&L == change in equity
        assert tr["net"].sum() == pytest.approx(res.equity.iloc[-1] - res.initial_capital, abs=1e-6)
        # costs == fills x per-fill cost
        n_fills = 2 * len(tr)
        n_market = len(tr) + int((tr["exit_kind"] == "market").sum())
        assert tr["commission"].sum() == pytest.approx(n_fills * res.commission_per_fill)
        assert tr["slippage"].sum() == pytest.approx(n_market * res.slippage_per_fill)
        assert (tr["net"] - (tr["gross"] - tr["slippage"] - tr["commission"])).abs().max() < 1e-6
        s = res.summary
        assert s["net"] == pytest.approx(s["gross"] - s["slippage"] - s["commission"])
        assert res.position.abs().max() <= cfg.strategy.qty


def test_engine_causality(synth_globex: pd.DataFrame) -> None:
    cfg = AppConfig()
    cfg.strategy.band_k = 1.5
    df = synth_globex.iloc[:6000]
    rng = np.random.default_rng(99)
    cuts = rng.choice(np.arange(500, len(df) - 1), size=25, replace=False)
    assert check_engine_causality(df, cfg, cuts) == []


def test_no_trades_summary() -> None:
    df = bars_from(day_rows(DAY))
    res = run(df, signals_at(df, {}), nq_cfg())
    assert res.summary["trades"] == 0 and res.summary["net"] == 0
