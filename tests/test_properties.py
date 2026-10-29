"""Property tests on random OHLC data and random signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nqbot.config import AppConfig
from nqbot.data import prepare_bars
from nqbot.engine import run_signals
from nqbot.features import compute_features
from nqbot.strategies.base import Signals

TICK = 0.25


def random_bars(seed: int, n_days: int, vol_ticks: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-04-01", periods=n_days)
    idx = []
    for d in days:
        t0 = pd.Timestamp(f"{d.date()} 09:00", tz="America/New_York")
        idx.extend(t0 + pd.Timedelta(minutes=i) for i in range(7 * 60 + 30))
    n = len(idx)
    steps = np.round(rng.standard_normal(n) * vol_ticks) * TICK
    close = 15000 + np.cumsum(steps)
    gaps = np.where(rng.random(n) < 0.02, np.round(rng.standard_normal(n) * 8) * TICK, 0.0)
    open_ = np.concatenate(([close[0]], close[:-1])) + gaps
    hi = np.maximum(open_, close) + np.round(np.abs(rng.standard_normal(n)) * vol_ticks) * TICK
    lo = np.minimum(open_, close) - np.round(np.abs(rng.standard_normal(n)) * vol_ticks) * TICK
    vol = rng.integers(0, 50, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol, "contract": "NQM4", "roll_spread": np.nan},
        index=pd.DatetimeIndex(idx).tz_convert("UTC"),
    )
    return prepare_bars(df)


def random_signals(df: pd.DataFrame, seed: int, p: float) -> Signals:
    rng = np.random.default_rng(seed + 1)
    n = len(df)
    c = df["close"].to_numpy()
    d = np.where(rng.random(n) < p, rng.choice([-1, 1], n), 0).astype(np.int8)
    dist_s = (1 + rng.integers(0, 40, n)) * TICK
    dist_t = (1 + rng.integers(0, 40, n)) * TICK
    stop = np.where(d > 0, c - dist_s, np.where(d < 0, c + dist_s, np.nan))
    tgt = np.where(d > 0, c + dist_t, np.where(d < 0, c - dist_t, np.nan))
    return Signals(d, stop, tgt)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    n_days=st.integers(1, 4),
    vol=st.floats(0.5, 12.0),
    p=st.floats(0.0, 0.5),
    qty=st.integers(1, 3),
    slip=st.sampled_from([0.0, 0.5, 1.0, 3.0]),
    max_trades=st.integers(0, 6),
    cooldown=st.integers(0, 20),
    instrument=st.sampled_from(["NQ", "MNQ"]),
)
def test_equity_identity_and_position_limits(
    seed: int, n_days: int, vol: float, p: float, qty: int, slip: float,
    max_trades: int, cooldown: int, instrument: str,
) -> None:
    df = random_bars(seed, n_days, vol)
    cfg = AppConfig()
    cfg.costs.instrument = instrument
    cfg.costs.slippage_ticks = slip
    cfg.strategy.qty = qty
    cfg.strategy.max_trades_per_day = max_trades
    cfg.strategy.cooldown_minutes = cooldown
    feat = compute_features(df, cfg.strategy)
    res = run_signals(df, feat, random_signals(df, seed, p), cfg.strategy, cfg.costs)
    tr = res.trades

    # equity identity
    assert abs(tr["net"].sum() - (res.equity.iloc[-1] - res.initial_capital)) < 1e-6
    assert ((tr["gross"] - tr["slippage"] - tr["commission"] - tr["net"]).abs() < 1e-6).all()
    # costs: every fill pays commission, every market fill pays slippage
    n_market = len(tr) + int((tr["exit_kind"] == "market").sum())
    assert abs(tr["commission"].sum() - 2 * len(tr) * res.commission_per_fill) < 1e-6
    assert abs(tr["slippage"].sum() - n_market * res.slippage_per_fill) < 1e-6
    # position limits, one position at a time, ordering
    assert res.position.abs().max() <= qty
    assert (tr["exit_idx"] >= tr["entry_idx"]).all()
    assert (tr["entry_idx"].to_numpy()[1:] > tr["exit_idx"].to_numpy()[:-1]).all()
    assert res.position.iloc[-1] == 0
    # flat outside the trading window
    mod = feat.time["et_mod"]
    assert (res.position.to_numpy()[(mod >= 955) | (mod < 585)] == 0).all()
    # max trades per day
    if len(tr):
        per_day = tr.groupby(tr["entry_time"].dt.tz_convert("America/New_York").dt.date).size()
        assert per_day.max() <= max_trades
    # stops never fill better than the stop price
    stops = tr[tr["exit_reason"] == "stop"]
    assert ((stops["exit_fill"] - stops["stop"]) * stops["dir"] <= 1e-9).all()
    targets = tr[tr["exit_reason"] == "target"]
    assert (targets["exit_fill"] == targets["target"]).all()
