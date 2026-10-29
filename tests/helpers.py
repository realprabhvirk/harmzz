"""Hand-crafted scenarios for execution tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nqbot.config import AppConfig
from nqbot.engine import BacktestResult, run_signals
from nqbot.features import compute_features
from nqbot.strategies.base import Signals
from tests.conftest import make_bars


def day_rows(day: str, price: float = 100.0, start: str = "09:30", end: str = "16:00") -> dict[str, list]:
    t0, t1 = pd.Timestamp(f"{day} {start}"), pd.Timestamp(f"{day} {end}")
    out: dict[str, list] = {}
    t = t0
    while t < t1:
        out[t.strftime("%Y-%m-%d %H:%M")] = [price, price, price, price, 10.0]
        t += pd.Timedelta(minutes=1)
    return out


def bars_from(rows: dict[str, list], contracts: dict[str, str] | None = None) -> pd.DataFrame:
    keys = sorted(rows)
    data = [(k, *rows[k]) for k in keys]
    cons = [contracts.get(k, "NQZ4") if contracts else "NQZ4" for k in keys]
    if contracts:
        current = "NQZ4"
        cons = []
        for k in keys:
            current = contracts.get(k, current)
            cons.append(current)
    return make_bars(data, contract=cons)


def signals_at(df: pd.DataFrame, entries: dict[str, tuple[int, float, float]]) -> Signals:
    n = len(df)
    d = np.zeros(n, dtype=np.int8)
    stop = np.full(n, np.nan)
    tgt = np.full(n, np.nan)
    local = df.index.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
    pos = {k: i for i, k in enumerate(local)}
    for k, (dd, s, t) in entries.items():
        i = pos[k]
        d[i], stop[i], tgt[i] = dd, s, t
    return Signals(d, stop, tgt)


def nq_cfg(**strategy: object) -> AppConfig:
    cfg = AppConfig()
    cfg.costs.instrument = "NQ"
    cfg.costs.slippage_ticks = 1.0
    cfg.costs.commission_per_side = 2.25
    cfg.strategy.cooldown_minutes = 0
    for k, v in strategy.items():
        setattr(cfg.strategy, k, v)
    return cfg


def run(df: pd.DataFrame, sig: Signals, cfg: AppConfig) -> BacktestResult:
    feat = compute_features(df, cfg.strategy)
    return run_signals(df, feat, sig, cfg.strategy, cfg.costs)
