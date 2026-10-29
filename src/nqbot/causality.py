"""Causality checks: a signal at t must not change when data after t is removed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nqbot.config import AppConfig, StrategyConfig
from nqbot.features import compute_features
from nqbot.instruments import get_instrument
from nqbot.strategies.base import Strategy


@dataclass
class Violation:
    t: int
    full: tuple[int, float, float]
    truncated: tuple[int, float, float]


def _same(a: tuple[int, float, float], b: tuple[int, float, float]) -> bool:
    if a[0] != b[0]:
        return False
    for x, y in zip(a[1:], b[1:]):
        if not (x == y or (np.isnan(x) and np.isnan(y))):
            return False
    return True


def check_signal_causality(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: StrategyConfig,
    cut_points: list[int] | np.ndarray,
    tick: float = 0.25,
) -> list[Violation]:
    """For each t: signal(t | data[:t+1]) must equal signal(t | full data)."""
    # The roll guard is calendar information; give the truncated run the same guard.
    full_feat = compute_features(df, cfg)
    full = strategy.signals(full_feat, cfg, tick)
    out: list[Violation] = []
    for t in cut_points:
        t = int(t)
        part = df.iloc[: t + 1]
        feat = compute_features(part, cfg, guard=full_feat.roll_guard[: t + 1])
        sig = strategy.signals(feat, cfg, tick)
        a, b = full.at(t), sig.at(t)
        if not _same(a, b):
            out.append(Violation(t, a, b))
    return out


def check_engine_causality(
    df: pd.DataFrame,
    cfg: AppConfig,
    cut_points: list[int] | np.ndarray,
) -> list[int]:
    """Entries decided up to bar t must be identical when the engine only sees data[:t+1].

    Returns the cut points where entries (index, dir, fill, stop, target) differ.
    """
    from nqbot.engine import run_signals
    from nqbot.strategies import get_strategy

    strat = get_strategy(cfg.strategy)
    tick = get_instrument(cfg.costs.instrument).tick_size
    full_feat = compute_features(df, cfg.strategy)
    full = run_signals(df, full_feat, strat.signals(full_feat, cfg.strategy, tick), cfg.strategy, cfg.costs)
    cols = ["entry_idx", "dir", "entry_fill", "stop", "target"]
    bad: list[int] = []
    for t in cut_points:
        t = int(t)
        part = df.iloc[: t + 1]
        feat = compute_features(part, cfg.strategy, guard=full_feat.roll_guard[: t + 1])
        res = run_signals(part, feat, strat.signals(feat, cfg.strategy, tick), cfg.strategy, cfg.costs)
        a = full.trades.loc[full.trades["entry_idx"] <= t, cols].reset_index(drop=True)
        b = res.trades[cols].reset_index(drop=True)
        if not a.equals(b):
            bad.append(t)
    return bad
