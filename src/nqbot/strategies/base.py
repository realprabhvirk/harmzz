"""Strategy interface: a pure, causal map from features to per-bar entry proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nqbot.config import StrategyConfig
from nqbot.features import Features


@dataclass
class Signals:
    """Per-bar proposal decided at bar close: direction (+1/-1/0), raw stop and target."""

    dir: np.ndarray  # int8
    stop: np.ndarray  # float64, NaN where dir == 0
    target: np.ndarray  # float64, NaN where dir == 0

    def __len__(self) -> int:
        return len(self.dir)

    def at(self, t: int) -> tuple[int, float, float]:
        return int(self.dir[t]), float(self.stop[t]), float(self.target[t])


class Strategy(Protocol):
    name: str

    def signals(self, feat: Features, cfg: StrategyConfig, tick: float) -> Signals: ...


def round_tick(x: np.ndarray, tick: float) -> np.ndarray:
    return np.round(x / tick) * tick


def finalize(
    long_: np.ndarray,
    short: np.ndarray,
    stop_l: np.ndarray,
    tgt_l: np.ndarray,
    stop_s: np.ndarray,
    tgt_s: np.ndarray,
    close: np.ndarray,
    tick: float,
) -> Signals:
    """Round levels to tick, drop proposals whose bracket is not stop < close < target."""
    stop_l, tgt_l = round_tick(stop_l, tick), round_tick(tgt_l, tick)
    stop_s, tgt_s = round_tick(stop_s, tick), round_tick(tgt_s, tick)
    with np.errstate(invalid="ignore"):
        long_ = long_ & (stop_l < close) & (close < tgt_l)
        short = short & (stop_s > close) & (close > tgt_s)
    both = long_ & short
    long_, short = long_ & ~both, short & ~both
    d = np.zeros(len(close), dtype=np.int8)
    d[long_] = 1
    d[short] = -1
    stop = np.where(long_, stop_l, np.where(short, stop_s, np.nan))
    tgt = np.where(long_, tgt_l, np.where(short, tgt_s, np.nan))
    return Signals(d, stop.astype(np.float64), tgt.astype(np.float64))
