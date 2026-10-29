"""Strategy registry."""

from __future__ import annotations

from nqbot.config import StrategyConfig
from nqbot.strategies.base import Signals, Strategy
from nqbot.strategies.vwap import VwapFade, VwapPullback

STRATEGIES: dict[str, type] = {"vwap_fade": VwapFade, "vwap_pullback": VwapPullback}


def get_strategy(cfg: StrategyConfig) -> Strategy:
    try:
        return STRATEGIES[cfg.variant]()
    except KeyError as exc:
        raise ValueError(f"unknown strategy variant {cfg.variant!r}") from exc


__all__ = ["Signals", "Strategy", "get_strategy", "STRATEGIES", "VwapFade", "VwapPullback"]
