from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqbot.causality import check_signal_causality
from nqbot.config import StrategyConfig
from nqbot.features import Features, compute_features
from nqbot.strategies import get_strategy
from nqbot.strategies.base import Signals, finalize
from nqbot.strategies.vwap import VwapFade, VwapPullback


def feats(**arrays: list[float]) -> Features:
    n = len(arrays["close"])
    base = {
        "open": arrays.get("open", arrays["close"]),
        "high": arrays.get("high", arrays["close"]),
        "low": arrays.get("low", arrays["close"]),
        "volume": [1.0] * n,
        "vwap": [100.0] * n,
        "sd": [1.0] * n,
        "atr": [2.0] * n,
        "slope": [0.0] * n,
    }
    base.update({k: v for k, v in arrays.items() if k in base or k == "close"})
    idx = pd.date_range("2024-03-12 14:00", periods=n, freq="min", tz="UTC")
    return Features(
        index=idx,
        **{k: np.asarray(v, dtype=float) for k, v in base.items()},
        sess_bar=np.arange(n),
        cid=np.zeros(n, dtype=np.int64),
        roll_guard=np.zeros(n, dtype=bool),
        time={"in_session": np.ones(n, dtype=bool)},
    )


def test_fade_long_short_levels() -> None:
    cfg = StrategyConfig(band_k=2.0, fade_stop_atr=1.5, fade_trend_filter=False)
    f = feats(close=[97.5, 100.0, 102.5])
    s = VwapFade().signals(f, cfg, 0.25)
    assert list(s.dir) == [1, 0, -1]
    assert s.at(0) == (1, 94.5, 100.0)  # stop = close - 1.5*ATR, target = VWAP
    assert s.at(2) == (-1, 105.5, 100.0)
    assert np.isnan(s.stop[1])


def test_fade_trend_filter_blocks_falling_knife() -> None:
    cfg = StrategyConfig(band_k=2.0, fade_trend_filter=True, fade_trend_max_slope_atr=1.0)
    f = feats(close=[97.5, 97.5, 102.5, 102.5], slope=[-4.0, -1.0, 4.0, np.nan])
    s = VwapFade().signals(f, cfg, 0.25)
    assert list(s.dir) == [0, 1, 0, 0]


def test_pullback_signals() -> None:
    cfg = StrategyConfig(variant="vwap_pullback", pb_min_slope_atr=0.3, pb_touch_atr=0.25, pb_stop_atr=1.0, pb_target_r=2.0)
    f = feats(
        close=[101.0, 101.0, 99.0, 101.0],
        open=[100.5, 101.5, 99.5, 100.5],
        low=[100.25, 100.25, 98.5, 101.0],
        high=[101.0, 101.5, 99.75, 101.0],
        slope=[1.0, 1.0, -1.0, 1.0],
    )
    s = VwapPullback().signals(f, cfg, 0.25)
    # bar0: uptrend, low within 0.5 of VWAP, bullish close -> long
    # bar1: bearish bar -> none; bar2: short; bar3: low too far from VWAP -> none
    assert list(s.dir) == [1, 0, -1, 0]
    assert s.at(0) == (1, 98.0, 107.0)  # stop vwap-2, risk 3, target close+6
    assert s.at(2) == (-1, 102.0, 93.0)


def test_finalize_rejects_inverted_brackets() -> None:
    c = np.array([100.0, 100.0])
    s = finalize(
        np.array([True, False]), np.array([False, True]),
        np.array([101.0, np.nan]), np.array([105.0, np.nan]),
        np.array([np.nan, 99.0]), np.array([np.nan, 95.0]), c, 0.25,
    )
    assert list(s.dir) == [0, 0]
    both = finalize(np.array([True]), np.array([True]), np.array([99.0]), np.array([101.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]), 0.25)
    assert list(both.dir) == [0]


def test_registry() -> None:
    assert get_strategy(StrategyConfig(variant="vwap_pullback")).name == "vwap_pullback"
    bad = StrategyConfig()
    bad.variant = "x"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        get_strategy(bad)


class LeakyStrategy:
    """Deliberately uses the NEXT bar's close. The causality test must catch it."""

    name = "leaky"

    def signals(self, feat: Features, cfg: StrategyConfig, tick: float) -> Signals:
        c = feat.close
        nxt = np.append(c[1:], np.nan)
        with np.errstate(invalid="ignore"):
            long_ = feat.in_session & (nxt > c + 2 * tick)
        stop = c - 10.0
        tgt = c + 10.0
        return finalize(long_, np.zeros_like(long_), stop, tgt, stop, tgt, c, tick)


@pytest.mark.parametrize("variant", ["vwap_fade", "vwap_pullback"])
@pytest.mark.parametrize("session", ["rth", "globex"])
def test_signal_causality(synth_globex: pd.DataFrame, variant: str, session: str) -> None:
    cfg = StrategyConfig(variant=variant, session=session, band_k=1.5, pb_min_slope_atr=0.1)
    strat = get_strategy(cfg)
    rng = np.random.default_rng(2024)
    df = synth_globex.iloc[:8000]
    cuts = rng.choice(np.arange(100, len(df) - 1), size=60, replace=False)
    # ensure we test bars that actually carry signals
    full_sig = strat.signals(compute_features(df, cfg), cfg, 0.25)
    active = np.flatnonzero(full_sig.dir != 0)
    assert len(active) > 10
    cuts = np.concatenate([cuts, rng.choice(active, size=20, replace=False)])
    assert check_signal_causality(df, strat, cfg, cuts) == []


def test_leaky_strategy_fails_causality(synth_globex: pd.DataFrame) -> None:
    cfg = StrategyConfig()
    df = synth_globex.iloc[:6000]
    rng = np.random.default_rng(7)
    cuts = rng.choice(np.arange(100, len(df) - 1), size=60, replace=False)
    violations = check_signal_causality(df, LeakyStrategy(), cfg, cuts)
    assert len(violations) > 0
    assert all(v.truncated[0] == 0 for v in violations)

