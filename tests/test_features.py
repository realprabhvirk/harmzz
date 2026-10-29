from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqbot.config import StrategyConfig
from nqbot.data import prepare_bars
from nqbot.features import compute_features
from tests.conftest import make_bars


def _vwap_sd(tps: list[float], vols: list[float]) -> tuple[float, float]:
    """Hand formula: VWAP and volume-weighted standard deviation."""
    v = sum(vols)
    m = sum(t * w for t, w in zip(tps, vols)) / v
    var = sum(w * (t - m) ** 2 for t, w in zip(tps, vols)) / v
    return m, var**0.5


def test_vwap_hand_computed_with_dst_reset() -> None:
    rows = [
        ("2024-03-08 09:29", 99, 99, 99, 99, 50),  # pre-open: not in RTH session
        ("2024-03-08 09:30", 100, 101, 99, 100, 10),  # tp 100
        ("2024-03-08 09:31", 100, 103, 97, 103, 30),  # tp 101
        ("2024-03-08 09:32", 104, 104, 104, 104, 20),  # tp 104
        # Monday after the US DST switch (EST -> EDT): new session must reset
        ("2024-03-11 09:30", 200, 202, 198, 200, 5),  # tp 200
        ("2024-03-11 09:31", 200, 206, 200, 203, 15),  # tp 203
    ]
    df = make_bars(rows)
    f = compute_features(df, StrategyConfig(session="rth"))
    assert np.isnan(f.vwap[0])
    assert f.vwap[1] == pytest.approx(100.0) and f.sd[1] == pytest.approx(0.0)
    m, s = _vwap_sd([100, 101], [10, 30])
    assert (f.vwap[2], f.sd[2]) == pytest.approx((m, s))
    assert m == pytest.approx(100.75) and s**2 == pytest.approx(0.1875)
    m, s = _vwap_sd([100, 101, 104], [10, 30, 20])
    assert (f.vwap[3], f.sd[3]) == pytest.approx((m, s))
    assert f.vwap[4] == pytest.approx(200.0)
    m, s = _vwap_sd([200, 203], [5, 15])
    assert (f.vwap[5], f.sd[5]) == pytest.approx((m, s))
    # bands
    k = 2.0
    assert f.upper(k)[3] == pytest.approx(f.vwap[3] + k * f.sd[3])
    assert f.lower(k)[3] == pytest.approx(f.vwap[3] - k * f.sd[3])
    assert list(f.sess_bar) == [-1, 0, 1, 2, 0, 1]
    # UTC hour of the 09:30 bar differs across the DST switch
    assert df.index[1].hour == 14 and df.index[4].hour == 13


def test_globex_vwap_resets_at_1800_on_dst_sunday() -> None:
    rows = [
        ("2024-03-08 16:58", 100, 100, 100, 100, 10),
        ("2024-03-08 16:59", 102, 102, 102, 102, 10),
        ("2024-03-10 18:00", 110, 110, 110, 110, 10),  # first EDT session
        ("2024-03-10 18:01", 112, 112, 112, 112, 30),
    ]
    f = compute_features(make_bars(rows), StrategyConfig(session="globex"))
    assert f.vwap[1] == pytest.approx(101.0)
    assert f.vwap[2] == pytest.approx(110.0)
    assert f.vwap[3] == pytest.approx((110 * 10 + 112 * 30) / 40)


def test_atr_true_range_and_slope() -> None:
    rows = [
        ("2024-03-12 09:30", 100, 101, 99, 100, 1),
        ("2024-03-12 09:31", 100, 104, 100, 103, 1),  # TR = 4
        ("2024-03-12 09:32", 106, 107, 105, 106, 1),  # TR = max(107,103)-min(105,103) = 4
        ("2024-03-12 09:33", 106, 106, 100, 101, 1),  # TR = 6
    ]
    cfg = StrategyConfig(atr_len=2, slope_lookback=2)
    f = compute_features(make_bars(rows), cfg)
    assert np.isnan(f.atr[0]) and np.isnan(f.atr[1])
    assert f.atr[2] == pytest.approx(4.0)
    assert f.atr[3] == pytest.approx(5.0)
    assert np.isnan(f.slope[1])
    assert f.slope[2] == pytest.approx(f.vwap[2] - f.vwap[0])
    assert f.slope_atr[3] == pytest.approx(f.slope[3] / 5.0)


def test_roll_adjusts_true_range_and_resets_vwap() -> None:
    rows = [
        ("2024-03-12 09:30", 100, 100, 100, 100, 1),
        ("2024-03-12 09:31", 100, 100, 100, 100, 1),
        ("2024-03-12 09:32", 150, 151, 149, 150, 1),  # new contract trades 50 higher
    ]
    df = make_bars(rows, contract=["NQH4", "NQH4", "NQM4"])
    df.loc[df.index[2], "roll_spread"] = 50.0
    f = compute_features(df, StrategyConfig(atr_len=1))
    assert f.atr[2] == pytest.approx(2.0)  # H-L only, the 50pt roll gap is removed
    assert f.vwap[2] == pytest.approx(150.0)  # reset on contract change
    assert f.roll_guard[0] and f.roll_guard[1] and not f.roll_guard[2]


def test_zero_volume_bar_does_not_move_vwap() -> None:
    rows = [
        ("2024-03-12 09:30", 100, 100, 100, 100, 10),
        ("2024-03-12 09:31", 120, 120, 120, 120, 0),
    ]
    df = make_bars(rows)
    assert len(df) == 1  # dropped at ingestion
    raw = prepare_bars(df, drop_zero_volume=False)
    f = compute_features(raw, StrategyConfig())
    assert f.vwap[0] == 100.0


def test_truncated_buffer_is_bit_identical(synth_globex: pd.DataFrame) -> None:
    """Live parity foundation: features of the last bar from a 2000-bar buffer == full history."""
    cfg = StrategyConfig()
    full = compute_features(synth_globex, cfg)
    for end in (5000, 12345, len(synth_globex)):
        part = synth_globex.iloc[end - 2000 : end]
        f = compute_features(part, cfg)
        for name in ("vwap", "sd", "atr", "slope"):
            a, b = getattr(full, name)[end - 1], getattr(f, name)[-1]
            assert (a == b) or (np.isnan(a) and np.isnan(b)), name
    frame = full.frame()
    assert {"vwap", "atr", "et_mod"} <= set(frame.columns)
