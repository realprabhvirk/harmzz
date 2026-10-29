from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqbot.data import (
    back_adjust,
    contract_ids,
    effective_roll_spread,
    load_bars,
    prepare_bars,
    roll_guard,
    synthetic_bars,
)


def test_synthetic_deterministic_and_on_tick() -> None:
    a = synthetic_bars(days=5, seed=1)
    b = synthetic_bars(days=5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    for col in ("open", "high", "low", "close"):
        assert np.allclose(a[col] / 0.25, np.round(a[col] / 0.25))
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert len(synthetic_bars(days=2, mode="rth")) == 780
    with pytest.raises(ValueError):
        synthetic_bars(days=2, mode="x")


def test_prepare_bars_cleans() -> None:
    df = synthetic_bars(days=1, mode="rth", zero_volume_frac=0.1, seed=2)
    dup = pd.concat([df, df.iloc[:3]])
    out = prepare_bars(dup.drop(columns=["roll_spread", "contract"]))
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert (out["volume"] > 0).all()
    assert len(out) < len(df)
    assert (out["contract"] == "UNKNOWN").all()
    kept = prepare_bars(df, drop_zero_volume=False)
    assert len(kept) == len(df)
    with pytest.raises(ValueError):
        prepare_bars(df.tz_localize(None))
    with pytest.raises(ValueError):
        prepare_bars(df.drop(columns=["volume"]))
    with pytest.raises(ValueError):
        prepare_bars(df.reset_index(drop=True))
    ts = df.reset_index()
    assert len(prepare_bars(ts)) == len(prepare_bars(df))


def test_load_parquet_roundtrip(tmp_path) -> None:
    df = synthetic_bars(days=2, mode="rth")
    p = tmp_path / "bars.parquet"
    df.to_parquet(p)
    out = load_bars(p)
    assert len(out) == len(df)
    assert str(out.index.tz) == "UTC"


def test_roll_guard_and_ids() -> None:
    ids = contract_ids(np.array(["A", "A", "A", "B", "B", "C"]))
    assert list(ids) == [0, 0, 0, 1, 1, 2]
    g = roll_guard(ids)
    assert list(g) == [False, True, True, True, True, False]
    assert len(contract_ids(np.array([]))) == 0


def test_back_adjust_removes_roll_jump() -> None:
    df = prepare_bars(synthetic_bars(start="2024-03-01", days=10, seed=5, gap_vol=0.0))
    roll_i = int(np.flatnonzero(df["roll_spread"].notna())[0])
    spread = df["roll_spread"].iloc[roll_i]
    adj = back_adjust(df)
    # after the roll prices are untouched; before it they are shifted by the spread
    assert adj["close"].iloc[-1] == df["close"].iloc[-1]
    assert adj["close"].iloc[roll_i - 1] == pytest.approx(df["close"].iloc[roll_i - 1] + spread)
    raw_jump = abs(df["open"].iloc[roll_i] - df["close"].iloc[roll_i - 1])
    adj_jump = abs(adj["open"].iloc[roll_i] - adj["close"].iloc[roll_i - 1])
    assert raw_jump > 20 and adj_jump < 10


def test_spread_fallback_uses_gap() -> None:
    df = prepare_bars(synthetic_bars(start="2024-03-01", days=10, seed=5))
    roll_i = int(np.flatnonzero(df["roll_spread"].notna())[0])
    df2 = df.copy()
    df2["roll_spread"] = np.nan
    s = effective_roll_spread(df2)
    assert s[roll_i] == df2["open"].iloc[roll_i] - df2["close"].iloc[roll_i - 1]
    assert effective_roll_spread(df.iloc[:1]).tolist() == [0.0]
