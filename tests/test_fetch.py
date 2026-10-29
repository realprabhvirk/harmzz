from __future__ import annotations

import numpy as np
import pandas as pd

from nqbot.data import back_adjust, prepare_bars
from nqbot.data_quality import quality_report
from nqbot.fetch import build_canonical


def _frame(times: list[str], closes: list[float], ids: list[int]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(times, tz="UTC")
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"instrument_id": ids, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 5.0}, index=idx)


def test_build_canonical_roll_spread_from_second_month() -> None:
    t = ["2024-03-13 21:58", "2024-03-13 21:59", "2024-03-13 23:00", "2024-03-13 23:01"]
    v0 = _frame(t, [100, 101, 160, 161], [1, 1, 2, 2])  # contract 2 becomes front at 23:00
    v1 = _frame(t[:2] + t[2:], [158, 160, 200, 201], [2, 2, 3, 3])
    out = build_canonical(v0, v1, {1: "NQH4", 2: "NQM4"})
    assert list(out["contract"]) == ["NQH4", "NQH4", "NQM4", "NQM4"]
    assert np.isnan(out["roll_spread"].iloc[1])
    assert out["roll_spread"].iloc[2] == 160 - 101  # v1 - v0 at 21:59
    canon = prepare_bars(out)
    adj = back_adjust(canon)
    assert adj["close"].iloc[1] == 101 + 59
    assert quality_report(out)["rolls"]["rolls"] == 1


def test_build_canonical_without_v1_or_overlap() -> None:
    t = ["2024-03-13 21:59", "2024-03-13 23:00"]
    v0 = _frame(t, [100, 160], [1, 2])
    out = build_canonical(v0, None)
    assert list(out["contract"]) == ["id1", "id2"] and out["roll_spread"].isna().all()
    v1 = _frame(["2024-03-13 10:00"], [150], [2])  # too old to use
    assert build_canonical(v0, v1)["roll_spread"].isna().all()
