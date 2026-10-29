from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from nqbot.sessions import (
    HALF_DAY_CLOSE,
    HOLIDAY_CLOSE,
    NO_ENTRY,
    et_now_mod,
    half_days,
    in_window,
    session_arrays,
    session_close_mod,
    us_holidays,
)


def _idx(*local: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(s, tz="America/New_York") for s in local]).tz_convert("UTC")


def test_rth_open_is_0930_et_across_dst() -> None:
    # Before and after the March 2024 DST change the 09:30 ET bar has different UTC times
    idx = _idx("2024-03-08 09:30", "2024-03-11 09:30", "2024-11-01 09:30", "2024-11-04 09:30")
    assert [t.hour for t in idx] == [14, 13, 13, 14]
    s = session_arrays(idx, "rth")
    assert (s["et_mod"] == 570).all()
    assert (s["sess_el"] == 0).all()
    assert s["in_session"].all()
    assert len(np.unique(s["session_id"])) == 4


def test_globex_session_assignment() -> None:
    idx = _idx("2024-03-10 18:00", "2024-03-11 09:30", "2024-03-11 16:59", "2024-03-11 18:00", "2024-03-11 17:30")
    s = session_arrays(idx, "globex")
    sid = s["session_id"]
    assert sid[0] == sid[1] == sid[2]
    assert sid[3] == sid[0] + 1
    assert s["sess_el"][0] == 0
    assert s["sess_el"][1] == 15 * 60 + 30
    assert not s["in_session"][4]  # maintenance halt


def test_rth_out_of_session_and_flat_mod() -> None:
    idx = _idx("2024-03-12 09:29", "2024-03-12 15:59", "2024-03-12 16:00")
    s = session_arrays(idx, "rth")
    assert list(s["in_session"]) == [False, True, False]
    assert (s["flat_mod"] == 955).all()
    assert (s["entry_end_mod"] == 930).all()


def test_calendar_half_day_and_holiday() -> None:
    assert dt.date(2024, 11, 29) in half_days(2024)
    assert dt.date(2024, 7, 3) in half_days(2024)
    assert dt.date(2024, 12, 24) in half_days(2024)
    assert dt.date(2024, 11, 28) in us_holidays(2024)
    assert dt.date(2022, 6, 20) in us_holidays(2022)  # Juneteenth observed (Sun -> Mon)
    assert dt.date(2021, 6, 18) not in us_holidays(2021)
    assert dt.date(2026, 7, 3) in us_holidays(2026)  # July 4 on Saturday -> Friday
    assert session_close_mod(dt.date(2024, 11, 29)) == (HALF_DAY_CLOSE, False)
    assert session_close_mod(dt.date(2024, 1, 15)) == (HOLIDAY_CLOSE, True)
    s = session_arrays(_idx("2024-11-29 10:00", "2024-01-15 10:00", "2024-01-15 10:00"), "rth")
    assert s["flat_mod"][0] == HALF_DAY_CLOSE - 5
    assert s["entry_end_mod"][0] == HALF_DAY_CLOSE - 5 - 25
    assert s["entry_end_mod"][1] == NO_ENTRY
    s2 = session_arrays(_idx("2024-01-15 10:00"), "rth", trade_holidays=True)
    assert s2["entry_end_mod"][0] == HOLIDAY_CLOSE - 5 - 25


def test_naive_index_rejected_and_bad_mode() -> None:
    import pytest

    with pytest.raises(ValueError):
        session_arrays(pd.DatetimeIndex(["2024-01-02 10:00"]), "rth")
    with pytest.raises(ValueError):
        session_arrays(_idx("2024-01-02 10:00"), "asia")


def test_window_helpers() -> None:
    assert in_window(10, 5, 20)
    assert not in_window(20, 5, 20)
    assert in_window(1000, 956, 1085)
    assert in_window(1430, 1400, 60) and in_window(30, 1400, 60) and not in_window(600, 1400, 60)
    assert et_now_mod(pd.Timestamp("2024-07-01 13:30", tz="UTC")) == 9 * 60 + 30
