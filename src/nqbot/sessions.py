"""Exchange-time utilities: ET minute-of-day, session ids, approximate CME calendar.

Bars are labelled by their START minute. All local-time logic uses
America/New_York via zoneinfo, so DST is handled by the tz database.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
EPOCH = dt.date(1970, 1, 1)

RTH_START = 9 * 60 + 30  # 09:30
RTH_END = 16 * 60  # 16:00
GLOBEX_START = 18 * 60  # 18:00 (previous calendar day)
HALT_START = 17 * 60  # daily maintenance halt 17:00-18:00
HALF_DAY_CLOSE = 13 * 60 + 15  # 13:15 ET (12:15 CT) early close
HOLIDAY_CLOSE = 13 * 60  # 13:00 ET (12:00 CT) halt on US holidays

NO_ENTRY = -1


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def us_holidays(year: int) -> frozenset[dt.date]:
    """US holidays on which CME equity futures trade an abbreviated session (approximate)."""
    days = {
        _nth_weekday(year, 1, 0, 3),  # MLK
        _nth_weekday(year, 2, 0, 3),  # Presidents
        _last_weekday(year, 5, 0),  # Memorial
        _observed(dt.date(year, 7, 4)),  # Independence
        _nth_weekday(year, 9, 0, 1),  # Labor
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
    }
    if year >= 2022:
        days.add(_observed(dt.date(year, 6, 19)))  # Juneteenth
    return frozenset(days)


@lru_cache(maxsize=None)
def half_days(year: int) -> frozenset[dt.date]:
    """13:15 ET early-close days (approximate)."""
    days = {_nth_weekday(year, 11, 3, 4) + dt.timedelta(days=1)}  # day after Thanksgiving
    xmas_eve = dt.date(year, 12, 24)
    if xmas_eve.weekday() <= 3:
        days.add(xmas_eve)
    jul3 = dt.date(year, 7, 3)
    if jul3.weekday() <= 3:
        days.add(jul3)
    return frozenset(days)


def session_close_mod(day: dt.date) -> tuple[int, bool]:
    """(close minute-of-day ET, is_us_holiday) for a session date."""
    if day in us_holidays(day.year):
        return HOLIDAY_CLOSE, True
    if day in half_days(day.year):
        return HALF_DAY_CLOSE, False
    return RTH_END, False


def to_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("bar index must be timezone-aware (UTC)")
    return index.tz_convert("UTC")


def local_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since epoch on the ET wall clock (DST-aware)."""
    wall = index.tz_convert(ET).tz_localize(None).as_unit("ns")
    return wall.asi8 // 60_000_000_000


def session_arrays(
    index: pd.DatetimeIndex,
    mode: str = "rth",
    flat_time: int = 15 * 60 + 55,
    entry_end: int = 15 * 60 + 30,
    trade_holidays: bool = False,
) -> dict[str, np.ndarray]:
    """Per-bar time arrays used by features and the decision kernel.

    Returns utc_min, et_mod (ET minute-of-day of the bar label), session_id
    (ordinal of the session date), in_session, sess_el (minutes from session
    start to the bar label), flat_mod and entry_end_mod (per-bar, calendar aware;
    entry_end_mod == NO_ENTRY on no-trade days).
    """
    index = to_utc_index(index).as_unit("ns")
    utc_min = index.asi8 // 60_000_000_000
    lmin = local_minutes(index)
    et_mod = lmin % 1440
    if mode == "rth":
        session_id = lmin // 1440
        sess_el = et_mod - RTH_START
    elif mode == "globex":
        session_id = (lmin + (1440 - GLOBEX_START)) // 1440
        sess_el = (et_mod - GLOBEX_START) % 1440
    else:
        raise ValueError(f"unknown session mode {mode!r}")

    days, inverse = np.unique(session_id, return_inverse=True)
    close = np.empty(len(days), dtype=np.int64)
    holiday = np.empty(len(days), dtype=bool)
    for i, d in enumerate(days):
        close[i], holiday[i] = session_close_mod(EPOCH + dt.timedelta(days=int(d)))
    close_b = close[inverse]
    holiday_b = holiday[inverse]

    if mode == "rth":
        in_session = (et_mod >= RTH_START) & (et_mod < close_b)
    else:
        in_session = ~((et_mod >= HALT_START) & (et_mod < GLOBEX_START))

    gap = flat_time - entry_end
    flat_mod = np.minimum(flat_time, close_b - 5).astype(np.int64)
    entry_end_mod = np.minimum(entry_end, flat_mod - gap).astype(np.int64)
    if not trade_holidays:
        entry_end_mod = np.where(holiday_b, NO_ENTRY, entry_end_mod)
    return {
        "utc_min": utc_min.astype(np.int64),
        "et_mod": et_mod.astype(np.int64),
        "session_id": session_id.astype(np.int64),
        "in_session": in_session,
        "sess_el": sess_el.astype(np.int64),
        "flat_mod": flat_mod,
        "entry_end_mod": entry_end_mod.astype(np.int64),
    }


_EPOCH_NAIVE = dt.datetime(1970, 1, 1)


def bar_time_fields(
    utc_min: int,
    mode: str = "rth",
    flat_time: int = 15 * 60 + 55,
    entry_end: int = 15 * 60 + 30,
    trade_holidays: bool = False,
) -> dict[str, int | bool]:
    """Scalar version of ``session_arrays`` for one bar (used by the live runner).

    Must agree exactly with ``session_arrays``; tests check this across DST days.
    """
    wall = dt.datetime.fromtimestamp(int(utc_min) * 60, tz=ET).replace(tzinfo=None)
    lmin = int((wall - _EPOCH_NAIVE).total_seconds()) // 60
    et_mod = lmin % 1440
    if mode == "rth":
        session_id = lmin // 1440
        sess_el = et_mod - RTH_START
    elif mode == "globex":
        session_id = (lmin + (1440 - GLOBEX_START)) // 1440
        sess_el = (et_mod - GLOBEX_START) % 1440
    else:
        raise ValueError(f"unknown session mode {mode!r}")
    close, holiday = session_close_mod(EPOCH + dt.timedelta(days=session_id))
    if mode == "rth":
        in_session = RTH_START <= et_mod < close
    else:
        in_session = not (HALT_START <= et_mod < GLOBEX_START)
    flat_mod = min(flat_time, close - 5)
    entry_end_mod = min(entry_end, flat_mod - (flat_time - entry_end))
    if holiday and not trade_holidays:
        entry_end_mod = NO_ENTRY
    return {
        "utc_min": int(utc_min),
        "et_mod": et_mod,
        "session_id": session_id,
        "in_session": in_session,
        "sess_el": sess_el,
        "flat_mod": flat_mod,
        "entry_end_mod": entry_end_mod,
    }


def in_window(mod: int, start: int, end: int) -> bool:
    """Is minute-of-day ``mod`` in [start, end)? Handles windows that wrap midnight."""
    if start <= end:
        return start <= mod < end
    return mod >= start or mod < end


def et_now_mod(ts: pd.Timestamp) -> int:
    t = ts.tz_convert(ET)
    return t.hour * 60 + t.minute
