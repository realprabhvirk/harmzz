"""Bars for the live runner: replay feed, 5-second -> 1-minute aggregation, IB feed."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nqbot.data import contract_ids, effective_roll_spread, roll_guard


@dataclass
class Bar:
    label_min: float  # UTC minute of the bar START
    open: float
    high: float
    low: float
    close: float
    volume: float
    contract: str
    roll_spread: float = float("nan")  # on the first bar of a new contract
    roll_guard: bool = False  # calendar: roll within the next two bars

    @property
    def ts(self) -> pd.Timestamp:
        return pd.Timestamp(int(self.label_min) * 60_000_000_000, tz="UTC")


def bars_from_frame(df: pd.DataFrame) -> list[Bar]:
    """Replay bars from a canonical frame, carrying calendar roll info."""
    cids = contract_ids(df["contract"])
    guard = roll_guard(cids)
    spread = effective_roll_spread(df)
    utc_min = df.index.as_unit("ns").asi8 // 60_000_000_000
    o, h, low, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    con = df["contract"].to_numpy()
    out = []
    for i in range(len(df)):
        rs = spread[i] if i > 0 and cids[i] != cids[i - 1] else float("nan")
        out.append(Bar(float(utc_min[i]), o[i], h[i], low[i], c[i], v[i], str(con[i]), rs, bool(guard[i])))
    return out


class ReplayFeed:
    def __init__(self, df: pd.DataFrame) -> None:
        self.bars = bars_from_frame(df)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self.bars)

    def __len__(self) -> int:
        return len(self.bars)


class MinuteAggregator:
    """Aggregates sub-minute bars (e.g. IB 5-second real-time bars) into 1-minute bars.

    A minute is emitted when data for a later minute arrives, or when the wall
    clock passes the minute end + ``grace_s``. Minutes with no volume are dropped.
    """

    def __init__(self, contract: str, grace_s: float = 3.0) -> None:
        self.contract = contract
        self.grace_s = grace_s
        self._cur: list[float] | None = None  # [label_min, o, h, l, c, v]

    def add(self, ts: dt.datetime, o: float, h: float, low: float, c: float, v: float) -> list[Bar]:
        label = math.floor(ts.timestamp() / 60.0)
        out: list[Bar] = []
        if self._cur is not None and label > self._cur[0]:
            out += self._emit()
        if self._cur is None:
            self._cur = [float(label), o, h, low, c, v]
        else:
            cur = self._cur
            cur[2] = max(cur[2], h)
            cur[3] = min(cur[3], low)
            cur[4] = c
            cur[5] += v
        return out

    def flush_if_due(self, now: dt.datetime) -> list[Bar]:
        if self._cur is not None and now.timestamp() >= (self._cur[0] + 1) * 60 + self.grace_s:
            return self._emit()
        return []

    def _emit(self) -> list[Bar]:
        cur, self._cur = self._cur, None
        if cur is None or cur[5] <= 0:
            return []
        return [Bar(cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], self.contract)]


class IBBarFeed:
    """Polls an ib_insync/ib_async RealTimeBarList (5-second TRADES bars)."""

    def __init__(self, ib: Any, contract: Any, contract_code: str, grace_s: float = 3.0) -> None:
        self.ib = ib
        self.contract = contract
        self.agg = MinuteAggregator(contract_code, grace_s)
        self.rt = ib.reqRealTimeBars(contract, 5, "TRADES", False)
        self._seen = 0
        self.last_update: dt.datetime | None = None

    def history(self, duration: str = "2 D") -> list[Bar]:
        bars = self.ib.reqHistoricalData(
            self.contract, endDateTime="", durationStr=duration, barSizeSetting="1 min",
            whatToShow="TRADES", useRTH=False, formatDate=2,
        )
        out = []
        for b in bars:
            ts = b.date if isinstance(b.date, dt.datetime) else pd.Timestamp(b.date).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            if b.volume and b.volume > 0:
                out.append(Bar(float(math.floor(ts.timestamp() / 60)), b.open, b.high, b.low, b.close, float(b.volume), self.agg.contract))
        return out

    def poll(self, now: dt.datetime) -> list[Bar]:
        out: list[Bar] = []
        new = list(self.rt)[self._seen :]
        self._seen += len(new)
        for b in new:
            ts = b.time if b.time.tzinfo else b.time.replace(tzinfo=dt.timezone.utc)
            self.last_update = ts
            out += self.agg.add(ts, b.open_, b.high, b.low, b.close, float(b.volume))
        out += self.agg.flush_if_due(now)
        return out

