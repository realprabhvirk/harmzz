"""Causal features: session VWAP, volume-weighted sigma bands, ATR, VWAP slope.

Every value at bar t depends only on bars <= t. Loops are explicit so the
floating-point result depends only on the current session (VWAP) or a fixed
window (ATR), which makes a truncated live buffer bit-identical to full history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from nqbot.config import StrategyConfig, hhmm
from nqbot.data import contract_ids, roll_guard
from nqbot.sessions import session_arrays


@njit(cache=True)
def _feature_kernel(
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    cid: np.ndarray,
    spread: np.ndarray,
    session_id: np.ndarray,
    in_session: np.ndarray,
    atr_len: int,
    slope_lb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(c)
    vwap = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    sess_bar = np.full(n, -1, dtype=np.int64)
    tr = np.full(n, np.nan)

    anchor = 0.0
    spv = 0.0
    spv2 = 0.0
    sv = 0.0
    k = -1
    for t in range(n):
        # --- true range with roll-adjusted previous close ---
        if t > 0:
            pc = c[t - 1]
            if cid[t] != cid[t - 1]:
                pc = pc + spread[t]
            hi = h[t] if h[t] > pc else pc
            lo = low[t] if low[t] < pc else pc
            tr[t] = hi - lo
        if t >= atr_len:
            s = 0.0
            for j in range(t - atr_len + 1, t + 1):
                s += tr[j]
            atr[t] = s / atr_len

        # --- session VWAP (resets on new session or contract change) ---
        if in_session[t]:
            new = (
                t == 0
                or (not in_session[t - 1])
                or session_id[t] != session_id[t - 1]
                or cid[t] != cid[t - 1]
            )
            tp = (h[t] + low[t] + c[t]) / 3.0
            if new:
                anchor = tp
                spv = 0.0
                spv2 = 0.0
                sv = 0.0
                k = -1
            k += 1
            x = tp - anchor
            if v[t] > 0:
                spv += v[t] * x
                spv2 += v[t] * x * x
                sv += v[t]
            sess_bar[t] = k
            if sv > 0:
                mean = spv / sv
                var = spv2 / sv - mean * mean
                if var < 0.0:
                    var = 0.0
                vwap[t] = anchor + mean
                sd[t] = np.sqrt(var)
            if k >= slope_lb and np.isfinite(vwap[t - slope_lb]):
                slope[t] = vwap[t] - vwap[t - slope_lb]
    return vwap, sd, atr, slope, sess_bar


@dataclass
class Features:
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray
    sd: np.ndarray
    atr: np.ndarray
    slope: np.ndarray  # VWAP change over slope_lookback bars (points)
    sess_bar: np.ndarray
    cid: np.ndarray
    roll_guard: np.ndarray
    time: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.close)

    @property
    def in_session(self) -> np.ndarray:
        return self.time["in_session"]

    def upper(self, k: float) -> np.ndarray:
        return self.vwap + k * self.sd

    def lower(self, k: float) -> np.ndarray:
        return self.vwap - k * self.sd

    @property
    def slope_atr(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.slope / self.atr

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "close": self.close,
                "vwap": self.vwap,
                "sd": self.sd,
                "atr": self.atr,
                "slope": self.slope,
                "sess_bar": self.sess_bar,
                "roll_guard": self.roll_guard,
                **{k: v for k, v in self.time.items()},
            },
            index=self.index,
        )


def compute_features(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    guard: np.ndarray | None = None,
) -> Features:
    """Features for a canonical bar frame (see nqbot.data).

    ``guard`` overrides the roll guard (live passes its calendar-derived guard).
    """
    from nqbot.data import effective_roll_spread

    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    cid = contract_ids(df["contract"])
    spread = effective_roll_spread(df)
    time = session_arrays(
        df.index,
        cfg.session,
        flat_time=hhmm(cfg.flat_time),
        entry_end=hhmm(cfg.entry_end),
        trade_holidays=cfg.trade_holidays,
    )
    vwap, sd, atr, slope, sess_bar = _feature_kernel(
        o, h, low, c, v, cid, spread, time["session_id"], time["in_session"],
        int(cfg.atr_len), int(cfg.slope_lookback),
    )
    if guard is None:
        guard = roll_guard(cid)
    return Features(
        index=df.index, open=o, high=h, low=low, close=c, volume=v,
        vwap=vwap, sd=sd, atr=atr, slope=slope, sess_bar=sess_bar,
        cid=cid, roll_guard=np.asarray(guard, dtype=bool), time=time,
    )
