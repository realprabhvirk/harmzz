"""Bar data: loading, preparation, roll handling and a deterministic synthetic generator.

Canonical bar frame:
    index        tz-aware UTC DatetimeIndex, bar START time, unique, sorted
    open/high/low/close   float64 raw prices of the contract traded at that time
    volume       float64 (> 0 after preparation)
    contract     str, e.g. 'NQZ5'
    roll_spread  float64, on the first bar of a new contract: new - old price at
                 the roll instant (NaN elsewhere / when unknown)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nqbot.instruments import contract_code, front_contract
from nqbot.sessions import ET

PRICE_COLS = ["open", "high", "low", "close"]
REQUIRED = PRICE_COLS + ["volume"]


def prepare_bars(df: pd.DataFrame, drop_zero_volume: bool = True) -> pd.DataFrame:
    """Normalise a raw bar frame to the canonical schema (no silent price repairs)."""
    out = df.copy()
    if "ts" in out.columns:
        out = out.set_index("ts")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("bars need a DatetimeIndex (or a 'ts' column)")
    if out.index.tz is None:
        raise ValueError("bar timestamps must be timezone-aware; refusing to guess")
    out.index = out.index.tz_convert("UTC")
    out.index.name = "ts"
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = out.sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="first")]
    for col in REQUIRED:
        out[col] = out[col].astype(np.float64)
    if "contract" not in out.columns:
        out["contract"] = "UNKNOWN"
    out["contract"] = out["contract"].astype(str)
    if "roll_spread" not in out.columns:
        out["roll_spread"] = np.nan
    out["roll_spread"] = out["roll_spread"].astype(np.float64)
    if drop_zero_volume:
        out = out[out["volume"] > 0]
    return out[REQUIRED + ["contract", "roll_spread"]]


def load_bars(path: str | Path, drop_zero_volume: bool = True) -> pd.DataFrame:
    return prepare_bars(pd.read_parquet(path), drop_zero_volume=drop_zero_volume)


def contract_ids(contract: pd.Series | np.ndarray) -> np.ndarray:
    """Integer id that increments whenever the contract changes."""
    arr = np.asarray(contract)
    if len(arr) == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.empty(len(arr), dtype=bool)
    change[0] = False
    change[1:] = arr[1:] != arr[:-1]
    return np.cumsum(change).astype(np.int64)


def roll_guard(cids: np.ndarray) -> np.ndarray:
    """True at bar t if bar t+1 or t+2 belongs to a different contract.

    Roll dates are calendar knowledge (fixed exchange schedule), not price
    information, so using them ahead of time is not lookahead. A decision at t
    under the guard is filled at t+1, still in the outgoing contract.
    """
    n = len(cids)
    g = np.zeros(n, dtype=bool)
    if n > 1:
        g[:-1] |= cids[1:] != cids[:-1]
    if n > 2:
        g[:-2] |= cids[2:] != cids[:-2]
    return g


def effective_roll_spread(df: pd.DataFrame) -> np.ndarray:
    """Roll spread per bar (0 except on the first bar of a new contract).

    Falls back to open[t] - close[t-1] when the true spread is unknown, which
    wrongly attributes the whole gap to the roll; data_quality flags this.
    """
    cids = contract_ids(df["contract"])
    spread = np.zeros(len(df))
    if len(df) < 2:
        return spread
    roll_idx = np.flatnonzero(cids[1:] != cids[:-1]) + 1
    rs = df["roll_spread"].to_numpy()
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    for i in roll_idx:
        spread[i] = rs[i] if np.isfinite(rs[i]) else o[i] - c[i - 1]
    return spread


def back_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Additive back-adjustment: earlier contracts shifted so the latest prices are raw."""
    spread = effective_roll_spread(df)
    offset = np.zeros(len(df))
    if len(df) > 1:
        offset[:-1] = np.cumsum(spread[::-1])[::-1][1:]
    out = df.copy()
    for col in PRICE_COLS:
        out[col] = out[col].to_numpy() + offset
    out["adj_offset"] = offset
    return out


# ---------------------------------------------------------------------------
# Synthetic data (pipeline testing ONLY; never a strategy result)
# ---------------------------------------------------------------------------


def _intraday_vol_shape(et_mod: np.ndarray) -> np.ndarray:
    shape = np.full(len(et_mod), 0.45)
    rth = (et_mod >= 570) & (et_mod < 960)
    shape[rth] = 1.0
    shape[(et_mod >= 570) & (et_mod < 600)] = 2.2
    shape[(et_mod >= 930) & (et_mod < 960)] = 1.5
    return shape


def synthetic_bars(
    start: str = "2024-01-02",
    days: int = 20,
    seed: int = 7,
    mode: str = "globex",
    start_price: float = 17_000.0,
    daily_vol: float = 0.012,
    gap_vol: float = 0.003,
    roll_spread_mean: float = 60.0,
    roll_spread_sd: float = 10.0,
    symbol: str = "NQ",
    zero_volume_frac: float = 0.0,
    tick: float = 0.25,
) -> pd.DataFrame:
    """Deterministic random-walk 1-minute NQ-like bars with quarterly rolls.

    The walk has no edge by construction. ``mode`` 'globex' gives 18:00-17:00 ET
    sessions (1380 bars), 'rth' gives 09:30-16:00 ET (390 bars).
    """
    rng = np.random.default_rng(seed)
    session_days = pd.bdate_range(start, periods=days)
    if mode == "globex":
        first_min, n_per = -360, 1380
    elif mode == "rth":
        first_min, n_per = 570, 390
    else:
        raise ValueError(mode)
    base = session_days.values.astype("datetime64[m]").astype(np.int64)
    offs = np.arange(n_per, dtype=np.int64) + first_min
    wall = (base[:, None] + offs[None, :]).ravel()
    naive = pd.DatetimeIndex(wall.astype("datetime64[m]"))
    idx = naive.tz_localize(ET, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    n = len(idx)
    et_mod = wall % 1440
    sess_of_bar = np.repeat(np.arange(days), n_per)
    first_of_session = np.zeros(n, dtype=bool)
    first_of_session[::n_per] = True

    shape = _intraday_vol_shape(et_mod)
    shape = shape / np.sqrt((shape[:n_per] ** 2).sum())
    sigma = daily_vol * shape
    r = rng.standard_normal(n) * sigma
    gap = np.where(first_of_session, rng.standard_normal(n) * gap_vol, 0.0)
    gap[0] = 0.0
    log_x = np.log(start_price) + np.cumsum(r + gap)
    x = np.exp(log_x)
    x_prev_gapped = np.empty(n)
    x_prev_gapped[0] = start_price
    x_prev_gapped[1:] = x[:-1] * np.exp(gap[1:])

    # contracts by session date + roll basis
    labels = []
    for d in session_days:
        y, m = front_contract(d.date())
        labels.append(contract_code(symbol, y, m))
    labels_arr = np.array(labels)[sess_of_bar]
    cids = contract_ids(labels_arr)
    n_contracts = int(cids[-1]) + 1 if n else 0
    spreads = np.round(rng.normal(roll_spread_mean, roll_spread_sd, n_contracts) / tick) * tick
    spreads[0] = 0.0
    basis = np.cumsum(spreads)[cids]

    def rt(a: np.ndarray) -> np.ndarray:
        return np.round(a / tick) * tick

    close = rt(x + basis)
    open_ = rt(x_prev_gapped + basis)
    ext = np.abs(rng.standard_normal((2, n))) * sigma * x * 0.6
    high = np.maximum(np.maximum(open_, close), np.ceil((np.maximum(open_, close) + ext[0]) / tick) * tick)
    low = np.minimum(np.minimum(open_, close), np.floor((np.minimum(open_, close) - ext[1]) / tick) * tick)

    vol_shape = _intraday_vol_shape(et_mod)
    volume = np.maximum(1, np.round(400 * vol_shape * rng.lognormal(0.0, 0.5, n)))
    if zero_volume_frac > 0:
        volume[rng.random(n) < zero_volume_frac] = 0

    roll_spread = np.full(n, np.nan)
    roll_bars = np.flatnonzero(np.diff(cids) != 0) + 1
    roll_spread[roll_bars] = spreads[cids[roll_bars]]
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(np.float64),
            "contract": labels_arr,
            "roll_spread": roll_spread,
        },
        index=pd.DatetimeIndex(idx, name="ts"),
    )
    return df
