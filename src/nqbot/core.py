"""Shared decision + execution kernel.

These numba functions are the ONLY implementation of position gating, bracket
fill rules and trade bookkeeping. The backtest loop calls them inside numba;
the live runner and SimBroker call the very same functions from Python.

State lives in a float64 vector ``st`` (indices below) so it can be mutated in
place from both worlds and persisted by the live bot.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from nqbot.config import CostConfig, StrategyConfig, hhmm
from nqbot.instruments import get_instrument

# ---- state vector -----------------------------------------------------------
S_POS = 0  # -1 / 0 / +1
S_QTY = 1
S_ENTRY_THEO = 2
S_ENTRY_FILL = 3
S_STOP = 4
S_TGT = 5
S_POS_SESSION = 6
S_SESSION = 7
S_TRADES_TODAY = 8
S_DAY_PNL = 9
S_COOLDOWN_UNTIL = 10  # utc minute; entries allowed when decision clock >= this
S_PEND = 11
S_PEND_DIR = 12
S_PEND_STOP = 13
S_PEND_TGT = 14
S_PEND_REASON = 15
S_ENTRY_MIN = 16
N_STATE = 17

# ---- pending actions ---------------------------------------------------------
PEND_NONE = 0
PEND_ENTRY = 1
PEND_FLATTEN = 2

# ---- exit reasons --------------------------------------------------------------
EXIT_NONE = 0
EXIT_STOP = 1
EXIT_TARGET = 2
EXIT_FLAT_TIME = 3
EXIT_ROLL = 4
EXIT_SESSION = 5
EXIT_END_OF_DATA = 6
EXIT_RISK = 7

EXIT_NAMES = {
    EXIT_STOP: "stop",
    EXIT_TARGET: "target",
    EXIT_FLAT_TIME: "flat_time",
    EXIT_ROLL: "roll",
    EXIT_SESSION: "session",
    EXIT_END_OF_DATA: "end_of_data",
    EXIT_RISK: "risk",
}

# ---- parameter vector ------------------------------------------------------------
P_WARMUP = 0
P_ENTRY_START = 1
P_MAX_TRADES = 2
P_MAX_DAILY_LOSS = 3
P_COOLDOWN = 4
P_QTY = 5
P_TICK = 6
P_POINT_VALUE = 7
P_COMMISSION = 8  # per contract per side
P_SLIP_TICKS = 9
P_THROUGH_TICKS = 10
N_PARAMS = 11


def pack_params(strategy: StrategyConfig, costs: CostConfig) -> np.ndarray:
    inst = get_instrument(costs.instrument)
    p = np.zeros(N_PARAMS, dtype=np.float64)
    p[P_WARMUP] = strategy.warmup_minutes
    p[P_ENTRY_START] = hhmm(strategy.entry_start)
    p[P_MAX_TRADES] = strategy.max_trades_per_day
    p[P_MAX_DAILY_LOSS] = strategy.max_daily_loss
    p[P_COOLDOWN] = strategy.cooldown_minutes
    p[P_QTY] = strategy.qty
    p[P_TICK] = inst.tick_size
    p[P_POINT_VALUE] = inst.point_value
    p[P_COMMISSION] = costs.commission
    p[P_SLIP_TICKS] = costs.slippage_ticks
    p[P_THROUGH_TICKS] = costs.target_through_ticks
    return p


@njit(cache=True)
def new_state() -> np.ndarray:
    st = np.zeros(N_STATE, dtype=np.float64)
    st[S_SESSION] = -1.0
    st[S_POS_SESSION] = -1.0
    st[S_STOP] = np.nan
    st[S_TGT] = np.nan
    st[S_COOLDOWN_UNTIL] = -1.0
    return st


@njit(cache=True)
def roll_session(st: np.ndarray, session_id: int) -> None:
    """Reset per-session counters when a new session starts."""
    if st[S_SESSION] != session_id:
        st[S_SESSION] = session_id
        st[S_TRADES_TODAY] = 0.0
        st[S_DAY_PNL] = 0.0


@njit(cache=True)
def market_fill(theo: float, side: int, p: np.ndarray) -> float:
    """Fill price of a market order (side +1 buy, -1 sell): theo moved against us."""
    return theo + side * p[P_SLIP_TICKS] * p[P_TICK]


@njit(cache=True)
def bracket_check(
    pos: int, stop: float, tgt: float, o: float, h: float, low: float, p: np.ndarray
) -> tuple[int, float, float]:
    """Check one bar against a working bracket. Returns (exit_code, theo, fill).

    Rules: gap through stop -> open; stop touched -> stop price; both stop and
    target touched -> stop (conservative). Stops are market orders and pay
    slippage. The limit target fills at its price, only if traded through by
    ``through`` ticks, and never with price improvement.
    """
    through = p[P_THROUGH_TICKS] * p[P_TICK]
    if pos > 0:
        if o <= stop:
            return EXIT_STOP, o, market_fill(o, -1, p)
        if low <= stop:
            return EXIT_STOP, stop, market_fill(stop, -1, p)
        if h >= tgt + through:
            return EXIT_TARGET, tgt, tgt
    elif pos < 0:
        if o >= stop:
            return EXIT_STOP, o, market_fill(o, 1, p)
        if h >= stop:
            return EXIT_STOP, stop, market_fill(stop, 1, p)
        if low <= tgt - through:
            return EXIT_TARGET, tgt, tgt
    return EXIT_NONE, np.nan, np.nan


@njit(cache=True)
def commission_per_fill(p: np.ndarray, qty: float) -> float:
    return p[P_COMMISSION] * qty


@njit(cache=True)
def on_entry(
    st: np.ndarray, direction: int, qty: float, theo: float, fill: float,
    stop: float, tgt: float, t_min: float,
) -> None:
    st[S_POS] = direction
    st[S_QTY] = qty
    st[S_ENTRY_THEO] = theo
    st[S_ENTRY_FILL] = fill
    st[S_STOP] = stop
    st[S_TGT] = tgt
    st[S_POS_SESSION] = st[S_SESSION]
    st[S_ENTRY_MIN] = t_min
    st[S_TRADES_TODAY] += 1.0


@njit(cache=True)
def on_exit(
    st: np.ndarray, theo: float, fill: float, reason: int, t_min: float, p: np.ndarray
) -> tuple[float, float, float, float]:
    """Close the position. Returns (gross, slippage, commission, net) for the trade.

    gross uses theoretical prices (open / stop / target / close), slippage is
    the market-order cost on both fills, commission covers both sides, and
    net = (exit_fill - entry_fill) * dir * qty * point_value - commission.
    """
    pos = st[S_POS]
    qty = st[S_QTY]
    pv = p[P_POINT_VALUE]
    gross = (theo - st[S_ENTRY_THEO]) * pos * qty * pv
    net_px = (fill - st[S_ENTRY_FILL]) * pos * qty * pv
    comm = 2.0 * commission_per_fill(p, qty)
    slip = gross - net_px
    net = net_px - comm
    st[S_DAY_PNL] += net
    if reason == EXIT_STOP:
        st[S_COOLDOWN_UNTIL] = t_min + 1.0 + p[P_COOLDOWN]
    st[S_POS] = 0.0
    st[S_QTY] = 0.0
    st[S_STOP] = np.nan
    st[S_TGT] = np.nan
    st[S_POS_SESSION] = -1.0
    return gross, slip, comm, net


@njit(cache=True)
def decide(
    st: np.ndarray,
    p: np.ndarray,
    label_min: float,
    et_mod: int,
    sess_el: int,
    session_id: int,
    entry_end_mod: int,
    flat_mod: int,
    guard: bool,
    sig_dir: int,
    sig_stop: float,
    sig_tgt: float,
) -> None:
    """Decide at the close of the bar labelled ``label_min``; sets the pending action.

    The decision clock is label + 1 minute (the bar's close).
    """
    st[S_PEND] = PEND_NONE
    st[S_PEND_REASON] = EXIT_NONE
    now = label_min + 1.0
    dmod = et_mod + 1
    el = sess_el + 1
    pos = st[S_POS]
    if pos != 0:
        reason = EXIT_NONE
        if guard:
            reason = EXIT_ROLL
        elif session_id != st[S_POS_SESSION]:
            reason = EXIT_SESSION
        elif dmod >= flat_mod or dmod < p[P_ENTRY_START]:
            reason = EXIT_FLAT_TIME
        if reason != EXIT_NONE:
            st[S_PEND] = PEND_FLATTEN
            st[S_PEND_REASON] = reason
        return
    if guard or sig_dir == 0:
        return
    if st[S_TRADES_TODAY] >= p[P_MAX_TRADES]:
        return
    if p[P_MAX_DAILY_LOSS] > 0 and st[S_DAY_PNL] <= -p[P_MAX_DAILY_LOSS]:
        return
    if now < st[S_COOLDOWN_UNTIL]:
        return
    if el < p[P_WARMUP]:
        return
    if dmod < p[P_ENTRY_START] or dmod > entry_end_mod:
        return
    if not (np.isfinite(sig_stop) and np.isfinite(sig_tgt)):
        return
    st[S_PEND] = PEND_ENTRY
    st[S_PEND_DIR] = sig_dir
    st[S_PEND_STOP] = sig_stop
    st[S_PEND_TGT] = sig_tgt
