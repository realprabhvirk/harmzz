"""Bracket-order backtester (numba). Uses the shared kernel in nqbot.core.

Per bar t:
  1. a pending action decided at t-1 fills at open[t] (entry or flatten)
  2. an open position (including one entered at this open) is checked against
     the bracket using high/low[t]
  3. on the last bar any open position is closed at close[t] (end_of_data)
  4. equity is marked at close[t]
  5. the decision for t+1 is made at the close of t
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from nqbot.config import AppConfig, CostConfig, StrategyConfig
from nqbot.core import (
    EXIT_END_OF_DATA,
    EXIT_NAMES,
    EXIT_NONE,
    EXIT_TARGET,
    P_POINT_VALUE,
    P_QTY,
    PEND_ENTRY,
    PEND_FLATTEN,
    S_ENTRY_FILL,
    S_PEND,
    S_PEND_DIR,
    S_PEND_REASON,
    S_PEND_STOP,
    S_PEND_TGT,
    S_POS,
    S_QTY,
    S_STOP,
    S_TGT,
    bracket_check,
    commission_per_fill,
    decide,
    market_fill,
    new_state,
    on_entry,
    on_exit,
    pack_params,
    roll_session,
)
from nqbot.features import Features, compute_features
from nqbot.strategies import Signals, get_strategy
from nqbot.instruments import get_instrument

# trade record columns
T_ENTRY_IDX, T_EXIT_IDX, T_DIR, T_QTY = 0, 1, 2, 3
T_ENTRY_THEO, T_ENTRY_FILL, T_EXIT_THEO, T_EXIT_FILL = 4, 5, 6, 7
T_STOP, T_TGT, T_REASON = 8, 9, 10
T_GROSS, T_SLIP, T_COMM, T_NET = 11, 12, 13, 14
N_TCOLS = 15


@njit(cache=True)
def _close_trade(
    rec: np.ndarray, k: int, st: np.ndarray, t: int, theo: float, fill: float,
    reason: int, t_min: float, p: np.ndarray,
) -> float:
    """Record and close; returns the cash change of the exit fill."""
    pos = st[S_POS]
    qty = st[S_QTY]
    entry_fill = st[S_ENTRY_FILL]
    rec[k, T_EXIT_IDX] = t
    rec[k, T_EXIT_THEO] = theo
    rec[k, T_EXIT_FILL] = fill
    rec[k, T_REASON] = reason
    gross, slip, comm, net = on_exit(st, theo, fill, reason, t_min, p)
    rec[k, T_GROSS] = gross
    rec[k, T_SLIP] = slip
    rec[k, T_COMM] = comm
    rec[k, T_NET] = net
    return (fill - entry_fill) * pos * qty * p[P_POINT_VALUE] - commission_per_fill(p, qty)


@njit(cache=True)
def _run_kernel(
    o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    utc_min: np.ndarray, et_mod: np.ndarray, sess_el: np.ndarray,
    session_id: np.ndarray, entry_end_mod: np.ndarray, flat_mod: np.ndarray,
    guard: np.ndarray, sig_dir: np.ndarray, sig_stop: np.ndarray, sig_tgt: np.ndarray,
    p: np.ndarray, initial: float, capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = len(o)
    st = new_state()
    rec = np.full((capacity, N_TCOLS), np.nan)
    equity = np.empty(n)
    position = np.zeros(n, dtype=np.int8)
    cash = initial
    k = 0  # number of trades opened
    pv = p[P_POINT_VALUE]
    for t in range(n):
        roll_session(st, session_id[t])
        t_min = float(utc_min[t])
        pend = st[S_PEND]
        if pend == PEND_ENTRY:
            d = int(st[S_PEND_DIR])
            qty = p[P_QTY]
            fill = market_fill(o[t], d, p)
            if k >= capacity:
                raise RuntimeError("trade capacity exceeded")
            rec[k, T_ENTRY_IDX] = t
            rec[k, T_DIR] = d
            rec[k, T_QTY] = qty
            rec[k, T_ENTRY_THEO] = o[t]
            rec[k, T_ENTRY_FILL] = fill
            rec[k, T_STOP] = st[S_PEND_STOP]
            rec[k, T_TGT] = st[S_PEND_TGT]
            on_entry(st, d, qty, o[t], fill, st[S_PEND_STOP], st[S_PEND_TGT], t_min)
            cash -= commission_per_fill(p, qty)
            k += 1
        elif pend == PEND_FLATTEN and st[S_POS] != 0:
            side = -int(st[S_POS])
            fill = market_fill(o[t], side, p)
            cash += _close_trade(rec, k - 1, st, t, o[t], fill, int(st[S_PEND_REASON]), t_min, p)
        st[S_PEND] = 0.0

        if st[S_POS] != 0:
            code, theo, fill = bracket_check(int(st[S_POS]), st[S_STOP], st[S_TGT], o[t], h[t], low[t], p)
            if code != EXIT_NONE:
                cash += _close_trade(rec, k - 1, st, t, theo, fill, code, t_min, p)

        if t == n - 1 and st[S_POS] != 0:
            side = -int(st[S_POS])
            cash += _close_trade(rec, k - 1, st, t, c[t], market_fill(c[t], side, p), EXIT_END_OF_DATA, t_min, p)

        position[t] = int(st[S_POS] * st[S_QTY])
        unreal = 0.0
        if st[S_POS] != 0:
            unreal = (c[t] - st[S_ENTRY_FILL]) * st[S_POS] * st[S_QTY] * pv
        equity[t] = cash + unreal

        if t < n - 1:
            decide(
                st, p, t_min, et_mod[t], sess_el[t], session_id[t], entry_end_mod[t],
                flat_mod[t], guard[t], sig_dir[t], sig_stop[t], sig_tgt[t],
            )
    return rec[:k], equity, position, k


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    position: pd.Series
    initial_capital: float
    point_value: float
    commission_per_fill: float
    slippage_per_fill: float  # currency per market fill per contract
    summary: dict[str, float] = field(default_factory=dict)


def trades_frame(rec: np.ndarray, index: pd.DatetimeIndex, contract: np.ndarray | None = None) -> pd.DataFrame:
    cols = [
        "entry_idx", "exit_idx", "dir", "qty", "entry_theo", "entry_fill", "exit_theo",
        "exit_fill", "stop", "target", "reason", "gross", "slippage", "commission", "net",
    ]
    df = pd.DataFrame(rec, columns=cols)
    for col in ("entry_idx", "exit_idx", "dir", "qty", "reason"):
        df[col] = df[col].astype(np.int64)
    df["entry_time"] = index[df["entry_idx"].to_numpy()] if len(df) else pd.DatetimeIndex([], tz="UTC")
    df["exit_time"] = index[df["exit_idx"].to_numpy()] if len(df) else pd.DatetimeIndex([], tz="UTC")
    df["exit_reason"] = df["reason"].map(EXIT_NAMES)
    df["exit_kind"] = np.where(df["reason"] == EXIT_TARGET, "limit", "market")
    if contract is not None and len(df):
        df["contract"] = contract[df["entry_idx"].to_numpy()]
    return df


def run_signals(
    df: pd.DataFrame,
    feat: Features,
    sig: Signals,
    strategy: StrategyConfig,
    costs: CostConfig,
) -> BacktestResult:
    """Run the kernel on precomputed features + signals."""
    p = pack_params(strategy, costs)
    t = feat.time
    n_sessions = len(np.unique(t["session_id"])) if len(feat) else 0
    capacity = n_sessions * max(strategy.max_trades_per_day, 1) + 16
    rec, equity, position, _ = _run_kernel(
        feat.open, feat.high, feat.low, feat.close,
        t["utc_min"], t["et_mod"], t["sess_el"], t["session_id"],
        t["entry_end_mod"], t["flat_mod"], feat.roll_guard,
        sig.dir.astype(np.int64), sig.stop, sig.target,
        p, float(costs.initial_capital), int(capacity),
    )
    inst = get_instrument(costs.instrument)
    trades = trades_frame(rec, feat.index, df["contract"].to_numpy())
    res = BacktestResult(
        trades=trades,
        equity=pd.Series(equity, index=feat.index, name="equity"),
        position=pd.Series(position, index=feat.index, name="position"),
        initial_capital=float(costs.initial_capital),
        point_value=inst.point_value,
        commission_per_fill=costs.commission * strategy.qty,
        slippage_per_fill=costs.slippage_ticks * inst.tick_size * inst.point_value * strategy.qty,
    )
    from nqbot.metrics import summarize

    res.summary = summarize(res)
    return res


def run_backtest(df: pd.DataFrame, cfg: AppConfig) -> BacktestResult:
    """Full pipeline: features -> strategy signals -> bracket engine."""
    strat = get_strategy(cfg.strategy)
    tick = get_instrument(cfg.costs.instrument).tick_size
    feat = compute_features(df, cfg.strategy)
    sig = strat.signals(feat, cfg.strategy, tick)
    return run_signals(df, feat, sig, cfg.strategy, cfg.costs)
