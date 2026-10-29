"""Performance statistics, breakdowns and the trade-order bootstrap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nqbot.sessions import ET

if TYPE_CHECKING:
    from nqbot.engine import BacktestResult


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline (positive number, currency)."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def summarize(res: BacktestResult) -> dict[str, float]:
    tr = res.trades
    eq = res.equity
    n = len(tr)
    net = tr["net"].to_numpy() if n else np.zeros(0)
    wins = net[net > 0]
    losses = net[net < 0]
    daily = eq.groupby(eq.index.tz_convert(ET).date).last() if len(eq) else pd.Series(dtype=float)
    daily_ret = daily.diff().dropna()
    sd = float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else 0.0
    return {
        "trades": float(n),
        "long_trades": float((tr["dir"] > 0).sum()) if n else 0.0,
        "short_trades": float((tr["dir"] < 0).sum()) if n else 0.0,
        "gross": float(tr["gross"].sum()) if n else 0.0,
        "slippage": float(tr["slippage"].sum()) if n else 0.0,
        "commission": float(tr["commission"].sum()) if n else 0.0,
        "net": float(net.sum()),
        "avg_trade": float(net.mean()) if n else 0.0,
        "win_rate": float(len(wins) / n) if n else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan"),
        "max_drawdown": max_drawdown(eq.to_numpy()),
        "daily_sharpe_ann": float(daily_ret.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
        "days": float(len(daily)),
        "final_equity": float(eq.iloc[-1]) if len(eq) else res.initial_capital,
    }


def _group_stats(tr: pd.DataFrame, key: pd.Series | np.ndarray) -> pd.DataFrame:
    if len(tr) == 0:
        return pd.DataFrame(columns=["trades", "net", "avg", "win_rate", "gross", "costs"])
    g = tr.groupby(key)
    out = pd.DataFrame(
        {
            "trades": g["net"].count(),
            "net": g["net"].sum(),
            "avg": g["net"].mean(),
            "win_rate": g["net"].apply(lambda s: float((s > 0).mean())),
            "gross": g["gross"].sum(),
            "costs": g["slippage"].sum() + g["commission"].sum(),
        }
    )
    return out


def by_year(tr: pd.DataFrame) -> pd.DataFrame:
    return _group_stats(tr, tr["entry_time"].dt.tz_convert(ET).dt.year if len(tr) else [])


def by_hour(tr: pd.DataFrame) -> pd.DataFrame:
    return _group_stats(tr, tr["entry_time"].dt.tz_convert(ET).dt.hour if len(tr) else [])


def by_side(tr: pd.DataFrame) -> pd.DataFrame:
    return _group_stats(tr, np.where(tr["dir"] > 0, "long", "short") if len(tr) else [])


def by_exit_reason(tr: pd.DataFrame) -> pd.DataFrame:
    return _group_stats(tr, tr["exit_reason"] if len(tr) else [])


def daily_range_regime(bars: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Causal volatility measure per ET date: trailing mean of prior days' high-low range.

    The value for date D uses only days before D.
    """
    d = bars.index.tz_convert(ET).date
    rng = (bars["high"].groupby(d).max() - bars["low"].groupby(d).min()).astype(float)
    return rng.shift(1).rolling(lookback, min_periods=5).mean()


def by_vol_regime(tr: pd.DataFrame, bars: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Trades bucketed into terciles of the causal trailing daily-range measure."""
    if len(tr) == 0:
        return _group_stats(tr, [])
    reg = daily_range_regime(bars, lookback)
    dates = tr["entry_time"].dt.tz_convert(ET).dt.date
    val = dates.map(reg)
    valid = val.notna()
    labels = pd.Series("unknown", index=tr.index, dtype=object)
    if valid.sum() >= 3:
        q = pd.qcut(val[valid].astype(float), 3, labels=["low", "mid", "high"], duplicates="drop")
        labels[valid] = q.astype(str)
    return _group_stats(tr, labels)


def bootstrap_drawdown(
    trade_pnl: np.ndarray, n_samples: int = 2000, seed: int = 12345
) -> dict[str, float]:
    """Distribution of max drawdown under random reorderings of the trade sequence."""
    pnl = np.asarray(trade_pnl, dtype=float)
    if len(pnl) == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "observed": 0.0}
    rng = np.random.default_rng(seed)
    dds = np.empty(n_samples)
    for i in range(n_samples):
        path = np.concatenate(([0.0], np.cumsum(rng.permutation(pnl))))
        dds[i] = max_drawdown(path)
    observed = max_drawdown(np.concatenate(([0.0], np.cumsum(pnl))))
    return {
        "p50": float(np.percentile(dds, 50)),
        "p90": float(np.percentile(dds, 90)),
        "p95": float(np.percentile(dds, 95)),
        "p99": float(np.percentile(dds, 99)),
        "observed": observed,
    }
