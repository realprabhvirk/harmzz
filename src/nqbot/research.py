"""Research harness, honest by design.

- Time-ordered split by ET date: development (~60%), walk-forward validation
  (~20%), holdout (~20%).
- ``develop`` never loads holdout bars: they are sliced off before any
  backtest runs. Coarse grids, neighbour/plateau scoring, anchored
  walk-forward over the validation folds, cost sensitivity, breakdowns,
  trade-order bootstrap. Every configuration tried is counted and logged.
- ``holdout`` evaluates the frozen selection exactly once; a lock file keyed on
  the data hash refuses a second evaluation.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nqbot.config import AppConfig, config_from_dict, config_to_dict
from nqbot.engine import run_signals
from nqbot.features import Features, compute_features
from nqbot.instruments import get_instrument
from nqbot.metrics import (
    bootstrap_drawdown,
    by_hour,
    by_side,
    by_vol_regime,
    by_year,
    max_drawdown,
)
from nqbot.sessions import ET
from nqbot.strategies import get_strategy

DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "vwap_fade": {
        "band_k": [1.5, 2.0, 2.5, 3.0],
        "fade_stop_atr": [1.0, 2.0, 3.0],
        "fade_trend_filter": [True, False],
    },
    "vwap_pullback": {
        "pb_min_slope_atr": [0.2, 0.5],
        "pb_touch_atr": [0.25, 0.5],
        "pb_stop_atr": [0.5, 1.0, 1.5],
        "pb_target_r": [1.5, 2.0],
    },
}
COST_SLIPPAGE = [0.5, 1.0, 2.0, 3.0]
COST_COMMISSION_MULT = [1.0, 1.5, 2.0]
WARMUP_DAYS = 3
SYNTH_BANNER = (
    "> **SYNTHETIC DATA — pipeline check only. These numbers are NOT strategy "
    "results and say nothing about NQ.**"
)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    dev: tuple[pd.Timestamp, pd.Timestamp]
    val: tuple[pd.Timestamp, pd.Timestamp]
    holdout: tuple[pd.Timestamp, pd.Timestamp]

    def as_dict(self) -> dict[str, list[str]]:
        return {k: [str(a), str(b)] for k, (a, b) in (("dev", self.dev), ("val", self.val), ("holdout", self.holdout))}


def split_by_date(df: pd.DataFrame, fracs: tuple[float, float] = (0.6, 0.2)) -> Split:
    """Split on ET calendar dates (midnight ET boundaries; no position spans midnight)."""
    dates = np.unique(np.asarray(df.index.tz_convert(ET).date))
    if len(dates) < 10:
        raise ValueError("need at least 10 trading dates to split")
    i1 = int(round(len(dates) * fracs[0]))
    i2 = int(round(len(dates) * (fracs[0] + fracs[1])))

    def ts(d: dt.date) -> pd.Timestamp:
        return pd.Timestamp(d, tz=ET).tz_convert("UTC")

    end = ts(dates[-1] + dt.timedelta(days=1))
    return Split((ts(dates[0]), ts(dates[i1])), (ts(dates[i1]), ts(dates[i2])), (ts(dates[i2]), end))


def slice_with_warmup(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, warmup_days: int = WARMUP_DAYS) -> pd.DataFrame:
    """Bars in [start - warmup, end). Warm-up uses only earlier (past) data."""
    lo = start - pd.Timedelta(days=warmup_days)
    return df[(df.index >= lo) & (df.index < end)]


def in_window(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if len(trades) == 0:
        return trades
    m = (trades["entry_time"] >= start) & (trades["entry_time"] < end)
    return trades[m]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def trade_stats(trades: pd.DataFrame, n_days: int | None = None) -> dict[str, float]:
    n = len(trades)
    net = trades["net"].to_numpy() if n else np.zeros(0)
    wins, losses = net[net > 0], net[net < 0]
    out = {
        "trades": float(n),
        "net": float(net.sum()),
        "gross": float(trades["gross"].sum()) if n else 0.0,
        "costs": float((trades["slippage"] + trades["commission"]).sum()) if n else 0.0,
        "avg_trade": float(net.mean()) if n else 0.0,
        "win_rate": float((net > 0).mean()) if n else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("nan"),
        "max_dd": max_drawdown(np.concatenate(([0.0], np.cumsum(net)))),
    }
    if n and n_days:
        daily = trades.groupby(trades["exit_time"].dt.tz_convert(ET).dt.date)["net"].sum()
        vals = np.concatenate([daily.to_numpy(), np.zeros(max(0, n_days - len(daily)))])
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        out["sharpe_ann"] = float(vals.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    else:
        out["sharpe_ann"] = float("nan")
    return out


def n_trading_days(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    part = df[(df.index >= start) & (df.index < end)]
    return int(len(np.unique(np.asarray(part.index.tz_convert(ET).date))))


# ---------------------------------------------------------------------------
# grid + plateau
# ---------------------------------------------------------------------------


@dataclass
class GridRun:
    variant: str
    dims: dict[str, list[Any]]
    keys: list[tuple[int, ...]]
    params: list[dict[str, Any]]
    trades: list[pd.DataFrame]


def _feature_cache_key(cfg: AppConfig) -> tuple[Any, ...]:
    s = cfg.strategy
    return (s.session, s.atr_len, s.slope_lookback, s.flat_time, s.entry_end, s.trade_holidays)


def run_config(df: pd.DataFrame, cfg: AppConfig, feat: Features | None = None) -> pd.DataFrame:
    if feat is None:
        feat = compute_features(df, cfg.strategy)
    tick = get_instrument(cfg.costs.instrument).tick_size
    sig = get_strategy(cfg.strategy).signals(feat, cfg.strategy, tick)
    return run_signals(df, feat, sig, cfg.strategy, cfg.costs).trades


def with_params(base: AppConfig, variant: str, params: dict[str, Any]) -> AppConfig:
    cfg = copy.deepcopy(base)
    cfg.strategy.variant = variant  # type: ignore[assignment]
    for k, v in params.items():
        setattr(cfg.strategy, k, v)
    cfg.validate()
    return cfg


def run_grid(df: pd.DataFrame, base: AppConfig, variant: str, dims: dict[str, list[Any]]) -> GridRun:
    """Run every grid point once over ``df`` (which must exclude holdout bars)."""
    names = list(dims)
    keys = list(itertools.product(*[range(len(dims[k])) for k in names]))
    feats: dict[tuple[Any, ...], Features] = {}
    params, trades = [], []
    for key in keys:
        p = {k: dims[k][i] for k, i in zip(names, key)}
        cfg = with_params(base, variant, p)
        ck = _feature_cache_key(cfg)
        if ck not in feats:
            feats[ck] = compute_features(df, cfg.strategy)
        params.append(p)
        trades.append(run_config(df, cfg, feats[ck]))
    return GridRun(variant, dims, keys, params, trades)


def neighbours(key: tuple[int, ...], dims: dict[str, list[Any]]) -> list[tuple[int, ...]]:
    sizes = [len(v) for v in dims.values()]
    out = []
    for d, size in enumerate(sizes):
        for step in (-1, 1):
            j = key[d] + step
            if 0 <= j < size:
                out.append(key[:d] + (j,) + key[d + 1 :])
    return out


def score_grid(
    gr: GridRun, start: pd.Timestamp, end: pd.Timestamp, min_trades: int = 30
) -> pd.DataFrame:
    """Per-config net P&L in [start, end) plus neighbour statistics and plateau score.

    plateau = mean net over the config and its one-step neighbours; configs
    with fewer than ``min_trades`` trades are not eligible.
    """
    rows = []
    metric: dict[tuple[int, ...], float] = {}
    counts: dict[tuple[int, ...], int] = {}
    for key, tr in zip(gr.keys, gr.trades):
        w = in_window(tr, start, end)
        metric[key] = float(w["net"].sum()) if len(w) else 0.0
        counts[key] = len(w)
    for key, p in zip(gr.keys, gr.params):
        nb = neighbours(key, gr.dims)
        nb_vals = [metric[k] for k in nb]
        eligible = counts[key] >= min_trades
        plateau = float(np.mean([metric[key], *nb_vals])) if eligible else float("nan")
        rows.append(
            {
                **p,
                "trades": counts[key],
                "net": metric[key],
                "nb_median": float(np.median(nb_vals)) if nb_vals else float("nan"),
                "nb_min": float(np.min(nb_vals)) if nb_vals else float("nan"),
                "plateau": plateau,
                "key": key,
            }
        )
    return pd.DataFrame(rows)


def pick(table: pd.DataFrame) -> pd.Series | None:
    elig = table[table["plateau"].notna()]
    if len(elig) == 0:
        return None
    return elig.sort_values(["plateau", "net"], ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


@dataclass
class WalkForward:
    folds: list[dict[str, Any]] = field(default_factory=list)
    oos_trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def fold_bounds(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, k: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    part = df[(df.index >= start) & (df.index < end)]
    dates = np.unique(np.asarray(part.index.tz_convert(ET).date))
    chunks = np.array_split(dates, k)
    out = []
    for i, ch in enumerate(chunks):
        if len(ch) == 0:
            continue
        a = pd.Timestamp(ch[0], tz=ET).tz_convert("UTC")
        b = end if i == len(chunks) - 1 else pd.Timestamp(chunks[i + 1][0], tz=ET).tz_convert("UTC")
        out.append((max(a, start), b))
    return out


def walk_forward(
    gr: GridRun, dev_start: pd.Timestamp, folds: list[tuple[pd.Timestamp, pd.Timestamp]], min_trades: int
) -> WalkForward:
    """Anchored: params for fold i are chosen on [dev_start, fold_i_start) only."""
    wf = WalkForward()
    oos = []
    for a, b in folds:
        table = score_grid(gr, dev_start, a, min_trades)
        best = pick(table)
        if best is None:
            wf.folds.append({"start": str(a), "end": str(b), "params": None, "oos": None})
            continue
        idx = gr.keys.index(best["key"])
        tr = in_window(gr.trades[idx], a, b)
        oos.append(tr)
        wf.folds.append(
            {
                "start": str(a),
                "end": str(b),
                "params": gr.params[idx],
                "in_sample_net": float(best["net"]),
                "plateau": float(best["plateau"]),
                "oos": trade_stats(tr),
            }
        )
    wf.oos_trades = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    return wf


# ---------------------------------------------------------------------------
# cost sensitivity
# ---------------------------------------------------------------------------


def cost_sensitivity(
    df: pd.DataFrame, cfg: AppConfig, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    part = slice_with_warmup(df, start, end)
    feat = compute_features(part, cfg.strategy)
    days = n_trading_days(df, start, end)
    rows = []
    for slip in COST_SLIPPAGE:
        for mult in COST_COMMISSION_MULT:
            c = copy.deepcopy(cfg)
            c.costs.slippage_ticks = slip
            c.costs.commission_mult = mult
            tr = in_window(run_config(part, c, feat), start, end)
            s = trade_stats(tr, days)
            rows.append({"slippage_ticks": slip, "commission_x": mult, **{k: s[k] for k in ("trades", "net", "avg_trade", "profit_factor", "max_dd", "sharpe_ann")}})
    return pd.DataFrame(rows)


def wf_cost_sensitivity(
    df: pd.DataFrame, base: AppConfig, variant: str, wf: WalkForward, n_days: int
) -> pd.DataFrame:
    """Cost grid on the walk-forward OOS path (each fold uses the params chosen before it)."""
    rows = []
    for slip in COST_SLIPPAGE:
        for mult in COST_COMMISSION_MULT:
            parts = []
            for f in wf.folds:
                if f["params"] is None:
                    continue
                a, b = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
                c = with_params(base, variant, f["params"])
                c.costs.slippage_ticks = slip
                c.costs.commission_mult = mult
                parts.append(in_window(run_config(slice_with_warmup(df, a, b), c), a, b))
            tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            s = trade_stats(tr, n_days) if len(tr) else trade_stats(_empty_trades(), None)
            rows.append({"slippage_ticks": slip, "commission_x": mult, **{k: s[k] for k in ("trades", "net", "avg_trade", "profit_factor", "max_dd", "sharpe_ann")}})
    return pd.DataFrame(rows)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in ("net", "gross", "slippage", "commission")})


# ---------------------------------------------------------------------------
# bookkeeping: trial log + holdout lock
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def log_trials(out: Path, data_id: str, n: int, note: str) -> int:
    """Append to the trial log; returns the cumulative count for this dataset."""
    out.mkdir(parents=True, exist_ok=True)
    p = out / "trials_log.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": dt.datetime.now(dt.timezone.utc).isoformat(), "data": data_id, "configs": n, "note": note}) + "\n")
    total = 0
    for line in p.read_text().splitlines():
        rec = json.loads(line)
        if rec["data"] == data_id:
            total += int(rec["configs"])
    return total


class HoldoutAlreadyUsed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------


def _md(df: pd.DataFrame, floatfmt: int = 2) -> str:
    if df is None or len(df) == 0:
        return "_(none)_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    cols = [str(c) for c in d.columns]
    idx_name = d.index.name or ""
    show_index = not isinstance(d.index, pd.RangeIndex)
    head = ([idx_name] if show_index else []) + cols
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in d.iterrows():
        vals = ([str(i)] if show_index else []) + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _stats_line(s: dict[str, float]) -> str:
    return (
        f"trades {int(s['trades'])}, net {s['net']:.2f}, gross {s['gross']:.2f}, costs {s['costs']:.2f}, "
        f"avg {s['avg_trade']:.2f}, win {s['win_rate']:.1%}, PF {s['profit_factor']:.2f}, "
        f"maxDD {s['max_dd']:.2f}, Sharpe(ann, daily) {s['sharpe_ann']:.2f}"
    )


def breakdown_section(tr: pd.DataFrame, bars: pd.DataFrame) -> str:
    if len(tr) == 0:
        return "_(no trades)_\n"
    bs = bootstrap_drawdown(tr["net"].to_numpy(), 2000, seed=20240601)
    parts = [
        "**By year**\n\n" + _md(by_year(tr)),
        "**By volatility regime** (terciles of trailing 20-day mean daily range, causal)\n\n" + _md(by_vol_regime(tr, bars)),
        "**By entry hour (ET)**\n\n" + _md(by_hour(tr)),
        "**Long vs short**\n\n" + _md(by_side(tr)),
        "**Trade-order bootstrap of max drawdown** (2000 permutations)\n\n"
        + _md(pd.DataFrame([bs])),
    ]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# develop / holdout
# ---------------------------------------------------------------------------


@dataclass
class DevelopResult:
    split: Split
    selected: AppConfig | None
    selected_variant: str | None
    wf: dict[str, WalkForward]
    tables: dict[str, pd.DataFrame]
    total_configs: int
    cumulative_configs: int
    verdict_profitable: bool
    report: str


def develop(
    df: pd.DataFrame,
    base: AppConfig,
    out_dir: str | Path,
    data_id: str,
    grids: dict[str, dict[str, list[Any]]] | None = None,
    wf_folds: int = 4,
    min_trades: int = 30,
    synthetic: bool = False,
) -> DevelopResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grids = grids or DEFAULT_GRIDS
    split = split_by_date(df)
    pre = df[df.index < split.holdout[0]]  # holdout bars physically removed
    assert pre.index.max() < split.holdout[0]
    folds = fold_bounds(pre, split.val[0], split.val[1], wf_folds)
    val_days = n_trading_days(pre, split.val[0], split.val[1])

    n_configs = sum(int(np.prod([len(v) for v in g.values()])) for g in grids.values())
    cumulative = log_trials(out, data_id, n_configs, "develop grid")

    wfs: dict[str, WalkForward] = {}
    tables: dict[str, pd.DataFrame] = {}
    final_tables: dict[str, pd.DataFrame] = {}
    sections: list[str] = []
    for variant, dims in grids.items():
        gr = run_grid(pre, base, variant, dims)
        tables[variant] = score_grid(gr, split.dev[0], split.dev[1], min_trades)
        wf = walk_forward(gr, split.dev[0], folds, min_trades)
        wfs[variant] = wf
        final_tables[variant] = score_grid(gr, split.dev[0], split.val[1], min_trades)
        sections.append(_variant_section(variant, tables[variant], wf, val_days, pre))

    # choose the variant by walk-forward OOS net (validation is for this), then
    # its parameters by plateau on all pre-holdout data
    wf_net = {v: (float(w.oos_trades["net"].sum()) if len(w.oos_trades) else float("-inf")) for v, w in wfs.items()}
    best_variant = max(wf_net, key=lambda v: wf_net[v]) if wf_net else None
    selected = None
    if best_variant is not None:
        best = pick(final_tables[best_variant])
        if best is not None:
            dims = grids[best_variant]
            params = {k: dims[k][i] for k, i in zip(dims, best["key"])}
            selected = with_params(base, best_variant, params)
    profitable = best_variant is not None and wf_net[best_variant] > 0

    cost_md = "_(no walk-forward path)_"
    if best_variant is not None and len(wfs[best_variant].oos_trades):
        cost_md = _md(wf_cost_sensitivity(pre, base, best_variant, wfs[best_variant], val_days))
    if selected is not None:
        with open(out / "selected.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config_to_dict(selected), fh, sort_keys=False)

    lines = ["# Research report — development + walk-forward validation", ""]
    if synthetic:
        lines += [SYNTH_BANNER, ""]
    lines += [
        f"- data: `{data_id}`",
        f"- split (UTC): dev {split.dev[0]} → {split.dev[1]}, validation {split.val[0]} → {split.val[1]}, "
        f"holdout {split.holdout[0]} → {split.holdout[1]} (**not loaded in this run**)",
        f"- configurations tried in this run: **{n_configs}**; cumulative on this dataset: **{cumulative}**",
        f"- walk-forward: anchored, {len(folds)} folds over validation; min trades for eligibility: {min_trades}",
        f"- costs: {base.costs.instrument}, slippage {base.costs.slippage_ticks} tick/market fill, "
        f"commission {base.costs.commission:.2f}/side",
        "",
        "## Verdict",
        "",
    ]
    if profitable:
        lines.append(
            f"Walk-forward OOS net after costs is positive for `{best_variant}` "
            f"({wf_net[best_variant]:.2f}). This is ONE validation path; check cost sensitivity and the "
            "holdout before believing it."
        )
    else:
        lines.append(
            "**Nothing is net-profitable out-of-sample after realistic costs.** "
            f"Walk-forward OOS net by variant: {', '.join(f'{k}: {v:.2f}' for k, v in wf_net.items())}. "
            "Per the research rules, no further variants or cost loosening should be tried."
        )
    lines += ["", f"Selected for holdout: `{best_variant}` {selected and {k: getattr(selected.strategy, k) for k in grids[best_variant]}}", ""]
    lines += [
        f"## Cost sensitivity (`{best_variant}`, walk-forward OOS path)",
        "",
        "Each fold uses the parameters chosen on data before it, so this is out-of-sample.",
        "",
        cost_md,
        "",
    ]
    lines += sections
    report = "\n".join(lines)
    (out / "report_develop.md").write_text(report, encoding="utf-8")
    return DevelopResult(split, selected, best_variant, wfs, tables, n_configs, cumulative, profitable, report)


def _variant_section(variant: str, table: pd.DataFrame, wf: WalkForward, val_days: int, bars: pd.DataFrame) -> str:
    show = table.drop(columns=["key"]).sort_values("plateau", ascending=False, na_position="last")
    lines = [
        f"## {variant}",
        "",
        "### Development grid (net after costs, with one-step neighbours)",
        "",
        _md(show),
        "",
        "### Walk-forward folds",
        "",
    ]
    for f in wf.folds:
        if f["params"] is None:
            lines.append(f"- {f['start']} → {f['end']}: no eligible config")
        else:
            lines.append(f"- {f['start']} → {f['end']}: params {f['params']} → OOS {_stats_line(f['oos'])}")
    lines += ["", f"**Stitched OOS:** {_stats_line(trade_stats(wf.oos_trades, val_days)) if len(wf.oos_trades) else 'no trades'}", ""]
    lines += ["### OOS breakdowns", "", breakdown_section(wf.oos_trades, bars)]
    return "\n".join(lines)


def holdout(
    df: pd.DataFrame,
    selected: AppConfig,
    out_dir: str | Path,
    data_id: str,
    synthetic: bool = False,
    force: bool = False,
) -> str:
    """Evaluate the frozen selection on the holdout. Refuses to run twice per dataset."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "holdout_lock.json"
    prior: list[dict[str, Any]] = json.loads(lock.read_text()) if lock.exists() else []
    used = [r for r in prior if r["data"] == data_id]
    if used and not force:
        raise HoldoutAlreadyUsed(
            f"holdout for this dataset was already evaluated at {used[0]['time']}; "
            f"result is in {out / 'report_holdout.md'}. Re-running would make it not a holdout."
        )
    split = split_by_date(df)
    a, b = split.holdout
    part = slice_with_warmup(df, a, b)
    days = n_trading_days(df, a, b)
    tr = in_window(run_config(part, selected), a, b)
    stats = trade_stats(tr, days)
    cs = cost_sensitivity(df, selected, a, b)
    rec = {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data": data_id,
        "config": config_to_dict(selected),
        "stats": stats,
        "forced_rerun": bool(used),
    }
    lock.write_text(json.dumps(prior + [rec], indent=2, default=str))
    lines = ["# Holdout evaluation (one shot)", ""]
    if synthetic:
        lines += [SYNTH_BANNER, ""]
    if used:
        lines += [
            f"> **WARNING: holdout re-evaluated (run #{len(used) + 1}). It is no longer a clean holdout.**",
            "",
        ]
    lines += [
        f"- data: `{data_id}`; holdout {a} → {b}; {days} trading days",
        f"- config: `{selected.strategy.variant}` " + json.dumps(config_to_dict(selected)["strategy"]),
        "",
        "## Result (reported whatever it is)",
        "",
        _stats_line(stats),
        "",
        "**Verdict:** "
        + ("net-profitable on holdout after base costs." if stats["net"] > 0 else "NOT net-profitable on holdout after base costs."),
        "",
        "## Cost sensitivity",
        "",
        _md(cs),
        "",
        "## Breakdowns",
        "",
        breakdown_section(tr, part),
    ]
    report = "\n".join(lines)
    (out / "report_holdout.md").write_text(report, encoding="utf-8")
    return report


def load_selected(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))
