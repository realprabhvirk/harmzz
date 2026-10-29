"""Databento downloader -> data/nq_1m.parquet (canonical schema, see nqbot.data).

Downloads front (``NQ.v.0``) and second (``NQ.v.1``) volume-rolled continuous
1-minute OHLCV from CME Globex (GLBX.MDP3). Prices are raw contract prices.
The roll spread (new - old contract price at the same minute, just before the
roll) comes from v.1, which before the roll IS the incoming contract.

Needs ``pip install databento`` and DATABENTO_API_KEY in the environment.
Always prints the cost estimate first and asks before downloading (or --yes).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
DATASET_START = "2010-06-06"


def build_canonical(
    v0: pd.DataFrame,
    v1: pd.DataFrame | None,
    id_to_symbol: dict[int, str] | None = None,
    max_lookback: str = "2h",
) -> pd.DataFrame:
    """Combine front/second continuous frames into the canonical bar frame.

    Both inputs: tz-aware index (bar start), columns open/high/low/close/volume
    and instrument_id. Returns open/high/low/close/volume/contract/roll_spread.
    """
    id_to_symbol = id_to_symbol or {}
    f = v0.sort_index()
    f = f[~f.index.duplicated(keep="first")]
    ids = f["instrument_id"].to_numpy()
    out = pd.DataFrame(
        {k: f[k].astype(np.float64).to_numpy() for k in ("open", "high", "low", "close", "volume")},
        index=f.index.tz_convert("UTC"),
    )
    out.index.name = "ts"
    out["contract"] = [id_to_symbol.get(int(i), f"id{int(i)}") for i in ids]
    out["roll_spread"] = np.nan
    if v1 is None or len(f) < 2:
        return out
    s = v1.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    roll_pos = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    lookback = pd.Timedelta(max_lookback)
    for i in roll_pos:
        new_id, t_old = ids[i], f.index[i - 1]
        # last minute <= t_old where v.1 is the incoming contract and both have a bar
        cand = s[(s.index <= t_old) & (s.index >= t_old - lookback) & (s["instrument_id"] == new_id)]
        common = cand.index.intersection(f.index[(f.index <= t_old) & (f["instrument_id"].to_numpy() == ids[i - 1])])
        if len(common):
            t = common.max()
            out.iloc[i, out.columns.get_loc("roll_spread")] = float(s.loc[t, "close"]) - float(f.loc[t, "close"])
    return out


def _client() -> Any:  # pragma: no cover - network
    try:
        import databento as db  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("pip install databento") from exc
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("set DATABENTO_API_KEY (never commit it)")
    return db.Historical(key)


def _symbol_map(client: Any, ids: list[int], start: str, end: str) -> dict[int, str]:  # pragma: no cover - network
    try:
        res = client.symbology.resolve(
            dataset=DATASET, symbols=[str(i) for i in ids], stype_in="instrument_id",
            stype_out="raw_symbol", start_date=start[:10], end_date=end[:10],
        )
        out = {}
        for k, intervals in res.get("result", {}).items():
            if intervals:
                out[int(k)] = intervals[-1]["s"]
        return out
    except Exception as exc:  # noqa: BLE001 - symbol names are cosmetic
        print(f"warning: could not resolve raw symbols ({exc}); using instrument ids", file=sys.stderr)
        return {}


def fetch(
    root: str, start: str, end: str, out: Path, yes: bool, input_fn: Callable[[str], str] = input
) -> Path:  # pragma: no cover - network
    client = _client()
    syms = [f"{root}.v.0", f"{root}.v.1"]
    cost = client.metadata.get_cost(dataset=DATASET, symbols=syms, stype_in="continuous", schema=SCHEMA, start=start, end=end)
    print(f"Databento cost estimate for {syms} {start} -> {end}: ${cost:.2f}")
    if not yes and input_fn("Download? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")
    frames = {}
    for sym in syms:
        parts = []
        for y0 in range(int(start[:4]), int(end[:4]) + 1):
            a = max(start, f"{y0}-01-01")
            b = min(end, f"{y0 + 1}-01-01")
            if a >= b:
                continue
            print(f"  {sym} {a} -> {b}")
            data = client.timeseries.get_range(dataset=DATASET, symbols=[sym], stype_in="continuous", schema=SCHEMA, start=a, end=b)
            parts.append(data.to_df()[["instrument_id", "open", "high", "low", "close", "volume"]])
        frames[sym] = pd.concat(parts).sort_index()
    ids = sorted(set(frames[syms[0]]["instrument_id"].astype(int)))
    canon = build_canonical(frames[syms[0]], frames[syms[1]], _symbol_map(client, ids, start, end))
    out.parent.mkdir(parents=True, exist_ok=True)
    canon.to_parquet(out)
    print(f"wrote {len(canon):,} bars to {out}; next: python scripts/data_quality.py --data {out}")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(description="Download NQ 1-minute bars from Databento")
    ap.add_argument("--root", default="NQ", help="NQ (longer history) or MNQ (from 2019)")
    ap.add_argument("--start", default=DATASET_START)
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="data/nq_1m.parquet")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args(argv)
    fetch(args.root, args.start, args.end, Path(args.out), args.yes)
    return 0
