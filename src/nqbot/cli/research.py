from __future__ import annotations

import argparse
from pathlib import Path

from nqbot.cli.common import add_data_args, load_cfg, load_data
from nqbot.research import (
    HoldoutAlreadyUsed,
    develop,
    file_sha256,
    frame_sha256,
    holdout,
    load_selected,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research harness: develop (grid + walk-forward) then a one-shot holdout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("develop", help="grid + plateau + walk-forward on dev/validation (holdout untouched)")
    add_data_args(d)
    d.add_argument("--out", default=None)
    d.add_argument("--folds", type=int, default=4)
    d.add_argument("--min-trades", type=int, default=30)
    h = sub.add_parser("holdout", help="evaluate the frozen selection on the holdout, ONCE")
    add_data_args(h)
    h.add_argument("--out", default=None)
    h.add_argument("--selected", default=None, help="selected.yaml from develop (default: <out>/selected.yaml)")
    h.add_argument("--force-rerun-holdout", action="store_true", help="burns the holdout; flagged in the report")
    args = ap.parse_args(argv)

    df, synthetic = load_data(args)
    data_id = f"synthetic:{args.synthetic_days}d:seed{args.seed}:{frame_sha256(df)[:12]}" if synthetic else f"{args.data}:{file_sha256(args.data)[:16]}"
    out = Path(args.out or ("research_output/synthetic" if synthetic else "research_output/real"))
    if args.cmd == "develop":
        res = develop(df, load_cfg(args.config), out, data_id, wf_folds=args.folds, min_trades=args.min_trades, synthetic=synthetic)
        print(res.report)
        print(f"\nreport: {out / 'report_develop.md'}")
        return 0
    sel_path = Path(args.selected) if args.selected else out / "selected.yaml"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} not found; run `develop` first")
    try:
        print(holdout(df, load_selected(sel_path), out, data_id, synthetic=synthetic, force=args.force_rerun_holdout))
    except HoldoutAlreadyUsed as exc:
        print(f"REFUSED: {exc}")
        return 2
    return 0
