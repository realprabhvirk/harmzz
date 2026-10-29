from __future__ import annotations

import argparse
import json

import pandas as pd

from nqbot.cli.common import DEFAULT_DATA
from nqbot.data_quality import format_report, quality_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Data quality report for 1-minute bars")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="also write the markdown report here")
    args = ap.parse_args(argv)
    rep = quality_report(pd.read_parquet(args.data))
    text = json.dumps(rep, indent=2, default=str) if args.json else format_report(rep)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(format_report(rep))
    return 0 if rep.get("ok") else 1
