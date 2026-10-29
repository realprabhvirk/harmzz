from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nqbot.config import AppConfig, load_config
from nqbot.data import load_bars, prepare_bars, synthetic_bars

DEFAULT_DATA = "data/nq_1m.parquet"


def add_data_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", default=DEFAULT_DATA, help="bar parquet (default: %(default)s)")
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    ap.add_argument(
        "--synthetic-days", type=int, default=0,
        help="use N days of SYNTHETIC random-walk data instead of --data (pipeline testing only)",
    )
    ap.add_argument("--seed", type=int, default=7)


def load_cfg(path: str | None) -> AppConfig:
    return load_config(path) if path else AppConfig().validate()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    """Returns (bars, is_synthetic)."""
    if args.synthetic_days > 0:
        return prepare_bars(synthetic_bars(days=args.synthetic_days, seed=args.seed)), True
    if not Path(args.data).exists():
        raise SystemExit(
            f"{args.data} not found. Run `python fetch_nq_data.py` first, "
            "or pass --synthetic-days N for a synthetic pipeline check."
        )
    return load_bars(args.data), False
