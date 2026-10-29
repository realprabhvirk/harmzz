#!/usr/bin/env python3
"""Backtest entry point. Example:

    python backtester.py --config configs/default.yaml
    python backtester.py --synthetic-days 60      # pipeline check, not a result
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nqbot.cli.backtest import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
