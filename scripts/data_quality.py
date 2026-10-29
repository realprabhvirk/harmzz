#!/usr/bin/env python3
"""Data quality report: python scripts/data_quality.py --data data/nq_1m.parquet"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqbot.cli.quality import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
