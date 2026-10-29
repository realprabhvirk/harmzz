#!/usr/bin/env python3
"""Download NQ 1-minute bars from Databento into data/nq_1m.parquet.

    export DATABENTO_API_KEY=...        # never commit it
    python fetch_nq_data.py             # prints the cost estimate and asks first
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nqbot.fetch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
