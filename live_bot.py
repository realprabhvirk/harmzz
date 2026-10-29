#!/usr/bin/env python3
"""Live runner entry point.

    python live_bot.py replay --synthetic-days 20 --state-dir /tmp/s   # replay + parity check
    python live_bot.py paper --config configs/default.yaml           # IB paper (default)
    NQBOT_ENABLE_LIVE=I_ACCEPT_REAL_MONEY_RISK python live_bot.py live --live --config ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nqbot.cli.live import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
