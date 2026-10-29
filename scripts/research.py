#!/usr/bin/env python3
"""Research harness.

    python scripts/research.py develop --config configs/default.yaml
    python scripts/research.py holdout          # one shot, after develop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqbot.cli.research import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
