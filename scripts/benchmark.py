#!/usr/bin/env python3
"""Engine speed on ~15 years of synthetic 1-minute bars (~5.2M bars)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqbot.config import AppConfig  # noqa: E402
from nqbot.data import prepare_bars, synthetic_bars  # noqa: E402
from nqbot.engine import run_backtest  # noqa: E402


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 15 * 252
    t0 = time.perf_counter()
    df = prepare_bars(synthetic_bars(days=days, seed=1))
    t1 = time.perf_counter()
    print(f"generated {len(df):,} bars in {t1 - t0:.1f}s")
    for variant in ("vwap_fade", "vwap_pullback"):
        cfg = AppConfig()
        cfg.strategy.variant = variant
        run_backtest(df.iloc[:5000], cfg)  # JIT warm-up (cached afterwards)
        t2 = time.perf_counter()
        res = run_backtest(df, cfg)
        t3 = time.perf_counter()
        print(f"{variant}: {len(df):,} bars, {len(res.trades):,} trades in {t3 - t2:.1f}s")


if __name__ == "__main__":
    main()
