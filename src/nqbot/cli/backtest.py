from __future__ import annotations

import argparse
import json

from nqbot.cli.common import add_data_args, load_cfg, load_data
from nqbot.engine import run_backtest
from nqbot.metrics import by_exit_reason, by_side, by_year

SYNTH_BANNER = "*** SYNTHETIC DATA: pipeline check only, NOT a strategy result ***"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the bracket backtester")
    add_data_args(ap)
    ap.add_argument("--trades-csv", default=None, help="write trades to this CSV")
    ap.add_argument("--json", action="store_true", help="print summary as JSON")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    df, synthetic = load_data(args)
    res = run_backtest(df, cfg)
    s = res.summary
    if args.json:
        print(json.dumps({"synthetic": synthetic, **s}, indent=2))
    else:
        if synthetic:
            print(SYNTH_BANNER)
        print(f"strategy={cfg.strategy.variant} instrument={cfg.costs.instrument} bars={len(df)}")
        print(f"trades={int(s['trades'])} (long {int(s['long_trades'])} / short {int(s['short_trades'])})")
        print(f"gross       {s['gross']:>12.2f}")
        print(f"slippage    {-s['slippage']:>12.2f}")
        print(f"commission  {-s['commission']:>12.2f}")
        print(f"net         {s['net']:>12.2f}")
        print(f"max DD {s['max_drawdown']:.2f}  win rate {s['win_rate']:.1%}  PF {s['profit_factor']:.2f}")
        if len(res.trades):
            print("\nby year:\n" + by_year(res.trades).round(2).to_string())
            print("\nby side:\n" + by_side(res.trades).round(2).to_string())
            print("\nby exit:\n" + by_exit_reason(res.trades).round(2).to_string())
        if synthetic:
            print(SYNTH_BANNER)
    if args.trades_csv:
        res.trades.to_csv(args.trades_csv, index=False)
    return 0
