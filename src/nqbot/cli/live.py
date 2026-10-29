from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd

from nqbot.cli.common import add_data_args, load_cfg, load_data
from nqbot.config import AppConfig
from nqbot.core import pack_params
from nqbot.engine import run_backtest
from nqbot.instruments import front_contract, ib_contract_month
from nqbot.live.alerts import Alerter
from nqbot.live.broker import SimBroker
from nqbot.live.feed import bars_from_frame
from nqbot.live.runner import LiveRunner
from nqbot.live.state import StateStore

LIVE_ENV = "NQBOT_ENABLE_LIVE"
LIVE_ENV_VALUE = "I_ACCEPT_REAL_MONEY_RISK"


class GateError(SystemExit):
    pass


def confirmation_phrase(account: str, cfg: AppConfig) -> str:
    return f"LIVE {account} {cfg.costs.instrument} {cfg.strategy.qty}"


def check_mode_gate(
    mode: str,
    live_flag: bool,
    account: str,
    cfg: AppConfig,
    env: dict[str, str],
    input_fn: Callable[[str], str],
) -> None:
    """Paper is the default. Live needs the env var AND --live AND a typed confirmation."""
    if mode == "paper":
        if live_flag:
            raise GateError("--live given but mode is paper; use `live` mode")
        if not account.startswith("DU"):
            raise GateError(f"paper mode refuses account {account!r}: IB paper accounts start with 'DU'")
        return
    if mode != "live":
        raise GateError(f"unknown mode {mode}")
    if env.get(LIVE_ENV) != LIVE_ENV_VALUE:
        raise GateError(f"live trading needs {LIVE_ENV}={LIVE_ENV_VALUE} in the environment")
    if not live_flag:
        raise GateError("live trading needs the --live flag")
    phrase = confirmation_phrase(account, cfg)
    typed = input_fn(f"Type exactly '{phrase}' to trade REAL money: ").strip()
    if typed != phrase:
        raise GateError("confirmation did not match; not trading")


def _import_ib() -> tuple[Any, Any]:
    try:
        import ib_async as mod  # type: ignore[import-not-found]
    except ImportError:
        try:
            import ib_insync as mod  # type: ignore[import-not-found,no-redef]
        except ImportError as exc:
            raise SystemExit("install ib_async (pip install ib_async) for paper/live mode") from exc
    return mod.IB, mod


def replay(args: argparse.Namespace, cfg: AppConfig) -> int:
    df, synthetic = load_data(args)
    store = StateStore(args.state_dir)
    if store.load() and not args.resume:
        raise SystemExit(f"{args.state_dir} already has state; use a fresh --state-dir or --resume")
    runner = LiveRunner(cfg, SimBroker(pack_params(cfg.strategy, cfg.costs)), store, Alerter(cfg.live.alert_kind))
    runner.start()
    trades = pd.DataFrame(runner.run_replay(bars_from_frame(df)))
    bt = run_backtest(df, cfg).trades
    net = float(trades["net"].sum()) if len(trades) else 0.0
    print(("*** SYNTHETIC DATA ***\n" if synthetic else "") + f"replay trades {len(trades)}, net {net:.2f}")
    same = len(trades) == len(bt) and (
        len(bt) == 0 or (trades["net"].to_numpy() == bt["net"].to_numpy()).all()
    )
    print(f"backtester trades {len(bt)}, net {bt['net'].sum():.2f} -> parity {'OK' if same else 'MISMATCH'}")
    return 0 if same else 1


def realtime(args: argparse.Namespace, cfg: AppConfig) -> int:  # pragma: no cover - needs TWS/Gateway
    from nqbot.live.feed import IBBarFeed
    from nqbot.live.ib_broker import IBBroker
    from nqbot.live.loop import calendar_roll_guard, run_live

    if args.mode == "live" and cfg.live.port == 7497:
        cfg.live.port = 7496
    month = cfg.live.contract_month or ib_contract_month(*front_contract(dt.date.today()))
    IB, api = _import_ib()
    ib = IB()
    broker = IBBroker(ib, api, cfg.live, cfg.costs.instrument, month)
    broker.connect()
    accounts = broker.accounts()
    account = cfg.live.account or (accounts[0] if accounts else "")
    check_mode_gate(args.mode, args.live, account, cfg, dict(os.environ), input)
    store = StateStore(cfg.live.state_dir)
    runner = LiveRunner(cfg, broker, store, Alerter(cfg.live.alert_kind))
    if args.clear_halt:
        runner.restore()
        runner.clear_halt()
    feed = IBBarFeed(ib, broker.contract, f"{cfg.costs.instrument}{month}")
    print(f"{args.mode.upper()} trading {cfg.costs.instrument} {month} account {account}; KILL file: {cfg.risk.kill_file}")
    run_live(runner, feed, ib.sleep, guard_fn=calendar_roll_guard(month))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live runner: replay (SimBroker), paper (default) or live (IBKR)")
    ap.add_argument("mode", nargs="?", default="paper", choices=["replay", "paper", "live"])
    add_data_args(ap)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="replay: continue from existing state")
    ap.add_argument("--live", action="store_true", help="required (with env var + confirmation) for live mode")
    ap.add_argument("--clear-halt", action="store_true", help="clear a persisted halt after investigating it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    cfg = load_cfg(args.config)
    if args.state_dir:
        cfg.live.state_dir = args.state_dir
    args.state_dir = cfg.live.state_dir if args.mode != "replay" or args.state_dir else "state/replay"
    if args.mode == "replay":
        return replay(args, cfg)
    return realtime(args, cfg)
