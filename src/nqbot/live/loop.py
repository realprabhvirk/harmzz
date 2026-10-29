"""Real-time loop for paper/live trading (IB). Replay uses LiveRunner.run_replay."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from typing import Any

from nqbot.instruments import roll_date
from nqbot.live.feed import Bar
from nqbot.live.runner import LiveRunner
from nqbot.sessions import ET


def calendar_roll_guard(contract_month: str) -> Callable[[Bar], bool]:
    """True from the CME roll date of the traded contract on: no new entries, flatten."""
    year, month = int(contract_month[:4]), int(contract_month[4:6])
    rd = roll_date(year, month)

    def guard(bar: Bar) -> bool:
        day = bar.ts.tz_convert(ET).date()
        return day >= rd

    return guard


def run_live(
    runner: LiveRunner,
    feed: Any,
    pump: Callable[[float], None],
    now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    guard_fn: Callable[[Bar], bool] | None = None,
    poll_s: float = 0.5,
    grace_s: float = 5.0,
    max_iterations: int | None = None,
) -> None:
    """Event loop. ``pump`` must service the IB event loop (``ib.sleep``)."""
    runner.start()
    runner.seed(feed.history())
    next_open: float | None = None
    it = 0
    while max_iterations is None or it < max_iterations:
        it += 1
        if not runner.ensure_connected():
            break
        now = now_fn()
        for bar in feed.poll(now):
            if guard_fn is not None:
                bar.roll_guard = guard_fn(bar)
            runner.on_bar_close(bar)
            runner.on_bar_open(bar.label_min + 1)
            next_open = bar.label_min + 2
        cur = math.floor(now.timestamp() / 60)
        if next_open is not None and cur >= next_open and now.timestamp() - cur * 60 >= grace_s:
            runner.on_bar_open(cur)  # no bar arrived: still enforce the flat deadline on the clock
            next_open = cur + 1
        runner.process_broker()
        if not runner.halted:
            runner.ensure_protected()
            runner.check_entry_timeout()
            runner.check_stale(now.timestamp())
            if runner.risk.kill_switch():
                runner.halt("kill file present")
        pump(poll_s)
