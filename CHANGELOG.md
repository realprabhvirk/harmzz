# Changelog

## Unreleased

### Context
- The brief referenced existing `backtester.py`, `live_bot.py` and
  `fetch_nq_data.py`. They were not in the repository, so there was no prior
  behaviour to pin. Everything was written from scratch following the brief's
  conventions; those conventions are pinned by tests instead.

### Stage 1 — scaffold
- Package `src/nqbot`, typed YAML config with strict key/type validation.
- NQ/MNQ specs, quarterly contract and roll calendar (8 days before expiry).
- ET session arrays (DST via zoneinfo), approximate CME holiday/half-day calendar.
- Canonical bar schema, loader, additive back-adjustment, roll guard,
  deterministic synthetic generator (pipeline testing only).

### Bugs found and fixed (with regression tests)
- `session_arrays` returned no in-session bars under pandas 3 because
  `DatetimeIndex.asi8` is no longer always nanoseconds; now forced to ns
  (`tests/test_sessions.py::test_rth_open_is_0930_et_across_dst`).
- Config loader accepted a YAML list for a section (`strategy: []`) as empty;
  now rejected (`tests/test_config.py::test_bad_config_rejected`).

### Stage 2 — features and strategies
- Numba feature kernel: session VWAP (resets on session start and contract
  change), volume-weighted sigma bands, ATR (SMA of roll-adjusted true range),
  VWAP slope. Finite-memory, bit-identical on truncated buffers.
- `vwap_fade` and `vwap_pullback`, pure vectorised signal functions.
- Causality checker (signal-level and engine-level); a deliberately leaky
  strategy is shown to fail it.

### Stage 3 — engine
- Shared numba kernel (`nqbot.core`): gating/decision, bracket fill rules,
  trade bookkeeping. Backtest loop in `nqbot.engine`.
- Stop/target simulation: next-open entries, entry bar checked, stop-first on
  ambiguous bars, gap-through-stop at the open, slippage on market fills only,
  target needs a 1-tick trade-through, time/roll/session flatten at next open,
  end-of-data flatten at last close.
- Gross / slippage / commission / net breakdown, breakdowns by year, hour,
  side, exit reason, volatility regime; trade-order bootstrap of drawdown.
- Data-quality script (gaps, duplicates, bad ticks, coverage, rolls, tz/DST).
- Benchmark: 5.2M synthetic bars backtest in ~4-7 s after JIT warm-up.

### Stage 4 — research harness
- 60/20/20 split by ET date; holdout bars physically removed during `develop`.
- Coarse grids (24 + 24 configs), neighbour/plateau scoring, anchored
  walk-forward over 4 validation folds, cumulative trial log per dataset.
- Cost sensitivity (slippage 0.5-3 ticks x commission x1/x1.5/x2) on the
  walk-forward OOS path and on the holdout.
- Breakdowns by year, volatility regime, entry hour, side; trade-order bootstrap.
- One-shot holdout with a lock file; `--force-rerun-holdout` is flagged in the report.

### Bugs found and fixed (with regression tests)
- Property test found that a missing bar right before the flat deadline (e.g.
  zero-volume 15:54 bar) let a position survive into the 15:55 bar. The kernel
  now also checks at each bar's open and flattens positions past the deadline
  or carried into a new session (`test_missing_bar_before_flat_time_regression`,
  `test_session_gap_closes_at_first_open_of_new_session`). Entry orders whose
  next bar starts > `entry_max_delay_minutes` after the decision are cancelled
  (`test_entry_cancelled_after_data_gap`).
- First draft of the research report ran cost sensitivity with the config
  selected on dev+validation, evaluated on validation (in-sample). It now runs
  on the walk-forward OOS path.

### Stage 5 — live bot
- `LiveRunner` drives the same `nqbot.core` kernel and strategy signal code as
  the backtester (no separate live strategy implementation).
- `SimBroker` replay uses the backtester's fill functions; replay == backtest
  to the tick (parity tests incl. stops, targets, rolls, gaps, zero-volume bars,
  qty > 1, globex session, strategy daily-loss gating, crash/restart mid-position).
- `IBBroker` (ib_async/ib_insync API): market-parent bracket with GTC OCO
  stop/target, stop acknowledgement required (else cancel + `OrderRejected`),
  fill polling from executions, own-fill position tracking with delayed-report
  tolerance, partial-fill resize, flatten with confirmation. Tested against a
  fake IB only.
- "Never without a resting stop": invariant enforced every bar/tick; protective
  stop placement failure -> flatten + halt.
- Startup/reconnect reconciliation, exponential-backoff reconnect, stale-data
  levels, KILL file, flat window, daily loss, max position, max orders, reject
  limit; state persisted atomically (save on change), halts persist until cleared.
- Paper by default; live needs env var + `--live` + typed confirmation; paper
  refuses non-`DU` accounts. Optional Slack/Telegram alerts from env vars.

### Bugs found and fixed
- Replay persisted state every bar because NaNs in the state vector made the
  change detection always fire (and `decide` stamped a timestamp every bar);
  both fixed, replay is ~8x faster.
