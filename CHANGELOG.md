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
