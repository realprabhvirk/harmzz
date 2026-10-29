# PLAN — NQ/MNQ VWAP trading system

## Context

The brief assumed three existing files (`backtester.py`, `live_bot.py`,
`fetch_nq_data.py`). They were **not in the repo**, so everything is written
from scratch. The step "pin the original behaviour with tests before
refactoring" has nothing to pin; instead the conventions in the brief are
encoded as tests from day one:

- signals are decided at bar close and filled at the **next bar's open**
- costs (commission + slippage) are paid on **every fill**
- positions are **flat before contract rolls**
- signals are computed on roll-adjusted prices; fills use raw prices

The three file names are kept as thin CLI entry points over the `nqbot` package.

## Stages (each ends with green tests + a commit)

1. **Scaffold**: `pyproject.toml`, config dataclasses + YAML, instrument specs,
   ET session/calendar utilities, data loading, synthetic data generator.
2. **Features + strategies**: session VWAP, volume-weighted sigma bands, ATR,
   VWAP slope (numba, strictly causal); `vwap_fade` and `vwap_pullback`;
   causality checker + deliberately leaky strategy.
3. **Engine**: bracket-order backtester (stop/target/time flatten/roll/EOD),
   cost breakdown, metrics, data-quality script, property tests, 5M-bar benchmark.
4. **Research harness**: 60/20/20 time split, coarse grids with neighbour
   (plateau) scoring, anchored walk-forward, cost sensitivity, breakdowns,
   trade-order bootstrap, one-shot holdout lock.
5. **Live**: broker interface, SimBroker replay, IBKR adapter (tested against a
   fake IB), runner, risk limits, state persistence, reconnect, alerts,
   paper/live gating; parity test (replay == backtest, to the tick).
6. **Docs**: README, RESULTS, CHANGELOG, final verification summary.

## Key design decisions

### One decision kernel, two callers
`nqbot.core` holds numba-jitted functions that mutate a small state vector:
`decide()` (gating + entry/flatten decision), `bracket_check()` (stop/target
fill rules), `on_entry()` / `on_exit()` (bookkeeping, P&L, cooldown). The
backtest loop calls them inside numba; the live runner calls the *same*
functions from Python once per bar. SimBroker uses the same fill functions.
There is no second implementation of the strategy for live.

### Strategy = pure vectorised signal + shared gating
- `compute_features()` → VWAP, bands, ATR, slope (finite memory, causal).
- `strategy.signals(features)` → per-bar `(dir, stop, target)` proposals.
- Gating that depends on position state (one position, max trades/day,
  cooldown, warm-up, entry window, flat time, daily loss) lives in `decide()`.

### Exact parity by construction
Features are computed with explicit loops whose floating-point order depends
only on the current session (VWAP) or a fixed window (ATR = SMA of true range).
So features from a 2000-bar live buffer are bit-identical to features from the
full history. No global back-adjustment is used for signals: the only
cross-roll quantity (previous close in true range) is shifted by the roll
spread, and VWAP sessions also reset on a contract change. That is exactly
equivalent to back-adjusting, but needs no knowledge of future rolls.

### Execution rules (backtest and SimBroker)
- entry: market at next open ± slippage
- the entry bar itself is checked against the bracket
- stop and target touched in the same bar → **stop first**
- gap through stop → fill at the open (worse than the stop) ± slippage
- stop = market order → pays slippage; limit target pays none and fills only if
  price trades through by `target_through_ticks` (default 1); a gap through the
  target fills at the target price (no price improvement — conservative)
- time flatten / roll flatten / session carry → market at next open ± slippage
- end of data → market at last close ± slippage
- zero-volume bars are dropped at ingestion (backtest and live alike)

### Time
All session logic in `America/New_York` via `zoneinfo`; bars are labelled by
their **start** minute (Databento convention), so the decision clock at bar `t`
is `label + 1 min`. "Flat by 15:55" = decided at the close of the 15:54 bar,
filled at the 15:55 open. An approximate CME calendar handles early closes;
US holidays are no-trade days by default.

### Honesty guards in research
Holdout segment is physically sliced off during selection and evaluated once;
a lock file keyed on the data hash refuses a second evaluation. Every config
tried is counted and reported.

## Not in scope / explicitly not done
Martingale, averaging down, pyramiding, leverage tricks, profit claims.
