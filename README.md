# nqbot — NQ/MNQ VWAP research, backtesting and live trading

Two session-VWAP strategies (`vwap_fade`, `vwap_pullback`) for Nasdaq-100
futures, a bracket-order backtester, an honest research harness and a live
runner for Interactive Brokers that runs **the same strategy code** as the
backtester.

> No profit claims. As of this commit **no real market data has been run
> through the research harness** (see [RESULTS.md](RESULTS.md)). Everything
> verified so far is mechanics, not edge.

## Layout

```
backtester.py  live_bot.py  fetch_nq_data.py      thin CLI entry points
scripts/research.py  scripts/data_quality.py  scripts/benchmark.py
configs/default.yaml (vwap_fade)  configs/pullback.yaml
src/nqbot/
  config.py        typed YAML config (strict)
  instruments.py   NQ/MNQ specs, contract months, roll dates
  sessions.py      ET session math (DST-safe), approximate CME calendar
  data.py          canonical bars, back-adjustment, roll guard, synthetic data
  data_quality.py  gaps, duplicates, bad ticks, coverage, rolls, tz/DST
  features.py      session VWAP, sigma bands, ATR, VWAP slope (numba, causal)
  strategies/      vwap_fade, vwap_pullback (pure signal functions)
  core.py          THE shared decision + fill kernel (backtest and live)
  engine.py        backtest loop
  metrics.py       stats, breakdowns, trade-order bootstrap
  causality.py     truncation-based look-ahead checks
  research.py      split / grid / plateau / walk-forward / costs / holdout
  live/            runner, SimBroker, IB adapter, feed, risk, state, alerts
tests/             pytest + hypothesis, fake IB, no network
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # numpy pandas numba pyarrow pyyaml + pytest/hypothesis
pip install -e ".[data]"             # databento (for fetch_nq_data.py)
pip install -e ".[live]"             # ib_async (for paper/live)
pytest -q                            # ~25 s, no network
NUMBA_DISABLE_JIT=1 pytest -q --cov=nqbot   # coverage (pure Python, ~6 min)
```

## Data

```bash
export DATABENTO_API_KEY=...          # environment only; never commit secrets
python fetch_nq_data.py               # prints the Databento cost and asks first
python scripts/data_quality.py        # exit code 1 if anything needs a look
```

`fetch_nq_data.py` downloads `NQ.v.0` and `NQ.v.1` (volume-rolled continuous)
1-minute OHLCV from GLBX.MDP3 since 2010 and writes `data/nq_1m.parquet` with
raw contract prices, the contract per bar and the roll spread (from v.1).
Use `--root MNQ` for MNQ (history from 2019). NQ and MNQ track the same index;
research on NQ history and trade MNQ by setting `costs.instrument: MNQ`.

Until that file exists every command accepts `--synthetic-days N`, a
deterministic random walk with no edge. It exists to test plumbing; output is
banner-labelled SYNTHETIC and must never be read as a strategy result.

## Backtest

```bash
python backtester.py --config configs/default.yaml --trades-csv trades.csv
python backtester.py --config configs/pullback.yaml
python backtester.py --synthetic-days 60          # plumbing check only
```

Prints gross, slippage, commission, net, and breakdowns by year, side and exit reason.

Execution model (identical in backtest and live replay):

| event | fill |
|---|---|
| entry | market, next bar open ± slippage |
| stop | market: stop price ± slippage; gap through stop → open ± slippage |
| stop and target touched in the same bar | stop (conservative) |
| target | limit at the target, only if traded through by `target_through_ticks` (1); no price improvement |
| time flatten (15:55), roll, session carry | market, next open ± slippage |
| end of data | market, last close ± slippage |
| commission | every fill |

Signals are decided at the bar close (bars are labelled by start time, so the
decision clock is label + 1 min). The entry bar itself is checked against
the bracket. Zero-volume bars are dropped. Positions are flat before rolls
(roll dates are calendar knowledge). Benchmark: 5.2M bars (15 years of
synthetic minutes) backtest in 4–7 s after JIT warm-up (`python scripts/benchmark.py`).

## Strategies (all parameters in `StrategyConfig`, YAML-loadable)

Session VWAP = Σ(typical·volume)/Σ(volume), typical = (H+L+C)/3. It resets at 09:30 ET
(`session: rth`) or 18:00 ET (`session: globex`) and on contract changes, with
volume-weighted ±k·σ bands.

- **vwap_fade**: close below VWAP − kσ → long (mirror short). Optional trend
  filter skips fades against a VWAP slope steeper than `fade_trend_max_slope_atr`.
  Target = VWAP at the signal bar, stop = `fade_stop_atr`·ATR beyond the signal close.
- **vwap_pullback**: trend = close on one side of VWAP and the VWAP slope (in ATR
  over `slope_lookback`) agrees. The bar dips to within `pb_touch_atr`·ATR of VWAP and
  closes in the trend direction. Stop = `pb_stop_atr`·ATR beyond VWAP,
  target = `pb_target_r` × risk.
- Common rules: 15-minute warm-up, entries 09:45–15:30 ET, hard flat 15:55,
  max trades/day, one position, cooldown after a stop-out, optional daily-loss
  gate, US holidays no-trade by default, early-close days handled.

## Research

```bash
python scripts/research.py develop --config configs/default.yaml
#   -> research_output/real/report_develop.md, selected.yaml, trials_log.jsonl
python scripts/research.py holdout
#   -> research_output/real/report_holdout.md   (ONE shot; a second run is refused)
```

- Split by ET date: 60% development, 20% walk-forward validation, 20% holdout.
- `develop` physically removes holdout bars before running anything. It runs 48
  coarse configs (24 per variant), each shown with its one-step neighbours,
  and picks by plateau score (mean of config + neighbours), not by peak. It runs an
  anchored walk-forward over 4 validation folds, where each fold's parameters come
  only from earlier data, plus cost sensitivity (slippage 0.5/1/2/3 ticks ×
  commission ×1/×1.5/×2) on the walk-forward path, breakdowns (year, volatility
  regime, hour, side) and a trade-order bootstrap of max drawdown. Every
  configuration tried is logged cumulatively per dataset.
- `holdout` evaluates the frozen `selected.yaml` once and reports the result
  whatever it is. `--force-rerun-holdout` works but stamps the report as no
  longer a clean holdout.
- If walk-forward OOS is not net-positive after costs, the report says so in
  plain words. The rule: stop there. No new variants, no loosening costs.

## Paper trading (IBKR)

1. Run TWS or IB Gateway logged into a **paper** account (account id starts
   with `DU`). Enable API on port 7497. Allow the client id in `live.client_id`.
2. Make sure you have market data permissions for CME Nasdaq futures.
3. `python live_bot.py paper --config configs/default.yaml -v`

Behaviour:
- Seeds 2 days of 1-minute history, then aggregates IB 5-second bars into 1-minute bars.
- Every entry is a market parent plus a GTC OCO stop/target (outsideRth). The stop
  must be acknowledged or the bracket is cancelled. A position without a resting
  stop for its full size gets one placed immediately, or the bot flattens and halts.
- Startup and every reconnect reconcile positions and working orders. Orphans
  are cancelled, unknown positions flatten and halt, missing stops are
  replaced, and stale entry orders are cancelled.
- Reconnect uses exponential backoff (1 s → 60 s, 10 attempts), then halts.
  Broker-side stops stay working.
- State in `state/` (atomic JSON + `trades.jsonl` + `events.jsonl`). Halts
  persist across restarts. Clear one with `--clear-halt` after you've investigated it.
- Risk: realised + unrealised daily loss, max position, max orders/day, reject
  limit, stale data (no entries after 120 s, flatten+halt after 600 s in a
  position), flat window 15:56–18:05, and a `KILL` file in the working directory
  (flattens and halts).
- Alerts: `live.alert_kind: slack|telegram` with `NQBOT_SLACK_WEBHOOK` or
  `NQBOT_TELEGRAM_TOKEN` + `NQBOT_TELEGRAM_CHAT` in the environment.
- Rolls: trades the front month by the CME roll calendar (8 days before
  expiry). From the roll date it stops entering the old contract. Restart the bot
  to pick up the new front month (or set `live.contract_month`).

Replay through the live runner (no IB needed). It checks parity against the backtester:

```bash
python live_bot.py replay --config configs/default.yaml --state-dir /tmp/replay
```

## Go-live checklist

Do not skip items. Each one is a reason real money has been lost.

- [ ] Real data downloaded and `scripts/data_quality.py` is clean, or every flagged item is understood.
- [ ] `research.py develop` on real data shows walk-forward OOS **net-positive after costs**,
      robust across the cost grid (incl. 2–3 ticks slippage and ×2 commission), not driven by one
      year, one hour, or one side, and the selected config sits on a plateau.
- [ ] `research.py holdout` run **once**, result acceptable, and you decided in advance what "acceptable" means.
- [ ] Bootstrap p95 drawdown is survivable at your size. Size from that, not from the observed drawdown.
- [ ] `pytest -q` green on the exact commit you deploy.
- [ ] `python live_bot.py replay` on real data reports `parity OK`.
- [ ] ≥ 4 weeks of IB **paper** trading. Compare paper fills to the backtest over the same days
      (slippage, missed targets, stop fills) and update cost assumptions if they're worse.
- [ ] You've watched it handle: a restart mid-position, a TWS disconnect, a rejected order, the
      KILL file, a half-day, a DST change, a contract roll.
- [ ] TWS/Gateway auto-restart configured. You know what happens at the nightly IB reset.
- [ ] Alerts arrive on your phone. Test them.
- [ ] Commissions in config match your actual IB statement.
- [ ] Risk limits set for **MNQ, 1 contract**. Start there.
- [ ] Live gate: `export NQBOT_ENABLE_LIVE=I_ACCEPT_REAL_MONEY_RISK`, then
      `python live_bot.py live --live --config ...`, then type `LIVE <account> MNQ 1` when prompted.

## Tests

`pytest -q` covers:
- a hand-computed VWAP/σ example, including a reset across the DST switch
- causality by truncation at random cut points (signals and engine), plus a
  deliberately leaky strategy that the test catches
- execution: next-open fills, gap through stop, stop+target in one bar,
  trade-through targets, per-fill costs, end-of-data and roll flatten, time
  flatten, holidays and half days, data-gap regressions
- reconciliation identities (Σ trade P&L = Δequity; costs = fills × per-fill cost)
- live-replay **parity** to the tick
- risk limits
- the IB adapter against a fake IB (fills, partials, rejects, disconnects,
  stale ticks, delayed positions)
- hypothesis property tests on random OHLC
- the research harness's honesty guards
