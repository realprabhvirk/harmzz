# RESULTS

## Strategy results on real data: **none yet**

There's no real market data in this environment: no `data/nq_1m.parquet`,
no Databento key, and the download costs money and needs your account. So
**no strategy result exists.** Nothing here says whether `vwap_fade` or
`vwap_pullback` makes money on NQ/MNQ, and nothing should be read that way.

To produce the real numbers:

```bash
python fetch_nq_data.py
python scripts/data_quality.py
python scripts/research.py develop --config configs/default.yaml   # read report_develop.md
python scripts/research.py holdout                                 # once
```

Then replace this section with the numbers from `research_output/real/`,
including the trial count, whatever they show. If the walk-forward OOS is not
net-positive after costs, write exactly that here and stop. Per the brief, no
new variants and no cost loosening.

## What has been verified (mechanics, on synthetic and hand-made data)

| check | result |
|---|---|
| test suite | 144 tests pass under JIT and with `NUMBA_DISABLE_JIT=1` |
| line coverage of `nqbot` (no-JIT run) | 97% overall; core/engine/features/strategies/risk/brokers ≥ 97% |
| VWAP/σ vs hand computation, incl. DST-day reset | exact |
| causality (random cut points, both strategies, both session modes; engine-level too) | no violations; leaky control strategy is detected |
| live replay vs backtester | identical trades to the tick (entry/exit time, prices, reason, P&L) incl. stops, targets, rolls, data gaps, zero-volume bars, qty 2, restart mid-position |
| Σ trade net = Δ equity; costs = fills × per-fill cost | holds (also as hypothesis property over random OHLC) |
| look-ahead sanity: random walk with zero edge, default configs, 2,500 days | mean gross per trade 0.18 ± 0.65 pts (fade, n=5,800), −0.25 ± 1.46 pts (pullback, n=1,939): no spurious edge |
| speed | 5.2M 1-minute bars (15 years) backtested in 4.1 s (fade) / 6.6 s (pullback) after JIT warm-up |

A warning the synthetic pipeline run showed: on a 300-day random walk, the
research harness picked `vwap_pullback` with a **positive** walk-forward OOS
(+$1,006 on 44 trades). The one-shot holdout then lost $1,921 on 71 trades. That's
noise passing a walk-forward check. It's why the holdout exists and why one
positive validation path proves nothing.

## Limitations and things NOT modelled

Execution and market:
- **Queue position / partial fills on limit targets.** Targets need a 1-tick
  trade-through. That approximates queue risk but doesn't model it. Real fills
  at the target can be worse (missed fills, then a later stop).
- **Slippage is a constant** number of ticks per market fill. Fast markets,
  news (CPI, FOMC, NFP), the open and thin overnight books slip far more.
  Gap fills use the bar open, which is optimistic in a true fast market.
- **Intrabar path is unknown.** Stop-first on ambiguous bars is conservative
  for P&L, but the order of events inside a minute is still guessed.
- **Gap through target fills at the target price.** Conservative: no price improvement.
- **Latency.** Decisions are assumed to reach the exchange at the next open.
  Live, the market order goes out a few seconds after the minute closes.
- **Market impact** is zero. Fine for 1–few MNQ. Not fine at size.
- Stop and target levels are referenced to the signal-bar close, not the
  actual fill. The bracket's distance from the real entry differs by the
  close→open move.
- Live bars come from IB 5-second TRADES bars. Historical bars come from Databento.
  Small OHLC differences are possible, and live signals can differ from a later
  backtest of the same day.

Data and calendar:
- The CME holiday/half-day calendar is rule-based and approximate (not an
  exchange feed). Unusual closures (e.g. national days of mourning) are missing.
- Roll spread comes from the v.1 series at the last common minute before the
  roll. If that's missing, the open-vs-previous-close gap is used and flagged by the data-quality script.
- The volume-based continuous roll (`.v.0`) is decided by Databento. Live
  uses the CME calendar roll (8 days before expiry). They can differ by a few days.
- No survivorship issue for a single futures root, but **regime change** is real:
  NQ microstructure in 2010 is not NQ in 2026.

Costs:
- Commission defaults (NQ $2.25, MNQ $0.62 per side all-in) are assumptions
  close to IBKR tiered retail. Check them against your statements.
- No exchange data fees, platform fees or margin interest.

Live adapter:
- `IBBroker` is tested only against `tests/fake_ib.py`, which is my model of
  how IB behaves. It has **not** been run against real TWS/Gateway. In
  particular: bracket child behaviour on parent partial fills, OCA handling,
  status strings, error codes and reconnect re-sync must be confirmed in paper.
- IB nightly server resets and TWS auto-restart are not simulated.
- If reconnect fails, the bot halts but **cannot flatten**. It relies on the
  broker-side GTC stop. That is by design, but know it.
- `ib_async` / `ib_insync` threading and event-loop details (e.g. running inside
  Jupyter) are not handled.

Research:
- Grids are deliberately coarse. Plateau scoring uses net P&L, which favours
  configurations that trade more. Read trade counts alongside it.
- Walk-forward is anchored with 4 folds. With ~3 years of validation data that's
  short and still one path.
- Volatility-regime breakdown uses terciles of a trailing daily-range measure.
  It's descriptive, not a tradable filter.
