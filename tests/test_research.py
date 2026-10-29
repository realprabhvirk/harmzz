from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nqbot.research as R
from nqbot.cli import research as cli_r
from nqbot.config import AppConfig
from nqbot.data import prepare_bars, synthetic_bars

SMALL_GRIDS = {
    "vwap_fade": {"band_k": [1.5, 2.0], "fade_stop_atr": [1.0, 2.0]},
    "vwap_pullback": {"pb_stop_atr": [0.5, 1.0], "pb_target_r": [1.5, 2.0]},
}


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return prepare_bars(synthetic_bars(start="2023-01-02", days=80, seed=21, mode="rth"))


def test_split_is_time_ordered(bars: pd.DataFrame) -> None:
    s = R.split_by_date(bars)
    assert s.dev[0] <= bars.index[0] and s.dev[1] == s.val[0] and s.val[1] == s.holdout[0]
    assert s.holdout[1] > bars.index[-1]
    n = len(bars)
    frac_dev = (bars.index < s.dev[1]).sum() / n
    frac_hold = (bars.index >= s.holdout[0]).sum() / n
    assert 0.55 < frac_dev < 0.65 and 0.15 < frac_hold < 0.25
    assert set(s.as_dict()) == {"dev", "val", "holdout"}
    with pytest.raises(ValueError):
        R.split_by_date(bars.iloc[:100])


def test_neighbours_and_plateau_prefers_plateau() -> None:
    dims = {"a": [1, 2, 3, 4, 5], "b": [True, False]}
    assert sorted(R.neighbours((0, 0), dims)) == [(0, 1), (1, 0)]
    assert len(R.neighbours((2, 1), dims)) == 3
    # isolated peak at a=1 (index 0) vs broad plateau around a=4
    nets = {0: 1000.0, 1: -500.0, 2: 100.0, 3: 300.0, 4: 300.0}
    t0 = pd.Timestamp("2024-01-02 15:00", tz="UTC")
    keys, params, trades = [], [], []
    for i in range(5):
        for j in range(2):
            keys.append((i, j))
            params.append({"a": dims["a"][i], "b": dims["b"][j]})
            n = 40
            net = -500.0 if (i == 0 and j == 1) else nets[i]  # the peak is isolated in both dims
            trades.append(pd.DataFrame({"entry_time": [t0] * n, "net": [net / n] * n}))
    gr = R.GridRun("vwap_fade", dims, keys, params, trades)
    table = R.score_grid(gr, t0, t0 + pd.Timedelta(days=1), min_trades=30)
    best = R.pick(table)
    assert table.loc[table["net"].idxmax(), "a"] == 1  # raw peak
    assert best["a"] in (4, 5)  # plateau wins over the isolated peak
    none = R.score_grid(gr, t0, t0 + pd.Timedelta(days=1), min_trades=100)
    assert R.pick(none) is None


def test_develop_never_loads_holdout(bars: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = R.split_by_date(bars)
    seen: list[pd.Timestamp] = []
    real = R.run_config

    def spy(df: pd.DataFrame, cfg: AppConfig, feat=None):  # type: ignore[no-untyped-def]
        seen.append(df.index.max())
        return real(df, cfg, feat)

    monkeypatch.setattr(R, "run_config", spy)
    res = R.develop(bars, AppConfig(), tmp_path, "test-data", grids=SMALL_GRIDS, min_trades=5)
    assert seen and max(seen) < split.holdout[0]
    assert res.total_configs == 8
    assert (tmp_path / "report_develop.md").exists()
    assert "configurations tried in this run: **8**" in res.report
    if res.selected is not None:
        assert (tmp_path / "selected.yaml").exists()


def test_walk_forward_is_anchored(bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    split = R.split_by_date(bars)
    pre = bars[bars.index < split.holdout[0]]
    gr = R.run_grid(pre, AppConfig(), "vwap_fade", SMALL_GRIDS["vwap_fade"])
    folds = R.fold_bounds(pre, split.val[0], split.val[1], 3)
    assert len(folds) == 3 and folds[0][0] == split.val[0] and folds[-1][1] == split.val[1]
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    real = R.score_grid

    def spy(gr_, start, end, min_trades=30):  # type: ignore[no-untyped-def]
        calls.append((start, end))
        return real(gr_, start, end, min_trades)

    monkeypatch.setattr(R, "score_grid", spy)
    wf = R.walk_forward(gr, split.dev[0], folds, min_trades=1)
    assert [c[1] for c in calls] == [f[0] for f in folds]  # selection ends where the fold starts
    for f, (a, b) in zip(wf.folds, folds):
        if f["params"] is not None:
            assert pd.Timestamp(f["start"]) == a
    if len(wf.oos_trades):
        assert wf.oos_trades["entry_time"].min() >= split.val[0]
        assert wf.oos_trades["entry_time"].max() < split.val[1]


def test_verdict_when_nothing_profitable(bars: pd.DataFrame, tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.costs.instrument = "NQ"
    cfg.costs.commission_per_side = 1000.0  # absurd costs -> nothing can be profitable
    res = R.develop(bars, cfg, tmp_path, "costly", grids=SMALL_GRIDS, min_trades=1)
    assert not res.verdict_profitable
    assert "Nothing is net-profitable out-of-sample" in res.report


def test_holdout_lock(bars: pd.DataFrame, tmp_path: Path) -> None:
    cfg = AppConfig()
    rep = R.holdout(bars, cfg, tmp_path, "data-A", synthetic=True)
    assert "reported whatever it is" in rep and "SYNTHETIC" in rep
    with pytest.raises(R.HoldoutAlreadyUsed):
        R.holdout(bars, cfg, tmp_path, "data-A")
    R.holdout(bars, cfg, tmp_path, "data-B")  # different dataset is fine
    forced = R.holdout(bars, cfg, tmp_path, "data-A", force=True)
    assert "no longer a clean holdout" in forced
    lock = json.loads((tmp_path / "holdout_lock.json").read_text())
    assert [r["data"] for r in lock] == ["data-A", "data-B", "data-A"]
    assert lock[-1]["forced_rerun"] is True


def test_trial_log_is_cumulative(tmp_path: Path) -> None:
    assert R.log_trials(tmp_path, "d1", 10, "x") == 10
    assert R.log_trials(tmp_path, "d2", 5, "x") == 5
    assert R.log_trials(tmp_path, "d1", 7, "y") == 17


def test_trade_stats_and_cost_sensitivity(bars: pd.DataFrame) -> None:
    split = R.split_by_date(bars)
    cs = R.cost_sensitivity(bars, AppConfig(), split.val[0], split.val[1])
    assert len(cs) == len(R.COST_SLIPPAGE) * len(R.COST_COMMISSION_MULT)
    # more costs never help
    base = cs[(cs.commission_x == 1.0)].sort_values("slippage_ticks")["net"].to_numpy()
    assert (np.diff(base) <= 1e-9).all()
    s = R.trade_stats(R._empty_trades())
    assert s["trades"] == 0 and np.isnan(s["sharpe_ann"])


def test_research_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "DEFAULT_GRIDS", SMALL_GRIDS)
    out = tmp_path / "rs"
    with pytest.raises(SystemExit):
        cli_r.main(["holdout", "--synthetic-days", "60", "--out", str(out)])  # no selection yet
    assert cli_r.main(["develop", "--synthetic-days", "60", "--out", str(out), "--min-trades", "1"]) == 0
    assert cli_r.main(["holdout", "--synthetic-days", "60", "--out", str(out)]) == 0
    assert cli_r.main(["holdout", "--synthetic-days", "60", "--out", str(out)]) == 2
    text = capsys.readouterr().out
    assert "REFUSED" in text and "SYNTHETIC" in text
