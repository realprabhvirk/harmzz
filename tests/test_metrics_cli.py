from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nqbot.cli import backtest as cli_bt
from nqbot.cli import quality as cli_q
from nqbot.config import AppConfig
from nqbot.data import synthetic_bars
from nqbot.engine import run_backtest
from nqbot.metrics import (
    bootstrap_drawdown,
    by_exit_reason,
    by_hour,
    by_side,
    by_vol_regime,
    by_year,
    daily_range_regime,
    max_drawdown,
)


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([0, 10, 5, 12, 2, 8])) == 10
    assert max_drawdown(np.array([])) == 0.0


def test_bootstrap_deterministic() -> None:
    pnl = np.array([10, -5, 3, -8, 12, -2, -6, 4.0])
    a = bootstrap_drawdown(pnl, 500, seed=1)
    assert a == bootstrap_drawdown(pnl, 500, seed=1)
    assert a["p50"] <= a["p90"] <= a["p99"]
    assert a["observed"] == max_drawdown(np.concatenate(([0], np.cumsum(pnl))))
    assert bootstrap_drawdown(np.array([]))["p95"] == 0.0


def test_breakdowns(synth_globex: pd.DataFrame) -> None:
    cfg = AppConfig()
    cfg.strategy.band_k = 1.5
    res = run_backtest(synth_globex, cfg)
    tr = res.trades
    for fn in (by_year, by_hour, by_side, by_exit_reason):
        out = fn(tr)
        assert out["trades"].sum() == len(tr)
        assert out["net"].sum() == pytest.approx(tr["net"].sum())
    reg = by_vol_regime(tr, synth_globex)
    assert reg["trades"].sum() == len(tr)
    r = daily_range_regime(synth_globex)
    assert r.iloc[:5].isna().all()  # needs history, causal
    empty = tr.iloc[:0]
    assert len(by_year(empty)) == 0 and len(by_vol_regime(empty, synth_globex)) == 0


def test_backtest_cli_synthetic(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out_csv = tmp_path / "t.csv"
    assert cli_bt.main(["--synthetic-days", "15", "--trades-csv", str(out_csv)]) == 0
    text = capsys.readouterr().out
    assert "SYNTHETIC" in text and "net" in text
    assert out_csv.exists()
    assert cli_bt.main(["--synthetic-days", "5", "--json", "--config", "configs/pullback.yaml"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["synthetic"] is True


def test_backtest_cli_real_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "bars.parquet"
    synthetic_bars(days=3, mode="rth").to_parquet(p)
    assert cli_bt.main(["--data", str(p)]) == 0
    assert "SYNTHETIC" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_bt.main(["--data", str(tmp_path / "missing.parquet")])


def test_quality_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "bars.parquet"
    synthetic_bars(start="2024-04-01", days=5, mode="globex").to_parquet(p)
    rc = cli_q.main(["--data", str(p), "--out", str(tmp_path / "q.md")])
    assert rc in (0, 1)
    assert (tmp_path / "q.md").read_text().startswith("# Data quality")
    capsys.readouterr()
    cli_q.main(["--data", str(p), "--json"])
    assert json.loads(capsys.readouterr().out)["rows"] == 5 * 1380
