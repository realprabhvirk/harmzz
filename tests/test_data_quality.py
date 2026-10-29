from __future__ import annotations

import numpy as np
import pandas as pd

from nqbot.data import synthetic_bars
from nqbot.data_quality import format_report, quality_report


def _clean() -> pd.DataFrame:
    # spans the March 2024 DST change and the March roll
    return synthetic_bars(start="2024-02-26", days=25, seed=4, mode="globex")


def test_clean_synthetic_passes_most_checks() -> None:
    rep = quality_report(_clean())
    assert rep["timezone"]["ok"], rep["timezone"]
    assert rep["timezone"]["volume_peak_minute_et"] == {"est": 570, "edt": 570}
    assert rep["duplicates"]["ok"]
    assert rep["bad_ticks"]["ok"]
    assert rep["gaps"]["ok"]
    assert rep["coverage"]["ok"]
    assert rep["rolls"]["rolls"] == 1
    assert rep["rolls"]["sequence"] == ["NQH4", "NQM4"]
    assert "OVERALL" in format_report(rep)


def test_detects_injected_problems() -> None:
    df = _clean()
    df = pd.concat([df, df.iloc[100:103]])  # duplicates
    i = df.index.get_loc(df.index[5000])
    df.iloc[i, df.columns.get_loc("high")] = df.iloc[i]["low"] - 1  # high < low
    df.iloc[i + 1, df.columns.get_loc("close")] = 1.13  # off tick + spike
    df.iloc[i + 2, df.columns.get_loc("volume")] = 0
    # drop 20 minutes of RTH on one day
    loc = df.index.tz_convert("America/New_York")
    hole = (loc.date == pd.Timestamp("2024-03-05").date()) & (loc.hour == 11) & (loc.minute < 20)
    df = df[~hole]
    df.loc[df.index[df.contract == "NQM4"][0], "roll_spread"] = np.nan
    rep = quality_report(df)
    assert rep["duplicates"]["duplicate_timestamps"] == 3
    b = rep["bad_ticks"]
    assert b["high_below_low"] >= 1 and b["off_tick"] >= 1 and b["return_spikes"] >= 1
    assert b["zero_volume"] >= 1 and not b["ok"]
    assert rep["gaps"]["largest_rth_gaps"][0] == ("2024-03-05 11:00", 20)
    assert any("roll_spread" in s for s in rep["rolls"]["issues"])
    assert not rep["ok"]


def test_detects_wrong_timezone() -> None:
    df = _clean()
    # Classic bug: ET wall-clock times stored as if they were UTC
    wrong = df.copy()
    wrong.index = df.index.tz_convert("America/New_York").tz_localize(None).tz_localize("UTC")
    rep = quality_report(wrong)
    assert not rep["timezone"]["ok"]
    naive = df.tz_localize(None)
    assert quality_report(naive)["ok"] is False


def test_detects_bad_roll_sequence() -> None:
    df = synthetic_bars(start="2024-02-26", days=25, seed=4, mode="globex")
    df.loc[df.contract == "NQM4", "contract"] = "NQZ4"
    rep = quality_report(df)
    assert any("unexpected roll" in s for s in rep["rolls"]["issues"])


def test_low_coverage_session() -> None:
    df = synthetic_bars(start="2024-04-01", days=5, seed=2, mode="rth")
    loc = df.index.tz_convert("America/New_York")
    df = df[~((loc.date == pd.Timestamp("2024-04-02").date()) & (loc.hour >= 12))]
    rep = quality_report(df)
    assert rep["coverage"]["n_low_coverage"] == 1
