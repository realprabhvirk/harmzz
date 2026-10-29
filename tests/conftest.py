from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqbot.config import AppConfig
from nqbot.data import prepare_bars, synthetic_bars


def make_bars(
    rows: list[tuple[str, float, float, float, float, float]],
    contract: str | list[str] = "NQZ4",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Hand-made bars: (local time, open, high, low, close, volume)."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz=tz) for r in rows]).tz_convert("UTC")
    df = pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
            "contract": contract if isinstance(contract, list) else [contract] * len(rows),
            "roll_spread": np.nan,
        },
        index=idx,
    )
    return prepare_bars(df)


def flat_session(day: str, n: int = 390, price: float = 100.0, start: str = "09:30") -> list:
    t0 = pd.Timestamp(f"{day} {start}")
    return [((t0 + pd.Timedelta(minutes=i)).isoformat(), price, price, price, price, 10.0) for i in range(n)]


@pytest.fixture(scope="session")
def synth_globex() -> pd.DataFrame:
    return prepare_bars(synthetic_bars(start="2024-02-26", days=30, seed=11, mode="globex"))


@pytest.fixture(scope="session")
def synth_rth() -> pd.DataFrame:
    return prepare_bars(synthetic_bars(start="2024-01-02", days=40, seed=3, mode="rth"))


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()
