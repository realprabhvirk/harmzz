"""Contract specifications and quarterly contract/roll calendar for NQ and MNQ."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
CODE_MONTHS = {v: k for k, v in MONTH_CODES.items()}


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size: float
    point_value: float
    # All-in per contract per side (broker + exchange + regulatory). Assumption,
    # roughly IBKR tiered retail pricing; override in config.
    commission_per_side: float

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "NQ": InstrumentSpec("NQ", 0.25, 20.0, 2.25),
    "MNQ": InstrumentSpec("MNQ", 0.25, 2.0, 0.62),
}


def get_instrument(symbol: str) -> InstrumentSpec:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown instrument {symbol!r}; known: {sorted(INSTRUMENTS)}") from exc


def third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (4 - first.weekday()) % 7  # Friday = 4
    return first + dt.timedelta(days=offset + 14)


def expiry(year: int, month: int) -> dt.date:
    """Equity index futures expire on the third Friday of the contract month."""
    return third_friday(year, month)


def roll_date(year: int, month: int) -> dt.date:
    """CME equity-index roll convention: 8 calendar days before expiry (a Thursday)."""
    return expiry(year, month) - dt.timedelta(days=8)


def front_contract(day: dt.date) -> tuple[int, int]:
    """(year, month) of the contract to trade on ``day`` (switches on the roll date)."""
    year = day.year
    for month in (3, 6, 9, 12):
        if day < roll_date(year, month):
            return year, month
    return year + 1, 3


def contract_code(symbol: str, year: int, month: int) -> str:
    """e.g. ``contract_code('NQ', 2025, 12) == 'NQZ5'``."""
    return f"{symbol}{MONTH_CODES[month]}{year % 10}"


def ib_contract_month(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"
