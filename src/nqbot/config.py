"""Typed configuration. Every tunable lives in a dataclass and loads from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

import yaml

from nqbot.instruments import get_instrument

Variant = Literal["vwap_fade", "vwap_pullback"]
SessionMode = Literal["rth", "globex"]


def hhmm(s: str) -> int:
    """'09:45' -> 585 (minutes after midnight)."""
    hh, mm = s.split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad time {s!r}")
    return h * 60 + m


@dataclass
class StrategyConfig:
    variant: Variant = "vwap_fade"
    session: SessionMode = "rth"
    band_k: float = 2.0
    atr_len: int = 14
    slope_lookback: int = 15
    # --- vwap_fade ---
    fade_stop_atr: float = 2.0
    fade_trend_filter: bool = True
    fade_trend_max_slope_atr: float = 1.0
    # --- vwap_pullback ---
    pb_min_slope_atr: float = 0.3
    pb_touch_atr: float = 0.25
    pb_stop_atr: float = 1.0
    pb_target_r: float = 2.0
    # --- common gating ---
    warmup_minutes: int = 15
    entry_start: str = "09:45"
    entry_end: str = "15:30"
    flat_time: str = "15:55"
    max_trades_per_day: int = 3
    cooldown_minutes: int = 15
    max_daily_loss: float = 0.0  # currency, realised net; 0 disables
    trade_holidays: bool = False
    qty: int = 1

    def validate(self) -> None:
        if self.variant not in get_args(Variant):
            raise ValueError(f"variant must be one of {get_args(Variant)}")
        if self.session not in get_args(SessionMode):
            raise ValueError(f"session must be one of {get_args(SessionMode)}")
        if self.atr_len < 1 or self.slope_lookback < 1:
            raise ValueError("atr_len and slope_lookback must be >= 1")
        if self.band_k <= 0:
            raise ValueError("band_k must be > 0")
        if self.qty < 1:
            raise ValueError("qty must be >= 1")
        if self.max_trades_per_day < 0 or self.cooldown_minutes < 0 or self.warmup_minutes < 0:
            raise ValueError("counts/minutes must be >= 0")
        s, e, f = hhmm(self.entry_start), hhmm(self.entry_end), hhmm(self.flat_time)
        if not s <= e < f:
            raise ValueError("need entry_start <= entry_end < flat_time")
        for name in ("fade_stop_atr", "pb_stop_atr", "pb_target_r"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass
class CostConfig:
    instrument: str = "MNQ"
    slippage_ticks: float = 1.0  # per market fill (entry, stop, flatten)
    commission_per_side: float | None = None  # None -> instrument default
    commission_mult: float = 1.0
    target_through_ticks: int = 1  # limit target fills only if traded through by this
    initial_capital: float = 50_000.0

    def validate(self) -> None:
        get_instrument(self.instrument)
        if self.slippage_ticks < 0 or self.commission_mult < 0 or self.target_through_ticks < 0:
            raise ValueError("cost parameters must be >= 0")

    @property
    def commission(self) -> float:
        base = (
            self.commission_per_side
            if self.commission_per_side is not None
            else get_instrument(self.instrument).commission_per_side
        )
        return base * self.commission_mult


@dataclass
class RiskConfig:
    max_daily_loss: float = 500.0  # realised + unrealised, currency; breach -> flatten & halt
    max_position: int = 1
    max_orders_per_day: int = 20
    kill_file: str = "KILL"
    flat_window_start: str = "15:56"
    flat_window_end: str = "18:05"
    stale_seconds: float = 120.0  # no bar for this long in session -> no new entries
    stale_flatten_seconds: float = 600.0  # still stale while in a position -> flatten & halt


@dataclass
class LiveConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 = TWS paper, 7496 = TWS live
    client_id: int = 17
    account: str = ""
    exchange: str = "CME"
    contract_month: str = ""  # YYYYMM; empty -> front month by roll calendar
    state_dir: str = "state"
    buffer_bars: int = 2000
    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 60.0
    reconnect_max_attempts: int = 10
    order_confirm_timeout_s: float = 5.0
    entry_timeout_s: float = 30.0
    emergency_stop_points: float = 40.0
    alert_kind: Literal["none", "slack", "telegram"] = "none"
    max_rejects_per_day: int = 3


@dataclass
class AppConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def validate(self) -> AppConfig:
        self.strategy.validate()
        self.costs.validate()
        return self


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any] | None, path: str) -> T:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        hint = hints[key]
        if dataclasses.is_dataclass(hint):
            kwargs[key] = _build(hint, value, f"{path}.{key}")  # type: ignore[arg-type]
        else:
            kwargs[key] = _coerce(hint, value, f"{path}.{key}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = get_origin(hint)
    if origin is Literal:
        if value not in get_args(hint):
            raise ValueError(f"{path}: {value!r} not in {get_args(hint)}")
        return value
    args = get_args(hint)
    if value is None and type(None) in args:
        return None
    base = next((a for a in args if a is not type(None)), hint) if args else hint
    if base is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected bool")
        return value
    if base is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected int")
        return value
    if base is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected number")
        return float(value)
    if base is str:
        return str(value)
    return value


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    return _build(AppConfig, data, "config").validate()


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data)


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
