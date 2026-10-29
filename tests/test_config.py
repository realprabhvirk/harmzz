from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nqbot.config import AppConfig, config_from_dict, config_to_dict, hhmm, load_config
from nqbot.instruments import (
    contract_code,
    expiry,
    front_contract,
    get_instrument,
    ib_contract_month,
    roll_date,
)
import datetime as dt


def test_hhmm() -> None:
    assert hhmm("09:45") == 585
    assert hhmm("00:00") == 0
    with pytest.raises(ValueError):
        hhmm("24:00")


def test_defaults_validate() -> None:
    cfg = AppConfig().validate()
    assert cfg.strategy.variant == "vwap_fade"
    assert cfg.costs.commission == pytest.approx(0.62)


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.strategy.variant = "vwap_pullback"
    cfg.costs.instrument = "NQ"
    cfg.costs.commission_mult = 1.5
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(config_to_dict(cfg)))
    back = load_config(p)
    assert back == cfg
    assert back.costs.commission == pytest.approx(2.25 * 1.5)


def test_repo_configs_load() -> None:
    for p in Path("configs").glob("*.yaml"):
        load_config(p)


@pytest.mark.parametrize(
    "data",
    [
        {"strategy": {"nope": 1}},
        {"strategy": {"variant": "magic"}},
        {"strategy": {"atr_len": "x"}},
        {"strategy": {"qty": 0}},
        {"strategy": {"entry_start": "16:00"}},
        {"strategy": {"fade_trend_filter": 1}},
        {"strategy": {"band_k": True}},
        {"costs": {"instrument": "ES"}},
        {"costs": {"slippage_ticks": -1}},
        {"strategy": []},
        {"strategy": {"fade_stop_atr": 0}},
        {"strategy": {"atr_len": 0}},
        {"strategy": {"band_k": -1}},
        {"strategy": {"cooldown_minutes": -1}},
    ],
)
def test_bad_config_rejected(data: dict) -> None:
    with pytest.raises(ValueError):
        config_from_dict(data)


def test_optional_none_and_session() -> None:
    cfg = config_from_dict({"costs": {"commission_per_side": None}, "strategy": {"session": "globex"}})
    assert cfg.costs.commission_per_side is None
    with pytest.raises(ValueError):
        config_from_dict({"strategy": {"session": "asia"}})


def test_instruments_and_roll_calendar() -> None:
    nq = get_instrument("nq")
    assert nq.tick_value == 5.0
    assert get_instrument("MNQ").tick_value == 0.5
    with pytest.raises(ValueError):
        get_instrument("ES")
    assert expiry(2024, 12) == dt.date(2024, 12, 20)
    assert roll_date(2024, 12) == dt.date(2024, 12, 12)
    assert front_contract(dt.date(2024, 12, 11)) == (2024, 12)
    assert front_contract(dt.date(2024, 12, 12)) == (2025, 3)
    assert front_contract(dt.date(2024, 1, 5)) == (2024, 3)
    assert contract_code("NQ", 2025, 12) == "NQZ5"
    assert ib_contract_month(2025, 3) == "202503"
