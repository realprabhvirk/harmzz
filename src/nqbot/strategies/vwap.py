"""VWAP strategies. Both are pure functions of causal features.

vwap_fade     close beyond the -k sigma band -> long (mirror short); target = VWAP
              at the signal bar, stop = N*ATR beyond the signal close. Optional
              trend filter skips fades against a steep VWAP slope.
vwap_pullback trend = close on one side of VWAP and VWAP slope agrees; the bar
              pulls back to within x*ATR of VWAP and closes in the trend
              direction. Stop = m*ATR beyond VWAP, target = R multiple of risk.

Levels are referenced to the signal bar's close because the bracket must be
known when the order is sent; the actual fill is the next open.
"""

from __future__ import annotations

import numpy as np

from nqbot.config import StrategyConfig
from nqbot.features import Features
from nqbot.strategies.base import Signals, finalize


def _valid(feat: Features) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (
            feat.in_session
            & np.isfinite(feat.vwap)
            & np.isfinite(feat.atr)
            & (feat.atr > 0)
            & (feat.sd > 0)
            & (feat.volume > 0)
        )


class VwapFade:
    name = "vwap_fade"

    def signals(self, feat: Features, cfg: StrategyConfig, tick: float) -> Signals:
        c = feat.close
        valid = _valid(feat)
        with np.errstate(invalid="ignore"):
            long_ = valid & (c < feat.lower(cfg.band_k))
            short = valid & (c > feat.upper(cfg.band_k))
            if cfg.fade_trend_filter:
                s = feat.slope_atr
                lim = cfg.fade_trend_max_slope_atr
                long_ &= np.isfinite(s) & (s > -lim)
                short &= np.isfinite(s) & (s < lim)
        dist = cfg.fade_stop_atr * feat.atr
        return finalize(long_, short, c - dist, feat.vwap, c + dist, feat.vwap, c, tick)


class VwapPullback:
    name = "vwap_pullback"

    def signals(self, feat: Features, cfg: StrategyConfig, tick: float) -> Signals:
        c, o, h, low, vw, atr = feat.close, feat.open, feat.high, feat.low, feat.vwap, feat.atr
        valid = _valid(feat)
        s = feat.slope_atr
        zone = cfg.pb_touch_atr * atr
        with np.errstate(invalid="ignore"):
            up = valid & np.isfinite(s) & (c > vw) & (s >= cfg.pb_min_slope_atr)
            dn = valid & np.isfinite(s) & (c < vw) & (s <= -cfg.pb_min_slope_atr)
            long_ = up & (low <= vw + zone) & (c > o)
            short = dn & (h >= vw - zone) & (c < o)
        stop_l = vw - cfg.pb_stop_atr * atr
        stop_s = vw + cfg.pb_stop_atr * atr
        tick_round = lambda x: np.round(x / tick) * tick  # noqa: E731
        risk_l = c - tick_round(stop_l)
        risk_s = tick_round(stop_s) - c
        tgt_l = c + cfg.pb_target_r * risk_l
        tgt_s = c - cfg.pb_target_r * risk_s
        return finalize(long_, short, stop_l, tgt_l, stop_s, tgt_s, c, tick)
