"""Data quality checks for 1-minute NQ bars.

Run on the RAW parquet (before prepare_bars) so duplicates/zero-volume bars are seen.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nqbot.data import contract_ids
from nqbot.instruments import CODE_MONTHS
from nqbot.sessions import ET, HALT_START, GLOBEX_START, RTH_END, RTH_START, session_close_mod


def _local(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(ET)


def check_timezone(df: pd.DataFrame) -> dict[str, Any]:
    idx = df.index
    out: dict[str, Any] = {"tz": str(idx.tz), "tz_aware": idx.tz is not None}
    if idx.tz is None:
        out["ok"] = False
        return out
    loc = _local(idx)
    mod = loc.hour * 60 + loc.minute
    out["bars_in_halt_17_18_et"] = int(((mod >= HALT_START) & (mod < GLOBEX_START)).sum())
    wd = loc.weekday
    out["weekend_bars"] = int(((wd == 5) | ((wd == 6) & (mod < GLOBEX_START)) | ((wd == 4) & (mod >= HALT_START))).sum())
    # DST sanity: the busiest minute around the open should be 09:30 ET in both
    # winter (UTC-5) and summer (UTC-4). A wrong tz shifts it by an hour.
    offs = (loc.tz_localize(None).as_unit("ns").asi8 - idx.as_unit("ns").asi8) // 60_000_000_000
    dst = offs == -240
    vol = df["volume"].to_numpy()
    mod = np.asarray(mod)
    peaks = {}
    for label, mask in (("est", ~dst), ("edt", dst)):
        m = mask & (mod >= RTH_START - 60) & (mod < RTH_START + 60)
        if m.sum() == 0:
            continue
        prof = pd.Series(vol[m]).groupby(mod[m]).sum()
        peaks[label] = int(prof.idxmax())
    out["volume_peak_minute_et"] = peaks
    out["ok"] = (
        out["bars_in_halt_17_18_et"] == 0
        and out["weekend_bars"] == 0
        and all(abs(v - RTH_START) <= 1 for v in peaks.values())
    )
    return out


def check_duplicates_and_order(df: pd.DataFrame) -> dict[str, Any]:
    dup = int(df.index.duplicated().sum())
    unsorted = int((np.diff(df.index.as_unit("ns").asi8) < 0).sum()) if len(df) > 1 else 0
    return {"duplicate_timestamps": dup, "out_of_order": unsorted, "ok": dup == 0 and unsorted == 0}


def check_bad_ticks(df: pd.DataFrame, tick: float = 0.25, spike_sigma: float = 12.0) -> dict[str, Any]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(float)
    nan = int(np.isnan(np.c_[o, h, l, c, v]).any(axis=1).sum())
    nonpos = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
    hl = int((h < l).sum())
    oc_out = int(((o > h) | (o < l) | (c > h) | (c < l)).sum())
    prices = np.c_[o, h, l, c]
    off_tick = int((np.abs(prices / tick - np.round(prices / tick)) > 1e-6).any(axis=1).sum())
    zero_vol = int((v <= 0).sum())
    neg_vol = int((v < 0).sum())
    # spikes: |1-min log return| vs a robust scale, excluding contract changes
    cid = contract_ids(df["contract"]) if "contract" in df else np.zeros(len(df), dtype=np.int64)
    spikes: list[str] = []
    if len(c) > 50:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(c))
        r[cid[1:] != cid[:-1]] = np.nan
        mad = np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826
        if mad > 0:
            z = np.abs(r) / mad
            bad = np.flatnonzero(z > spike_sigma) + 1
            spikes = [str(df.index[i]) for i in bad[:20]]
            n_spikes = int(len(bad))
        else:
            n_spikes = 0
        # range spikes: bar range >> typical
        rng = h - l
        med = np.nanmedian(rng)
        range_spikes = int((rng > max(med, tick) * 40).sum())
    else:
        n_spikes = 0
        range_spikes = 0
    return {
        "nan_rows": nan,
        "non_positive": nonpos,
        "high_below_low": hl,
        "open_close_outside_range": oc_out,
        "off_tick": off_tick,
        "zero_volume": zero_vol,
        "negative_volume": neg_vol,
        "return_spikes": n_spikes,
        "range_spikes": range_spikes,
        "spike_examples": spikes,
        "ok": nan == 0 and nonpos == 0 and hl == 0 and oc_out == 0 and off_tick == 0 and neg_vol == 0,
    }


def check_gaps(df: pd.DataFrame, min_gap: int = 5) -> dict[str, Any]:
    """Missing minutes inside RTH (09:30-16:00 ET, calendar aware)."""
    if len(df) == 0:
        return {"rth_missing_minutes": 0, "largest_rth_gaps": [], "ok": True}
    loc = _local(df.index)
    mod = np.asarray(loc.hour * 60 + loc.minute)
    dates = np.asarray(loc.date)
    rth = (mod >= RTH_START) & (mod < RTH_END)
    missing = 0
    big: list[tuple[str, int]] = []
    s = pd.Series(mod[rth], index=pd.Index(dates[rth]))
    for d, mins in s.groupby(level=0):
        close, _ = session_close_mod(d)
        arr = np.sort(mins.to_numpy())
        arr = arr[arr < close]
        expected = close - RTH_START
        missing += expected - len(np.unique(arr))
        full = np.concatenate(([RTH_START - 1], arr, [close]))
        gaps = np.diff(full) - 1
        for i in np.flatnonzero(gaps >= min_gap):
            big.append((f"{d} {int(full[i] + 1) // 60:02d}:{int(full[i] + 1) % 60:02d}", int(gaps[i])))
    big.sort(key=lambda x: -x[1])
    return {"rth_missing_minutes": int(missing), "largest_rth_gaps": big[:15], "ok": len(big) == 0}


def check_session_coverage(df: pd.DataFrame, min_frac: float = 0.9) -> dict[str, Any]:
    if len(df) == 0:
        return {"sessions": 0, "low_coverage_sessions": [], "ok": True}
    loc = _local(df.index)
    mod = np.asarray(loc.hour * 60 + loc.minute)
    dates = np.asarray(loc.date)
    rth = (mod >= RTH_START) & (mod < RTH_END)
    counts = pd.Series(1, index=pd.Index(dates[rth])).groupby(level=0).sum()
    low = []
    for d, n in counts.items():
        close, _ = session_close_mod(d)
        exp = close - RTH_START
        if n < min_frac * exp:
            low.append((str(d), int(n), int(exp)))
    weekdays = pd.bdate_range(min(dates), max(dates))
    present = set(counts.index)
    missing_days = [str(d.date()) for d in weekdays if d.date() not in present]
    return {
        "sessions": int(len(counts)),
        "median_rth_bars": float(counts.median()),
        "low_coverage_sessions": low[:30],
        "n_low_coverage": len(low),
        "weekdays_without_rth_bars": missing_days[:40],
        "n_weekdays_without_rth_bars": len(missing_days),
        "ok": len(low) == 0,
    }


def check_rolls(df: pd.DataFrame) -> dict[str, Any]:
    if "contract" not in df or len(df) == 0:
        return {"rolls": 0, "ok": True}
    cid = contract_ids(df["contract"])
    roll_idx = np.flatnonzero(np.diff(cid) != 0) + 1
    contracts = df["contract"].to_numpy()
    seq = [contracts[0]] + [contracts[i] for i in roll_idx]
    issues: list[str] = []
    if len(set(seq)) != len(seq):
        issues.append("a contract appears in more than one segment (roll back-and-forth)")
    months = []
    for code in seq:
        m = CODE_MONTHS.get(code[-2]) if len(code) >= 3 else None
        months.append(m)
    for a, b, ca, cb in zip(months, months[1:], seq, seq[1:]):
        if a is not None and b is not None and (b - a) % 12 != 3:
            issues.append(f"unexpected roll {ca} -> {cb}")
    loc = _local(df.index)
    roll_times = [str(loc[i]) for i in roll_idx]
    days_between = np.diff([df.index[i].value for i in roll_idx]) / 86_400e9 if len(roll_idx) > 1 else np.array([])
    rs = df["roll_spread"].to_numpy() if "roll_spread" in df else np.full(len(df), np.nan)
    missing_spread = int(np.isnan(rs[roll_idx]).sum()) if len(roll_idx) else 0
    mid_rth = [t for i, t in zip(roll_idx, roll_times) if RTH_START <= loc[i].hour * 60 + loc[i].minute < RTH_END]
    if missing_spread:
        issues.append(f"{missing_spread} roll(s) without roll_spread; gap fallback used for signals")
    if len(days_between) and ((days_between < 70) | (days_between > 110)).any():
        issues.append("roll interval outside 70-110 days")
    if mid_rth:
        issues.append(f"{len(mid_rth)} roll(s) during RTH")
    return {
        "rolls": int(len(roll_idx)),
        "sequence": seq[:80],
        "roll_times_et": roll_times[:80],
        "days_between_rolls": [round(float(x), 1) for x in days_between[:80]],
        "spreads": [float(x) for x in rs[roll_idx][:80]],
        "issues": issues,
        "ok": not issues,
    }


def quality_report(df: pd.DataFrame, tick: float = 0.25) -> dict[str, Any]:
    raw = df.copy()
    if "ts" in raw.columns:
        raw = raw.set_index("ts")
    rep = {
        "rows": int(len(raw)),
        "start": str(raw.index.min()) if len(raw) else None,
        "end": str(raw.index.max()) if len(raw) else None,
        "timezone": check_timezone(raw),
    }
    if not rep["timezone"]["tz_aware"]:
        rep["ok"] = False
        return rep
    raw.index = raw.index.tz_convert("UTC")
    rep["duplicates"] = check_duplicates_and_order(raw)
    clean = raw[~raw.index.duplicated()].sort_index()
    rep["bad_ticks"] = check_bad_ticks(clean, tick)
    rep["gaps"] = check_gaps(clean)
    rep["coverage"] = check_session_coverage(clean)
    rep["rolls"] = check_rolls(clean)
    rep["ok"] = all(rep[k]["ok"] for k in ("timezone", "duplicates", "bad_ticks", "gaps", "coverage", "rolls"))
    return rep


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"# Data quality report", "", f"rows: {rep['rows']}  range: {rep['start']} -> {rep['end']}", ""]
    for key in ("timezone", "duplicates", "bad_ticks", "gaps", "coverage", "rolls"):
        if key not in rep:
            continue
        sec = rep[key]
        lines.append(f"## {key}: {'OK' if sec.get('ok') else 'CHECK'}")
        for k, v in sec.items():
            if k == "ok":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"OVERALL: {'OK' if rep.get('ok') else 'ISSUES FOUND'}")
    return "\n".join(lines)
