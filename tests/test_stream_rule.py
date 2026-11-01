"""Streaming decision rule helpers: the fast (precomputed-EMA) path and the indexed scorer must equal the originals."""
import ctypes
import random

import numpy as np
import pytest

from train.kws.eval_c import load_engine, MMContext
from train.kws import stream_eval as S


def _reference_score(detections, events):
    """The original full scan over every event, kept as the reference."""
    matched, tp, dup, wrong, outside, lat = set(), 0, 0, 0, 0, []
    for t, cls in detections:
        hit = False
        for i, (s0, e0, c0) in enumerate(events):
            if s0 - 0.2 <= t <= e0 + 0.5 and cls == c0 and i not in matched:
                matched.add(i); tp += 1; lat.append((t - e0) * 1000.0); hit = True
                break
        if hit:
            continue
        if any(s0 - 0.2 <= t <= e0 + 0.5 and c0 == cls and i in matched for i, (s0, e0, c0) in enumerate(events)):
            dup += 1
        elif any(s0 - 0.2 <= t <= e0 + 0.5 for s0, e0, _ in events):
            wrong += 1
        else:
            outside += 1
    return tp, dup, wrong, outside, lat


@pytest.mark.parametrize("seed", range(5))
def test_indexed_scorer_matches_full_scan(seed):
    rng = random.Random(seed)
    # sorted events with some overlapping windows (gap < 0.7 s) on purpose
    events, t = [], 1.0
    for _ in range(120):
        events.append((t, t + 1.0, rng.randrange(10)))
        t += 1.0 + rng.uniform(0.3, 3.0)
    dets = sorted((rng.uniform(0, t), rng.randrange(10)) for _ in range(600))
    got = S.score_detections(dets, events, total_hours=t / 3600.0)
    tp, dup, wrong, outside, lat = _reference_score(dets, events)
    assert (got["true_positives"], got["fa_duplicate"], got["fa_wrong_class"], got["fa_outside"]) == (tp, dup, wrong, outside)
    assert got["latency_ms_median"] == (float(np.median(lat)) if lat else None)


def test_substitution_and_background_rates_are_reported():
    events = [(10.0, 11.0, 3)]
    dets = [(10.5, 3), (10.6, 4), (30.0, 2)]  # hit, substitution, pure background
    r = S.score_detections(dets, events, total_hours=3600.0 / 3600.0)
    assert (r["fa_wrong_class"], r["fa_outside"]) == (1, 1)
    bg_hours = 1.0 - 1.0 / 3600.0
    assert r["fa_wrong_class_per_background_hour"] == pytest.approx(1 / bg_hours)
    assert r["fa_outside_per_background_hour"] == pytest.approx(1 / bg_hours)


@pytest.mark.parametrize("alpha,refr,th", [(0.7, 25, 1.5), (0.85, 50, 0.5), (0.93, 75, 2.0), (0.85, 50, 4.0)])
def test_fast_rule_matches_c_rule(alpha, refr, th):
    lib = load_engine()
    lib.mm_decide_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int16]
    lib.mm_decide_ex.restype = ctypes.c_int
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2.5, size=(4000, 12)) * 4096
    logits[:, 10:] += 1500  # background classes usually win, keywords burst now and then
    bursts = rng.integers(0, 3900, 60)
    for b in bursts:
        logits[b:b + 25, rng.integers(0, 10)] += 45000  # margin ~11 in real units: even alpha 0.93 / threshold 4 fire
    logits = logits.astype(np.int32)

    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx)); lib.mm_reset(ctypes.byref(ctx))
    ref = []
    for t, l in enumerate(logits):
        d = lib.mm_decide_ex(ctypes.byref(ctx), (ctypes.c_int32 * 12)(*l), int(round(th * 4096)),
                             int(round(alpha * 32768)), refr)
        if d >= 0:
            ref.append(((t * 320 + 512) / 16000.0, d))
    sm = S.smoothed_margins(lib, list(logits), alpha)
    assert S.detections_from_margins(sm, th, refr) == ref and len(ref) > 0


def _row(det, fa, th, **kw):
    return {"detection_rate_pct": det, "fa_per_background_hour": fa, "threshold": th, **kw}


def test_budget_rule_picks_highest_detection_within_budget():
    rows = [_row(95, 300, 0.5), _row(80, 90, 1.0), _row(70, 20, 2.0), _row(80, 60, 1.5), _row(0, 0, 10.0)]
    best, ok = S.select_setting(rows, fa_budget=100)
    assert ok and (best["detection_rate_pct"], best["fa_per_background_hour"]) == (80, 60)  # tie on detection -> fewer FA


def test_budget_unreachable_picks_lowest_fa_not_max_detection():
    rows = [_row(95, 900, 0.25), _row(60, 400, 3.0), _row(40, 250, 4.0), _row(40, 250, 5.0, extra=1)]
    best, ok = S.select_setting(rows, fa_budget=100)
    assert not ok and best["fa_per_background_hour"] == 250 and best["threshold"] == 4.0


def test_poisson_rate_ci_exact_values():
    lo, hi = S.poisson_rate_ci(70, 1.5473217708)  # 70 false alarms over the 1.547 background hours of the test stream
    assert (round(lo, 1), round(hi, 1)) == (35.3, 57.2)
    lo, hi = S.poisson_rate_ci(14, 1.5473217708)
    assert (round(lo, 2), round(hi, 1)) == (4.95, 15.2)
    assert S.poisson_rate_ci(0, 2.0)[0] == 0.0 and S.poisson_rate_ci(0, 2.0)[1] == pytest.approx(3.6889 / 2.0, abs=1e-3)
    r = S.score_detections([(10.5, 3), (10.6, 4), (30.0, 2)], [(10.0, 11.0, 3)], total_hours=1.0)
    assert r["fa_outside_per_background_hour_ci95"][0] < r["fa_outside_per_background_hour"] < r["fa_outside_per_background_hour_ci95"][1]
