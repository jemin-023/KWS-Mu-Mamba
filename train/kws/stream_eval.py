import argparse
import bisect
import json
import random
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict, Sequence
import numpy as np
from scipy.stats import chi2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.kws.data import SpeechCommandsDataset, TARGET_KEYWORDS, CLASS_TO_IDX, IDX_TO_CLASS, build_continuous_stream
from train.kws.eval_c import load_engine, MMContext, ModelWeightsQ, populate_c_weights
from train.kws.model import MuMambaKWS

def poisson_rate_ci(count: int, hours: float, conf: float = 0.95):
    """Exact (Garwood, chi-square) two-sided interval for a Poisson count, divided by the observation time in hours."""
    if hours <= 0:
        return [0.0, 0.0]
    alpha = 1.0 - conf
    lo = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2.0
    hi = chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2.0
    return [float(lo / hours), float(hi / hours)]

def score_detections(
    detections: List[Tuple[float, int]],
    events: List[Tuple[float, float, int]],
    total_hours: float,
) -> Dict[str, float]:
    """
    Matches detections to ground-truth keyword events.

    A detection at time t matches an event (start, end, cls) when
    start - 0.2 <= t <= end + 0.5; the first unmatched event with the
    right class is consumed (true positive).

    Every detection that is not a true positive is a false alarm, classified
    into exactly one category (checked in this order):
      fa_duplicate   - inside the window of an event of the detected class
                       which was already matched
      fa_wrong_class - inside at least one event window, but no window of the
                       detected class contains it
      fa_outside     - inside no event window at all

    Returns a dict with events, true_positives, false_negatives,
    detection_rate_pct, fa_duplicate, fa_wrong_class, fa_outside,
    false_alarms, fa_per_hour, fa_per_hour_excl_duplicates, total_hours,
    background_hours (total minus the summed event durations),
    fa_per_background_hour, fa_per_background_hour_excl_duplicates,
    fa_wrong_class_per_background_hour (substitutions: a wrong keyword inside a keyword window) and
    fa_outside_per_background_hour (pure background: inside no keyword window), the matching `*_ci95` entries (exact
    Poisson 95 % intervals of the counts, divided by the hours; keyword-clip time is excluded for the background rates), and
    detection latency of the true positives (detection time minus event END, ms):
    latency_ms_mean / latency_ms_median / latency_ms_p90 (None without true positives).
    fa_per_hour divides by the WHOLE stream (keywords included); the per-background-hour
    figures divide by the stream minus the keyword events.
    """
    matched_events = set()
    latencies_ms = []
    true_positives = 0
    fa_duplicate = 0
    fa_wrong_class = 0
    fa_outside = 0

    # Event windows [start - 0.2, end + 0.5], indexed by start so each detection only looks at the few events whose
    # window can contain it (same decisions as scanning every event; tests/test_stream_rule.py keeps the full scan).
    starts_w = [start_s - 0.2 for start_s, _, _ in events]
    ends_w = [end_s + 0.5 for _, end_s, _ in events]
    max_len = max((e - s_ for s_, e in zip(starts_w, ends_w)), default=0.0)

    for det_time, det_cls in detections:
        k = bisect.bisect_right(starts_w, det_time)
        cands = []
        j = k - 1
        while j >= 0 and starts_w[j] >= det_time - max_len:
            if det_time <= ends_w[j]:
                cands.append(j)
            j -= 1
        cands.reverse()  # ascending event order, as in a full scan

        match_found = False
        for ev_idx in cands:
            true_cls = events[ev_idx][2]
            if det_cls == true_cls and ev_idx not in matched_events:
                matched_events.add(ev_idx)
                true_positives += 1
                latencies_ms.append((det_time - events[ev_idx][1]) * 1000.0)
                match_found = True
                break

        if match_found:
            continue

        # Duplicate: right class, but that event was already matched
        if any(events[ev_idx][2] == det_cls and ev_idx in matched_events for ev_idx in cands):
            fa_duplicate += 1
            continue

        # Inside at least one event window, but none of the detected class
        if cands:
            fa_wrong_class += 1
        else:
            fa_outside += 1

    total_events = len(events)
    false_negatives = total_events - len(matched_events)
    false_alarms = fa_duplicate + fa_wrong_class + fa_outside
    detection_rate_pct = (true_positives / max(1, total_events)) * 100.0
    fa_per_hour = false_alarms / total_hours if total_hours > 0 else 0.0
    fa_per_hour_excl_duplicates = (fa_wrong_class + fa_outside) / total_hours if total_hours > 0 else 0.0

    background_hours = total_hours - sum(end_s - start_s for start_s, end_s, _ in events) / 3600.0
    fa_per_bg_hour = false_alarms / background_hours if background_hours > 0 else 0.0
    fa_per_bg_hour_excl_dup = (fa_wrong_class + fa_outside) / background_hours if background_hours > 0 else 0.0

    fa_wrong_per_bg_hour = fa_wrong_class / background_hours if background_hours > 0 else 0.0
    ci_fa_hour = poisson_rate_ci(false_alarms, total_hours)
    ci_fa_bg = poisson_rate_ci(false_alarms, background_hours)
    ci_wrong_bg = poisson_rate_ci(fa_wrong_class, background_hours)
    ci_outside_bg = poisson_rate_ci(fa_outside, background_hours)
    fa_outside_per_bg_hour = fa_outside / background_hours if background_hours > 0 else 0.0

    return {
        "events": total_events,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "detection_rate_pct": detection_rate_pct,
        "fa_duplicate": fa_duplicate,
        "fa_wrong_class": fa_wrong_class,
        "fa_outside": fa_outside,
        "false_alarms": false_alarms,
        "fa_per_hour": fa_per_hour,
        "fa_per_hour_excl_duplicates": fa_per_hour_excl_duplicates,
        "total_hours": float(total_hours),
        "background_hours": float(background_hours),
        "fa_per_background_hour": fa_per_bg_hour,
        "fa_per_background_hour_excl_duplicates": fa_per_bg_hour_excl_dup,
        "fa_wrong_class_per_background_hour": fa_wrong_per_bg_hour,
        "fa_outside_per_background_hour": fa_outside_per_bg_hour,
        "fa_per_hour_ci95": ci_fa_hour,
        "fa_per_background_hour_ci95": ci_fa_bg,
        "fa_wrong_class_per_background_hour_ci95": ci_wrong_bg,
        "fa_outside_per_background_hour_ci95": ci_outside_bg,
        "latency_ms_mean": float(np.mean(latencies_ms)) if latencies_ms else None,
        "latency_ms_median": float(np.median(latencies_ms)) if latencies_ms else None,
        "latency_ms_p90": float(np.percentile(latencies_ms, 90)) if latencies_ms else None,
    }

def run_stream_inference(
    lib,
    audio: np.ndarray,
    engine_type: str = "fixed",
    model_f32_weights = None,
) -> List[np.ndarray]:
    """
    Streams audio through the C engine (fixed or float), returning all per-token Q12 logits.
    """
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    if engine_type == "fixed":
        c_weights_q = ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")
        lib.mm_set_weights_q(ctypes.byref(c_weights_q))
    else:
        lib.mm_set_weights_f32(ctypes.byref(model_f32_weights))

    lib.mm_reset(ctypes.byref(ctx))
    all_token_logits = []

    # Stream 512 initial, then 320 hops
    pushed = 0
    total_samples = len(audio)

    # Initial chunk
    chunk0 = np.ascontiguousarray(audio[:512], dtype=np.int16)
    lib.mm_push_audio(ctypes.byref(ctx), chunk0.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(chunk0))
    pushed = 512

    logits_buf_q = (ctypes.c_int32 * 12)()
    logits_buf_f = (ctypes.c_float * 12)()

    if engine_type == "fixed":
        if lib.mm_step_q(ctypes.byref(ctx), None, logits_buf_q):
            all_token_logits.append(np.array(logits_buf_q, dtype=np.int32).copy())
    else:
        if lib.mm_step_f32(ctypes.byref(ctx), None, logits_buf_f):
            arr_f = np.array(logits_buf_f, dtype=np.float32)
            all_token_logits.append((arr_f * 4096.0).astype(np.int32))

    num_hops = max(0, (total_samples - 512) // 320)
    pbar = tqdm(total=num_hops, desc="Streaming inference")
    while pushed + 320 <= total_samples:
        chunk = np.ascontiguousarray(audio[pushed : pushed + 320], dtype=np.int16)
        p = lib.mm_push_audio(ctypes.byref(ctx), chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(chunk))
        if engine_type == "fixed":
            if lib.mm_step_q(ctypes.byref(ctx), None, logits_buf_q):
                all_token_logits.append(np.array(logits_buf_q, dtype=np.int32).copy())
        else:
            if lib.mm_step_f32(ctypes.byref(ctx), None, logits_buf_f):
                arr_f = np.array(logits_buf_f, dtype=np.float32)
                all_token_logits.append((arr_f * 4096.0).astype(np.int32))
        if p < len(chunk):
            rem = np.ascontiguousarray(chunk[p:], dtype=np.int16)
            lib.mm_push_audio(ctypes.byref(ctx), rem.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(rem))
        pushed += 320
        pbar.update(1)
    pbar.close()

    return all_token_logits

def evaluate_decision_rule(
    lib,
    token_logits: List[np.ndarray],
    events: List[Tuple[float, float, int]],
    total_hours: float,
    threshold: float = 1.5,
    ema_alpha: float = 0.93,
    refractory_tokens: int = 75,
) -> Dict[str, float]:
    """
    Runs the exact shipped C decision rule mm_decide_ex over precomputed token logits.
    """
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    lib.mm_reset(ctypes.byref(ctx))

    # Same fixed-point conversions as the C defaults (mm_decide: 6144 = 1.5 * 4096, 30474 = round(0.93 * 32768))
    th_q12 = int(round(threshold * 4096))
    alpha_q15 = int(round(ema_alpha * 32768))

    detections = []
    hop = 320
    frame_len = 512

    for t, l_q in enumerate(token_logits):
        c_logits = (ctypes.c_int32 * 12)(*l_q)
        dec = lib.mm_decide_ex(ctypes.byref(ctx), c_logits, th_q12, alpha_q15, refractory_tokens)
        if dec >= 0:
            time_s = (t * hop + frame_len) / 16000.0
            detections.append((time_s, dec))

    return score_detections(detections, events, total_hours)

def smoothed_margins(lib, token_logits, ema_alpha: float) -> np.ndarray:
    """Per-token EMA of the 10 keyword margins (int32), read from the C rule's own state. Depends only on alpha."""
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    lib.mm_reset(ctypes.byref(ctx))
    alpha_q15 = int(round(ema_alpha * 32768))
    never = 2 ** 31 - 1  # threshold nobody exceeds: only the EMA state is of interest here
    out = np.empty((len(token_logits), 10), dtype=np.int32)
    c_logits = (ctypes.c_int32 * 12)()
    for t, l_q in enumerate(token_logits):
        c_logits[:] = l_q
        lib.mm_decide_ex(ctypes.byref(ctx), c_logits, never, alpha_q15, 0)
        out[t] = ctx.smoothed_margins[:10]
    return out

def detections_from_margins(sm: np.ndarray, threshold: float, refractory_tokens: int) -> List[Tuple[float, int]]:
    """
    Mirror of mm_decide_ex's threshold + refractory logic on precomputed smoothed margins: a token fires when its best
    keyword margin is strictly above the threshold (first maximum wins), then the next `refractory_tokens` tokens are
    suppressed. Equivalent to calling the C rule token by token (see tests/test_stream_rule.py).
    """
    th_q12 = int(round(threshold * 4096))
    best = sm.max(axis=1)
    cand = np.flatnonzero(best > th_q12)
    arg = sm.argmax(axis=1)
    dets = []
    i = 0
    while i < len(cand):
        t = int(cand[i])
        dets.append(((t * 320 + 512) / 16000.0, int(arg[t])))
        i = int(np.searchsorted(cand, t + refractory_tokens + 1))
    return dets

def build_eval_stream(dataset: SpeechCommandsDataset, num_keywords: int, stream_seed: int):
    """Continuous evaluation stream: (audio, events, total_hours, composition info)."""
    info = {}
    audio, _, events = build_continuous_stream(
        dataset,
        num_keywords=num_keywords,
        is_training=False,
        rng=random.Random(stream_seed),
        with_targets=False,
        info=info,
    )
    info.pop("timeline", None)
    total_hours = len(audio) / (16000.0 * 3600.0)
    print(f"Constructed {dataset.split} stream: {len(audio)/16000.0:.1f}s ({total_hours*60:.2f} min, {total_hours:.2f} hrs), "
          f"{len(events)} keyword events, {info['n_unknown_distractors']} unknown distractors")
    return audio, events, total_hours, info

def load_float_weights(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})
    model = MuMambaKWS(
        in_dim=40,
        d_model=args_dict.get("d_model", 32),
        d_inner=args_dict.get("d_inner", 64),
        d_state=args_dict.get("d_state", 8),
        dt_rank=args_dict.get("dt_rank", 4),
        d_conv=args_dict.get("d_conv", 4),
        num_layers=args_dict.get("num_layers", 2),
        num_classes=12,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return populate_c_weights(model, stats=ckpt.get("stats"))

TUNE_GRID = [round(0.25 * i, 2) for i in range(1, 41)]  # thresholds 0.25 .. 10.0

_GRID_ROW_KEYS = (
    "detection_rate_pct", "true_positives", "false_alarms", "fa_duplicate", "fa_wrong_class", "fa_outside",
    "fa_per_hour", "fa_per_background_hour", "fa_wrong_class_per_background_hour", "fa_outside_per_background_hour",
    "fa_per_background_hour_ci95", "latency_ms_median",
)

def tune_operating_point(lib, token_logits, events, total_hours, fa_budget, alphas, refractories):
    """
    Picks (alpha, refractory, threshold) on a validation stream under a false-alarm budget: among the settings with
    fa_per_background_hour <= fa_budget, the one with the highest detection (ties -> fewer false alarms, then lower
    threshold). If no setting fits the budget, "budget not reachable": the setting with the fewest false alarms
    (ties -> higher detection). The EMA depends only on alpha, so the margins are computed once per alpha.
    Returns (best_row, budget_reachable, rows).
    """
    rows = []
    for alpha in alphas:
        sm = smoothed_margins(lib, token_logits, alpha)
        for refr in refractories:
            for th in TUNE_GRID:
                score = score_detections(detections_from_margins(sm, th, refr), events, total_hours)
                row = {"ema_alpha": alpha, "refractory_tokens": refr, "threshold": th}
                row.update({k: score[k] for k in _GRID_ROW_KEYS})
                rows.append(row)
    best, reachable = select_setting(rows, fa_budget)
    return best, reachable, rows

def select_setting(rows, fa_budget):
    """False-alarm-budget rule over grid rows. Returns (best_row, budget_reachable)."""
    fits = [r for r in rows if r["fa_per_background_hour"] <= fa_budget]
    if fits:
        best = min(fits, key=lambda r: (-r["detection_rate_pct"], r["fa_per_background_hour"], r["threshold"]))
    else:
        best = min(rows, key=lambda r: (r["fa_per_background_hour"], -r["detection_rate_pct"], r["threshold"]))
    return best, bool(fits)

def _print_result(title, result):
    print(f"\n=== {title} ===")
    for key, val in result.items():
        if isinstance(val, float):
            print(f"  {key + ':':<40} {val:.2f}")
        else:
            print(f"  {key + ':':<40} {val}")

def evaluate_streaming(
    ckpt_path: Path,
    dataset: SpeechCommandsDataset,
    engine_type: str = "fixed",
    threshold: float = 1.5,
    ema_alpha: float = 0.93,
    refractory_tokens: int = 75,
    num_keywords: int = 2000,
    sweep: bool = False,
    stream_seed: int = 42,
    tune_dataset: SpeechCommandsDataset | None = None,
    fa_budget: float = 100.0,
    tune_alphas: Sequence[float] = (0.85, 0.93, 0.96, 0.98),
    tune_refractories: Sequence[int] = (50, 75, 100, 150),
):
    """
    With `tune_dataset` (the validation split) alpha, refractory and threshold are selected per engine on a validation
    stream built with the same seed and size, under a false-alarm budget (false alarms per background hour), then a
    single run on `dataset` (the test stream) uses them.
    """
    lib = load_engine()
    lib.mm_decide_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int16]
    lib.mm_decide_ex.restype = ctypes.c_int

    engines = ["fixed", "float"] if engine_type == "both" else [engine_type]

    model_f32_weights = None
    if "float" in engines and ckpt_path.exists():
        model_f32_weights = load_float_weights(ckpt_path)
    if "float" in engines and model_f32_weights is None:
        raise FileNotFoundError(f"Float engine needs the checkpoint, {ckpt_path} does not exist")

    print(f"Constructing continuous evaluation stream with {num_keywords} keywords...")
    audio, events, total_hours, composition = build_eval_stream(dataset, num_keywords, stream_seed)

    run_params = {
        "engine": engine_type,
        "split": dataset.split,
        "threshold": threshold,
        "ema_alpha": ema_alpha,
        "refractory_tokens": refractory_tokens,
        "num_keywords": num_keywords,
        "stream_seed": stream_seed,
        "composition": composition,
    }

    if sweep:
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

        print(f"\n================ STREAMING THRESHOLD SWEEP ================")
        results = {}
        for eng in engines:
            print(f"\n--- Engine: {eng.upper()} ---")
            print(f"{'Threshold':>10} | {'Det %':>7} | {'FA/hr':>7} | {'FA/bg-hr':>9} | {'subst/bg-hr':>11} | {'bgnd/bg-hr':>10} | {'dup':>5} | {'wrong':>6} | {'outside':>7}")
            print("-" * 96)
            token_logits = run_stream_inference(lib, audio, engine_type=eng, model_f32_weights=model_f32_weights)
            results[eng] = []
            for th in thresholds:
                score = evaluate_decision_rule(
                    lib, token_logits, events, total_hours,
                    threshold=th, ema_alpha=ema_alpha, refractory_tokens=refractory_tokens
                )
                row = dict(score)
                row["engine"] = eng
                row["threshold"] = th
                results[eng].append(row)
                print(f"{th:>10.2f} | {score['detection_rate_pct']:>6.1f}% | {score['fa_per_hour']:>7.2f} | {score['fa_per_background_hour']:>9.2f} | {score['fa_wrong_class_per_background_hour']:>11.2f} | {score['fa_outside_per_background_hour']:>10.2f} | {score['fa_duplicate']:>5} | {score['fa_wrong_class']:>6} | {score['fa_outside']:>7}")
        print("===========================================================\n")
        return {"run_parameters": run_params, "engines": results}

    tuned = {}
    if tune_dataset is not None:
        print(f"\nTuning on the {tune_dataset.split} stream (false-alarm budget {fa_budget:.1f} per background hour; "
              f"alpha {list(tune_alphas)}, refractory {list(tune_refractories)})...")
        val_audio, val_events, val_hours, val_comp = build_eval_stream(tune_dataset, num_keywords, stream_seed)
        for eng in engines:
            val_logits = run_stream_inference(lib, val_audio, engine_type=eng, model_f32_weights=model_f32_weights)
            best, reachable, rows = tune_operating_point(lib, val_logits, val_events, val_hours, fa_budget,
                                                         tune_alphas, tune_refractories)
            tuned[eng] = {
                "ema_alpha": best["ema_alpha"],
                "refractory_tokens": best["refractory_tokens"],
                "threshold": best["threshold"],
                "fa_budget_per_background_hour": fa_budget,
                "budget_reachable": reachable,
                "validation_detection_rate_pct": best["detection_rate_pct"],
                "validation_fa_per_background_hour": best["fa_per_background_hour"],
                "validation_fa_wrong_class_per_background_hour": best["fa_wrong_class_per_background_hour"],
                "validation_fa_outside_per_background_hour": best["fa_outside_per_background_hour"],
                "validation_composition": val_comp,
                "validation_grid": rows,
            }
            print(f"  {eng}: alpha {best['ema_alpha']}, refractory {best['refractory_tokens']}, threshold {best['threshold']} "
                  f"(validation detection {best['detection_rate_pct']:.1f}%, {best['fa_per_background_hour']:.2f} FA per "
                  f"background hour of which substitutions {best['fa_wrong_class_per_background_hour']:.2f} and pure background "
                  f"{best['fa_outside_per_background_hour']:.2f}; budget reachable: {reachable})")
        del val_audio

    out = {}
    for eng in engines:
        th = tuned[eng]["threshold"] if eng in tuned else threshold
        alpha_used = tuned[eng]["ema_alpha"] if eng in tuned else ema_alpha
        refr_used = tuned[eng]["refractory_tokens"] if eng in tuned else refractory_tokens
        print(f"Running streaming inference ({eng} engine)...")
        token_logits = run_stream_inference(lib, audio, engine_type=eng, model_f32_weights=model_f32_weights)
        result = evaluate_decision_rule(
            lib, token_logits, events, total_hours,
            threshold=th, ema_alpha=alpha_used, refractory_tokens=refr_used
        )
        result.update(run_params)
        result["engine"] = eng
        result["threshold"] = th
        result["ema_alpha"] = alpha_used
        result["refractory_tokens"] = refr_used
        if eng in tuned:
            result["threshold_tuned_on_validation"] = tuned[eng]
        _print_result(f"Streaming Evaluation ({eng.upper()})", result)
        out[eng] = result
    if engine_type == "both":
        return {"run_parameters": run_params, "engines": out}
    return out[engine_type]

def main():
    parser = argparse.ArgumentParser(description="Continuous Streaming Evaluation with Shipped C Decision Rule")
    parser.add_argument("--ckpt", type=str, default="checkpoints/quant_model.pt")
    parser.add_argument("--engine", choices=["float", "fixed", "both"], default="fixed")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--threshold", type=float, default=1.5, help="Fixed threshold (ignored with --tune-on-validation)")
    parser.add_argument("--ema-alpha", type=float, default=0.93)
    parser.add_argument("--refractory-tokens", type=int, default=75)
    parser.add_argument("--num-keywords", type=int, default=2000)
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="Report the whole threshold curve on --split (a curve, not an operating point)")
    parser.add_argument("--tune-on-validation", action="store_true",
                        help="Select alpha, refractory and threshold per engine on a VALIDATION stream under a false-alarm "
                             "budget (--fa-budget), then run one --split stream with them. Report this, not a sweep.")
    parser.add_argument("--fa-budget", type=float, default=100.0,
                        help="False alarms per BACKGROUND hour (substitutions included) allowed when tuning on validation")
    parser.add_argument("--tune-grid-alpha", type=float, nargs="+", default=[0.85, 0.93, 0.96, 0.98])
    parser.add_argument("--tune-grid-refractory", type=int, nargs="+", default=[50, 75, 100, 150])
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON report")
    args = parser.parse_args()
    if args.tune_on_validation and args.sweep:
        parser.error("--tune-on-validation and --sweep are mutually exclusive")
    if args.tune_on_validation and args.split == "validation":
        parser.error("--tune-on-validation needs --split test (tuning and reporting on the same stream is not allowed)")

    dataset = SpeechCommandsDataset(data_dir=args.data_dir, split=args.split, augment=False)
    tune_dataset = SpeechCommandsDataset(data_dir=args.data_dir, split="validation", augment=False) if args.tune_on_validation else None
    result = evaluate_streaming(
        ckpt_path=Path(args.ckpt),
        dataset=dataset,
        engine_type=args.engine,
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        refractory_tokens=args.refractory_tokens,
        num_keywords=args.num_keywords,
        sweep=args.sweep,
        stream_seed=args.stream_seed,
        tune_dataset=tune_dataset,
        fa_budget=args.fa_budget,
        tune_alphas=args.tune_grid_alpha,
        tune_refractories=args.tune_grid_refractory,
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote JSON report to {args.json}")

if __name__ == "__main__":
    main()
