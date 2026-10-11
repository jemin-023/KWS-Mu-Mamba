import argparse
import random
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F

from train.kws.data import SpeechCommandsDataset, TARGET_KEYWORDS, CLASS_TO_IDX, IDX_TO_CLASS, build_continuous_stream
from train.kws.eval_c import load_engine, MMContext, ModelWeightsQ, populate_c_weights
from train.kws.model import MuMambaKWS

def score_detections(
    detections: List[Tuple[float, int]],
    events: List[Tuple[float, float, int]],
    total_hours: float,
) -> Tuple[float, float, int, int, int]:
    """
    Matches detections to ground-truth keyword events.
    Returns:
      detection_acc (%), fa_per_hour, true_positives, false_alarms, false_negatives
    """
    matched_events = set()
    true_positives = 0
    false_alarms = 0

    for det_time, det_cls in detections:
        match_found = False
        for ev_idx, (start_s, end_s, true_cls) in enumerate(events):
            # Keyword duration tolerance: detected between start-0.2s and end+0.5s
            if start_s - 0.2 <= det_time <= end_s + 0.5:
                if det_cls == true_cls and ev_idx not in matched_events:
                    matched_events.add(ev_idx)
                    true_positives += 1
                    match_found = True
                    break

        if not match_found:
            false_alarms += 1

    total_events = len(events)
    false_negatives = total_events - len(matched_events)
    detection_acc = (true_positives / max(1, total_events)) * 100.0
    fa_per_hour = false_alarms / total_hours if total_hours > 0 else 0.0
    return detection_acc, fa_per_hour, true_positives, false_alarms, false_negatives

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
    chunk0 = audio[:512]
    c_chunk0 = (ctypes.c_int16 * len(chunk0))(*chunk0)
    lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
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

    while pushed + 320 <= total_samples:
        chunk = audio[pushed : pushed + 320]
        c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
        p = lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
        if engine_type == "fixed":
            if lib.mm_step_q(ctypes.byref(ctx), None, logits_buf_q):
                all_token_logits.append(np.array(logits_buf_q, dtype=np.int32).copy())
        else:
            if lib.mm_step_f32(ctypes.byref(ctx), None, logits_buf_f):
                arr_f = np.array(logits_buf_f, dtype=np.float32)
                all_token_logits.append((arr_f * 4096.0).astype(np.int32))
        if p < len(chunk):
            rem = chunk[p:]
            c_rem = (ctypes.c_int16 * len(rem))(*rem)
            lib.mm_push_audio(ctypes.byref(ctx), c_rem, len(rem))
        pushed += 320

    return all_token_logits

def evaluate_decision_rule(
    lib,
    token_logits: List[np.ndarray],
    events: List[Tuple[float, float, int]],
    total_hours: float,
    threshold: float = 1.5,
    ema_alpha: float = 0.7,
    refractory_tokens: int = 25,
) -> Tuple[float, float]:
    """
    Runs the exact shipped C decision rule mm_decide_ex over precomputed token logits.
    """
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    lib.mm_reset(ctypes.byref(ctx))

    th_q12 = int(threshold * 4096)
    alpha_q15 = int(ema_alpha * 32768)

    detections = []
    hop = 320
    frame_len = 512

    for t, l_q in enumerate(token_logits):
        c_logits = (ctypes.c_int32 * 12)(*l_q)
        dec = lib.mm_decide_ex(ctypes.byref(ctx), c_logits, th_q12, alpha_q15, refractory_tokens)
        if dec >= 0:
            time_s = (t * hop + frame_len) / 16000.0
            detections.append((time_s, dec))

    det_acc, fa_per_hour, tp, fa, fn = score_detections(detections, events, total_hours)
    return det_acc, fa_per_hour

def evaluate_streaming(
    ckpt_path: Path,
    dataset: SpeechCommandsDataset,
    engine_type: str = "fixed",
    threshold: float = 1.5,
    ema_alpha: float = 0.7,
    refractory_tokens: int = 25,
    num_keywords: int = 50,
    sweep: bool = False,
):
    lib = load_engine()
    lib.mm_decide_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int16]
    lib.mm_decide_ex.restype = ctypes.c_int

    model_f32_weights = None
    if engine_type == "float" or sweep:
        if ckpt_path.exists():
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
            model_f32_weights = populate_c_weights(model, stats=ckpt.get("stats"))

    print(f"Constructing continuous evaluation stream with {num_keywords} keywords...")
    audio, _, events = build_continuous_stream(
        dataset,
        num_keywords=num_keywords,
        is_training=False,
        rng=random.Random(42),
    )
    total_hours = len(audio) / (16000.0 * 3600.0)
    print(f"Constructed stream: {len(audio)/16000.0:.1f}s ({total_hours*60:.2f} min), {len(events)} keyword events")

    if not sweep:
        print(f"Running streaming inference ({engine_type} engine)...")
        token_logits = run_stream_inference(lib, audio, engine_type=engine_type, model_f32_weights=model_f32_weights)
        det_acc, fa_per_hour = evaluate_decision_rule(
            lib, token_logits, events, total_hours,
            threshold=threshold, ema_alpha=ema_alpha, refractory_tokens=refractory_tokens
        )
        print(f"\n=== Streaming Evaluation ({engine_type.upper()}) ===")
        print(f"Threshold:         {threshold:.2f} (margin in Q12)")
        print(f"Detection Rate:    {det_acc:.2f}%")
        print(f"False Alarms/Hour: {fa_per_hour:.2f} FA/hr")
        return det_acc, fa_per_hour
    else:
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        engines = ["fixed", "float"] if engine_type == "both" else [engine_type]

        print(f"\n================ STREAMING THRESHOLD SWEEP ================")
        for eng in engines:
            print(f"\n--- Engine: {eng.upper()} ---")
            print(f"{'Threshold':>10} | {'Detection Rate':>14} | {'False Alarms/hr':>15}")
            print("-" * 46)
            token_logits = run_stream_inference(lib, audio, engine_type=eng, model_f32_weights=model_f32_weights)
            for th in thresholds:
                det_acc, fa_per_hour = evaluate_decision_rule(
                    lib, token_logits, events, total_hours,
                    threshold=th, ema_alpha=ema_alpha, refractory_tokens=refractory_tokens
                )
                print(f"{th:>10.2f} | {det_acc:>13.1f}% | {fa_per_hour:>14.2f}")
        print("===========================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Continuous Streaming Evaluation with Shipped C Decision Rule")
    parser.add_argument("--ckpt", type=str, default="checkpoints/quant_model.pt")
    parser.add_argument("--engine", choices=["float", "fixed", "both"], default="fixed")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--ema-alpha", type=float, default=0.7)
    parser.add_argument("--refractory-tokens", type=int, default=25)
    parser.add_argument("--num-keywords", type=int, default=50)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    dataset = SpeechCommandsDataset(data_dir=args.data_dir, split="test", augment=False)
    evaluate_streaming(
        ckpt_path=Path(args.ckpt),
        dataset=dataset,
        engine_type=args.engine,
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        refractory_tokens=args.refractory_tokens,
        num_keywords=args.num_keywords,
        sweep=args.sweep,
    )

if __name__ == "__main__":
    main()
