import argparse
import ctypes
import hashlib
import json
import math
import random
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from train.kws.data import SpeechCommandsDataset, NUM_CLASSES, IDX_TO_CLASS, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.model import MuMambaKWS


def select_eval_indices(total: int, subset: int | None, seed: int = 0) -> list[int]:
    """All indices if subset is None or >= total; otherwise a uniformly random sample of `subset` indices drawn without replacement with random.Random(seed), returned sorted ascending."""
    if subset is None or subset >= total:
        return list(range(total))
    rng = random.Random(seed)
    indices = rng.sample(range(total), subset)
    indices.sort()
    return indices


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, returned as (low, high) in percent. total == 0 -> (0.0, 100.0)."""
    if total == 0:
        return (0.0, 100.0)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    low = max(0.0, centre - half) * 100.0
    high = min(1.0, centre + half) * 100.0
    return (low, high)


class BlockWeightsF32(ctypes.Structure):
    _fields_ = [
        ("norm_w", ctypes.c_float * 32),
        ("in_proj_w", (ctypes.c_float * 32) * 128),
        ("conv1d_w", (ctypes.c_float * 4) * 64),
        ("conv1d_b", ctypes.c_float * 64),
        ("x_proj_w", (ctypes.c_float * 64) * 20),
        ("dt_proj_w", (ctypes.c_float * 4) * 64),
        ("dt_proj_b", ctypes.c_float * 64),
        ("D_skip", ctypes.c_float * 64),
        ("out_proj_w", (ctypes.c_float * 64) * 32),
    ]

class ModelWeightsF32(ctypes.Structure):
    _fields_ = [
        ("stem_w", (ctypes.c_float * 40) * 32),
        ("stem_b", ctypes.c_float * 32),
        ("blocks", BlockWeightsF32 * 2),
        ("final_norm_w", ctypes.c_float * 32),
        ("head_w", (ctypes.c_float * 32) * 12),
        ("head_b", ctypes.c_float * 12),
    ]

class MMContext(ctypes.Structure):
    _fields_ = [
        ("audio_buf", ctypes.c_int16 * 512),
        ("audio_buf_count", ctypes.c_uint16),
        ("mel_features", ctypes.c_int16 * 40),
        ("conv_ring_q", ((ctypes.c_int16 * 64) * 3) * 2),
        ("ssm_h_q", ((ctypes.c_int32 * 8) * 64) * 2),
        ("logits_q", ctypes.c_int32 * 12),
        ("smoothed_margins", ctypes.c_int32 * 12),
        ("refractory_timer", ctypes.c_int16),
        ("conv_ring_f", ((ctypes.c_float * 64) * 3) * 2),
        ("ssm_h_f", ((ctypes.c_float * 8) * 64) * 2),
        ("logits_f", ctypes.c_float * 12),
    ]

class BlockWeightsQ(ctypes.Structure):
    _fields_ = [
        ("norm_w", ctypes.c_int16 * 32),
        ("in_proj_w", (ctypes.c_int8 * 32) * 128),
        ("in_proj_scale_mult", ctypes.c_int16 * 128),
        ("in_proj_scale_shift", ctypes.c_uint8 * 128),
        ("conv1d_w", (ctypes.c_int8 * 4) * 64),
        ("conv1d_b", ctypes.c_int32 * 64),
        ("conv1d_scale_mult", ctypes.c_int16 * 64),
        ("conv1d_scale_shift", ctypes.c_uint8 * 64),
        ("x_proj_w", (ctypes.c_int8 * 64) * 20),
        ("x_proj_scale_mult", ctypes.c_int16 * 20),
        ("x_proj_scale_shift", ctypes.c_uint8 * 20),
        ("dt_proj_w", (ctypes.c_int8 * 4) * 64),
        ("dt_proj_b", ctypes.c_int32 * 64),
        ("dt_proj_scale_mult", ctypes.c_int16 * 64),
        ("dt_proj_scale_shift", ctypes.c_uint8 * 64),
        ("D_skip", ctypes.c_int16 * 64),
        ("out_proj_w", (ctypes.c_int8 * 64) * 32),
        ("out_proj_scale_mult", ctypes.c_int16 * 32),
        ("out_proj_scale_shift", ctypes.c_uint8 * 32),
    ]

class ModelWeightsQ(ctypes.Structure):
    _fields_ = [
        ("stem_w", (ctypes.c_int8 * 40) * 32),
        ("stem_b", ctypes.c_int32 * 32),
        ("stem_scale_mult", ctypes.c_int16 * 32),
        ("stem_scale_shift", ctypes.c_uint8 * 32),
        ("blocks", BlockWeightsQ * 2),
        ("final_norm_w", ctypes.c_int16 * 32),
        ("head_w", (ctypes.c_int8 * 32) * 12),
        ("head_b", ctypes.c_int32 * 12),
        ("head_scale_mult", ctypes.c_int16 * 12),
        ("head_scale_shift", ctypes.c_uint8 * 12),
    ]

class MMSatStats(ctypes.Structure):
    _fields_ = [
        ("stem_total", ctypes.c_uint64),
        ("stem_sat", ctypes.c_uint64),
        ("in_proj_total", ctypes.c_uint64),
        ("in_proj_sat", ctypes.c_uint64),
        ("conv_total", ctypes.c_uint64),
        ("conv_sat", ctypes.c_uint64),
        ("x_proj_total", ctypes.c_uint64),
        ("x_proj_sat", ctypes.c_uint64),
        ("resid_total", ctypes.c_uint64),
        ("resid_sat", ctypes.c_uint64),
        ("h_total", ctypes.c_uint64),
        ("h_sat", ctypes.c_uint64),
        ("norm_out_total", ctypes.c_uint64),
        ("norm_out_sat", ctypes.c_uint64),
        ("y_total", ctypes.c_uint64),
        ("y_sat", ctypes.c_uint64),
        ("y_gate_total", ctypes.c_uint64),
        ("y_gate_sat", ctypes.c_uint64),
        ("gate_z_total", ctypes.c_uint64),
        ("gate_z_sat", ctypes.c_uint64),
        ("p_clamp_total", ctypes.c_uint64),
        ("p_clamp_sat", ctypes.c_uint64),
        ("logits_total", ctypes.c_uint64),
        ("logits_sat", ctypes.c_uint64),
    ]

def load_engine():
    lib_path = Path("engine/libmumamba.so").resolve()
    if not lib_path.exists():
        raise FileNotFoundError(f"{lib_path} does not exist. Run 'make -C engine' first.")
    lib = ctypes.CDLL(str(lib_path))

    lib.mm_context_size.argtypes = []
    lib.mm_context_size.restype = ctypes.c_size_t

    assert ctypes.sizeof(MMContext) == lib.mm_context_size(), (
        f"MMContext size mismatch: python={ctypes.sizeof(MMContext)}, C={lib.mm_context_size()}"
    )

    lib.mm_init.argtypes = [ctypes.POINTER(MMContext)]
    lib.mm_init.restype = None

    lib.mm_reset.argtypes = [ctypes.POINTER(MMContext)]
    lib.mm_reset.restype = None

    lib.mm_set_weights_f32.argtypes = [ctypes.POINTER(ModelWeightsF32)]
    lib.mm_set_weights_f32.restype = None

    lib.mm_set_weights_q.argtypes = [ctypes.POINTER(ModelWeightsQ)]
    lib.mm_set_weights_q.restype = None

    lib.mm_push_audio.argtypes = [
        ctypes.POINTER(MMContext),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
    ]
    lib.mm_push_audio.restype = ctypes.c_size_t

    lib.mm_step_f32.argtypes = [
        ctypes.POINTER(MMContext),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.mm_step_f32.restype = ctypes.c_int

    lib.mm_step_q.argtypes = [
        ctypes.POINTER(MMContext),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.mm_step_q.restype = ctypes.c_int

    lib.mm_weights_id.argtypes = []
    lib.mm_weights_id.restype = ctypes.c_char_p

    # The ctypes weight structs hard-code the model dimensions: refuse to run on a layout mismatch
    lib.mm_weights_q_size.argtypes = []
    lib.mm_weights_q_size.restype = ctypes.c_size_t
    assert ctypes.sizeof(ModelWeightsQ) == lib.mm_weights_q_size(), (
        f"ModelWeightsQ size mismatch: python={ctypes.sizeof(ModelWeightsQ)}, C={lib.mm_weights_q_size()}"
    )
    if hasattr(lib, "mm_weights_f32_size"):
        lib.mm_weights_f32_size.argtypes = []
        lib.mm_weights_f32_size.restype = ctypes.c_size_t
        assert ctypes.sizeof(ModelWeightsF32) == lib.mm_weights_f32_size(), (
            f"ModelWeightsF32 size mismatch: python={ctypes.sizeof(ModelWeightsF32)}, C={lib.mm_weights_f32_size()}"
        )

    lib.mm_export_id.argtypes = []
    lib.mm_export_id.restype = ctypes.c_char_p

    lib.mm_get_sat_stats.argtypes = [ctypes.POINTER(MMSatStats)]
    lib.mm_get_sat_stats.restype = None

    lib.mm_reset_sat_stats.argtypes = []
    lib.mm_reset_sat_stats.restype = None

    return lib

def get_ckpt_id(ckpt_path: Path) -> str:
    if not ckpt_path.exists():
        return "unknown"
    with open(ckpt_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

def check_weights_id(lib, ckpt_path: Path):
    if not ckpt_path.exists():
        return
    expected_id = get_ckpt_id(ckpt_path)
    engine_id = lib.mm_weights_id().decode("utf-8")
    if engine_id != expected_id:
        print(f"ERROR: C engine weights ID '{engine_id}' does not match checkpoint ID '{expected_id}'.")
        print(f"Refusing to run. Please run 'make export' to regenerate C weights from {ckpt_path}.")
        sys.exit(1)

def populate_c_weights(model: MuMambaKWS, stats: dict = None) -> ModelWeightsF32:
    w = ModelWeightsF32()

    # Stem weights
    stem_w = model.stem.weight.detach().cpu().numpy().copy() # (32, 40)
    stem_b = model.stem.bias.detach().cpu().numpy().copy()   # (32,)

    # If trained with normalized features (mel - mean) / std, fold normalization into stem:
    if stats is not None and "mean" in stats and "std" in stats:
        mean_val = stats["mean"].detach().cpu().numpy() if hasattr(stats["mean"], "detach") else np.array(stats["mean"])
        std_val = stats["std"].detach().cpu().numpy() if hasattr(stats["std"], "detach") else np.array(stats["std"])
        mean = np.array(mean_val, dtype=np.float32)
        std = np.array(std_val, dtype=np.float32)
        stem_w_folded = stem_w / std[np.newaxis, :]
        stem_b_folded = stem_b - np.sum(stem_w_folded * mean[np.newaxis, :], axis=1)
        stem_w = stem_w_folded
        stem_b = stem_b_folded

    for d in range(32):
        w.stem_b[d] = float(stem_b[d])
        for m in range(40):
            w.stem_w[d][m] = float(stem_w[d, m])

    # Blocks
    for l_idx, layer in enumerate(model.layers):
        bw = w.blocks[l_idx]
        norm_w = layer.norm.weight.detach().cpu().numpy()
        for d in range(32):
            bw.norm_w[d] = float(norm_w[d])

        in_proj_w = layer.in_proj.weight.detach().cpu().numpy()
        for i in range(128):
            for d in range(32):
                bw.in_proj_w[i][d] = float(in_proj_w[i, d])

        conv_w = layer.conv1d.weight.detach().squeeze(1).cpu().numpy()
        conv_b = layer.conv1d.bias.detach().cpu().numpy()
        for d in range(64):
            bw.conv1d_b[d] = float(conv_b[d])
            for k in range(4):
                bw.conv1d_w[d][k] = float(conv_w[d, k])

        x_proj_w = layer.x_proj.weight.detach().cpu().numpy()
        for i in range(20):
            for d in range(64):
                bw.x_proj_w[i][d] = float(x_proj_w[i, d])

        dt_proj_w = layer.dt_proj.weight.detach().cpu().numpy()
        dt_proj_b = layer.dt_proj.bias.detach().cpu().numpy()
        for d in range(64):
            bw.dt_proj_b[d] = float(dt_proj_b[d])
            for r in range(4):
                bw.dt_proj_w[d][r] = float(dt_proj_w[d, r])

        for d in range(64):
            bw.D_skip[d] = float(layer.D_skip[d].detach().cpu())

        out_proj_w = layer.out_proj.weight.detach().cpu().numpy()
        for o in range(32):
            for d in range(64):
                bw.out_proj_w[o][d] = float(out_proj_w[o, d])

    # Final norm & head
    final_norm_w = model.final_norm.weight.detach().cpu().numpy()
    for d in range(32):
        w.final_norm_w[d] = float(final_norm_w[d])

    head_w = model.head.weight.detach().cpu().numpy()
    head_b = model.head.bias.detach().cpu().numpy()
    for c in range(12):
        w.head_b[c] = float(head_b[c])
        for d in range(32):
            w.head_w[c][d] = float(head_w[c, d])

    return w

def run_clip_stream(lib, ctx, audio, is_float: bool, last_logits, collect_features: bool = False):
    chunk0 = audio[:512]
    c_chunk0 = (ctypes.c_int16 * len(chunk0))(*chunk0)
    lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
    features = None
    if is_float:
        if collect_features:
            feat0 = (ctypes.c_int16 * 40)()
            lib.mm_step_f32(ctypes.byref(ctx), feat0, last_logits)
            features = [np.array(feat0, dtype=np.int16).copy()]
        else:
            lib.mm_step_f32(ctypes.byref(ctx), None, last_logits)
    else:
        lib.mm_step_q(ctypes.byref(ctx), None, last_logits)

    for t in range(1, 50):
        start = 512 + (t - 1) * 320
        chunk = audio[start : start + 320]
        c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
        lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
        if is_float:
            if collect_features:
                feat = (ctypes.c_int16 * 40)()
                lib.mm_step_f32(ctypes.byref(ctx), feat, last_logits)
                features.append(np.array(feat, dtype=np.int16).copy())
            else:
                lib.mm_step_f32(ctypes.byref(ctx), None, last_logits)
        else:
            lib.mm_step_q(ctypes.byref(ctx), None, last_logits)
    return features

SAT_SITES = [
    ("stem", "Stem"), ("in_proj", "in_proj"), ("conv", "conv"), ("x_proj", "x_proj"), ("resid", "residual"),
    ("h", "h"), ("norm_out", "norm out"), ("y", "y"), ("y_gate", "y_gate"), ("gate_z", "gate z>|8|"),
    ("p_clamp", "p clamp"), ("logits", "logits"),
]

def print_saturation_stats(lib):
    sat = MMSatStats()
    lib.mm_get_sat_stats(ctypes.byref(sat))
    print("\nSaturation rates:")
    for site, label in SAT_SITES:
        n_sat, n_total = getattr(sat, f"{site}_sat"), getattr(sat, f"{site}_total")
        print(f"  {label + ':':<12} {n_sat}/{n_total} ({n_sat/max(1, n_total)*100:.3f}%)")

def sat_stats_dict(lib) -> dict:
    """All saturation counters of the fixed engine as {field: int}, plus a rate in percent per site."""
    sat = MMSatStats()
    lib.mm_get_sat_stats(ctypes.byref(sat))
    out = {name: int(getattr(sat, name)) for name, _ in MMSatStats._fields_}
    for name in list(out):
        if name.endswith("_total"):
            site = name[: -len("_total")]
            out[f"{site}_sat_pct"] = out[f"{site}_sat"] / max(1, out[name]) * 100.0
    return out

def evaluate_c(
    ckpt_path: Path,
    engine_type: str = "both",
    split: str = "test",
    data_dir: str = "data/speech_commands/v0.02",
    subset: int = None,
    seed: int = 0,
    json_path: str | None = None,
    dump_npz: str | None = None,
):
    lib = load_engine()
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))

    if engine_type in ["fixed", "both"]:
        check_weights_id(lib, ckpt_path)

    stats = None
    if ckpt_path.exists():
        print(f"Loading checkpoint {ckpt_path} (ID: {get_ckpt_id(ckpt_path)})...")
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
            num_classes=NUM_CLASSES,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        stats = ckpt.get("stats", None)
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with template initialized model.")
        model = MuMambaKWS()

    if stats is None:
        stats_file = Path("train/kws/frontend_stats.pt")
        if stats_file.exists():
            stats = torch.load(stats_file, weights_only=False)

    model.eval()

    c_weights_f = populate_c_weights(model, stats=stats)
    c_weights_q = ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")

    ds = SpeechCommandsDataset(data_dir=data_dir, split=split, augment=False)
    total_samples = len(ds)
    indices = select_eval_indices(total_samples, subset, seed)
    num_eval = len(indices)

    print(f"Evaluating C engine ({engine_type}) on split '{split}' ({num_eval} clips)...")
    lib.mm_reset_sat_stats()

    correct_f, correct_q, agree = 0, 0, 0
    total = 0
    logit_diffs_fq = []  # Fixed vs Float logit diffs
    logit_diffs_fp = []  # Float C vs PyTorch logit diffs
    agree_fp = 0  # Float C vs PyTorch top-1 agreement
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}
    rec = {"index": [], "label": [], "logits_float": [], "logits_fixed": [], "pred_float": [], "pred_fixed": [], "logits_pytorch": []}

    # Prepare normalization stats for FloatC-vs-PyTorch comparison
    if stats is not None and "mean" in stats and "std" in stats:
        mean_arr = stats["mean"].detach().cpu().numpy() if hasattr(stats["mean"], "detach") else np.array(stats["mean"])
        std_arr = stats["std"].detach().cpu().numpy() if hasattr(stats["std"], "detach") else np.array(stats["std"])
        mean_norm = np.array(mean_arr, dtype=np.float32)
        std_norm = np.array(std_arr, dtype=np.float32)
    else:
        mean_norm = np.zeros(40, dtype=np.float32)
        std_norm = np.ones(40, dtype=np.float32)

    pbar = tqdm(indices, desc=f"C Engine ({engine_type})")
    for idx in pbar:
        fpath, label, silence_idx = ds.epoch_samples[idx]
        if fpath is None:
            if silence_idx is not None and silence_idx < len(ds.val_silence_crops):
                audio = ds.val_silence_crops[silence_idx]
            else:
                audio = ds._get_silence_crop()
        else:
            try:
                audio = read_wav_int16(fpath)
            except Exception:
                audio = np.zeros(TARGET_AUDIO_LEN, dtype=np.int16)

        if len(audio) < TARGET_AUDIO_LEN:
            audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)), mode="constant")
        elif len(audio) > TARGET_AUDIO_LEN:
            audio = audio[:TARGET_AUDIO_LEN]

        audio = np.ascontiguousarray(audio, dtype=np.int16)

        # Float engine
        if engine_type in ["float", "both"]:
            lib.mm_set_weights_f32(ctypes.byref(c_weights_f))
            lib.mm_reset(ctypes.byref(ctx))
            last_logits_f = (ctypes.c_float * 12)()
            features = run_clip_stream(lib, ctx, audio, is_float=True, last_logits=last_logits_f, collect_features=(engine_type == "both"))
            pred_f = int(np.argmax(np.array(last_logits_f, dtype=np.float32)))
            if pred_f == label:
                correct_f += 1

        # Fixed engine
        if engine_type in ["fixed", "both"]:
            lib.mm_set_weights_q(ctypes.byref(c_weights_q))
            lib.mm_reset(ctypes.byref(ctx))
            last_logits_q = (ctypes.c_int32 * 12)()
            run_clip_stream(lib, ctx, audio, is_float=False, last_logits=last_logits_q)
            pred_q = int(np.argmax(np.array(last_logits_q, dtype=np.int32)))
            if pred_q == label:
                correct_q += 1
            class_total[label] = class_total.get(label, 0) + 1
            if pred_q == label:
                class_correct[label] = class_correct.get(label, 0) + 1

        if engine_type == "both":
            if pred_f == pred_q:
                agree += 1
            lf = np.array(last_logits_f, dtype=np.float32)
            lq = np.array(last_logits_q, dtype=np.float32) / 4096.0
            logit_diffs_fq.append(np.abs(lf - lq))

            # Float C vs PyTorch comparison
            # features is a list of 50 (c_int16 * 40) arrays from the C float engine
            if features is not None and len(features) == 50:
                # Normalize features exactly like the dataset does: (mel - mean) / std
                feats_np = np.stack([f.astype(np.float32) for f in features], axis=0)  # (50, 40)
                feats_norm = (feats_np - mean_norm) / std_norm
                feats_tensor = torch.from_numpy(feats_norm).unsqueeze(0)  # (1, 50, 40)
                with torch.no_grad():
                    logits_pt = model.forward_seq(feats_tensor)  # (1, 50, 12)
                    logits_pt_last = logits_pt[0, -1, :].numpy()  # (12,)
                lf_float = np.array(last_logits_f, dtype=np.float32)
                diff_fp = np.abs(lf_float - logits_pt_last)
                logit_diffs_fp.append(diff_fp)
                pred_pt = int(np.argmax(logits_pt_last))
                if pred_f == pred_pt:
                    agree_fp += 1

        rec["index"].append(idx)
        rec["label"].append(label)
        if engine_type in ["float", "both"]:
            rec["logits_float"].append(np.array(last_logits_f, dtype=np.float32))
            rec["pred_float"].append(pred_f)
        if engine_type in ["fixed", "both"]:
            rec["logits_fixed"].append(np.array(last_logits_q, dtype=np.int32))
            rec["pred_fixed"].append(pred_q)
        if engine_type == "both":
            rec["logits_pytorch"].append(logits_pt_last if features is not None and len(features) == 50 else np.full(12, np.nan, dtype=np.float32))

        total += 1
        if engine_type == "both":
            pbar.set_postfix({"f_acc": f"{correct_f/total*100:.1f}%", "q_acc": f"{correct_q/total*100:.1f}%", "agree": f"{agree/total*100:.1f}%"})
        elif engine_type == "float":
            pbar.set_postfix({"acc": f"{correct_f/total*100:.2f}%"})
        else:
            pbar.set_postfix({"acc": f"{correct_q/total*100:.2f}%"})

    if engine_type == "float":
        acc = correct_f / total * 100.0 if total > 0 else 0.0
        lo, hi = wilson_interval(correct_f, total)
        print(f"\nFinal C Engine (float) Accuracy on {split}: {acc:.2f}% ({correct_f}/{total})  95% CI [{lo:.2f}, {hi:.2f}]")
        if json_path:
            _write_json_report(json_path, {
                "engine": "float",
                "split": split,
                "subset": subset,
                "seed": seed,
                "total_clips": total,
                "checkpoint_id": get_ckpt_id(ckpt_path),
                "export_id": lib.mm_export_id().decode("utf-8"),
                "float_correct": correct_f,
                "float_accuracy_pct": acc,
                "float_wilson_ci": [lo, hi],
            })
        if dump_npz:
            _dump_npz(dump_npz, rec)
        return acc
    elif engine_type == "fixed":
        acc = correct_q / total * 100.0 if total > 0 else 0.0
        lo, hi = wilson_interval(correct_q, total)
        print(f"\nFinal C Engine (fixed) Accuracy on {split}: {acc:.2f}% ({correct_q}/{total})  95% CI [{lo:.2f}, {hi:.2f}]")
        print_saturation_stats(lib)
        if json_path:
            _write_json_report(json_path, {
                "engine": "fixed",
                "split": split,
                "subset": subset,
                "seed": seed,
                "total_clips": total,
                "checkpoint_id": get_ckpt_id(ckpt_path),
                "export_id": lib.mm_export_id().decode("utf-8"),
                "fixed_correct": correct_q,
                "fixed_accuracy_pct": acc,
                "fixed_wilson_ci": [lo, hi],
                "saturation": sat_stats_dict(lib),
            })
        if dump_npz:
            _dump_npz(dump_npz, rec)
        return acc
    else:  # both
        acc_f = correct_f / total * 100.0
        acc_q = correct_q / total * 100.0
        drop = acc_f - acc_q
        agreement = agree / total * 100.0
        med_err = float(np.median(logit_diffs_fq))

        # Fixed vs Float logit error stats
        logit_diffs_fq_arr = np.stack(logit_diffs_fq, axis=0)  # (N, 12)
        linf_fq = np.max(logit_diffs_fq_arr, axis=1)
        mean_abs_fq = np.mean(logit_diffs_fq_arr, axis=1)
        avg_linf_fq = float(np.mean(linf_fq))
        avg_mean_abs_fq = float(np.mean(mean_abs_fq))
        worst_linf_fq = float(np.max(linf_fq))

        # Float C vs PyTorch logit error stats
        if logit_diffs_fp:
            logit_diffs_fp_arr = np.stack(logit_diffs_fp, axis=0)  # (N, 12)
            linf_fp = np.max(logit_diffs_fp_arr, axis=1)
            mean_abs_fp = np.mean(logit_diffs_fp_arr, axis=1)
            avg_linf_fp = float(np.mean(linf_fp))
            avg_mean_abs_fp = float(np.mean(mean_abs_fp))
            worst_linf_fp = float(np.max(linf_fp))
            agreement_fp = agree_fp / total * 100.0
        else:
            avg_linf_fp = avg_mean_abs_fp = worst_linf_fp = 0.0
            agreement_fp = 0.0

        # Wilson intervals
        lo_f, hi_f = wilson_interval(correct_f, total)
        lo_q, hi_q = wilson_interval(correct_q, total)

        print(f"\n================ C ENGINE VERIFICATION GATE ================")
        print(f"Float Accuracy:     {acc_f:.2f}% ({correct_f}/{total})  95% CI [{lo_f:.2f}, {hi_f:.2f}]")
        print(f"Fixed Accuracy:     {acc_q:.2f}% ({correct_q}/{total})  95% CI [{lo_q:.2f}, {hi_q:.2f}]")
        print(f"Accuracy Drop:      {drop:+.2f} pt (gate: <= 0.50 pt)")
        print(f"Top-1 Agreement:    {agreement:.2f}% (gate: >= 99.00%)")
        print(f"Median Logit Error: {med_err:.4f}")
        print(f"Fixed-vs-Float  avg Linf: {avg_linf_fq:.3e}   avg mean-abs: {avg_mean_abs_fq:.3e}   worst Linf: {worst_linf_fq:.3e}")
        print(f"FloatC-vs-PyTorch avg Linf: {avg_linf_fp:.3e}   avg mean-abs: {avg_mean_abs_fp:.3e}   worst Linf: {worst_linf_fp:.3e}")
        print(f"FloatC-vs-PyTorch top-1 agreement: {agreement_fp:.2f}% ({agree_fp}/{total})")
        print_saturation_stats(lib)

        # Per-class accuracy table for fixed engine
        print(f"\n--- Per-Class Accuracy (Fixed Engine) ---")
        for class_idx in range(NUM_CLASSES):
            c_total = class_total.get(class_idx, 0)
            c_correct = class_correct.get(class_idx, 0)
            if c_total > 0:
                c_acc = c_correct / c_total * 100.0
                class_name = IDX_TO_CLASS.get(class_idx, f"class_{class_idx}")
                print(f"  {class_name:8s}: {c_acc:.2f}% ({c_correct}/{c_total})")
            else:
                class_name = IDX_TO_CLASS.get(class_idx, f"class_{class_idx}")
                print(f"  {class_name:8s}: N/A (0/0)")

        gate_passed = (drop <= 0.50) and (agreement >= 99.0)
        if gate_passed:
            print(f"\n>>> GATE PASSED: drop <= 0.50 pt and agreement >= 99.00% <<<")
        else:
            print(f"\n>>> GATE FAILED: criteria not met (drop <= 0.50 pt, agreement >= 99.00%) <<<")

        if json_path:
            _write_json_report(json_path, {
                "engine": "both",
                "split": split,
                "subset": subset,
                "seed": seed,
                "total_clips": total,
                "checkpoint_id": get_ckpt_id(ckpt_path),
                "export_id": lib.mm_export_id().decode("utf-8"),
                "gate_passed": gate_passed,
                "float_correct": correct_f,
                "float_accuracy_pct": acc_f,
                "float_wilson_ci": [lo_f, hi_f],
                "fixed_correct": correct_q,
                "fixed_accuracy_pct": acc_q,
                "fixed_wilson_ci": [lo_q, hi_q],
                "accuracy_drop_pt": drop,
                "top1_agreement_pct": agreement,
                "median_logit_error": med_err,
                "fixed_vs_float_avg_linf": avg_linf_fq,
                "fixed_vs_float_avg_mean_abs": avg_mean_abs_fq,
                "fixed_vs_float_worst_linf": worst_linf_fq,
                "floatc_vs_pytorch_avg_linf": avg_linf_fp,
                "floatc_vs_pytorch_avg_mean_abs": avg_mean_abs_fp,
                "floatc_vs_pytorch_worst_linf": worst_linf_fp,
                "floatc_vs_pytorch_top1_agreement_pct": agreement_fp,
                "floatc_vs_pytorch_top1_agree_count": agree_fp,
                "per_class_fixed": {IDX_TO_CLASS[i]: {"correct": class_correct.get(i, 0), "total": class_total.get(i, 0), "accuracy_pct": class_correct.get(i, 0) / class_total.get(i, 0) * 100.0 if class_total.get(i, 0) > 0 else 0.0} for i in range(NUM_CLASSES)},
                "saturation": sat_stats_dict(lib),
            })
        if dump_npz:
            _dump_npz(dump_npz, rec)
        if not gate_passed:
            sys.exit(1)
        return acc_q


def _dump_npz(path: str, rec: dict):
    arrays = {}
    for key, vals in rec.items():
        if vals:
            arrays[key] = np.stack(vals, axis=0) if isinstance(vals[0], np.ndarray) else np.array(vals)
    np.savez_compressed(path, **arrays)
    print(f"Wrote per-clip arrays to {path}")

def _write_json_report(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate C engine on Speech Commands")
    parser.add_argument("--ckpt", type=str, default="checkpoints/quant_model.pt")
    parser.add_argument("--engine", choices=["float", "fixed", "both"], default="both")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection (default: 0)")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON report")
    parser.add_argument("--dump-npz", type=str, default=None, help="Path to save per-clip labels, predictions and last-token logits (.npz)")
    args = parser.parse_args()

    evaluate_c(
        ckpt_path=Path(args.ckpt),
        engine_type=args.engine,
        split=args.split,
        data_dir=args.data_dir,
        subset=args.subset,
        seed=args.seed,
        json_path=args.json,
        dump_npz=args.dump_npz,
    )

if __name__ == "__main__":
    main()
