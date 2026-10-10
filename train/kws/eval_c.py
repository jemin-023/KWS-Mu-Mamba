import argparse
import ctypes
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from train.kws.data import SpeechCommandsDataset, NUM_CLASSES, IDX_TO_CLASS, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.model import MuMambaKWS

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
        ("conv_ring_f", ((ctypes.c_float * 64) * 3) * 2),
        ("conv_ring_q", ((ctypes.c_int16 * 64) * 3) * 2),
        ("conv_ring_idx", ctypes.c_uint8 * 2),
        ("ssm_h_f", ((ctypes.c_float * 8) * 64) * 2),
        ("ssm_h_q", ((ctypes.c_int32 * 8) * 64) * 2),
        ("logits_q", ctypes.c_int32 * 12),
        ("logits_f", ctypes.c_float * 12),
        ("smoothed_probs", ctypes.c_int32 * 12),
        ("refractory_timer", ctypes.c_int16),
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
    ]

def load_engine():
    lib_path = Path("engine/libmumamba.so").resolve()
    if not lib_path.exists():
        raise FileNotFoundError(f"{lib_path} does not exist. Run 'make -C engine' first.")
    lib = ctypes.CDLL(str(lib_path))

    lib.mm_context_size.argtypes = []
    lib.mm_context_size.restype = ctypes.c_size_t

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

def run_clip_stream(lib, ctx, audio, is_float: bool, last_logits):
    chunk0 = audio[:512]
    c_chunk0 = (ctypes.c_int16 * len(chunk0))(*chunk0)
    lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
    if is_float:
        lib.mm_step_f32(ctypes.byref(ctx), None, last_logits)
    else:
        lib.mm_step_q(ctypes.byref(ctx), None, last_logits)

    for t in range(1, 50):
        start = 512 + (t - 1) * 320
        chunk = audio[start : start + 320]
        c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
        lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
        if is_float:
            lib.mm_step_f32(ctypes.byref(ctx), None, last_logits)
        else:
            lib.mm_step_q(ctypes.byref(ctx), None, last_logits)

def print_saturation_stats(lib):
    sat = MMSatStats()
    lib.mm_get_sat_stats(ctypes.byref(sat))
    print("\nSaturation rates:")
    print(f"  Stem:     {sat.stem_sat}/{sat.stem_total} ({sat.stem_sat/max(1, sat.stem_total)*100:.3f}%)")
    print(f"  in_proj:  {sat.in_proj_sat}/{sat.in_proj_total} ({sat.in_proj_sat/max(1, sat.in_proj_total)*100:.3f}%)")
    print(f"  conv:     {sat.conv_sat}/{sat.conv_total} ({sat.conv_sat/max(1, sat.conv_total)*100:.3f}%)")
    print(f"  x_proj:   {sat.x_proj_sat}/{sat.x_proj_total} ({sat.x_proj_sat/max(1, sat.x_proj_total)*100:.3f}%)")
    print(f"  residual: {sat.resid_sat}/{sat.resid_total} ({sat.resid_sat/max(1, sat.resid_total)*100:.3f}%)")
    print(f"  h:        {sat.h_sat}/{sat.h_total} ({sat.h_sat/max(1, sat.h_total)*100:.3f}%)")

def evaluate_c(
    ckpt_path: Path,
    engine_type: str = "both",
    split: str = "test",
    data_dir: str = "data/speech_commands/v0.02",
    subset: int = None,
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
    if subset is not None and subset < total_samples:
        indices = range(subset)
    else:
        indices = range(total_samples)

    print(f"Evaluating C engine ({engine_type}) on split '{split}' ({len(indices)} clips)...")
    lib.mm_reset_sat_stats()

    correct_f, correct_q, agree = 0, 0, 0
    total = 0
    logit_diffs = []

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

        if engine_type in ["float", "both"]:
            lib.mm_set_weights_f32(ctypes.byref(c_weights_f))
            lib.mm_reset(ctypes.byref(ctx))
            last_logits_f = (ctypes.c_float * 12)()
            run_clip_stream(lib, ctx, audio, is_float=True, last_logits=last_logits_f)
            pred_f = int(np.argmax(np.array(last_logits_f, dtype=np.float32)))
            if pred_f == label:
                correct_f += 1

        if engine_type in ["fixed", "both"]:
            lib.mm_set_weights_q(ctypes.byref(c_weights_q))
            lib.mm_reset(ctypes.byref(ctx))
            last_logits_q = (ctypes.c_int32 * 12)()
            run_clip_stream(lib, ctx, audio, is_float=False, last_logits=last_logits_q)
            pred_q = int(np.argmax(np.array(last_logits_q, dtype=np.int32)))
            if pred_q == label:
                correct_q += 1

        if engine_type == "both":
            if pred_f == pred_q:
                agree += 1
            lf = np.array(last_logits_f, dtype=np.float32)
            lq = np.array(last_logits_q, dtype=np.float32) / 4096.0
            logit_diffs.append(np.abs(lf - lq))

        total += 1
        if engine_type == "both":
            pbar.set_postfix({"f_acc": f"{correct_f/total*100:.1f}%", "q_acc": f"{correct_q/total*100:.1f}%", "agree": f"{agree/total*100:.1f}%"})
        elif engine_type == "float":
            pbar.set_postfix({"acc": f"{correct_f/total*100:.2f}%"})
        else:
            pbar.set_postfix({"acc": f"{correct_q/total*100:.2f}%"})

    if engine_type == "float":
        acc = correct_f / total * 100.0 if total > 0 else 0.0
        print(f"\nFinal C Engine (float) Accuracy on {split}: {acc:.2f}% ({correct_f}/{total})")
        return acc
    elif engine_type == "fixed":
        acc = correct_q / total * 100.0 if total > 0 else 0.0
        print(f"\nFinal C Engine (fixed) Accuracy on {split}: {acc:.2f}% ({correct_q}/{total})")
        print_saturation_stats(lib)
        return acc
    else: # both
        acc_f = correct_f / total * 100.0
        acc_q = correct_q / total * 100.0
        drop = acc_f - acc_q
        agreement = agree / total * 100.0
        med_err = float(np.median(logit_diffs))

        print(f"\n================ C ENGINE VERIFICATION GATE ================")
        print(f"Float Accuracy:     {acc_f:.2f}% ({correct_f}/{total})")
        print(f"Fixed Accuracy:     {acc_q:.2f}% ({correct_q}/{total})")
        print(f"Accuracy Drop:      {drop:+.2f} pt (gate: <= 0.50 pt)")
        print(f"Top-1 Agreement:    {agreement:.2f}% (gate: >= 99.00%)")
        print(f"Median Logit Error: {med_err:.4f}")
        print_saturation_stats(lib)

        passed = (drop <= 0.50) and (agreement >= 99.0)
        if passed:
            print(f"\n>>> GATE PASSED: drop <= 0.50 pt and agreement >= 99.00% <<<")
        else:
            print(f"\n>>> GATE FAILED: criteria not met (drop <= 0.50 pt, agreement >= 99.00%) <<<")
            sys.exit(1)
        return acc_q

def main():
    parser = argparse.ArgumentParser(description="Evaluate C engine on Speech Commands")
    parser.add_argument("--ckpt", type=str, default="checkpoints/quant_model.pt")
    parser.add_argument("--engine", choices=["float", "fixed", "both"], default="both")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    evaluate_c(
        ckpt_path=Path(args.ckpt),
        engine_type=args.engine,
        split=args.split,
        data_dir=args.data_dir,
        subset=args.subset,
    )

if __name__ == "__main__":
    main()
