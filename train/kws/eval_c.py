import argparse
import ctypes
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

    return lib

def populate_c_weights(model: MuMambaKWS, stats: dict = None) -> ModelWeightsF32:
    w = ModelWeightsF32()

    # Stem weights
    stem_w = model.stem.weight.detach().numpy().copy() # (32, 40)
    stem_b = model.stem.bias.detach().numpy().copy()   # (32,)

    # If trained with normalized features (mel - mean) / std, fold normalization into stem:
    # stem(x) = W * (x - mean)/std + b = (W / std) * x + (b - W * mean / std)
    if stats is not None and "mean" in stats and "std" in stats:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        # Fold scale
        stem_w_folded = stem_w / std[np.newaxis, :]
        # Fold mean offset
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
        norm_w = layer.norm.weight.detach().numpy()
        for d in range(32):
            bw.norm_w[d] = float(norm_w[d])

        in_proj_w = layer.in_proj.weight.detach().numpy()
        for i in range(128):
            for d in range(32):
                bw.in_proj_w[i][d] = float(in_proj_w[i, d])

        conv_w = layer.conv1d.weight.detach().squeeze(1).numpy()
        conv_b = layer.conv1d.bias.detach().numpy()
        for d in range(64):
            bw.conv1d_b[d] = float(conv_b[d])
            for k in range(4):
                bw.conv1d_w[d][k] = float(conv_w[d, k])

        x_proj_w = layer.x_proj.weight.detach().numpy()
        for i in range(20):
            for d in range(64):
                bw.x_proj_w[i][d] = float(x_proj_w[i, d])

        dt_proj_w = layer.dt_proj.weight.detach().numpy()
        dt_proj_b = layer.dt_proj.bias.detach().numpy()
        for d in range(64):
            bw.dt_proj_b[d] = float(dt_proj_b[d])
            for r in range(4):
                bw.dt_proj_w[d][r] = float(dt_proj_w[d, r])

        for d in range(64):
            bw.D_skip[d] = float(layer.D_skip[d])

        out_proj_w = layer.out_proj.weight.detach().numpy()
        for o in range(32):
            for d in range(64):
                bw.out_proj_w[o][d] = float(out_proj_w[o, d])

    # Final norm & head
    final_norm_w = model.final_norm.weight.detach().numpy()
    for d in range(32):
        w.final_norm_w[d] = float(final_norm_w[d])

    head_w = model.head.weight.detach().numpy()
    head_b = model.head.bias.detach().numpy()
    for c in range(12):
        w.head_b[c] = float(head_b[c])
        for d in range(32):
            w.head_w[c][d] = float(head_w[c, d])

    return w

def evaluate_c(
    ckpt_path: Path,
    engine_type: str = "float",
    split: str = "test",
    data_dir: str = "data/speech_commands/v0.02",
    subset: int = None,
):
    lib = load_engine()
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))

    stats = None
    if ckpt_path.exists():
        print(f"Loading checkpoint {ckpt_path}...")
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

    # Load fallback stats if not in checkpoint
    if stats is None:
        stats_file = Path("train/kws/frontend_stats.pt")
        if stats_file.exists():
            stats = torch.load(stats_file, weights_only=False)

    model.eval()

    if engine_type == "float":
        c_weights = populate_c_weights(model, stats=stats)
        lib.mm_set_weights_f32(ctypes.byref(c_weights))
    elif engine_type == "fixed":
        # Load exported quantized weights directly from C library rodata
        c_weights_q = ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")
        lib.mm_set_weights_q(ctypes.byref(c_weights_q))
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")

    ds = SpeechCommandsDataset(data_dir=data_dir, split=split, augment=False)
    total_samples = len(ds)
    if subset is not None and subset < total_samples:
        indices = range(subset)
    else:
        indices = range(total_samples)

    print(f"Evaluating C engine ({engine_type}) on split '{split}' ({len(indices)} clips)...")

    correct = 0
    total = 0

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

        # Pad or truncate to 16,192 samples (exactly 50 tokens)
        if len(audio) < TARGET_AUDIO_LEN:
            audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)), mode="constant")
        elif len(audio) > TARGET_AUDIO_LEN:
            audio = audio[:TARGET_AUDIO_LEN]

        audio = np.ascontiguousarray(audio, dtype=np.int16)

        # Reset streaming state for clip evaluation
        lib.mm_reset(ctypes.byref(ctx))

        tokens_fired = 0

        if engine_type == "float":
            last_logits_f = (ctypes.c_float * 12)()
            chunk0 = audio[:512]
            c_chunk0 = (ctypes.c_int16 * len(chunk0))(*chunk0)
            lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
            if lib.mm_step_f32(ctypes.byref(ctx), None, last_logits_f):
                tokens_fired += 1

            for t in range(1, 50):
                start = 512 + (t - 1) * 320
                chunk = audio[start : start + 320]
                c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
                lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
                if lib.mm_step_f32(ctypes.byref(ctx), None, last_logits_f):
                    tokens_fired += 1
            pred = int(np.argmax(np.array(last_logits_f, dtype=np.float32)))
        else:
            last_logits_q = (ctypes.c_int32 * 12)()
            chunk0 = audio[:512]
            c_chunk0 = (ctypes.c_int16 * len(chunk0))(*chunk0)
            lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
            if lib.mm_step_q(ctypes.byref(ctx), None, last_logits_q):
                tokens_fired += 1

            for t in range(1, 50):
                start = 512 + (t - 1) * 320
                chunk = audio[start : start + 320]
                c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
                lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
                if lib.mm_step_q(ctypes.byref(ctx), None, last_logits_q):
                    tokens_fired += 1
            pred = int(np.argmax(np.array(last_logits_q, dtype=np.int32)))

        assert tokens_fired == 50, f"Expected 50 tokens, fired {tokens_fired}"

        if pred == label:
            correct += 1
        total += 1

        pbar.set_postfix({"acc": f"{correct / total * 100:.2f}%"})

    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\nFinal C Engine ({engine_type}) Accuracy on {split}: {acc:.2f}% ({correct}/{total})")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate C engine on Speech Commands")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--engine", choices=["float", "fixed"], default="float")
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
