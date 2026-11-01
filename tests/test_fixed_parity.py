import ctypes
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch
import pytest

from train.kws.eval_c import load_engine, populate_c_weights, MMContext, ModelWeightsQ, select_eval_indices
from train.kws.data import SpeechCommandsDataset, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.model import MuMambaKWS

def _find_checkpoint(engine_id: str):
    """The checkpoint the engine was exported from: $KWS_CKPT first, then checkpoints/*.pt, matched by SHA-256 prefix."""
    cands = ([Path(os.environ["KWS_CKPT"])] if os.environ.get("KWS_CKPT") else []) + sorted(Path("checkpoints").glob("*.pt"))
    for p in cands:
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()[:16] == engine_id:
            return p
    return None

def test_fixed_vs_float_parity():
    """
    REGRESSION SMOKE TEST, not the acceptance gate. Fixed-point engine vs Float32 reference on 300 clips (seed 0).
    The acceptance gate is the full-test eval_c gate (accuracy drop <= 0.50 pt AND top-1 agreement >= 99 % on all 4,890
    test clips), which every export must pass and record in its JSON. A 300-clip sample is too small for a 99 % gate
    (at a true agreement of 99.35 % it fails by chance about 4 % of the time), so this test only catches broken exports
    or scale bugs, which give far lower agreement.
    Asserts:
      - C weights ID matches the SHA-256 prefix of the exported checkpoint
      - Top-1 agreement >= 97.5 %
      - Median logit error < 0.20
    """
    lib = load_engine()
    lib.mm_weights_id.restype = ctypes.c_char_p
    engine_id = lib.mm_weights_id().decode("utf-8")
    ckpt_path = _find_checkpoint(engine_id)
    assert ckpt_path is not None, (
        f"No checkpoint matches the engine weights ID '{engine_id}' (searched $KWS_CKPT and checkpoints/*.pt). "
        "Run the export for a checkpoint, or set KWS_CKPT to the exported one."
    )

    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})
    model = MuMambaKWS(
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

    c_weights_f = populate_c_weights(model, stats=ckpt.get("stats"))
    c_weights_q = ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")

    ds = SpeechCommandsDataset(data_dir="data/speech_commands/v0.02", split="test", augment=False)
    num_eval = min(300, len(ds))
    eval_indices = select_eval_indices(len(ds), 300, seed=0)

    agree_count = 0
    correct_f = 0
    correct_q = 0
    logit_diffs = []

    for idx in eval_indices:
        fpath, label, silence_idx = ds.epoch_samples[idx]
        if fpath is None:
            audio = ds.val_silence_crops[silence_idx] if silence_idx is not None else ds._get_silence_crop()
        else:
            audio = read_wav_int16(fpath)

        if len(audio) < TARGET_AUDIO_LEN:
            audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)))
        else:
            audio = audio[:TARGET_AUDIO_LEN]
        audio = np.ascontiguousarray(audio, dtype=np.int16)

        # 1. Float forward
        lib.mm_set_weights_f32(ctypes.byref(c_weights_f))
        lib.mm_reset(ctypes.byref(ctx))
        last_f = (ctypes.c_float * 12)()
        chunk0 = audio[:512]
        c_chunk0 = (ctypes.c_int16 * len(chunk0))(*chunk0)
        lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
        lib.mm_step_f32(ctypes.byref(ctx), None, last_f)
        for t in range(1, 50):
            chunk = audio[512 + (t - 1) * 320 : 512 + t * 320]
            c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
            lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
            lib.mm_step_f32(ctypes.byref(ctx), None, last_f)
        pred_f = int(np.argmax(last_f))

        # 2. Fixed forward
        lib.mm_set_weights_q(ctypes.byref(c_weights_q))
        lib.mm_reset(ctypes.byref(ctx))
        last_q = (ctypes.c_int32 * 12)()
        lib.mm_push_audio(ctypes.byref(ctx), c_chunk0, len(chunk0))
        lib.mm_step_q(ctypes.byref(ctx), None, last_q)
        for t in range(1, 50):
            chunk = audio[512 + (t - 1) * 320 : 512 + t * 320]
            c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
            lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
            lib.mm_step_q(ctypes.byref(ctx), None, last_q)
        pred_q = int(np.argmax(last_q))

        if pred_f == label:
            correct_f += 1
        if pred_q == label:
            correct_q += 1
        if pred_f == pred_q:
            agree_count += 1

        lf = np.array(last_f, dtype=np.float32)
        lq = np.array(last_q, dtype=np.float32) / 4096.0
        logit_diffs.append(np.abs(lf - lq))

    agreement = agree_count / num_eval
    median_logit_err = float(np.median(logit_diffs))
    print(f"\nParity 300 clips: float_acc={correct_f/num_eval*100:.2f}%, fixed_acc={correct_q/num_eval*100:.2f}%, agreement={agreement*100:.2f}%, median_err={median_logit_err:.4f}")

    assert agreement >= 0.975, (
        f"Agreement {agreement*100:.2f}% below the 97.5% smoke threshold. (The acceptance gate is the full-test eval_c gate: "
        f"drop <= 0.50 pt and agreement >= 99%.)"
    )
    assert median_logit_err < 0.20, f"Median logit error {median_logit_err:.4f} too large!"

if __name__ == "__main__":
    test_fixed_vs_float_parity()
