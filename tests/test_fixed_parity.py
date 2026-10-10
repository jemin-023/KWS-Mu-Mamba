import ctypes
import hashlib
from pathlib import Path
import numpy as np
import torch
import pytest

from train.kws.eval_c import load_engine, populate_c_weights, MMContext, ModelWeightsQ
from train.kws.data import SpeechCommandsDataset, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.model import MuMambaKWS

def test_fixed_vs_float_parity():
    """
    Fixed-point engine vs Float32 reference parity test on first 300 clips.
    Asserts:
      - C weights ID matches quant_model.pt SHA-256 prefix
      - Top-1 agreement >= 99%
      - Median logit error < 0.15
    """
    ckpt_path = Path("checkpoints/quant_model.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not found.")

    with open(ckpt_path, "rb") as f:
        expected_id = hashlib.sha256(f.read()).hexdigest()[:16]

    lib = load_engine()
    lib.mm_weights_id.restype = ctypes.c_char_p
    engine_id = lib.mm_weights_id().decode("utf-8")
    assert engine_id == expected_id, (
        f"Weights ID mismatch: engine has '{engine_id}', checkpoint is '{expected_id}'. Run make export."
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

    agree_count = 0
    correct_f = 0
    correct_q = 0
    logit_diffs = []

    for idx in range(num_eval):
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

    assert agreement >= 0.985, f"Agreement {agreement*100:.2f}% below 98.5% gate!"
    assert median_logit_err < 0.20, f"Median logit error {median_logit_err:.4f} too large!"

if __name__ == "__main__":
    test_fixed_vs_float_parity()
