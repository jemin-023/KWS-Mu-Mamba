"""Golden per-token logits of the host fixed-point engine, used to prove that engine edits are bit-identical.

Usage (from the repo root, after `make -C engine clean all`):
    PYTHONPATH=. python tests/make_golden.py

Writes tests/data/golden_fixed_<weights_id>_<export tag>.npz holding every per-token int32 Q12 logit vector for
  - 24 test clips (eval_c.select_eval_indices(len(ds), 24, seed=123)), fed like eval_c.run_clip_stream
  - one 60 s validation stream (build_continuous_stream(num_keywords=15, is_training=False, Random(5))),
    fed like stream_eval.run_stream_inference
The feeding loops are re-implemented here on purpose so later edits to eval_c / stream_eval cannot move the golden.
"""
import ctypes
import random
import sys
from pathlib import Path

import numpy as np

from train.kws.data import SpeechCommandsDataset, build_continuous_stream, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.export_id import compute_export_id
from train.kws.eval_c import load_engine, MMContext, ModelWeightsQ, select_eval_indices

DATA_DIR = "data/speech_commands/v0.02"
GOLDEN_DIR = Path(__file__).parent / "data"
GEN_DIR = Path(__file__).resolve().parents[1] / "engine" / "gen"
NUM_CLIPS = 24
CLIP_SEED = 123
STREAM_KEYWORDS = 15
STREAM_SEED = 5


def golden_path(weights_id: str) -> Path:
    return GOLDEN_DIR / f"golden_fixed_{weights_id}_{compute_export_id(GEN_DIR)}.npz"


def _step(lib, ctx, out_buf, tokens):
    if lib.mm_step_q(ctypes.byref(ctx), None, out_buf):
        tokens.append(np.array(out_buf, dtype=np.int32).copy())


def _feed(lib, audio: np.ndarray) -> np.ndarray:
    """512-sample first frame, then 320-sample hops; one logit vector per step. Returns (T, 12) int32."""
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    lib.mm_set_weights_q(ctypes.byref(ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")))
    lib.mm_reset(ctypes.byref(ctx))
    out_buf = (ctypes.c_int32 * 12)()
    tokens = []

    first = np.ascontiguousarray(audio[:512], dtype=np.int16)
    lib.mm_push_audio(ctypes.byref(ctx), first.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(first))
    _step(lib, ctx, out_buf, tokens)

    pushed = 512
    while pushed + 320 <= len(audio):
        chunk = np.ascontiguousarray(audio[pushed:pushed + 320], dtype=np.int16)
        p = lib.mm_push_audio(ctypes.byref(ctx), chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(chunk))
        _step(lib, ctx, out_buf, tokens)
        if p < len(chunk):
            rem = np.ascontiguousarray(chunk[p:], dtype=np.int16)
            lib.mm_push_audio(ctypes.byref(ctx), rem.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(rem))
        pushed += 320
    return np.stack(tokens, axis=0)


def _clip_audio(ds, idx: int) -> np.ndarray:
    fpath, _, silence_idx = ds.epoch_samples[idx]
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
    return np.ascontiguousarray(audio[:TARGET_AUDIO_LEN], dtype=np.int16)


def compute_golden(lib) -> dict:
    ds_test = SpeechCommandsDataset(data_dir=DATA_DIR, split="test", augment=False)
    indices = select_eval_indices(len(ds_test), NUM_CLIPS, seed=CLIP_SEED)
    clip_logits = np.stack([_feed(lib, _clip_audio(ds_test, i)) for i in indices], axis=0)  # (24, 50, 12)

    ds_val = SpeechCommandsDataset(data_dir=DATA_DIR, split="validation", augment=False)
    stream_audio, _, _ = build_continuous_stream(
        ds_val, num_keywords=STREAM_KEYWORDS, is_training=False, rng=random.Random(STREAM_SEED)
    )
    stream_logits = _feed(lib, stream_audio)  # (T, 12)
    return {
        "clip_indices": np.array(indices, dtype=np.int64),
        "clip_logits": clip_logits,
        "stream_logits": stream_logits,
    }


def main():
    lib = load_engine()
    weights_id = lib.mm_weights_id().decode("utf-8")
    data = compute_golden(lib)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(weights_id)
    np.savez_compressed(path, weights_id=np.array(weights_id), **data)
    print(f"weights id {weights_id}: {data['clip_logits'].shape[0]} clips x {data['clip_logits'].shape[1]} tokens, "
          f"stream {data['stream_logits'].shape[0]} tokens -> {path}")


if __name__ == "__main__":
    sys.exit(main())
