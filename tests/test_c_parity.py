import ctypes
from pathlib import Path
import torch
import numpy as np
from train.kws.model import MuMambaKWS

# Define C structs matching mm_block.h
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

def load_c_engine():
    lib_path = Path("engine/libmumamba.so").resolve()
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

    lib.mm_push_audio.argtypes = [
        ctypes.POINTER(MMContext),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t
    ]
    lib.mm_push_audio.restype = ctypes.c_size_t

    lib.mm_step_f32.argtypes = [
        ctypes.POINTER(MMContext),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.mm_step_f32.restype = ctypes.c_int

    # Block step:
    lib.mm_block_step_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(BlockWeightsF32),
        ctypes.POINTER((ctypes.c_float * 64) * 3),
        ctypes.POINTER((ctypes.c_float * 8) * 64),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.mm_block_step_f32.restype = None
    return lib

def copy_weights_to_c(model: MuMambaKWS) -> ModelWeightsF32:
    w = ModelWeightsF32()
    # Stem
    stem_b = model.stem.bias.detach().cpu().numpy()
    stem_w = model.stem.weight.detach().cpu().numpy()
    for d in range(32):
        w.stem_b[d] = float(stem_b[d])
        for m in range(40):
            w.stem_w[d][m] = float(stem_w[d, m])

    # Blocks
    for l_idx, layer in enumerate(model.layers):
        bw = w.blocks[l_idx]
        norm_w = layer.norm.weight.detach().cpu().numpy()
        in_proj_w = layer.in_proj.weight.detach().cpu().numpy()
        conv1d_b = layer.conv1d.bias.detach().cpu().numpy()
        conv1d_w = layer.conv1d.weight.detach().cpu().numpy()
        x_proj_w = layer.x_proj.weight.detach().cpu().numpy()
        dt_proj_b = layer.dt_proj.bias.detach().cpu().numpy()
        dt_proj_w = layer.dt_proj.weight.detach().cpu().numpy()
        D_skip = layer.D_skip.detach().cpu().numpy()
        out_proj_w = layer.out_proj.weight.detach().cpu().numpy()

        for d in range(32):
            bw.norm_w[d] = float(norm_w[d])
        for i in range(128):
            for d in range(32):
                bw.in_proj_w[i][d] = float(in_proj_w[i, d])
        for d in range(64):
            bw.conv1d_b[d] = float(conv1d_b[d])
            for k in range(4):
                bw.conv1d_w[d][k] = float(conv1d_w[d, 0, k])
        for i in range(20):
            for d in range(64):
                bw.x_proj_w[i][d] = float(x_proj_w[i, d])
        for d in range(64):
            bw.dt_proj_b[d] = float(dt_proj_b[d])
            for r in range(4):
                bw.dt_proj_w[d][r] = float(dt_proj_w[d, r])
        for d in range(64):
            bw.D_skip[d] = float(D_skip[d])
        for o in range(32):
            for d in range(64):
                bw.out_proj_w[o][d] = float(out_proj_w[o, d])

    # Final norm & head
    final_norm_w = model.final_norm.weight.detach().cpu().numpy()
    head_b = model.head.bias.detach().cpu().numpy()
    head_w = model.head.weight.detach().cpu().numpy()
    for d in range(32):
        w.final_norm_w[d] = float(final_norm_w[d])
    for c in range(12):
        w.head_b[c] = float(head_b[c])
        for d in range(32):
            w.head_w[c][d] = float(head_w[c, d])

    return w

def test_check_3_c_parity():
    """
    Check 3: Host C float engine vs PyTorch step()
    - Block parity across 200 clips
    - Full engine mm_step_f32 parity
    Pass criterion: max diff < 1e-5
    """
    torch.manual_seed(123)
    np.random.seed(123)

    model = MuMambaKWS()
    # Randomize all weights (including dt_proj.weight to non-zero)
    for p in model.parameters():
        p.data.normal_(0.0, 0.2)
    model.eval()

    lib = load_c_engine()
    c_weights = copy_weights_to_c(model)
    lib.mm_set_weights_f32(ctypes.byref(c_weights))

    # 1. Block parity check on 200 random clips
    num_clips = 200
    max_block_diff = 0.0

    print(f"Running Check 3: MambaC1Block C vs PyTorch step() on {num_clips} clips...")
    for _ in range(num_clips):
        tok_np = np.random.randn(50, 32).astype(np.float32)
        tok_t = torch.from_numpy(tok_np).unsqueeze(0) # (1, 50, 32)

        # PyTorch block step
        layer = model.layers[0]
        conv_ring = torch.zeros(1, layer.d_conv - 1, layer.d_inner)
        h = torch.zeros(1, layer.d_inner, layer.d_state)
        py_state = (conv_ring, h)
        py_outs = []
        with torch.no_grad():
            for t in range(50):
                out_tok, py_state = layer.step(tok_t[:, t, :], py_state)
                py_outs.append(out_tok.squeeze(0).numpy())

        # C block step
        c_ring = ((ctypes.c_float * 64) * 3)()
        c_h = ((ctypes.c_float * 8) * 64)()
        c_outs = []
        for t in range(50):
            x_in = (ctypes.c_float * 32)(*tok_np[t])
            x_out = (ctypes.c_float * 32)()
            lib.mm_block_step_f32(
                x_in,
                ctypes.byref(c_weights.blocks[0]),
                ctypes.byref(c_ring),
                ctypes.byref(c_h),
                x_out
            )
            c_outs.append(np.array(x_out, dtype=np.float32))

        diff = np.max(np.abs(np.array(py_outs) - np.array(c_outs)))
        if diff > max_block_diff:
            max_block_diff = diff

    print(f"Check 3 (Block): Maximum block diff across {num_clips} clips = {max_block_diff:.3e}")
    assert max_block_diff < 1e-5, f"Check 3 block failed: max diff {max_block_diff} >= 1e-5"

    # 2. End-to-end full engine mm_step_f32 parity check on real audio frames
    print("Running Check 3: Full engine mm_step_f32 vs PyTorch step()...")
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))

    # Generate 16,192 audio samples (50 tokens: 512 + 49 * 320)
    audio = (np.random.randn(16192) * 5000).astype(np.int16)

    # Run C engine: push audio and step
    c_engine_logits = []
    c_features = []
    pushed = 0
    while pushed < len(audio):
        chunk = audio[pushed : pushed + 320]
        c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
        consumed = lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
        pushed += consumed

        feats_out = (ctypes.c_int16 * 40)()
        logits_out = (ctypes.c_float * 12)()
        if lib.mm_step_f32(ctypes.byref(ctx), feats_out, logits_out):
            c_engine_logits.append(np.array(logits_out, dtype=np.float32))
            c_features.append(np.array(feats_out, dtype=np.int16))

    assert len(c_engine_logits) == 50, f"Expected 50 tokens, got {len(c_engine_logits)}"

    # Run PyTorch model.step on the exact same extracted features
    py_features = torch.from_numpy(np.array(c_features, dtype=np.float32)).unsqueeze(0) # (1, 50, 40)
    py_states = model.init_streaming_state(batch_size=1)
    py_engine_logits = []
    with torch.no_grad():
        for t in range(50):
            py_log, py_states = model.step(py_features[:, t, :], py_states)
            py_engine_logits.append(py_log.squeeze(0).numpy())

    engine_diff = np.max(np.abs(np.array(c_engine_logits) - np.array(py_engine_logits)))
    print(f"Check 3 (Full Engine): Maximum logit diff across 50 tokens = {engine_diff:.3e}")
    assert engine_diff < 1e-5, f"Check 3 engine failed: max diff {engine_diff} >= 1e-5"
    print("Check 3 PASSED! C float engine matches PyTorch step() end-to-end!")

if __name__ == "__main__":
    test_check_3_c_parity()
