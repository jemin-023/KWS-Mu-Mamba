import argparse
import hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from train.kws.model import MuMambaKWS
from train.kws.data import SpeechCommandsDataset

def quantize_row_symmetric(weight_row, max_int=127):
    """Symmetric per-row int8 quantization."""
    max_val = np.max(np.abs(weight_row))
    if max_val < 1e-8:
        scale = 1.0 / max_int
        q_w = np.zeros_like(weight_row, dtype=np.int8)
    else:
        scale = float(max_val / max_int)
        q_w = np.clip(np.round(weight_row / scale), -max_int, max_int).astype(np.int8)
    return q_w, scale

def compute_mult_shift(scale):
    """
    Compute integer multiplier (int16: 1..32767) and shift (uint8: 0..31)
    such that scale ~= mult * 2^(-shift).
    """
    if scale <= 0:
        return 0, 0
    shift = int(np.floor(-np.log2(scale) + 14))
    shift = max(0, min(31, shift))
    mult = int(np.round(scale * (2 ** shift)))
    mult = max(1, min(32767, mult))
    return mult, shift

def calibrate_formats(model: MuMambaKWS, data_dir: str, margin: float = 2.0):
    """
    Runs checkpoint over validation features, takes the 99th percentile magnitude
    with 2x margin, and computes fractional bits for MM_Q_RESID and MM_Q_XPROJ.
    """
    val_ds = SpeechCommandsDataset(data_dir=data_dir, split="validation", augment=False)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    resids = []
    xprojs = []
    inprojs = []

    with torch.no_grad():
        for features, _ in loader:
            h = model.stem(features)
            resids.append(h.abs().reshape(-1))
            for layer in model.layers:
                x_norm = layer.norm(h)
                xz = layer.in_proj(x_norm)
                inprojs.append(xz.abs().reshape(-1))
                u, z = torch.split(xz, [layer.d_inner, layer.d_inner], dim=-1)
                u_padded = torch.nn.functional.pad(u.transpose(1, 2), (layer.d_conv - 1, 0))
                u_conv = layer.conv1d(u_padded)
                u_prime = torch.nn.functional.silu(u_conv.transpose(1, 2))
                x_ssm = layer.x_proj(u_prime)
                xprojs.append(x_ssm.abs().reshape(-1))

                dt_rank_val, B_val, C_val = torch.split(x_ssm, [layer.dt_rank, layer.d_state, layer.d_state], dim=-1)
                p = layer.dt_proj(dt_rank_val)
                y_scan, _ = layer._c1_scan(u_prime, p, B_val, C_val)
                y_scan = y_scan + layer.D_skip.view(1, 1, -1) * u_prime
                y_gate = y_scan * torch.nn.functional.silu(z)
                out = layer.out_proj(y_gate)
                h = h + out
                resids.append(h.abs().reshape(-1))

    all_resids = torch.cat(resids).numpy()
    all_xprojs = torch.cat(xprojs).numpy()
    all_inprojs = torch.cat(inprojs).numpy()

    mag_r = float(np.percentile(all_resids, 99.0))
    mag_x = float(np.percentile(all_xprojs, 99.0))
    mag_inp = float(np.percentile(all_inprojs, 99.0))

    q_resid = int(np.floor(np.log2(32767.0 / (margin * mag_r))))
    q_xproj = int(np.floor(np.log2(32767.0 / (margin * mag_x))))
    q_inproj = int(np.floor(np.log2(32767.0 / (margin * mag_inp))))

    q_resid = max(4, min(12, q_resid))
    q_xproj = max(8, min(12, q_xproj))
    q_inproj = max(8, min(12, q_inproj))
    print(f"Calibrated formats: MM_Q_RESID={q_resid} (mag={mag_r:.2f}), MM_Q_XPROJ={q_xproj} (mag={mag_x:.2f}), MM_Q_INPROJ={q_inproj} (mag={mag_inp:.2f})")
    return q_resid, q_xproj, q_inproj

def export_model(ckpt_path: Path, out_dir: Path, data_dir: str = "data/speech_commands/v0.02"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute checkpoint SHA-256 prefix
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    else:
        ckpt_hash = "0000000000000000"

    # 1. Load checkpoint
    if ckpt_path.exists():
        print(f"Loading checkpoint {ckpt_path} (ID: {ckpt_hash})...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args_dict = ckpt.get("args", {})
        d_model = args_dict.get("d_model", 32)
        d_inner = args_dict.get("d_inner", 64)
        d_state = args_dict.get("d_state", 8)
        dt_rank = args_dict.get("dt_rank", 4)
        d_conv = args_dict.get("d_conv", 4)
        num_layers = args_dict.get("num_layers", 2)
        num_classes = args_dict.get("num_classes", 12)

        model = MuMambaKWS(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
            num_layers=num_layers,
            num_classes=num_classes,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        # Load normalization stats from checkpoint if present, or fallback
        if "stats" in ckpt and ckpt["stats"] is not None:
            mean = ckpt["stats"]["mean"].numpy()
            std = ckpt["stats"]["std"].numpy()
        elif Path("train/kws/frontend_stats.pt").exists():
            st = torch.load("train/kws/frontend_stats.pt", map_location="cpu", weights_only=False)
            mean = st["mean"].numpy()
            std = st["std"].numpy()
        else:
            mean = np.zeros(40, dtype=np.float32)
            std = np.ones(40, dtype=np.float32)
    else:
        print(f"Checkpoint {ckpt_path} not found. Generating template from default model.")
        model = MuMambaKWS()
        mean = np.zeros(40, dtype=np.float32)
        std = np.ones(40, dtype=np.float32)

    model.eval()
    print(f"Model: D={model.d_model}, D_inner={model.layers[0].d_inner}, N={model.layers[0].d_state}, L={model.num_layers}")

    # Calibrate formats
    q_resid, q_xproj, q_inproj = calibrate_formats(model, data_dir=data_dir)

    # 2. Fold feature normalization into stem:
    raw_stem_w = model.stem.weight.detach().cpu().numpy() # (32, 40)
    raw_stem_b = model.stem.bias.detach().cpu().numpy()   # (32,)

    stem_w_folded = raw_stem_w / std[np.newaxis, :]
    stem_b_folded = raw_stem_b - np.sum(raw_stem_w * (mean / std)[np.newaxis, :], axis=1)

    # 3. Export gen/mm_config.h
    config_h = out_dir / "mm_config.h"
    with open(config_h, "w") as f:
        f.write("/* Auto-generated by export.py - DO NOT EDIT */\n")
        f.write("#ifndef MM_CONFIG_H\n#define MM_CONFIG_H\n\n")
        f.write(f"#define MM_DIM         {model.d_model}\n")
        f.write(f"#define MM_D_INNER     {model.layers[0].d_inner}\n")
        f.write(f"#define MM_D_STATE     {model.layers[0].d_state}\n")
        f.write(f"#define MM_DT_RANK     {model.layers[0].dt_rank}\n")
        f.write(f"#define MM_D_CONV      {model.layers[0].d_conv}\n")
        f.write(f"#define MM_NUM_LAYERS  {model.num_layers}\n")
        f.write(f"#define MM_NUM_CLASSES {model.num_classes}\n\n")
        f.write(f"#define MM_Q_RESID     {q_resid}\n")
        f.write(f"#define MM_Q_XPROJ     {q_xproj}\n")
        f.write(f"#define MM_Q_INPROJ    {q_inproj}\n")
        f.write(f'#define MM_WEIGHTS_ID  "{ckpt_hash}"\n\n')
        f.write("#endif /* MM_CONFIG_H */\n")
    print(f"Exported {config_h}")

    # 4. Export gen/mm_weights.c (Float32 Reference Weights)
    weights_c = out_dir / "mm_weights.c"
    with open(weights_c, "w") as f:
        f.write("/* Auto-generated by export.py - DO NOT EDIT */\n")
        f.write('#include "mu_mamba.h"\n')
        f.write('#include "mm_block.h"\n\n')

        def fmt_f(val):
            return f"{float(val):.9g}f"

        f.write("const mm_model_weights_f32_t g_default_model_weights_f32 = {\n")

        # Stem
        f.write("    .stem_w = {\n")
        for r in range(model.d_model):
            f.write("        { " + ", ".join(fmt_f(x) for x in stem_w_folded[r]) + " },\n")
        f.write("    },\n")
        f.write("    .stem_b = { " + ", ".join(fmt_f(x) for x in stem_b_folded) + " },\n")

        # Blocks
        f.write("    .blocks = {\n")
        for l_idx, layer in enumerate(model.layers):
            norm_w = layer.norm.weight.detach().cpu().numpy()
            in_proj_w = layer.in_proj.weight.detach().cpu().numpy()
            conv1d_w = layer.conv1d.weight.detach().squeeze(1).cpu().numpy()
            conv1d_b = layer.conv1d.bias.detach().cpu().numpy()
            x_proj_w = layer.x_proj.weight.detach().cpu().numpy()
            dt_proj_w = layer.dt_proj.weight.detach().cpu().numpy()
            dt_proj_b = layer.dt_proj.bias.detach().cpu().numpy()
            D_skip = layer.D_skip.detach().cpu().numpy()
            out_proj_w = layer.out_proj.weight.detach().cpu().numpy()

            f.write(f"        /* Block {l_idx} */\n        {{\n")
            f.write("            .norm_w = { " + ", ".join(fmt_f(x) for x in norm_w) + " },\n")

            f.write("            .in_proj_w = {\n")
            for r in range(len(in_proj_w)):
                f.write("                { " + ", ".join(fmt_f(x) for x in in_proj_w[r]) + " },\n")
            f.write("            },\n")

            f.write("            .conv1d_w = {\n")
            for r in range(len(conv1d_w)):
                f.write("                { " + ", ".join(fmt_f(x) for x in conv1d_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .conv1d_b = { " + ", ".join(fmt_f(x) for x in conv1d_b) + " },\n")

            f.write("            .x_proj_w = {\n")
            for r in range(len(x_proj_w)):
                f.write("                { " + ", ".join(fmt_f(x) for x in x_proj_w[r]) + " },\n")
            f.write("            },\n")

            f.write("            .dt_proj_w = {\n")
            for r in range(len(dt_proj_w)):
                f.write("                { " + ", ".join(fmt_f(x) for x in dt_proj_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .dt_proj_b = { " + ", ".join(fmt_f(x) for x in dt_proj_b) + " },\n")

            f.write("            .D_skip = { " + ", ".join(fmt_f(x) for x in D_skip) + " },\n")

            f.write("            .out_proj_w = {\n")
            for r in range(len(out_proj_w)):
                f.write("                { " + ", ".join(fmt_f(x) for x in out_proj_w[r]) + " },\n")
            f.write("            },\n")
            f.write("        },\n")
        f.write("    },\n")

        # Final norm & head
        final_norm_w = model.final_norm.weight.detach().cpu().numpy()
        head_w = model.head.weight.detach().cpu().numpy()
        head_b = model.head.bias.detach().cpu().numpy()

        f.write("    .final_norm_w = { " + ", ".join(fmt_f(x) for x in final_norm_w) + " },\n")
        f.write("    .head_w = {\n")
        for r in range(len(head_w)):
            f.write("        { " + ", ".join(fmt_f(x) for x in head_w[r]) + " },\n")
        f.write("    },\n")
        f.write("    .head_b = { " + ", ".join(fmt_f(x) for x in head_b) + " },\n")
        f.write("};\n")

    print(f"Exported {weights_c}")

    # 5. Export gen/mm_weights_q.c (Quantized W8A16 Weights)
    weights_q_c = out_dir / "mm_weights_q.c"
    with open(weights_q_c, "w") as f:
        f.write("/* Auto-generated by export.py - DO NOT EDIT */\n")
        f.write('#include "mu_mamba.h"\n')
        f.write('#include "mm_block_q.h"\n\n')

        f.write("const mm_model_weights_q_t g_default_model_weights_q = {\n")

        # Stem: mel features (raw) -> output in Q_RESID: scale * 2**MM_Q_RESID
        q_stem_w = np.zeros_like(stem_w_folded, dtype=np.int8)
        stem_mults = []
        stem_shifts = []
        stem_biases = []
        for d in range(model.d_model):
            qw, scale = quantize_row_symmetric(stem_w_folded[d])
            q_stem_w[d] = qw
            mult, shift = compute_mult_shift(scale * (2 ** q_resid))
            stem_mults.append(mult)
            stem_shifts.append(shift)
            b_scaled = int(np.round(stem_b_folded[d] / scale)) if scale > 0 else 0
            stem_biases.append(b_scaled)

        f.write("    .stem_w = {\n")
        for r in range(model.d_model):
            f.write("        { " + ", ".join(str(int(x)) for x in q_stem_w[r]) + " },\n")
        f.write("    },\n")
        f.write("    .stem_b = { " + ", ".join(str(int(x)) for x in stem_biases) + " },\n")
        f.write("    .stem_scale_mult = { " + ", ".join(str(int(x)) for x in stem_mults) + " },\n")
        f.write("    .stem_scale_shift = { " + ", ".join(str(int(x)) for x in stem_shifts) + " },\n")

        # Blocks
        f.write("    .blocks = {\n")
        for l_idx, layer in enumerate(model.layers):
            norm_w = layer.norm.weight.detach().cpu().numpy()
            in_proj_w = layer.in_proj.weight.detach().cpu().numpy()
            conv1d_w = layer.conv1d.weight.detach().squeeze(1).cpu().numpy()
            conv1d_b = layer.conv1d.bias.detach().cpu().numpy()
            x_proj_w = layer.x_proj.weight.detach().cpu().numpy()
            dt_proj_w = layer.dt_proj.weight.detach().cpu().numpy()
            dt_proj_b = layer.dt_proj.bias.detach().cpu().numpy()
            D_skip = layer.D_skip.detach().cpu().numpy()
            out_proj_w = layer.out_proj.weight.detach().cpu().numpy()

            f.write(f"        /* Block {l_idx} */\n        {{\n")
            # RMSNorm gain: Q14 (scale 16384.0) instead of Q15
            norm_q14 = np.clip(np.round(norm_w * 16384.0), -32768, 32767).astype(np.int16)
            f.write("            .norm_w = { " + ", ".join(str(int(x)) for x in norm_q14) + " },\n")

            # in_proj: row-wise, scale -> scale / 2**(12 - MM_Q_INPROJ)
            q_in_w = np.zeros_like(in_proj_w, dtype=np.int8)
            in_mults, in_shifts = [], []
            for r in range(len(in_proj_w)):
                qw, scale = quantize_row_symmetric(in_proj_w[r])
                q_in_w[r] = qw
                m, s = compute_mult_shift(scale / (2 ** (12 - q_inproj)))
                in_mults.append(m)
                in_shifts.append(s)

            f.write("            .in_proj_w = {\n")
            for r in range(len(q_in_w)):
                f.write("                { " + ", ".join(str(int(x)) for x in q_in_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .in_proj_scale_mult = { " + ", ".join(str(int(x)) for x in in_mults) + " },\n")
            f.write("            .in_proj_scale_shift = { " + ", ".join(str(int(x)) for x in in_shifts) + " },\n")

            # conv1d: scale -> scale * 2**(12 - MM_Q_INPROJ), bias -> b / scale * 2**MM_Q_INPROJ
            q_conv_w = np.zeros_like(conv1d_w, dtype=np.int8)
            conv_mults, conv_shifts, conv_biases = [], [], []
            for d in range(len(conv1d_w)):
                qw, scale = quantize_row_symmetric(conv1d_w[d])
                q_conv_w[d] = qw
                m, s = compute_mult_shift(scale * (2 ** (12 - q_inproj)))
                conv_mults.append(m)
                conv_shifts.append(s)
                b_scaled = int(np.round(conv1d_b[d] / scale * (2 ** q_inproj))) if scale > 0 else 0
                conv_biases.append(b_scaled)

            f.write("            .conv1d_w = {\n")
            for r in range(len(q_conv_w)):
                f.write("                { " + ", ".join(str(int(x)) for x in q_conv_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .conv1d_b = { " + ", ".join(str(int(x)) for x in conv_biases) + " },\n")
            f.write("            .conv1d_scale_mult = { " + ", ".join(str(int(x)) for x in conv_mults) + " },\n")
            f.write("            .conv1d_scale_shift = { " + ", ".join(str(int(x)) for x in conv_shifts) + " },\n")

            # x_proj: scale -> scale / 2**(12 - MM_Q_XPROJ)
            q_x_w = np.zeros_like(x_proj_w, dtype=np.int8)
            x_mults, x_shifts = [], []
            for r in range(len(x_proj_w)):
                qw, scale = quantize_row_symmetric(x_proj_w[r])
                q_x_w[r] = qw
                m, s = compute_mult_shift(scale / (2 ** (12 - q_xproj)))
                x_mults.append(m)
                x_shifts.append(s)

            f.write("            .x_proj_w = {\n")
            for r in range(len(q_x_w)):
                f.write("                { " + ", ".join(str(int(x)) for x in q_x_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .x_proj_scale_mult = { " + ", ".join(str(int(x)) for x in x_mults) + " },\n")
            f.write("            .x_proj_scale_shift = { " + ", ".join(str(int(x)) for x in x_shifts) + " },\n")

            # dt_proj: scale becomes scale * 2**(11 - MM_Q_XPROJ), bias becomes b / scale * 2**MM_Q_XPROJ
            q_dt_w = np.zeros_like(dt_proj_w, dtype=np.int8)
            dt_mults, dt_shifts, dt_biases = [], [], []
            for d in range(len(dt_proj_w)):
                qw, scale = quantize_row_symmetric(dt_proj_w[d])
                q_dt_w[d] = qw
                m, s = compute_mult_shift(scale * (2 ** (11 - q_xproj)))
                dt_mults.append(m)
                dt_shifts.append(s)
                b_scaled = int(np.round(dt_proj_b[d] / scale * (2 ** q_xproj))) if scale > 0 else 0
                dt_biases.append(b_scaled)

            f.write("            .dt_proj_w = {\n")
            for r in range(len(q_dt_w)):
                f.write("                { " + ", ".join(str(int(x)) for x in q_dt_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .dt_proj_b = { " + ", ".join(str(int(x)) for x in dt_biases) + " },\n")
            f.write("            .dt_proj_scale_mult = { " + ", ".join(str(int(x)) for x in dt_mults) + " },\n")
            f.write("            .dt_proj_scale_shift = { " + ", ".join(str(int(x)) for x in dt_shifts) + " },\n")

            # D_skip in Q12
            d_skip_q12 = np.clip(np.round(D_skip * 4096.0), -32768, 32767).astype(np.int16)
            f.write("            .D_skip = { " + ", ".join(str(int(x)) for x in d_skip_q12) + " },\n")

            # out_proj: scale -> scale / 2**(12 - MM_Q_RESID)
            q_out_w = np.zeros_like(out_proj_w, dtype=np.int8)
            out_mults, out_shifts = [], []
            for r in range(len(out_proj_w)):
                qw, scale = quantize_row_symmetric(out_proj_w[r])
                q_out_w[r] = qw
                m, s = compute_mult_shift(scale / (2 ** (12 - q_resid)))
                out_mults.append(m)
                out_shifts.append(s)

            f.write("            .out_proj_w = {\n")
            for r in range(len(q_out_w)):
                f.write("                { " + ", ".join(str(int(x)) for x in q_out_w[r]) + " },\n")
            f.write("            },\n")
            f.write("            .out_proj_scale_mult = { " + ", ".join(str(int(x)) for x in out_mults) + " },\n")
            f.write("            .out_proj_scale_shift = { " + ", ".join(str(int(x)) for x in out_shifts) + " },\n")
            f.write("        },\n")
        f.write("    },\n")

        # final_norm & head
        final_norm_q14 = np.clip(np.round(final_norm_w * 16384.0), -32768, 32767).astype(np.int16)
        f.write("    .final_norm_w = { " + ", ".join(str(int(x)) for x in final_norm_q14) + " },\n")

        q_head_w = np.zeros_like(head_w, dtype=np.int8)
        head_mults, head_shifts, head_biases = [], [], []
        for c in range(len(head_w)):
            qw, scale = quantize_row_symmetric(head_w[c])
            q_head_w[c] = qw
            m, s = compute_mult_shift(scale)
            head_mults.append(m)
            head_shifts.append(s)
            b_scaled = int(np.round(head_b[c] / scale * 4096.0)) if scale > 0 else 0
            head_biases.append(b_scaled)

        f.write("    .head_w = {\n")
        for r in range(len(q_head_w)):
            f.write("        { " + ", ".join(str(int(x)) for x in q_head_w[r]) + " },\n")
        f.write("    },\n")
        f.write("    .head_b = { " + ", ".join(str(int(x)) for x in head_biases) + " },\n")
        f.write("    .head_scale_mult = { " + ", ".join(str(int(x)) for x in head_mults) + " },\n")
        f.write("    .head_scale_shift = { " + ", ".join(str(int(x)) for x in head_shifts) + " },\n")
        f.write("};\n")

    print(f"Exported {weights_q_c}")

    # 6. Always regenerate engine/gen/mm_luts.c
    luts_c = out_dir / "mm_luts.c"
    print(f"Generating {luts_c}...")
    # SiLU: domain [-8, 8] in Q12 (513 pts, step 128 = 2^7)
    pts_silu = np.linspace(-8.0, 8.0, 513)
    vals_silu = pts_silu / (1.0 + np.exp(-pts_silu))
    vals_silu_q12 = np.clip(np.round(vals_silu * 4096.0), -32768, 32767).astype(np.int16)

    # q = sigmoid(-p): domain [-16, 16] in Q11 (513 pts, step 128 = 2^7)
    pts_q = np.linspace(-16.0, 16.0, 513)
    vals_q = 1.0 / (1.0 + np.exp(pts_q))
    vals_q_q31 = np.clip(np.round(vals_q * 2147483647.0), 0, 2147483647).astype(np.int32)

    # softplus(p): domain [-16, 16] in Q11 (513 pts, step 128 = 2^7)
    vals_sp = np.log1p(np.exp(pts_q))
    vals_sp_q11 = np.clip(np.round(vals_sp * 2048.0), 0, 32767).astype(np.int16)
    vals_sp_q24 = np.clip(np.round(vals_sp * (2 ** 24)), 0, 2147483647).astype(np.int32)

    with open(luts_c, "w") as f:
        f.write("/* Auto-generated power-of-two LUTs for Mu-Mamba C1 fixed-point runtime */\n")
        f.write('#include <stdint.h>\n\n')
        f.write("const int16_t mm_lut_silu_q12[513] = {\n")
        for i in range(0, 513, 8):
            f.write("    " + ", ".join(f"{x:6d}" for x in vals_silu_q12[i:i+8]) + ",\n")
        f.write("};\n\n")

        f.write("const int32_t mm_lut_q_q31[513] = {\n")
        for i in range(0, 513, 8):
            f.write("    " + ", ".join(f"{x:11d}" for x in vals_q_q31[i:i+8]) + ",\n")
        f.write("};\n\n")

        f.write("const int16_t mm_lut_softplus_q11[513] = {\n")
        for i in range(0, 513, 8):
            f.write("    " + ", ".join(f"{x:6d}" for x in vals_sp_q11[i:i+8]) + ",\n")
        f.write("};\n\n")

        f.write("const int32_t mm_lut_softplus_q24[513] = {\n")
        for i in range(0, 513, 8):
            f.write("    " + ", ".join(f"{x:11d}" for x in vals_sp_q24[i:i+8]) + ",\n")
        f.write("};\n\n")
    print(f"Exported {luts_c}")

def main():
    parser = argparse.ArgumentParser(description="Export Mu-Mamba model to C header/sources")
    parser.add_argument("--ckpt", type=str, default="checkpoints/quant_model.pt")
    parser.add_argument("--out-dir", type=str, default="engine/gen")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    args = parser.parse_args()
    export_model(Path(args.ckpt), Path(args.out_dir), args.data_dir)

if __name__ == "__main__":
    main()
