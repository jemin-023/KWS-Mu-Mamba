"""Float activation ranges at every fixed-point clamp site, on clips and long streams.

Usage (repo root): PYTHONPATH=. python paper/scripts/activation_ranges.py paper/data/activation_ranges_<suffix>.json [--ckpt PATH]
Q formats are read from engine/gen/mm_config.h; the checkpoint defaults to checkpoints/quant_model.pt.
"""
import json, random, re, sys
import numpy as np, torch, torch.nn.functional as F
from train.kws.data import SpeechCommandsDataset, build_continuous_stream
from train.kws.frontend import extract_log_mel_c
from train.kws.model import MuMambaKWS

torch.set_num_threads(4)
_ckpt = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else "checkpoints/quant_model.pt"
ck = torch.load(_ckpt, map_location="cpu", weights_only=False)
m = MuMambaKWS(); m.load_state_dict(ck["model_state_dict"]); m.eval()
_cfg = open("engine/gen/mm_config.h").read()
QR, QX, QI = (int(re.search(rf"#define {k}\s+(\d+)", _cfg).group(1)) for k in ("MM_Q_RESID", "MM_Q_XPROJ", "MM_Q_INPROJ"))
lim = {"resid": 2**(15-QR), "x_norm": 8.0, "u_inproj": 2**(15-QI), "z_inproj": 2**(15-QI),
       "z_gate_silu_domain": 8.0, "conv_out": 8.0, "x_proj": 2**(15-QX), "p_clamp": 12.0,
       "h": 2**(31-12-QX), "final_norm": 8.0}

def trace(x):
    out = {k: [] for k in lim}; extra = {"one_minus_q": [], "h_abs": [], "deadband_frac": []}
    h = m.stem(x); out["resid"].append(h)
    for L in m.layers:
        xn = L.norm(h); out["x_norm"].append(xn)
        xz = L.in_proj(xn); u, z = xz.split([64, 64], -1)
        out["u_inproj"].append(u); out["z_inproj"].append(z); out["z_gate_silu_domain"].append(z)
        uc = L.conv1d(F.pad(u.transpose(1, 2), (3, 0))).transpose(1, 2); out["conv_out"].append(uc)
        up = F.silu(uc); xs = L.x_proj(up); out["x_proj"].append(xs)
        r, B, C = xs.split([4, 8, 8], -1); p = L.dt_proj(r); out["p_clamp"].append(p)
        pc = p.clamp(-12, 12); q = torch.sigmoid(-pc); D = F.softplus(pc)
        Bsz, T, _ = up.shape; hs = torch.zeros(Bsz, 64, 8); ys = []
        for t in range(T):
            P = torch.stack([q[:, t] ** (n + 1) for n in range(8)], -1)
            hs = P * hs + (D[:, t] * up[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(-2)
            out["h"].append(hs)
            # dead band of round-half-up decay in LSB of Q(12+QX): |h| < 0.5/(1-P)
            db = 0.5 / (1 - P).clamp(min=1e-12) / 2**(12 + QX)
            extra["deadband_frac"].append(((hs.abs() < db) & (hs.abs() > 0)).float().mean().reshape(1))
            ys.append(torch.einsum("bdn,bn->bd", hs, C[:, t]))
        extra["one_minus_q"].append(torch.sigmoid(pc).reshape(-1))
        extra["h_abs"].append(torch.stack(ys, 1).abs().reshape(-1))
        y = torch.stack(ys, 1) + L.D_skip * up
        h = h + L.out_proj(y * F.silu(z)); out["resid"].append(h)
    out["final_norm"].append(m.final_norm(h))
    return out, extra

def summarize(tag, xs):
    tot = {k: [] for k in lim}; ex = {"one_minus_q": [], "deadband_frac": []}
    with torch.no_grad():
        for x in xs:
            o, e = trace(x)
            for k in lim: tot[k].append(torch.cat([t.reshape(-1) for t in o[k]]).abs())
            ex["one_minus_q"] += e["one_minus_q"]; ex["deadband_frac"] += e["deadband_frac"]
    res = {}
    for k in lim:
        a = torch.cat(tot[k]).numpy()
        res[k] = {"limit": lim[k], "max": float(a.max()), "p99": float(np.percentile(a, 99)),
                  "p99.99": float(np.percentile(a, 99.99)), "frac_over": float((a > lim[k]).mean())}
    omq = torch.cat(ex["one_minus_q"]).numpy()
    res["one_minus_q"] = {"min": float(omq.min()), "p1": float(np.percentile(omq, 1)), "median": float(np.median(omq))}
    res["deadband_frac_mean"] = float(torch.cat(ex["deadband_frac"]).mean())
    print(f"== {tag}")
    for k, v in res.items(): print(f"  {k:20s} {v}")
    return res

ds = SpeechCommandsDataset("data/speech_commands/v0.02", split="validation", augment=False)
rng = random.Random(0); idx = rng.sample(range(len(ds)), 600)
clips = torch.stack([torch.as_tensor(ds[i][0]) for i in idx]).float()
R = {"clips_600_val": summarize("600 validation clips (50 tokens)", [clips[i:i+100] for i in range(0, 600, 100)])}
srng = random.Random(7); streams = []
for _ in range(3):
    a, _, _ = build_continuous_stream(ds, num_keywords=40, is_training=False, rng=srng)
    mel = extract_log_mel_c(a, target_frames=(len(a) - 512) // 320 + 1).astype(np.float32)
    streams.append(torch.from_numpy((mel - ds.mean) / ds.std).unsqueeze(0))
R["streams_3x40kw"] = summarize("3 validation streams x 40 keywords (~2.5 min each)", streams)
json.dump(R, open(sys.argv[1], "w"), indent=1)
