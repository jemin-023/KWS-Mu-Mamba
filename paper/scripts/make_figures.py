"""Regenerate every paper figure from the repo state and paper/data/*.json.

Run from the repo root:
    PYTHONPATH=. python paper/scripts/make_figures.py [--suffix pre_fix] [--logit-clips 1000]

Checkpoint-independent figures (LUT / recurrence / rounding math) read engine/gen/mm_luts.c.
Checkpoint-dependent figures read paper/data/*_<suffix>.json; the logit-error histogram
runs the host C engine (engine/libmumamba.so) on a seeded test subset.
"""
import argparse
import ctypes
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
FIGS = ROOT / "paper" / "figures"

# Reference categorical slots 1-3 (dataviz palette, light mode) + neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})

H_Q = 1.0 / 16.0  # q / softplus grid step (p units)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote paper/figures/{name}.pdf/.png")


def parse_lut(name):
    src = (ROOT / "engine" / "gen" / "mm_luts.c").read_text()
    m = re.search(r"const\s+\w+\s+" + re.escape(name) + r"\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\}\s*;", src, re.S)
    return np.array([int(v) for v in re.findall(r"-?\d+", m.group(1))], dtype=np.int64)


def interp(lut, x):
    """Bit-exact mirror of mm_lut_interp_q31 / _s16 (x_min = -32768, step 2^7, 513 entries)."""
    diff = x - (-32768)
    idx = np.clip(diff >> 7, 0, 511)
    frac = diff & 127
    out = lut[idx] + (((lut[idx + 1] - lut[idx]) * frac + 64) >> 7)
    out = np.where(diff <= 0, lut[0], out)
    return np.where((diff >> 7) >= 512, lut[512], out)


def fig_lut_error():
    lut = parse_lut("mm_lut_q_q31")
    p_int = np.arange(-24576, 24577, dtype=np.int64)
    p = p_int / 2048.0
    q = 1.0 / (1.0 + np.exp(p))
    qh = interp(lut, p_int) / 2.0 ** 31
    err = np.abs(qh - q)
    rel = err / (1.0 / (1.0 + np.exp(-p)))  # |dq| / (1 - q)

    cell = lambda v: v[: len(v) // 128 * 128].reshape(-1, 128).max(1)  # envelope: max per 1/16 interpolation cell
    pc = cell(p)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a.semilogy(pc, cell(err), color=BLUE)
    a.axhline(H_Q ** 2 / 8 / (6 * np.sqrt(3)), color=INK2, ls="--", lw=1)
    a.text(-4, H_Q ** 2 / 8 / (6 * np.sqrt(3)) * 2.0, r"$\frac{h^2}{8}\max|f''|$", color=INK2)
    a.set(xlabel=r"pre-activation $\tilde p$", ylabel=r"$|\hat q-\sigma(-\tilde p)|$",
          title="(a) Absolute error of interpolated $q$ (Q31)", ylim=(1e-9, 3e-4))
    b.semilogy(pc, cell(rel), color=BLUE)
    b.axhline(H_Q ** 2 / 8, color=INK2, ls="--", lw=1)
    b.text(1, H_Q ** 2 / 8 * 1.6, r"$h^2/8=4.88\times10^{-4}$", color=INK2)
    i = int(rel.argmax())
    b.plot(p[i], rel[i], "o", ms=4, color=ORANGE)
    b.annotate(f"max {rel[i]:.2e} at p={p[i]:.2f}", (p[i], rel[i]), xytext=(-9, 2e-5), color=INK, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    b.set(xlabel=r"pre-activation $\tilde p$", ylabel=r"$|\delta q|/(1-q)$",
          title="(b) Relative error of the decay rate", ylim=(1e-7, 3e-3))
    save(fig, "fig_lut_q_error")
    return {"q_abs_err_max": float(err.max()), "decay_rate_rel_err_max": float(rel[i]), "at_p": float(p[i])}


def fig_power_error():
    lut = parse_lut("mm_lut_q_q31")
    p_int = np.arange(-24576, 24577, 3, dtype=np.int64)
    q = 1.0 / (1.0 + np.exp(p_int / 2048.0))
    ql = interp(lut, p_int)
    P = ql.copy()
    abs_err, rel_err = [], []
    for n in range(8):
        e = np.abs(P / 2.0 ** 31 - q ** (n + 1))
        abs_err.append(e.max())
        rel_err.append((e / -np.expm1((n + 1) * np.log(q))).max())
        P = (P * ql + (1 << 30)) >> 31
    n = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(n, abs_err, "o-", color=BLUE, ms=4, label=r"measured max $|\hat P_n-q^{n+1}|$")
    ax.plot(n, n * 4.6965e-5, "--", color=INK2, lw=1, label=r"first-order bound $(n{+}1)\,\delta q_{\max}$")
    ax.set(xlabel=r"power $n+1$", ylabel="absolute error", title="Repeated-multiplication decay powers (Q31)")
    ax.legend(frameon=False)
    save(fig, "fig_power_recurrence_error")
    return {"abs_err_per_n": [float(x) for x in abs_err], "decay_rate_rel_err_per_n": [float(x) for x in rel_err]}


def fig_rounding_drift(eps=1e-3, steps=40000, seeds=16):
    """Mean (over seeds) of h_fixed - h_exact for h <- P*h + noise, zero-mean integer noise."""
    P = round((1.0 - eps) * 2 ** 31)
    noise = np.random.default_rng(0).integers(-3, 4, size=(seeds, steps)).astype(np.int64)
    hf = np.zeros(seeds, np.int64); hr = np.zeros(seeds, np.int64); he = np.zeros(seeds)
    ef, er = np.empty(steps), np.empty(steps)
    for i in range(steps):
        hf = ((P * hf) >> 31) + noise[:, i]
        hr = ((P * hr + (1 << 30)) >> 31) + noise[:, i]
        he = (P / 2.0 ** 31) * he + noise[:, i]
        ef[i], er[i] = (hf - he).mean(), (hr - he).mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    db = 0.5 / eps
    ax.axhspan(-db, db, color=GRID, alpha=0.5, lw=0)
    ax.text(steps * 0.55, db * 0.75, "round dead band", color=INK2, fontsize=8)
    ax.plot(er, color=BLUE, label="round-half-up (shipped)")
    ax.plot(ef, color=ORANGE, label="floor (arithmetic shift)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xlabel="time step", ylabel=r"mean $h_{\mathrm{fixed}}-h_{\mathrm{exact}}$ (LSB)",
           title=fr"State bias, $1-P=10^{{-3}}$, {seeds} noise seeds")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig_decay_rounding_drift")
    return {"floor_bias_lsb_last10k": float(ef[-10000:].mean()), "round_bias_lsb_last10k": float(er[-10000:].mean())}


def fig_activation_ranges(suffix):
    f = DATA / f"activation_ranges_{suffix}.json"
    if not f.exists():
        print(f"skip activation ranges ({f.name} missing)")
        return
    R = json.loads(f.read_text())
    sites = ["resid", "x_norm", "u_inproj", "z_gate_silu_domain", "conv_out", "x_proj", "p_clamp", "h", "final_norm"]
    labels = ["residual (Q7)", "RMSNorm out (Q12)", "in_proj u (Q11)", "gate z, SiLU dom.", "conv out / SiLU dom.",
              "x_proj (Q10)", r"$\tilde p$ clamp $\pm$12", "state h (Q22)", "final norm (Q12)"]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    y = np.arange(len(sites))
    for off, key, col, lab in ((-0.2, "clips_600_val", BLUE, "600 val. clips"), (0.2, "streams_3x40kw", ORANGE, "3 val. streams (~2.5 min)")):
        s = R[key]
        r99 = [s[k]["p99.99"] / s[k]["limit"] for k in sites]
        rmax = [s[k]["max"] / s[k]["limit"] for k in sites]
        ax.barh(y + off, r99, height=0.36, color=col, label=f"p99.99, {lab}")
        ax.plot(rmax, y + off, "|", color=INK, ms=7, mew=1.5)
    ax.plot([], [], "|", color=INK, ms=7, mew=1.5, label="max")
    ax.axvline(1.0, color=INK2, ls="--", lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set(xlabel="float magnitude / fixed-point clamp limit  (>1 saturates)",
           title="Float activation magnitude relative to each fixed-point clamp")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    save(fig, f"fig_activation_ranges_{suffix}")


def fig_per_class(suffix):
    f = DATA / f"eval_full_test_{suffix}.json"
    if not f.exists():
        print(f"skip per-class ({f.name} missing)")
        return
    R = json.loads(f.read_text())
    import math
    def wilson(c, n, z=1.96):
        p = c / n; d = 1 + z * z / n
        m = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return 100 * (m - h), 100 * (m + h)
    names = list(R["per_class_fixed"].keys())
    acc = np.array([R["per_class_fixed"][k]["accuracy_pct"] for k in names])
    ci = np.array([wilson(R["per_class_fixed"][k]["correct"], R["per_class_fixed"][k]["total"]) for k in names])
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    y = np.arange(len(names))
    ax.errorbar(acc, y, xerr=[acc - ci[:, 0], ci[:, 1] - acc], fmt="o", color=BLUE, ecolor=INK2, ms=4, capsize=2, lw=1)
    ax.axvline(R["fixed_accuracy_pct"], color=INK2, ls="--", lw=1)
    ax.text(R["fixed_accuracy_pct"] + 0.3, -0.8, f"overall {R['fixed_accuracy_pct']:.2f}%", color=INK2, fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel="test accuracy, W8A16 engine (%) with 95% Wilson CI",
           title=f"Per-class accuracy (n={R['total_clips']})")
    save(fig, f"fig_per_class_{suffix}")


def collect_logit_errors(n_clips, seed=0):
    """Per-clip fixed-vs-float logit error from the host C engines (last token, zero state)."""
    import torch
    from train.kws.data import SpeechCommandsDataset, read_wav_int16, TARGET_AUDIO_LEN
    from train.kws import eval_c as E
    from train.kws.model import MuMambaKWS
    lib = E.load_engine()
    ck = torch.load(ROOT / "checkpoints" / "quant_model.pt", map_location="cpu", weights_only=False)
    m = MuMambaKWS(); m.load_state_dict(ck["model_state_dict"]); m.eval()
    wf = E.populate_c_weights(m, stats=ck.get("stats"))
    wq = E.ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")
    ds = SpeechCommandsDataset(str(ROOT / "data/speech_commands/v0.02"), split="test", augment=False)
    idx = E.select_eval_indices(len(ds), n_clips, seed)
    ctx = E.MMContext(); lib.mm_init(ctypes.byref(ctx))
    linf, labels, pf, pq = [], [], [], []
    for i in idx:
        fpath, label, _ = ds.epoch_samples[i]
        a = read_wav_int16(fpath)
        a = np.pad(a, (0, max(0, TARGET_AUDIO_LEN - len(a))))[:TARGET_AUDIO_LEN].astype(np.int16)
        lf = (ctypes.c_float * 12)(); lq = (ctypes.c_int32 * 12)()
        lib.mm_set_weights_f32(ctypes.byref(wf)); lib.mm_reset(ctypes.byref(ctx))
        E.run_clip_stream(lib, ctx, a, True, lf)
        lib.mm_set_weights_q(ctypes.byref(wq)); lib.mm_reset(ctypes.byref(ctx))
        E.run_clip_stream(lib, ctx, a, False, lq)
        f, q = np.array(lf), np.array(lq) / 4096.0
        linf.append(np.abs(f - q).max()); labels.append(label); pf.append(f.argmax()); pq.append(q.argmax())
    return {"linf": np.array(linf), "label": np.array(labels), "pred_f": np.array(pf), "pred_q": np.array(pq)}


def logit_errors_from_dump(dump):
    """Per-clip fixed-vs-float L-inf from an eval_c --dump-npz file (fixed logits are Q12 integers)."""
    d = np.load(dump)
    f, q = d["logits_float"], d["logits_fixed"].astype(np.float64) / 4096.0
    return {"linf": np.abs(f - q).max(1), "label": d["label"], "pred_f": d["pred_float"], "pred_q": d["pred_fixed"]}


def fig_logit_error(suffix, n_clips):
    cache = DATA / f"logit_errors_{suffix}.npz"
    dump = DATA / f"logits_{suffix}.npz"
    if not cache.exists() and dump.exists():
        np.savez(cache, **logit_errors_from_dump(dump))
    if not cache.exists():
        np.savez(cache, **collect_logit_errors(n_clips))
    d = np.load(cache)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(d["linf"], bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
    med = np.median(d["linf"])
    ax.axvline(med, color=INK2, ls="--", lw=1)
    ax.text(med * 1.1, ax.get_ylim()[1] * 0.85, f"median {med:.3f}", color=INK2, fontsize=8)
    ax.set(xlabel=r"$L_\infty$ logit error, W8A16 vs FP32 C engine", ylabel="clips",
           title=f"Per-clip max logit error (n={len(d['linf'])} test clips, seed 0)")
    save(fig, f"fig_logit_error_{suffix}")


def fig_latency(suffix):
    f = DATA / f"board_bench_{suffix}.json"
    if not f.exists():
        print(f"skip latency ({f.name} missing)")
        return
    B = json.loads(f.read_text())
    mhz = B["cpu_freq_hz"] / 1e6
    fx, fl = B["fixed"], B["float_softfloat_same_model"]
    rows = [("fixed point (int8 weights)", fx["frontend_cycles_median"], fx["step_cycles_median"]),
            ("FP32 soft-float\n(same model)", fl["frontend_cycles_median"], fl["step_f32_cycles_median"])]
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    for i, (name, fe, tot) in enumerate(rows):
        fe_ms, tot_ms = fe / mhz / 1e3, tot / mhz / 1e3
        ax.barh(i, fe_ms, color=AQUA, height=0.5, label="front-end" if i == 0 else None)
        ax.barh(i, tot_ms - fe_ms, left=fe_ms, color=BLUE, height=0.5, label="model" if i == 0 else None)
        ax.text(tot_ms + 0.8, i, f"{tot_ms:.2f} ms ({fe_ms:.2f} front-end)", va="center", color=INK, fontsize=8)
    ax.axvline(20, color=INK2, ls="--", lw=1)
    ax.text(20.5, 0.5, "20 ms hop\n(real-time budget)", color=INK2, fontsize=8, va="center")
    ax.set_xlim(0, 62)
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set(xlabel="per-token latency on ESP32-C3 @160 MHz (ms)", title="On-device per-token latency")
    ax.legend(frameon=False, loc="upper right")
    save(fig, f"fig_latency_{suffix}")


def fig_stream_det(suffix):
    f = DATA / f"stream_sweep_{suffix}.json"
    if not f.exists():
        print(f"skip streaming DET ({f.name} missing; produce it with stream_eval --sweep --json)")
        return
    R = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for eng, col in (("fixed", BLUE), ("float", ORANGE)):
        rows = R["engines"].get(eng)
        if not rows:
            continue
        key = "fa_per_background_hour" if "fa_per_background_hour" in rows[0] else "fa_per_hour"
        fa = [max(r[key], 0.1) for r in rows]
        miss = [100 - r["detection_rate_pct"] for r in rows]
        ax.plot(fa, miss, "o-", ms=3, color=col, label=f"{eng} engine")
    ax.set_xscale("log")
    ax.set(xlabel="false alarms per background hour (log)" if key == "fa_per_background_hour" else "false alarms per hour (log)",
           ylabel="miss rate (%)", title="Streaming detection trade-off")
    ax.legend(frameon=False)
    save(fig, f"fig_stream_det_{suffix}")


def fig_c1_ablation(suffix):
    f = DATA / f"board_bench_{suffix}.json"
    if not f.exists():
        return
    B = json.loads(f.read_text())
    A = B.get("c1_ablation")
    if not A:
        print("skip C1 ablation (no c1_ablation in board_bench json)")
        return
    mhz = B["cpu_freq_hz"] / 1e6
    rows = [("C1 power recurrence\n(1 sigmoid + 1 softplus,\nN-1 multiplies)", A["c1_step_f32_cycles_median"], BLUE),
            ("per-(d, n) exp(-Delta (n+1))\n(N expf per channel)", A["zoh_expf_step_cycles_median"], ORANGE)]
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    for i, (name, cyc, col) in enumerate(rows):
        ms = cyc / mhz / 1e3
        ax.barh(i, ms, color=col, height=0.5)
        ax.text(ms + 1.0, i, f"{ms:.1f} ms ({cyc/1e6:.2f} M cycles)", va="center", color=INK, fontsize=8)
    ratio = A["zoh_expf_step_cycles_median"] / A["c1_step_f32_cycles_median"]
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[1] for r in rows) / mhz / 1e3 * 1.75)
    ax.set(xlabel="FP32 soft-float per-token step on ESP32-C3 @160 MHz (ms)",
           title=f"C1 ablation: same model, ratio {ratio:.2f}x")
    save(fig, f"fig_c1_ablation_{suffix}")


def fig_learned_a(suffix):
    f = DATA / "ablation_c1_vs_learned_a.json"
    if not f.exists():
        print("skip learned-A figure (ablation_c1_vs_learned_a.json missing)")
        return
    R = json.loads(f.read_text())
    if "pairs" not in R:  # old single-pair schema
        R = {"pairs": {"seed42": {"c1": R["c1"], "learned_a": R["learned_a"]}}}
    names = list(R["pairs"])
    arms = (("c1", "frozen C1 A", BLUE), ("learned_a", "learned A", ORANGE))
    marks = ["o", "s", "^", "D", "v"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for x, (k, label, col) in enumerate(arms):
        vals = np.array([R["pairs"][n][k]["test_clip"]["accuracy_pct"] for n in names])
        a.bar(x, vals.mean(), color=col, width=0.5, alpha=0.35,
              yerr=vals.std(ddof=1) if len(vals) > 1 else None, capsize=3, ecolor=INK2)
        for j, n in enumerate(names):
            t = R["pairs"][n][k]["test_clip"]
            a.errorbar(x + (j - (len(names) - 1) / 2) * 0.14, t["accuracy_pct"],
                       yerr=[[t["accuracy_pct"] - t["wilson_ci"][0]], [t["wilson_ci"][1] - t["accuracy_pct"]]],
                       fmt=marks[j % 5], color=col, ms=4, capsize=2, lw=1, label=n if x == 0 else None)
    a.set_xticks([0, 1], [x[1] for x in arms])
    a.set_ylim(min(R["pairs"][n][k]["test_clip"]["wilson_ci"][0] for n in names for k in ("c1", "learned_a")) - 2, None)
    a.set(ylabel="test clip accuracy (%)", title="(a) clip accuracy per seed")
    a.legend(frameon=False, fontsize=7, title="seed", title_fontsize=7)
    S = R.get("streaming_pytorch")
    if S:
        for x, (k, label, col) in enumerate(arms):
            for j, n in enumerate(names):
                t = S[f"{n}/{k}"]["test"]
                b.scatter(t["fa_bg_h"], t["det"], color=col, marker=marks[j % 5], s=22, label=label if j == 0 else None)
        b.set(xlabel="false alarms per background hour (test)", ylabel="detection (%)",
              title="(b) streaming, val-tuned budget")
        b.legend(frameon=False, fontsize=7)
    else:
        b.axis("off")
    save(fig, f"fig_learned_a_{suffix}")


def fig_stream_models():
    """Detection vs false alarms per background hour for every trained model (PyTorch logits, 500-keyword test
    stream, EMA alpha 0.85, refractory 50 tokens), from paper/scripts/diagnose_streaming.py."""
    series = [("diagnose_streaming.json", "shipped", "shipped clip model", BLUE),
              ("diagnose_streaming_warm.json", "warm", "shipped + stream fine-tune", ORANGE),
              ("diagnose_streaming.json", "stream", "stream recipe, C1", AQUA),
              ("diagnose_streaming.json", "learned_a", "stream recipe, learned A", "#eda100")]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    drawn = 0
    for fname, key, label, col in series:
        f = DATA / fname
        if not f.exists() or key not in json.loads(f.read_text()):
            continue
        rows = [r for r in json.loads(f.read_text())[key]["test_grid"]
                if r["alpha"] == 0.85 and r["refractory"] == 50 and r["fa_bg_h"] > 0]   # 0 FA cannot sit on a log axis
        rows.sort(key=lambda r: r["fa_bg_h"])
        ax.plot([r["fa_bg_h"] for r in rows], [r["det"] for r in rows], "o-", ms=3, color=col, label=label)
        drawn += 1
    if not drawn:
        plt.close(fig)
        print("skip stream model comparison (diagnose_streaming*.json missing)")
        return
    ax.set_xscale("log")
    ax.set(xlabel="false alarms per background hour (log)", ylabel="detection (%)",
           title=r"Streaming trade-off ($\alpha$=0.85, refractory 1 s)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    save(fig, "fig_stream_models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="pre_fix")
    ap.add_argument("--logit-clips", type=int, default=1000)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    summary = {"lut_q": fig_lut_error(), "power_recurrence": fig_power_error()}
    summary["rounding_drift"] = fig_rounding_drift()
    fig_activation_ranges(a.suffix)
    fig_per_class(a.suffix)
    fig_logit_error(a.suffix, a.logit_clips)
    fig_latency(a.suffix)
    fig_stream_det(a.suffix)
    fig_c1_ablation(a.suffix)
    fig_learned_a(a.suffix)
    fig_stream_models()
    (DATA / "math_numbers.json").write_text(json.dumps(summary, indent=1))
    print("wrote paper/data/math_numbers.json")


if __name__ == "__main__":
    main()
