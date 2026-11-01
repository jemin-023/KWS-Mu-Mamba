"""PyTorch-only ablation: frozen C1 harmonic A against a learned per-(d, n) A, trained with the identical recipe.

Usage (repo root, memory-capped); one --pair NAME=C1_CKPT,LEARNED_CKPT per seed:
    PYTHONPATH=. python paper/scripts/ablation_learned_a.py \
        --pair seed42=checkpoints/stream_model.pt,~/.cache/kws-final/learned_a_ckpt/best_model.pt \
        --pair seed1=~/.cache/kws-final/c1_seed1_ckpt/best_model.pt,~/.cache/kws-final/learned_a_seed1_ckpt/best_model.pt \
        --out paper/data/ablation_c1_vs_learned_a.json [--stream-keywords 500]

Reports per seed the test clip accuracy with Wilson CIs, the paired difference, mean +- sd per arm over seeds, and (with
--stream-keywords > 0) the PyTorch-float streaming row at the engine decision defaults (alpha 0.85, refractory 50) with the
threshold chosen on a validation stream under a false-alarm budget (100 per background hour), reported on a test stream.

Both models are evaluated on the full test set (zero-state 50-token clip protocol, last-token argmax, as in the
paper's clip accuracy) and on the same 300 seeded validation sequences used for checkpoint selection in training.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.kws.data import SpeechCommandsDataset
from train.kws.eval_c import wilson_interval
from train.kws.model import MuMambaKWS
from train.kws.train import build_val_sequences, evaluate_val_sequences_detailed

DATA_DIR = "data/speech_commands/v0.02"


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = MuMambaKWS(
        d_model=a.get("d_model", 32), d_inner=a.get("d_inner", 64), d_state=a.get("d_state", 8),
        dt_rank=a.get("dt_rank", 4), d_conv=a.get("d_conv", 4), num_layers=a.get("num_layers", 2),
        learned_a=a.get("learned_a", False),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def test_clips(model, loader):
    correct = total = 0
    with torch.no_grad():
        for feats, labels in loader:
            correct += (model(feats).argmax(-1) == labels).sum().item()
            total += len(labels)
    lo, hi = wilson_interval(correct, total)
    return {"correct": correct, "total": total, "accuracy_pct": correct / total * 100.0, "wilson_ci": [lo, hi]}


def describe(path, val_items, loader):
    model, ck = load(path)
    path = Path(path).expanduser()
    rec = {
        "checkpoint": str(path),
        "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": ck.get("epoch"),
        "learned_a": bool(ck.get("args", {}).get("learned_a", False)),
        "init_ckpt": ck.get("init_ckpt"),
        "git_commit": ck.get("git_commit"),
        "seed": ck.get("args", {}).get("seed"),
        "test_clip": test_clips(model, loader),
        "validation_sequences": evaluate_val_sequences_detailed(model, val_items, torch.device("cpu")),
    }
    if rec["learned_a"]:
        # |A| per state index n, over channels and layers; the C1 value is n + 1
        A = torch.stack([torch.exp(l.A_log.detach()) for l in model.layers])  # (L, D_i, N)
        rec["learned_abs_A_per_n"] = {
            "c1_value": list(range(1, A.shape[-1] + 1)),
            "mean": A.mean(dim=(0, 1)).tolist(), "std": A.std(dim=(0, 1)).tolist(),
            "min": A.amin(dim=(0, 1)).tolist(), "max": A.amax(dim=(0, 1)).tolist(),
        }
    return rec


def stream_row(models_by_name, num_keywords, budget=100.0, alpha=0.85, refractory=50):
    """Per model: threshold picked on a validation stream (highest detection within `budget` FA per background hour,
    else the fewest false alarms), then the test-stream row at that setting. PyTorch float logits, 0.4 background h."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import diagnose_streaming as D  # reviewer's exploratory streaming code: same streams, scoring and rule mirror

    val_ds = SpeechCommandsDataset(DATA_DIR, split="validation", augment=False)
    test_ds = SpeechCommandsDataset(DATA_DIR, split="test", augment=False)
    va, ve, vh, _ = D.build_eval_stream(val_ds, num_keywords, 42)
    ta, te, th_, _ = D.build_eval_stream(test_ds, num_keywords, 42)
    out = {"num_keywords": num_keywords, "alpha": alpha, "refractory": refractory, "fa_budget_per_background_hour": budget,
           "val_hours": vh, "test_hours": th_}
    for name, path in models_by_name.items():
        m, stats = D.load(path)
        rv = [r for r in D.grid(D.stream_logits(m, stats, va), ve, vh) if r["alpha"] == alpha and r["refractory"] == refractory]
        rt = [r for r in D.grid(D.stream_logits(m, stats, ta), te, th_) if r["alpha"] == alpha and r["refractory"] == refractory]
        fits = [r for r in rv if r["fa_bg_h"] <= budget]
        pick = (max(fits, key=lambda r: (r["det"], -r["fa_bg_h"])) if fits else min(rv, key=lambda r: (r["fa_bg_h"], -r["det"])))
        test = next(r for r in rt if r["threshold"] == pick["threshold"])
        bg_h = th_ - len(te) / 3600.0
        out[name] = {"budget_reachable": bool(fits), "threshold": pick["threshold"], "validation": pick,
                     "test": dict(test, fa_wrong_class_per_background_hour=test["wrong"] / bg_h,
                                  fa_outside_per_background_hour=test["outside"] / bg_h)}
        print(f"  stream {name}: threshold {pick['threshold']} val det {pick['det']:.1f}% fa/bg-h {pick['fa_bg_h']:.1f} | "
              f"test det {test['det']:.1f}% fa/bg-h {test['fa_bg_h']:.1f}", flush=True)
    return out


def mean_sd(xs):
    xs = np.array(xs, dtype=float)
    return {"n": int(xs.size), "mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if xs.size > 1 else None, "values": xs.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="NAME=C1_CKPT,LEARNED_A_CKPT (one per seed)")
    ap.add_argument("--out", default="paper/data/ablation_c1_vs_learned_a.json")
    ap.add_argument("--trained-torch", action="append", default=[],
                    help="NAME=TORCH_VERSION[,GPU] the pair NAME was trained with (checkpoints do not record it), e.g. seed2=2.11.0+cu128,Tesla T4")
    ap.add_argument("--stream-keywords", type=int, default=0, help="> 0: also report PyTorch streaming rows (e.g. 500)")
    a = ap.parse_args()

    torch.set_num_threads(2)
    test_ds = SpeechCommandsDataset(data_dir=DATA_DIR, split="test", augment=False)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    val_ds = SpeechCommandsDataset(data_dir=DATA_DIR, split="validation", augment=False)
    val_items = build_val_sequences(val_ds, num_sequences=300, seed=1337)

    trained = dict(kv.split("=", 1) for kv in a.trained_torch)
    pairs, paths = {}, {}
    for spec in a.pair:
        name, ckpts = spec.split("=", 1)
        c1, learned = (Path(x).expanduser() for x in ckpts.split(","))
        pairs[name] = {"c1": describe(c1, val_items, loader), "learned_a": describe(learned, val_items, loader)}
        pairs[name]["trained_with"] = trained.get(name, "not recorded")
        pairs[name]["difference_pt"] = (pairs[name]["learned_a"]["test_clip"]["accuracy_pct"]
                                        - pairs[name]["c1"]["test_clip"]["accuracy_pct"])
        paths[f"{name}/c1"], paths[f"{name}/learned_a"] = c1, learned

    out = {
        "note": "PyTorch only (no C engine for learned A). Identical recipe per seed (stream-from-scratch, 60 epochs, 160 "
                "samples/class, batch 16, AdamW wd 1e-4 incl. A_log); test clips use the zero-state 50-token last-token "
                "protocol; validation sequences are the 300 seeded sequences used for checkpoint selection.",
        "evaluated_with_torch": torch.__version__,
        "pairs": pairs,
        "summary": {
            "c1_test_clip_accuracy_pct": mean_sd([p["c1"]["test_clip"]["accuracy_pct"] for p in pairs.values()]),
            "learned_a_test_clip_accuracy_pct": mean_sd([p["learned_a"]["test_clip"]["accuracy_pct"] for p in pairs.values()]),
            "paired_difference_pt": mean_sd([p["difference_pt"] for p in pairs.values()]),
        },
    }
    if a.stream_keywords > 0:
        out["streaming_pytorch"] = stream_row({k: v for k, v in paths.items()}, a.stream_keywords)
    Path(a.out).write_text(json.dumps(out, indent=2))
    for name, p in pairs.items():
        for k in ("c1", "learned_a"):
            t = p[k]["test_clip"]
            print(f"{name:8s} {k:10s} test {t['accuracy_pct']:.2f}% CI {t['wilson_ci'][0]:.2f}-{t['wilson_ci'][1]:.2f}")
        print(f"{name:8s} learned-A minus C1: {p['difference_pt']:+.2f} pt")
    sm = out["summary"]
    print("mean +- sd  C1 {:.2f} +- {}  learned-A {:.2f} +- {}  paired diff {:+.2f} +- {}".format(
        sm["c1_test_clip_accuracy_pct"]["mean"], sm["c1_test_clip_accuracy_pct"]["sd"],
        sm["learned_a_test_clip_accuracy_pct"]["mean"], sm["learned_a_test_clip_accuracy_pct"]["sd"],
        sm["paired_difference_pt"]["mean"], sm["paired_difference_pt"]["sd"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
