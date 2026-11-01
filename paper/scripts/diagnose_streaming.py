"""Diagnose the streaming / clip-accuracy failures and test decision-rule fixes that need no retraining.

Usage (repo root, GPU, memory-capped):
    PYTHONPATH=. python paper/scripts/diagnose_streaming.py --out paper/data/diagnose_streaming.json \
        --model shipped=checkpoints/quant_model.pt --model stream=checkpoints/stream_model.pt \
        --model learned_a=~/.cache/kws-final/learned_a_ckpt/best_model.pt

Uses PyTorch float logits (the fixed engine agrees on 99.7 % of top-1 decisions), so the decision-rule numbers are
exploratory; confirm the chosen operating point with train/kws/stream_eval.py on the C fixed engine.
Parts:
  1. clip accuracy under three read-outs: last token from zero state (paper protocol), last token after a 1 s
     background pre-roll, and the max keyword margin over the clip;
  2. background accuracy as a function of time since the last keyword ended (does the state "forget"?);
  3. decision-rule grid (EMA alpha x refractory x threshold) selected on a validation stream, reported on a test
     stream: FA per background hour, detection, duplicates.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter

from train.kws.data import SpeechCommandsDataset, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.frontend import extract_log_mel_c
from train.kws.model import MuMambaKWS
from train.kws.stream_eval import build_eval_stream, score_detections

DATA = "data/speech_commands/v0.02"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHAS = [0.5, 0.7, 0.85, 0.93]
REFRACTORY = [25, 50, 75, 100]
THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


def load(path):
    ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    m = MuMambaKWS(d_model=a.get("d_model", 32), d_inner=a.get("d_inner", 64), d_state=a.get("d_state", 8),
                   dt_rank=a.get("dt_rank", 4), d_conv=a.get("d_conv", 4), num_layers=a.get("num_layers", 2),
                   learned_a=a.get("learned_a", False))
    m.load_state_dict(ck["model_state_dict"])
    return m.eval().to(DEV), ck["stats"]


def norm(mel, stats):
    return (mel.astype(np.float32) - stats["mean"].numpy()) / stats["std"].numpy()


@torch.no_grad()
def logits_seq(model, feats):
    return model.forward_seq(torch.from_numpy(feats).float().unsqueeze(0).to(DEV))[0].cpu().numpy()


def clip_protocols(models, test_ds, noise):
    """Returns per-model accuracy for the three read-outs; features are computed once."""
    pre = (noise[:16000].astype(np.float32) * 0.10).astype(np.int16)
    feats0, feats1, labels = [], [], []
    for f, label, _ in test_ds.epoch_samples:
        a = read_wav_int16(f)
        a = np.pad(a, (0, max(0, TARGET_AUDIO_LEN - len(a))))[:TARGET_AUDIO_LEN]
        feats0.append(extract_log_mel_c(a, 50))
        a1 = np.concatenate([pre, a]).astype(np.int16)
        feats1.append(extract_log_mel_c(a1, 100))
        labels.append(label)
    labels = np.array(labels)
    out = {}
    for name, (m, stats) in models.items():
        res = {}
        for tag, F in (("zero_state_last", feats0), ("preroll_1s_last", feats1)):
            x = torch.from_numpy(np.stack([norm(f, stats) for f in F])).float()
            preds, margins = [], []
            for i in range(0, len(x), 256):
                with torch.no_grad():
                    lg = m.forward_seq(x[i:i + 256].to(DEV)).cpu().numpy()
                preds.append(lg[:, -1].argmax(-1))
                if tag == "zero_state_last":
                    margins.append(lg)
            res[tag] = float((np.concatenate(preds) == labels).mean() * 100)
            if margins:
                lg = np.concatenate(margins)                         # (N, 50, 12)
                bg = np.maximum(lg[..., 10], lg[..., 11])
                kw_m = (lg[..., :10] - bg[..., None]).max(1)          # best margin over time per keyword
                pred = np.where(kw_m.max(-1) > 0, kw_m.argmax(-1), np.where(lg[:, -1, 10] > lg[:, -1, 11], 10, 11))
                res["max_margin_over_clip"] = float((pred == labels).mean() * 100)
        out[name] = res
        print(name, res, flush=True)
    return out


def stream_logits(model, stats, audio):
    n = (len(audio) - 512) // 320 + 1
    return logits_seq(model, norm(extract_log_mel_c(audio, n), stats))


def bg_vs_time(logits, events):
    """Background-token accuracy binned by seconds since the previous keyword clip ended (keyword windows excluded)."""
    t = (np.arange(len(logits)) * 320 + 512) / 16000.0
    in_ev = np.zeros(len(t), bool)
    last_end = np.full(len(t), -1e9)
    for s, e, _ in events:
        in_ev |= (t >= s - 0.2) & (t <= e + 0.5)
        last_end = np.where(t > e + 0.5, np.maximum(last_end, e), last_end)
    since = t - last_end
    pred_bg = logits.argmax(-1) >= 10
    bins = [0, 0.5, 1, 2, 3, 4, 6, 1e9]
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (~in_ev) & (since >= lo) & (since < hi)
        if sel.sum():
            out[f"{lo}-{hi if hi < 1e8 else 'inf'}s"] = float(pred_bg[sel].mean() * 100)
    return out


def detect(smooth, th, refractory):
    """Greedy mirror of mm_decide_ex given precomputed smoothed margins (T, 10)."""
    best = smooth.max(1)
    cand = np.nonzero(best > th)[0]
    dets, nxt = [], -1
    for t in cand:
        if t >= nxt:
            dets.append(((t * 320 + 512) / 16000.0, int(smooth[t].argmax())))
            nxt = t + refractory + 1
    return dets


def smoothed(logits, alpha):
    bg = np.maximum(logits[:, 10], logits[:, 11])
    m = logits[:, :10] - bg[:, None]
    return lfilter([1 - alpha], [1, -alpha], m, axis=0)


def grid(logits, events, hours):
    bg_hours = hours - len(events) / 3600.0
    rows = []
    for a in ALPHAS:
        sm = smoothed(logits, a)
        for r in REFRACTORY:
            for th in THRESHOLDS:
                s = score_detections(detect(sm, th, r), events, hours)
                rows.append({"alpha": a, "refractory": r, "threshold": th,
                             "det": s["detection_rate_pct"], "fa_bg_h": s["false_alarms"] / bg_hours,
                             "dup": s["fa_duplicate"], "wrong": s["fa_wrong_class"], "outside": s["fa_outside"]})
    return rows


def pick(rows, target):
    ok = [r for r in rows if r["det"] >= target]
    return min(ok, key=lambda r: (r["fa_bg_h"], -r["det"])) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--num-keywords", type=int, default=500)
    ap.add_argument("--out", default="paper/data/diagnose_streaming.json")
    args = ap.parse_args()
    models = {kv.split("=", 1)[0]: load(kv.split("=", 1)[1]) for kv in args.model}

    val_ds = SpeechCommandsDataset(DATA, split="validation", augment=False)
    test_ds = SpeechCommandsDataset(DATA, split="test", augment=False)
    R = {"clip_protocols": clip_protocols(models, test_ds, val_ds.noise_wavs[0])}

    va, ve, vh, _ = build_eval_stream(val_ds, args.num_keywords, 42)
    ta, te, th_, _ = build_eval_stream(test_ds, args.num_keywords, 42)
    R["streams"] = {"num_keywords": args.num_keywords, "val_hours": vh, "test_hours": th_}
    for name, (m, stats) in models.items():
        vl, tl = stream_logits(m, stats, va), stream_logits(m, stats, ta)
        rv, rt = grid(vl, ve, vh), grid(tl, te, th_)
        res = {"bg_acc_vs_time_since_keyword_test": bg_vs_time(tl, te),
               "shipped_rule_test": next(r for r in rt if r["alpha"] == 0.7 and r["refractory"] == 25 and r["threshold"] == 1.5)}
        for target in (70, 80, 90):
            pv = pick(rv, target)
            res[f"val_pick_det{target}"] = pv
            res[f"test_at_val_pick_det{target}"] = None if pv is None else next(
                r for r in rt if (r["alpha"], r["refractory"], r["threshold"]) == (pv["alpha"], pv["refractory"], pv["threshold"]))
        res["test_grid"] = rt
        R[name] = res
        print(name, {k: v for k, v in res.items() if k != "test_grid"}, flush=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
