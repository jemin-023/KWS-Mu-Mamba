#!/usr/bin/env python3
"""
Compute per-bin mean and scale (standard deviation) across the Speech Commands training set
using the C BFP frontend.
Saves to:
  - train/kws/frontend_stats.pt (PyTorch dict with 'mean' and 'std' tensors)
  - engine/gen/mm_stats.json (JSON for C export)
"""
import glob
import json
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from pathlib import Path
from tqdm import tqdm
from train.kws.frontend import extract_log_mel_c

def main():
    wav_paths = glob.glob("data/speech_commands/v0.02/train/*/*.wav")
    print(f"Computing per-bin statistics over {len(wav_paths)} training clips...")

    # We compute online running sum and sum of squares for 40 mel bins
    total_frames = 0
    sum_feats = np.zeros(40, dtype=np.float64)
    sum_sq_feats = np.zeros(40, dtype=np.float64)

    for p in tqdm(wav_paths, desc="Processing WAVs"):
        try:
            sr, audio = wavfile.read(p)
        except Exception:
            continue
        feats = extract_log_mel_c(audio, target_frames=50) # (50, 40), int16
        feats_f = feats.astype(np.float64)
        sum_feats += np.sum(feats_f, axis=0)
        sum_sq_feats += np.sum(feats_f ** 2, axis=0)
        total_frames += feats.shape[0]

    mean = sum_feats / total_frames
    variance = (sum_sq_feats / total_frames) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 1e-6))

    print(f"Computed stats over {total_frames} frames ({len(wav_paths)} clips):")
    print("Mean (first 10 bins):", np.round(mean[:10], 1))
    print("Std  (first 10 bins):", np.round(std[:10], 1))

    # Save to PyTorch pt file
    pt_path = Path("train/kws/frontend_stats.pt")
    torch.save({
        "mean": torch.from_numpy(mean).float(),
        "std": torch.from_numpy(std).float(),
    }, pt_path)
    print(f"Saved PyTorch stats to {pt_path}")

    # Save to JSON
    json_path = Path("engine/gen/mm_stats.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist(),
        }, f, indent=2)
    print(f"Saved JSON stats to {json_path}")

if __name__ == "__main__":
    main()
