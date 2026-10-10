import argparse
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from train.kws.data import SpeechCommandsDataset, TARGET_KEYWORDS, CLASS_TO_IDX, IDX_TO_CLASS, read_wav_int16
from train.kws.frontend import extract_log_mel_c, TARGET_FRAMES
from train.kws.model import MuMambaKWS

def build_continuous_stream(
    dataset: SpeechCommandsDataset,
    num_keywords: int = 100,
    min_silence_s: float = 1.0,
    max_silence_s: float = 4.0,
) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
    """
    Constructs a continuous audio stream by joining randomly selected test clips
    interspersed with background noise crops.
    Returns:
      audio_stream: 1D int16 array
      ground_truth_events: list of (start_sec, end_sec, class_idx)
    """
    audio_chunks = []
    events = []
    current_time_s = 0.0

    keyword_indices = [
        i for i, (p, l, *_) in enumerate(dataset.epoch_samples)
        if l < 10 and p is not None
    ]
    selected_indices = random.choices(keyword_indices, k=num_keywords)

    for idx in selected_indices:
        # 1. Preceding silence/noise chunk
        silence_dur = random.uniform(min_silence_s, max_silence_s)
        n_silence_samples = int(silence_dur * 16000)
        silence_chunk = np.zeros(n_silence_samples, dtype=np.int16)
        if dataset.noise_wavs:
            full_noise = dataset._get_silence_crop()
            # Repeat or slice noise to match length
            repeats = (n_silence_samples // len(full_noise)) + 1
            silence_chunk = np.tile(full_noise, repeats)[:n_silence_samples]

        audio_chunks.append(silence_chunk)
        current_time_s += silence_dur

        # 2. Keyword clip
        fpath, label, *_ = dataset.epoch_samples[idx]
        kw_audio = read_wav_int16(fpath)
        if len(kw_audio) < 16000:
            kw_audio = np.pad(kw_audio, (0, 16000 - len(kw_audio)), mode="constant")
        else:
            kw_audio = kw_audio[:16000]

        kw_dur = len(kw_audio) / 16000.0
        events.append((current_time_s, current_time_s + kw_dur, label))
        audio_chunks.append(kw_audio)
        current_time_s += kw_dur

    # Trailing noise
    tail_samples = int(2.0 * 16000)
    tail_silence = np.zeros(tail_samples, dtype=np.int16)
    if dataset.noise_wavs:
        full_noise = dataset._get_silence_crop()
        repeats = (tail_samples // len(full_noise)) + 1
        tail_silence = np.tile(full_noise, repeats)[:tail_samples]
    audio_chunks.append(tail_silence)

    full_audio = np.concatenate(audio_chunks, axis=0)
    return full_audio, events

def evaluate_streaming(
    model: MuMambaKWS,
    dataset: SpeechCommandsDataset,
    threshold: float = 0.6,
    ema_alpha: float = 0.8,
    refractory_tokens: int = 25, # 0.5 s refractory period
    num_keywords: int = 100,
):
    model.eval()
    print("Building continuous audio stream...")
    audio, events = build_continuous_stream(dataset, num_keywords=num_keywords)
    total_hours = len(audio) / (16000.0 * 3600.0)
    print(f"Constructed stream: {len(audio)/16000.0:.1f} seconds ({total_hours:.3f} hours), {len(events)} keyword events")

    # Extract all frames using C log-mel frontend
    hop = 320
    frame_len = 512
    n_frames = (len(audio) - frame_len) // hop + 1

    print("Extracting C log-mel features across stream...")
    mel_features = extract_log_mel_c(audio, target_frames=n_frames) # (T, 40)

    # Streaming evaluation with persistent state (no reset)
    states = model.init_streaming_state(batch_size=1)
    smoothed_probs = np.zeros(12, dtype=np.float32)
    refractory_timer = 0
    detections = [] # list of (time_s, detected_class)

    mean_t = torch.from_numpy(dataset.mean).float().unsqueeze(0)
    std_t = torch.from_numpy(dataset.std).float().unsqueeze(0)

    print("Running sequential streaming inference...")
    with torch.no_grad():
        for t in range(n_frames):
            tok = torch.from_numpy(mel_features[t : t + 1, :]).float() # (1, 40)
            tok = (tok - mean_t) / std_t
            logits, states = model.step(tok, states)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy() # (12,)

            # Exponential Moving Average smoothing
            smoothed_probs = ema_alpha * smoothed_probs + (1.0 - ema_alpha) * probs
            time_s = (t * hop + frame_len) / 16000.0

            if refractory_timer > 0:
                refractory_timer -= 1
                continue

            # Check target keywords (classes 0..9)
            best_kw_idx = int(np.argmax(smoothed_probs[:10]))
            if smoothed_probs[best_kw_idx] >= threshold:
                detections.append((time_s, best_kw_idx))
                refractory_timer = refractory_tokens

    # Match detections to ground-truth events
    matched_events = set()
    true_positives = 0
    false_alarms = 0

    for det_time, det_cls in detections:
        match_found = False
        for ev_idx, (start_s, end_s, true_cls) in enumerate(events):
            # Keyword duration tolerance: detected between start and end + 0.5s
            if start_s - 0.2 <= det_time <= end_s + 0.5:
                if det_cls == true_cls and ev_idx not in matched_events:
                    matched_events.add(ev_idx)
                    true_positives += 1
                    match_found = True
                    break

        if not match_found:
            false_alarms += 1

    total_events = len(events)
    false_negatives = total_events - len(matched_events)
    detection_acc = (true_positives / total_events) * 100.0
    fa_per_hour = false_alarms / total_hours if total_hours > 0 else 0.0

    print("\n=== Continuous Streaming Evaluation Results ===")
    print(f"Total Stream Duration: {total_hours*60:.2f} minutes")
    print(f"Total Target Keywords: {total_events}")
    print(f"True Positives:        {true_positives}")
    print(f"False Negatives:       {false_negatives}")
    print(f"False Alarms:          {false_alarms}")
    print(f"Detection Accuracy:    {detection_acc:.2f}%")
    print(f"False Alarms / Hour:   {fa_per_hour:.2f} FA/hr")
    return detection_acc, fa_per_hour

def main():
    parser = argparse.ArgumentParser(description="Continuous Streaming Evaluation")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--ema-alpha", type=float, default=0.7)
    parser.add_argument("--num-keywords", type=int, default=50)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        print(f"Loading checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args_dict = ckpt.get("args", {})
        model = MuMambaKWS(
            in_dim=40,
            d_model=args_dict.get("d_model", 32),
            d_inner=args_dict.get("d_inner", 64),
            d_state=args_dict.get("d_state", 8),
            dt_rank=args_dict.get("dt_rank", 4),
            d_conv=args_dict.get("d_conv", 4),
            num_layers=args_dict.get("num_layers", 2),
            num_classes=args_dict.get("num_classes", 12),
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with initial model.")
        model = MuMambaKWS()

    dataset = SpeechCommandsDataset(data_dir=args.data_dir, split="test", augment=False)
    evaluate_streaming(
        model=model,
        dataset=dataset,
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        num_keywords=args.num_keywords,
    )

if __name__ == "__main__":
    main()
