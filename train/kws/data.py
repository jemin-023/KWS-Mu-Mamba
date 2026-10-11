import os
import wave
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .frontend import extract_log_mel_c, TARGET_FRAMES, MEL_BINS

TARGET_KEYWORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]
CLASS_TO_IDX = {kw: i for i, kw in enumerate(TARGET_KEYWORDS)}
CLASS_TO_IDX["unknown"] = 10
CLASS_TO_IDX["silence"] = 11

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = 12

# Audio sample rate and exact length for 50 tokens (512 frame length + 49 hops of 320)
SAMPLE_RATE = 16000
TARGET_AUDIO_LEN = 16192

def read_wav_int16(path: Path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit WAV file into an int16 numpy array."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        audio = audio[::n_channels]
    return audio

class SpeechCommandsDataset(Dataset):
    """
    12-class Speech Commands v0.02 Dataset:
      10 keywords + unknown + silence.
      - 25 other classes are mapped to 'unknown'.
      - Deterministic, frozen validation set (fixed unknown sampling, fixed silence crops).
      - Test set uses official 4,890 clips (balanced 408 per class).
      - Audio padded to 16,192 samples so C frontend extracts exactly 50 tokens.
      - Per-bin mean and scale normalization using frontend_stats.pt.
    """
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        augment: bool = False,
        target_frames: int = TARGET_FRAMES,
        noise_prob: float = 0.8,
        stats_path: Optional[Path] = None,
        mode: str = "clip",
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.augment = augment
        self.target_frames = target_frames
        self.noise_prob = noise_prob
        self.mode = mode
        self.sample_rate = SAMPLE_RATE
        self.target_audio_len = TARGET_AUDIO_LEN

        # Normalization stats
        if stats_path is None:
            stats_path = Path(__file__).parent / "frontend_stats.pt"
        if stats_path.exists():
            stats = torch.load(stats_path)
            self.mean = stats["mean"].numpy() # (40,)
            self.std = stats["std"].numpy()   # (40,)
        else:
            self.mean = np.zeros(MEL_BINS, dtype=np.float32)
            self.std = np.ones(MEL_BINS, dtype=np.float32)

        self.keyword_files: Dict[str, List[Path]] = {kw: [] for kw in TARGET_KEYWORDS}
        self.unknown_files: List[Path] = []
        self.noise_wavs: List[np.ndarray] = []
        self.test_silence_files: List[Path] = []
        self.val_silence_crops: List[np.ndarray] = []

        self._scan_files()
        self.epoch_samples: List[Tuple[Optional[Path], int, Optional[int]]] = []
        self.resample_epoch()

    def _scan_files(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset split dir {self.data_dir} does not exist.")

        for p in self.data_dir.iterdir():
            if not p.is_dir():
                continue
            cname = p.name
            if cname == "_silence_":
                silence_files = sorted(list(p.glob("*.wav")))
                if self.split == "test":
                    self.test_silence_files = silence_files
                for nw in silence_files:
                    try:
                        self.noise_wavs.append(read_wav_int16(nw))
                    except Exception as e:
                        pass
            elif cname in TARGET_KEYWORDS:
                self.keyword_files[cname].extend(sorted(list(p.glob("*.wav"))))
            else:
                self.unknown_files.extend(sorted(list(p.glob("*.wav"))))

        # If validation has few noise files, fallback to train noise files for background
        if not self.noise_wavs and self.split == "validation":
            train_silence = self.data_dir.parent / "train" / "_silence_"
            if train_silence.exists():
                for nw in sorted(list(train_silence.glob("*.wav"))):
                    try:
                        self.noise_wavs.append(read_wav_int16(nw))
                    except Exception:
                        pass

    def resample_epoch(self, max_samples_per_class: Optional[int] = None):
        """
        Rebalance unknown and silence class samples for the epoch.
        For validation: runs ONCE deterministically (fixed seed 42).
        For test: uses all test clips (no resampling needed).
        For train: resamples unknown and synthetic silence dynamically.
        """
        self.epoch_samples = []

        # 1. Target keywords
        counts = []
        for kw, files in self.keyword_files.items():
            if max_samples_per_class is not None and self.split == "train":
                sampled_kw = random.sample(files, min(max_samples_per_class, len(files)))
                for f in sampled_kw:
                    self.epoch_samples.append((f, CLASS_TO_IDX[kw], None))
                counts.append(len(sampled_kw))
            else:
                for f in files:
                    self.epoch_samples.append((f, CLASS_TO_IDX[kw], None))
                counts.append(len(files))

        avg_count = int(np.mean(counts)) if counts else 2000
        if max_samples_per_class is not None and self.split == "train":
            avg_count = min(avg_count, max_samples_per_class)

        # 2. Unknown class
        if self.unknown_files:
            if self.split == "test":
                for f in self.unknown_files:
                    self.epoch_samples.append((f, CLASS_TO_IDX["unknown"], None))
            elif self.split == "validation":
                val_rng = random.Random(42)
                sampled_unknown = val_rng.sample(self.unknown_files, min(avg_count, len(self.unknown_files)))
                for f in sampled_unknown:
                    self.epoch_samples.append((f, CLASS_TO_IDX["unknown"], None))
            else:
                sampled_unknown = random.sample(self.unknown_files, min(avg_count, len(self.unknown_files)))
                for f in sampled_unknown:
                    self.epoch_samples.append((f, CLASS_TO_IDX["unknown"], None))

        # 3. Silence class
        if self.split == "test":
            for f in self.test_silence_files:
                self.epoch_samples.append((f, CLASS_TO_IDX["silence"], None))
        elif self.split == "validation":
            # Generate deterministic fixed silence crops
            if not self.val_silence_crops:
                val_rng = random.Random(1337)
                for _ in range(avg_count):
                    if self.noise_wavs:
                        noise = val_rng.choice(self.noise_wavs)
                        if len(noise) <= self.target_audio_len:
                            crop = np.pad(noise, (0, self.target_audio_len - len(noise)))
                        else:
                            st = val_rng.randint(0, len(noise) - self.target_audio_len)
                            crop = noise[st : st + self.target_audio_len]
                    else:
                        crop = np.zeros(self.target_audio_len, dtype=np.int16)
                    self.val_silence_crops.append(crop)
            for i in range(len(self.val_silence_crops)):
                self.epoch_samples.append((None, CLASS_TO_IDX["silence"], i))
        else:
            # Training: generate synthetic silence indicators
            for _ in range(avg_count):
                self.epoch_samples.append((None, CLASS_TO_IDX["silence"], None))

        if self.split == "train":
            random.shuffle(self.epoch_samples)

    def __len__(self) -> int:
        return len(self.epoch_samples)

    def _get_silence_crop(self) -> np.ndarray:
        if not self.noise_wavs:
            return np.zeros(self.target_audio_len, dtype=np.int16)
        noise = random.choice(self.noise_wavs)
        if len(noise) <= self.target_audio_len:
            return np.pad(noise, (0, self.target_audio_len - len(noise)))
        start = random.randint(0, len(noise) - self.target_audio_len)
        return noise[start : start + self.target_audio_len]

    def _apply_audio_augmentations(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)

        # 1. Time shift: +/- 100 ms (+/- 1600 samples)
        shift = random.randint(-1600, 1600)
        if shift > 0:
            audio = np.pad(audio[shift:], (0, shift), mode="constant")
        elif shift < 0:
            audio = np.pad(audio[:shift], (-shift, 0), mode="constant")

        # 2. Background noise mixing
        if self.noise_wavs and random.random() < self.noise_prob:
            noise_crop = self._get_silence_crop().astype(np.float32)
            snr_factor = random.uniform(0.0, 0.15)
            audio = audio + snr_factor * noise_crop

        # 3. Random gain (+/- 3 dB, ~0.7 to 1.3)
        gain = random.uniform(0.7, 1.3)
        audio = audio * gain

        return np.clip(audio, -32768.0, 32767.0).astype(np.int16)

    def _apply_spec_augment(self, mel: np.ndarray) -> np.ndarray:
        """Time and frequency masking on (T, 40) features."""
        mel = mel.copy()
        # Frequency masking (up to 4 bins)
        f_width = random.randint(0, 4)
        f_start = random.randint(0, max(0, MEL_BINS - f_width))
        mel[:, f_start : f_start + f_width] = 0.0

        # Time masking (up to 8 frames)
        t_width = random.randint(0, 8)
        t_start = random.randint(0, max(0, self.target_frames - t_width))
        mel[t_start : t_start + t_width, :] = 0.0

        return mel

    def __getitem__(self, idx: int):
        fpath, label, silence_idx = self.epoch_samples[idx]

        if fpath is None:
            if silence_idx is not None and silence_idx < len(self.val_silence_crops):
                audio = self.val_silence_crops[silence_idx]
            else:
                audio = self._get_silence_crop()
        else:
            try:
                audio = read_wav_int16(fpath)
            except Exception:
                audio = np.zeros(self.target_audio_len, dtype=np.int16)

        if self.mode == "sequence":
            audio, targets, _ = build_continuous_stream(
                dataset=self,
                clip_info=(fpath, label, silence_idx),
                is_training=(self.split == "train"),
            )
            n_tokens = len(targets)
            mel = extract_log_mel_c(audio, target_frames=n_tokens)
            mel_float = mel.astype(np.float32)
            mel_norm = (mel_float - self.mean) / self.std
            return torch.from_numpy(mel_norm).float(), torch.from_numpy(targets).long()

        # Mode == "clip": pad or truncate to 16,192 samples (exactly 50 tokens)
        if len(audio) < self.target_audio_len:
            audio = np.pad(audio, (0, self.target_audio_len - len(audio)), mode="constant")
        elif len(audio) > self.target_audio_len:
            audio = audio[:self.target_audio_len]

        if self.augment and label != CLASS_TO_IDX["silence"]:
            audio = self._apply_audio_augmentations(audio)

        # Extract bit-exact C log-mel features: shape (50, 40), dtype int16
        mel = extract_log_mel_c(audio, target_frames=self.target_frames)
        mel_float = mel.astype(np.float32)

        # Per-bin feature normalization: (mel - mean) / std
        mel_norm = (mel_float - self.mean) / self.std

        if self.augment:
            mel_norm = self._apply_spec_augment(mel_norm)

        return mel_norm, label

def build_continuous_stream(
    dataset: SpeechCommandsDataset,
    num_keywords: Optional[int] = None,
    clip_info: Optional[Tuple[Optional[Path], int, Optional[int]]] = None,
    is_training: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, int]]]:
    """
    Generalized sequence composer for training and streaming evaluation.
    Returns:
      audio: 1D int16 array
      targets: 1D int64 per-token target array (-100 = ignore)
      events: list of (start_sec, end_sec, class_idx)
    """
    if rng is None:
        rng = random.Random()

    events: List[Tuple[float, float, int]] = []
    timeline_clips = []
    current_sample = 0

    if num_keywords is None:
        # Single-clip training/validation sequence
        preroll_s = rng.uniform(0.0, 1.5) if is_training else 0.5
        preroll_samples = int(preroll_s * SAMPLE_RATE)

        # Distractor in pre-roll with probability 0.3
        has_unk = is_training and (rng.random() < 0.3) and len(dataset.unknown_files) > 0
        if has_unk:
            gap1_samples = preroll_samples // 2
            current_sample += gap1_samples
            unk_path = rng.choice(dataset.unknown_files)
            try:
                unk_audio = read_wav_int16(unk_path)
            except Exception:
                unk_audio = np.zeros(16000, dtype=np.int16)
            if len(unk_audio) < 16000:
                unk_audio = np.pad(unk_audio, (0, 16000 - len(unk_audio)))
            else:
                unk_audio = unk_audio[:16000]
            timeline_clips.append((current_sample, unk_audio, CLASS_TO_IDX["unknown"]))
            current_sample += 16000
            current_sample += (preroll_samples - gap1_samples)
        else:
            current_sample += preroll_samples

        # Main clip
        if clip_info is not None:
            fpath, label, s_idx = clip_info
        else:
            sample_item = rng.choice(dataset.epoch_samples)
            fpath, label = sample_item[0], sample_item[1]
            s_idx = sample_item[2] if len(sample_item) > 2 else None

        if fpath is None:
            if s_idx is not None and s_idx < len(dataset.val_silence_crops):
                main_audio = dataset.val_silence_crops[s_idx]
            else:
                main_audio = dataset._get_silence_crop()
        else:
            try:
                main_audio = read_wav_int16(fpath)
            except Exception:
                main_audio = np.zeros(16000, dtype=np.int16)

        if len(main_audio) < 16000:
            main_audio = np.pad(main_audio, (0, 16000 - len(main_audio)))
        else:
            main_audio = main_audio[:16000]

        timeline_clips.append((current_sample, main_audio, label))
        if label < 10:
            events.append((current_sample / SAMPLE_RATE, (current_sample + 16000) / SAMPLE_RATE, label))
        current_sample += 16000

        # Post-roll
        postroll_s = rng.uniform(0.0, 1.0) if is_training else 0.5
        postroll_samples = int(postroll_s * SAMPLE_RATE)
        current_sample += postroll_samples
        total_samples = current_sample
    else:
        # Multi-keyword stream for continuous evaluation
        keyword_indices = [
            i for i, (p, l, *_) in enumerate(dataset.epoch_samples)
            if l < 10 and p is not None
        ]
        selected_indices = [rng.choice(keyword_indices) for _ in range(num_keywords)]

        for idx in selected_indices:
            silence_dur = rng.uniform(1.0, 4.0)
            gap_samples = int(silence_dur * SAMPLE_RATE)

            if rng.random() < 0.3 and dataset.unknown_files:
                gap1 = gap_samples // 2
                current_sample += gap1
                unk_path = rng.choice(dataset.unknown_files)
                try:
                    unk_audio = read_wav_int16(unk_path)
                except Exception:
                    unk_audio = np.zeros(16000, dtype=np.int16)
                if len(unk_audio) < 16000:
                    unk_audio = np.pad(unk_audio, (0, 16000 - len(unk_audio)))
                else:
                    unk_audio = unk_audio[:16000]
                timeline_clips.append((current_sample, unk_audio, CLASS_TO_IDX["unknown"]))
                current_sample += 16000
                current_sample += (gap_samples - gap1)
            else:
                current_sample += gap_samples

            fpath, label, *_ = dataset.epoch_samples[idx]
            try:
                kw_audio = read_wav_int16(fpath)
            except Exception:
                kw_audio = np.zeros(16000, dtype=np.int16)
            if len(kw_audio) < 16000:
                kw_audio = np.pad(kw_audio, (0, 16000 - len(kw_audio)))
            else:
                kw_audio = kw_audio[:16000]

            timeline_clips.append((current_sample, kw_audio, label))
            events.append((current_sample / SAMPLE_RATE, (current_sample + 16000) / SAMPLE_RATE, label))
            current_sample += 16000

        current_sample += int(2.0 * SAMPLE_RATE)
        total_samples = current_sample

    speech_track = np.zeros(total_samples, dtype=np.float32)
    for st_samp, clip_pcm, _ in timeline_clips:
        ed_samp = min(total_samples, st_samp + len(clip_pcm))
        speech_track[st_samp : ed_samp] += clip_pcm[: ed_samp - st_samp].astype(np.float32)

    is_digital_silence = is_training and (rng.random() < 0.10)
    if is_digital_silence or not dataset.noise_wavs:
        noise_track = np.zeros(total_samples, dtype=np.float32)
    else:
        raw_noise = rng.choice(dataset.noise_wavs).astype(np.float32)
        repeats = (total_samples // len(raw_noise)) + 2
        tiled = np.tile(raw_noise, repeats)
        st_offset = rng.randint(0, len(raw_noise) - 1)
        noise_crop = tiled[st_offset : st_offset + total_samples]
        noise_level = rng.uniform(0.05, 0.20) if is_training else 0.10
        noise_track = noise_crop * noise_level

    audio_float = speech_track + noise_track

    # One gain per sequence, log-uniform from -18 dB to +6 dB
    gain_db = rng.uniform(-18.0, 6.0) if is_training else 0.0
    gain_linear = 10.0 ** (gain_db / 20.0)
    audio_float = audio_float * gain_linear
    audio = np.clip(audio_float, -32768.0, 32767.0).astype(np.int16)

    # Compute per-token targets
    hop = 320
    frame_len = 512
    n_tokens = (len(audio) - frame_len) // hop + 1
    targets = np.full(n_tokens, CLASS_TO_IDX["silence"], dtype=np.int64)

    for st_samp, _, clip_label in timeline_clips:
        for t in range(n_tokens):
            tok_start = t * hop
            rel_samp = tok_start - st_samp
            if 0 <= rel_samp < 16000:
                rel_tok = rel_samp // hop
                if rel_tok < 40:
                    targets[t] = -100
                else:
                    targets[t] = clip_label
            elif 16000 <= rel_samp < 16000 + 4800:
                targets[t] = -100

    return audio, targets, events

def collate_sequences(batch):
    """Pads variable length sequence features and targets in batch."""
    max_len = max(item[0].shape[0] for item in batch)
    b_sz = len(batch)
    feats = torch.zeros(b_sz, max_len, 40, dtype=torch.float32)
    targets = torch.full((b_sz, max_len), -100, dtype=torch.long)
    for i, (f, tg) in enumerate(batch):
        L = f.shape[0]
        feats[i, :L] = f
        targets[i, :L] = tg
    return feats, targets
