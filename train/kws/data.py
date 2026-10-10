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
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.augment = augment
        self.target_frames = target_frames
        self.noise_prob = noise_prob
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

    def resample_epoch(self):
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
            for f in files:
                self.epoch_samples.append((f, CLASS_TO_IDX[kw], None))
            counts.append(len(files))

        avg_count = int(np.mean(counts)) if counts else 2000

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

        # Pad or truncate to 16,192 samples (exactly 50 tokens)
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
