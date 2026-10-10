import ctypes
import os
from pathlib import Path
import numpy as np

# Find or load libmumamba.so
_LIB_PATH = Path(__file__).resolve().parents[2] / "engine" / "libmumamba.so"

def _load_lib():
    if not _LIB_PATH.exists():
        raise FileNotFoundError(f"libmumamba.so not found at {_LIB_PATH}. Run 'make -C engine' first.")
    lib = ctypes.CDLL(str(_LIB_PATH))
    
    # void mm_frontend_init(void)
    lib.mm_frontend_init.argtypes = []
    lib.mm_frontend_init.restype = None

    # void mm_frontend_process_frame(const int16_t *pcm_512, int16_t *mel_40_out)
    lib.mm_frontend_process_frame.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16)
    ]
    lib.mm_frontend_process_frame.restype = None
    return lib

_LIB = None

def get_lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
        _LIB.mm_frontend_init()
    return _LIB

FRAME_LEN = 512
FRAME_HOP = 320
MEL_BINS = 40
TARGET_FRAMES = 50  # 50 tokens/s for 1-second clips

def extract_log_mel_c(audio: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Extract 40-bin log-mel features using the bit-exact C frontend.
    Args:
        audio: 1D array of int16 or float32 audio samples at 16 kHz.
        target_frames: Number of output frames (default 50).
    Returns:
        np.ndarray of shape (target_frames, 40), dtype int16.
    """
    lib = get_lib()

    if audio.dtype != np.int16:
        if np.issubdtype(audio.dtype, np.floating):
            # Scale float [-1.0, 1.0] to int16
            audio = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        else:
            audio = audio.astype(np.int16)

    # Required length for target_frames: FRAME_LEN + (target_frames - 1) * FRAME_HOP
    required_len = FRAME_LEN + (target_frames - 1) * FRAME_HOP
    if len(audio) < required_len:
        # Pad with zeros
        pad_len = required_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")
    elif len(audio) > required_len:
        audio = audio[:required_len]

    # Ensure contiguous int16 array
    audio = np.ascontiguousarray(audio, dtype=np.int16)

    features = np.zeros((target_frames, MEL_BINS), dtype=np.int16)
    
    pcm_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    mel_out = np.zeros(MEL_BINS, dtype=np.int16)
    mel_ptr = mel_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

    for t in range(target_frames):
        offset = t * FRAME_HOP
        # Pass pointer to audio slice
        sub_pcm_ptr = ctypes.cast(ctypes.byref(audio.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)).contents, offset * 2),
                                  ctypes.POINTER(ctypes.c_int16))
        lib.mm_frontend_process_frame(sub_pcm_ptr, mel_ptr)
        features[t] = mel_out

    return features
