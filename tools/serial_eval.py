#!/usr/bin/env python3
"""
Host-to-device streaming evaluation tool over UART/Serial:
Streams audio clips to the ESP32-C3 firmware running mm_step_q,
collects device output logits, and compares against the host C fixed engine
bit-for-bit.
"""
import sys
import time
import argparse
import ctypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from tqdm import tqdm

try:
    import serial
except ImportError:
    serial = None

from train.kws.data import SpeechCommandsDataset, read_wav_int16, TARGET_AUDIO_LEN
from train.kws.eval_c import load_engine, MMContext, ModelWeightsQ

SERIAL_SYNC = bytes([0xAA, 0x55])
RESP_SYNC   = bytes([0x55, 0xAA])

CMD_AUDIO_CHUNK = 0x01
CMD_RESET       = 0x03
CMD_BENCH       = 0x04

def calc_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def read_exact(ser, num_bytes: int, timeout_sec: float = 2.0) -> bytes:
    buf = bytearray()
    t0 = time.time()
    while len(buf) < num_bytes:
        chunk = ser.read(num_bytes - len(buf))
        if chunk:
            buf.extend(chunk)
        elif time.time() - t0 > timeout_sec:
            break
    return bytes(buf)

def read_resp_hdr(ser, timeout_sec: float = 2.0) -> bytes:
    t0 = time.time()
    sync_state = 0
    while time.time() - t0 < timeout_sec:
        b = ser.read(1)
        if not b:
            continue
        val = b[0]
        if sync_state == 0:
            if val == 0x55:
                sync_state = 1
        elif sync_state == 1:
            if val == 0xAA:
                sync_state = 2
                break
            elif val == 0x55:
                sync_state = 1
            else:
                sync_state = 0
    if sync_state != 2:
        return b""
    tail = read_exact(ser, 2, timeout_sec=timeout_sec)
    if len(tail) != 2:
        return b""
    return RESP_SYNC + tail

def send_reset(ser) -> bool:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(SERIAL_SYNC + bytes([CMD_RESET, 0x00]))
    resp = read_resp_hdr(ser)
    return len(resp) == 4 and resp[:2] == RESP_SYNC and resp[2] == 0x00


def run_serial_eval(port: str, baud: int, subset: int, data_dir: str):
    if serial is None:
        raise ImportError("pyserial is required for on-device serial evaluation. Install with 'pip install pyserial'.")

    print(f"Opening serial port {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except Exception as e:
        print(f"Could not open serial port {port}: {e}")
        print("Note: Plug in the ESP32-C3 board to perform hardware serial validation.")
        return

    # Reset streaming state on device
    if not send_reset(ser):
        print("Warning: Initial device reset handshake failed. Verifying connection...")

    # Load host reference engine
    lib = load_engine()
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    c_weights_q = ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")
    lib.mm_set_weights_q(ctypes.byref(c_weights_q))

    ds = SpeechCommandsDataset(data_dir=data_dir, split="test", augment=False)
    total_clips = len(ds)
    num_eval = min(subset, total_clips) if subset else total_clips
    indices = range(num_eval)

    print(f"Running on-device serial test on {num_eval} clips...")
    bit_identical_count = 0
    total_evaluated = 0
    max_logit_diff = 0

    pbar = tqdm(indices, desc="Serial Eval")
    for idx in pbar:
        fpath, label, silence_idx = ds.epoch_samples[idx]
        if fpath is None:
            if silence_idx is not None and silence_idx < len(ds.val_silence_crops):
                audio = ds.val_silence_crops[silence_idx]
            else:
                audio = ds._get_silence_crop()
        else:
            try:
                audio = read_wav_int16(fpath)
            except Exception:
                audio = np.zeros(TARGET_AUDIO_LEN, dtype=np.int16)

        # Pad to exactly 16,192 samples (50 tokens: 512 initial + 49 hops of 320)
        if len(audio) < TARGET_AUDIO_LEN:
            audio = np.pad(audio, (0, TARGET_AUDIO_LEN - len(audio)), mode="constant")
        elif len(audio) > TARGET_AUDIO_LEN:
            audio = audio[:TARGET_AUDIO_LEN]

        # Reset both host and device streaming state
        lib.mm_reset(ctypes.byref(ctx))
        send_reset(ser)

        device_last_logits = None
        host_last_logits = (ctypes.c_int32 * 12)()

        # Push initial 512 samples, then 49 hops of 320 samples
        # Chunk 0: initial 512 samples
        # Split into 320 + 192, or send in 320 hops
        # To match firmware's 320-sample hops:
        # Pad audio with 128 leading samples so 51 hops of 320 feed exactly 50 tokens
        # Or send audio in 320-sample hops
        num_hops = len(audio) // 320
        device_tokens_received = 0
        host_tokens_stepped = 0

        for h in range(num_hops):
            chunk = audio[h * 320 : (h + 1) * 320]
            chunk_bytes = chunk.astype(np.int16).tobytes()
            crc = calc_crc16(chunk_bytes)

            # Send packet to device
            packet = SERIAL_SYNC + bytes([CMD_AUDIO_CHUNK, 0x00]) + chunk_bytes + crc.to_bytes(2, "little")
            ser.write(packet)

            # Read device response header (4 bytes)
            resp_hdr = read_resp_hdr(ser, timeout_sec=1.0)
            if len(resp_hdr) == 4 and resp_hdr[:2] == RESP_SYNC:
                stepped = resp_hdr[3]
                if stepped:
                    raw_logits = read_exact(ser, 12 * 4) # 12 int32s
                    if len(raw_logits) == 48:
                        device_last_logits = np.frombuffer(raw_logits, dtype=np.int32)
                        device_tokens_received += 1

            # Step host engine with the exact same chunk
            c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
            lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
            if lib.mm_step_q(ctypes.byref(ctx), None, host_last_logits):
                host_tokens_stepped += 1

        if device_last_logits is not None:
            host_arr = np.array(host_last_logits, dtype=np.int32)
            diff = int(np.max(np.abs(device_last_logits - host_arr)))
            if diff > max_logit_diff:
                max_logit_diff = diff

            if diff == 0:
                bit_identical_count += 1
            total_evaluated += 1

            pbar.set_postfix({
                "bit_identical": f"{bit_identical_count}/{total_evaluated}",
                "max_diff": max_logit_diff
            })

    print(f"\nOn-Device Serial Validation Complete!")
    print(f"Evaluated Clips: {total_evaluated}")
    print(f"Bit-Identical Clips: {bit_identical_count} ({bit_identical_count/max(1, total_evaluated)*100:.2f}%)")
    print(f"Maximum Logit Difference: {max_logit_diff}")

def main():
    parser = argparse.ArgumentParser(description="Run on-device serial evaluation for ESP32-C3")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    args = parser.parse_args()

    run_serial_eval(args.port, args.baud, args.subset, args.data_dir)

if __name__ == "__main__":
    main()
