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
    bit_identical_clips = 0
    total_clips_evaluated = 0
    total_tokens_evaluated = 0
    bit_identical_tokens = 0
    device_correct = 0
    host_correct = 0
    all_cycles = []
    max_logit_diff = 0

    TARGET_EVAL_LEN = 51 * 320 # 16,320 samples (51 hops)

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

        # Pad to exactly 16,320 samples (51 hops of 320 samples -> 50 tokens)
        if len(audio) < TARGET_EVAL_LEN:
            audio = np.pad(audio, (0, TARGET_EVAL_LEN - len(audio)), mode="constant")
        elif len(audio) > TARGET_EVAL_LEN:
            audio = audio[:TARGET_EVAL_LEN]

        # Reset both host and device streaming state
        lib.mm_reset(ctypes.byref(ctx))
        send_reset(ser)

        clip_bit_identical = True
        device_last_logits = None
        host_last_logits = None

        num_hops = 51
        for h in range(num_hops):
            chunk = audio[h * 320 : (h + 1) * 320]
            chunk_bytes = chunk.astype(np.int16).tobytes()
            crc = calc_crc16(chunk_bytes)

            # Send packet to device
            packet = SERIAL_SYNC + bytes([CMD_AUDIO_CHUNK, 0x00]) + chunk_bytes + crc.to_bytes(2, "little")
            ser.write(packet)

            # Read device response header (4 bytes)
            resp_hdr = read_resp_hdr(ser, timeout_sec=1.0)
            dev_stepped = False
            dev_token_logits = None
            if len(resp_hdr) == 4 and resp_hdr[:2] == RESP_SYNC:
                if resp_hdr[3]:
                    dev_stepped = True
                    payload = read_exact(ser, 12 * 4 + 4) # 48 bytes logits + 4 bytes cycles
                    if len(payload) == 52:
                        dev_token_logits = np.frombuffer(payload[:48], dtype=np.int32)
                        cycles = int.from_bytes(payload[48:52], "little")
                        all_cycles.append(cycles)
                        device_last_logits = dev_token_logits

            # Step host engine with exact same push/step/remainder logic
            c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
            pushed = lib.mm_push_audio(ctypes.byref(ctx), c_chunk, len(chunk))
            host_token_logits = (ctypes.c_int32 * 12)()
            host_stepped = lib.mm_step_q(ctypes.byref(ctx), None, host_token_logits)
            if pushed < len(chunk):
                rem_chunk = chunk[pushed:]
                c_rem = (ctypes.c_int16 * len(rem_chunk))(*rem_chunk)
                lib.mm_push_audio(ctypes.byref(ctx), c_rem, len(rem_chunk))

            if host_stepped:
                host_arr = np.array(host_token_logits, dtype=np.int32)
                host_last_logits = host_arr
                if dev_stepped and dev_token_logits is not None:
                    total_tokens_evaluated += 1
                    diff = int(np.max(np.abs(dev_token_logits - host_arr)))
                    if diff > max_logit_diff:
                        max_logit_diff = diff
                    if diff == 0:
                        bit_identical_tokens += 1
                    else:
                        clip_bit_identical = False
                else:
                    clip_bit_identical = False

        if device_last_logits is not None and host_last_logits is not None:
            dev_pred = int(np.argmax(device_last_logits))
            host_pred = int(np.argmax(host_last_logits))
            if dev_pred == label:
                device_correct += 1
            if host_pred == label:
                host_correct += 1
            if clip_bit_identical:
                bit_identical_clips += 1
            total_clips_evaluated += 1

            pbar.set_postfix({
                "dev_acc": f"{device_correct/total_clips_evaluated*100:.1f}%",
                "tok_bit_id": f"{bit_identical_tokens/max(1, total_tokens_evaluated)*100:.1f}%",
                "max_diff": max_logit_diff
            })

    print(f"\n================ ON-DEVICE EVALUATION REPORT ================")
    print(f"Total Clips Evaluated:    {total_clips_evaluated}")
    print(f"Total Tokens Evaluated:   {total_tokens_evaluated}")
    dev_acc = device_correct / max(1, total_clips_evaluated) * 100.0
    host_acc = host_correct / max(1, total_clips_evaluated) * 100.0
    tok_id = bit_identical_tokens / max(1, total_tokens_evaluated) * 100.0
    clip_id = bit_identical_clips / max(1, total_clips_evaluated) * 100.0
    print(f"Device Accuracy:          {dev_acc:.2f}% ({device_correct}/{total_clips_evaluated})")
    print(f"Host Accuracy:            {host_acc:.2f}% ({host_correct}/{total_clips_evaluated})")
    print(f"Per-Token Bit-Identity:   {tok_id:.2f}% ({bit_identical_tokens}/{total_tokens_evaluated})")
    print(f"Bit-Identical Clips:      {clip_id:.2f}% ({bit_identical_clips}/{total_clips_evaluated})")
    print(f"Maximum Logit Difference: {max_logit_diff}")

    if all_cycles:
        cycles_arr = np.array(all_cycles, dtype=np.float64)
        min_ms = np.min(cycles_arr) / 160000.0
        med_ms = np.median(cycles_arr) / 160000.0
        mean_ms = np.mean(cycles_arr) / 160000.0
        print(f"\n--- On-Board Timing (@ 160 MHz) ---")
        print(f"Min Per-Token Time:       {min_ms:.2f} ms ({int(np.min(cycles_arr))} cycles)")
        print(f"Median Per-Token Time:    {med_ms:.2f} ms ({int(np.median(cycles_arr))} cycles)")
        print(f"Mean Per-Token Time:      {mean_ms:.2f} ms ({int(np.mean(cycles_arr))} cycles)")
    print(f"=============================================================\n")

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
