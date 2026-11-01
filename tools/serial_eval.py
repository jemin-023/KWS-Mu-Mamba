#!/usr/bin/env python3
"""
Host-to-device streaming evaluation tool over UART/Serial:
Streams audio clips to the ESP32-C3 firmware running mm_step_q,
collects device output logits, and compares against the host C fixed engine
bit-for-bit.
"""
import sys
import json
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
from train.kws.eval_c import load_engine, MMContext, ModelWeightsQ, select_eval_indices, wilson_interval

SERIAL_SYNC = bytes([0xAA, 0x55])
RESP_SYNC   = bytes([0x55, 0xAA])

CMD_AUDIO_CHUNK = 0x01
CMD_RESET       = 0x03
CMD_BENCH       = 0x04

BENCH_FIELD_NAMES = [
    "version",
    "cpu_freq_hz",
    "ctx_bytes",
    "frontend_cycles_min",
    "frontend_cycles_median",
    "step_cycles_min",
    "step_cycles_median",
    "step_cycles_max",
    "step_stack_used_bytes",
    "free_heap_before",
    "free_heap_after",
    "min_free_heap_before",
    "min_free_heap_after",
    "main_stack_high_water",
    "step_f32_cycles_median",
    "step_f32_rounds",
]

CYCLE_FIELDS = {
    "frontend_cycles_min",
    "frontend_cycles_median",
    "step_cycles_min",
    "step_cycles_median",
    "step_cycles_max",
    "step_f32_cycles_median",
}

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

BENCH_VERSION = 2
BENCH_PAYLOAD_BYTES = 64

def send_bench(ser, timeout_sec: float = 30.0) -> dict | None:
    """Send CMD_BENCH and parse the response. Returns dict of bench values, or None on any protocol problem."""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(SERIAL_SYNC + bytes([CMD_BENCH, 0x00]))
    hdr = read_resp_hdr(ser, timeout_sec)
    if len(hdr) != 4:
        print("Benchmark: no response header")
        return None
    status, length = hdr[2], hdr[3]
    if status != 0:
        print(f"Benchmark: device returned status {status}")
        return None
    payload = read_exact(ser, length, timeout_sec=timeout_sec)
    if len(payload) != length:
        print(f"Benchmark: expected {length} bytes, got {len(payload)}")
        return None
    if length != BENCH_PAYLOAD_BYTES:
        print(f"Benchmark: unexpected payload length {length}, raw bytes: {payload.hex()}")
        return None
    # Decode 16 little-endian uint32 values
    values = [int.from_bytes(payload[i:i + 4], "little") for i in range(0, BENCH_PAYLOAD_BYTES, 4)]
    bench_dict = dict(zip(BENCH_FIELD_NAMES, values))
    if bench_dict["version"] != BENCH_VERSION:
        print(f"Benchmark: unexpected stats version {bench_dict['version']} (expected {BENCH_VERSION})")
        return None
    return bench_dict

def cycle_stats(cycles: list[int]) -> dict | None:
    if not cycles:
        return None
    arr = np.array(cycles, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "median": int(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }

def _write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote JSON report to {path}")

def run_serial_eval(port: str, baud: int, subset: int, data_dir: str, seed: int = 0, bench: bool = False,
                    json_path: str | None = None, ser=None) -> int:
    """Returns the process exit code: 0 only if every requested step completed and every token was bit-identical."""
    if ser is None:
        if serial is None:
            raise ImportError("pyserial is required for on-device serial evaluation. Install with 'pip install pyserial'.")

        print(f"Opening serial port {port} at {baud} baud...")
        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = baud
            ser.timeout = 1.0
            ser.dtr = False
            ser.rts = False
            ser.open()
        except Exception as e:
            print(f"Could not open serial port {port}: {e}")
            print("Note: Plug in the ESP32-C3 board to perform hardware serial validation.")
            return 1

    cpu_freq_hz = 160_000_000  # default 160 MHz
    bench_data = None
    report = {"port": port, "baud": baud, "subset": subset, "seed": seed, "bench_requested": bench}
    exit_code = 0

    if bench:
        print("Running on-device benchmark...")
        bench_data = send_bench(ser, timeout_sec=30.0)
        if bench_data:
            cpu_freq_hz = bench_data.get("cpu_freq_hz", cpu_freq_hz)
            print("Benchmark results:")
            for name in BENCH_FIELD_NAMES:
                val = bench_data[name]
                if name in CYCLE_FIELDS and cpu_freq_hz > 0:
                    ms = val / (cpu_freq_hz / 1000.0)
                    print(f"  {name}: {val}  ({ms:.3f} ms)")
                else:
                    print(f"  {name}: {val}")
        else:
            print("Benchmark FAILED: no valid version-2 stats from the device (old firmware, no sync or short read).")
            exit_code = 1
        report["bench"] = bench_data
        report["bench_ok"] = bench_data is not None

    report["cpu_freq_hz"] = cpu_freq_hz

    if subset == 0 and bench:
        print("Subset is 0 and --bench was used. Exiting after benchmark.")
        ser.close()
        if json_path:
            _write_json(json_path, report)
        return exit_code

    # Load host reference engine
    lib = load_engine()
    ctx = MMContext()
    lib.mm_init(ctypes.byref(ctx))
    c_weights_q = ModelWeightsQ.in_dll(lib, "g_default_model_weights_q")
    lib.mm_set_weights_q(ctypes.byref(c_weights_q))
    report["host_weights_id"] = lib.mm_weights_id().decode("utf-8")
    report["host_export_id"] = lib.mm_export_id().decode("utf-8")

    ds = SpeechCommandsDataset(data_dir=data_dir, split="test", augment=False)
    total_clips = len(ds)
    indices = select_eval_indices(total_clips, subset if subset else None, seed)
    num_eval = len(indices)

    print(f"Running on-device serial test on {num_eval} clips...")
    bit_identical_clips = 0
    total_clips_evaluated = 0
    host_tokens = 0                # every token the HOST stepped: the parity denominator
    bit_identical_tokens = 0
    mismatched_tokens = 0
    device_missing_tokens = 0      # host stepped, device gave no (or garbled) logits
    desync_tokens = 0              # device stepped, host did not
    crc_errors = 0
    reset_failures = 0
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

        # Reset both host and device streaming state; a clip whose device reset fails is evaluated-and-failed
        lib.mm_reset(ctypes.byref(ctx))
        if not (send_reset(ser) or send_reset(ser)):
            reset_failures += 1
            total_clips_evaluated += 1
            continue

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

            # Read device response header (4 bytes): sync, status, stepped
            resp_hdr = read_resp_hdr(ser, timeout_sec=1.0)
            dev_stepped = False
            dev_token_logits = None
            if len(resp_hdr) == 4 and resp_hdr[:2] == RESP_SYNC:
                if resp_hdr[2] != 0:
                    crc_errors += 1
                elif resp_hdr[3]:
                    dev_stepped = True
                    payload = read_exact(ser, 12 * 4 + 4) # 48 bytes logits + 4 bytes cycles
                    if len(payload) == 52:
                        dev_token_logits = np.frombuffer(payload[:48], dtype=np.int32)
                        all_cycles.append(int.from_bytes(payload[48:52], "little"))

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
                host_tokens += 1
                host_arr = np.array(host_token_logits, dtype=np.int32)
                host_last_logits = host_arr
                device_last_logits = dev_token_logits
                if dev_token_logits is None:
                    device_missing_tokens += 1
                    clip_bit_identical = False
                else:
                    diff = int(np.max(np.abs(dev_token_logits.astype(np.int64) - host_arr)))
                    if diff > max_logit_diff:
                        max_logit_diff = diff
                    if diff == 0:
                        bit_identical_tokens += 1
                    else:
                        mismatched_tokens += 1
                        clip_bit_identical = False
            elif dev_stepped:
                desync_tokens += 1
                clip_bit_identical = False

        if host_last_logits is not None:
            if device_last_logits is not None and int(np.argmax(device_last_logits)) == label:
                device_correct += 1
            if int(np.argmax(host_last_logits)) == label:
                host_correct += 1
        if clip_bit_identical:
            bit_identical_clips += 1
        total_clips_evaluated += 1

        pbar.set_postfix({
            "dev_acc": f"{device_correct/total_clips_evaluated*100:.1f}%",
            "tok_bit_id": f"{bit_identical_tokens/max(1, host_tokens)*100:.1f}%",
            "max_diff": max_logit_diff
        })

    print(f"\n================ ON-DEVICE EVALUATION REPORT ================")
    print(f"Total Clips Evaluated:    {total_clips_evaluated} / {num_eval}")
    print(f"Host-Stepped Tokens:      {host_tokens}")
    dev_acc = device_correct / max(1, total_clips_evaluated) * 100.0
    host_acc = host_correct / max(1, total_clips_evaluated) * 100.0
    tok_id = bit_identical_tokens / max(1, host_tokens) * 100.0
    clip_id = bit_identical_clips / max(1, total_clips_evaluated) * 100.0

    dev_lo, dev_hi = wilson_interval(device_correct, total_clips_evaluated)
    host_lo, host_hi = wilson_interval(host_correct, total_clips_evaluated)

    print(f"Device Accuracy:          {dev_acc:.2f}% ({device_correct}/{total_clips_evaluated})  95% CI [{dev_lo:.2f}, {dev_hi:.2f}]")
    print(f"Host Accuracy:            {host_acc:.2f}% ({host_correct}/{total_clips_evaluated})  95% CI [{host_lo:.2f}, {host_hi:.2f}]")
    print(f"Per-Token Bit-Identity:   {tok_id:.2f}% ({bit_identical_tokens}/{host_tokens})")
    print(f"Bit-Identical Clips:      {clip_id:.2f}% ({bit_identical_clips}/{total_clips_evaluated})")
    print(f"Maximum Logit Difference: {max_logit_diff}")
    print(f"Mismatched tokens: {mismatched_tokens}   device-missing: {device_missing_tokens}   desync: {desync_tokens}   "
          f"CRC errors: {crc_errors}   reset failures: {reset_failures}")

    cstats = cycle_stats(all_cycles)
    if cstats:
        to_ms = lambda c: c / (cpu_freq_hz / 1000.0)
        print(f"\n--- On-Board Timing (@ {cpu_freq_hz/1e6:.0f} MHz) ---")
        print(f"Min Per-Token Time:       {to_ms(cstats['min']):.2f} ms ({cstats['min']} cycles)")
        print(f"Median Per-Token Time:    {to_ms(cstats['median']):.2f} ms ({cstats['median']} cycles)")
        print(f"Mean Per-Token Time:      {to_ms(cstats['mean']):.2f} ms ({int(cstats['mean'])} cycles)")
        print(f"p99 Per-Token Time:       {to_ms(cstats['p99']):.2f} ms ({cstats['p99']} cycles)")
        print(f"Max Per-Token Time:       {to_ms(cstats['max']):.2f} ms ({cstats['max']} cycles)")
    print(f"=============================================================\n")

    parity_ok = (
        host_tokens > 0
        and total_clips_evaluated == num_eval
        and bit_identical_clips == num_eval
        and bit_identical_tokens == host_tokens
        and device_missing_tokens == 0
        and desync_tokens == 0
        and reset_failures == 0
    )
    report.update({
        "num_eval": num_eval,
        "total_clips_evaluated": total_clips_evaluated,
        "bit_identical_clips": bit_identical_clips,
        "host_tokens": host_tokens,
        "bit_identical_tokens": bit_identical_tokens,
        "mismatched_tokens": mismatched_tokens,
        "device_missing_tokens": device_missing_tokens,
        "desync_tokens": desync_tokens,
        "crc_errors": crc_errors,
        "reset_failures": reset_failures,
        "max_logit_diff": max_logit_diff,
        "device_correct": device_correct,
        "device_accuracy_pct": dev_acc,
        "device_wilson_ci": [dev_lo, dev_hi],
        "host_correct": host_correct,
        "host_accuracy_pct": host_acc,
        "host_wilson_ci": [host_lo, host_hi],
        "cycles": cstats,
        "parity_ok": parity_ok,
    })
    if json_path:
        _write_json(json_path, report)

    if not parity_ok:
        print("Parity check FAILED: incomplete run or tokens not bit-identical")
        return 1
    return exit_code

def main():
    parser = argparse.ArgumentParser(description="Run on-device serial evaluation for ESP32-C3")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection (default: 0)")
    parser.add_argument("--bench", action="store_true", help="Run on-device benchmark before evaluation")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON report")
    args = parser.parse_args()

    sys.exit(run_serial_eval(args.port, args.baud, args.subset, args.data_dir, seed=args.seed, bench=args.bench,
                             json_path=args.json))

if __name__ == "__main__":
    main()
