"""serial_eval parity/bench gates, driven by a simulated device that runs the host fixed engine.

The fake device speaks the same framing as firmware/main/audio_serial.c, so these tests check that
serial_eval fails (exit 1) on an incomplete or old-firmware run instead of reporting a pass.
"""
import ctypes
import json

import numpy as np

from tools import serial_eval as se
from train.kws.eval_c import load_engine, MMContext, ModelWeightsQ

DATA_DIR = "data/speech_commands/v0.02"
FAKE_CYCLES = 650_000


class FakeDevice:
    """pyserial-like object. Responses are queued on write(); read() never blocks."""

    def __init__(self, drop_audio_packet=None, bench="ok"):
        self.lib = load_engine()
        self.ctx = MMContext()
        self.lib.mm_init(ctypes.byref(self.ctx))
        self.lib.mm_set_weights_q(ctypes.byref(ModelWeightsQ.in_dll(self.lib, "g_default_model_weights_q")))
        self.rx = bytearray()
        self.drop_audio_packet = drop_audio_packet
        self.bench = bench
        self.audio_packets = 0

    def reset_input_buffer(self):
        self.rx.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        pass

    def read(self, n=1):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def write(self, data):
        cmd = data[2]
        if cmd == se.CMD_RESET:
            self.lib.mm_reset(ctypes.byref(self.ctx))
            self.rx += bytes([0x55, 0xAA, 0x00, 0x00])
        elif cmd == se.CMD_BENCH:
            self._bench_reply()
        elif cmd == se.CMD_AUDIO_CHUNK:
            self._audio_reply(np.frombuffer(data[4:4 + 640], dtype=np.int16))

    def _bench_reply(self):
        if self.bench == "short":  # old 24-byte firmware
            self.rx += bytes([0x55, 0xAA, 0x00, 24]) + bytes(24)
            return
        version = 1 if self.bench == "old_version" else se.BENCH_VERSION
        vals = [version, 160_000_000] + [1] * 14
        self.rx += bytes([0x55, 0xAA, 0x00, 64]) + b"".join(v.to_bytes(4, "little") for v in vals)

    def _audio_reply(self, chunk):
        c_chunk = (ctypes.c_int16 * len(chunk))(*chunk)
        pushed = self.lib.mm_push_audio(ctypes.byref(self.ctx), c_chunk, len(chunk))
        logits = (ctypes.c_int32 * 12)()
        stepped = self.lib.mm_step_q(ctypes.byref(self.ctx), None, logits)
        if pushed < len(chunk):
            rem = chunk[pushed:]
            self.lib.mm_push_audio(ctypes.byref(self.ctx), (ctypes.c_int16 * len(rem))(*rem), len(rem))
        self.audio_packets += 1
        if self.audio_packets == self.drop_audio_packet:
            return
        self.rx += bytes([0x55, 0xAA, 0x00, int(bool(stepped))])
        if stepped:
            self.rx += np.array(logits, dtype=np.int32).tobytes() + FAKE_CYCLES.to_bytes(4, "little")


def _run(tmp_path, **kw):
    out = tmp_path / "report.json"
    dev = FakeDevice(**{k: v for k, v in kw.items() if k in ("drop_audio_packet", "bench")})
    code = se.run_serial_eval("fake", 0, kw.get("subset", 2), DATA_DIR, seed=0, bench=kw.get("bench") is not None,
                              json_path=str(out), ser=dev)
    return code, json.loads(out.read_text())


def test_clean_run_passes(tmp_path):
    code, rep = _run(tmp_path)
    assert code == 0
    assert rep["parity_ok"] and rep["host_tokens"] == rep["bit_identical_tokens"] > 0
    assert rep["device_missing_tokens"] == rep["desync_tokens"] == rep["reset_failures"] == 0
    assert rep["cycles"]["median"] == FAKE_CYCLES


def test_dropped_hop_fails(tmp_path):
    code, rep = _run(tmp_path, drop_audio_packet=5)
    assert code == 1
    assert rep["device_missing_tokens"] == 1
    assert rep["bit_identical_tokens"] == rep["host_tokens"] - 1
    assert not rep["parity_ok"]


def test_short_bench_fails(tmp_path):
    code, rep = _run(tmp_path, bench="short", subset=0)
    assert code == 1
    assert rep["bench_ok"] is False


def test_old_bench_version_fails(tmp_path):
    code, rep = _run(tmp_path, bench="old_version", subset=0)
    assert code == 1


def test_good_bench_subset_zero_passes(tmp_path):
    code, rep = _run(tmp_path, bench="ok", subset=0)
    assert code == 0
    assert rep["bench"]["cpu_freq_hz"] == 160_000_000
