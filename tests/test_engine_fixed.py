"""Tests against the built host library engine/libmumamba.so.

Exported C symbols (see engine/include/mm_math.h, engine/include/mu_mamba.h, engine/src/mm_math.c):
  int32_t mm_lut_interp_q31(const int32_t *lut, int32_t x, int32_t x_min, int32_t step_shift, int32_t num_entries);
  int32_t mm_lut_interp_s16(const int16_t *lut, int32_t x, int32_t x_min, int32_t step_shift, int32_t num_entries);
  int32_t mm_log2_q8(uint32_t x);
  int32_t mm_rsqrt_q12(uint32_t x);
  int16_t mm_silu_q12(int16_t x);
  size_t mm_context_size(void);
  data symbols: const int16_t mm_lut_silu_q12[513];
                const int32_t mm_lut_q_q31[513];
                const int32_t mm_lut_softplus_q24[513];
"""

import ctypes
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared constants and helpers (mirror from test_math_claims.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LUTS_C = _REPO_ROOT / "engine" / "gen" / "mm_luts.c"
_ENGINE_LIB = _REPO_ROOT / "engine" / "libmumamba.so"
_INCLUDE_DIR = _REPO_ROOT / "engine" / "include"
_GEN_DIR = _REPO_ROOT / "engine" / "gen"

_X_MIN = -32768
_STEP_SHIFT = 7
_NUM_ENTRIES = 513


def _parse_c_lut(name):
    """Return the integer entries of ``const <type> <name>[N] = { ... };``."""
    src = _LUTS_C.read_text()
    pat = r"const\s+\w+\s+" + re.escape(name) + r"\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\}\s*;"
    m = re.search(pat, src, re.DOTALL)
    assert m is not None, f"array {name} not found in {_LUTS_C}"
    return [int(v) for v in re.findall(r"-?\d+", m.group(1))]


# Parsed LUTs from generated C source
LUT_SILU_Q12_PY = _parse_c_lut("mm_lut_silu_q12")
LUT_Q_Q31_PY = _parse_c_lut("mm_lut_q_q31")
LUT_SOFTPLUS_Q24_PY = _parse_c_lut("mm_lut_softplus_q24")


def lut_interp_python(lut, x, x_min=_X_MIN, step_shift=_STEP_SHIFT, num_entries=_NUM_ENTRIES):
    """Exact mirror of mm_lut_interp_q31 / mm_lut_interp_s16 (scalar, ints)."""
    diff = x - x_min
    if diff <= 0:
        return lut[0]
    idx = diff >> step_shift
    if idx >= num_entries - 1:
        return lut[num_entries - 1]
    frac = diff & ((1 << step_shift) - 1)
    return lut[idx] + (((lut[idx + 1] - lut[idx]) * frac + (1 << (step_shift - 1))) >> step_shift)


# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lib():
    """Load engine/libmumamba.so once per test module."""
    assert _ENGINE_LIB.exists(), f"{_ENGINE_LIB} does not exist"
    lib = ctypes.CDLL(str(_ENGINE_LIB))

    # Function signatures
    lib.mm_lut_interp_q31.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.mm_lut_interp_q31.restype = ctypes.c_int32

    lib.mm_lut_interp_s16.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.mm_lut_interp_s16.restype = ctypes.c_int32

    lib.mm_log2_q8.argtypes = [ctypes.c_uint32]
    lib.mm_log2_q8.restype = ctypes.c_int32

    lib.mm_rsqrt_q12.argtypes = [ctypes.c_uint32]
    lib.mm_rsqrt_q12.restype = ctypes.c_int32

    lib.mm_silu_q12.argtypes = [ctypes.c_int16]
    lib.mm_silu_q12.restype = ctypes.c_int16

    lib.mm_context_size.argtypes = []
    lib.mm_context_size.restype = ctypes.c_size_t

    # Data symbols
    silu_arr = (ctypes.c_int16 * 513).in_dll(lib, "mm_lut_silu_q12")
    q_arr = (ctypes.c_int32 * 513).in_dll(lib, "mm_lut_q_q31")
    softplus_arr = (ctypes.c_int32 * 513).in_dll(lib, "mm_lut_softplus_q24")

    return {
        "lib": lib,
        "silu": silu_arr,
        "q": q_arr,
        "softplus": softplus_arr,
    }


# ---------------------------------------------------------------------------
# Test a) test_lut_symbols_match_generated_source
# ---------------------------------------------------------------------------

def test_lut_symbols_match_generated_source(lib):
    """The three in-library tables equal, element by element, the arrays
    parsed from engine/gen/mm_luts.c."""
    silu_lib = list(lib["silu"])
    q_lib = list(lib["q"])
    softplus_lib = list(lib["softplus"])

    assert silu_lib == LUT_SILU_Q12_PY, "mm_lut_silu_q12 mismatch"
    assert q_lib == LUT_Q_Q31_PY, "mm_lut_q_q31 mismatch"
    assert softplus_lib == LUT_SOFTPLUS_Q24_PY, "mm_lut_softplus_q24 mismatch"
    print("test_lut_symbols_match_generated_source: all three tables match")


# ---------------------------------------------------------------------------
# Test b) test_c_interpolation_matches_python_mirror
# ---------------------------------------------------------------------------

def test_c_interpolation_matches_python_mirror(lib):
    """For the q table and the softplus table, for EVERY integer x in
    [-33000, 33000] (covers both clamps), the C mm_lut_interp_q31(table, x,
    -32768, 7, 513) equals the Python mirror, and the same for
    mm_lut_interp_s16 with the silu table. Also check mm_silu_q12(x) ==
    the mirror for every int16 x."""

    lib_handle = lib["lib"]

    # Prepare array pointers
    q_ptr = lib["q"]
    softplus_ptr = lib["softplus"]
    silu_ptr = lib["silu"]

    # x range: [-33000, 33000] covers both clamps
    x_vals = np.arange(-33000, 33001, dtype=np.int32)

    # Test q table (Q31 interpolation)
    print("Testing q table interpolation...")
    for x in x_vals:
        c_result = lib_handle.mm_lut_interp_q31(q_ptr, int(x), _X_MIN, _STEP_SHIFT, _NUM_ENTRIES)
        py_result = lut_interp_python(LUT_Q_Q31_PY, int(x))
        assert c_result == py_result, f"q table mismatch at x={x}: C={c_result}, Python={py_result}"

    # Test softplus table (Q31 interpolation)
    print("Testing softplus table interpolation...")
    for x in x_vals:
        c_result = lib_handle.mm_lut_interp_q31(softplus_ptr, int(x), _X_MIN, _STEP_SHIFT, _NUM_ENTRIES)
        py_result = lut_interp_python(LUT_SOFTPLUS_Q24_PY, int(x))
        assert c_result == py_result, f"softplus table mismatch at x={x}: C={c_result}, Python={py_result}"

    # Test silu table (S16 interpolation)
    print("Testing silu table interpolation...")
    for x in x_vals:
        c_result = lib_handle.mm_lut_interp_s16(silu_ptr, int(x), _X_MIN, _STEP_SHIFT, _NUM_ENTRIES)
        py_result = lut_interp_python(LUT_SILU_Q12_PY, int(x))
        assert c_result == py_result, f"silu table mismatch at x={x}: C={c_result}, Python={py_result}"

    # Test mm_silu_q12 for every int16 x
    print("Testing mm_silu_q12 for all int16...")
    for x in range(-32768, 32768):
        c_result = lib_handle.mm_silu_q12(ctypes.c_int16(x))
        py_result = lut_interp_python(LUT_SILU_Q12_PY, x)
        assert c_result == py_result, f"mm_silu_q12 mismatch at x={x}: C={c_result}, Python={py_result}"

    print("test_c_interpolation_matches_python_mirror: all checks passed")


# ---------------------------------------------------------------------------
# Test c) test_rsqrt_q12_accuracy
# ---------------------------------------------------------------------------

def test_rsqrt_q12_accuracy(lib):
    """Test mm_rsqrt_q12 accuracy.

    For x in a broad sample (all x in 1..5000, plus 20000 log-uniformly
    spaced values up to 2**30), r = mm_rsqrt_q12(x) satisfies
    |r / (2**24 / sqrt(x)) - 1| <= 2**-7 for x >= 16, and r is clamped to
    2**24 at x == 0 and x == 1. Print the measured maximum relative error
    for x >= 16 and for x >= 4096 (print only; assert only the stated bound).
    """
    lib_handle = lib["lib"]

    # Test x = 0 and x = 1: should return 2**24 (clamped)
    r0 = lib_handle.mm_rsqrt_q12(0)
    r1 = lib_handle.mm_rsqrt_q12(1)
    assert r0 == (1 << 24), f"mm_rsqrt_q12(0) = {r0}, expected {1<<24}"
    assert r1 == (1 << 24), f"mm_rsqrt_q12(1) = {r1}, expected {1<<24}"
    print(f"mm_rsqrt_q12(0) = {r0}, mm_rsqrt_q12(1) = {r1} (clamped to 2^24)")

    # Generate test points: all x in 1..5000, plus 20000 log-uniform up to 2**30
    xs = list(range(1, 5001))
    # Log-uniform samples from 5000 to 2**30
    log_min = math.log(5000)
    log_max = math.log(2**30)
    for i in range(20000):
        u = i / 19999.0
        val = int(math.exp(log_min + u * (log_max - log_min)))
        xs.append(val)

    xs = sorted(set(xs))
    print(f"Testing mm_rsqrt_q12 on {len(xs)} values...")

    max_rel_err_ge16 = 0.0
    max_rel_err_ge4096 = 0.0
    max_rel_err_x = 0
    max_rel_err_x_ge4096 = 0

    for x in xs:
        r = lib_handle.mm_rsqrt_q12(x)
        # Expected: 2^24 / sqrt(x)
        expected = (1 << 24) / math.sqrt(x)
        rel_err = abs(r / expected - 1.0)

        if x >= 16 and rel_err > max_rel_err_ge16:
            max_rel_err_ge16 = rel_err
            max_rel_err_x = x

        if x >= 4096 and rel_err > max_rel_err_ge4096:
            max_rel_err_ge4096 = rel_err
            max_rel_err_x_ge4096 = x

    bound = 2**-7  # 1/128
    print(f"test_rsqrt_q12_accuracy: max rel err for x>=16 = {max_rel_err_ge16:.6e} at x={max_rel_err_x} (bound = {bound:.6e})")
    print(f"test_rsqrt_q12_accuracy: max rel err for x>=4096 = {max_rel_err_ge4096:.6e} at x={max_rel_err_x_ge4096} (bound = {bound:.6e})")

    assert max_rel_err_ge16 <= bound, f"max rel err for x>=16 = {max_rel_err_ge16:.6e} > {bound:.6e}"


# ---------------------------------------------------------------------------
# Test d) test_log2_q8_accuracy
# ---------------------------------------------------------------------------

def test_log2_q8_accuracy(lib):
    """Test mm_log2_q8 accuracy.

    For x in 1..70000 and 20000 log-uniform values up to 2**32 - 1:
    |mm_log2_q8(x) / 256 - log2(x)| <= 2/256, and mm_log2_q8(0) == 0,
    and mm_log2_q8(2**k) == 256*k for k = 0..31. Print the measured
    maximum error.
    """
    lib_handle = lib["lib"]

    # Test x = 0
    assert lib_handle.mm_log2_q8(0) == 0, f"mm_log2_q8(0) = {lib_handle.mm_log2_q8(0)}, expected 0"

    # Test exact powers of two: mm_log2_q8(2**k) == 256*k
    for k in range(32):
        r = lib_handle.mm_log2_q8(1 << k)
        expected = 256 * k
        assert r == expected, f"mm_log2_q8(2^{k}) = {r}, expected {expected}"

    # Generate test points: all x in 1..70000, plus 20000 log-uniform up to 2**32-1
    xs = list(range(1, 70001))
    log_min = math.log(70000)
    log_max = math.log(2**32 - 1)
    for i in range(20000):
        u = i / 19999.0
        val = int(math.exp(log_min + u * (log_max - log_min)))
        xs.append(val)

    xs = sorted(set(xs))
    print(f"Testing mm_log2_q8 on {len(xs)} values...")

    max_abs_err = 0.0
    max_abs_err_x = 0

    for x in xs:
        r = lib_handle.mm_log2_q8(x)
        exact = math.log2(x)
        approx = r / 256.0
        abs_err = abs(approx - exact)
        if abs_err > max_abs_err:
            max_abs_err = abs_err
            max_abs_err_x = x

    bound = 2.0 / 256.0
    print(f"test_log2_q8_accuracy: max abs err = {max_abs_err:.6e} at x={max_abs_err_x} (bound = {bound:.6e})")

    assert max_abs_err <= bound, f"max abs err = {max_abs_err:.6e} > {bound:.6e}"


# ---------------------------------------------------------------------------
# Test e) test_context_size_matches_documented_layout
# ---------------------------------------------------------------------------

def test_context_size_matches_documented_layout(tmp_path):
    """Compile a one-line C program that includes engine/include/mu_mamba.h
    WITHOUT defining MM_ENABLE_F32 and prints sizeof(mm_context_t). Assert it
    prints 6072, and assert the arithmetic that documents it:
    512*2 + 2 + 40*2 + 2*3*64*2 + 2 (alignment padding before the int32 state)
    + 2*64*8*4 + 12*4 + 12*4 + 2 + 2 (tail padding) == 6072.

    Also compile it WITH -DMM_ENABLE_F32 and assert the printed value equals
    lib.mm_context_size() (the host library is built with the float reference
    enabled).
    """
    # Check if gcc is available
    if not _has_gcc():
        pytest.skip("gcc not found")

    # The documented arithmetic for mm_context_t without MM_ENABLE_F32
    # audio_buf[512] (int16_t) = 512 * 2 = 1024
    # audio_buf_count (uint16_t) = 2
    # mel_features[40] (int16_t) = 40 * 2 = 80
    # conv_ring_q[2][3][64] (int16_t) = 2 * 3 * 64 * 2 = 768
    # alignment padding before int32 state = 2
    # ssm_h_q[2][64][8] (int32_t) = 2 * 64 * 8 * 4 = 4096
    # logits_q[12] (int32_t) = 12 * 4 = 48
    # smoothed_margins[12] (int32_t) = 12 * 4 = 48
    # refractory_timer (int16_t) = 2
    # tail padding = 2
    # Total = 1024 + 2 + 80 + 768 + 2 + 4096 + 48 + 48 + 2 + 2 = 6072

    documented_size = (
        512 * 2 +      # audio_buf
        2 +            # audio_buf_count
        40 * 2 +       # mel_features
        2 * 3 * 64 * 2 +  # conv_ring_q
        2 +            # alignment padding before int32 state
        2 * 64 * 8 * 4 +  # ssm_h_q
        12 * 4 +       # logits_q
        12 * 4 +       # smoothed_margins
        2 +            # refractory_timer
        2              # tail padding
    )
    assert documented_size == 6072, f"Documented size arithmetic = {documented_size}, expected 6072"

    # Compile and run without MM_ENABLE_F32
    c_code = """
#include <stdio.h>
#include <mu_mamba.h>
int main(void) {
    printf("%zu\\n", sizeof(mm_context_t));
    return 0;
}
"""
    result = _compile_and_run_c(tmp_path, c_code, extra_flags=["-I", str(_INCLUDE_DIR), "-I", str(_GEN_DIR)])
    size_without_f32 = int(result.stdout.strip())
    print(f"sizeof(mm_context_t) without MM_ENABLE_F32 = {size_without_f32}")
    assert size_without_f32 == 6072, f"Expected 6072, got {size_without_f32}"

    # Compile and run with MM_ENABLE_F32
    c_code_f32 = """
#define MM_ENABLE_F32
#include <stdio.h>
#include <mu_mamba.h>
int main(void) {
    printf("%zu\\n", sizeof(mm_context_t));
    return 0;
}
"""
    result_f32 = _compile_and_run_c(tmp_path, c_code_f32, extra_flags=["-I", str(_INCLUDE_DIR), "-I", str(_GEN_DIR)])
    size_with_f32 = int(result_f32.stdout.strip())
    print(f"sizeof(mm_context_t) with MM_ENABLE_F32 = {size_with_f32}")

    # Load library and check mm_context_size()
    lib = ctypes.CDLL(str(_ENGINE_LIB))
    lib.mm_context_size.argtypes = []
    lib.mm_context_size.restype = ctypes.c_size_t
    lib_size = lib.mm_context_size()
    print(f"lib.mm_context_size() = {lib_size}")

    assert size_with_f32 == lib_size, f"sizeof with MM_ENABLE_F32 ({size_with_f32}) != lib.mm_context_size() ({lib_size})"


def _has_gcc():
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _compile_and_run_c(tmp_path, c_code, extra_flags=None):
    """Compile a C program and run it, returning the CompletedProcess."""
    c_file = tmp_path / "test_context.c"
    c_file.write_text(c_code)

    exe_file = tmp_path / "test_context"
    cmd = ["gcc", "-std=c99", "-o", str(exe_file), str(c_file)]
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Compilation failed: {result.stderr}")

    run_result = subprocess.run([str(exe_file)], capture_output=True, text=True)
    if run_result.returncode != 0:
        pytest.fail(f"Execution failed: {run_result.stderr}")

    return run_result