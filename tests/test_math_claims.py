"""Numerical verification of the mathematical claims made for the KWS-Mu-Mamba
C1 fixed-point engine.

This module deliberately does NOT load ``engine/libmumamba.so`` (which may be
under concurrent rebuild).  The three lookup tables are instead parsed straight
out of the generated C source ``engine/gen/mm_luts.c`` and the interpolation
routine is re-implemented bit-for-bit in Python integers.

All fixed-point arithmetic is done with Python ints (arbitrary precision) or
numpy int64 (where products provably stay below 2**63).  Reals are float64.
"""

import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lookup table parsing (regex over the C source - no shared library involved)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LUTS_C = _REPO_ROOT / "engine" / "gen" / "mm_luts.c"


def _parse_c_lut(name):
    """Return the integer entries of ``const <type> <name>[N] = { ... };``."""
    src = _LUTS_C.read_text()
    pat = r"const\s+\w+\s+" + re.escape(name) + r"\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\}\s*;"
    m = re.search(pat, src, re.DOTALL)
    assert m is not None, f"array {name} not found in {_LUTS_C}"
    return [int(v) for v in re.findall(r"-?\d+", m.group(1))]


# Only these three arrays are relevant; mm_lut_softplus_q11 and the log2 table
# in mm_math.c are intentionally ignored.
LUT_SILU_Q12 = _parse_c_lut("mm_lut_silu_q12")
LUT_Q_Q31 = _parse_c_lut("mm_lut_q_q31")
LUT_SOFTPLUS_Q24 = _parse_c_lut("mm_lut_softplus_q24")

# Interpolation parameters, identical for all three tables (step 128 = 2**7).
_X_MIN = -32768
_STEP_SHIFT = 7
_NUM_ENTRIES = 513
_H_Q = 1.0 / 16.0     # q / softplus grid step in real p units
_H_SILU = 1.0 / 32.0  # silu grid step in real x units


def lut_interp(lut, x):
    """Exact mirror of mm_lut_interp_q31 / mm_lut_interp_s16 (scalar, ints)."""
    diff = x - _X_MIN
    if diff <= 0:
        return lut[0]
    idx = diff >> _STEP_SHIFT
    if idx >= _NUM_ENTRIES - 1:
        return lut[_NUM_ENTRIES - 1]
    frac = diff & ((1 << _STEP_SHIFT) - 1)
    return lut[idx] + (((lut[idx + 1] - lut[idx]) * frac + (1 << (_STEP_SHIFT - 1))) >> _STEP_SHIFT)


def lut_interp_vec(lut, x):
    """Vectorised exact mirror of the C interpolation (int64 arithmetic)."""
    lut = np.asarray(lut, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    diff = x - _X_MIN
    idx = diff >> _STEP_SHIFT
    frac = diff & ((1 << _STEP_SHIFT) - 1)
    ic = np.clip(idx, 0, _NUM_ENTRIES - 2)
    y0 = lut[ic]
    y1 = lut[ic + 1]
    out = y0 + (((y1 - y0) * frac + (1 << (_STEP_SHIFT - 1))) >> _STEP_SHIFT)
    out = np.where(diff <= 0, lut[0], out)
    out = np.where(idx >= _NUM_ENTRIES - 1, lut[_NUM_ENTRIES - 1], out)
    return out


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_c1_identity():
    """exp(-softplus(p)) == sigmoid(-p) and powers thereof (rel err < 1e-12)."""
    p = np.linspace(-12.0, 12.0, 20001)
    softplus = np.log1p(np.exp(p))
    q = 1.0 / (1.0 + np.exp(p))

    lhs0 = np.exp(-softplus)
    rel0 = np.max(np.abs(lhs0 - q) / np.abs(q))
    print(f"T1 n=0: max rel err exp(-softplus(p)) vs sigmoid(-p) = {rel0:.3e}")

    max_rel = rel0
    for n in range(8):
        lhs = np.exp(-(n + 1) * softplus)
        rhs = q ** (n + 1)
        rel = np.max(np.abs(lhs - rhs) / np.abs(rhs))
        max_rel = max(max_rel, rel)
        print(f"T1 n={n}: max rel err = {rel:.3e}")

    assert max_rel < 1e-12


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------

def test_zoh_identity():
    """ZOH input gain equals (1 - q**(n+1)) / (n+1); saturates to 1/(n+1)."""
    p = np.linspace(-6.0, 12.0, 20001)
    Delta = np.log1p(np.exp(p))
    q = 1.0 / (1.0 + np.exp(p))
    B = 1.0

    max_rel = 0.0
    for n in range(8):
        A = -(n + 1)
        lhs = (Delta * A) ** -1 * np.expm1(Delta * A) * Delta * B
        rhs = -np.expm1((n + 1) * np.log(q)) / (n + 1)
        rel = np.max(np.abs(lhs - rhs) / np.abs(rhs))
        max_rel = max(max_rel, rel)
        print(f"T2 n={n}: max rel err = {rel:.3e}")
    assert max_rel < 1e-9

    # Saturation limit at p = 12 (q -> 0): value 1 / (n+1).
    D12 = math.log1p(math.exp(12.0))
    sat_err = 0.0
    for n in range(8):
        A = -(n + 1)
        val = (D12 * A) ** -1 * math.expm1(D12 * A) * D12 * B
        sat_err = max(sat_err, abs(val - 1.0 / (n + 1)))
    print(f"T2 saturation at p=12: max |value - 1/(n+1)| = {sat_err:.3e}")
    assert sat_err < 1e-4


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------

def test_lut_tables_shape_and_footprint():
    """Table sizes, flash footprint and entries match their definitions."""
    assert len(LUT_SILU_Q12) == 513
    assert len(LUT_Q_Q31) == 513
    assert len(LUT_SOFTPLUS_Q24) == 513

    total_flash = 513 * 2 + 513 * 4 + 513 * 4
    assert total_flash == 5130
    assert round(5130 / 1024.0, 2) == 5.01
    print(f"T3 flash footprint = {total_flash} bytes = {round(5130/1024.0, 2)} KiB")

    # q table: x = linspace(-16, 16, 513) is p; round(sigmoid(-p) * 2**31 - 1)
    p = np.linspace(-16.0, 16.0, 513)
    ref_q = np.round((1.0 / (1.0 + np.exp(p))) * 2147483647.0)
    ref_sp = np.round(np.log1p(np.exp(p)) * (2.0 ** 24))

    # silu table: x = linspace(-8, 8, 513); round(x*sigmoid(x) * 4096)
    xs = np.linspace(-8.0, 8.0, 513)
    ref_silu = np.round((xs / (1.0 + np.exp(-xs))) * 4096.0)

    q_arr = np.asarray(LUT_Q_Q31, dtype=np.int64)
    sp_arr = np.asarray(LUT_SOFTPLUS_Q24, dtype=np.int64)
    silu_arr = np.asarray(LUT_SILU_Q12, dtype=np.int64)

    dq = np.max(np.abs(q_arr - ref_q))
    dsp = np.max(np.abs(sp_arr - ref_sp))
    dsilu = np.max(np.abs(silu_arr - ref_silu))
    print(f"T3 max deviations from definitions (LSB): q={dq}, softplus={dsp}, silu={dsilu}")
    assert dq <= 1
    assert dsp <= 1
    assert dsilu <= 1

    assert np.all(np.diff(q_arr) <= 0), "q table must be non-increasing"
    assert np.all(np.diff(sp_arr) >= 0), "softplus table must be non-decreasing"
    assert np.all(q_arr >= 0)
    assert np.all(q_arr <= 2 ** 31 - 1)
    print("T3 q non-increasing, softplus non-decreasing, q entries within [0, 2**31-1]")


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------

def test_sigmoid_second_derivative_peak():
    """max |f''| for f(x)=sigmoid(-x) is 1/(6*sqrt(3)) at f = 0.2113 / 0.7887."""
    x = np.linspace(-20.0, 20.0, 2000001)
    f = 1.0 / (1.0 + np.exp(x))          # sigmoid(-x)
    fpp = f * (1.0 - f) * (1.0 - 2.0 * f)  # f''
    absf = np.abs(fpp)

    peak = float(absf.max())
    theory = 1.0 / (6.0 * math.sqrt(3.0))
    print(f"T4 max |f''| = {peak:.12f}; 1/(6*sqrt(3)) = {theory:.12f}; diff = {abs(peak-theory):.3e}")
    assert abs(peak - theory) < 1e-6

    pos = np.where(x > 0.0)[0]
    neg = np.where(x < 0.0)[0]
    i_pos = pos[np.argmax(absf[pos])]
    i_neg = neg[np.argmax(absf[neg])]
    f_lo = (3.0 - math.sqrt(3.0)) / 6.0
    f_hi = (3.0 + math.sqrt(3.0)) / 6.0
    print(f"T4 peaks at f = {f[i_pos]:.6f} (target {f_lo:.6f}) and f = {f[i_neg]:.6f} (target {f_hi:.6f})")
    assert abs(f[i_pos] - f_lo) < 1e-3
    assert abs(f[i_neg] - f_hi) < 1e-3


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------

def test_q_lut_absolute_error():
    """Interpolated Q31 sigmoid LUT error bound over the clamped domain."""
    p_int = np.arange(-24576, 24577, dtype=np.int64)
    p = p_int.astype(np.float64) / 2048.0
    qh = lut_interp_vec(LUT_Q_Q31, p_int) / float(2 ** 31)
    q_exact = 1.0 / (1.0 + np.exp(p))  # sigmoid(-p)

    err = np.abs(qh - q_exact)
    bound = (_H_Q ** 2 / 8.0) * (1.0 / (6.0 * math.sqrt(3.0))) + 2.0 ** -30
    i = int(err.argmax())
    print(f"T5 max |q_lut/2**31 - sigmoid(-p)| = {err[i]:.9e} at p_int={int(p_int[i])} (p={p[i]:.5f})")
    print(f"T5 bound = {bound:.9e}")
    assert err[i] <= bound


# ---------------------------------------------------------------------------
# T6
# ---------------------------------------------------------------------------

def test_decay_rate_relative_error():
    """Relative error of the continuous decay rate |dq|/(1-q)."""
    p_int = np.arange(-24576, 24577, dtype=np.int64)
    p = p_int.astype(np.float64) / 2048.0
    q = 1.0 / (1.0 + np.exp(p))              # sigmoid(-p)
    qh = lut_interp_vec(LUT_Q_Q31, p_int) / float(2 ** 31)
    one_minus_q = 1.0 / (1.0 + np.exp(-p))   # sigmoid(p), accurate form of 1 - q

    ratio = np.abs(qh - q) / one_minus_q
    bound = _H_Q ** 2 / 8.0 + 3.0 * (2.0 ** -31) / one_minus_q

    i = int(ratio.argmax())
    print(f"T6 whole-sweep max |dq|/(1-q) = {ratio[i]:.6e} at p_int={int(p_int[i])} (p={p[i]:.4f})")
    print(f"T6 whole-sweep bound min = {float(bound.min()):.6e} (at p_int={int(p_int[int(bound.argmin())])})")

    mask = p_int <= -4 * 2048
    ratio_long = ratio[mask]
    p_long = p_int[mask]
    j = int(ratio_long.argmax())
    print(f"T6 long-memory (p<=-4) max |dq|/(1-q) = {ratio_long[j]:.6e} at p_int={int(p_long[j])} (p={p_long[j]/2048.0:.4f})")
    print(f"T6 paper yardstick h^2/8 = {_H_Q**2/8.0:.6e}")

    assert np.all(ratio <= bound)


# ---------------------------------------------------------------------------
# T7
# ---------------------------------------------------------------------------

def test_power_recurrence_error():
    """Integer repeated-multiplication P_n vs exact q**(n+1)."""
    p_int = np.arange(-24576, 24577, 7, dtype=np.int64)
    p = p_int.astype(np.float64) / 2048.0
    q_exact = 1.0 / (1.0 + np.exp(p))  # sigmoid(-p)
    q_lut = lut_interp_vec(LUT_Q_Q31, p_int)

    bound_step = 4.698e-5 + 2.0 ** -30
    P = q_lut.copy()
    for n in range(8):
        Pn = P / float(2 ** 31)
        exact = q_exact ** (n + 1)
        abserr = np.abs(Pn - exact)
        # 1 - q**(n+1) computed accurately as -expm1((n+1) log q)
        one_minus = -np.expm1((n + 1) * np.log(q_exact))
        rel = abserr / one_minus

        ia = int(abserr.argmax())
        ir = int(rel.argmax())
        print(f"T7 n={n}: max abs err = {abserr[ia]:.6e} (p_int={int(p_int[ia])})  "
              f"bound = {(n+1)*bound_step:.6e}")
        print(f"T7 n={n}: max decay-rate rel err = {rel[ir]:.6e} (p_int={int(p_int[ir])})")
        assert abserr[ia] <= (n + 1) * bound_step

        P = (P * q_lut + 2 ** 30) >> 31


# ---------------------------------------------------------------------------
# T8
# ---------------------------------------------------------------------------

def test_softplus_lut_error():
    """Interpolated Q24 softplus LUT error bound."""
    p_int = np.arange(-24576, 24577, dtype=np.int64)
    p = p_int.astype(np.float64) / 2048.0
    lut = lut_interp_vec(LUT_SOFTPLUS_Q24, p_int) / float(2 ** 24)
    exact = np.log1p(np.exp(p))

    err = np.abs(lut - exact)
    bound = (_H_Q ** 2 / 8.0) * 0.25 + 2.0 ** -23
    i = int(err.argmax())
    print(f"T8 max |softplus_lut/2**24 - softplus(p)| = {err[i]:.9e} at p_int={int(p_int[i])}")
    print(f"T8 bound = {bound:.9e}")
    assert err[i] <= bound


# ---------------------------------------------------------------------------
# T9
# ---------------------------------------------------------------------------

def test_silu_lut_error():
    """Interpolated Q12 SiLU LUT error bound with numerically measured M."""
    x_int = np.arange(-32768, 32768, dtype=np.int64)
    x = x_int.astype(np.float64) / 4096.0
    lut = lut_interp_vec(LUT_SILU_Q12, x_int) / 4096.0
    silu = x / (1.0 + np.exp(-x))

    # Numerical max |silu''| on a fine grid over [-8, 8].
    xg = np.linspace(-8.0, 8.0, 2000001)
    sg = 1.0 / (1.0 + np.exp(-xg))
    silu_pp = sg * (1.0 - sg) * (2.0 + xg * (1.0 - 2.0 * sg))
    abssp = np.abs(silu_pp)
    M = float(abssp.max())
    iM = int(abssp.argmax())
    print(f"T9 measured max |silu''| M = {M:.12f} attained at x = {xg[iM]:.9f}")
    assert abs(M - 0.5) < 1e-3
    assert abs(xg[iM]) < 1e-9

    err = np.abs(lut - silu)
    bound = (_H_SILU ** 2 / 8.0) * M + 1.0 / 4096.0
    i = int(err.argmax())
    print(f"T9 max |silu_lut/4096 - silu(x)| = {err[i]:.9e} at x_int={int(x_int[i])}")
    print(f"T9 bound = {bound:.9e}")
    assert err[i] <= bound


# ---------------------------------------------------------------------------
# T10
# ---------------------------------------------------------------------------

def test_accumulator_bounds():
    """32-bit accumulator safety and the need for a 64-bit requant product."""
    acc = 64 * 127 * 32768
    assert acc == 266338304
    assert acc < 2 ** 31 - 1

    prod = acc * 32767
    assert prod > 2 ** 31 - 1, "32-bit product must overflow"
    assert prod < 2 ** 63 - 1

    lg = math.log2(prod)
    print(f"T10 acc = {acc} (< 2**31-1); acc*32767 = {prod}; log2 = {lg:.6f}")
    assert 42.9 < lg < 43.0


# ---------------------------------------------------------------------------
# T11
# ---------------------------------------------------------------------------

# Cache for decay noise experiments: (seed, rounding) -> mean after burn-in
_DECAY_NOISE_CACHE = {}


def _decay_noise_mean(seed, rounding, eps=1e-3, steps=200000, burn_in=50000):
    """Simulate h <- decay(h) + noise and return mean of h over steps after burn_in.

    Uses Python ints for exact fixed-point arithmetic. Caches results per (seed, rounding).
    """
    key = (seed, rounding)
    if key in _DECAY_NOISE_CACHE:
        return _DECAY_NOISE_CACHE[key]

    P = round((1.0 - eps) * 2 ** 31)
    noise = np.random.default_rng(seed).integers(-3, 4, size=steps)
    h = 0
    total = 0
    count = steps - burn_in
    for i in range(steps):
        if rounding:
            h = (P * h + 2 ** 30) >> 31
        else:
            h = (P * h) >> 31
        h += int(noise[i])
        if i >= burn_in:
            total += h
    mean = total / count
    _DECAY_NOISE_CACHE[key] = mean
    return mean


def test_floor_decay_is_asymmetric():
    """Arithmetic-shift decay loses LSBs on positives and stagnates on negatives.

    (c) Multi-seed zero-mean noise experiment: floor decay drifts strongly negative.
        For each seed in 0..7, the mean after burn-in must be < -150 LSB.
        The average over seeds must be < -250 LSB (theory: about -0.35/eps .. -0.5/eps).
    """
    positives = (1, 2, 5, 100, 10 ** 4, 10 ** 6)
    for eps in (1e-3, 1e-4):
        P = round((1.0 - eps) * 2 ** 31)
        d = 2 ** 31 - P

        # (a) strictly decreasing for positive h.
        for h in positives:
            assert ((P * h) >> 31) < h, f"positive h={h} did not decrease (eps={eps})"

        # (b) negative stagnation while k*d < 2**31; changes above the threshold.
        kmax = (2 ** 31 - 1) // d
        for k in range(1, kmax + 1):
            assert ((P * (-k)) >> 31) == -k, f"negative h=-{k} should stagnate (eps={eps})"
        kbig = 2 ** 31 // d + 2
        assert (P * (-kbig)) >> 31 != -kbig, f"negative h=-{kbig} should change (eps={eps})"
        print(f"T11 eps={eps}: d={d}, stagnation up to |h|={kmax}, changes at k={kbig}")

    # (c) zero-mean noise experiment: floor decay drifts strongly negative.
    seeds = range(8)
    floor_means = [_decay_noise_mean(s, rounding=False) for s in seeds]
    for s, m in zip(seeds, floor_means):
        assert m < -150, f"seed {s}: floor mean {m:.1f} not < -150"
    avg_floor = sum(floor_means) / len(floor_means)
    assert avg_floor < -250, f"average floor mean {avg_floor:.1f} not < -250"
    print(f"T11 eps=1e-3 noise: floor means per seed = {[f'{m:.1f}' for m in floor_means]}")
    print(f"T11 eps=1e-3 noise: average floor mean = {avg_floor:.1f} LSB (theory ~ -0.35/eps .. -0.5/eps)")


# ---------------------------------------------------------------------------
# T12
# ---------------------------------------------------------------------------

def test_round_decay_is_unbiased():
    """Round-to-nearest decay is point-symmetric, dead-banded and drift-free.

    (b) Multi-seed zero-mean noise experiment: round-to-nearest removes the systematic
        negative drift of floor decay, but a symmetric dead-band of ±0.5/(1-P) LSB
        remains in which the state does not decay.
        For each seed in 0..7, |round_mean| < 0.5/eps (= 500, the dead-band half-width).
        The average over seeds has |avg| < 150.
        For every seed, round_mean - floor_mean > 150 (rounding removes the systematic bias).
    """
    hlist = list(range(1, 2001)) + [10 ** 4, 10 ** 6]
    for eps in (1e-3, 1e-4):
        P = round((1.0 - eps) * 2 ** 31)
        d = 2 ** 31 - P

        def step(x, P=P):
            return (P * x + 2 ** 30) >> 31

        # (a) symmetry: step(-h) and -step(h) differ by at most 1 LSB.
        for h in hlist:
            assert abs(step(-h) + step(h)) <= 1, f"asymmetry at h={h} (eps={eps})"

        # (c) dead-band holds for both signs.
        kmax = (2 ** 30 - 1) // d
        for k in range(1, kmax + 1):
            assert step(k) == k, f"dead-band failed for +{k} (eps={eps})"
            assert step(-k) == -k, f"dead-band failed for -{k} (eps={eps})"
        print(f"T12 eps={eps}: symmetric, dead-band up to |h|={kmax}")

    # (b) same noise experiment as T11: round-to-nearest removes the drift.
    seeds = range(8)
    round_means = [_decay_noise_mean(s, rounding=True) for s in seeds]
    floor_means = [_decay_noise_mean(s, rounding=False) for s in seeds]
    for s, m in zip(seeds, round_means):
        assert abs(m) < 500, f"seed {s}: round mean {m:.1f} not in (-500, 500)"
    avg_round = sum(round_means) / len(round_means)
    assert abs(avg_round) < 150, f"average round mean {avg_round:.1f} not in (-150, 150)"
    for s, (rm, fm) in enumerate(zip(round_means, floor_means)):
        diff = rm - fm
        assert diff > 150, f"seed {s}: round_mean - floor_mean = {diff:.1f} not > 150"
    print(f"T12 eps=1e-3 noise: round means per seed = {[f'{m:.1f}' for m in round_means]}")
    print(f"T12 eps=1e-3 noise: average round mean = {avg_round:.1f} LSB")
    print(f"T12 eps=1e-3 noise: round - floor per seed = {[f'{rm-fm:.1f}' for rm, fm in zip(round_means, floor_means)]}")


# ---------------------------------------------------------------------------
# T13
# ---------------------------------------------------------------------------

def test_parameter_counts():
    """Model parameter counts match the paper's closed forms."""
    from train.kws.model import MuMambaKWS

    model = MuMambaKWS()
    n = sum(p.numel() for p in model.parameters())
    print(f"T13 MuMambaKWS() parameters = {n}")
    assert n == 18060

    closed = (
        40 * 32 + 32
        + 2 * (32 + 128 * 32 + 64 * 4 + 64 + 20 * 64 + 64 * 4 + 64 + 64 + 32 * 64)
        + 32
        + 32 * 12 + 12
    )
    assert closed == 18060

    baseline = (
        (40 * 64 + 64)
        + (256 * 64 + 128 * 4 + 128 + 36 * 128 + 128 * 4 + 128 + 128 * 16 + 128 + 64 * 128)
        + (64 * 10 + 10)
    )
    assert baseline == 35914
    assert 100 * 128 * 16 == 204800
    print(f"T13 closed form = {closed}; baseline = {baseline}; expf/clip = {100*128*16}")


# ---------------------------------------------------------------------------
# T14
# ---------------------------------------------------------------------------

def test_state_memory_arithmetic():
    """Working-memory arithmetic from the paper."""
    assert 64 * 8 * 2 * 4 == 4096
    assert 3 * 64 * 2 * 2 == 768
    assert 4096 + 768 == 4864
    assert 51200 * 4 + 8192 * 2 + 6400 * 2 == 233984
    assert 278980 - 38992 == 239988
    assert round(239988 / 1024.0, 2) == 234.36
    print("T14 state memory identities hold: 4096, 768, 4864, 233984, 239988, 234.36 KiB")


# ---------------------------------------------------------------------------
# T15
# ---------------------------------------------------------------------------

def test_requant_rounding():
    """requant(acc, M, S) is exact round-half-up (ties toward +infinity)."""
    rng = np.random.default_rng(2024)
    n = 20000
    accs = rng.integers(-2 ** 28, 2 ** 28 + 1, size=n).astype(np.int64)
    mults = rng.integers(1, 32768, size=n).astype(np.int64)
    shifts = rng.integers(0, 32, size=n).astype(np.int64)

    max_dev = Fraction(0)
    for a, m, s in zip(accs.tolist(), mults.tolist(), shifts.tolist()):
        r = (a * m + ((1 << (s - 1)) if s > 0 else 0)) >> s
        exact = Fraction(a * m, 2 ** s)
        target = math.floor(exact + Fraction(1, 2))
        assert r == target, f"requant mismatch: acc={a}, M={m}, S={s}: {r} != {target}"
        dev = abs(Fraction(r) - exact)
        if dev > max_dev:
            max_dev = dev

    print(f"T15 checked {n} random (acc, M, S); max |requant - acc*M/2**S| = "
          f"{max_dev.numerator}/{max_dev.denominator} = {float(max_dev):.6f}")
    assert max_dev <= Fraction(1, 2)
