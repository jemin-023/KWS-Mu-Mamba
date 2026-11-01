#ifndef MM_MATH_H
#define MM_MATH_H

#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define MM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MM_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__)
#define MM_OPAQUE32(x) __asm__("" : "+r"(x))      /* empty asm: tells GCC the value of x is unknown from here on */
#else
#define MM_OPAQUE32(x) ((void)0)
#endif

/* Log2 in Q8 fixed-point (i.e. result * 256 = log2(x) * 256)
 * Returns 0 if x == 0.
 */
int32_t mm_log2_q8(uint32_t x);

/* Requantization with round-half-up:
 *   Scaled = (acc * M + R) >> S with R = 2^(S-1) if S > 0 else 0.
 * 64-bit intermediate, arithmetic shift.
 * Identity: floor((p + 2^(s-1)) / 2^s) == floor((floor(p / 2^(s-1)) + 1) / 2).
 */
static inline int64_t mm_requant(int64_t acc, int32_t mult, uint32_t shift) {
    int64_t prod = acc * (int64_t)mult;
    if (shift == 0) return prod;
    return ((prod >> (shift - 1)) + 1) >> 1;
}

static inline int64_t mm_requant32(int32_t acc, int32_t mult, uint32_t shift) {
    int64_t prod = (int64_t)acc * mult;
    if (shift == 0) return prod;
    return ((prod >> (shift - 1)) + 1) >> 1;
}

static inline int32_t mm_sat32(int64_t x) {
    if (MM_UNLIKELY(x != (int32_t)x)) return x < 0 ? INT32_MIN : INT32_MAX;
    return (int32_t)x;
}

static inline int16_t mm_sat16(int64_t x) {
    if (MM_UNLIKELY(x != (int16_t)x)) return x < 0 ? INT16_MIN : INT16_MAX;
    return (int16_t)x;
}

/* Fast integer inverse square root: returns floor(2^32 / isqrt(x << 16))
 * ~= 2^24 / sqrt(x), so that (v * result) >> 12 equals v / sqrt(x) in Q12;
 * for x == 0 it returns 2^24.
 * For RMSNorm scaling.
 */
int32_t mm_rsqrt_q12(uint32_t x);

/* Linear interpolation helper for lookup tables.
 * lut: array of int16_t of length num_entries
 * x: input value
 * x_min: minimum input domain
 * step_shift: log2 of spacing between entries (i.e. step = 1 << step_shift)
 * num_entries: number of points in table
 */
int32_t mm_lut_interp_s16(const int16_t *lut, int32_t x, int32_t x_min, int32_t step_shift, int32_t num_entries);

/* Q31 linear interpolation helper for probability/decay tables (e.g. q = sigmoid(-p)) */
int32_t mm_lut_interp_q31(const int32_t *lut, int32_t x, int32_t x_min, int32_t step_shift, int32_t num_entries);

#ifdef __cplusplus
}
#endif

#endif /* MM_MATH_H */
