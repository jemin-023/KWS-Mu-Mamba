#ifndef MM_MATH_H
#define MM_MATH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Log2 in Q8 fixed-point (i.e. result * 256 = log2(x) * 256)
 * Returns -32768 if x == 0.
 */
int32_t mm_log2_q8(uint32_t x);

/* Fast integer inverse square root: returns (1 / sqrt(x)) in Q15 format,
 * scaled such that output * sqrt(x) ~= 32768.
 * For RMSNorm scaling.
 */
int32_t mm_rsqrt_q15(uint32_t x);

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
