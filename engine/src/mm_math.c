#include "mm_math.h"

/* 257-entry table for log2(1 + i/256.0) in Q8 (scaled by 256).
 * Generated for i = 0 .. 256: round(256 * log2(1.0 + i / 256.0))
 */
static const uint16_t log2_mantissa_lut[257] = {
      0,   1,   3,   4,   6,   7,   9,  10,  11,  13,  14,  16,  17,  18,  20,  21,
     22,  24,  25,  26,  28,  29,  30,  32,  33,  34,  36,  37,  38,  40,  41,  42,
     44,  45,  46,  47,  49,  50,  51,  52,  54,  55,  56,  57,  59,  60,  61,  62,
     63,  65,  66,  67,  68,  69,  71,  72,  73,  74,  75,  77,  78,  79,  80,  81,
     82,  84,  85,  86,  87,  88,  89,  90,  92,  93,  94,  95,  96,  97,  98,  99,
    100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 178,
    179, 180, 181, 182, 183, 184, 185, 185, 186, 187, 188, 189, 190, 191, 192, 192,
    193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 203, 204, 205, 206,
    207, 208, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217, 218, 219,
    220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231, 231,
    232, 233, 234, 234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 244,
    244, 245, 246, 247, 247, 248, 249, 249, 250, 251, 252, 252, 253, 254, 255, 255,
    256
};

int32_t mm_log2_q8(uint32_t x) {
    if (x == 0) {
        return 0; /* Smooth floor at zero */
    }
    int lz = __builtin_clz(x);
    int integer_part = 31 - lz;

    uint32_t x_norm = x << lz;

    uint32_t idx = (x_norm >> 23) & 0xFF;
    uint32_t frac = (x_norm >> 16) & 0x7F;

    uint32_t y0 = log2_mantissa_lut[idx];
    uint32_t y1 = log2_mantissa_lut[idx + 1];

    uint32_t mantissa = y0 + (((y1 - y0) * frac + 64) >> 7);

    return (integer_part << 8) + (int32_t)mantissa;
}


int32_t mm_rsqrt_q12(uint32_t x) {            /* returns 4096 / rms_raw; input Q format cancels */
    if (x == 0) return 1 << 24;
    uint64_t n = (uint64_t)x << 16, res = 0, bit = (uint64_t)1 << 46;
    while (bit > n) bit >>= 2;
    while (bit) { if (n >= res + bit) { n -= res + bit; res = (res >> 1) + bit; } else res >>= 1; bit >>= 2; }
    if (res == 0) return 1 << 24;
    uint64_t r = ((uint64_t)1 << 32) / res;
    return r > (1u << 24) ? (1 << 24) : (int32_t)r;
}

int32_t mm_rsqrt_q15(uint32_t x) {
    return mm_rsqrt_q12(x);
}

int32_t mm_lut_interp_s16(const int16_t *lut, int32_t x, int32_t x_min, int32_t step_shift, int32_t num_entries) {
    int32_t diff = x - x_min;
    if (diff <= 0) return lut[0];

    int32_t idx = diff >> step_shift;
    if (idx >= num_entries - 1) return lut[num_entries - 1];

    int32_t frac = diff & ((1 << step_shift) - 1);
    int32_t y0 = lut[idx];
    int32_t y1 = lut[idx + 1];

    return y0 + (((y1 - y0) * frac + (1 << (step_shift - 1))) >> step_shift);
}

int32_t mm_lut_interp_q31(const int32_t *lut, int32_t x, int32_t x_min, int32_t step_shift, int32_t num_entries) {
    int32_t diff = x - x_min;
    if (diff <= 0) return lut[0];

    int32_t idx = diff >> step_shift;
    if (idx >= num_entries - 1) return lut[num_entries - 1];

    int32_t frac = diff & ((1 << step_shift) - 1);
    int64_t y0 = lut[idx];
    int64_t y1 = lut[idx + 1];

    return (int32_t)(y0 + (((y1 - y0) * frac + (1 << (step_shift - 1))) >> step_shift));
}
