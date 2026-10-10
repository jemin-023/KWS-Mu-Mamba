#include "mu_mamba.h"
#include "mm_math.h"
#include "mm_frontend_tables.h"

void mm_frontend_init(void) {
    /* Tables are statically declared in mm_frontend_tables.h */
}

void mm_frontend_process_frame(const int16_t *pcm_512, int16_t *mel_40_out) {
    int16_t re[MM_FFT_SIZE];
    int16_t im[MM_FFT_SIZE];

    /* 1. Block Floating Point: Find peak magnitude in input frame */
    int32_t max_val = 0;
    for (int n = 0; n < MM_FFT_SIZE; n++) {
        int32_t v = (int32_t)pcm_512[n];
        if (v < 0) v = -v;
        if (v > max_val) max_val = v;
    }
    if (max_val > 32767) max_val = 32767;

    /* Compute input pre-shift (leave 1 bit headroom for windowing) */
    int shift = 0;
    if (max_val > 0) {
        int lz = __builtin_clz((uint32_t)max_val) - 16;
        shift = lz - 1;
        if (shift < 0) shift = 0;
        if (shift > 14) shift = 14;
    }

    /* 2. Apply pre-shift, Hann window, and bit-reversal into FFT buffers */
    for (int n = 0; n < MM_FFT_SIZE; n++) {
        int32_t s = (int32_t)pcm_512[n] << shift;
        int32_t win_val = (s * mm_hann_window_q15[n]) >> 15;
        uint16_t rev_idx = mm_bitrev_512[n];
        re[rev_idx] = (int16_t)win_val;
        im[rev_idx] = 0;
    }

    /* 3. Radix-2 Cooley-Tukey DIT FFT with per-stage 1-bit scaling */
    /* 9 stages: 2^1 = 2 up to 2^9 = 512 */
    for (int s = 1; s <= 9; s++) {
        int len = 1 << s;
        int half = len >> 1;
        int twiddle_step = 256 / half;

        for (int i = 0; i < MM_FFT_SIZE; i += len) {
            for (int j = 0; j < half; j++) {
                int k = j * twiddle_step;
                int16_t w_re = mm_twiddle_re_q15[k];
                int16_t w_im = mm_twiddle_im_q15[k];

                int u_idx = i + j;
                int v_idx = i + j + half;

                int16_t v_re = re[v_idx];
                int16_t v_im = im[v_idx];

                /* Complex multiplication: v * W */
                int32_t t_re = ((int32_t)v_re * w_re - (int32_t)v_im * w_im) >> 15;
                int32_t t_im = ((int32_t)v_re * w_im + (int32_t)v_im * w_re) >> 15;

                int16_t u_re = re[u_idx];
                int16_t u_im = im[u_idx];

                /* Butterfly with 1-bit right shift */
                re[u_idx] = (int16_t)((u_re + t_re) >> 1);
                im[u_idx] = (int16_t)((u_im + t_im) >> 1);
                re[v_idx] = (int16_t)((u_re - t_re) >> 1);
                im[v_idx] = (int16_t)((u_im - t_im) >> 1);
            }
        }
    }

    /* 4. Compute Power Spectrum: 257 bins (k = 0 .. 256) */
    uint32_t power[MM_FFT_SIZE / 2 + 1];
    for (int k = 0; k <= MM_FFT_SIZE / 2; k++) {
        int32_t r = re[k];
        int32_t m = im[k];
        power[k] = (uint32_t)(r * r + m * m);
    }

    /* 5. Sparse Mel Filterbank accumulation and log2(energy + 1) */
    int32_t shift_adj_q8 = (2 * shift) << 8;

    for (int m = 0; m < MM_MEL_BINS; m++) {
        uint16_t start = mm_mel_filters[m].start_bin;
        uint16_t num   = mm_mel_filters[m].num_bins;
        uint16_t off   = mm_mel_filters[m].weight_offset;

        uint64_t acc = 0;
        for (uint16_t j = 0; j < num; j++) {
            acc += (uint64_t)power[start + j] * mm_mel_weights_q15[off + j];
        }

        uint32_t energy = (uint32_t)(acc >> 15);
        /* Compute log2(energy + 1) in Q8 (smooth floor, 0 for zero energy) */
        int32_t log_val = mm_log2_q8(energy + 1);

        /* Remove input pre-shift exponent in Q8 log domain */
        int32_t adj = log_val - shift_adj_q8;
        if (adj < 0) adj = 0;
        mel_40_out[m] = (int16_t)adj;
    }
}
