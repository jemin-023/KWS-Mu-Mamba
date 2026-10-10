#include <string.h>
#include "mu_mamba.h"
#include "mm_math.h"
#include "mm_block_q.h"

extern const int16_t mm_lut_silu_q12[513];
extern const int32_t mm_lut_q_q31[513];
extern const int32_t mm_lut_softplus_q24[513];

#ifdef MM_DEBUG_SAT
extern mm_sat_stats_t g_mm_sat_stats;
#endif

void mm_block_step_q(
    const int16_t *x_in,
    const mm_block_weights_q_t *w,
    int16_t conv_ring[MM_D_CONV - 1][MM_D_INNER],
    int32_t h[MM_D_INNER][MM_D_STATE],
    int16_t *x_out
) {
    /* 1. Integer RMSNorm */
    uint64_t sum_sq = 0;
    for (int d = 0; d < MM_DIM; d++) {
        int64_t val = x_in[d];
        sum_sq += (uint64_t)(val * val);
    }
    uint32_t var = (uint32_t)(sum_sq >> 5); /* / 32 */
    int32_t rsqrt_val = mm_rsqrt_q12(var);

    int16_t x_norm[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        int32_t normalized = (int32_t)(((int64_t)x_in[d] * rsqrt_val) >> 12);
        x_norm[d] = (int16_t)(((int32_t)normalized * w->norm_w[d]) >> 14);
    }

    /* 2. in_proj: W8A16 matvec */
    int16_t xz[2 * MM_D_INNER];
    for (int i = 0; i < 2 * MM_D_INNER; i++) {
        int32_t acc = 0;
        for (int d = 0; d < MM_DIM; d++) {
            acc += (int32_t)w->in_proj_w[i][d] * x_norm[d];
        }
        int64_t scaled = ((int64_t)acc * w->in_proj_scale_mult[i]) >> w->in_proj_scale_shift[i];
#ifdef MM_DEBUG_SAT
        g_mm_sat_stats.in_proj_total++;
        if (scaled > 32767 || scaled < -32768) g_mm_sat_stats.in_proj_sat++;
#endif
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;
        xz[i] = (int16_t)scaled;
    }

    const int16_t *u = &xz[0];
    const int16_t *z = &xz[MM_D_INNER];

    /* 3. 1D Causal Conv with ring buffer & SiLU */
    int16_t u_prime[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        int32_t acc = w->conv1d_b[d];
        acc += (int32_t)conv_ring[0][d] * w->conv1d_w[d][0];
        acc += (int32_t)conv_ring[1][d] * w->conv1d_w[d][1];
        acc += (int32_t)conv_ring[2][d] * w->conv1d_w[d][2];
        acc += (int32_t)u[d]            * w->conv1d_w[d][3];

        conv_ring[0][d] = conv_ring[1][d];
        conv_ring[1][d] = conv_ring[2][d];
        conv_ring[2][d] = u[d];

        int64_t scaled = ((int64_t)acc * w->conv1d_scale_mult[d]) >> w->conv1d_scale_shift[d];
#ifdef MM_DEBUG_SAT
        g_mm_sat_stats.conv_total++;
        if (scaled > 32767 || scaled < -32768) g_mm_sat_stats.conv_sat++;
#endif
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;

        /* SiLU LUT interpolation: domain [-8, 8] in Q12, min = -(8 << 12) = -32768, step_shift = 7 */
        u_prime[d] = (int16_t)mm_lut_interp_s16(mm_lut_silu_q12, (int32_t)scaled, -(8 << 12), 7, 513);
    }

    /* 4. x_proj: W8A16 matvec -> 20 outputs in Q_XPROJ */
    int16_t x_ssm[MM_DT_RANK + 2 * MM_D_STATE];
    for (int i = 0; i < (MM_DT_RANK + 2 * MM_D_STATE); i++) {
        int32_t acc = 0;
        for (int d = 0; d < MM_D_INNER; d++) {
            acc += (int32_t)w->x_proj_w[i][d] * u_prime[d];
        }
        int64_t scaled = ((int64_t)acc * w->x_proj_scale_mult[i]) >> w->x_proj_scale_shift[i];
#ifdef MM_DEBUG_SAT
        g_mm_sat_stats.x_proj_total++;
        if (scaled > 32767 || scaled < -32768) g_mm_sat_stats.x_proj_sat++;
#endif
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;
        x_ssm[i] = (int16_t)scaled;
    }

    const int16_t *dt_rank_val = &x_ssm[0];
    const int16_t *B = &x_ssm[MM_DT_RANK];
    const int16_t *C = &x_ssm[MM_DT_RANK + MM_D_STATE];

    /* 5 & 6. dt_proj -> p, C1 Q31 Recurrence */
    int32_t y[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        int32_t p = w->dt_proj_b[d];
        for (int r = 0; r < MM_DT_RANK; r++) {
            p += (int32_t)w->dt_proj_w[d][r] * dt_rank_val[r];
        }
        int32_t p_scaled = (int32_t)(((int64_t)p * w->dt_proj_scale_mult[d]) >> w->dt_proj_scale_shift[d]);
        if (p_scaled > (12 << 11)) p_scaled = (12 << 11);
        if (p_scaled < -(12 << 11)) p_scaled = -(12 << 11);

        /* Lookup q in Q31: domain [-16, 16] in Q11, min = -(16 << 11) = -32768, step_shift = 7 */
        int32_t q = mm_lut_interp_q31(mm_lut_q_q31, p_scaled, -(16 << 11), 7, 513);

        /* Lookup Delta in Q24: domain [-16, 16] in Q11, min = -(16 << 11) = -32768, step_shift = 7 */
        int32_t delta = mm_lut_interp_q31(mm_lut_softplus_q24, p_scaled, -(16 << 11), 7, 513);

        /* v = (delta * u_prime[d]) >> 24 in Q12 */
        int32_t v = (int32_t)(((int64_t)delta * u_prime[d] + (1 << 23)) >> 24);

        /* C1 Recurrence: P_n = P_{n-1} * q in Q31 */
        int32_t P = q;
        int64_t dot_y = 0;
        for (int n = 0; n < MM_D_STATE; n++) {
            /* h is in Q(12 + MM_Q_XPROJ) */
#ifdef MM_DEBUG_SAT
            g_mm_sat_stats.h_total++;
            int64_t new_h = (int64_t)(((int64_t)P * h[d][n]) >> 31) + ((int64_t)v * (int32_t)B[n]);
            if (new_h > 2147483647LL || new_h < -2147483648LL) g_mm_sat_stats.h_sat++;
#endif
            h[d][n] = (int32_t)(((int64_t)P * h[d][n]) >> 31) + (v * (int32_t)B[n]);
            dot_y += (int64_t)h[d][n] * C[n]; /* Q(12 + MM_Q_XPROJ) * Q(MM_Q_XPROJ) = Q(12 + 2 * MM_Q_XPROJ) */
            P = (int32_t)(((int64_t)P * q) >> 31);
        }

        /* dot_y >> (2 * MM_Q_XPROJ) is in Q12. Add D_skip in Q12: (D_skip * u_prime) >> 12 */
        int64_t skip_val = ((int64_t)w->D_skip[d] * u_prime[d]) >> 12;
        int64_t dot_y_scaled = (dot_y + (1LL << (2 * MM_Q_XPROJ - 1))) >> (2 * MM_Q_XPROJ);
        y[d] = (int32_t)(dot_y_scaled + skip_val);
    }

    /* 7. Gating: y_gate = (y * SiLU(z)) >> 12 */
    int32_t y_gate[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        int32_t z_q12 = (int32_t)z[d] << (12 - MM_Q_INPROJ);
        int16_t silu_z = (int16_t)mm_lut_interp_s16(mm_lut_silu_q12, z_q12, -(8 << 12), 7, 513);
        y_gate[d] = (int32_t)(((int64_t)y[d] * silu_z) >> 12);
    }

    /* 8. out_proj + residual */
    for (int o = 0; o < MM_DIM; o++) {
        int64_t acc = 0;
        for (int d = 0; d < MM_D_INNER; d++) {
            acc += (int64_t)w->out_proj_w[o][d] * y_gate[d];
        }
        int64_t scaled = (acc * w->out_proj_scale_mult[o]) >> w->out_proj_scale_shift[o];
        int32_t out_val = (int32_t)x_in[o] + (int32_t)scaled;
#ifdef MM_DEBUG_SAT
        g_mm_sat_stats.resid_total++;
        if (out_val > 32767 || out_val < -32768) g_mm_sat_stats.resid_sat++;
#endif
        if (out_val > 32767) out_val = 32767;
        if (out_val < -32768) out_val = -32768;
        x_out[o] = (int16_t)out_val;
    }
}
