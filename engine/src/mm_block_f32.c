#include <math.h>
#include <string.h>
#include "mu_mamba.h"
#include "mm_block.h"

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

void mm_block_step_f32(
    const float *x_in,
    const mm_block_weights_f32_t *w,
    float conv_ring[MM_D_CONV - 1][MM_D_INNER],
    float h[MM_D_INNER][MM_D_STATE],
    float *x_out
) {
    /* 1. RMSNorm: x_norm = x_in * rsqrt(mean(x^2) + eps) * norm_w */
    float sum_sq = 0.0f;
    for (int d = 0; d < MM_DIM; d++) {
        sum_sq += x_in[d] * x_in[d];
    }
    float rsqrt = 1.0f / sqrtf((sum_sq / (float)MM_DIM) + 1e-5f);

    float x_norm[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        x_norm[d] = x_in[d] * rsqrt * w->norm_w[d];
    }

    /* 2. in_proj: projects D -> 2 * D_inner (x and z) */
    float xz[2 * MM_D_INNER];
    for (int i = 0; i < 2 * MM_D_INNER; i++) {
        float acc = 0.0f;
        for (int d = 0; d < MM_DIM; d++) {
            acc += w->in_proj_w[i][d] * x_norm[d];
        }
        xz[i] = acc;
    }

    /* u = xz[0 .. D_inner - 1], z = xz[D_inner .. 2 * D_inner - 1] */
    const float *u = &xz[0];
    const float *z = &xz[MM_D_INNER]; /* z is untouched until gate */

    /* 3. 1D Causal Convolution on u with ring buffer */
    float u_prime[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        /* Conv taps: ring[0]=t-3, ring[1]=t-2, ring[2]=t-1, u[d]=t */
        float conv_val = w->conv1d_b[d];
        conv_val += conv_ring[0][d] * w->conv1d_w[d][0];
        conv_val += conv_ring[1][d] * w->conv1d_w[d][1];
        conv_val += conv_ring[2][d] * w->conv1d_w[d][2];
        conv_val += u[d]            * w->conv1d_w[d][3];

        /* Update ring buffer for this channel */
        conv_ring[0][d] = conv_ring[1][d];
        conv_ring[1][d] = conv_ring[2][d];
        conv_ring[2][d] = u[d];

        /* Activation: SiLU(conv_val) */
        u_prime[d] = silu_f(conv_val);
    }

    /* 4. x_proj: projects u' (D_inner) -> (dt_rank + 2 * d_state) = 20 */
    float x_ssm[MM_DT_RANK + 2 * MM_D_STATE];
    for (int i = 0; i < (MM_DT_RANK + 2 * MM_D_STATE); i++) {
        float acc = 0.0f;
        for (int d = 0; d < MM_D_INNER; d++) {
            acc += w->x_proj_w[i][d] * u_prime[d];
        }
        x_ssm[i] = acc;
    }

    const float *dt_rank_val = &x_ssm[0];
    const float *B = &x_ssm[MM_DT_RANK];
    const float *C = &x_ssm[MM_DT_RANK + MM_D_STATE];

    /* 5 & 6. dt_proj -> p, stable softplus/sigmoid, C1 recurrence */
    float y[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        float p = w->dt_proj_b[d];
        for (int r = 0; r < MM_DT_RANK; r++) {
            p += w->dt_proj_w[d][r] * dt_rank_val[r];
        }

        /* Clamp p to representative range */
        if (p > 12.0f) p = 12.0f;
        if (p < -12.0f) p = -12.0f;

        /* Stable softplus and sigmoid */
        float delta = log1pf(expf(p));
        float q = 1.0f / (1.0f + expf(p));

        float v = delta * u_prime[d];

        /* C1 power recurrence: P_n = P_{n-1} * q, P_0 = q */
        float P = q;
        /* Start accumulator with Dskip[d] * u_prime[d] */
        float dot_y = w->D_skip[d] * u_prime[d];
        for (int n = 0; n < MM_D_STATE; n++) {
            h[d][n] = P * h[d][n] + v * B[n];
            dot_y += h[d][n] * C[n];
            P *= q;
        }
        y[d] = dot_y;
    }

    /* 7. Gating with z: y_gate = y * SiLU(z) */
    float y_gate[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        y_gate[d] = y[d] * silu_f(z[d]);
    }

    /* 8. out_proj & residual: x_out = x_in + out_proj(y_gate) */
    for (int o = 0; o < MM_DIM; o++) {
        float acc = 0.0f;
        for (int d = 0; d < MM_D_INNER; d++) {
            acc += w->out_proj_w[o][d] * y_gate[d];
        }
        x_out[o] = x_in[o] + acc;
    }
}
