/* C1 speed ablation (float build only): the shipped decay recurrence P_n = P_{n-1} * q against the textbook
 * per-(d, n) decay exp(-Delta * (n + 1)), i.e. one expf per state element. Both are the same model
 * mathematically (q^(n+1) == exp(-(n+1) * softplus(p))); only the cost differs.
 * Enabled with -DMM_ENABLE_F32 -DMM_BENCH_ZOH_ABLATION (platformio env esp32-c3-ablation). */
#if defined(ESP_PLATFORM) && defined(MM_ENABLE_F32) && defined(MM_BENCH_ZOH_ABLATION)
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "bench.h"
#include "mu_mamba.h"
#include "mm_block.h"
#include "esp_cpu.h"
#include "esp_private/esp_clk.h"

extern const mm_model_weights_f32_t g_default_model_weights_f32;

#define ABLATION_ROUNDS 11

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* Copy of mm_block_step_f32 with the recurrence replaced by exp(-delta * (n + 1)) */
static void block_step_zoh(
    const float *x_in,
    const mm_block_weights_f32_t *w,
    float conv_ring[MM_D_CONV - 1][MM_D_INNER],
    float h[MM_D_INNER][MM_D_STATE],
    float *x_out
) {
    float sum_sq = 0.0f;
    for (int d = 0; d < MM_DIM; d++) {
        sum_sq += x_in[d] * x_in[d];
    }
    float rsqrt = 1.0f / sqrtf((sum_sq / (float)MM_DIM) + 1e-5f);

    float x_norm[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        x_norm[d] = x_in[d] * rsqrt * w->norm_w[d];
    }

    float xz[2 * MM_D_INNER];
    for (int i = 0; i < 2 * MM_D_INNER; i++) {
        float acc = 0.0f;
        for (int d = 0; d < MM_DIM; d++) {
            acc += w->in_proj_w[i][d] * x_norm[d];
        }
        xz[i] = acc;
    }

    const float *u = &xz[0];
    const float *z = &xz[MM_D_INNER];

    float u_prime[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        float conv_val = w->conv1d_b[d];
        conv_val += conv_ring[0][d] * w->conv1d_w[d][0];
        conv_val += conv_ring[1][d] * w->conv1d_w[d][1];
        conv_val += conv_ring[2][d] * w->conv1d_w[d][2];
        conv_val += u[d]            * w->conv1d_w[d][3];

        conv_ring[0][d] = conv_ring[1][d];
        conv_ring[1][d] = conv_ring[2][d];
        conv_ring[2][d] = u[d];

        u_prime[d] = silu_f(conv_val);
    }

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

    float y[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        float p = w->dt_proj_b[d];
        for (int r = 0; r < MM_DT_RANK; r++) {
            p += w->dt_proj_w[d][r] * dt_rank_val[r];
        }
        if (p > 12.0f) p = 12.0f;
        if (p < -12.0f) p = -12.0f;

        float delta = log1pf(expf(p));
        float v = delta * u_prime[d];

        float dot_y = w->D_skip[d] * u_prime[d];
        for (int n = 0; n < MM_D_STATE; n++) {
            float decay = expf(-delta * (float)(n + 1));
            h[d][n] = decay * h[d][n] + v * B[n];
            dot_y += h[d][n] * C[n];
        }
        y[d] = dot_y;
    }

    float y_gate[MM_D_INNER];
    for (int d = 0; d < MM_D_INNER; d++) {
        y_gate[d] = y[d] * silu_f(z[d]);
    }

    for (int o = 0; o < MM_DIM; o++) {
        float acc = 0.0f;
        for (int d = 0; d < MM_D_INNER; d++) {
            acc += w->out_proj_w[o][d] * y_gate[d];
        }
        x_out[o] = x_in[o] + acc;
    }
}

/* Copy of mm_step_f32 using block_step_zoh; shares the front-end and the audio buffer handling */
static int step_zoh(mm_context_t *ctx, float *logits_out) {
    const mm_model_weights_f32_t *W = &g_default_model_weights_f32;
    if (ctx->audio_buf_count < MM_FRAME_LEN) return 0;

    mm_frontend_process_frame(ctx->audio_buf, ctx->mel_features);

    float h_cur[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        float acc = W->stem_b[d];
        for (int m = 0; m < MM_MEL_BINS; m++) {
            acc += W->stem_w[d][m] * (float)ctx->mel_features[m];
        }
        h_cur[d] = acc;
    }

    float h_next[MM_DIM];
    for (int l = 0; l < MM_NUM_LAYERS; l++) {
        block_step_zoh(h_cur, &W->blocks[l], ctx->conv_ring_f[l], ctx->ssm_h_f[l], h_next);
        memcpy(h_cur, h_next, sizeof(h_cur));
    }

    float sum_sq = 0.0f;
    for (int d = 0; d < MM_DIM; d++) {
        sum_sq += h_cur[d] * h_cur[d];
    }
    float rsqrt = 1.0f / sqrtf((sum_sq / (float)MM_DIM) + 1e-5f);
    float h_norm[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        h_norm[d] = h_cur[d] * rsqrt * W->final_norm_w[d];
    }

    for (int c = 0; c < MM_NUM_CLASSES; c++) {
        float acc = W->head_b[c];
        for (int d = 0; d < MM_DIM; d++) {
            acc += W->head_w[c][d] * h_norm[d];
        }
        logits_out[c] = acc;
    }

    memmove(ctx->audio_buf, &ctx->audio_buf[MM_FRAME_HOP], (MM_FRAME_LEN - MM_FRAME_HOP) * sizeof(int16_t));
    ctx->audio_buf_count -= MM_FRAME_HOP;
    return 1;
}

static void sort_u32(uint32_t *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[i]) {
                uint32_t tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }
}

static int16_t s_noise[MM_FRAME_LEN];
static mm_context_t s_c1_ctx;
static mm_context_t s_zoh_ctx;
static uint32_t s_c1_cycles[ABLATION_ROUNDS];
static uint32_t s_zoh_cycles[ABLATION_ROUNDS];
static float s_c1_logits[MM_NUM_CLASSES];
static float s_zoh_logits[MM_NUM_CLASSES];

void mm_bench_ablation_print(void) {
    uint32_t lcg = 0x12345678u; /* same noise frame as the main bench */
    for (int i = 0; i < MM_FRAME_LEN; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        s_noise[i] = (int16_t)(lcg >> 16);
    }

    mm_init(&s_c1_ctx);
    mm_set_weights_f32(&g_default_model_weights_f32);
    mm_init(&s_zoh_ctx);
    mm_push_audio(&s_c1_ctx, s_noise, MM_FRAME_LEN);
    mm_push_audio(&s_zoh_ctx, s_noise, MM_FRAME_LEN);
    (void)mm_step_f32(&s_c1_ctx, NULL, s_c1_logits);   /* warm-up, also advances the state */
    (void)step_zoh(&s_zoh_ctx, s_zoh_logits);

    float max_diff = 0.0f;
    int all_stepped = 1;
    for (int r = 0; r < ABLATION_ROUNDS; r++) {
        mm_push_audio(&s_c1_ctx, s_noise, MM_FRAME_HOP);
        mm_push_audio(&s_zoh_ctx, s_noise, MM_FRAME_HOP);

        uint32_t c0 = esp_cpu_get_cycle_count();
        int ret_c1 = mm_step_f32(&s_c1_ctx, NULL, s_c1_logits);
        uint32_t c1 = esp_cpu_get_cycle_count();
        int ret_zoh = step_zoh(&s_zoh_ctx, s_zoh_logits);
        uint32_t c2 = esp_cpu_get_cycle_count();
        if (ret_c1 != 1 || ret_zoh != 1) all_stepped = 0;
        s_c1_cycles[r] = c1 - c0;
        s_zoh_cycles[r] = c2 - c1;

        for (int c = 0; c < MM_NUM_CLASSES; c++) {
            float diff = fabsf(s_c1_logits[c] - s_zoh_logits[c]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    sort_u32(s_c1_cycles, ABLATION_ROUNDS);
    sort_u32(s_zoh_cycles, ABLATION_ROUNDS);

    double mhz = (double)esp_clk_cpu_freq() / 1000000.0;
    uint32_t c1_med = s_c1_cycles[ABLATION_ROUNDS / 2];
    uint32_t zoh_med = s_zoh_cycles[ABLATION_ROUNDS / 2];
    printf("==== MU-MAMBA ZOH ABLATION BEGIN ====\n");
    printf("ABLATION_ROUNDS: %d\n", ABLATION_ROUNDS);
    printf("ABLATION_ALL_STEPPED: %d\n", all_stepped);
    printf("C1_STEP_CYCLES_MEDIAN: %lu (%.3f ms)\n", (unsigned long)c1_med, (double)c1_med / 1000.0 / mhz);
    printf("ZOH_EXPF_STEP_CYCLES_MEDIAN: %lu (%.3f ms)\n", (unsigned long)zoh_med, (double)zoh_med / 1000.0 / mhz);
    printf("MAX_ABS_LOGIT_DIFF: %.9g\n", (double)max_diff);
    printf("==== MU-MAMBA ZOH ABLATION END ====\n");
}
#endif
