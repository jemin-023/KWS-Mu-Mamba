#include <string.h>
#include <math.h>
#include "mu_mamba.h"
#include "mm_block.h"
#include "mm_block_q.h"
#include "mm_math.h"

/* Model weights pointers - zero RAM copy, accessed in place from flash const */
static const mm_model_weights_f32_t *g_model_weights_f32 = NULL;
static const mm_model_weights_q_t   *g_model_weights_q   = NULL;

extern const int16_t mm_lut_silu_q12[513];

void mm_set_weights_f32(const mm_model_weights_f32_t *w) {
    g_model_weights_f32 = w;
}

void mm_set_weights_q(const mm_model_weights_q_t *w) {
    g_model_weights_q = w;
}

/* Saturation tracking counters */
mm_sat_stats_t g_mm_sat_stats = {0};

void mm_get_sat_stats(mm_sat_stats_t *out) {
    if (out) *out = g_mm_sat_stats;
}

void mm_reset_sat_stats(void) {
    memset(&g_mm_sat_stats, 0, sizeof(g_mm_sat_stats));
}

const char *mm_weights_id(void) {
#ifdef MM_WEIGHTS_ID
    return MM_WEIGHTS_ID;
#else
    return "unknown";
#endif
}

int16_t mm_silu_q15(int16_t x) {
    return (int16_t)mm_lut_interp_s16(mm_lut_silu_q12, (int32_t)x, -(8 << 12), 7, 513);
}

size_t mm_context_size(void) {
    return sizeof(mm_context_t);
}

void mm_init(mm_context_t *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(mm_context_t));
    mm_frontend_init();
}

void mm_reset(mm_context_t *ctx) {
    if (!ctx) return;
    /* Reset streaming audio and layer persistent states without wiping weights */
    ctx->audio_buf_count = 0;
    memset(ctx->conv_ring_f, 0, sizeof(ctx->conv_ring_f));
    memset(ctx->conv_ring_q, 0, sizeof(ctx->conv_ring_q));
    memset(ctx->conv_ring_idx, 0, sizeof(ctx->conv_ring_idx));
    memset(ctx->ssm_h_f, 0, sizeof(ctx->ssm_h_f));
    memset(ctx->ssm_h_q, 0, sizeof(ctx->ssm_h_q));
    memset(ctx->logits_q, 0, sizeof(ctx->logits_q));
    memset(ctx->logits_f, 0, sizeof(ctx->logits_f));
    memset(ctx->smoothed_probs, 0, sizeof(ctx->smoothed_probs));
    ctx->refractory_timer = 0;
}

size_t mm_push_audio(mm_context_t *ctx, const int16_t *samples, size_t n_samples) {
    if (!ctx || !samples || n_samples == 0) return 0;

    size_t space = MM_FRAME_LEN - ctx->audio_buf_count;
    size_t to_copy = (n_samples < space) ? n_samples : space;

    memcpy(&ctx->audio_buf[ctx->audio_buf_count], samples, to_copy * sizeof(int16_t));
    ctx->audio_buf_count += (uint16_t)to_copy;
    return to_copy;
}

int mm_step_f32(mm_context_t *ctx, int16_t *features_out, float *logits_out) {
    if (!ctx || !g_model_weights_f32 || ctx->audio_buf_count < MM_FRAME_LEN) {
        return 0; /* Need more audio samples or weights not loaded */
    }

    /* 1. Extract 40 log-mel features from current 512-sample window */
    mm_frontend_process_frame(ctx->audio_buf, ctx->mel_features);
    if (features_out) {
        memcpy(features_out, ctx->mel_features, MM_MEL_BINS * sizeof(int16_t));
    }

    /* 2. Model Forward Step (Float32 Reference) */
    float mel_f[MM_MEL_BINS];
    for (int m = 0; m < MM_MEL_BINS; m++) {
        mel_f[m] = (float)ctx->mel_features[m];
    }

    /* Stem: Linear 40 -> 32 */
    float h_cur[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        float acc = g_model_weights_f32->stem_b[d];
        for (int m = 0; m < MM_MEL_BINS; m++) {
            acc += g_model_weights_f32->stem_w[d][m] * mel_f[m];
        }
        h_cur[d] = acc;
    }

    /* Residual Blocks: L=2 */
    float h_next[MM_DIM];
    for (int l = 0; l < MM_NUM_LAYERS; l++) {
        mm_block_step_f32(
            h_cur,
            &g_model_weights_f32->blocks[l],
            ctx->conv_ring_f[l],
            ctx->ssm_h_f[l],
            h_next
        );
        memcpy(h_cur, h_next, sizeof(h_cur));
    }

    /* Final RMSNorm */
    float sum_sq = 0.0f;
    for (int d = 0; d < MM_DIM; d++) {
        sum_sq += h_cur[d] * h_cur[d];
    }
    float rsqrt = 1.0f / sqrtf((sum_sq / (float)MM_DIM) + 1e-5f);
    float h_norm[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        h_norm[d] = h_cur[d] * rsqrt * g_model_weights_f32->final_norm_w[d];
    }

    /* Head: Linear 32 -> 12 */
    for (int c = 0; c < MM_NUM_CLASSES; c++) {
        float acc = g_model_weights_f32->head_b[c];
        for (int d = 0; d < MM_DIM; d++) {
            acc += g_model_weights_f32->head_w[c][d] * h_norm[d];
        }
        ctx->logits_f[c] = acc;
    }

    if (logits_out) {
        memcpy(logits_out, ctx->logits_f, MM_NUM_CLASSES * sizeof(float));
    }

    /* 3. Shift audio buffer by 320 samples (hop size) */
    memmove(
        ctx->audio_buf,
        &ctx->audio_buf[MM_FRAME_HOP],
        (MM_FRAME_LEN - MM_FRAME_HOP) * sizeof(int16_t)
    );
    ctx->audio_buf_count -= MM_FRAME_HOP;

    return 1;
}

int mm_step_q(mm_context_t *ctx, int16_t *features_out, int32_t *logits_out) {
    if (!ctx || !g_model_weights_q || ctx->audio_buf_count < MM_FRAME_LEN) {
        return 0; /* Need more audio samples or weights not loaded */
    }

    /* 1. Extract 40 log-mel features */
    mm_frontend_process_frame(ctx->audio_buf, ctx->mel_features);
    if (features_out) {
        memcpy(features_out, ctx->mel_features, MM_MEL_BINS * sizeof(int16_t));
    }

    /* 2. Model Forward Step (W8A16 Fixed Point) */
    /* Stem: W8A16 matvec with folded normalization, outputs Q12 */
    int16_t h_cur[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        int64_t acc = g_model_weights_q->stem_b[d];
        for (int m = 0; m < MM_MEL_BINS; m++) {
            acc += (int32_t)g_model_weights_q->stem_w[d][m] * ctx->mel_features[m];
        }
        int64_t scaled = (acc * g_model_weights_q->stem_scale_mult[d]) >> g_model_weights_q->stem_scale_shift[d];
#ifdef MM_DEBUG_SAT
        g_mm_sat_stats.stem_total++;
        if (scaled > 32767 || scaled < -32768) g_mm_sat_stats.stem_sat++;
#endif
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;
        h_cur[d] = (int16_t)scaled;
    }

    /* Residual Blocks: L=2 */
    int16_t h_next[MM_DIM];
    for (int l = 0; l < MM_NUM_LAYERS; l++) {
        mm_block_step_q(
            h_cur,
            &g_model_weights_q->blocks[l],
            ctx->conv_ring_q[l],
            ctx->ssm_h_q[l],
            h_next
        );
        memcpy(h_cur, h_next, sizeof(h_cur));
    }

    /* Final RMSNorm */
    uint64_t sum_sq = 0;
    for (int d = 0; d < MM_DIM; d++) {
        int32_t val = h_cur[d];
        sum_sq += (uint64_t)((int64_t)val * val);
    }
    uint32_t var = (uint32_t)(sum_sq >> 5); /* / 32 */
    int32_t rsqrt_val = mm_rsqrt_q12(var);

    int16_t h_norm[MM_DIM];
    for (int d = 0; d < MM_DIM; d++) {
        int32_t norm = (int32_t)(((int64_t)h_cur[d] * rsqrt_val) >> 12);
        h_norm[d] = (int16_t)(((int32_t)norm * g_model_weights_q->final_norm_w[d]) >> 14);
    }

    /* Head: W8A16 matvec -> 12 class logits (int32) */
    for (int c = 0; c < MM_NUM_CLASSES; c++) {
        int64_t acc = g_model_weights_q->head_b[c];
        for (int d = 0; d < MM_DIM; d++) {
            acc += (int32_t)g_model_weights_q->head_w[c][d] * h_norm[d];
        }
        int64_t scaled = (acc * g_model_weights_q->head_scale_mult[c]) >> g_model_weights_q->head_scale_shift[c];
        ctx->logits_q[c] = (int32_t)scaled;
    }

    if (logits_out) {
        memcpy(logits_out, ctx->logits_q, MM_NUM_CLASSES * sizeof(int32_t));
    }

    /* 3. Shift audio buffer by 320 samples (hop size) */
    memmove(
        ctx->audio_buf,
        &ctx->audio_buf[MM_FRAME_HOP],
        (MM_FRAME_LEN - MM_FRAME_HOP) * sizeof(int16_t)
    );
    ctx->audio_buf_count -= MM_FRAME_HOP;

    return 1;
}
