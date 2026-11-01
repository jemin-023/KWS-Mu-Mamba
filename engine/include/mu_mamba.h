#ifndef MU_MAMBA_H
#define MU_MAMBA_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Audio & Front-end configuration */
#define MM_SAMPLE_RATE    16000
#define MM_FRAME_LEN      512   /* 32 ms */
#define MM_FRAME_HOP      320   /* 20 ms -> 50 tokens/s */
#define MM_FFT_SIZE       512
#define MM_MEL_BINS       40

/* Model Architecture Configuration */
#if defined(__has_include)
#if __has_include("mm_config.h")
#include "mm_config.h"
#elif __has_include("gen/mm_config.h")
#include "gen/mm_config.h"
#endif
#endif

/* Fallback Defaults if not defined by mm_config.h (the generated header is the single source of truth) */
#ifndef MM_NUM_CLASSES
#define MM_NUM_CLASSES    12
#endif
#ifndef MM_DIM
#define MM_DIM            32    /* D = 32 */
#endif
#ifndef MM_D_INNER
#define MM_D_INNER        64    /* D_i = 64 */
#endif
#ifndef MM_D_STATE
#define MM_D_STATE        8     /* N = 8 */
#endif
#ifndef MM_DT_RANK
#define MM_DT_RANK        4     /* R = 4 */
#endif
#ifndef MM_D_CONV
#define MM_D_CONV         4     /* K = 4 (kernel size) */
#endif
#ifndef MM_NUM_LAYERS
#define MM_NUM_LAYERS     2     /* L = 2 */
#endif
#ifndef MM_Q_RESID
#define MM_Q_RESID        6     /* Fractional bits for residual stream */
#endif
#ifndef MM_Q_XPROJ
#define MM_Q_XPROJ        10    /* Fractional bits for x_proj output */
#endif
#ifndef MM_Q_INPROJ
#define MM_Q_INPROJ       11    /* Fractional bits for in_proj output */
#endif

/* Class labels */
typedef enum {
    MM_CLASS_YES = 0,
    MM_CLASS_NO,
    MM_CLASS_UP,
    MM_CLASS_DOWN,
    MM_CLASS_LEFT,
    MM_CLASS_RIGHT,
    MM_CLASS_ON,
    MM_CLASS_OFF,
    MM_CLASS_STOP,
    MM_CLASS_GO,
    MM_CLASS_UNKNOWN,
    MM_CLASS_SILENCE,
    MM_CLASS_COUNT = 12
} mm_class_t;

/* Streaming context */
typedef struct {
    /* Audio buffer: holds incoming PCM samples until a hop of 320 is processed */
    int16_t audio_buf[MM_FRAME_LEN];
    uint16_t audio_buf_count;

    /* Front-end temporary state */
    int16_t mel_features[MM_MEL_BINS];

    /* Persistent state for each layer */
    /* Conv ring buffer: (K - 1) tokens x D_inner per layer */
    int16_t conv_ring_q[MM_NUM_LAYERS][MM_D_CONV - 1][MM_D_INNER];

    /* State H: D_inner x D_state per layer */
    int32_t ssm_h_q[MM_NUM_LAYERS][MM_D_INNER][MM_D_STATE];

    /* Output logits */
    int32_t logits_q[MM_NUM_CLASSES];

    /* Continuous streaming decision rule: EMA of logit margins */
    int32_t smoothed_margins[MM_NUM_CLASSES];
    int16_t refractory_timer;
#ifdef MM_ENABLE_F32
    /* Float32 reference state (host only) */
    float   conv_ring_f[MM_NUM_LAYERS][MM_D_CONV - 1][MM_D_INNER];
    float   ssm_h_f[MM_NUM_LAYERS][MM_D_INNER][MM_D_STATE];
    float   logits_f[MM_NUM_CLASSES];
#endif
} mm_context_t;

/* Public API */
size_t mm_context_size(void);
void mm_init(mm_context_t *ctx);
void mm_reset(mm_context_t *ctx);
size_t mm_push_audio(mm_context_t *ctx, const int16_t *samples, size_t n_samples);

/* Model Weights API */
#ifdef MM_ENABLE_F32
struct mm_model_weights_f32_s;
typedef struct mm_model_weights_f32_s mm_model_weights_f32_t;
void mm_set_weights_f32(const mm_model_weights_f32_t *w);
#endif
struct mm_model_weights_q_s;
typedef struct mm_model_weights_q_s mm_model_weights_q_t;
void mm_set_weights_q(const mm_model_weights_q_t *w);

/* Step returns:
 *   1 if a new frame was processed (output written to features/logits)
 *   0 if more audio samples are needed
 */
#ifdef MM_ENABLE_F32
int mm_step_f32(mm_context_t *ctx, int16_t *features_out, float *logits_out);
#endif
int mm_step_q(mm_context_t *ctx, int16_t *features_out, int32_t *logits_out);

/* Standalone Front-end function (used by ctypes frontend.py and unit tests) */
void mm_frontend_init(void);
void mm_frontend_process_frame(const int16_t *pcm_512, int16_t *mel_40_out);

/* Mathematical helper functions */
int32_t mm_log2_q8(uint32_t x);
/* mm_rsqrt_q12: returns floor(2^32 / isqrt(x << 16)) ~= 2^24 / sqrt(x),
 * so that (v * result) >> 12 equals v / sqrt(x) in Q12;
 * for x == 0 it returns 2^24. */
int32_t mm_rsqrt_q12(uint32_t x);
int16_t mm_silu_q12(int16_t x);

/* Continuous streaming decision rule: returns class (0..9) or -1 */
int mm_decide(mm_context_t *ctx, const int32_t *logits_q12);
int mm_decide_ex(mm_context_t *ctx, const int32_t *logits_q12, int32_t threshold_q12, int32_t ema_alpha_q15, int16_t refractory_tokens);

const char *mm_weights_id(void);
/* Hash of the exported fixed-point model (Q formats, weights, LUTs): differs between exports of one checkpoint */
const char *mm_export_id(void);

/* sizeof of the weight structs, so a host binding can check its mirror of the layout */
size_t mm_weights_q_size(void);
#ifdef MM_ENABLE_F32
size_t mm_weights_f32_size(void);
#endif

/* Saturation tracking */
typedef struct {
    uint64_t stem_total;
    uint64_t stem_sat;
    uint64_t in_proj_total;
    uint64_t in_proj_sat;
    uint64_t conv_total;
    uint64_t conv_sat;
    uint64_t x_proj_total;
    uint64_t x_proj_sat;
    uint64_t resid_total;
    uint64_t resid_sat;
    uint64_t h_total;
    uint64_t h_sat;
    /* Appended: clamps that are not requantisation outputs (norm out, SSM y, gate, p clamp, logits) */
    uint64_t norm_out_total;   /* both block RMSNorms and the final RMSNorm, int16 output */
    uint64_t norm_out_sat;
    uint64_t y_total;          /* SSM output y, int32 */
    uint64_t y_sat;
    uint64_t y_gate_total;     /* gated output y * SiLU(z), int32 */
    uint64_t y_gate_sat;
    uint64_t gate_z_total;     /* gate z outside the SiLU LUT domain [-8, 8] (clamped to SiLU(+-8)) */
    uint64_t gate_z_sat;
    uint64_t p_clamp_total;    /* dt_proj output p before the +-12 clamp */
    uint64_t p_clamp_sat;
    uint64_t logits_total;     /* head output, int32 */
    uint64_t logits_sat;
} mm_sat_stats_t;

void mm_get_sat_stats(mm_sat_stats_t *out);
void mm_reset_sat_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* MU_MAMBA_H */
