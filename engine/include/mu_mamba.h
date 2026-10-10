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
#define MM_NUM_CLASSES    12

/* Model Architecture Configuration */
#if defined(__has_include)
#if __has_include("mm_config.h")
#include "mm_config.h"
#elif __has_include("gen/mm_config.h")
#include "gen/mm_config.h"
#endif
#endif

/* Fallback Defaults if not defined by mm_config.h */
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
    float   conv_ring_f[MM_NUM_LAYERS][MM_D_CONV - 1][MM_D_INNER];
    int16_t conv_ring_q[MM_NUM_LAYERS][MM_D_CONV - 1][MM_D_INNER];
    uint8_t conv_ring_idx[MM_NUM_LAYERS];

    /* State H: D_inner x D_state per layer */
    float   ssm_h_f[MM_NUM_LAYERS][MM_D_INNER][MM_D_STATE];
    int32_t ssm_h_q[MM_NUM_LAYERS][MM_D_INNER][MM_D_STATE];

    /* Output logits */
    int32_t logits_q[MM_NUM_CLASSES];
    float   logits_f[MM_NUM_CLASSES];

    /* Continuous streaming posterior smoothing */
    int32_t smoothed_probs[MM_NUM_CLASSES];
    int16_t refractory_timer;
} mm_context_t;

/* Public API */
size_t mm_context_size(void);
void mm_init(mm_context_t *ctx);
void mm_reset(mm_context_t *ctx);
size_t mm_push_audio(mm_context_t *ctx, const int16_t *samples, size_t n_samples);

/* Model Weights API */
struct mm_model_weights_f32_s;
typedef struct mm_model_weights_f32_s mm_model_weights_f32_t;
struct mm_model_weights_q_s;
typedef struct mm_model_weights_q_s mm_model_weights_q_t;

void mm_set_weights_f32(const mm_model_weights_f32_t *w);
void mm_set_weights_q(const mm_model_weights_q_t *w);

/* Step returns:
 *   1 if a new frame was processed (output written to features/logits)
 *   0 if more audio samples are needed
 */
int mm_step_f32(mm_context_t *ctx, int16_t *features_out, float *logits_out);
int mm_step_q(mm_context_t *ctx, int16_t *features_out, int32_t *logits_out);

/* Standalone Front-end function (used by ctypes frontend.py and unit tests) */
void mm_frontend_init(void);
void mm_frontend_process_frame(const int16_t *pcm_512, int16_t *mel_40_out);

/* Mathematical helper functions */
int32_t mm_log2_q8(uint32_t x);
int32_t mm_rsqrt_q12(uint32_t x);
int32_t mm_rsqrt_q15(uint32_t x);
int16_t mm_silu_q15(int16_t x);

#ifdef __cplusplus
}
#endif

#endif /* MU_MAMBA_H */
