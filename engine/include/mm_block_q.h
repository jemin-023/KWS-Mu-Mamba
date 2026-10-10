#ifndef MM_BLOCK_Q_H
#define MM_BLOCK_Q_H

#include <stdint.h>
#include "mu_mamba.h"

/* Quantized block weights in W8A16 format */
typedef struct {
    /* RMSNorm gain in Q15 */
    int16_t norm_w[MM_DIM];

    /* in_proj: 128 x 32 int8 weights, per-row Q15 scale multiplier + shift */
    int8_t  in_proj_w[2 * MM_D_INNER][MM_DIM];
    int16_t in_proj_scale_mult[2 * MM_D_INNER];
    uint8_t in_proj_scale_shift[2 * MM_D_INNER];

    /* conv1d: 64 x 4 int8 weights, int16 bias, per-channel scale */
    int8_t  conv1d_w[MM_D_INNER][MM_D_CONV];
    int32_t conv1d_b[MM_D_INNER];
    int16_t conv1d_scale_mult[MM_D_INNER];
    uint8_t conv1d_scale_shift[MM_D_INNER];

    /* x_proj: 20 x 64 int8 weights, per-row scale */
    int8_t  x_proj_w[MM_DT_RANK + 2 * MM_D_STATE][MM_D_INNER];
    int16_t x_proj_scale_mult[MM_DT_RANK + 2 * MM_D_STATE];
    uint8_t x_proj_scale_shift[MM_DT_RANK + 2 * MM_D_STATE];

    /* dt_proj: 64 x 4 int8 weights, int32 bias in p domain */
    int8_t  dt_proj_w[MM_D_INNER][MM_DT_RANK];
    int32_t dt_proj_b[MM_D_INNER];
    int16_t dt_proj_scale_mult[MM_D_INNER];
    uint8_t dt_proj_scale_shift[MM_D_INNER];

    /* D_skip: 64 skip weights in Q12 */
    int16_t D_skip[MM_D_INNER];

    /* out_proj: 32 x 64 int8 weights, per-row scale */
    int8_t  out_proj_w[MM_DIM][MM_D_INNER];
    int16_t out_proj_scale_mult[MM_DIM];
    uint8_t out_proj_scale_shift[MM_DIM];
} mm_block_weights_q_t;

/* Quantized model weights */
struct mm_model_weights_q_s {
    int8_t  stem_w[MM_DIM][MM_MEL_BINS];
    int32_t stem_b[MM_DIM];
    int16_t stem_scale_mult[MM_DIM];
    uint8_t stem_scale_shift[MM_DIM];

    mm_block_weights_q_t blocks[MM_NUM_LAYERS];

    int16_t final_norm_w[MM_DIM];

    int8_t  head_w[MM_NUM_CLASSES][MM_DIM];
    int32_t head_b[MM_NUM_CLASSES];
    int16_t head_scale_mult[MM_NUM_CLASSES];
    uint8_t head_scale_shift[MM_NUM_CLASSES];
};

/* Set quantized model weights */
void mm_set_weights_q(const mm_model_weights_q_t *w);

/* W8A16 single-token block execution */
void mm_block_step_q(
    const int16_t *x_in,              /* Q12 */
    const mm_block_weights_q_t *w,
    int16_t conv_ring[MM_D_CONV - 1][MM_D_INNER], /* Q12 */
    int32_t h[MM_D_INNER][MM_D_STATE],            /* int32 persistent state */
    int16_t *x_out                     /* Q12 */
);

#endif /* MM_BLOCK_Q_H */
