#ifndef MM_BLOCK_H
#define MM_BLOCK_H

#include "mu_mamba.h"

/* Floating-point weights structure for one block */
typedef struct {
    float norm_w[MM_DIM];                                      /* 32 */
    float in_proj_w[2 * MM_D_INNER][MM_DIM];                  /* 128 x 32 */
    float conv1d_w[MM_D_INNER][MM_D_CONV];                    /* 64 x 4 */
    float conv1d_b[MM_D_INNER];                               /* 64 */
    float x_proj_w[MM_DT_RANK + 2 * MM_D_STATE][MM_D_INNER];  /* 20 x 64 */
    float dt_proj_w[MM_D_INNER][MM_DT_RANK];                  /* 64 x 4 */
    float dt_proj_b[MM_D_INNER];                              /* 64 */
    float D_skip[MM_D_INNER];                                 /* 64 */
    float out_proj_w[MM_DIM][MM_D_INNER];                     /* 32 x 64 */
} mm_block_weights_f32_t;

/* Floating-point model weights */
struct mm_model_weights_f32_s {
    float stem_w[MM_DIM][MM_MEL_BINS];                        /* 32 x 40 */
    float stem_b[MM_DIM];                                     /* 32 */
    mm_block_weights_f32_t blocks[MM_NUM_LAYERS];             /* 2 layers */
    float final_norm_w[MM_DIM];                               /* 32 */
    float head_w[MM_NUM_CLASSES][MM_DIM];                     /* 12 x 32 */
    float head_b[MM_NUM_CLASSES];                             /* 12 */
};

/* Step function for float32 reference Mamba C1 block */
void mm_block_step_f32(
    const float *x_in,
    const mm_block_weights_f32_t *w,
    float conv_ring[MM_D_CONV - 1][MM_D_INNER],
    float h[MM_D_INNER][MM_D_STATE],
    float *x_out
);

#endif /* MM_BLOCK_H */
