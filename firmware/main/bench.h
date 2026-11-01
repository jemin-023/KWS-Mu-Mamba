#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t version;                 /* = 2 */
    uint32_t cpu_freq_hz;             /* esp_clk_cpu_freq() */
    uint32_t ctx_bytes;               /* sizeof(mm_context_t) */
    uint32_t frontend_cycles_min;
    uint32_t frontend_cycles_median;
    uint32_t step_cycles_min;         /* mm_step_q: front-end + model */
    uint32_t step_cycles_median;
    uint32_t step_cycles_max;
    uint32_t step_stack_used_bytes;   /* peak stack used by a task that only runs mm_step_q */
    uint32_t free_heap_before;        /* esp_get_free_heap_size() just before the timed loop */
    uint32_t free_heap_after;         /* ... just after it */
    uint32_t min_free_heap_before;    /* esp_get_minimum_free_heap_size() just before the timed loop */
    uint32_t min_free_heap_after;     /* ... just after it */
    uint32_t main_stack_high_water;   /* uxTaskGetStackHighWaterMark(NULL) of the calling task */
    uint32_t step_f32_cycles_median;  /* mm_step_f32 median; 0 when built without MM_ENABLE_F32 */
    uint32_t step_f32_rounds;         /* number of float rounds measured; 0 when built without MM_ENABLE_F32 */
} mm_bench_stats_t;

_Static_assert(sizeof(mm_bench_stats_t) == 64, "mm_bench_stats_t must be 64 bytes");

void mm_bench_run(mm_bench_stats_t *out_stats);
void mm_bench_print(const mm_bench_stats_t *stats);
void mm_bench_softfloat_print(void);
/* C1 vs per-(d, n) expf decay, float build with -DMM_BENCH_ZOH_ABLATION only */
void mm_bench_ablation_print(void);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_H */
