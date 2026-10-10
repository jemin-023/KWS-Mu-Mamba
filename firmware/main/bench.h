#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t cycles_frontend;
    uint32_t cycles_block_q[2];
    uint32_t cycles_block_f32[2];
    uint32_t cycles_head;
    uint32_t cycles_total_token;
    uint32_t free_heap_bytes;
    uint32_t min_free_heap_bytes;
    uint32_t stack_high_water_bytes;
    float    float_vs_fixed_ratio;
} mm_bench_stats_t;

void mm_bench_run(mm_bench_stats_t *out_stats);
void mm_bench_print(const mm_bench_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_H */
