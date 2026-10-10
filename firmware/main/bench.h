#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t cycles_min;
    uint32_t cycles_median;
    uint32_t cycles_frontend;
    uint32_t free_heap_bytes;
    uint32_t min_free_heap_bytes;
    uint32_t stack_high_water_bytes;
#ifdef BENCH_FLOAT_BLOCK
    uint32_t cycles_block_f32;
#endif
} mm_bench_stats_t;

void mm_bench_run(mm_bench_stats_t *out_stats);
void mm_bench_print(const mm_bench_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_H */
