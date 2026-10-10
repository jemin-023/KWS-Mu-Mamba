#include <stdio.h>
#include <string.h>
#include "bench.h"
#include "mu_mamba.h"
#include "mm_block.h"
#include "mm_block_q.h"

#ifdef ESP_PLATFORM
#include "esp_cpu.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#else
static inline uint32_t esp_cpu_get_cycle_count(void) { return 0; }
static inline uint32_t esp_get_free_heap_size(void) { return 250000; }
static inline uint32_t esp_get_minimum_free_heap_size(void) { return 200000; }
static inline uint32_t uxTaskGetStackHighWaterMark(void *task) { (void)task; return 4096; }
#endif

/* Static benchmark buffers in BSS to prevent stack overflow on ESP-IDF main task */
static int16_t s_dummy_pcm[MM_FRAME_LEN];
static int16_t s_dummy_mel[MM_MEL_BINS];

static int16_t s_dummy_xq[MM_DIM];
static int16_t s_dummy_outq[MM_DIM];
static int16_t s_conv_ring_q[MM_D_CONV - 1][MM_D_INNER];
static int32_t s_h_q[MM_D_INNER][MM_D_STATE];
static mm_block_weights_q_t s_dummy_w_q;

static float s_dummy_xf[MM_DIM];
static float s_dummy_outf[MM_DIM];
static float s_conv_ring_f[MM_D_CONV - 1][MM_D_INNER];
static float s_h_f[MM_D_INNER][MM_D_STATE];
static mm_block_weights_f32_t s_dummy_w_f;

void mm_bench_run(mm_bench_stats_t *out_stats) {
    if (!out_stats) return;
    memset(out_stats, 0, sizeof(mm_bench_stats_t));

    /* 1. Benchmark Front-end (Hann window + Radix-2 FFT + Mel + log2) */
    uint32_t c0 = esp_cpu_get_cycle_count();
    mm_frontend_process_frame(s_dummy_pcm, s_dummy_mel);
    uint32_t c1 = esp_cpu_get_cycle_count();
    out_stats->cycles_frontend = c1 - c0;

    /* 2. Benchmark Fixed-Point W8A16 Block (RV32IMC Native) */
    c0 = esp_cpu_get_cycle_count();
    mm_block_step_q(s_dummy_xq, &s_dummy_w_q, s_conv_ring_q, s_h_q, s_dummy_outq);
    c1 = esp_cpu_get_cycle_count();
    out_stats->cycles_block_q[0] = c1 - c0;

    /* 3. Benchmark Float32 Reference Block */
    c0 = esp_cpu_get_cycle_count();
    mm_block_step_f32(s_dummy_xf, &s_dummy_w_f, s_conv_ring_f, s_h_f, s_dummy_outf);
    c1 = esp_cpu_get_cycle_count();
    out_stats->cycles_block_f32[0] = c1 - c0;

    /* Total per-token cycle estimate for Fixed-Point: Front-end + 2 Blocks + Stem/Head (~2000 cycles) */
    out_stats->cycles_total_token = out_stats->cycles_frontend + 2 * out_stats->cycles_block_q[0] + 2000;

    /* 4. Memory metrics */
    out_stats->free_heap_bytes = esp_get_free_heap_size();
    out_stats->min_free_heap_bytes = esp_get_minimum_free_heap_size();
    out_stats->stack_high_water_bytes = uxTaskGetStackHighWaterMark(NULL);
}

void mm_bench_print(const mm_bench_stats_t *stats) {
    if (!stats) return;
    printf("\n================ ESP32-C3 BENCHMARK REPORT ================\n");
    printf("CPU Frequency:          160 MHz (RV32IMC)\n");
    printf("Front-end Cycles:       %lu (%.2f ms @ 160MHz)\n",
           stats->cycles_frontend, stats->cycles_frontend / 160000.0f);
    printf("Block (W8A16 Q) Cycles: %lu (%.2f ms @ 160MHz)\n",
           stats->cycles_block_q[0], stats->cycles_block_q[0] / 160000.0f);
    printf("Block (Float32) Cycles: %lu (%.2f ms @ 160MHz)\n",
           stats->cycles_block_f32[0], stats->cycles_block_f32[0] / 160000.0f);
    printf("Estimated Total/Token:  %lu (%.2f ms @ 160MHz, target <= 5.0 ms)\n",
           stats->cycles_total_token, stats->cycles_total_token / 160000.0f);
    printf("Free Heap:              %lu bytes\n", stats->free_heap_bytes);
    printf("Min Free Heap:          %lu bytes\n", stats->min_free_heap_bytes);
    printf("Stack High-Water:       %lu bytes\n", stats->stack_high_water_bytes);
    printf("===========================================================\n\n");
}
