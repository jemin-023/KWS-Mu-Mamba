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

extern const mm_model_weights_q_t g_default_model_weights_q;

#define BENCH_ROUNDS 11

void mm_bench_run(mm_bench_stats_t *out_stats) {
    if (!out_stats) return;
    memset(out_stats, 0, sizeof(mm_bench_stats_t));

    /* Generate pseudo-random noise frame */
    static int16_t noise_pcm[MM_FRAME_LEN];
    uint32_t lcg = 0x12345678;
    for (int i = 0; i < MM_FRAME_LEN; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        noise_pcm[i] = (int16_t)(lcg >> 16);
    }

    /* 1. Benchmark Front-end (Hann window + Radix-2 FFT + Mel + log2) */
    static int16_t mel_out[MM_MEL_BINS];
    uint32_t c0 = esp_cpu_get_cycle_count();
    mm_frontend_process_frame(noise_pcm, mel_out);
    uint32_t c1 = esp_cpu_get_cycle_count();
    out_stats->cycles_frontend = c1 - c0;

    /* 2. Benchmark real mm_step_q on real weights */
    static mm_context_t bench_ctx;
    mm_init(&bench_ctx);
    mm_set_weights_q(&g_default_model_weights_q);
    mm_push_audio(&bench_ctx, noise_pcm, MM_FRAME_LEN);

    int32_t logits[MM_NUM_CLASSES];
    uint32_t step_cycles[BENCH_ROUNDS];

    /* Warm-up step */
    mm_step_q(&bench_ctx, NULL, logits);

    for (int r = 0; r < BENCH_ROUNDS; r++) {
        mm_push_audio(&bench_ctx, noise_pcm, MM_FRAME_HOP);
        c0 = esp_cpu_get_cycle_count();
        mm_step_q(&bench_ctx, NULL, logits);
        c1 = esp_cpu_get_cycle_count();
        step_cycles[r] = c1 - c0;
    }

    /* Sort cycles to find min and median */
    for (int i = 0; i < BENCH_ROUNDS - 1; i++) {
        for (int j = i + 1; j < BENCH_ROUNDS; j++) {
            if (step_cycles[j] < step_cycles[i]) {
                uint32_t tmp = step_cycles[i];
                step_cycles[i] = step_cycles[j];
                step_cycles[j] = tmp;
            }
        }
    }
    out_stats->cycles_min = step_cycles[0];
    out_stats->cycles_median = step_cycles[BENCH_ROUNDS / 2];

#ifdef BENCH_FLOAT_BLOCK
    static float dummy_xf[MM_DIM];
    static float dummy_outf[MM_DIM];
    static float conv_ring_f[MM_D_CONV - 1][MM_D_INNER];
    static float h_f[MM_D_INNER][MM_D_STATE];
    extern const mm_model_weights_f32_t g_default_model_weights_f32;
    c0 = esp_cpu_get_cycle_count();
    mm_block_step_f32(dummy_xf, &g_default_model_weights_f32.blocks[0], conv_ring_f, h_f, dummy_outf);
    c1 = esp_cpu_get_cycle_count();
    out_stats->cycles_block_f32 = c1 - c0;
#endif

    /* Memory metrics */
    out_stats->free_heap_bytes = esp_get_free_heap_size();
    out_stats->min_free_heap_bytes = esp_get_minimum_free_heap_size();
    out_stats->stack_high_water_bytes = uxTaskGetStackHighWaterMark(NULL);
}

void mm_bench_print(const mm_bench_stats_t *stats) {
    if (!stats) return;
    printf("\n================ ESP32-C3 BENCHMARK REPORT ================\n");
    printf("CPU Frequency:          160 MHz (RV32IMC)\n");
    printf("Front-end Cycles:       %lu (%.2f ms @ 160MHz)\n",
           (unsigned long)stats->cycles_frontend, stats->cycles_frontend / 160000.0f);
    printf("mm_step_q Min:          %lu cycles (%.2f ms @ 160MHz)\n",
           (unsigned long)stats->cycles_min, stats->cycles_min / 160000.0f);
    printf("mm_step_q Median:       %lu cycles (%.2f ms @ 160MHz)\n",
           (unsigned long)stats->cycles_median, stats->cycles_median / 160000.0f);
#ifdef BENCH_FLOAT_BLOCK
    printf("Float32 Block Cycles:   %lu (%.2f ms @ 160MHz)\n",
           (unsigned long)stats->cycles_block_f32, stats->cycles_block_f32 / 160000.0f);
#endif
    printf("Free Heap:              %lu bytes\n", (unsigned long)stats->free_heap_bytes);
    printf("Min Free Heap:          %lu bytes\n", (unsigned long)stats->min_free_heap_bytes);
    printf("Stack High-Water:       %lu bytes\n", (unsigned long)stats->stack_high_water_bytes);
    printf("===========================================================\n\n");
}
