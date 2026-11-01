#include <stdio.h>
#include <string.h>
#include "bench.h"
#include "mu_mamba.h"
#include "mm_block_q.h"
#ifdef MM_ENABLE_F32
#include "mm_block.h"
#endif

#ifdef ESP_PLATFORM
#include "esp_cpu.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_private/esp_clk.h"
#else
static inline uint32_t esp_cpu_get_cycle_count(void) { return 0; }
static inline uint32_t esp_clk_cpu_freq(void) { return 160000000u; }
static inline uint32_t esp_get_free_heap_size(void) { return 250000; }
static inline uint32_t esp_get_minimum_free_heap_size(void) { return 200000; }
static inline uint32_t uxTaskGetStackHighWaterMark(void *task) { (void)task; return 4096; }
#endif

extern const mm_model_weights_q_t g_default_model_weights_q;
#ifdef MM_ENABLE_F32
extern const mm_model_weights_f32_t g_default_model_weights_f32;
#endif

#define BENCH_FRONT_ROUNDS 21
#define BENCH_STEP_ROUNDS 51
#define BENCH_F32_ROUNDS 11
#define BENCH_STACK_SIZE 8192

static int16_t s_noise_pcm[MM_FRAME_LEN];
static int16_t s_mel_out[MM_MEL_BINS];
static mm_context_t s_bench_ctx;
#ifdef ESP_PLATFORM
static mm_context_t s_stack_ctx;
#endif
static uint32_t s_front_cycles[BENCH_FRONT_ROUNDS];
static uint32_t s_step_cycles[BENCH_STEP_ROUNDS];
static int32_t s_logits[MM_NUM_CLASSES];
#ifdef MM_ENABLE_F32
static uint32_t s_f32_cycles[BENCH_F32_ROUNDS];
static mm_context_t s_f32_ctx;
static float s_f32_logits[MM_NUM_CLASSES];
#endif

static void sort_u32(uint32_t *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[i]) {
                uint32_t tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }
}

#ifdef ESP_PLATFORM
static volatile uint32_t s_stack_hw = 0;
static TaskHandle_t s_creator_handle = NULL;

static void bench_stack_task(void *arg) {
    (void)arg;
    mm_init(&s_stack_ctx);
    mm_set_weights_q(&g_default_model_weights_q);
    mm_push_audio(&s_stack_ctx, s_noise_pcm, MM_FRAME_LEN);
    {
        int32_t logits[MM_NUM_CLASSES];
        for (int i = 0; i < 5; i++) {
            mm_push_audio(&s_stack_ctx, s_noise_pcm, MM_FRAME_HOP);
            mm_step_q(&s_stack_ctx, NULL, logits);
        }
    }
    s_stack_hw = (uint32_t)uxTaskGetStackHighWaterMark(NULL);
    xTaskNotifyGive(s_creator_handle);
    vTaskDelete(NULL);
}
#endif

void mm_bench_run(mm_bench_stats_t *out_stats) {
    if (!out_stats) return;
    memset(out_stats, 0, sizeof(mm_bench_stats_t));

    out_stats->version = 2;
#ifdef ESP_PLATFORM
    out_stats->cpu_freq_hz = (uint32_t)esp_clk_cpu_freq();
#else
    out_stats->cpu_freq_hz = esp_clk_cpu_freq();
#endif
    out_stats->ctx_bytes = (uint32_t)sizeof(mm_context_t);

    /* Deterministic LCG noise frame */
    {
        uint32_t lcg = 0x12345678u;
        for (int i = 0; i < MM_FRAME_LEN; i++) {
            lcg = lcg * 1664525u + 1013904223u;
            s_noise_pcm[i] = (int16_t)(lcg >> 16);
        }
    }

    /* Front-end: time mm_frontend_process_frame 21 times */
    for (int r = 0; r < BENCH_FRONT_ROUNDS; r++) {
        uint32_t c0 = esp_cpu_get_cycle_count();
        mm_frontend_process_frame(s_noise_pcm, s_mel_out);
        uint32_t c1 = esp_cpu_get_cycle_count();
        s_front_cycles[r] = c1 - c0;
    }
    sort_u32(s_front_cycles, BENCH_FRONT_ROUNDS);
    out_stats->frontend_cycles_min = s_front_cycles[0];
    out_stats->frontend_cycles_median = s_front_cycles[BENCH_FRONT_ROUNDS / 2];

    /* mm_step_q: init, weights, full frame, one warm-up, then 51 timed rounds */
    mm_init(&s_bench_ctx);
    mm_set_weights_q(&g_default_model_weights_q);
    mm_push_audio(&s_bench_ctx, s_noise_pcm, MM_FRAME_LEN);
    (void)mm_step_q(&s_bench_ctx, NULL, s_logits);

    out_stats->free_heap_before = esp_get_free_heap_size();
    out_stats->min_free_heap_before = esp_get_minimum_free_heap_size();
    {
        int all_stepped = 1;
        for (int r = 0; r < BENCH_STEP_ROUNDS; r++) {
            mm_push_audio(&s_bench_ctx, s_noise_pcm, MM_FRAME_HOP);
            uint32_t c0 = esp_cpu_get_cycle_count();
            int ret = mm_step_q(&s_bench_ctx, NULL, s_logits);
            uint32_t c1 = esp_cpu_get_cycle_count();
            if (ret != 1) {
                all_stepped = 0;
            }
            s_step_cycles[r] = c1 - c0;
        }
        out_stats->free_heap_after = esp_get_free_heap_size();
        out_stats->min_free_heap_after = esp_get_minimum_free_heap_size();
        if (!all_stepped) {
            out_stats->step_cycles_min = 0;
            out_stats->step_cycles_median = 0;
            out_stats->step_cycles_max = 0;
        } else {
            sort_u32(s_step_cycles, BENCH_STEP_ROUNDS);
            out_stats->step_cycles_min = s_step_cycles[0];
            out_stats->step_cycles_median = s_step_cycles[BENCH_STEP_ROUNDS / 2];
            out_stats->step_cycles_max = s_step_cycles[BENCH_STEP_ROUNDS - 1];
        }
    }

    out_stats->main_stack_high_water = (uint32_t)uxTaskGetStackHighWaterMark(NULL);

    /* Stack used by inference: dedicated task, measured AFTER heap reads */
#ifdef ESP_PLATFORM
    {
        s_stack_hw = 0;
        s_creator_handle = xTaskGetCurrentTaskHandle();
        TaskHandle_t h = NULL;
        if (xTaskCreate(bench_stack_task, "mm_bench_stack", BENCH_STACK_SIZE, NULL,
                        tskIDLE_PRIORITY + 1, &h) == pdPASS && h != NULL) {
            (void)ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
            out_stats->step_stack_used_bytes = BENCH_STACK_SIZE - (uint32_t)s_stack_hw;
        } else {
            out_stats->step_stack_used_bytes = 0;
        }
    }
#else
    out_stats->step_stack_used_bytes = BENCH_STACK_SIZE - uxTaskGetStackHighWaterMark(NULL);
#endif

    /* Float reference */
#ifdef MM_ENABLE_F32
    {
        mm_init(&s_f32_ctx);
        mm_set_weights_f32(&g_default_model_weights_f32);
        mm_push_audio(&s_f32_ctx, s_noise_pcm, MM_FRAME_LEN);
        (void)mm_step_f32(&s_f32_ctx, NULL, s_f32_logits);
        int all_stepped = 1;
        for (int r = 0; r < BENCH_F32_ROUNDS; r++) {
            mm_push_audio(&s_f32_ctx, s_noise_pcm, MM_FRAME_HOP);
            uint32_t c0 = esp_cpu_get_cycle_count();
            int ret = mm_step_f32(&s_f32_ctx, NULL, s_f32_logits);
            uint32_t c1 = esp_cpu_get_cycle_count();
            if (ret != 1) {
                all_stepped = 0;
            }
            s_f32_cycles[r] = c1 - c0;
        }
        if (!all_stepped) {
            out_stats->step_f32_cycles_median = 0;
        } else {
            sort_u32(s_f32_cycles, BENCH_F32_ROUNDS);
            out_stats->step_f32_cycles_median = s_f32_cycles[BENCH_F32_ROUNDS / 2];
        }
        out_stats->step_f32_rounds = BENCH_F32_ROUNDS;
        mm_set_weights_q(&g_default_model_weights_q);
    }
#else
    out_stats->step_f32_cycles_median = 0;
    out_stats->step_f32_rounds = 0;
#endif
}

void mm_bench_print(const mm_bench_stats_t *stats) {
    if (!stats) return;
    double mhz = stats->cpu_freq_hz / 1000000.0;
    printf("==== MU-MAMBA BENCH BEGIN ====\n");
    printf("BENCH_VERSION: %lu\n", (unsigned long)stats->version);
    printf("CPU_FREQ_HZ: %lu\n", (unsigned long)stats->cpu_freq_hz);
    printf("CTX_BYTES: %lu\n", (unsigned long)stats->ctx_bytes);
    if (stats->cpu_freq_hz != 0) {
        printf("FRONTEND_CYCLES_MIN: %lu (%.3f ms)\n",
               (unsigned long)stats->frontend_cycles_min,
               (double)stats->frontend_cycles_min / 1000.0 / mhz);
        printf("FRONTEND_CYCLES_MEDIAN: %lu (%.3f ms)\n",
               (unsigned long)stats->frontend_cycles_median,
               (double)stats->frontend_cycles_median / 1000.0 / mhz);
        printf("STEP_CYCLES_MIN: %lu (%.3f ms)\n",
               (unsigned long)stats->step_cycles_min,
               (double)stats->step_cycles_min / 1000.0 / mhz);
        printf("STEP_CYCLES_MEDIAN: %lu (%.3f ms)\n",
               (unsigned long)stats->step_cycles_median,
               (double)stats->step_cycles_median / 1000.0 / mhz);
        printf("STEP_CYCLES_MAX: %lu (%.3f ms)\n",
               (unsigned long)stats->step_cycles_max,
               (double)stats->step_cycles_max / 1000.0 / mhz);
    } else {
        printf("FRONTEND_CYCLES_MIN: %lu\n", (unsigned long)stats->frontend_cycles_min);
        printf("FRONTEND_CYCLES_MEDIAN: %lu\n", (unsigned long)stats->frontend_cycles_median);
        printf("STEP_CYCLES_MIN: %lu\n", (unsigned long)stats->step_cycles_min);
        printf("STEP_CYCLES_MEDIAN: %lu\n", (unsigned long)stats->step_cycles_median);
        printf("STEP_CYCLES_MAX: %lu\n", (unsigned long)stats->step_cycles_max);
    }
    printf("STEP_STACK_USED_BYTES: %lu\n", (unsigned long)stats->step_stack_used_bytes);
    printf("FREE_HEAP_BEFORE: %lu\n", (unsigned long)stats->free_heap_before);
    printf("FREE_HEAP_AFTER: %lu\n", (unsigned long)stats->free_heap_after);
    printf("MIN_FREE_HEAP_BEFORE: %lu\n", (unsigned long)stats->min_free_heap_before);
    printf("MIN_FREE_HEAP_AFTER: %lu\n", (unsigned long)stats->min_free_heap_after);
    printf("MAIN_STACK_HIGH_WATER: %lu\n", (unsigned long)stats->main_stack_high_water);
    if (stats->cpu_freq_hz != 0) {
        printf("STEP_F32_CYCLES_MEDIAN: %lu (%.3f ms)\n",
               (unsigned long)stats->step_f32_cycles_median,
               (double)stats->step_f32_cycles_median / 1000.0 / mhz);
    } else {
        printf("STEP_F32_CYCLES_MEDIAN: %lu\n", (unsigned long)stats->step_f32_cycles_median);
    }
    printf("STEP_F32_ROUNDS: %lu\n", (unsigned long)stats->step_f32_rounds);
    printf("==== MU-MAMBA BENCH END ====\n");
}

#if defined(ESP_PLATFORM) && defined(MM_ENABLE_F32)
#include <math.h>

#define MM_SOFTFLOAT_EXPF_N 1024
#define MM_SOFTFLOAT_MAC_N 4096
#define MM_SOFTFLOAT_REPEATS 5

__attribute__((noinline)) static float mm_softfloat_expf_body(const float *x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += expf(x[i]);
    }
    return acc;
}

__attribute__((noinline)) static float mm_softfloat_expf_ref_body(const float *x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += x[i];
    }
    return acc;
}

__attribute__((noinline)) static float mm_softfloat_fmac_body(const float *a, const float *b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

__attribute__((noinline)) static int32_t mm_softfloat_imac_body(const int8_t *w, const int16_t *x, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) {
        acc += (int32_t)w[i] * (int32_t)x[i];
    }
    return acc;
}
#endif

void mm_bench_softfloat_print(void) {
#if defined(ESP_PLATFORM) && defined(MM_ENABLE_F32)
    static float s_expf_x[MM_SOFTFLOAT_EXPF_N];
    static float s_fmac_a[MM_SOFTFLOAT_MAC_N];
    static float s_fmac_b[MM_SOFTFLOAT_MAC_N];
    static int8_t s_imac_w[MM_SOFTFLOAT_MAC_N];
    static int16_t s_imac_a[MM_SOFTFLOAT_MAC_N];
    static volatile float s_sink_f;
    static volatile int32_t s_sink_i;

    for (int i = 0; i < MM_SOFTFLOAT_EXPF_N; i++) {
        s_expf_x[i] = -8.0f + (8.0f * (float)i) / 1023.0f;
    }
    {
        uint32_t lcg = 0x12345678u;
        for (int i = 0; i < MM_SOFTFLOAT_MAC_N; i++) {
            lcg = lcg * 1664525u + 1013904223u;
            s_fmac_a[i] = ((float)((lcg >> 16) & 0xFFFFu) / 32768.0f) - 1.0f;
            lcg = lcg * 1664525u + 1013904223u;
            s_fmac_b[i] = ((float)((lcg >> 16) & 0xFFFFu) / 32768.0f) - 1.0f;
            lcg = lcg * 1664525u + 1013904223u;
            s_imac_w[i] = (int8_t)(lcg >> 16);
            lcg = lcg * 1664525u + 1013904223u;
            s_imac_a[i] = (int16_t)(lcg >> 16);
        }
    }

    {
        uint32_t expf_best = 0xFFFFFFFFu;
        uint32_t ref_best = 0xFFFFFFFFu;
        uint32_t fmac_best = 0xFFFFFFFFu;
        uint32_t imac_best = 0xFFFFFFFFu;

        for (int r = 0; r < MM_SOFTFLOAT_REPEATS; r++) {
            uint32_t c0 = esp_cpu_get_cycle_count();
            float res = mm_softfloat_expf_body(s_expf_x, MM_SOFTFLOAT_EXPF_N);
            uint32_t c1 = esp_cpu_get_cycle_count();
            s_sink_f = res;
            if ((c1 - c0) < expf_best) {
                expf_best = c1 - c0;
            }
        }
        for (int r = 0; r < MM_SOFTFLOAT_REPEATS; r++) {
            uint32_t c0 = esp_cpu_get_cycle_count();
            float res = mm_softfloat_expf_ref_body(s_expf_x, MM_SOFTFLOAT_EXPF_N);
            uint32_t c1 = esp_cpu_get_cycle_count();
            s_sink_f = res;
            if ((c1 - c0) < ref_best) {
                ref_best = c1 - c0;
            }
        }
        for (int r = 0; r < MM_SOFTFLOAT_REPEATS; r++) {
            uint32_t c0 = esp_cpu_get_cycle_count();
            float res = mm_softfloat_fmac_body(s_fmac_a, s_fmac_b, MM_SOFTFLOAT_MAC_N);
            uint32_t c1 = esp_cpu_get_cycle_count();
            s_sink_f = res;
            if ((c1 - c0) < fmac_best) {
                fmac_best = c1 - c0;
            }
        }
        for (int r = 0; r < MM_SOFTFLOAT_REPEATS; r++) {
            uint32_t c0 = esp_cpu_get_cycle_count();
            int32_t res = mm_softfloat_imac_body(s_imac_w, s_imac_a, MM_SOFTFLOAT_MAC_N);
            uint32_t c1 = esp_cpu_get_cycle_count();
            s_sink_i = res;
            if ((c1 - c0) < imac_best) {
                imac_best = c1 - c0;
            }
        }

        {
            uint32_t expf_per_call = 0;
            uint32_t fmac_per_mac = fmac_best / (uint32_t)MM_SOFTFLOAT_MAC_N;
            uint32_t imac_per_mac_x100 =
                (uint32_t)(((uint64_t)imac_best * 100u) / (uint64_t)MM_SOFTFLOAT_MAC_N);
            if (expf_best > ref_best) {
                expf_per_call = (expf_best - ref_best) / (uint32_t)MM_SOFTFLOAT_EXPF_N;
            }
            printf("==== SOFTFLOAT MICRO BEGIN ====\n");
            printf("EXPF_CALLS: %lu\n", (unsigned long)MM_SOFTFLOAT_EXPF_N);
            printf("EXPF_CYCLES_TOTAL: %lu\n", (unsigned long)expf_best);
            printf("EXPF_LOOP_OVERHEAD_CYCLES: %lu\n", (unsigned long)ref_best);
            printf("EXPF_CYCLES_PER_CALL: %lu\n", (unsigned long)expf_per_call);
            printf("FMAC_COUNT: %lu\n", (unsigned long)MM_SOFTFLOAT_MAC_N);
            printf("FMAC_CYCLES_TOTAL: %lu\n", (unsigned long)fmac_best);
            printf("FMAC_CYCLES_PER_MAC: %lu\n", (unsigned long)fmac_per_mac);
            printf("IMAC_COUNT: %lu\n", (unsigned long)MM_SOFTFLOAT_MAC_N);
            printf("IMAC_CYCLES_TOTAL: %lu\n", (unsigned long)imac_best);
            printf("IMAC_CYCLES_PER_MAC_X100: %lu\n", (unsigned long)imac_per_mac_x100);
            printf("==== SOFTFLOAT MICRO END ====\n");
        }
    }
#else
    (void)0;
#endif
}
