#include <stdio.h>
#include <string.h>
#include "mu_mamba.h"
#include "bench.h"
#include "audio_source.h"
#include "mm_block_q.h"

extern const mm_model_weights_q_t g_default_model_weights_q;

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
static const char *TAG = "KWS_MAMBA";
#endif

extern int audio_serial_process_frame(mm_context_t *ctx);

static mm_context_t g_ctx;

void app_main(void) {
    printf("\n========================================\n");
    printf("   Mu-Mamba KWS Engine (ESP32-C3)       \n");
    printf("   Model %s export %s (D=%d L=%d N=%d C=%d)   \n",
           mm_weights_id(), mm_export_id(), MM_DIM, MM_NUM_LAYERS, MM_D_STATE, MM_NUM_CLASSES);
    printf("========================================\n\n");

    /* Initialize engine context, front-end, and load flash weights */
    mm_init(&g_ctx);
    mm_set_weights_q(&g_default_model_weights_q);

    /* Run initial power-on benchmark */
    mm_bench_stats_t bench_stats;
    mm_bench_run(&bench_stats);
    mm_bench_print(&bench_stats);
    mm_bench_softfloat_print();
#ifdef MM_BENCH_ZOH_ABLATION
    mm_bench_ablation_print();
#endif

    /* Initialize audio source */
    if (audio_source_serial.init) {
        audio_source_serial.init();
    }

    printf("Waiting for serial audio frames...\n");

    while (1) {
        audio_serial_process_frame(&g_ctx);
#ifdef ESP_PLATFORM
        vTaskDelay(pdMS_TO_TICKS(1));
#endif
    }
}
