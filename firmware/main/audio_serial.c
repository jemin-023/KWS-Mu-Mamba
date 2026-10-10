#include <stdio.h>
#include <string.h>
#include "audio_source.h"
#include "mu_mamba.h"
#include "bench.h"
#include "mm_block_q.h"


#define SERIAL_SYNC_0 0xAA
#define SERIAL_SYNC_1 0x55
#define RESP_SYNC_0   0x55
#define RESP_SYNC_1   0xAA

#define CMD_AUDIO_CHUNK 0x01
#define CMD_MEL_TOKEN   0x02
#define CMD_RESET       0x03
#define CMD_BENCH       0x04

static uint16_t calc_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = crc << 1;
            }
        }
    }
    return crc;
}

#ifdef ESP_PLATFORM
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"
#include "esp_timer.h"
#define UART_NUM UART_NUM_0

static int s_active_interface = 0; // 0: detect, 1: USB_SERIAL_JTAG, 2: UART0
#endif

static int serial_init(void) {
#ifdef ESP_PLATFORM
    usb_serial_jtag_driver_config_t usb_config = {
        .tx_buffer_size = 4096,
        .rx_buffer_size = 4096,
    };
    usb_serial_jtag_driver_install(&usb_config);

    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_param_config(UART_NUM, &uart_config);
    uart_driver_install(UART_NUM, 4096, 4096, 0, NULL, 0);
#endif
    return 0;
}

static int serial_read_exact(uint8_t *buf, size_t len) {
#ifdef ESP_PLATFORM
    size_t read_bytes = 0;
    while (read_bytes < len) {
        int r = 0;
        if (s_active_interface == 1) {
            r = usb_serial_jtag_read_bytes(buf + read_bytes, len - read_bytes, pdMS_TO_TICKS(500));
        } else if (s_active_interface == 2) {
            r = uart_read_bytes(UART_NUM, buf + read_bytes, len - read_bytes, pdMS_TO_TICKS(500));
        } else {
            r = usb_serial_jtag_read_bytes(buf + read_bytes, len - read_bytes, pdMS_TO_TICKS(10));
            if (r > 0) {
                s_active_interface = 1;
            } else {
                r = uart_read_bytes(UART_NUM, buf + read_bytes, len - read_bytes, pdMS_TO_TICKS(10));
                if (r > 0) {
                    s_active_interface = 2;
                }
            }
        }
        if (r <= 0) return -1;
        read_bytes += r;
    }
    return 0;
#else
    (void)buf; (void)len;
    return 0;
#endif
}

static void serial_write_bytes(const uint8_t *buf, size_t len) {
#ifdef ESP_PLATFORM
    if (s_active_interface == 1) {
        usb_serial_jtag_write_bytes(buf, len, pdMS_TO_TICKS(500));
    } else if (s_active_interface == 2) {
        uart_write_bytes(UART_NUM, (const char *)buf, len);
    } else {
        usb_serial_jtag_write_bytes(buf, len, pdMS_TO_TICKS(50));
        uart_write_bytes(UART_NUM, (const char *)buf, len);
    }
#else
    (void)buf; (void)len;
#endif
}

int audio_serial_process_frame(mm_context_t *ctx) {
    /* Resync loop state machine */
    int sync_state = 0;
    while (sync_state < 2) {
        uint8_t b = 0;
        if (serial_read_exact(&b, 1) != 0) return 0;
        if (sync_state == 0) {
            if (b == SERIAL_SYNC_0) sync_state = 1;
        } else if (sync_state == 1) {
            if (b == SERIAL_SYNC_1) {
                sync_state = 2;
            } else if (b == SERIAL_SYNC_0) {
                sync_state = 1;
            } else {
                sync_state = 0;
            }
        }
    }

    uint8_t header_tail[2];
    if (serial_read_exact(header_tail, 2) != 0) return -1;

    uint8_t cmd = header_tail[0];
    (void)header_tail[1]; /* payload indicator */

    if (cmd == CMD_RESET) {
        mm_reset(ctx);
        uint8_t resp[] = { RESP_SYNC_0, RESP_SYNC_1, 0x00, 0x00 };
        serial_write_bytes(resp, sizeof(resp));
        return 1;
    } else if (cmd == CMD_BENCH) {
        mm_bench_stats_t stats;
        mm_bench_run(&stats);
        uint8_t resp[] = { RESP_SYNC_0, RESP_SYNC_1, 0x00, (uint8_t)sizeof(stats) };
        serial_write_bytes(resp, sizeof(resp));
        serial_write_bytes((const uint8_t *)&stats, sizeof(stats));
        return 1;
    } else if (cmd == CMD_AUDIO_CHUNK) {
        uint16_t n_samples = MM_FRAME_HOP; // 320 samples per hop
        int16_t pcm_buf[MM_FRAME_HOP];

        if (serial_read_exact((uint8_t *)pcm_buf, n_samples * sizeof(int16_t)) != 0) {
            return -1;
        }

        uint16_t recv_crc;
        if (serial_read_exact((uint8_t *)&recv_crc, sizeof(recv_crc)) != 0) {
            return -1;
        }

        uint16_t exp_crc = calc_crc16((const uint8_t *)pcm_buf, n_samples * sizeof(int16_t));
        if (recv_crc != exp_crc) {
            uint8_t resp[] = { RESP_SYNC_0, RESP_SYNC_1, 0x01, 0x00 }; // CRC Error
            serial_write_bytes(resp, sizeof(resp));
            return -2;
        }

        mm_push_audio(ctx, pcm_buf, n_samples);

        /* Execute native fixed-point inference engine */
        int32_t logits_q[MM_NUM_CLASSES] = {0};
        int stepped = mm_step_q(ctx, NULL, logits_q);

        uint8_t resp_hdr[] = { RESP_SYNC_0, RESP_SYNC_1, 0x00, (uint8_t)stepped };
        serial_write_bytes(resp_hdr, sizeof(resp_hdr));
        if (stepped) {
            serial_write_bytes((const uint8_t *)logits_q, sizeof(logits_q));
        }
        return 1;
    }
    return 0;
}

const audio_source_t audio_source_serial = {
    .name = "serial_audio",
    .init = serial_init,
    .read = NULL,
    .deinit = NULL,
};
