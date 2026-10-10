#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "mu_mamba.h"
#include "mm_math.h"
#include "mm_frontend_tables.h"

static void test_log2(void) {
    printf("Testing mm_log2_q8...\n");
    for (uint32_t val = 1; val < 10000000; val += (val < 100 ? 1 : val / 10)) {
        int32_t q8_res = mm_log2_q8(val);
        double f_res = 256.0 * log2((double)val);
        double diff = fabs((double)q8_res - f_res);
        /* Q8 tolerance: within 2 Q8 units (less than 0.01 in log2) */
        if (diff > 2.0) {
            printf("log2 mismatch for %u: q8=%d, double=%.2f, diff=%.2f\n", val, q8_res, f_res, diff);
            exit(1);
        }
    }
    printf("mm_log2_q8 PASSED!\n");
}

static void test_frontend_silence(void) {
    printf("Testing mm_frontend on silence...\n");
    int16_t pcm[MM_FRAME_LEN] = {0};
    int16_t mel[MM_MEL_BINS] = {0};

    mm_frontend_init();
    mm_frontend_process_frame(pcm, mel);

    for (int i = 0; i < MM_MEL_BINS; i++) {
        /* On pure silence, energy is 0, smooth log2(energy + 1) outputs 0 */
        assert(mel[i] == 0);
    }
    printf("Silence test PASSED (mel=0 for all bins)!\n");
}

static void test_frontend_sine(void) {
    printf("Testing mm_frontend on 1000 Hz sine wave...\n");
    int16_t pcm[MM_FRAME_LEN];
    int16_t mel[MM_MEL_BINS];

    /* 1 kHz sine at 16 kHz sample rate: 1000 / 16000 = 1/16 cycle per sample */
    for (int n = 0; n < MM_FRAME_LEN; n++) {
        pcm[n] = (int16_t)(16000.0 * sin(2.0 * M_PI * 1000.0 * n / 16000.0));
    }

    mm_frontend_process_frame(pcm, mel);

    /* 1 kHz corresponds to ~1000 Hz mel bin (~bin 14 or 15) */
    int max_bin = 0;
    int16_t max_val = 0;
    for (int i = 0; i < MM_MEL_BINS; i++) {
        if (mel[i] > max_val) {
            max_val = mel[i];
            max_bin = i;
        }
    }
    printf("Max energy at 1000 Hz is bin %d with log2 value %d (%.2f log2 units)\n",
           max_bin, max_val, (double)max_val / 256.0);
    assert(max_bin >= 12 && max_bin <= 16);
    assert(max_val > 4000); /* High energy detected */
    printf("Sine 1kHz test PASSED!\n");
}

/* Reference float log-mel implementation for 1 frame */
static void compute_float_reference(const int16_t *pcm_512, float *ref_mel_40) {
    float win_pcm[MM_FRAME_LEN];
    for (int n = 0; n < MM_FRAME_LEN; n++) {
        float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)n / (float)MM_FRAME_LEN));
        win_pcm[n] = (float)pcm_512[n] * roundf(w * 32767.0f) / 32768.0f;
    }

    /* 512-point float DFT scaled by 1/512 */
    float pwr[MM_FFT_SIZE / 2 + 1];
    for (int k = 0; k <= MM_FFT_SIZE / 2; k++) {
        float re = 0.0f;
        float im = 0.0f;
        for (int n = 0; n < MM_FRAME_LEN; n++) {
            float angle = -2.0f * (float)M_PI * (float)(k * n) / (float)MM_FRAME_LEN;
            re += win_pcm[n] * cosf(angle);
            im += win_pcm[n] * sinf(angle);
        }
        re /= (float)MM_FFT_SIZE;
        im /= (float)MM_FFT_SIZE;
        pwr[k] = re * re + im * im;
    }

    /* Mel accumulation and log2(energy + 1) */
    for (int m = 0; m < MM_MEL_BINS; m++) {
        uint16_t start = mm_mel_filters[m].start_bin;
        uint16_t num   = mm_mel_filters[m].num_bins;
        uint16_t off   = mm_mel_filters[m].weight_offset;
        float energy = 0.0f;
        for (uint16_t j = 0; j < num; j++) {
            energy += pwr[start + j] * ((float)mm_mel_weights_q15[off + j] / 32768.0f);
        }
        ref_mel_40[m] = 256.0f * log2f(energy + 1.0f);
    }
}

static void test_frontend_float_reference_multilevel(void) {
    printf("Testing mm_frontend vs float reference across dynamic range...\n");
    const int levels_db[] = {0, -6, -12, -20, -30};
    const int num_levels = sizeof(levels_db) / sizeof(levels_db[0]);

    for (int l = 0; l < num_levels; l++) {
        int db = levels_db[l];
        double amp_scale = pow(10.0, (double)db / 20.0);
        double peak_amp = 30000.0 * amp_scale;

        int16_t pcm[MM_FRAME_LEN];
        for (int n = 0; n < MM_FRAME_LEN; n++) {
            double t = (double)n / 16000.0;
            double sig = 0.5 * sin(2.0 * M_PI * 500.0 * t) +
                         0.3 * sin(2.0 * M_PI * 1000.0 * t) +
                         0.2 * sin(2.0 * M_PI * 2500.0 * t);
            pcm[n] = (int16_t)round(peak_amp * sig);
        }

        int16_t c_mel[MM_MEL_BINS];
        float ref_mel[MM_MEL_BINS];

        mm_frontend_process_frame(pcm, c_mel);
        compute_float_reference(pcm, ref_mel);

        /* Compare active bins */
        int active_bins = 0;
        double sum_err = 0.0;
        double max_err = 0.0;

        for (int m = 0; m < MM_MEL_BINS; m++) {
            /* Active bins are those with significant energy (e.g. > 1 log2 unit) */
            if (ref_mel[m] > 256.0f) {
                active_bins++;
                double err = fabs((double)c_mel[m] - (double)ref_mel[m]) / 256.0;
                sum_err += err;
                if (err > max_err) max_err = err;
            }
        }

        double mean_err = (active_bins > 0) ? (sum_err / active_bins) : 0.0;
        printf("Level %3d dB | Active bins: %2d | Mean error: %.4f log2 | Max error: %.4f log2\n",
               db, active_bins, mean_err, max_err);

        assert(active_bins >= 3); /* At least harmonic bins active */
        assert(mean_err < 0.20);  /* Mean error well below 0.20 log2 units */
        assert(max_err < 0.80);   /* Peak harmonic bins match within ~0.8 log2 units */
    }
    printf("Float reference multi-level test PASSED!\n");
}

int main(void) {
    printf("=== Mu-Mamba Frontend Unit Tests ===\n");
    test_log2();
    test_frontend_silence();
    test_frontend_sine();
    test_frontend_float_reference_multilevel();
    printf("All frontend tests passed successfully!\n");
    return 0;
}
