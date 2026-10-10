#ifndef AUDIO_SOURCE_H
#define AUDIO_SOURCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct audio_source {
    const char *name;
    int (*init)(void);
    /* Reads up to n_samples into buf. Returns number of samples read, or negative on error */
    int (*read)(int16_t *buf, size_t n_samples);
    void (*deinit)(void);
} audio_source_t;

/* Serial audio source implementation */
extern const audio_source_t audio_source_serial;

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_SOURCE_H */
