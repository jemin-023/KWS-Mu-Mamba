CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -fPIC -Iinclude -Isrc -Igen -std=c99 -DMM_DEBUG_SAT
LDFLAGS ?= -shared -lm

SRCS = src/mm_math.c src/mm_frontend.c src/mm_block_f32.c src/mm_block_q.c src/mm_model.c gen/mm_luts.c gen/mm_weights.c gen/mm_weights_q.c
OBJS = $(SRCS:.c=.o)
TARGET = libmumamba.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	$(CC) -O3 -Iinclude -Isrc tests/test_frontend.c -L. -lmumamba -Wl,-rpath,. -lm -o tests/test_frontend
	./tests/test_frontend

clean:
	rm -f $(OBJS) $(TARGET) tests/test_frontend tests/*.o

.PHONY: all test clean
