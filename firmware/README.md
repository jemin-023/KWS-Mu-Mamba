# firmware/

ESP-IDF firmware (built with PlatformIO) for the ESP32-C3 (160 MHz RV32IMC, no FPU). The engine from `../engine` is
pulled in as a component.

| Environment | Purpose |
| :--- | :--- |
| `esp32-c3-devkitm-1` | fixed-point engine only: serial parity/benchmark firmware (this is what the paper's on-device numbers use) |
| `esp32-c3-f32bench` | adds the float reference block (`-DMM_ENABLE_F32`) for the software-float comparison |
| `esp32-c3-ablation` | float build plus the C1-vs-`expf` ZOH micro-benchmark (`-DMM_BENCH_ZOH_ABLATION`, `main/bench_ablation.c`) |

```bash
pio run -d firmware -e esp32-c3-devkitm-1 -t upload     # board expected on /dev/ttyACM0
python tools/serial_eval.py --bench --subset 200 --seed 0
```

`main/audio_serial.c` receives audio tokens over USB-serial and returns logits; `main/bench.c` reports per-token cycle
counts. Build output (`.pio/`, `sdkconfig.*`) is git-ignored.
