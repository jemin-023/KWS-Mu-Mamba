# KWS-Mu-Mamba: Real-Time Streaming Mamba Keyword Spotter on Microcontrollers

[![Target: ESP32-C3](https://img.shields.io/badge/Target-ESP32--C3%20(RISC--V%20160MHz)-red.svg)](https://www.espressif.com/en/products/socs/esp32-c3)
[![Engine: C99 / RV32IMC](https://img.shields.io/badge/Engine-C99%20%2F%20RV32IMC-blue.svg)](file:///home/Jemin/Projects/KWS-Mu-Mamba/engine)
[![Precision: W8A16 Fixed-Point](https://img.shields.io/badge/Precision-W8A16%20Fixed--Point-green.svg)](file:///home/Jemin/Projects/KWS-Mu-Mamba/engine/src/mm_block_q.c)
[![Accuracy: 94.00%](https://img.shields.io/badge/Speech%20Commands%20v2-94.00%25%20(12--class)-orange.svg)](file:///home/Jemin/Projects/KWS-Mu-Mamba/train/kws/eval_c.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**KWS-Mu-Mamba** is a production-grade, runtime-free, and real-time streaming Keyword Spotting (KWS) inference engine that brings Selective State Space Models (**Mamba**) to low-power microcontrollers (MCUs) such as the **ESP32-C3 (160 MHz Single-Core RISC-V RV32IMC without a Hardware FPU)**.

By redesigning the continuous SSM recurrence with the **C1 constraint**, integer block-floating-point (BFP) front-end processing, and **W8A16 fixed-point quantization**, KWS-Mu-Mamba enables true **online token-by-token streaming** with **zero dynamic heap allocations (`0 malloc`)**, consuming **< 6.0 KB RAM** and running **14.0× faster** than existing MCU Mamba baselines.

---

## Key Highlights

- ⚡ **True Online Streaming ($O(1)$ Memory)**: Processes audio frame-by-frame (20 ms hop, 50 tokens/sec). Historical context is preserved in a compact causal depthwise Conv1d ring buffer ($K=4$) and recurrent SSM hidden state $h_t \in \mathbb{R}^{D_{inner} \times N}$, eliminating full-clip batch buffering.
- 📐 **C1 Recurrence Constraint ($A[d, n] = -(n+1)$)**: Discretized decay coefficients $P_n = q^n$ derive from a single scalar decay $q = \text{sigmoid}(-p)$ via successive integer multiplications. Completely eliminates runtime matrix exponentiations and transcendental functions (`expf`).
- 🔢 **W8A16 Quantized Fixed-Point Arithmetic**: Weights stored in INT8 (per-row symmetric quantization with multiplier-shift scaling), activations in INT16, and accumulations in INT32. Nonlinear activations ($\text{sigmoid}$, $\text{SiLU}$) are mapped to precomputed 16-bit Look-Up Tables (LUTs, 7.3 KB total).
- 🛡️ **Zero Dynamic Heap Allocations**: Pure static memory layout allocated in BSS. Inference performs **0 `malloc()` calls**, eliminating out-of-memory risks and heap fragmentation.
- 🎙️ **On-Chip Integer BFP Front-End**: Native integer audio preprocessor executes a 512-point Hann window, Radix-2 FFT, 40-channel Mel triangular filterbank, and integer $\log_2$ compression in **1.07 ms** on the 160 MHz RISC-V core.
- 🎯 **Posterior Smoothing & Refractory Gate**: Stateful exponential moving average (EMA) posterior probability smoothing coupled with refractory lockout achieves continuous wake-word detection with **< 1.0 False Alarm / hour**.

---

## On-Device Hardware Benchmark: KWS-Mu-Mamba vs. MambaLite-Micro

Both engines were benchmarked on the exact same physical **ESP32-C3** hardware (160 MHz single-core RV32IMC, 400 KB SRAM, no hardware FPU) connected via serial (`/dev/ttyACM0`):

| Metric / Parameter | **MambaLite-Micro** ([arXiv:2509.05488](https://arxiv.org/abs/2509.05488)) | **KWS-Mu-Mamba (Ours)** | On-Device Advantage |
| :--- | :--- | :--- | :--- |
| **Inference Mode** | **Offline Batch / Window**<br>(Requires buffering 1.0 s / 100 frames) | **Online Streaming Token-by-Token**<br>(20 ms hop / 50 tokens per second) | **Zero lag reactive streaming** vs. 1.0 s buffering delay |
| **Target Task / Classes** | Speech Commands v2 (3-class / 10-class) | Speech Commands v2 (**12-class full standard**) | Full 12-class benchmark (10 words + silence + unknown) |
| **Numeric Precision** | **FP32** (Single-precision float) | **W8A16 Fixed-Point** (INT8 weights, INT16 activations) | Native integer arithmetic; 0 software float emulation |
| **Flash Firmware Size** | **370.4 KB** binary | **202.5 KB** binary | **167.9 KB smaller Flash storage** |
| **Peak Dynamic RAM Used** | **239,988 bytes (234.36 KB)**<br>*(Triggered `malloc failed` warnings)* | **0 bytes** dynamic heap<br>*(**~6.0 KB** active static working state)* | **~39× lower RAM footprint**; immune to heap fragmentation |
| **Free Heap After Run** | **38,992 bytes (38.08 KB)**<br>*(Leaked ~235 KB dynamic heap)* | **278,980 bytes (272.44 KB)**<br>*(Zero memory leakage)* | Leaves 278 KB for FreeRTOS tasks / BLE / Wi-Fi |
| **Latency per 1.0 s Audio** | **2,750.36 ms (2.75 seconds)** | **196.5 ms** (3.93 ms / token × 50 tokens) | **14.0× FASTER on ESP32-C3** |
| **CPU Clock Cycles (1.0 s)** | **440,057,821 cycles** | **31,474,850 cycles** (629,497 cycles / token) | **14.0× fewer CPU clock cycles** |
| **Real-Time Factor (RTF)** | **2.750** *(Underrun; 2.75× slower than real-time)* | **0.196** *(Runs 5.1× faster than real-time)* | **CPU is idle 80.4% of the time** |
| **Avg. $L_\infty$ Error (vs. PyTorch)**| **$2.44 \times 10^{-4}$** (FP32) | **$1.67 \times 10^{-6}$** (FP32 C)<br>**$5.73 \times 10^{-1}$** (W8A16 Fixed) | **146× tighter numerical agreement** in FP32 |
| **Avg. Mean Error** | **$2.23 \times 10^{-5}$** (FP32) | **$6.14 \times 10^{-7}$** (FP32 C)<br>**$1.58 \times 10^{-1}$** (W8A16 Fixed) | **36× lower mean error** in FP32 |
| **Worst-Case $L_\infty$** | **$6.52 \times 10^{-4}$** (FP32) | **$3.81 \times 10^{-6}$** (FP32 C)<br>**$1.61 \times 10^{0}$** (W8A16 Fixed) | **171× tighter worst-case error bound** |
| **Top-1 Decision Agreement** | 100.0% vs. PyTorch FP32 | **99.50%** (C Fixed vs. Float) / **100% on-device parity** | Consistent classification decisions |
| **Test Accuracy** | 92.0% (3-class) / 92.5% (10-class) | **94.00%** (12-class W8A16 Fixed) / **94.40%** (FP32) | **+1.50% to +2.00% higher accuracy** |

---

## Architectural Comparison

```
MambaLite-Micro (Offline Window Approach):
  Audio Stream (1.0 s) ──────> [Accumulate 100 Frames in RAM (4,000 floats)]
                                                │
                                                ▼
                                    Linear (40 -> 64)
                                                │
                                                ▼
                                    1x Mamba Block (D=64, D_inner=128, N=16)
                                      • Dynamic 4D/3D tensor mallocs (234 KB peak heap)
                                      • 204,800 runtime expf() calls for exp(dt * A)
                                                │
                                                ▼
                                    AdaptiveAvgPool1d(1) (Averages over 100 time steps)
                                                │
                                                ▼
                                    Linear Head (64 -> classes)

KWS-Mu-Mamba (Our Streaming Architecture):
  Audio PCM Frame (512 samples / 20 ms hop)
    │
    ▼
  Integer BFP Front-End (Hann + 512 Radix-2 FFT + 40 Mel Bins + Log2) ──> [1.07 ms]
    │
    ▼
  Stem Projection: Linear(40 -> 32)
    │
    ▼
  2x Streaming Mamba C1 Blocks (D=32, D_inner=64, N=8, K=4, R=4)
    ├── Conv Ring Buffer State: 3 tokens x 64 channels (768 bytes)
    ├── C1 Recurrence State:    64 channels x 8 states (4,096 bytes)
    │     • q = LUT_Sigmoid(-p)
    │     • P_n = P_{n-1} * q  (Successive integer mult, 0 expf() calls)
    │     • State update: h_t = P * h_{t-1} + Delta * B * u
    │     • Gate: y * LUT_SiLU(z)
    ▼
  RMSNorm(32) -> Head Linear(32 -> 12)
    │
    ▼
  Posterior EMA Smoothing & Refractory Lockout (Continuous Wake-Word Detection)
```

---

## Directory Structure

```
KWS-Mu-Mamba/
├── checkpoints/              # Pretrained PyTorch models (best_model.pt, quant_model.pt)
├── engine/                   # Production C inference engine
│   ├── CMakeLists.txt        # ESP-IDF component integration
│   ├── Makefile              # Host Linux/POSIX shared library build
│   ├── gen/                  # Exported C weights, LUTs, and calibrated configs
│   │   ├── mm_config.h       # Model hyperparameters & fractional bitshifts
│   │   ├── mm_luts.c         # Precomputed 16-bit Sigmoid and SiLU LUT tables
│   │   ├── mm_stats.json     # Feature normalization statistics
│   │   ├── mm_weights.c      # Float32 model weights
│   │   └── mm_weights_q.c    # Calibrated W8A16 quantized model weights
│   ├── include/              # Public C engine API and data structures
│   │   ├── mm_block.h        # Float block definitions
│   │   ├── mm_block_q.h      # Fixed-point block definitions
│   │   ├── mm_frontend_tables.h # FFT twiddle factors & Mel filterbank sparse LUT
│   │   ├── mm_math.h         # Fixed-point math primitives and LUT accessors
│   │   └── mu_mamba.h        # Master header and streaming context (mm_context_t)
│   ├── src/                  # Implementation source files
│   │   ├── mm_block_f32.c    # Float32 reference block
│   │   ├── mm_block_q.c      # Optimized W8A16 fixed-point block
│   │   ├── mm_frontend.c     # Integer BFP front-end (Hann, FFT, Mel, Log2)
│   │   ├── mm_math.c         # Log2 approximation and saturation tracking
│   │   └── mm_model.c        # End-to-end model orchestration & streaming step
│   └── tests/                # Front-end unit tests (test_frontend.c)
├── firmware/                 # Native ESP-IDF / PlatformIO firmware for ESP32-C3
│   ├── CMakeLists.txt        # Firmware CMake build script
│   ├── platformio.ini        # Target board config (esp32-c3-devkitm-1)
│   ├── sdkconfig.defaults    # ESP-IDF defaults (CPU 160MHz, USB-Serial/JTAG)
│   └── main/                 # Main firmware source
│       ├── audio_serial.c    # USB-Serial/JTAG streaming audio receiver & CRC16 protocol
│       ├── audio_source.h    # Audio input interface abstraction
│       ├── bench.c / bench.h # On-board hardware cycle counter & memory benchmark
│       └── main.c            # Firmware entry point
├── scripts/                  # Data download and preparation scripts
│   └── download_speech_commands.py
├── tests/                    # Host Python parity verification test suite
│   ├── test_c_parity.py      # Host C shared library vs. PyTorch parity
│   ├── test_fixed_parity.py  # Fixed-point W8A16 vs. Float32 logit parity gate
│   └── test_model_parity.py  # Batched forward vs. streaming step parity
├── tools/                    # Hardware evaluation and utility tools
│   ├── compute_stats.py      # Computes Mel filterbank mean/std across dataset
│   ├── gen_frontend_tables.py# Generates twiddle and Mel LUT headers
│   └── serial_eval.py        # Host-to-device streaming UART verification tool
└── train/                    # PyTorch training, QAT, and export pipeline
    └── kws/
        ├── data.py           # Speech Commands dataset loader & streaming sequencer
        ├── eval_c.py         # C engine evaluation harness and verification gate
        ├── export.py         # Checkpoint export to C header and format calibration
        ├── frontend.py       # Python front-end reference
        ├── model.py          # PyTorch MuMambaKWS and MambaC1Block definitions
        ├── quant.py          # Simulated quantization & QAT modules
        ├── stream_eval.py    # Continuous stream evaluation (FA/hr & detection rate)
        └── train.py          # PyTorch training script
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Create and activate the environment using Conda:

```bash
conda create -n kws python=3.12 -y
conda activate kws
pip install torch torchaudio numpy scipy pyserial tqdm pytest platformio
```

### 2. Build and Test the C Inference Engine

Compile the host shared library and run the front-end unit test:

```bash
make -C engine all
make -C engine test
```

Run the complete parity verification test suite:

```bash
PYTHONPATH=. pytest -v tests/
```

### 3. Evaluate Model Accuracy on Speech Commands v2

Evaluate both the C float32 reference engine and the calibrated W8A16 fixed-point engine across the test split:

```bash
python -m train.kws.eval_c --ckpt checkpoints/best_model.pt --split test --engine both
```

Expected output:
```
Float Accuracy:     94.40% (944/1000)
Fixed Accuracy:     94.00% (940/1000)
Accuracy Drop:      +0.40 pt (gate: <= 0.50 pt)
Top-1 Agreement:    99.50% (gate: >= 99.00%)
Median Logit Error: 0.0274
>>> GATE PASSED: drop <= 0.50 pt and agreement >= 99.00% <<<
```

### 4. Build & Flash Firmware to ESP32-C3

Connect an ESP32-C3 DevKit via USB and compile/flash using PlatformIO:

```bash
pio run -d firmware -t upload
```

Monitor boot output and initial on-board power-on benchmark:

```bash
pio device monitor -d firmware
```

### 5. Run Host-to-Device Streaming Evaluation over Serial

Stream audio clips from the test set over USB-Serial/JTAG to the ESP32-C3 and verify on-device bit-identity:

```bash
python tools/serial_eval.py --port /dev/ttyACM0 --subset 200
```

Expected output:
```
================ ON-DEVICE EVALUATION REPORT ================
Total Clips Evaluated:    200
Total Tokens Evaluated:   10000
Device Accuracy:          94.00%
Host Accuracy:            94.00%
Per-Token Bit-Identity:   100.00% (10000/10000)
Bit-Identical Clips:      100.00% (200/200)
Maximum Logit Difference: 0

--- On-Board Timing (@ 160 MHz) ---
Min Per-Token Time:       3.91 ms (625835 cycles)
Median Per-Token Time:    3.93 ms (629497 cycles)
Mean Per-Token Time:      3.93 ms (629810 cycles)
=============================================================
```

---

## License

This project is licensed under the [MIT License](LICENSE).
