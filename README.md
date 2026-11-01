# KWS-Mu-Mamba: Streaming Mamba Keyword Spotter for a Microcontroller Without an FPU

[![Target: ESP32-C3](https://img.shields.io/badge/Target-ESP32--C3%20(RISC--V%20160MHz)-red.svg)](https://www.espressif.com/en/products/socs/esp32-c3)
[![Engine: C99 / RV32IMC](https://img.shields.io/badge/Engine-C99%20%2F%20RV32IMC-blue.svg)](engine)
[![Fixed point: int8 weights](https://img.shields.io/badge/Fixed%20point-int8%20weights%20%2F%20int16%20activations-green.svg)](engine/src/mm_block_q.c)
[![Accuracy: 89.02%](https://img.shields.io/badge/Speech%20Commands%20v0.02-89.02%25%20(12--class%2C%204890%20test%20clips)-orange.svg)](train/kws/eval_c.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**KWS-Mu-Mamba** is a C99 inference engine for a small Mamba-style keyword spotter (18,060 parameters) that processes audio **one 20 ms token at a time**, targeting an ESP32-C3 (160 MHz single-core RV32IMC, no hardware FPU). The state-space recurrence is constrained ("C1") so that all `N` decay coefficients of a channel come from one table lookup and `N-1` multiplications, the front-end is integer, and the model runs in fixed-point arithmetic with lookup tables.

What is and is not established here (every number comes from `paper/data/`; sources are named next to each table):

- **Host measurements are complete** for the paper model: clip accuracy, fixed-vs-float agreement, saturation, streaming detection and false alarms, and a seed-replicated ablation.
- **On-device results are measured for the paper model** (export `6a9a3c92`, 2026-10-30): bit-identical to the host over 50,000 tokens, 4.14 ms per 20 ms token, 0 B heap. The third-party MambaLite-Micro baseline was also run on the board but gave no consistent latency (see the on-device section).
- **Streaming**: the paper model reaches about 80 % detection at 45 false alarms per background hour on a 2.1 h test stream (details and caveats below). An earlier claim of "< 1 false alarm per hour" was wrong and is withdrawn.

---

## The two models

| | Paper model (`checkpoints/paper_model.pt`) | Clip-trained baseline (`checkpoints/quant_model.pt`) |
| :--- | :--- | :--- |
| Training | Streaming fine-tune **warm-started from the baseline**: 25 epochs (best by validation-sequence score at epoch 9), 800 samples per class per epoch, lr 3e-4 (cosine, 1 warm-up epoch), batch 16, AdamW wd 1e-4, weight EMA 0.999, seed 42; training sequences are multi-clip streams with background between keywords | 60 epochs from scratch on short single-clip sequences, 100 samples per class per epoch, lr 1e-3 (cosine, 2 warm-up epochs), batch 64, AdamW wd 1e-4, EMA 0.999, seed 42 |
| Weights id / export id | `2f4bdc556afab73f` / `6a9a3c92` (residual Q5) | `3e3052cf5abd9675` / `17c660f7` (residual Q6) |
| Role | the model the paper reports | clip-accuracy reference; also the starting point of the paper model |

<!-- sources: checkpoint metadata; paper/data/model_selection.json -->
The paper model was chosen between two warm-started candidates (400 and 800 samples per class) by a rule that uses **validation data only**: the fixed-engine detection on a 2,000-keyword validation stream within a budget of 100 false alarms per background hour (78.1 % for the 800-sample candidate against 66.7 % for the 400-sample one, measured on the exports of the selection pipeline; the rule's tie margin is 1.0 pt) `paper/data/model_selection.json`. The winner was afterwards re-exported with a residual format chosen on validation (below); on the same validation stream that export gives 78.0 %.

---

## Results

### Accuracy (host C engines, 4,890 test clips)
<!-- sources: paper/data/eval_full_test_final.json (paper model), eval_full_test_shipped.json (baseline), eval_val_paper.json -->

| Metric | Paper model | Clip-trained baseline |
| :--- | :--- | :--- |
| FP32 C engine | 89.14 % (4359 / 4890), 95 % Wilson CI [88.24, 89.98] | 92.37 % (4517 / 4890), CI [91.59, 93.08] |
| Fixed-point engine | **89.02 %** (4353 / 4890), CI [88.11, 89.86] | **92.31 %** (4514 / 4890), CI [91.53, 93.02] |
| Drop from quantisation | 0.12 pt | 0.06 pt |
| Top-1 agreement, fixed vs float | 99.55 % | 99.67 % |
| Fixed vs float logits (real units) | avg L∞ 0.145, avg mean-abs 0.0529, worst L∞ 1.86, median 0.0291 | avg L∞ 0.132, avg mean-abs 0.0465, worst L∞ 1.61, median 0.0225 |
| FP32 C engine vs PyTorch | avg L∞ 2.43e-6, worst 1.45e-5, top-1 agreement 100 % | avg L∞ 1.73e-6, worst 9.89e-6, 100 % |
| Fixed-engine accuracy, validation split (4,443 clips) | 89.24 % (3965) | not measured |
| Acceptance gate (drop ≤ 0.50 pt and agreement ≥ 99 %) | passed | passed |

Protocol: zero initial state, 16,192 samples (50 tokens), argmax of the last-token logits. Turning the clip-trained model into the streaming model costs about 3.3 points of clip accuracy.

Per-class accuracy of the fixed engine, paper model (%): yes 90.93, no 88.89, up 86.82, down 88.92, left 87.38, right 86.36, on 91.92, off 84.33, stop 93.92, go 86.57, unknown 84.56, silence 97.55. (Baseline: yes 96.66, no 91.85, up 92.94, down 88.92, left 92.96, right 90.40, on 92.68, off 90.05, stop 97.57, go 90.80, unknown 87.25, silence 95.34.)

### Numerical precision (what is actually 8, 16 and 32 bits)
- Weights: int8, per-row symmetric scale, Q15 multiplier + shift.
- 16-bit activations: residual Q5 (baseline: Q6), RMSNorm output Q12, in_proj Q11, conv/SiLU Q12, x_proj Q10, dt pre-activation Q11.
- 32-bit: SSM state `h` (Q22), SSM output `y`, gated output, logits (Q12), all accumulators; the out_proj accumulator is 64-bit. Decay powers are Q31, Δ is Q24.
- Post-training quantisation only. No quantisation-aware training is used.
- The residual format is chosen per checkpoint: calibrated on validation clips and long validation streams (99.99th percentile, 1.25× margin), which gave Q6 for the paper model; then Q5, Q6 and Q7 were each exported and evaluated on the **validation** split, and the best fixed accuracy kept, then agreement, then lower residual saturation. Q5 won with 89.24 % against 89.22 % for Q6 and Q7 (a one-clip difference; residual saturation 0.000 %, 0.000 % and 0.004 %) `paper/data/resid_format_choice.json`. Test accuracy plays no role in this choice (an earlier version of this pipeline selected the format on test accuracy and picked Q7, 89.04 % against 89.02 %; those outputs are kept as `*_warm800.json`).

Saturation on the test set (fraction of values clamped, fixed engine, paper model) <!-- paper/data/eval_full_test_final.json -->: stem 0, in_proj 0, x_proj 0, state `h` 0, RMSNorm output 0, `y` 0, `y_gate` 0, logits 0, residual 0, conv output 0.046 % (clamped to SiLU(±8)), dt pre-activation clamp (±12) 0.0014 %, gate input outside the SiLU table domain 0.0003 %.

### Streaming detection (continuous operation)
<!-- sources: paper/data/stream_final.json (paper model), stream_warm400.json, stream_shipped_v2.json, stream_sweep_final.json -->

Setup: a 2.10 h stream of 2,000 test keyword clips separated by 1–4 s of noise, 30 % of the gaps containing an "unknown" word. The decision rule is the engine's (EMA of logit margins, then a refractory lockout). A detection counts if it falls within [start − 0.2 s, end + 0.5 s] of a keyword of the same class. **The operating point is chosen on a validation stream, never on the test stream**: the tuner picks α, refractory and a threshold from a grid, taking the highest detection among settings with at most **100 false alarms per background hour** (`--fa-budget`), then one test-stream run uses that setting. For the paper model the grid is α ∈ {0.85, 0.93, 0.96, 0.98} × refractory ∈ {50, 75, 100, 150} tokens × 40 thresholds (640 settings per engine, all stored in `stream_final.json`); the best setting is α 0.93, refractory 75, threshold 1.5, which lies inside the grid (the earlier, narrower grid had put it at the edge). "Background" is the stream time outside keyword clips; it still contains the unknown-word distractors, so detections on them count as false alarms. False alarms are split into substitutions (a wrong keyword inside a keyword's window) and pure-background detections. Rates are followed by an exact Poisson 95 % interval of the count: `value [low, high]`.

| Model (fixed engine unless noted) | Setting (α, refractory, threshold) | Detection | False alarms / background hour | of which substitutions | of which pure background |
| :--- | :--- | ---: | :--- | :--- | :--- |
| **Paper model** | 0.93, 75, 1.5 | **79.85 %** (1597 / 2000) | 47.2 [37.0, 59.3] | 38.1 [29.0, 49.2] | 9.0 [4.9, 15.2] |
| Paper model, float engine | 0.93, 100, 1.5 | 79.85 % | 45.9 [35.8, 57.9] | 38.8 [29.6, 49.9] | 7.1 [3.5, 12.7] |
| Candidate with 400 samples per class (narrower tuning grid, Q6 export) | 0.93, 75, 2.25 | 73.30 % | 41.4 [31.9, 52.8] | 34.3 [25.7, 44.8] | 7.1 [3.5, 12.7] |
| Clip-trained baseline, same budget (narrower tuning grid) | 0.7, 75, 6.25 | 10.60 % | 31.0 [22.9, 41.1] | 11.0 [6.4, 17.6] | 20.0 [13.6, 28.4] |

Per hour of the whole stream (keywords included) the paper model has 34.7 [27.2, 43.6] false alarms. There were no duplicate detections; the median detection time relative to the end of the keyword is −209 ms (p90 +49 ms), i.e. most detections fire while the word is still being spoken. The same budget leaves the clip-trained baseline at 10.6 % detection: its state does not return to "background" after a keyword, so no decision rule can rescue it (the activation scan on three long validation streams shows about 1 % of its channel-steps sitting at the dt clamp edge, i.e. near-perfect integrators; for the paper model the dt pre-activation never reaches the clamp on those streams and the 1st percentile of the per-step decay rate 1−q is 7.1e-5 instead of the clamp-edge value 6.1e-6, `paper/data/activation_ranges_final.json` and `activation_ranges_shipped.json`). The 400-sample candidate and the baseline were tuned on the previous grid (α ∈ {0.7, 0.85, 0.93} × refractory ∈ {25, 50, 75}); they were not re-tuned on the extended grid.

Detection against false alarms for the paper model on the test stream (a curve, not an operating point; α 0.93, refractory 75; fixed engine; 95 % Poisson interval of the false-alarm rate) <!-- stream_sweep_final.json -->:

| Threshold | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Detection (%) | 83.5 | 81.7 | 79.8 | 76.4 | 71.2 | 64.0 | 45.1 | 23.4 |
| False alarms / background hour | 107.9 [92.2, 125.6] | 65.9 [53.8, 80.0] | 47.2 [37.0, 59.3] | 33.0 [24.5, 43.3] | 23.3 [16.3, 32.2] | 20.0 [13.6, 28.4] | 7.8 [4.0, 13.5] | 3.2 [1.0, 7.5] |

Caveats that matter for how to read this:
- Counts are small (73 false alarms in 1.55 background hours for the operating point), hence the wide intervals; validation and test streams are single draws with one stream seed.
- A 1.5 s refractory suppresses a second keyword spoken within 1.5 s of the first.
- Most remaining false alarms are substitutions (a wrong keyword fired inside a real keyword's window), not noise.
- The 100-per-background-hour budget is a choice, not a product requirement. A real wake-word deployment typically needs far fewer false alarms per hour; the curve above shows the cost.

### Cost of freezing the decay matrix (PyTorch ablation, three seeds)
<!-- source: paper/data/ablation_c1_vs_learned_a.json (seeds 42 and 1 only: ablation_c1_vs_learned_a_seeds42_1.json; seed 42 only: ..._seed42.json) -->
Two otherwise identical models per seed, trained from scratch with the streaming-sequence recipe (60 epochs, 160 samples per class, batch 16): frozen C1 `A` against a learned per-(channel, state) `A` initialised to the C1 values (+1,024 parameters). Test clip accuracy, 4,890 clips, 95 % Wilson CI:

| Seed | Frozen C1 | Learned A | Difference |
| :--- | :--- | :--- | ---: |
| 42 | 79.47 % [78.31, 80.58] | 83.03 % [81.95, 84.05] | +3.56 pt |
| 1 | 72.80 % [71.54, 74.03] | 78.75 % [77.58, 79.88] | +5.95 pt |
| 2 | 80.65 % [79.52, 81.74] | 82.52 % [81.43, 83.55] | +1.86 pt |
| mean ± sd (n = 3) | 77.64 ± 4.23 | 81.43 ± 2.33 | +3.79 ± 2.05 |

Learned `A` was better in all three seeds, with a mean gain of 3.8 points, but the paired t-test over three seeds is not significant (t = 3.19, 2 degrees of freedom, p = 0.086), and the frozen model's seed-to-seed spread (7.9 points between its best and worst seed) is larger than the mean gap. Read this as a consistent direction with an imprecise size. On the PyTorch streaming rows in the same JSON (threshold chosen on a 500-keyword validation stream at the 100-per-hour budget, α 0.85, refractory 50), the two arms do not separate consistently. These models use the from-scratch streaming recipe, not the recipe of the paper model, so the penalty for the paper model itself is unknown. Seeds 42 and 1 were trained on the local RTX 4050 (torch 2.14.1+cu130) and seed 2 on a Tesla T4 (torch 2.11.0+cu128) in a notebook VM, so the seeds are independent runs, not exact reproductions. AdamW weight decay (1e-4) was applied to `A_log`. There is no C engine for learned `A`.

---

## How the C1 recurrence works

With `A[d, n] = -(n+1)` frozen and `Δ = softplus(p)`, `q = sigmoid(-p) = exp(-Δ)`, the discrete decay is `exp(Δ·A[d,n]) = q^(n+1)`: one lookup for `q`, then `P_n = P_{n-1}·q`. The engine never calls `expf`. The input gain is the Euler form `Δ·B`. See [`math.md`](math.md) for the derivation, the verified error bounds, the rounding behaviour of the state update (round-half-up with a symmetric dead band `|h| < 0.5/(1-P)` LSB) and the memory accounting.

```
Audio PCM (512-sample window, 320-sample / 20 ms hop)
  -> integer front-end: Hann (Q15), 512-pt radix-2 FFT (block floating point), 40 mel bins, log2 (Q8)
  -> stem Linear(40 -> 32)  (feature normalisation folded in)
  -> 2 x block: RMSNorm -> in_proj(32 -> 128) -> causal depthwise conv (K=4, ring buffer) + SiLU
                -> x_proj -> dt_proj -> C1 recurrence (D_inner=64, N=8, R=4) -> gate with SiLU(z) -> out_proj -> residual
  -> RMSNorm -> head Linear(32 -> 12)
  -> decision rule: EMA of logit margins (alpha 0.93) + refractory lockout (75 tokens), threshold 1.5
```

State per token: SSM state `int32[2][64][8]` (4,096 B) and conv ring `int16[2][3][64]` (768 B). Parameters: 18,060.

---

## On-device (ESP32-C3 @ 160 MHz) — paper model, measured 2026-10-30
<!-- sources: paper/data/board_bench_final.json, serial_eval_final.json (+ serial_eval_final_seed1.json), board_c1_ablation_final.json, baseline_mambalite_c3.json; raw captures board_*_boot.txt, board_mambalite_c3_raw.txt -->

ESP32-C3 DevKitM-1 (chip revision v0.4), ESP-IDF 6.1.0, 160 MHz. The boot banner of the flashed firmware shows weights id `2f4bdc556afab73f` and export id `6a9a3c92`, i.e. the paper model.

| Quantity | Value |
| :--- | :--- |
| Host vs device, 1,000 test clips (seed 0) | **50,000 / 50,000 tokens bit-identical**, max logit difference 0, no missing, desynchronised or CRC-failed token; device and host accuracy both 88.80 % (888 / 1000) |
| Host vs device, second run (200 clips, seed 1) | 10,000 / 10,000 tokens bit-identical; 91.50 % on both |
| Front-end per token (Hann, 512-pt FFT, 40 mel, log2) | 130,325 cycles = 0.815 ms |
| Whole step, fixed point (front-end + model) | median 662,595 cycles (min 659,727, max 665,421) = **4.141 ms** per 20 ms token, real-time factor 0.207 |
| Same, as reported by the device during the 50,000-token serial run | median 665,300 cycles (p99 671,132, max 675,502), which includes the serial protocol's own overhead on the device |
| Same model and front-end, FP32 soft-float | median 7,238,407 cycles = 45.24 ms, real-time factor 2.26 (cannot keep up) |
| Fixed-point vs soft-float speed-up (same model) | **10.9×** |
| `mm_context_t` (static) / recurrent state alone | 6,072 B / 4,864 B |
| Stack used by one step | 3,680 B |
| Static context + stack | 9,752 B |
| Heap allocated during inference | 0 B (free heap 300,500 B before and after, minimum equal) |
| Flash (`firmware.bin` / flash used per PlatformIO) | 219,072 B / 204,880 B; static RAM per PlatformIO 30,376 B (includes ESP-IDF) |

Soft-float micro-costs on the same core: `expf` 2,831 cycles, float multiply-accumulate 227 cycles, int8×int16 MAC 9.0 cycles (loop and loads included). The ESP32-C3 flash cache (16 KB) is smaller than the weights and tables read per token, so cache misses probably cost cycles; this was not measured.

**C1 recurrence against a per-state `expf` decay** (FP32 soft-float build, same noise frame, median of 11 steps; the two variants are the same model, maximum logit difference 2.7e-6): the C1 power recurrence takes 7,239,491 cycles (45.25 ms) per step, a per-(channel, state) `exp(-Δ(n+1))` takes 9,147,942 cycles (57.17 ms): **1.26× slower, +1.91 M cycles per token**. This is the saving of the recurrence itself, measured in soft-float; the fixed-point engine calls no `expf`.

**MambaLite-Micro on the same chip.** Built from the paper's repository (`github.com/Whiten-Rock/MambaLite-Micro`, commit `44d51fce`; all 9 library files byte-identical to it) with a thin benchmark wrapper, 10-class keyword model, offline 1 s window. Different model, different task: it has 35,914 parameters, 10 classes and classifies a whole 100-frame window, against our 18,060 parameters, 12 classes and a 20 ms hop. Static footprint: `firmware.bin` 308,720 B, flash used 308,054 B, static RAM 26,520 B. **It did not give a usable latency figure**: the first call from a fresh boot took 1,558,031,607 cycles (9.74 s, prediction 9, free heap 307,940 → 227,676 B, minimum free heap 10,376 B); the second took 74,288,568 cycles (0.46 s, prediction 0, heap down to 94,160 B) for the *same* input; the third call crashed with a `Load access fault` and the board rebooted, repeating the sequence identically on every boot. Heap use is therefore large and leaked across calls, and two runs on one input disagree. We did not investigate or alter the third-party code (a host-side run of its example was not possible here), so we report no baseline latency and compute no ratio against it. For scale only: our model needs about 207 ms of compute per second of audio (50 tokens × 4.141 ms) with no heap.

The fixed-point firmware was re-flashed after the baseline run and its banner and benchmark re-checked (`board_fixed_final_boot_after_baseline.txt`: step median 662,593 cycles).

---

## Directory structure

```
KWS-Mu-Mamba/
├── checkpoints/              # paper_model.pt (paper model), quant_model.pt (clip baseline), stream_model.pt, best_model.pt
├── engine/                   # (see engine/README.md) C inference engine (host shared library and ESP-IDF component)
│   ├── gen/                  # exported weights, LUTs and Q formats (mm_config.h incl. MM_WEIGHTS_ID, MM_EXPORT_ID)
│   ├── include/, src/        # front-end, float reference block, fixed-point block, model step, decision rule
│   └── tests/                # front-end unit test
├── firmware/                 # (see firmware/README.md) ESP-IDF / PlatformIO firmware (envs: esp32-c3-devkitm-1, esp32-c3-f32bench, esp32-c3-ablation)
├── paper/                    # RESULTS.md, PAPER_CONTEXT.md, IMPROVEMENTS.md, data/ (every reported number), figures/, scripts/
├── tests/                    # parity smoke test, golden-logit regression, math-claim tests, serial_eval mock-device test
├── tools/                    # serial_eval.py (host-to-device parity + timing), stats and table generators
└── train/kws/                # data, model, train, export (calibration), eval_c, stream_eval, quant (unused for the models above)
```

---

## Getting started

Prerequisites: Python 3.12 with `torch numpy scipy pyserial tqdm pytest platformio matplotlib`.

```bash
make -C engine -B all && make -C engine test
PYTHONPATH=. pytest -q tests/        # the golden test needs a golden file for the current export:
                                     # PYTHONPATH=. python tests/make_golden.py
```

Everything for one checkpoint (export with residual-format choice, golden file, full-test and validation evaluation, validation-tuned stream evaluation, test sweep, activation ranges) is `paper/scripts/final_pipeline.sh CKPT TAG`. Individual steps:

```bash
PYTHONPATH=. python -m train.kws.eval_c --ckpt checkpoints/paper_model.pt --engine both --split test \
    --json paper/data/eval_full_test_final.json --dump-npz paper/data/logits_final.npz
PYTHONPATH=. python -m train.kws.stream_eval --ckpt checkpoints/paper_model.pt --engine both \
    --tune-on-validation --fa-budget 100 --json paper/data/stream_final.json     # --sweep for the curve
```

`eval_c` writes its JSON even when the gate fails and then exits non-zero. A 2,000-keyword stream needs about 3 GB of RAM; run heavy jobs under a memory cap.

Firmware and on-device parity (board on `/dev/ttyACM0`). `serial_eval.py` exits non-zero if any token is missing, desynchronised or not bit-identical, if the benchmark reply is not a valid version-2 record, or if the port cannot be opened:

```bash
pio run -d firmware -e esp32-c3-devkitm-1 -t upload
python tools/serial_eval.py --bench --subset 200 --seed 0 --json paper/data/serial_eval_final.json
```

---

## Known limitations

1. On-device checks cover the fixed-point engine on one board (one chip, 1,200 clips in two parity runs); front-end/model latency is a median over repeated steps on a single noise input.
2. Streaming: 80 % detection at 47 [37, 59] false alarms per background hour (about 9 [5, 15] of them in pure background); the refractory (1.5 s) limits closely spaced keywords; operating points rest on one validation and one test stream.
3. The streaming model's clip accuracy (89.02 %) is 3.3 points below the clip-trained baseline (92.31 %).
4. The learned-`A` ablation has three seeds (consistent direction, paired p = 0.086) and uses the from-scratch recipe; the effect for the paper model's recipe is unknown.
5. The MambaLite-Micro baseline did not run reliably on the board (inconsistent repeated runs, then a crash); no baseline latency or speed ratio is reported, and its behaviour was not investigated.
6. The paper model was trained from a dirty working tree (checkpoint metadata records only the HEAD commit, no diff hash).

## License

This project is licensed under the [MIT License](LICENSE).
