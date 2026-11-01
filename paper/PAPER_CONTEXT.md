# Paper-Writing Context (handoff for a fresh session, 2026-10-30)

Start here. This file is self-contained context for writing the paper. Numbers live in **`paper/RESULTS.md`**, the
single source of truth, every value backed by `paper/data/*`. Never write a number that is not in RESULTS.md or in a
`paper/data/` file. Ask before measuring anything new.

## 1. Status
- Research and engineering are finished. All numbers (host and on-device) are measured and were reviewed twice
  (Opus as reviewer, Sonnet as implementer).
- Everything is committed on branch `fixed-point-and-streaming` (base `5c4020a`), in three commits: engine/firmware,
  training/tests/tools, and paper material plus documentation.
- Open decision: whether to report the MambaLite-Micro same-chip measurement (§6 below). If reported, the user should
  first approve a host run of the upstream example to check that its first-call prediction is valid (the
  permission system blocked it, and it must not be worked around).

## 2. Working title and abstract
**Exponential-Free Mamba: Integer-Only Streaming Keyword Spotting on an FPU-less Microcontroller**
Alternatives: "μ-Mamba: Real-Time Streaming KWS with a Fixed-Point SSM in 10 KB of RAM"; "Powers Instead of
Exponentials: …"; "Mamba Without expf: …". The abstract draft (~250 words) is in `paper/ABSTRACT.md`. Optional
MambaLite sentence is in an HTML comment there.

## 3. What the paper claims (contributions)
1. **C1 recurrence.** With A[d,n] = −(n+1) frozen, Δ = softplus(p) and q = σ(−p) = exp(−Δ), the ZOH decay is exactly
   exp(Δ·A) = q^(n+1). One 1-D LUT (q in Q31) plus N−1 multiplies, no runtime transcendental. Euler input B̄ = ΔB
   (Δ from a Q24 softplus LUT). The exact ZOH B̄ = (1−q^(n+1))/(n+1)·B is a derived remark only and is not implemented.
2. **Fixed-point design and analysis.** int8 per-row weights with Q15 multiplier+shift requant (round-half-up); 16-bit
   activations; int32 state h (Q22), y, y_gate, logits; 64-bit out_proj accumulator. Verified error results: q LUT
   absolute error 4.70e-5; relative decay-rate error max 5.97e-4 (h²/8 = 4.88e-4 is leading order only); powers P_n
   error ≤ 1.36e-4; floor decay is biased and stagnates negative states; the shipped round-half-up update is
   symmetric with dead band |h| < 0.5/(1−P) LSB (0.0194 real units at the clamp edge).
3. **Streaming finding.** A model trained on clips never returns to background after a keyword: background accuracy
   1–20 % for 0.5–3 s after it, and ~1 % of channel-steps sit at the p̃ clamp, i.e. integrators with a ~54 min time
   constant. A warm-start fine-tune on long multi-clip streams fixes this. A longer refractory removes duplicate triggers.
4. **System result.** An integer-only, bit-exact, real-time streaming engine on an MCU without an FPU, using 9.75 KB of RAM.
5. **Honest ablations.** Fixed vs soft-float on the same model (10.9×), C1 powers vs per-state expf (1.26×), frozen vs
   learned A (−3.8 ± 2.1 pt, 3 seeds, p = 0.086).

## 4. Headline numbers (details and CIs in RESULTS.md)
| | Value | RESULTS § |
|---|---|---|
| Parameters | 18,060 (2 blocks, D=32, D_inner=64, N=8, R=4, K=4) | 1 |
| Test clip accuracy, paper model (fixed / float) | **89.02 % / 89.14 %**, agreement 99.55 %, CI [88.11, 89.86], 4,890 clips, 12 classes | 2 |
| Clip-trained baseline (fixed) | 92.31 % (it fails at streaming) | 2 |
| Streaming, val-tuned rule (α 0.93, refractory 1.5 s, threshold 1.5) | **79.85 %** detection at **47.2 [37.0, 59.3]** FA per background hour: 38.1 substitutions + **9.0 [4.9, 15.2]** pure background; 0 duplicates; median latency −209 ms vs keyword end | 3 |
| Clip-trained baseline, same budget | 10.6 % detection | 3 |
| ESP32-C3 @160 MHz, per 20 ms token | **4.14 ms** (662,595 cycles), RTF 0.207; front-end 0.815 ms | 5 |
| RAM | 6,072 B static + 3,680 B stack = **9,752 B**; 0 B heap | 5 |
| Bit-exactness, device vs host | 50,000/50,000 tokens (1,000 clips) + 10,000/10,000 | 5 |
| Fixed vs FP32 soft-float, same model | **10.9×** (45.24 ms) | 5 |
| C1 vs per-(d,n) expf, float build | **1.26×** (45.25 vs 57.17 ms), logits within 2.7e-6 | 5 |
| Learned vs frozen A (stream-from-scratch recipe) | 81.43 ± 2.33 vs 77.64 ± 4.23 %, diff +3.79 ± 2.05, p = 0.086 (n=3) | 4 |
| Soft-float micro-costs | expf 2,831 cycles; float MAC 227; int8×int16 MAC 9.0 | 5 |

## 5. Do NOT write (claims audit, RESULTS §8)
- "First" anything (FEMBA on the Edge already runs a quantized Mamba on an MCU).
- "< 1 false alarm per hour", "production wake word", "low-power"/µW (power is not measured).
- 94 % accuracy; "better accuracy than MambaLite-Micro" (different class sets); state-of-the-art accuracy.
- "14× faster / 39× less RAM than MambaLite-Micro" or any ratio against numbers from other hardware.
- QAT (it is post-training quantisation); "EMA posterior probability" (it is an EMA of logit margins); "W8A16" without
  saying h/y/y_gate/logits are 32-bit; "near-zero cache-miss penalty"; "C1 costs no accuracy"; "C1 is the main
  speed-up" (fixed point gives 10.9×, C1 only 1.26× in float).

## 6. Related work notes
- **MambaLite-Micro** (Xu et al., arXiv 2509.05488, ML-for-Systems workshop): FP32 only; ESP32-S3 240 MHz and
  STM32H7 480 MHz (both have FPUs); 1 Mamba block, d_model 64, 100-frame windows; KWS SC v2 3-class 92.0 %, 10-class
  92.5 %; HAR 92.7 %; 1,133 ms per window on the S3, 934–964 ms on the H7; peak RAM 230–276 KB; flash ~305–372 KB;
  operator fusion + lifetime-aware buffers (−83 % RAM); FP32 error vs PyTorch L∞ ~1e-4. Future work they list:
  PTQ, fixed point, SIMD, which is exactly what we deliver. Streaming is claimed but not evaluated.
  **Our same-chip run (unmodified upstream 44d51fc on our C3):** first call after reset 1,558 M cycles = 9.74 s
  per 1 s window, reproducible across reboots and consistent with an operation-count estimate; the 2nd call gives a
  different answer, the heap leaks, the 3rd call crashes. If reported: "first invocation after reset", with model differences
  (35,914 vs 18,060 params, 1 vs 2 layers, 10 vs 12 classes, offline window vs 20 ms streaming).
- **FEMBA on the Edge** (arXiv 2603.26716): quantized bidirectional Mamba EEG model on an ultra-low-power MCU; notes
  that Mamba efficiency on MCUs is bounded by IPC/recurrence overhead (we see ~24 cycles per multiply, consistent).
- **Keyword Mamba** (arXiv 2508.07363): SSMs for KWS, accuracy-focused, not MCU.
- **arXiv 2607.20086** "Cumsum-Composable Phase Transport for Low-Cost Streaming KWS": recent, not yet read. Read it.
- **Hello Edge** (Zhang et al., 1711.07128): MCU KWS baselines (DS-CNN). BC-ResNet (~96.6 % on v2-12 with ~9k params,
  from memory): verify every competitor number from the source before citing. State honestly that our clip
  accuracy is below the CNN state of the art and that the contribution is the integer streaming SSM on a no-FPU MCU.
- Mamba (Gu & Dao 2023), S4D (Gu et al. 2022, S4D-Real init motivates A = −(n+1)), MambaQuant (ICLR 2025),
  Bi-Mamba / 1.58-bit Mamba (quantization of SSMs, not MCU).
- A proper literature search is still needed before submission.

## 7. Suggested outline
1. Introduction: always-on KWS on cheap no-FPU MCUs; SSMs are streaming by nature, but exp is costly; contributions (§3).
2. Background: selective SSM / Mamba discretisation; MCU constraints (RV32IMC, no FPU, 16 KB flash cache).
3. Method: 3.1 C1 recurrence (derivation, cost); 3.2 architecture + integer front-end; 3.3 fixed-point engine
   (formats table, requant, LUTs, round-half-up state update); 3.4 streaming training + decision rule.
4. Numerical analysis: LUT/decay errors, power recurrence, rounding dead band, saturation headroom.
5. Experiments: setup (data, protocols, validation-only tuning); clip accuracy; streaming; on-device latency/memory/bit-exactness;
   ablations (fixed vs float, C1 vs expf, frozen vs learned A); optional same-chip baseline.
6. Limitations: accuracy below CNN SOTA; 47 FA per background hour; one task; A ablation n=3 and only under the
   from-scratch recipe; no power measurement; synthetic streams built from SC clips.
7. Conclusion.
Venue advice: TinyML or edge-ML workshop / short paper fits now. A main conference would want more tasks, more seeds and the
same-chip baseline made fully valid.

## 8. Figures (`paper/figures/*.pdf`, regenerate with `PYTHONPATH=. python paper/scripts/make_figures.py --suffix final`)
| File | Use |
|---|---|
| `fig_lut_q_error` | (a) q LUT absolute error, (b) relative decay-rate error vs p̃, with h²/8 |
| `fig_power_recurrence_error` | P_n error vs n against first-order bound |
| `fig_decay_rounding_drift` | floor vs round-half-up state bias, dead band |
| `fig_activation_ranges_final` | float magnitudes vs fixed-point clamp limits (headroom) |
| `fig_per_class_final` | per-class accuracy with Wilson CIs |
| `fig_logit_error_final` | fixed vs float per-clip L∞ |
| `fig_stream_models` | detection vs FA/bg-h: clip model vs stream fine-tune vs from-scratch C1/learned-A (PyTorch, 500 kw) |
| `fig_stream_det_final` | paper model test sweep (fixed + float) |
| `fig_latency_final` | on-device latency: fixed vs soft-float, front-end split, 20 ms budget |
| `fig_c1_ablation_final` | C1 powers vs per-state expf (float build, on board) |
| `fig_learned_a_final` | frozen vs learned A over 3 seeds |
`*_pre_fix` / `*_shipped` figures are older or baseline versions; do not use them for the paper model.

## 9. Files
- `paper/RESULTS.md`: all numbers, protocols, verified math, claims audit.
- `paper/ABSTRACT.md`: abstract draft.
- `math.md`, `README.md`: long-form derivations and docs (sourced, consistent with RESULTS.md).
- `paper/data/`: every JSON/NPZ/raw board log. Key files: `eval_full_test_final.json`, `logits_final.npz`,
  `stream_final.json`, `stream_sweep_final.json`, `board_bench_final.json`, `board_c1_ablation_final.json`,
  `serial_eval_final.json`, `ablation_c1_vs_learned_a.json`, `baseline_mambalite_c3.json`, `model_selection.json`,
  `diagnose_streaming*.json`, `SONNET_LOG.md` (full execution log).
- History of the verification: `paper/IMPROVEMENTS.md` and `paper/data/SONNET_LOG.md` (the original fix plans were removed from the repository).
- Checkpoints: `checkpoints/paper_model.pt` (paper model), `quant_model.pt` (clip baseline), `stream_model.pt`.

## 10. Methods details to state in the paper
- Data: Speech Commands v0.02, 12 classes; test = official 4,890 clips; validation used for all selection and tuning
  (model choice, Q-format choice, decision rule). The test set was never used for a choice.
- Clip protocol: zero state, 16,192 samples = 50 tokens, last-token argmax.
- Streams: 2,000 keyword clips + 30 % unknown distractors in 1–4 s noise gaps (noise level 0.10), 2.10 h, 1.547 h
  background; detection window [start − 0.2 s, end + 0.5 s]; FA per background hour with exact Poisson CIs;
  substitutions vs pure-background FAs reported separately.
- Training: clip-trained base = 60 epochs from scratch (≤3.5 s sequences, 100 samples/class/epoch, batch 64, AdamW
  lr 1e-3, wd 1e-4, cosine, weight EMA 0.999). Paper model = that + stream fine-tune (3–19 s multi-clip sequences,
  lr 3e-4, 1 warm-up epoch, cosine over 25 epochs, 800 samples/class/epoch, batch 16, seed 42; best epoch 9 by
  validation-stream score). Loss: CE on the last 10 tokens of each keyword clip window (+ ignore regions) and on
  background tokens, averaged separately and summed. GPU: RTX 4050 laptop; seed 2 of the ablation on a T4
  (torch 2.11 vs 2.14, so it is an independent run).
- Export: per-row symmetric int8, Q formats calibrated on validation clips + 20 long validation streams (residual
  p99.99 × 1.25, others p99 × 2), Q5/Q6/Q7 residual chosen on validation (Q5).
- Hardware: ESP32-C3 rev v0.4 (RV32IMC, 160 MHz, no FPU), ESP-IDF 6.1.0 via PlatformIO, -O2.
