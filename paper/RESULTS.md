# KWS-Mu-Mamba — Results & Verified Math for the Paper

**Status (2026-10-30): FINAL — host results, 3-seed ablation and on-device measurements of the paper model.**
Only open decision: whether to report the MambaLite-Micro first-call number (§5).
Every number has a source in `paper/data/`; the reviewer recomputed the headline numbers from the raw per-clip
logits and stream counts. Branch `fixed-point-and-streaming`, base commit `5c4020a` + uncommitted tree.
Figures: `PYTHONPATH=. python paper/scripts/make_figures.py --suffix final`.

## 0. Models

| Name | File | What it is | Role in the paper |
|---|---|---|---|
| **Paper model** | `checkpoints/paper_model.pt` (weights id `2f4bdc556afab73f`, export `6a9a3c92`) | clip model below + stream fine-tune: warm start, lr 3e-4 (1 warm-up epoch, cosine over 25), 800 samples/class/epoch, batch 16, seed 42; best epoch 9 by validation stream score 0.918 | **the** model: all accuracy, streaming and (after the board session) on-device numbers |
| Clip-trained baseline | `checkpoints/quant_model.pt` (`3e3052cf5abd9675`) | 60 epochs from scratch on ≤ 3.5 s sequences, 100 samples/class/epoch, batch 64 | baseline row: why streaming training is needed |
| Stream-from-scratch C1 / learned-A | `stream_model.pt`, `~/.cache/kws-final/{learned_a,c1_seed1,learned_a_seed1}_ckpt` | 60 epochs from scratch on 3–19 s streams, 160/class | ablation only |

Selection of the paper model used **validation only** (`model_selection.json`): validation-stream detection within
100 false alarms per background hour was 78.1 % (warm800) vs 66.7 % (warm400). The residual Q format was chosen on
validation too (`resid_format_choice.json`; Q5/Q6/Q7 differ by one clip).

## 1. Setup (verified)

| Item | Value |
|---|---|
| Task | Speech Commands v0.02, 12 classes (10 keywords + unknown + silence) |
| Test set | 4,890 clips (official list); per class 396–425, unknown and silence 408 each |
| Front-end | 16 kHz; 512-sample frame (32 ms), 320 hop (20 ms → 50 tokens/s); Hann Q15; 512-pt radix-2 integer FFT with block floating point and 1-bit scaling per stage; 40 HTK-mel triangles 20 Hz–8 kHz (peak 1, unnormalised); log2 in Q8; per-bin mean/std folded into the stem |
| Architecture | stem Linear(40→32), 2 × Mamba-C1 block (D=32, D_inner=64, N=8, R=4, K=4), RMSNorm, Linear(32→12) |
| Parameters | **18,060** (stem 1,312; per block 8,160; final norm 32; head 396) |
| A | frozen, A[d,n] = −(n+1); p̃ clamped to [−12, 12] in training and in both C engines |
| Quantisation | post-training only (no QAT) |
| Clip protocol | zero initial state, 16,192 samples (50 tokens), argmax of the last-token logits |
| Multiplies / token | linear MACs 17,536 + SSM element-wise 4,480 ≈ **22.0 k** |

## 2. Clip accuracy (host C engines, 4,890 test clips)

| Model | FP32 C | W8A16 fixed | Fixed 95 % CI | Agreement | Validation (fixed) |
|---|---|---|---|---|---|
| **Paper model** | 89.14 % | **89.02 %** (4353) | [88.11, 89.86] | 99.55 % | 89.24 % |
| Clip-trained baseline (Q6 export) | 92.37 % | 92.31 % | [91.53, 93.02] | 99.67 % | — |

Paper model, fixed vs float logits: average L∞ 0.145, worst 1.86, median element error 0.029; FP32 C vs PyTorch
worst 1.45e-5, top-1 agreement 100 %. Evaluation gate (drop ≤ 0.5 pt, agreement ≥ 99 %) passes.
Per class (fixed): yes 90.93, no 88.89, up 86.82, down 88.92, left 87.38, right 86.36, on 91.92, off 84.33,
stop 93.92, go 86.57, unknown 84.56, silence 97.55.
Saturation on the test set: everything 0 except conv output 0.046 %, p̃ clamp 0.0014 %, gate-z domain 0.0003 %
(residual Q5: 0 %).
**Streaming training costs 3.3 pt of clip accuracy (92.31 → 89.02 %)**; that is the price of a model that resets.
Sources: `eval_full_test_final.json`, `logits_final.npz`, `eval_full_test_shipped.json`. Figures `fig_per_class_final`, `fig_logit_error_final`.

## 3. Streaming wake-word detection (host, fixed engine)

Protocol: 2,000 test keyword clips in noise, 1–4 s gaps, 30 % of gaps carry an unknown-word distractor; 2.10 h of
stream, 1.547 h of background. A detection is correct within [start − 0.2 s, end + 0.5 s] of a same-class event.
The decision rule is an EMA of logit margins (keyword − max(unknown, silence)) with a threshold and a refractory
period. All three settings are tuned **on a separate validation stream** (budget ≤ 100 false alarms per background
hour; alpha × refractory × 40 thresholds), then applied once to the test stream. Rates carry exact Poisson 95 % CIs.

| Model | Setting (val-tuned) | Detection | False alarms / background h | of which substitutions | of which pure background | Median latency vs keyword end |
|---|---|---|---|---|---|---|
| **Paper model** | α 0.93, refractory 75 (1.5 s), threshold 1.5 | **79.85 %** (1597/2000) | **47.2** [37.0, 59.3] | 38.1 [29.0, 49.2] | **9.0** [4.9, 15.2] | −209 ms (p90 +49) |
| Clip-trained baseline | α 0.7, refr. 75, threshold 6.25 | 10.6 % | 31.0 [22.9, 41.1] | 11.0 | 20.0 [13.6, 28.4] | −75 ms |
| warm400 candidate (narrower grid) | α 0.93, refr. 75, threshold 2.25 | 73.3 % | 41.4 [31.9, 52.8] | 34.3 | 7.1 [3.5, 12.7] | −157 ms |

Per whole-stream hour the paper model makes 34.7 [27.2, 43.6]. Duplicates: 0.
Float engine, same protocol: 79.85 % at 45.9 [35.8, 57.9].
"Substitution" = a wrong keyword fired inside a keyword's window, i.e. a confusion, not a trigger on silence.
Sources: `stream_final.json`, `stream_sweep_final.json`, `stream_shipped_v2.json`, `stream_warm400.json`. Figures `fig_stream_det_final`, `fig_stream_models`.

### Why the clip model fails, and what fixed it (diagnosis, `diagnose_streaming*.json`)
- The clip-trained model never returns to background after a keyword: background accuracy 1 % (0.5–1 s after),
  7 % (1–2 s), 20 % (2–3 s). About 1 % of its channel-steps on streams sit at the p̃ = −12 clamp (1−q = 6.1e-6,
  a time constant of ~54 min), i.e. near-perfect integrators. In the paper model the 1st percentile of 1−q is 7.1e-5
  and the clamp is essentially never reached.
- With the old rule (refractory 0.5 s), most false alarms were duplicate triggers on the same keyword. Refractory
  ≥ 1 s removes them all (about 9× fewer false alarms for stream-trained models).
- Training on streams from scratch gives correct resets but only 72.8–79.5 % clip accuracy. Warm-starting from the
  clip model and fine-tuning on streams keeps 89 % with resets that work.
- The earlier "validation 70.5 % vs test 91.3 %" gap at threshold 0.25 was a degenerate operating point (about 1.7
  triggers per background second).

**Do not claim "< 1 false alarm per hour".** At ≈ 80 % detection the paper model makes ~47 false alarms per background
hour, 9 of them on pure background. That is an honest small-model, 10-keyword result, not a production wake word.

## 4. Ablation: frozen C1 A vs learned A (PyTorch, stream-from-scratch recipe)

| Seed | C1 (frozen A) | Learned A (+1,024 params) | Difference |
|---|---|---|---|
| 42 | 79.47 % [78.31, 80.58] | 83.03 % [81.95, 84.05] | +3.56 |
| 1 | 72.80 % | 78.75 % | +5.95 |
| 2 (torch 2.11 / T4, independent run) | 80.65 % [79.52, 81.74] | 82.52 % [81.43, 83.55] | +1.86 |
| mean ± sd (n=3) | 77.64 ± 4.23 | 81.43 ± 2.33 | **+3.79 ± 2.05** |

Paired t-test (learned − C1): t = 3.19, 2 df, **p = 0.086**. Learned A wins in all three seeds, but at n = 3 the
gap is not significant at 0.05, and the C1 seed spread (7.9 pt) exceeds the mean gap. Write it as "a consistent
but not statistically established cost of about 2–6 pt". In PyTorch streaming (500 keywords) the arms do not
separate consistently. Under the paper model's own recipe (warm start) the cost was not measured. Learned A needs `exp(Δ·A)` per (d, n), i.e. 1,024 `expf` per
token in float, or a 2-D table in fixed point; C1 needs one 1-D lookup and N−1 multiplies. **Frame C1 as a
speed/simplicity vs accuracy trade-off.** On the board the `expf` variant costs 1.26× the C1 float step (§5).
Source: `ablation_c1_vs_learned_a.json`; figure `fig_learned_a_final`.

## 5. On-device (ESP32-C3 rev v0.4 @ 160 MHz, ESP-IDF 6.1.0) — PAPER MODEL, measured 2026-10-30

| Metric | Value |
|---|---|
| Firmware | weights `2f4bdc556afab73f`, export `6a9a3c92` (banner verified before and after the baseline run) |
| Front-end / token | 130,325 cycles = 0.815 ms |
| `mm_step_q` / token (front-end + model) | median **662,595 cycles = 4.14 ms** (min 4.12, max 4.16); **real-time factor 0.207** |
| Per-token cycles over 50,000 streamed tokens (serial_eval) | median 665,300, p99 671,132, max 675,502 |
| Same model in FP32 soft-float | 7,238,407 cycles = 45.24 ms (RTF 2.26) → fixed point **10.9×** faster |
| C1 power recurrence vs per-(d, n) `expf` (float build, same model) | 45.25 ms vs **57.17 ms**: **1.26×**, +1.91 M cycles/token; max logit difference 2.7e-6 |
| Soft-float micro-costs | `expf` 2,831 cycles; float MAC 227; int8×int16 MAC 9.0 (loop included) |
| Memory | context **6,072 B** + stack 3,680 B = **9,752 B**; heap during inference **0 B** (300,500 before = after = minimum) |
| Flash | firmware.bin 219,072 B (flash used 204,880 B; static RAM incl. IDF 30,376 B) |
| **Device vs host bit-exactness** | **1,000 clips / 50,000 tokens bit-identical** (seed 0; max diff 0; no missing/desync/CRC/reset errors), +200 clips / 10,000 tokens (seed 1). Device accuracy = host accuracy (88.80 % / 91.50 % on those subsets) |

Model cycles per multiply: (662,595 − 130,325) / ~22 k ≈ 24 (flash-cache effects plausible, not measured).
Sources: `board_bench_final.json`, `board_c1_ablation_final.json`, `serial_eval_final*.json`, raw `board_*_boot.txt`; figures `fig_latency_final`, `fig_c1_ablation_final`.

### MambaLite-Micro on the same C3 (upstream commit 44d51fc, library files byte-identical, `baseline_mambalite_c3.json`)
- First call after reset: **1,558,031,533 cycles = 9.74 s per 1 s window (RTF ≈ 9.7)**, reproducible across two
  reboots (cycles within 74, identical logits). The second call on the same input gives a different prediction in
  74 M cycles, the heap leaks (307,940 → 227,676 → 94,160 B free) and the third call crashes (Load access fault).
- Plausibility: their per-window operation count on this chip predicts ~1.3–1.5 G cycles (≈ 3 M MACs × 227 +
  204,800 `expf` × 2,831 + other `expf`), so the first-call number matches the cost model; the 74 M-cycle second call does not.
- Not verified: whether the first-call prediction is correct (a host run of the upstream example was blocked by the
  permission system and needs the user's OK).
- If reported, state it as "first invocation after reset; later invocations fail upstream", and give the differences:
  35,914 vs 18,060 parameters, 1 vs 2 layers, 10 vs 12 classes, offline 100-frame window vs 20 ms streaming hop.
  Normalised per second of audio that is 1.56 G vs 0.033 G cycles (≈ 47×). Per frame or token it is ≈ 23×. These
  models differ, so present it as a system comparison, not a like-for-like speed-up.

## 6. Verified math (checkpoint-independent)

| Claim | Result |
|---|---|
| exp(−(n+1)·softplus(p)) = σ(−p)^(n+1) | exact identity (max relative error 1.4e-14) |
| Exact ZOH B̄ = (1−q^(n+1))/(n+1)·B | verified, Δ cancels (not used; the engine uses Euler ΔB) |
| LUTs (SiLU Q12 int16, q Q31, softplus Q24; 513 entries each) | 5,130 B; entries exact |
| Interpolated q, absolute error | max 4.6965e-5 ≤ bound 4.6986e-5 |
| Decay-rate relative error \|δq\|/(1−q) | max **5.97e-4** at p̃ = −11.97 (h²/8 = 4.88e-4 is leading order only) |
| (n+1)q^n at q = 0.7887 | max **1.96** at n = 3 |
| Decay powers by repeated Q31 multiplication | abs error 4.70e-5 … 1.36e-4 (n=0…7); decay-rate relative error ≤ 6.13e-4 |
| Softplus / SiLU LUT errors | 1.22e-4 / 2.67e-4 (≈ 1.09 LSB) |
| Accumulator 64·127·32768 < 2³¹; acc·M ≈ 2^43 | the 64-bit requant product is required |
| Requant round-half-up | max \|error\| = ½ LSB |
| rsqrt_q12 relative error | ≤ 1.93e-3 |
| Floor decay | biased (−247 LSB mean at 1−P = 1e-3); negative states stagnate |
| Round-half-up decay (shipped) | symmetric dead band \|h\| < 0.5/(1−P) LSB; no systematic drift |
| Dead band in real units, Q22 | 0.0194 at the clamp edge; 1.19e-4 at 1−P = 1e-3 |
| Precision actually used | int8 weights; int16 activations (residual Q5 in the paper model, norm Q12, in_proj Q11, conv/SiLU Q12, x_proj Q10, p̃ Q11); **int32** state h (Q22), y, y_gate, logits; out_proj accumulator 64-bit; decay Q31; Δ Q24 |

Figures: `fig_lut_q_error`, `fig_power_recurrence_error`, `fig_decay_rounding_drift`, `fig_activation_ranges_final`.

## 7. Memory layout (`sizeof(mm_context_t)` = 6,072 B)

audio_buf 1,024 + count 2 + mel 80 + conv ring 768 + pad 2 + SSM state 4,096 + logits 48 + smoothed margins 48 +
timer + pad 4. Recurrent state alone: 4,864 B. Stack: FFT re/im 2 × 1,024 B + `power[257]` 1,028 B + block scratch.

## 8. Claims audit: what the paper may NOT say

- 94.00 / 94.40 % accuracy (biased subset), "< 1 FA/h", "14× faster / 39× less RAM than MambaLite-Micro" (old, unmeasured figures;
  use only the §5 same-chip measurement with its caveats), error-ratio or accuracy comparisons with
  MambaLite-Micro (different models and class sets), the old cycle estimates (`expf` 350–500, float MAC ~100,
  "single-cycle" int MAC), the 70/20/5/5 % speed-up attribution, "EMA posterior probability", QAT, the 6,066 B
  context, "q clamped to 2³¹−1", "near-zero cache-miss penalty", "C1 costs no accuracy".
- Comparisons with other KWS models must use their published 12-class v2 numbers and state parameter count and
  protocol. The clip accuracy (89 %) is below state-of-the-art CNN KWS models; the contribution is the integer-only
  streaming SSM on a no-FPU MCU, not accuracy.

## 9. Open items before the results section is final

1. (done) Board session. Open: user decision on the MambaLite-Micro first-call number and the host check of its prediction.
2. (done) Ablation seed 2.
3. (done) Final commit.

Diagnosis and history: `IMPROVEMENTS.md`; execution log `data/SONNET_LOG.md`.
