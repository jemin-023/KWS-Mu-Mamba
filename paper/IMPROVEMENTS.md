# Improvement Scan — what failed, why, and what fixes it (2026-10-30)

Evidence: `paper/data/diagnose_streaming.json`, `paper/data/diagnose_streaming_warm.json` (produced by
`paper/scripts/diagnose_streaming.py`: PyTorch float logits, 500-keyword validation/test streams of 32 min each,
0.40 h of background each), `paper/data/ablation_c1_vs_learned_a.json`, training logs in `~/.cache/kws-final/`.
Figure: `paper/figures/fig_stream_models.pdf`.
Stream counts are small (0.40 background hours), so confirm any adopted operating point on the 2,000-keyword
streams with the C fixed engine (`train/kws/stream_eval.py`).

## 1. Diagnosis

| Model | Recipe | Test clip acc. | Background acc. after a keyword (0.5–1 s / 1–2 s / 2–3 s / 4–6 s) | Verdict |
|---|---|---|---|---|
| shipped `quant_model.pt` | old: short sequences, 100/class/epoch, 60 ep | **92.37 %** | 1 % / 7 % / 20 % / 19 % | strong classifier, **never returns to background** |
| `stream_model.pt` | multi-clip streams from scratch, 160/class, 60 ep | 79.47 % | 97 % / 98 % / 98 % / 100 % | forgets correctly, weak classifier |
| learned-A | same as stream_model + learned A | 83.03 % | 97 % / 99 % / 99 % / 100 % | +3.6 pt over C1 (1 seed) |
| **warm** (new, this scan) | shipped + 20 ep stream fine-tune, lr 3e-4, 400/class | **89.33 %** | 68 % / 87 % / 86 % / 98 % | best trade-off so far; forgetting incomplete |

Findings:
- **F1. The shipped model fails at streaming because its state does not decay back to "background" after a keyword.**
  No decision rule can fix that: no validation setting reaches 80 % detection, and false alarms stay above 800 per background hour at
  every setting. This matches the activation scan: about 1 % of channel-steps on streams sit at the p̃ = −12 clamp edge,
  i.e. near-perfect integrators with a time constant of about 54 min.
- **F2. Most streaming false alarms are duplicates caused by the 0.5 s refractory period.** For the stream-trained models,
  refractory 25 → 50 tokens (1 s) removes **all** duplicates and cuts false alarms per background hour about 9×
  (C1 stream model at α 0.85, threshold 1.5: 709 → 93; learned-A: 699 → 58), with 1–3 points less detection.
  Most of what remains are **substitutions** (a wrong keyword inside a keyword window). False alarms in pure
  background are 0–8 per hour for the stream-trained models.
- **F3. The validation-vs-test gap Sonnet flagged (70.5 % vs 91.3 % detection at threshold 0.25) is an artefact.**
  At that threshold the shipped model fires about 1.7 times per background second, so "detections" are class guesses
  that happen to fall inside keyword windows. That detection rate is meaningless and depends on stream composition.
  The tuner then always fell back to "max detection", which picks this degenerate point.
- **F4. Every recipe is data-starved.** There are about 3,000 files per keyword, but 100–160 per class are sampled per
  epoch, which is only 2–3 passes over the data in 60 epochs.
- **F5. Freezing A costs accuracy under the streaming recipe**: 79.5 → 83.0 % with a learned A (non-overlapping
  CIs, but one seed per arm). It is unknown whether the cost holds with more data, warm starts or other seeds.
- **F6. The clip protocol is not the problem.** A 1 s background pre-roll changes clip accuracy by ≤ 1.8 pt for every model.

## 2. Ranked actions

| # | Action | Cost | Expected effect | Evidence |
|---|---|---|---|---|
| **A1** | Decision rule: refractory **50** tokens, EMA α **0.85** (Q15 27853), then tune the threshold on validation **at a false-alarm budget** (e.g. ≤ 100 per background hour), not at an unreachable detection target. Report substitutions and pure-background false alarms separately. Change `MM_DECIDE_*` in `mm_model.c` and the tuner in `stream_eval.py`. | 1 h, no retraining | ~9× fewer false alarms for any stream-trained model | F2, F3 |
| **A2** | Make one **streaming-trained** model the paper model: the warm fine-tune (89.3 % clip, best curve in `fig_stream_models`), after A3. Run the full C pipeline: export + calibration, gate, golden, 2,000-keyword C stream eval, board. | 1–2 h | one model with clip accuracy ≈ 89–90 % **and** a usable streaming curve | §1 |
| **A3** | Finish the forgetting: longer warm fine-tune with more data (`--max-samples-per-class 800`, 30 ep), optionally with the lower p̃ clamp tightened from −12 to −8 (time constant ≤ ~60 s instead of ~54 min). The clamp needs `MM_P_CLAMP_LO` exported in `mm_config.h` and used in `mm_block_q.c`/`mm_block_f32.c`/`model.py`. | 1–2 h GPU per run | background-after-keyword 68–87 % → >95 % (the scratch models show it is learnable) | F1, act. scan |
| **A4** | More data for the from-scratch stream recipe (`--max-samples-per-class 800+`) to recover keyword accuracy | ~2 h GPU | the scratch models' clean background plus better classification | F4 |
| **A5** | C1 vs learned A: 2 more seeds per arm, and the same comparison under the A3 recipe. Report mean ± sd. | 4 × 35 min GPU | turns F5 into a defensible trade-off claim (accuracy cost vs no `expf`) | F5 |
| **A6** | Board session: verify the final export (bit-exact, latency), run the C1-vs-`expf` speed ablation (`esp32-c3-ablation`), MambaLite-Micro on the C3, and optionally a weights-in-RAM latency test (~21 KB RAM vs flash-cache misses) | 1–2 h, board needed | the hardware half of the paper | FIX_PLAN Phase D |
| **A7** | QA fixes (below) | 30 min | — | Part 4 QA |

Recommended order: A1 → A3 (one warm run with more data, ± clamp) → A2 on the winner → A6 → A5 if time allows.

## 3. QA result of Sonnet's Part 3 work (fix-plan Part 4)

| Check | Result |
|---|---|
| C diff | only counters, `#error` guards, `mm_sat32` on logits and comments; no arithmetic change ✓ |
| Bit-exactness | current C code + the pre-change Q7 export reproduces the Q7 golden **and** the independent 2026-10-11 golden (400 test clips × 50 tokens) **bit-exactly** ✓ |
| Build / tests | 0 warnings, `make -C engine test` ✓, 38/38 pytest ✓ |
| Saturation counters | rates match the activation scan (residual 0.002 % at Q6, conv 0.061 %, p̃ clamp 0.26 %; the float scan gives 0.24 %) ✓ |
| Accuracy JSON | numbers self-consistent; Q6 choice justified ✓ |
| README / math.md | honest and sourced ✓; F3 explanation to be added |
| Board | not done (board absent) ✗ |

Issues found in QA (A7):
1. `MM_WEIGHTS_ID` is the checkpoint hash only, so the Q6 and Q7 exports print the same ID on the firmware banner
   and a stale board image cannot be recognised. Add `MM_EXPORT_ID` (the export hash already used for golden
   files) to `mm_config.h`, print it in the banner, and have `serial_eval.py` record it.
2. `stream_eval --tune-on-validation` falls back to max detection when the target is unreachable, which selects
   degenerate operating points (F3). Replace it with a false-alarm-budget rule (A1) and keep the fallback only as
   an explicit option.
3. Checkpoints store `git_commit` = HEAD while the tree is dirty: also store `git_dirty: true` and a hash of `git diff`.
4. The learned-A ablation applies weight decay to `A_log` (Mamba usually excludes it). The effect is small at wd 1e-4; state it, or exclude A_log.
5. Streaming metrics: report substitutions (`fa_wrong_class`) separately from pure-background false alarms (`fa_outside`) in every table, and say that "false alarms per background hour" in the README includes substitutions.
