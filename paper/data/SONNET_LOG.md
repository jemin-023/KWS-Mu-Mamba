# SONNET_LOG — execution of paper/FIX_PLAN.md Part 3 and FIX_PLAN_V2.md

## CURRENT STATUS (after Batch 2) — nothing committed

Paper model: `checkpoints/paper_model.pt` = warm-start streaming fine-tune (warm800), selected on validation only (`paper/data/model_selection.json`).
Engine/gen holds its export (weights id 2f4bdc556afab73f, export id 6a9a3c92, residual Q5 chosen on VALIDATION in Batch 3). Decision defaults: alpha 0.93, refractory 75, threshold 1.5 (confirmed on an extended grid).
Final checks: `make -C engine -B all` 0 warnings; `make -C engine test` pass; `pytest -q tests/` 53 passed; firmware envs
`esp32-c3-devkitm-1` (flash 204,880 B), `esp32-c3-f32bench`, `esp32-c3-ablation` all build.

| Item | State |
|---|---|
| Batch 1 (v2) | DONE, passed the reviewer's QA |
| B2-1 pipeline script, shipped_v2 baseline row | DONE |
| B2-2 warm400, B2-3 warm800 pipelines + validation-only selection | DONE (warm800 selected: 78.1 % vs 66.7 % validation detection within the budget) |
| B2-4 decision defaults | DONE (alpha 0.93, refractory 75, threshold 1.5; at the upper edge of the tuning grid) |
| B2-5 ablation | DONE for three seeds (+3.56, +5.95, +1.86 pt for learned A; mean +3.79 +- 2.05; paired p = 0.086). Seed 2 was trained on the user's notebook VM and retrieved (see "Seed 2" section) |
| B2-6 docs and figures | DONE (README, math.md, `--suffix final` and `--suffix shipped` figures) |
| B2-7 final checks | DONE (above) |
| Board items (on-device parity of the paper model, latency, C1-vs-expf, MambaLite baseline) | **DONE in the "Board session" below** (parity 50,000/50,000 tokens, 4.141 ms per token, C1 vs expf 1.26x; MambaLite-Micro did not run consistently) |

Paper-model headline numbers (Batch 3, Q5 export): clip fixed 89.02 % (float 89.14 %, agreement 99.55 %, gate pass), validation clip 89.24 %;
streaming on the 2.10 h test stream, validation-tuned at 100 FA per background hour: 79.85 % detection at 47.2 [37.0, 59.3] FA/bg-h
(38.1 [29.0, 49.2] substitutions, 9.0 [4.9, 15.2] pure background); same budget for the clip-trained baseline: 10.6 %.
Batch 3 (rigor items): see "Batch 3 (v2)" below.

---

## Older status table (end of Part 3, superseded where the rows above differ)

## FINAL STATUS (E3) — nothing committed

Final checks: `make -C engine clean all` 0 warnings; `make -C engine test` pass; `pytest -q tests/` 38 passed (incl. golden
logits for the final Q6 export); firmware envs `esp32-c3-devkitm-1`, `esp32-c3-f32bench`, `esp32-c3-ablation` all build.

| Item | State |
|---|---|
| A0–A6 | DONE |
| B1 streaming retrain | DONE, **NOT ADOPTED** (fixed clip acc 79.4 % < 91.0 %; FA 1,155 per background hour / 850 per hour at 77.9 % detection, ~3x better than the shipped model at similar detection, not 10x). `checkpoints/stream_model.pt`, `stream_stream_model.json`, `resid_format_choice_stream_model.json` |
| B2 learned-A | DONE: learned A 83.03 % [81.95, 84.05] vs frozen C1 79.47 % [78.31, 80.58] on 4,890 test clips (+3.56 pt, one seed, streaming recipe). `ablation_c1_vs_learned_a.json`. Learned ckpt in `~/.cache/kws-final/learned_a_ckpt` (not copied into checkpoints/) |
| C1 residual format | DONE: Q6 chosen for `quant_model.pt` (92.31 vs 92.27 %, resid sat 0.002 vs 0.048 %). `resid_format_choice.json` |
| C2 golden | DONE (`tests/data/golden_fixed_3e3052cf5abd9675_17c660f7.npz`, keyed by export hash; old Q7 golden kept) |
| C3 host measurements | DONE: `eval_full_test_final.json`, `logits_final.npz`, `stream_final.json` (validation-tuned, both engines), `stream_sweep_final.json`, `activation_ranges_final.json`, `math_claims_final.log` |
| D1 flash + on-device parity | **FAILED / NOT RUN: board not connected all session** |
| D2 float firmware bench | **NOT RUN (no board)** |
| D3 C1 vs expf on board | firmware written and builds; **NOT RUN (no board)** |
| D4 MambaLite baseline on C3 | **NOT RUN (no board)**; docs make no baseline comparison |
| D5 `board_bench_final.json` | **NOT PRODUCED**; README/math.md use the 2026-10-11 numbers and say so |
| E1 figures | DONE for everything with data (`--suffix final`: activation ranges, per-class, logit error, stream DET, learned-A); latency and C1-ablation figures skipped (need board JSON) |
| E2 docs | `README.md` and `math.md` rewritten from `paper/data` files; all RESULTS §8 items addressed except board-dependent ones, which are labelled pre-fix / not measured |
| E3 | DONE (this section) |

Open points for the reviewer:
1. Validation vs test stream at the same threshold 0.25: detection 70.5 % vs 91.3 %, FA/bg-hour 8,231 vs 6,129 — unexplained.
2. Tuning grid floor (0.25) was selected for every model; the 90 % target was never reached on validation, so "validation-tuned" is the fallback (max detection) rule, not a real operating-point selection. A finer/lower grid or a different target would change the table.
3. Stream FA/h of the shipped model (4,510/h at 91 % detection) contradicts the old "< 1 FA/h" claim by ~3 orders of magnitude; claim removed.
4. Learned-A ablation compares models under the streaming recipe only (not the shipped recipe), one seed.
5. Checkpoint `git_commit` is HEAD (5c4020a) while the tree is dirty.
6. `paper/data/_resid_work/` holds scratch exports/evals of the Q-format choice (kept for provenance).

Files changed/added (git status): README.md engine/Makefile engine/gen/mm_config.h engine/gen/mm_luts.c engine/gen/mm_weights.c engine/gen/mm_weights_q.c engine/include/mm_block.h engine/include/mm_block_q.h engine/include/mm_math.h engine/include/mu_mamba.h engine/src/mm_block_f32.c engine/src/mm_block_q.c engine/src/mm_math.c engine/src/mm_model.c firmware/main/CMakeLists.txt firmware/main/audio_serial.c firmware/main/bench.c firmware/main/bench.h firmware/main/main.c firmware/platformio.ini math.md tests/test_c_parity.py tests/test_fixed_parity.py tools/serial_eval.py train/kws/data.py train/kws/eval_c.py train/kws/export.py train/kws/model.py train/kws/quant.py train/kws/stream_eval.py train/kws/train.py checkpoints/stream_model.pt firmware/main/bench_ablation.c paper/ tests/make_golden.py tests/test_engine_fixed.py tests/test_golden_logits.py tests/test_learned_a.py tests/test_math_claims.py tests/test_serial_eval.py tests/test_stream_targets.py tools/check_gpu.py

## Batch 1 (v2) — FIX_PLAN_V2.md, executed after the reviewer's request

Authorisation: the request came from the reviewer session (`kws-mu-mamba-ea`), which stated that the user approved it;
I could not verify that from this session. All edits are local and uncommitted. Rules kept: no change to `model.py` /
`data.py` / training behaviour in `train.py` (metadata only), no training, no heavy job (largest run: a 464 MiB smoke test
with 2 training jobs active), `engine/gen` not re-exported (one define hand-added, see item 5).

Verification (item 8): clean build on a scratch copy of `engine/` = 0 warnings (the live `libmumamba.so` was only rebuilt
incrementally with `make -B`, never deleted, because training processes load it); `make -C engine test` pass;
`pytest -q tests/` **52 passed** (38 previous + 14 new, golden logits included, so the logits are still bit-identical);
`pio run -e esp32-c3-devkitm-1` builds (flash 204,880 B, RAM 30,376 B).

| # | Item | State |
|---|---|---|
| 1 | Decision defaults | DONE: `MM_DECIDE_REFRACTORY` 25 -> 50, `MM_DECIDE_ALPHA_Q15` 22938 -> 27853 (0.85), threshold kept at 6144. `stream_eval.py` argparse and function defaults mirror them (`--ema-alpha 0.85`, `--refractory-tokens 50`). |
| 2 | Budget tuner | DONE: `--fa-budget` (default 100 per background hour), `--tune-grid-alpha` (0.7 0.85 0.93), `--tune-grid-refractory` (25 50 75), 40 thresholds. Rule in `select_setting`: highest detection among settings within budget (ties: fewer FA, lower threshold); none fits -> "budget not reachable" (`budget_reachable: false`) and the lowest-FA setting. Inference runs once per engine; the EMA is computed once per alpha from the C rule's own state (`smoothed_margins`) and the threshold/refractory logic is replayed (`detections_from_margins`). Whole validation grid is stored in the JSON (`validation_grid`, 360 rows). The old `--target-detection` option is removed. |
| 3 | Substitution / background rates | DONE: `fa_wrong_class_per_background_hour` and `fa_outside_per_background_hour` in every result; sweep table has `subst/bg-hr` and `bgnd/bg-hr` columns; counts were already reported. |
| 4 | Checkpoint metadata | DONE: `git_commit`, `git_dirty` (status incl. untracked), `git_diff_sha256` (of `git diff HEAD`, tracked changes only). Unused `_git_commit` removed. |
| 5 | MM_EXPORT_ID | DONE: `train/kws/export_id.py` (`compute_export_id`, `write_export_id`) is the single hash used by `export.py` (writes the define after all files are written) and `tests/make_golden.py`. The hash excludes the `MM_EXPORT_ID` line itself. `mm_export_id()` in the engine, printed in the firmware banner, recorded in `eval_c` JSON and `serial_eval` JSON (`host_export_id`). The current `engine/gen/mm_config.h` got `#define MM_EXPORT_ID "17c660f7"` (computed with the helper, equal to the existing golden file suffix; no other byte of the export changed). `tests/test_export_id.py` checks define == helper hash == engine value == golden name. |
| 6 | A_log weight decay note | DONE: `--learned-a` help text and the learned-A paragraph in `math.md`. No behaviour change. |
| 7 | Validation/test gap | DONE in README (streaming section) and `math.md` (new §6.4): ~1.7 false alarms per background second at threshold 0.25 (9,484 in 1.547 h, from `stream_final.json`), so detection there is not meaningful. The "unexplained" sentence is gone. The 0.25 rows of the README table are labelled degenerate. |

Additions beyond the plan (needed to make item 2 usable): `score_detections` now looks up the few event windows around
each detection (bisect on window starts) instead of scanning all 2,000 events, which makes the 360-row grid cheap;
`tests/test_stream_rule.py` keeps the original full scan as a reference (random events incl. overlapping windows) and checks
`smoothed_margins` + `detections_from_margins` against `mm_decide_ex` token by token for four (alpha, refractory, threshold)
settings, plus tests of the budget rule and of the new per-hour rates. Smoke test of the whole tuner on an 80-keyword stream
(both engines) ran and wrote a 360-row grid.

Caveats for the reviewer:
- `paper/data/stream_final.json`, `stream_sweep_final.json`, `stream_stream_model.json` were produced by the OLD tuner and
  rule defaults (alpha 0.7, refractory 25) and lack the new per-hour keys. README says so. They have not been re-run (heavy job, forbidden in this batch).
- With the new rule the shipped model at an 80-keyword smoke stream reached the 100/h budget only at ~8 % detection, consistent with finding F1 (its state does not return to background); that stream is tiny, so quote nothing from it.
- `MM_DECIDE_THRESHOLD_Q12` is still 1.5 (6144) until Batch 2 tunes it.

## Board session (2026-10-30) — FIX_PLAN Part 3 Phase D on the paper model (weights 2f4bdc556afab73f, export 6a9a3c92)

Board: ESP32-C3 DevKitM-1 (chip revision v0.4) on /dev/ttyACM0, ESP-IDF 6.1.0, 160 MHz. One board job at a time; no other process held the port.

| Item | Result |
|---|---|
| D1 flash fixed firmware | `pio run -e esp32-c3-devkitm-1 -t upload` OK. PlatformIO: flash 204,880 B (19.5 %), RAM 30,376 B (9.3 %); `firmware.bin` 219,072 B. Boot banner: `Model 2f4bdc556afab73f export 6a9a3c92` (both ids present, so no stop). Boot bench: front-end median 130,325 cycles (0.815 ms), step min/median/max 659,727 / 662,595 / 665,421 (4.141 ms median), stack 3,680 B, free heap 300,500 B before = after = min (0 B allocated), context 6,072 B. Raw: `board_fixed_final_boot.txt` |
| D1 parity | `serial_eval.py --bench --subset 1000 --seed 0`: **exit 0**, 1000/1000 clips and 50,000/50,000 tokens bit-identical, max logit difference 0, device-missing 0, desync 0, CRC errors 0, reset failures 0, device accuracy = host accuracy = 88.80 % (888/1000); device-reported per-token cycles during the run: median 665,300, p99 671,132, max 675,502. `serial_eval_final.json`. Extra run seed 1, 200 clips: 10,000/10,000 tokens bit-identical, 91.50 % on both (`serial_eval_final_seed1.json`) |
| D2 float firmware | flash 283,078 B, RAM 108,416 B, `firmware.bin` 290,096 B. `STEP_F32_CYCLES_MEDIAN` 7,238,407 (45.240 ms), 11 rounds, context 11,752 B. Soft-float micro: `expf` 2,831 cycles/call, float MAC 227 cycles, int8xint16 MAC 9.00 cycles. Raw: `board_f32_final_boot.txt` |
| D3 C1 vs expf | `esp32-c3-ablation` (flash 285,834 B, `firmware.bin` 292,608 B): `C1_STEP_CYCLES_MEDIAN` 7,239,491 (45.247 ms), `ZOH_EXPF_STEP_CYCLES_MEDIAN` 9,147,942 (57.175 ms), `MAX_ABS_LOGIT_DIFF` 2.68220901e-06; ratio 1.2636, +1,908,451 cycles per token. My earlier derived estimate (2.5e6) was too high and is superseded. `board_c1_ablation_final.json` |
| D4 MambaLite-Micro on the C3 | Upstream: `https://github.com/Whiten-Rock/MambaLite-Micro`, commit `44d51fce0f17ddadb6111c5e5554d1f7f6c67aff` (2025-09-08), URL taken from the first page of the arXiv PDF (the original clone was no longer on disk, so I re-cloned it into its own directory and compared hashes only). All 9 library files of the port are **byte-identical** to upstream (sha256 per file in `baseline_mambalite_c3.json`); my wrapper `src/main.c` was read before flashing (compute and printf only). Built OK (flash 308,054 B, RAM 26,520 B, `firmware.bin` 308,720 B), flashed, captured 40 s. Result: run 1 (fresh heap): pred 9, 9,737,697 us, 1,558,031,607 cycles, heap 307,940 -> 227,676, min free 10,376; run 2 (same input): pred 0, 464,303 us, 74,288,568 cycles, heap 227,676 -> 94,160; run 3: **panic `Load access fault`**, reboot, same sequence on the next boot (runs 1 and 2 identical to within 1 us / 136 cycles across two boots). **No usable latency figure, no ratio computed.** A host-side run of the upstream example (to see the expected prediction) was attempted and **denied by the permission system**; I did not work around it, so the run1/run2 disagreement is unexplained. Fixed firmware re-flashed afterwards; banner `2f4bdc556afab73f export 6a9a3c92` and step median 662,593 re-confirmed (`board_fixed_final_boot_after_baseline.txt`). Raw: `board_mambalite_c3_raw.txt`, `baseline_mambalite_c3.json` |
| D5 | `board_bench_final.json` (schema of `board_bench_pre_fix.json` + `c1_ablation` + `derived` + parity/serial fields). Derived: fixed 4.141 ms per token (real-time factor 0.207), float 45.24 ms (2.262), float/fixed = 10.92x, static context + stack 9,752 B. Figures regenerated (`fig_latency_final`, `fig_c1_ablation_final`; no baseline-comparison figure, because the baseline gave no consistent numbers). README and math.md on-device sections rewritten from these files; the "earlier model / not re-measured" labels were removed where a number was re-measured |
| Final checks | `make -C engine -B all` 0 warnings; `make -C engine test` pass; `pytest -q tests/` 53 passed |

Process notes: two of my wait loops matched their own command lines (`pgrep -f`) and had to be killed by PID (no data affected). The board is left running the fixed-point paper-model firmware.

## Seed 2 of the ablation (remote notebook VM) — retrieved and added

The user re-opened a tunnel (new address) to the SAME notebook runtime, which had kept running. Both seed-2 runs finished (`exit=0`): C1 seed 2 best epoch 39, learned-A seed 2 best epoch 47 (recipe from checkpoint metadata:
seed 2, from scratch, 160 samples/class, 60 epochs, batch 16; no git metadata on the remote since it has no repository). Checkpoints fetched with `ssh ... cat` and verified by sha256 prefix against the remote copies
(`5d5037a5444c9469`, `834069e22d0a0fe3`) into `~/.cache/kws-final/{c1,learned_a}_seed2_ckpt/best_model.pt`. Remote environment: PyTorch 2.11.0+cu128, Tesla T4 (recorded per pair in the JSON via `--trained-torch`).
Re-measured locally with `ablation_learned_a.py` (not copied from the remote logs, whose own PyTorch numbers were 80.65 % and 82.52 % and agree):
seed 2 C1 80.65 % [79.52, 81.74], learned A 82.52 % [81.43, 83.55] (+1.86 pt). Three seeds: C1 77.64 +- 4.23, learned A 81.43 +- 2.33, paired difference +3.79 +- 2.05 (+3.56, +5.95, +1.86);
paired t-test t = 3.19, 2 df, p = 0.086 (stored in `summary.paired_ttest_learned_minus_c1`): consistent direction, not significant at 0.05 with n = 3. The PyTorch streaming rows do not separate the arms consistently.
`ablation_c1_vs_learned_a.json` is the 3-seed file; the 2-seed version is kept as `ablation_c1_vs_learned_a_seeds42_1.json`, the 1-seed one as `..._seed42.json`.
README (ablation section, limitation 4), math.md section 2.6 and `fig_learned_a_final` updated. `paper/RESULTS.md` (reviewer-owned) was NOT edited: its ablation numbers need the reviewer's update.
Process note: two `pkill -f` calls of mine matched their own command line and killed my shell/SSH session (no data lost); one fetch was slow, so the remote training logs were not copied (checkpoints and metadata are).

## Batch 3 (v2) — rigor items (residual format on validation, extended tuning grid, Poisson CIs)

1. **Residual Q-format on validation (item 2, done first because it can change the export).** `choose_resid_format.py` now selects on `--select-split` (default validation): fixed accuracy, then agreement, then LOWER residual saturation.
   Paper model, validation split (4,443 clips): Q5 89.24 % (agreement 99.26 %, resid sat 0.000 %), Q6 89.22 % (99.32 %, 0.000 %), Q7 89.22 % (99.26 %, 0.004 %) -> **Q5 wins by one clip**, so (as instructed for "a format other than Q7")
   the model was re-exported (export id `6a9a3c92`, engine/gen now holds it), re-goldened (`golden_fixed_2f4bdc556afab73f_6a9a3c92.npz`, older golden files kept) and stages 3-8 were re-run (tag `paper`, outputs copied to the `_final` names; the previous Q7 outputs remain as `*_warm800.*`).
   New test-set numbers (acceptance gate, recorded in `eval_full_test_final.json`): float 89.14 %, fixed 89.02 % (4353/4890, CI [88.11, 89.86]), drop 0.12 pt, agreement 99.55 %, gate pass; validation fixed 89.24 %.
   `resid_format_choice.json` is the validation-based record (rule and selection split stored in it); the old test-based record is `resid_format_choice_warm800.json`. `model_selection.json` got a `final_export` block; the model choice itself does not depend on the Q format.
2. **Extended tuning grid (item 1).** alpha {0.85, 0.93, 0.96, 0.98} x refractory {50, 75, 100, 150} x 40 thresholds = 640 settings per engine, budget 100 FA per background hour, validation only; defaults of `stream_eval --tune-grid-*` now are these grids; the whole grid is stored in `stream_final.json`.
   **Fixed-engine choice unchanged**: alpha 0.93, refractory 75, threshold 1.5 (validation 78.0 % detection at 94.5 FA/bg-h; next best alpha 0.93, refractory 100, threshold 1.5: 77.9 % at 85.5). The choice is therefore interior to the extended grid, `MM_DECIDE_*` and `stream_eval` defaults stay (alpha 30474/0.93, refractory 75, threshold 6144/1.5), nothing to update.
   (The float engine picks refractory 100 instead of 75.) Single test-stream run (stage 6): 79.85 % (1597/2000) at 47.2 [37.0, 59.3] FA per background hour (substitutions 38.1 [29.0, 49.2], pure background 9.0 [4.9, 15.2], 0 duplicates, median latency -209 ms, p90 +49 ms), 34.7 [27.2, 43.6] per hour of the whole stream.
   Sweep (stage 7) re-run on the new export at the final rule: `stream_sweep_final.json`. The 400-sample candidate and the clip-trained baseline rows keep the previous, narrower grid (documented in the README); they were not re-tuned.
3. **Poisson 95 % intervals (item 3).** `stream_eval.poisson_rate_ci` (exact Garwood chi-square interval of the count divided by the hours) feeds `fa_per_hour_ci95`, `fa_per_background_hour_ci95`, `fa_wrong_class_per_background_hour_ci95`, `fa_outside_per_background_hour_ci95` in every result/sweep row and in the validation grid rows;
   `paper/scripts/add_poisson_ci.py` added the same keys to the older stream JSONs (counts and hours were already in them; warm400, warm800, shipped_v2, shipped, stream model, sweeps). `tests/test_stream_rule.py::test_poisson_rate_ci_exact_values` checks the reviewer's reference values (70 FA over 1.547 h -> [35.3, 57.2]; 14 -> [4.95, 15.2]).
   README and math.md tables use the `value [low, high]` style.
Checks after Batch 3: `make -C engine -B all` 0 warnings; `make -C engine test` pass; `pytest -q tests/` **53 passed**; `pio run -e esp32-c3-devkitm-1` builds. Still open: ablation seed 2 (remote VM), all board items.

## Batch 2 (v2) — adopt a streaming-trained paper model (in progress; entries appended as stages finish)

Rules in force (reviewer's memory limits): one heavy job of mine at a time, cap <= 3.5G, started only when MemAvailable >= cap + 2G
(`paper/scripts/final_pipeline.sh` waits for that); `make -B all`, never `make clean`, while reviewer jobs run.

- B2-1 DONE: `paper/scripts/final_pipeline.sh CKPT TAG` (stages 1-8, per-stage logs in `~/.cache/kws-final/pipeline_TAG/`, `STAGES=` to subset).
  Supporting edits: `choose_resid_format.py` now tries q-1, q, q+1 and uses `make -B all` and a per-tag work dir; `activation_ranges.py --ckpt`;
  `tests/test_fixed_parity.py` finds the checkpoint matching the engine weights id ($KWS_CKPT, then checkpoints/*.pt).
  Shipped clip model re-run with the new rule (tag `shipped_v2`: `stream_shipped_v2.json`, `stream_sweep_shipped_v2.json`): budget-tuned
  setting alpha 0.7, refractory 75, threshold 6.25 reaches only 10.6 % test detection at 31 false alarms per background hour (fixed engine).
- Parity test decision (reviewer, option b): `tests/test_fixed_parity.py` is now a regression SMOKE test (300 clips, seed 0, agreement >= 97.5 %,
  median logit error < 0.20); the acceptance gate is the full-test `eval_c` gate (drop <= 0.50 pt and agreement >= 99 %), which every export must
  pass and record in its JSON. Reason: warm400 gave 98.33 % (295/300) on the smoke test but 99.35 % on the full-test gate (4,890 clips).
- B2-2 warm400 (tag `warm400`): stage 1 (calibrated Q_RESID 6; Q5/Q6/Q7 all pass the full gate; Q6 chosen: float 89.33 %, fixed 89.35 %,
  agreement 99.35 %, residual saturation 0.0004 %), stage 2 and the golden file done; stage 3 smoke test failed as described, relaxed, and the full
  suite then passed (52/52) on the warm400 export; stages 4-8 DONE (`eval_full_test_warm400.json`, `logits_warm400.npz`, `eval_val_warm400.json`,
  `stream_warm400.json`, `stream_sweep_warm400.json`, `activation_ranges_warm400.json`).
  warm400 numbers (fixed engine unless noted): validation clip accuracy 90.07 %; test clip accuracy 89.35 % (float 89.33 %, agreement 99.35 %, gate pass);
  budget-tuned on validation (100 FA per background hour; alpha 0.93, refractory 75, threshold 2.25, budget reachable): validation detection 66.7 %;
  test stream detection 73.30 % at 41.4 FA per background hour (substitutions 34.3, pure background 7.1, duplicates 0); median latency -157 ms.
  Float engine: 72.95 % at 38.1. Same budget on the shipped clip model (`shipped_v2`): 10.6 % detection.

- B2-3 DONE: warm800 pipeline (tag `warm800`, all 8 stages, 52/52 tests): calibrated Q6, full-test gate for Q5/Q6/Q7 all pass, Q7 chosen (89.04 % vs 89.02 %, one clip);
  test clip fixed 89.04 % / float 89.14 % / agreement 99.59 %; validation clip fixed 89.22 %; budget-tuned (fixed): alpha 0.93, refractory 75, threshold 1.5, validation detection 78.1 %
  at 95.8 FA/bg-h, TEST 80.05 % at 45.2 FA/bg-h (substitutions 36.2, pure background 9.0, duplicates 0, median latency -206 ms).
  SELECTION (validation only, `paper/data/model_selection.json`): warm800 78.1 % vs warm400 66.7 % validation detection within the budget (difference 11.4 pt > 1.0 pt) -> **warm800 selected**
  (no tie-break needed). Copied to `checkpoints/paper_model.pt` (sha prefix 2f4bdc556afab73f = engine weights id; the live engine/gen export is warm800's Q7 export, export id b944b043, so no re-export was needed;
  golden `tests/data/golden_fixed_2f4bdc556afab73f_b944b043.npz`; all older golden files kept). Previous `_final` files (clip-trained baseline) moved to `_shipped` names
  (eval_full_test, logits, stream (OLD tuner/rule), stream_sweep, activation_ranges, resid_format_choice, logit_errors); warm800 outputs copied to the `_final` names.
  Q-format choice note: the residual format (not the model) was chosen with test accuracy, as specified in Part 3 C1.
- B2-4 DONE: `MM_DECIDE_ALPHA_Q15` 30474 (0.93), `MM_DECIDE_REFRACTORY` 75, threshold unchanged 6144 (= the tuned 1.5); `stream_eval` defaults mirrored. Rebuilt, 0 warnings, 52/52 tests (golden unaffected).
  Both alpha and refractory are at the upper edge of the tuning grid (so the optimum may lie beyond it; documented). `stream_sweep_final.json` was re-run at the final rule (alpha 0.93, refractory 75);
  the pipeline's own sweep (alpha 0.85, refractory 50) is `stream_sweep_warm800.json`.
- B2-5 DONE for seeds 42 and 1 (seed 2 pending on the remote machine): `paper/data/ablation_c1_vs_learned_a.json` (seed 42 alone kept as `..._seed42.json`), re-measured with `ablation_learned_a.py`, 500-keyword PyTorch
  streaming rows included. Test clip accuracy: seed 42 C1 79.47 % vs learned-A 83.03 % (+3.56); seed 1 C1 72.80 % vs 78.75 % (+5.95); mean +- sd C1 76.13 +- 4.71, learned-A 80.89 +- 3.02, paired diff +4.75 +- 1.69.
  The C1 spread across seeds (6.7 pt) exceeds the arm difference, so with n=2 the direction is consistent but the size is poorly determined. The script records the evaluating torch version and, via `--trained-torch`, the training torch version per pair.
- B2-6 DONE: README.md rewritten for the paper model (baseline kept as the clip-trained row; new decision rule; substitutions vs background reported separately; validation-only selection; 2-seed ablation; on-device
  section clearly labelled as the EARLIER model/export), math.md updated (citations fixed to the right files, learned-A block, decision defaults, headroom numbers), figures regenerated (`--suffix final` for the paper model,
  `--suffix shipped` for the baseline). Finding recorded in the docs: for the paper model the dt pre-activation never hits the clamp on long streams (1st percentile of 1-q is 7.1e-5 vs the clamp edge 6.1e-6 for the baseline).
  ACCIDENT, REPAIRED: while rewriting `fig_learned_a` in `make_figures.py` I deleted the reviewer's `fig_stream_models()` function (the file is untracked, so git could not restore it); I recovered it verbatim from the copy
  of `paper/scripts/` that I had sent to the remote machine, and checked that the function lists of both files match.
- Remote notebook machine (user-provided via a free Pinggy SSH tunnel, root, 4 cores, 31 GB RAM, 2x Tesla T4 15 GB, Python 3.13, torch 2.11.0+cu128;
  local run used torch 2.14.1+cu130 on an RTX 4050): code copied (tar, no data/checkpoints/firmware), engine built there (0 warnings), Speech Commands v0.02
  downloaded directly on the remote (3.1 GB, same size as local). Started two extra ablation seeds with the SAME recipe as the local seed-1 runs
  (`train.kws.train --batch-size 16 --workers 2 --device cuda --epochs 60 --seed 2 --init-ckpt ""`, defaults otherwise, 160 samples/class):
  `/root/kws/ckpt_c1_seed2` and `/root/kws/ckpt_learned_a_seed2`, logs `/root/kws/logs/train_*_seed2.log`, ~55 s/epoch, started 10:38 remote time.
  NOT yet retrieved. The tunnel expires 60 min after it was created, so retrieval needs a fresh tunnel (the jobs keep running in the notebook while its runtime lives).
  Caveat: different torch/CUDA versions and GPU than the seed-1/seed-42 runs; GPU training is not bit-reproducible, so report the seeds as independent runs.

## Phase A — tooling and engine fixes (no retrain)

All Phase-A items are DONE. Evidence: `make -C engine clean all` (0 warnings), `make -C engine test` passed,
`pytest -q tests/` 34 passed, golden test (bit-identical fixed logits, weights id 3e3052cf5abd9675) passed after
every C edit. Both firmware envs still build (`esp32-c3-devkitm-1`: flash 204,836 B, RAM 30,376 B;
`esp32-c3-f32bench`: flash 283,044 B).

- **A0** `tests/make_golden.py`, `tests/test_golden_logits.py`, `tests/data/golden_fixed_3e3052cf5abd9675.npz`
  (24 clips x 50 tokens + 2,907-token validation stream). The feed loops are re-implemented in make_golden.py so
  later edits to eval_c/stream_eval cannot move the golden. The test never regenerates.
- **A1 (B1)** `eval_c.py`: `gate_passed` is computed and the JSON (with `gate_passed` and `saturation`) is written
  BEFORE exiting non-zero. Verified: 30-clip subset gives 96.7 % agreement -> `gate_passed: false`, JSON present,
  exit 1. `--dump-npz` saves index, label, float/fixed/PyTorch last-token logits, predictions. `import random` moved up.
- **A2 (B2, B3, B4)** `tools/serial_eval.py`: `send_bench` now uses `read_resp_hdr`, checks status == 0, payload
  length == 64 and version == 2. A failed `--bench` exits 1. The parity denominator is every token the host stepped;
  `device_missing_tokens`, `desync_tokens`, `crc_errors` and `reset_failures` (one retry) are counted; exit 0 only if
  all clips evaluated, all clips and tokens bit-identical, missing == desync == reset failures == 0. `--json` added
  (counters, accuracies + Wilson CIs, cycles min/median/mean/p99/max, cpu_freq, bench dict, host weights id).
  Two additions beyond the plan: failing to open the serial port now exits 1 (it used to return 0, a false pass), and
  `run_serial_eval` accepts an injected serial object. `tests/test_serial_eval.py` drives it with a simulated device:
  clean run passes; a dropped hop gives `device_missing_tokens == 1`, exit 1; 24-byte bench and version-1 bench exit 1.
- **A3 (B5, B6, B13)** `data.build_continuous_stream`: vectorised per-clip target assignment (`with_targets=False`
  skips it; optional `info` dict returns the composition and clip timeline). `tests/test_stream_targets.py` compares
  against the verbatim old loop on 30 seeds x 3 modes (train single-clip, validation single-clip, 20-keyword): equal.
  `num_keywords=2000` builds in 0.5 s (peak RSS 2.9 GB, so full stream jobs need a >= 5G cap).
  `stream_eval.py`: `--engine both` without `--sweep` runs both engines; results now carry `background_hours`,
  `fa_per_background_hour` (+ excl. duplicates), latency (detection time minus event END; mean/median/p90 in ms) and the
  stream composition; `--tune-on-validation` picks the threshold per engine on a validation stream (lowest FA per
  background hour with detection >= `--target-detection`, default 90 %, grid 0.25..10), then one test-stream run uses
  it and records the choice (`threshold_tuned_on_validation`, including `target_reached`). It refuses
  `--split validation` and `--sweep`.
- **A4 (B7, B9)** New counters (total + sat) in `mm_sat_stats_t`, appended at the end: `norm_out` (both block
  RMSNorms and the final one), `y`, `y_gate`, `gate_z` (z outside the SiLU LUT domain, true domain edge +-(8<<12)),
  `p_clamp` (before the +-12 clamp), `logits` (now `mm_sat32` instead of the bare cast). Mirrored in `MMSatStats`,
  `print_saturation_stats`, and the JSON (`sat_stats_dict`). `#error` guard for Q formats > 12 in `mm_block_q.h`;
  `MM_NUM_CLASSES` is defined only in the fallback block after the generated config; `mm_weights_q_size()` /
  `mm_weights_f32_size()` + asserts in `load_engine`. Comments fixed (conv1d_b int32, "EMA of logit margins").
  **Deviation:** `g_mm_sat_stats`, `mm_get_sat_stats`, `mm_reset_sat_stats` are now compiled only with
  `MM_DEBUG_SAT`. They were unconditional, so the firmware carried an unused 96 B `.bss` which the larger struct
  would have doubled. The firmware never calls them (checked), so its behaviour is unchanged; its static RAM is lower.
- **A5 (B10, B11, B12)** `train.py`: `--init-ckpt` default None, prints scratch vs warm start (a missing
  `--init-ckpt` path is an error now), checkpoint stores `init_ckpt` and `git_commit`. `quant.py`: dead
  `compute_multiplier_and_shift` removed, docstring states weight-only QAT / not used for the shipped model /
  unfolded stem, defaults batch 16 and workers 2. `test_fixed_parity.py`: agreement gate 0.99.
- **A6** see the evidence line at the top of this phase.

## Phase B onwards
(appended below)
