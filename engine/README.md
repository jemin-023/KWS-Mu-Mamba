# engine/

Portable C99 inference engine (host shared library and ESP-IDF component). No heap allocation, no FPU needed in the
fixed-point path.

| Path | Content |
| :--- | :--- |
| `include/mu_mamba.h` | public API: context, `mm_step` (one 20 ms token), decision rule, `mm_weights_id` / `mm_export_id` |
| `src/mm_frontend.c` | integer log-mel front-end (tables in `include/mm_frontend_tables.h`) |
| `src/mm_block_q.c` | fixed-point Mamba block with the C1 recurrence (int8 weights, int16 activations, int32 state) |
| `src/mm_block_f32.c` | float reference block (built with `-DMM_ENABLE_F32`; used for parity and the speed ablation) |
| `src/mm_math.c` | LUT lookups, fixed-point helpers |
| `src/mm_model.c` | model step and the streaming decision rule (EMA, threshold, refractory) |
| `gen/` | **generated** by `python -m train.kws.export` (weights, LUTs, `mm_config.h` with `MM_WEIGHTS_ID` / `MM_EXPORT_ID`); do not edit by hand |
| `tests/test_frontend.c` | front-end unit test |

```bash
make -C engine -B all     # builds libmumamba.so (ignored by git)
make -C engine test       # front-end unit test
```

Math background: [`../math.md`](../math.md). Bit-exactness is guarded by `tests/test_golden_logits.py`.
