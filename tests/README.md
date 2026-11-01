# tests/

```bash
make -C engine -B all
PYTHONPATH=. pytest -q tests/      # 53 tests
```

| File | Checks |
| :--- | :--- |
| `test_golden_logits.py`, `make_golden.py` | fixed engine reproduces stored per-token logits bit-exactly; golden file is keyed by weights id and export id (`tests/data/`). After re-exporting run `PYTHONPATH=. python tests/make_golden.py` |
| `test_fixed_parity.py`, `test_engine_fixed.py`, `test_c_parity.py`, `test_model_parity.py` | C engines vs. PyTorch / float reference |
| `test_math_claims.py` | numerical claims of `math.md` (power recurrence, LUT error, ...) |
| `test_stream_rule.py`, `test_stream_targets.py` | streaming decision rule and scoring |
| `test_export_id.py`, `test_learned_a.py`, `test_serial_eval.py` | export identity, learned-A ablation model, serial tool (mock device) |
