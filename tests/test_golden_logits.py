"""Bit-exactness guard: the host fixed engine must reproduce the stored per-token logits exactly.

The golden file is keyed by the engine's weights id. If none exists for the loaded library, run
`PYTHONPATH=. python tests/make_golden.py` for the new weights; this test never regenerates it.
"""
import numpy as np
import pytest

from tests.make_golden import compute_golden, golden_path
from train.kws.eval_c import load_engine


def test_fixed_engine_matches_golden_logits():
    lib = load_engine()
    weights_id = lib.mm_weights_id().decode("utf-8")
    path = golden_path(weights_id)
    if not path.exists():
        pytest.fail(
            f"No golden logits for weights id {weights_id} / this export ({path}). "
            "Run 'PYTHONPATH=. python tests/make_golden.py' for the new weights, then re-run."
        )
    golden = np.load(path)
    now = compute_golden(lib)
    assert np.array_equal(golden["clip_indices"], now["clip_indices"])
    assert np.array_equal(golden["clip_logits"], now["clip_logits"]), "clip logits changed"
    assert np.array_equal(golden["stream_logits"], now["stream_logits"]), "stream logits changed"
