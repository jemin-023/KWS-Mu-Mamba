"""The vectorised per-token target labelling of build_continuous_stream must equal the original nested loop."""
import random

import numpy as np
import pytest

from train.kws.data import SpeechCommandsDataset, build_continuous_stream, CLASS_TO_IDX

DATA_DIR = "data/speech_commands/v0.02"
HOP = 320
FRAME_LEN = 512


def _reference_targets(n_samples, timeline):
    """The original O(clips x tokens) loop, kept verbatim as the reference."""
    n_tokens = (n_samples - FRAME_LEN) // HOP + 1
    targets = np.full(n_tokens, CLASS_TO_IDX["silence"], dtype=np.int64)
    for st_samp, clip_label in timeline:
        for t in range(n_tokens):
            tok_start = t * HOP
            rel_samp = tok_start - st_samp
            if 0 <= rel_samp < 16000:
                rel_tok = rel_samp // HOP
                if rel_tok < 40:
                    targets[t] = -100
                else:
                    targets[t] = clip_label
            elif 16000 <= rel_samp < 16000 + 4800:
                targets[t] = -100
    return targets


@pytest.fixture(scope="module")
def datasets():
    return {
        "train": SpeechCommandsDataset(data_dir=DATA_DIR, split="train", augment=False),
        "validation": SpeechCommandsDataset(data_dir=DATA_DIR, split="validation", augment=False),
    }


@pytest.mark.parametrize("split,is_training,num_keywords", [
    ("train", True, None),
    ("validation", False, None),
    ("validation", False, 20),
])
def test_vectorised_targets_match_reference(datasets, split, is_training, num_keywords):
    for seed in range(30):
        info = {}
        audio, targets, _ = build_continuous_stream(
            datasets[split], num_keywords=num_keywords, is_training=is_training,
            rng=random.Random(seed), info=info,
        )
        ref = _reference_targets(len(audio), info["timeline"])
        assert targets.shape == ref.shape and np.array_equal(targets, ref), f"seed {seed}"


def test_without_targets_returns_same_audio(datasets):
    a1, t1, e1 = build_continuous_stream(datasets["validation"], num_keywords=5, is_training=False, rng=random.Random(3))
    a2, t2, e2 = build_continuous_stream(datasets["validation"], num_keywords=5, is_training=False, rng=random.Random(3),
                                         with_targets=False)
    assert t2 is None and t1 is not None
    assert np.array_equal(a1, a2) and e1 == e2
