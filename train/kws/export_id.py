"""Content hash of an exported fixed-point model, shared by export.py and tests/make_golden.py.

MM_WEIGHTS_ID is only the checkpoint hash, so two exports of one checkpoint with different Q formats print the same id.
The export id hashes what defines the fixed-point model: the config (Q formats, dimensions), the quantised weights and
the LUTs. The MM_EXPORT_ID line of mm_config.h is excluded, because it is derived from this hash.
"""
import hashlib
from pathlib import Path

_EXPORT_FILES = ("mm_config.h", "mm_weights_q.c", "mm_luts.c")


def compute_export_id(gen_dir) -> str:
    gen = Path(gen_dir)
    h = hashlib.sha256()
    for name in _EXPORT_FILES:
        data = (gen / name).read_bytes()
        if name == "mm_config.h":
            data = b"".join(l for l in data.splitlines(keepends=True) if b"MM_EXPORT_ID" not in l)
        h.update(data)
    return h.hexdigest()[:8]


def write_export_id(gen_dir) -> str:
    """Computes the export id of gen_dir and (re)writes the MM_EXPORT_ID define into its mm_config.h."""
    cfg = Path(gen_dir) / "mm_config.h"
    export_id = compute_export_id(gen_dir)
    lines = [l for l in cfg.read_text().splitlines(keepends=True) if "MM_EXPORT_ID" not in l]
    out = []
    for l in lines:
        out.append(l)
        if "#define MM_WEIGHTS_ID" in l:
            out.append(f'#define MM_EXPORT_ID   "{export_id}"\n')
    cfg.write_text("".join(out))
    return export_id
