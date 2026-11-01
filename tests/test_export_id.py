"""MM_EXPORT_ID in mm_config.h, the helper that names the golden files, and the engine must all agree."""
import re
from pathlib import Path

from tests.make_golden import GEN_DIR, golden_path
from train.kws.eval_c import load_engine
from train.kws.export_id import compute_export_id


def test_config_define_equals_helper_hash():
    cfg = (GEN_DIR / "mm_config.h").read_text()
    m = re.search(r'#define MM_EXPORT_ID\s+"([0-9a-f]{8})"', cfg)
    assert m, "mm_config.h has no MM_EXPORT_ID (run export, or write_export_id)"
    assert m.group(1) == compute_export_id(GEN_DIR)


def test_engine_reports_the_same_export_id_and_golden_uses_it():
    lib = load_engine()
    eid = lib.mm_export_id().decode("utf-8")
    assert eid == compute_export_id(GEN_DIR)
    assert eid in golden_path(lib.mm_weights_id().decode("utf-8")).name
