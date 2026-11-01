"""Pick the residual fixed-point format (MM_Q_RESID) for a checkpoint by running the full eval gate for each candidate.

Usage (repo root, run under the memory cap):
    PYTHONPATH=. python paper/scripts/choose_resid_format.py --ckpt checkpoints/stream_model.pt \
        --out paper/data/resid_format_choice.json [--work-dir paper/data/_resid_work] [--select-split validation]

Candidates: the calibrated Q_RESID q and q-1, q+1 (within 4..12). The residual format is chosen on `--select-split`
(default: validation, so test accuracy never influences it; "test" reproduces the earlier behaviour). For each: export -> rebuild the host engine -> eval_c --engine both on the full test
set. The winner has the higher fixed-engine accuracy (tie: higher top-1 agreement); it is exported and built last,
so engine/gen and engine/libmumamba.so are left on the chosen format.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw)


def export(ckpt, q_resid=None, calib_json=None):
    cmd = [PY, "-m", "train.kws.export", "--ckpt", ckpt]
    if q_resid is not None:
        cmd += ["--q-resid", str(q_resid)]
    if calib_json:
        cmd += ["--calib-json", str(calib_json)]
    run(cmd, check=True)
    # -B rebuilds everything but never deletes libmumamba.so first (other jobs may have it loaded)
    run(["make", "-C", "engine", "-B", "all"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="paper/data/resid_format_choice.json")
    ap.add_argument("--work-dir", default="paper/data/_resid_work")
    ap.add_argument("--select-split", choices=["validation", "test"], default="validation")
    a = ap.parse_args()

    scratch = ROOT / a.work_dir
    scratch.mkdir(parents=True, exist_ok=True)
    calib_json = scratch / "calibration.json"
    export(a.ckpt, calib_json=calib_json)
    calib = json.loads(calib_json.read_text())
    cal_q = calib["calibrated"]["q_resid"]
    cands = [cal_q] + [q for q in (cal_q - 1, cal_q + 1) if 4 <= q <= 12]

    results = {}
    for q in cands:
        export(a.ckpt, q_resid=q)
        rep = scratch / f"eval_q{q}.json"
        rc = run([PY, "-m", "train.kws.eval_c", "--ckpt", a.ckpt, "--engine", "both", "--split", a.select_split,
                  "--json", str(rep)]).returncode
        r = json.loads(rep.read_text())
        results[str(q)] = {
            "q_resid": q,
            "eval_exit_code": rc,
            "gate_passed": r["gate_passed"],
            "float_accuracy_pct": r["float_accuracy_pct"],
            "fixed_accuracy_pct": r["fixed_accuracy_pct"],
            "fixed_correct": r["fixed_correct"],
            "total_clips": r["total_clips"],
            "accuracy_drop_pt": r["accuracy_drop_pt"],
            "top1_agreement_pct": r["top1_agreement_pct"],
            "fixed_vs_float_avg_linf": r["fixed_vs_float_avg_linf"],
            "fixed_vs_float_worst_linf": r["fixed_vs_float_worst_linf"],
            "residual_saturation_pct": r["saturation"]["resid_sat_pct"],
            "saturation": r["saturation"],
        }

    # fixed accuracy, then top-1 agreement, then LOWER residual saturation
    winner = max(results.values(), key=lambda r: (r["fixed_accuracy_pct"], r["top1_agreement_pct"],
                                                 -r["residual_saturation_pct"]))["q_resid"]
    export(a.ckpt, q_resid=winner)
    out = {
        "checkpoint": a.ckpt,
        "calibration": calib,
        "candidates": results,
        "chosen_q_resid": winner,
        "selection_split": a.select_split,
        "rule": f"higher fixed-engine accuracy on the {a.select_split} split; tie -> higher top-1 agreement; tie -> lower residual saturation",
    }
    Path(ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"chosen MM_Q_RESID={winner}; wrote {a.out}")
    for q, r in results.items():
        print(f"  Q{q}: fixed {r['fixed_accuracy_pct']:.2f}%  agreement {r['top1_agreement_pct']:.2f}%  "
              f"resid sat {r['residual_saturation_pct']:.3f}%  gate {'PASS' if r['gate_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
