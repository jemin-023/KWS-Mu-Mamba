"""Adds exact (Garwood) Poisson 95 % intervals to false-alarm rates in stream JSONs written before stream_eval did it.

Usage (repo root): PYTHONPATH=. python paper/scripts/add_poisson_ci.py paper/data/stream_*.json

The intervals are a function of counts and observation time that the JSONs already contain, so nothing is re-run:
  - result and sweep rows (they carry false_alarms, fa_wrong_class, fa_outside, background_hours, total_hours);
  - validation_grid rows of a tuned run (background hours of the validation stream from `validation_composition`:
    stream seconds minus 1 s per keyword event).
Entries that already have a *_ci95 key are left untouched.
"""
import json
import sys

from train.kws.stream_eval import poisson_rate_ci


def add_row(row, bg_h, total_h=None):
    n = 0
    for count_key, out_key in (("false_alarms", "fa_per_background_hour_ci95"),
                               ("fa_wrong_class", "fa_wrong_class_per_background_hour_ci95"),
                               ("fa_outside", "fa_outside_per_background_hour_ci95")):
        if count_key in row and out_key not in row and bg_h:
            row[out_key] = poisson_rate_ci(row[count_key], bg_h)
            n += 1
    if total_h and "false_alarms" in row and "fa_per_hour_ci95" not in row:
        row["fa_per_hour_ci95"] = poisson_rate_ci(row["false_alarms"], total_h)
        n += 1
    return n


def walk(node):
    n = 0
    if isinstance(node, dict):
        if "false_alarms" in node and "background_hours" in node:
            n += add_row(node, node["background_hours"], node.get("total_hours"))
        comp = node.get("validation_composition")
        if comp and "validation_grid" in node:
            val_bg_h = (comp["total_seconds"] - comp["n_events"]) / 3600.0
            for row in node["validation_grid"]:
                n += add_row(row, val_bg_h)
            if "validation_fa_per_background_hour" in node and "validation_fa_per_background_hour_ci95" not in node:
                best = next((r for r in node["validation_grid"]
                             if r["threshold"] == node["threshold"] and r["ema_alpha"] == node["ema_alpha"]
                             and r["refractory_tokens"] == node["refractory_tokens"]), None)
                if best is not None:
                    node["validation_fa_per_background_hour_ci95"] = best["fa_per_background_hour_ci95"]
        for v in node.values():
            n += walk(v)
    elif isinstance(node, list):
        for v in node:
            n += walk(v)
    return n


def main():
    for path in sys.argv[1:]:
        data = json.load(open(path))
        n = walk(data)
        if n:
            json.dump(data, open(path, "w"), indent=2)
        print(f"{path}: added {n} interval(s)")


if __name__ == "__main__":
    main()
