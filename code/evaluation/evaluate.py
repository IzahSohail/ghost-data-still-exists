"""
Score a predictions CSV against a gold CSV.

From the code/ directory:

    python evaluation/evaluate.py \\
        --gold ../dataset/sample_output.csv \\
        --pred ../output.csv -v

Or with defaults (gold = dataset/sample_output.csv, pred = output.csv at repo root):

    python evaluation/evaluate.py -v

Reports:
  status accuracy       deletion_status equal
  sequence accuracy     steps_json identical as an ordered list (ignores step_no)
  step set accuracy     same steps as a set (ignores order and step_no)
  step recall/precision per-step match ignoring step_no
  field accuracy        per field over position-aligned steps
"""
import argparse
import csv
import json
import os
from collections import Counter

FIELDS = ["phase", "system_id", "resource_id", "operation", "state", "evidence_id"]

# code/evaluation/ → repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    # try as given (cwd-relative), then under repo root
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def load(path):
    path = resolve_path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty csv: {path}")
    for col in ("case_id", "deletion_status", "steps_json"):
        if col not in rows[0]:
            raise SystemExit(f"{path}: missing column {col}")
    return {r["case_id"]: r for r in rows}, path


def steps(row):
    raw = row.get("steps_json") or "[]"
    return json.loads(raw)


def normalize_step(s):
    return {k: s.get(k, "") for k in FIELDS}


def step_key(s):
    return json.dumps(normalize_step(s), sort_keys=True)


def score(gold, pred, verbose=False):
    status_ok = seq_ok = set_ok = 0
    gold_steps = pred_steps = matched = 0
    field_hits = Counter()
    field_total = 0
    missing = []
    for cid in sorted(gold):
        g = gold[cid]
        p = pred.get(cid)
        if p is None:
            missing.append(cid)
            continue
        gs, ps = steps(g), steps(p)
        status_ok += g["deletion_status"] == p["deletion_status"]
        # sequence: same ordered content; step_no ignored (renumbered by writers)
        seq_ok += [normalize_step(s) for s in gs] == [normalize_step(s) for s in ps]
        gset = {step_key(s) for s in gs}
        pset = {step_key(s) for s in ps}
        set_ok += gset == pset
        gold_steps += len(gs)
        pred_steps += len(ps)
        # multiset overlap so recall/precision match list lengths
        gc = Counter(step_key(s) for s in gs)
        pc = Counter(step_key(s) for s in ps)
        matched += sum(min(gc[k], pc[k]) for k in gc)
        for a, b in zip(gs, ps):
            field_total += 1
            for f in FIELDS:
                field_hits[f] += a.get(f) == b.get(f)
        if verbose and (g["deletion_status"] != p["deletion_status"] or gset != pset
                        or [normalize_step(s) for s in gs] != [normalize_step(s) for s in ps]):
            print(f"{cid}: status gold={g['deletion_status']} pred={p['deletion_status']} "
                  f"steps pred={len(ps)} gold={len(gs)}")
            for s in gs:
                if step_key(s) not in pset:
                    print(f"    missing  {s['phase']:12} {s['operation']:26} "
                          f"{s['resource_id']:18} ev={s['evidence_id']}")
            for s in ps:
                if step_key(s) not in gset:
                    print(f"    extra    {s['phase']:12} {s['operation']:26} "
                          f"{s['resource_id']:18} ev={s['evidence_id']}")
            if gset == pset and [normalize_step(s) for s in gs] != [normalize_step(s) for s in ps]:
                print("    same steps, different order")

    n = len(gold)
    out = {
        "cases": n,
        "missing_predictions": missing,
        "status_ok": status_ok,
        "status_accuracy": status_ok / n if n else 0,
        "sequence_ok": seq_ok,
        "sequence_accuracy": seq_ok / n if n else 0,
        "step_set_ok": set_ok,
        "step_set_accuracy": set_ok / n if n else 0,
        "gold_steps": gold_steps,
        "pred_steps": pred_steps,
        "matched_steps": matched,
        "step_recall": matched / gold_steps if gold_steps else 0,
        "step_precision": matched / pred_steps if pred_steps else 0,
        "field_accuracy": {
            f: field_hits[f] / field_total if field_total else 0 for f in FIELDS
        },
        "aligned_steps": field_total,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Score predictions against gold output.csv")
    ap.add_argument(
        "--gold",
        default=os.path.join("dataset", "sample_output.csv"),
        help="Gold CSV (default: dataset/sample_output.csv under repo root)",
    )
    ap.add_argument(
        "--pred",
        default="output.csv",
        help="Predictions CSV (default: output.csv under repo root)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Print per-case diffs")
    args = ap.parse_args()

    gold, gold_path = load(args.gold)
    pred, pred_path = load(args.pred)
    print(f"gold: {gold_path} ({len(gold)} cases)")
    print(f"pred: {pred_path} ({len(pred)} cases)")
    result = score(gold, pred, verbose=args.verbose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
