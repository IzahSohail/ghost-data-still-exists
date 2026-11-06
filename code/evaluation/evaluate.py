"""
Score a predictions file against a gold file.

    python evaluation/evaluate.py --gold ../dataset/sample_output.csv --pred /path/to/pred.csv

Reports:
  status accuracy      deletion_status equal
  sequence accuracy    steps_json identical as an ordered list (step_no included)
  step set accuracy    steps_json identical as a set (ignores order)
  step recall/precision  per-step match ignoring step_no
  field accuracy       per field over position-aligned steps
"""
import argparse
import csv
import json
from collections import Counter

FIELDS = ["phase", "system_id", "resource_id", "operation", "state", "evidence_id"]


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["case_id"]: r for r in csv.DictReader(fh)}


def steps(row):
    raw = row.get("steps_json") or "[]"
    return json.loads(raw)


def step_key(s):
    return json.dumps({k: s.get(k) for k in FIELDS}, sort_keys=True)


def score(gold, pred, verbose=False):
    status_ok = seq_ok = set_ok = 0
    gold_steps = pred_steps = matched = 0
    field_hits = Counter()
    field_total = 0
    missing = []
    for cid in sorted(gold, key=lambda c: (len(c), c)):
        g = gold[cid]
        p = pred.get(cid)
        if p is None:
            missing.append(cid)
            continue
        gs, ps = steps(g), steps(p)
        status_ok += g["deletion_status"] == p["deletion_status"]
        seq_ok += gs == ps
        gset = {step_key(s) for s in gs}
        pset = {step_key(s) for s in ps}
        set_ok += gset == pset
        gold_steps += len(gset)
        pred_steps += len(pset)
        matched += len(gset & pset)
        for a, b in zip(gs, ps):
            field_total += 1
            for f in FIELDS:
                field_hits[f] += a.get(f) == b.get(f)
        if verbose and (g["deletion_status"] != p["deletion_status"] or gs != ps):
            print(f"{cid}: status gold={g['deletion_status']} pred={p['deletion_status']} "
                  f"steps pred={len(ps)} gold={len(gs)}")
            for s in gs:
                if step_key(s) not in pset:
                    print(f"    missing  {s['phase']:12} {s['operation']:26} {s['resource_id']:18} ev={s['evidence_id']}")
            for s in ps:
                if step_key(s) not in gset:
                    print(f"    extra    {s['phase']:12} {s['operation']:26} {s['resource_id']:18} ev={s['evidence_id']}")
            if gset == pset:
                print("    same steps, different order")

    n = len(gold)
    out = {
        "cases": n,
        "missing_predictions": missing,
        "status_accuracy": status_ok / n if n else 0,
        "sequence_accuracy": seq_ok / n if n else 0,
        "step_set_accuracy": set_ok / n if n else 0,
        "step_recall": matched / gold_steps if gold_steps else 0,
        "step_precision": matched / pred_steps if pred_steps else 0,
        "field_accuracy": {f: field_hits[f] / field_total if field_total else 0 for f in FIELDS},
        "aligned_steps": field_total,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    result = score(load(args.gold), load(args.pred), verbose=args.verbose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
