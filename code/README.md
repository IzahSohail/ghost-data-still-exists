# Ghost Data Still Exists – deletion audit

Reads the files under `dataset/`, audits every request in
`dataset/deletion_requests.csv` and writes `output.csv` to the repository root.

## Layout

```
code/
  main.py              entry point: request classification, identity resolution, orchestration
  evidence_index.py    parsers for SQL / JSON / log / text evidence, activity stream helpers
  decision.py          policy tables, state reconstruction, step generation, output writer
  requirements.txt
  evaluation/
    evaluate.py        scorer (gold vs predictions)
    evaluation_report.md
    usage_report.md
```

## Setup

Python 3.10 or newer.

```
cd code
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: a Gemini key for request texts the rules cannot classify. Put it in
the environment or in `code/.env` (not committed):

```
GEMINI_API_KEY=...
```

Without a key the run still completes. If a text cannot be settled by the rules
and no key is set, a warning is printed and a conservative heuristic is used.

## Run

```
cd code
python main.py
```

Reads `../dataset/`, writes `../output.csv` (518 rows, sorted by `case_id`),
and prints classification counts, model usage and decision counts.

## Score against the samples

```
python evaluation/evaluate.py --gold ../dataset/sample_output.csv --pred <predictions.csv> -v
```

To produce predictions for the samples, run `main.py` on a requests file that
contains both the sample and the real requests (one tie-break uses other
requests reported in the same batch), then score the 32 sample rows.

## How it works

1. `request_text` is classified by multilingual rules (deletion verb + object,
   negation, retention and alternative-intent signals). Undecided texts go to
   the model. Non-deletion requests get `NOT_APPLICABLE` and `[]`.
2. The request email is resolved through `user_identities.csv` and
   `user_audit_log.csv` to a `user_id` plus every current and historical email,
   phone and email hash. Matching is case-insensitive.
3. Every evidence file in `evidence_files.csv` is parsed into observations
   keyed by identifier and by `resource_id`. SQL files are loaded with sqlite.
   Images and PDFs are indexed but not read.
4. For each case, observations that mention the subject, observations about
   the same resources, and activity rows about those resources are collected.
   Only records observed at or before `audit_at` are used; only files captured
   at or before `audit_at` are cited.
5. The primary account, when no evidence names it, is taken from the deletion
   worker's rows shortly before `reported_at`, excluding resources that
   evidence ties to another person.
6. `deletion_rules.csv` gives the treatment per data category. The latest
   in-time observation decides each resource's state. Violations become
   FINDING + REMEDIATION steps; VERIFICATION steps are added for the required
   final state. Steps are ordered by phase, operation table order, `system_id`,
   `resource_id`.

## Packaging

From the repository root:

```
zip -r code.zip README.md code -x "*/__pycache__/*" "*/.venv/*" "code/.env"
```
