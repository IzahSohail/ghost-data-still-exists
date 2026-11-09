# Ghost Data Still Exists

A deletion-audit tool that checks whether a user's personal data was actually
deleted, rather than trusting a workflow that reports "Complete".

Given a privacy-deletion request, it decides whether the request really asks
for deletion, reconstructs every identifier the person has used, searches a
mix of evidence (SQL snapshots, replicas, logs, JSON exports, PDFs and
screenshots), applies a retention policy, and returns one of:

- `COMPLETE` with verification steps that cite the evidence proving it,
- `INCOMPLETE` with the remaining records and recreation risks (active syncs,
  restorable backups), an ordered remediation plan, and the checks that must
  pass afterwards, or
- `NOT_APPLICABLE` when the request is for access, export, correction,
  opt-out, or anything other than deletion.

The dataset comes from the HackerRank Orchestrate (October 2026) challenge and
models a fictional commerce platform. All people and records in it are
synthetic.

## How it works

1. **Request intent.** An LLM (Gemini or any OpenAI-compatible provider)
   decides whether `request_text` authorizes deletion, in any language, and
   catches traps such as "send me a copy, do not delete". A small keyword
   classifier is used only when no LLM is available.
2. **Identity resolution** (`code/identity.py`). The request email is resolved
   through the current identity snapshot and the identity audit log to a
   `user_id` plus every current and historical email, phone number and email
   hash.
3. **Evidence indexing** (`code/evidence.py`). Every file in the evidence
   manifest is parsed into observations keyed by identifier and `resource_id`:
   - SQL dumps are loaded into in-memory SQLite (primary, replica and
     failover tables),
   - log lines from several formats are merged by trace id,
   - PDFs are read with `pypdf` and screenshots with PaddleOCR.
   The index is cached in `code/.cache/`.
4. **State reconstruction** (`code/main.py`). Only evidence observed and
   captured at or before the case's `audit_at` is used. The latest
   observation decides each resource's state: present, absent, redacted,
   anonymized, active or disabled. Reported statuses are treated as claims,
   not proof.
5. **Policy and planning.** `deletion_rules.csv` maps each data category to
   delete, redact, anonymize, retain, or keep an account tombstone. Each
   violation becomes a finding, a remediation step and a verification step.
   Data flows and restores are disabled before the records they could
   recreate are changed.

## Layout

```
code/
  main.py              CLI, request intent, state reconstruction, step planning
  identity.py          current + historical identity resolution
  evidence.py          SQL / JSON / log / text / PDF / OCR parsers and evidence index
  requirements.txt
  evaluation/
    evaluate.py        scorer: status, step, field and sequence accuracy
dataset/
  deletion_requests.csv          518 requests to audit
  sample_deletion_requests.csv   32 labelled requests
  sample_output.csv              gold answers for the samples
  user_identities.csv, user_audit_log.csv, system_activity.csv,
  deletion_rules.csv, evidence_files.csv, evidence/
```

## Setup

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
```

Optional: add an LLM key to a `.env` file (repo root or `code/`) to use the
LLM intent classifier. Any OpenAI-compatible provider works:

```bash
# Gemini
GEMINI_API_KEY=...

# or any other provider
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini                       # default when LLM_API_KEY is set
LLM_BASE_URL=https://api.groq.com/openai/v1 # omit for OpenAI
```

Without a key, with `--offline`, or if a call fails, the keyword fallback is
used.

## Run

```bash
python3 code/main.py                       # all 518 requests -> output.csv
python3 code/main.py --sample              # the 32 labelled samples
python3 code/main.py --case case_07        # a single case
python3 code/main.py --offline -o out.csv  # no API calls, custom output path
```

Output is a CSV with `case_id,deletion_status,steps_json`, where `steps_json`
is an ordered array of steps. For example, `case_05` (an account that was
reported deleted but still exists):

```json
[
  {"step_no": 1, "phase": "FINDING", "system_id": "account_service", "resource_id": "account_005",
   "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT", "evidence_id": "evidence_54"},
  {"step_no": 2, "phase": "REMEDIATION", "system_id": "account_service", "resource_id": "account_005",
   "operation": "DELETE_RECORD", "state": "ABSENT", "evidence_id": ""},
  {"step_no": 3, "phase": "VERIFICATION", "system_id": "account_service", "resource_id": "account_005",
   "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT", "evidence_id": ""}
]
```

## Evaluate

```bash
python3 code/main.py --sample --offline -o sample_pred.csv
python3 code/evaluation/evaluate.py --gold dataset/sample_output.csv --pred sample_pred.csv -v
```

The scorer reports status accuracy, exact step-sequence accuracy, step
recall and precision, and per-field accuracy. Only the 32 sample requests
have gold answers; the 518 main requests are unlabelled.
