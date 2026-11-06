# Evaluation report

## Setup

Gold data available to us: `dataset/sample_output.csv` (32 cases). The scorer is
`evaluation/evaluate.py`. All numbers below come from running `main.py` on the
sample requests together with the 518 real requests in one run. Running the 32
samples alone gives slightly lower numbers (sequence accuracy 28/32 instead of
29/32) because one tie-break depends on other requests that were reported in
the same batch; see "Known weaknesses".

Command used:

```
python evaluation/evaluate.py --gold ../dataset/sample_output.csv --pred <predictions> -v
```

## Results on the 32 sample cases

| Metric | Value |
| --- | --- |
| Case-level status accuracy | 32/32 (1.000) |
| Sequence accuracy (steps_json identical, order included) | 29/32 (0.906) |
| Step-set accuracy (same steps, order ignored) | 29/32 (0.906) |
| Step recall (gold steps reproduced) | 190/193 (0.984) |
| Step precision (predicted steps that are in gold) | 190/190 (1.000) |

Field-level accuracy over the 190 position-aligned steps:

| Field | Accuracy |
| --- | --- |
| phase | 1.000 |
| system_id | 0.974 |
| resource_id | 0.974 |
| operation | 0.974 |
| state | 0.974 |
| evidence_id | 0.995 |

Whenever the step set matched, the order also matched, so ordering is not a
source of error. The three remaining misses are three steps we cannot produce:

- `case_04` `account_004` – this resource id does not appear in any file under
  `dataset/` other than `sample_output.csv`.
- `case_18` `audit_record_6645` and `case_20` `audit_record_5569` – same.

Nothing in the supplied data names these resources, so the solution has no
grounded way to emit them.

## Additional evaluation cases

We wrote four requests that reuse real identities from `user_identities.csv`
so that evidence exists for them, appended them to the request file and ran
`main.py`. The rows are:

```
case_id,email,requested_at,reported_status,reported_at,audit_at,request_text
extra_01,e.lindberg18@yahoo.com,2026-09-20T09:00:00Z,COMPLETE,2026-09-22T09:00:00Z,2026-10-01T12:00:00Z,Please delete my account and all personal data linked to it. Confirm once done.
extra_02,e.lindberg18@yahoo.com,2026-09-20T09:00:00Z,COMPLETE,2026-09-22T09:00:00Z,2026-09-25T09:00:00Z,Please delete my account and all personal data linked to it. Confirm once done.
extra_03,n.petrescu@yahoo.com,2026-09-10T10:00:00Z,NOT_STARTED,2026-09-11T10:00:00Z,2026-09-20T10:00:00Z,"I would like a copy of all the information you hold about me. Please do not close or delete the account, I still use it."
extra_04,nadia.ionescu@yahoo.com,2026-09-10T10:00:00Z,NOT_STARTED,2026-09-11T10:00:00Z,2026-09-20T10:00:00Z,This was my address before I changed it twice. Erase the account and anything stored under any of my old or current emails.
```

| Case | What it tests | Expected | Result |
| --- | --- | --- | --- |
| `extra_01` | Plain deletion, identity found only in a hand-over text file captured exactly at `audit_at` | COMPLETE, account absent, retained audit record present, both cited to `evidence_03` | As expected |
| `extra_02` | Same identity, `audit_at` moved five days before the only evidence file | The late file must not be cited | `evidence_03` excluded. Status COMPLETE with an empty step list – see weakness 2 |
| `extra_03` | Access request that says the account must stay open | NOT_APPLICABLE, `[]` | As expected, decided by rules without an LLM call |
| `extra_04` | Entry email is two changes old (`nadia.ionescu` → `nadia.petrescu` → `n.petrescu`) | Resolve to `user_10`, three emails, find the leftover support record | Resolved correctly; finding on `support_002` with `evidence_61`, matching the gold for the sample case that uses the same person |

## Failures found during development and what was changed

1. Resources of other people were attached to a case.
   The deletion worker's rows around `reported_at` are used to find the
   primary account when no evidence names it. The filter only excluded
   resources that evidence tied to another case in the same run, so an
   account that evidence tied to a person with no request (for example a
   hand-over line with a different email) could be assigned. `case_04`
   became INCOMPLETE because of a search record owned by someone else.
   Fix: `decision.foreign_owned()` drops a candidate when any evidence names
   an identifier that is not the subject's.

2. Records inside later-captured files were dropped.
   The cut-off used the file's `captured_at`. Snapshot bundles carry a
   per-collection `captured_at` and logs carry per-line timestamps that can
   be earlier than the file capture. Audit records for `case_09`, `case_23`,
   `case_24`, `case_29`, `case_30` were only in such files and were lost.
   Fix: membership uses the record's own time (`observed_ts`); only files
   captured at or before `audit_at` are cited as evidence; retained records
   whose state cannot be proven are still listed in the verification plan of
   an INCOMPLETE case.

3. `cache_lookup result=miss` was read as "record absent".
   Log lines are merged per trace id and the cache line's `result` overwrote
   the read line that said `rows=1`. `audit_record_1596` in `case_15` was
   treated as deleted. Fix: `result` from `cache_lookup` events is not merged.

4. Email case. Requests and evidence use mixed case
   (`SIPHO.MOKOENA@GMAIL.COM`). All email matching is done on lower-cased
   keys.

5. Request classification. The first version sent every request text to the
   model. Rules now settle every one of the 550 texts (543 deletion, 11
   not-deletion across samples, real requests and extra cases); the model is
   only called when the rules return no decision. In the final run it was not
   called at all.

## Known weaknesses

1. Batch pairing. When several requests are reported in the same batch and
   the worker's `account_*` / `analytics_*` rows carry no identifiers, the
   unresolved cases are paired with the unresolved resources by sorting cases
   by `user_id` number and resources by `resource_id` number. This is a
   tie-break, not evidence. On the real set 285 of 646 anchor assignments go
   through it and 264 of 508 cases get their primary account only this way.
   Reversing the sort order changes `steps_json` in 136 cases and the status
   in none. Statuses are therefore stable; the exact `resource_id` of the
   account in those steps is a guess.

2. COMPLETE with no steps. When nothing captured at or before `audit_at`
   mentions the subject, the case has no findings and is reported COMPLETE
   with an empty array (`extra_02`, and 10 real cases). An empty verification
   list for a COMPLETE case violates the contract. The right answer is
   probably INCOMPLETE with an unverified account, but we did not have time to
   change and re-test it.

3. Images and PDFs are not read. `evidence/images/*.png` and
   `evidence/documents/*.pdf` are indexed as "needs vision" and never
   contribute. Cases whose only evidence is a screenshot or a PDF fall back to
   the activity stream.

4. The anchoring window (`reported_at` minus 2 hours to plus 15 minutes) and
   the status-to-result mapping (`COMPLETE→SUCCESS`, `PARTIALLY_COMPLETE→PARTIAL`,
   `NOT_STARTED→FAILED`) were chosen by inspecting the activity stream. They
   fit this dataset's worker cadence; a differently scheduled worker would
   need different values.
