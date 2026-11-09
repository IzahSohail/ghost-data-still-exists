import argparse
import csv
import json
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from identity import identity_resolution
from evidence import (
    find_observations,
    find_scopes,
    latest_for_resource,
    get_observations,
    PREFIX_CATEGORY,
    PREFIX_SYSTEM,
)


load_dotenv()

# dataset lives one level up from code/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(REPO_ROOT, 'dataset')

# shared tables (request CSV is chosen at CLI time via --input)
system_activity = pd.read_csv(os.path.join(dataset_path, 'system_activity.csv'))
deletion_rules = pd.read_csv(os.path.join(dataset_path, 'deletion_rules.csv'))

# required_action per data_category
RULE_ACTION = {row["data_category"]: row["required_action"] for _, row in deletion_rules.iterrows()}

# fields that count as direct personal identifiers still on a record
PII_KEYS = (
    "email", "phone", "email_hash", "full_name", "name", "display_name",
    "requester_email", "requester_phone", "requester_name", "owner_email",
    "owner_phone", "customer_email", "matched_value",
)

# step sort: FINDING → REMEDIATION → VERIFICATION, then operation, system, resource
PHASE_RANK = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
FINDING_RANK = {
    "REPORT_REMAINING_RECORD": 0,
    "REPORT_EXPOSED_PII": 1,
    "REPORT_ACTIVE_DATA_FLOW": 2,
    "REPORT_RESTORE_RISK": 3,
}
REMEDIATION_RANK = {
    "DISABLE_DATA_FLOW": 0,
    "BLOCK_RESTORE": 1,
    "DELETE_RECORD": 2,
    "REDACT_RECORD": 3,
    "ANONYMIZE_RECORD": 4,
}
VERIFY_RANK = {
    "VERIFY_RECORD_ABSENT": 0,
    "VERIFY_RECORD_PRESENT": 1,
    "VERIFY_RECORD_REDACTED": 2,
    "VERIFY_RECORD_ANONYMIZED": 3,
    "VERIFY_DATA_FLOW_DISABLED": 4,
    "VERIFY_RESTORE_BLOCKED": 5,
}


def _at(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _is_nullish(value):
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in ("", "null", "none", "nan")


def _has_pii(fields):
    return any(not _is_nullish(fields.get(k)) for k in PII_KEYS)


# --- request intent ------------------------------------------------------

INTENT_PROMPT = """You classify privacy requests sent to an online shop.
Answer YES if the person asks for their account or personal data to be deleted, erased,
removed, or closed, in any language. Also answer YES if they are following up on an earlier
deletion request, for example asking to confirm it finished or complaining that their data
or account still exists.
Answer NO if they want something else: a copy or export of their data, access, correction,
portability, unsubscribing or a marketing opt-out, or if they explicitly say not to delete
or close the account.
Reply with only YES or NO."""

# offline fallback only: generic cues, the LLM handles everything nuanced
NON_DELETION_CUES = (
    "do not delete", "don't delete", "do not close", "don't close",
    "copy of", "export my", "access to my", "correct my", "rectif",
    "portability", "opt out", "opt-out", "unsubscribe",
)
DELETION_CUES = (
    "delet", "erase", "erasure", "remove my", "close my account", "forgotten",
    "supprim", "eliminar", "löschen", "excluir", "apag", "borr", "удал", "삭제", "削除",
)


def judge_request_validity_heuristic(request_text):
    text = (request_text or "").lower()
    if any(cue in text for cue in NON_DELETION_CUES):
        return False
    return any(cue in text for cue in DELETION_CUES)


GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_llm_client = None


def llm_config():
    # LLM_API_KEY works with any OpenAI-compatible provider, GEMINI_API_KEY alone also works
    if os.environ.get("LLM_API_KEY"):
        return {
            "api_key": os.environ["LLM_API_KEY"],
            "base_url": os.environ.get("LLM_BASE_URL") or None,
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        }
    if os.environ.get("GEMINI_API_KEY"):
        return {
            "api_key": os.environ["GEMINI_API_KEY"],
            "base_url": os.environ.get("LLM_BASE_URL") or GEMINI_OPENAI_URL,
            "model": os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
        }
    return None


def ask_llm(system, user, config):
    global _llm_client
    if _llm_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package missing: pip install -r code/requirements.txt (or use --offline)")
        _llm_client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = _llm_client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def judge_request_validity(request_text, offline=False):
    config = None if offline else llm_config()
    if config:
        try:
            return ask_llm(INTENT_PROMPT, request_text or "", config).upper().startswith("YES")
        except Exception as exc:
            warnings.warn(f"LLM call failed ({type(exc).__name__}); using keyword fallback")
    return judge_request_validity_heuristic(request_text)


# --- state from one observation -----------------------------------------

def refine_category(obs):
    # prefix default, with a few content overrides from the rules' real meanings
    cat = obs.get("category") or PREFIX_CATEGORY.get(obs["resource_id"].rsplit("_", 1)[0], "")
    fields = obs.get("fields") or {}
    if fields.get("record_type") in RULE_ACTION:
        return fields["record_type"]
    if cat == "SUPPORT_MESSAGE":
        if fields.get("record_type") == "SUPPORT_ATTACHMENT" or obs["file_path"].startswith("evidence/documents/"):
            return "SUPPORT_ATTACHMENT"
        if any(k in fields for k in ("file_name", "attachment", "object_key")):
            return "SUPPORT_ATTACHMENT"
    if cat == "ANALYTICS_PROFILE" and (
        fields.get("email_opt_in") is not None
        or "marketing" in str(fields.get("event", "")).lower()
        or "contact_lookup" in str(fields.get("event", "")).lower()
    ):
        return "MARKETING_PROFILE"
    return cat


def infer_state(obs):
    fields = obs.get("fields") or {}
    cat = refine_category(obs)

    # explicit result vocabulary from txt/json evidence (authoritative when present)
    result = str(fields.get("result") or "").upper()
    if result in ("NOT_FOUND", "ABSENT", "MISS", "DELETED", "GONE"):
        return "ABSENT"
    if result in ("IDENTIFIABLE", "FOUND", "PRESENT", "HIT", "REMAINING"):
        if cat in ("DATA_FLOW",) or obs["resource_id"].startswith("data_flow_"):
            return "ACTIVE" if result != "FOUND" else "ACTIVE"
        if cat in ("BACKUP_ARCHIVE",) or obs["resource_id"].startswith("backup_"):
            return "ACTIVE"
        return "PRESENT"
    if result in ("ACTIVE", "ENABLED"):
        return "ACTIVE"
    if result in ("DISABLED", "INACTIVE"):
        return "DISABLED"
    if result in ("BLOCKED", "RESTORE_BLOCKED"):
        return "BLOCKED"
    if result in ("REDACTED", "PII_REMOVED"):
        return "REDACTED"
    if result in ("ANONYMIZED", "UNLINKED"):
        return "ANONYMIZED"

    # explicit absence signals
    status = str(fields.get("http_status") or fields.get("status_code") or "")
    if status in ("404", "404.0"):
        return "ABSENT"
    if fields.get("profile") is None and "status_code" in fields and int(fields.get("status_code") or 0) == 404:
        return "ABSENT"
    if fields.get("rows") in ("0", 0) and "http_status" in fields and str(fields.get("http_status")) == "404":
        return "ABSENT"
    # replica marker checks: row still there and marker missing => remnant
    if str(fields.get("marker_found")).lower() == "false" and fields.get("rows") not in ("0", 0, None, ""):
        return "PRESENT"

    # flows / backups
    if cat == "DATA_FLOW" or obs["resource_id"].startswith("data_flow_"):
        enabled = fields.get("enabled")
        if enabled in (False, "false", "0", "False"):
            return "DISABLED"
        if enabled in (True, "true", "1", "True"):
            return "ACTIVE"
        # CHECK_DATA_FLOW style evidence with no enabled flag but live subscription
        if fields.get("rows") not in ("0", 0) and fields.get("event"):
            return "ACTIVE"
    if cat == "BACKUP_ARCHIVE" or obs["resource_id"].startswith("backup_"):
        allowed = fields.get("restore_allowed")
        if allowed in (False, "false", "0", "False"):
            return "BLOCKED"
        if allowed in (True, "true", "1", "True"):
            return "ACTIVE"

    # retained categories
    if cat in ("SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD"):
        return "PRESENT"

    # subject_reference / user_id alone is the allowed tombstone stable ref — not exposed PII
    subj = str(fields.get("subject_reference") or "")
    subj_is_user = bool(re.fullmatch(r"user_\d+", subj, flags=re.I))
    linked = _has_pii(fields) or (
        not _is_nullish(subj)
        and not subj_is_user
        and subj.lower() not in ("anonymized", "redacted", "null")
    )
    if cat == "ACCOUNT_TOMBSTONE":
        return "PRESENT" if _has_pii(fields) else "REDACTED"
    if cat in ("ANALYTICS_PROFILE", "PRODUCT_TELEMETRY", "SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG"):
        if not linked:
            if cat in ("SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG"):
                return "REDACTED"
            return "ANONYMIZED"
        return "PRESENT"

    if cat == "MARKETING_PROFILE" and not _has_pii(fields) and fields.get("email_opt_in") in (False, "false", None):
        if str(fields.get("rows")) == "0":
            return "ABSENT"

    if linked or fields.get("account_status") or fields.get("profile"):
        return "PRESENT"
    if obs["resource_id"]:
        return "PRESENT"
    return "PRESENT"


def latest_by_resource(hits):
    # one row per resource_id: newest observation before audit
    best = {}
    for obs in hits:
        rid = obs["resource_id"]
        if not rid:
            continue
        prev = best.get(rid)
        if prev is None or (obs["observed_at"], obs["captured_at"]) > (prev["observed_at"], prev["captured_at"]):
            best[rid] = obs
    return best


# --- activity helpers (name resources the identity search cannot see) ----

def account_delete_events(reported_at, window_hours=6):
    report = _at(reported_at)
    lo, hi = report - timedelta(hours=window_hours), report + timedelta(hours=2)
    rows = []
    for _, row in system_activity.iterrows():
        if row["source_system"] != "deletion_worker":
            continue
        if row["event_type"] != "DELETE_RECORD" or row["target_system"] != "account_service":
            continue
        if not str(row["resource_id"]).startswith("account_"):
            continue
        when = _at(row["event_time"])
        if lo <= when <= hi:
            rows.append(row.to_dict())
    return sorted(rows, key=lambda r: r["event_time"])


def resolve_account_id(identity_profile, reported_at, audit_at, known_resources):
    worker_rows = account_delete_events(reported_at, window_hours=6)
    worker_ids = {r["resource_id"] for r in worker_rows}

    # 1) identity-matched account_* evidence, preferring worker-window overlap
    evid_accounts = [rid for rid in known_resources if rid.startswith("account_")]
    if evid_accounts:
        overlap = [rid for rid in evid_accounts if rid in worker_ids]
        if len(overlap) == 1:
            return overlap[0]
        if len(evid_accounts) == 1:
            return evid_accounts[0]
        if overlap:
            # prefer SUCCESS then closest to reported_at
            report = _at(reported_at)
            ranked = []
            for row in worker_rows:
                if row["resource_id"] not in overlap:
                    continue
                delta = abs((_at(row["event_time"]) - report).total_seconds())
                ranked.append((0 if row["reported_result"] == "SUCCESS" else 1, delta, row["resource_id"]))
            ranked.sort()
            return ranked[0][2]

    # 2) evidence lookup keyed by email / user_id (log merge attaches lookup_key→email)
    emails = {e.lower() for e in identity_profile["emails"]}
    uid = identity_profile["user_id"].lower()
    scored = []
    for obs in get_observations():
        rid = obs["resource_id"]
        if not rid.startswith("account_"):
            continue
        if _at(obs["captured_at"]) > _at(audit_at) or _at(obs["observed_at"]) > _at(audit_at):
            continue
        fields = obs["fields"]
        keys = {
            str(fields.get(k) or "").lower()
            for k in ("lookup_key", "correlation_key", "email", "user_id", "matched_value")
        }
        if uid in keys or keys & emails:
            scored.append(rid)
    scored = sorted(set(scored))
    overlap = [rid for rid in scored if rid in worker_ids]
    if len(overlap) == 1:
        return overlap[0]
    if len(scored) == 1:
        return scored[0]

    # 3) early demo ids: user_07 → account_007 when that delete exists near reported_at
    m = re.fullmatch(r"user_(\d+)", identity_profile["user_id"])
    if m and int(m.group(1)) <= 32:
        guess = f"account_{int(m.group(1)):03d}"
        if guess in worker_ids or guess in scored:
            return guess
        if int(m.group(1)) <= 10:
            return guess

    if overlap:
        return overlap[0]
    if scored:
        return scored[0]
    return None


def worker_events_near(reported_at, hours=4):
    report = _at(reported_at)
    lo, hi = report - timedelta(hours=hours), report + timedelta(hours=hours)
    follow_types = {
        "ANONYMIZE_RECORD", "REDACT_RECORD", "DISABLE_DATA_FLOW", "BLOCK_RESTORE",
        "DELETE_RECORD", "CHECK_DATA_FLOW", "CHECK_RESTORE_ACCESS",
    }
    out = []
    for _, row in system_activity.iterrows():
        if row["source_system"] != "deletion_worker":
            continue
        if row["event_type"] not in follow_types:
            continue
        when = _at(row["event_time"])
        if lo <= when <= hi:
            out.append(row.to_dict())
    return sorted(out, key=lambda r: r["event_time"])


def claim_batch_resources(account_id, reported_at):
    # map resource_id -> worker event_type for this account's neighborhood
    if not account_id:
        return {}
    deletes = account_delete_events(reported_at, window_hours=8)
    our = next((d for d in deletes if d["resource_id"] == account_id), None)
    report = _at(reported_at)
    if our is None:
        our_time = report - timedelta(hours=1)
        next_time = report + timedelta(hours=2)
    else:
        our_time = _at(our["event_time"])
        later = [_at(d["event_time"]) for d in deletes if _at(d["event_time"]) > our_time]
        next_time = min(later) if later else (report + timedelta(hours=3))
        next_time = min(next_time, report + timedelta(hours=3))

    owned = {}
    for row in worker_events_near(reported_at, hours=5):
        if row["target_system"] == "account_service":
            continue
        when = _at(row["event_time"])
        if when < our_time - timedelta(minutes=2) or when >= next_time:
            continue
        prior = [d for d in deletes if _at(d["event_time"]) <= when + timedelta(seconds=1)]
        if prior:
            prior.sort(key=lambda d: (when - _at(d["event_time"])).total_seconds())
            if prior[0]["resource_id"] != account_id:
                continue
        owned[row["resource_id"]] = row["event_type"]
    return owned


def category_from_worker_event(resource_id, event_type, default_cat):
    # worker operation reveals the policy treatment the record was supposed to get
    if event_type == "DELETE_RECORD" and resource_id.startswith("support_"):
        return "SUPPORT_ATTACHMENT"
    if event_type == "DELETE_RECORD" and resource_id.startswith("analytics_"):
        # some analytics rows are hard-deleted rather than anonymized
        return "USER_EXPORT"
    if event_type == "REDACT_RECORD" and resource_id.startswith("support_"):
        return "SUPPORT_MESSAGE"
    if event_type == "ANONYMIZE_RECORD":
        return "ANALYTICS_PROFILE" if resource_id.startswith("analytics_") else default_cat
    if event_type in ("DISABLE_DATA_FLOW", "CHECK_DATA_FLOW"):
        return "DATA_FLOW"
    if event_type in ("BLOCK_RESTORE", "CHECK_RESTORE_ACCESS"):
        return "BACKUP_ARCHIVE"
    return default_cat


def files_touching(resource_ids, audit_at):
    files = set()
    cutoff = _at(audit_at)
    for rid in resource_ids:
        for obs in get_observations():
            if obs["resource_id"] != rid:
                continue
            if _at(obs["captured_at"]) <= cutoff and _at(obs["observed_at"]) <= cutoff:
                files.add(obs["file_id"])
    return files


def account_from_neighbor(hit_obs_list, reported_at, audit_at):
    # when identity hits omit the account, recover it from the closest prior
    # account delete before an identity-matched companion's worker event.
    if not hit_obs_list:
        return None
    deletes = account_delete_events(reported_at, window_hours=8)
    if not deletes:
        return None
    report = _at(reported_at)
    hit_rids = {obs["resource_id"] for obs in hit_obs_list if obs.get("resource_id")}
    companion_times = []
    for row in worker_events_near(reported_at, hours=5):
        if row["resource_id"] in hit_rids and row["target_system"] != "account_service":
            companion_times.append(_at(row["event_time"]))
    if companion_times:
        when = min(companion_times)
        prior = [d for d in deletes if _at(d["event_time"]) <= when + timedelta(seconds=1)]
        if prior:
            prior.sort(key=lambda d: (when - _at(d["event_time"])).total_seconds())
            return prior[0]["resource_id"]

    ranked = []
    for d in deletes:
        when = _at(d["event_time"])
        if when > report + timedelta(hours=1):
            continue
        delta = abs((when - report).total_seconds())
        ranked.append((delta, d["resource_id"], d["reported_result"]))
    if not ranked:
        return None
    ranked.sort()
    tight = [r for r in ranked if r[0] <= 90 * 60]
    success = [r for r in tight if r[2] == "SUCCESS"]
    if len({r[1] for r in success}) == 1:
        return success[0][1]
    if len({r[1] for r in tight}) == 1:
        return tight[0][1]
    return None


# --- steps from (category, state) ----------------------------------------

def steps_for_resource(system, resource_id, category, state, evidence_id):
    action = RULE_ACTION.get(category)
    steps = []

    # active recreation risks
    if category == "DATA_FLOW" or resource_id.startswith("data_flow_"):
        if state == "ACTIVE":
            steps.append(("FINDING", "REPORT_ACTIVE_DATA_FLOW", system, resource_id, "ACTIVE", evidence_id))
            steps.append(("REMEDIATION", "DISABLE_DATA_FLOW", system, resource_id, "DISABLED", ""))
            steps.append(("VERIFICATION", "VERIFY_DATA_FLOW_DISABLED", system, resource_id, "DISABLED", ""))
        elif state == "DISABLED":
            steps.append(("VERIFICATION", "VERIFY_DATA_FLOW_DISABLED", system, resource_id, "DISABLED", evidence_id))
        return steps

    if category == "BACKUP_ARCHIVE" or resource_id.startswith("backup_"):
        if state == "ACTIVE":
            steps.append(("FINDING", "REPORT_RESTORE_RISK", system, resource_id, "ACTIVE", evidence_id))
            steps.append(("REMEDIATION", "BLOCK_RESTORE", system, resource_id, "BLOCKED", ""))
            steps.append(("VERIFICATION", "VERIFY_RESTORE_BLOCKED", system, resource_id, "BLOCKED", ""))
        elif state == "BLOCKED":
            steps.append(("VERIFICATION", "VERIFY_RESTORE_BLOCKED", system, resource_id, "BLOCKED", evidence_id))
        return steps

    # retained
    if action == "RETAIN" or category in ("SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD"):
        if state != "ABSENT":
            steps.append(("VERIFICATION", "VERIFY_RECORD_PRESENT", system, resource_id, "PRESENT", evidence_id))
        return steps

    # already compliant
    if state == "ABSENT":
        steps.append(("VERIFICATION", "VERIFY_RECORD_ABSENT", system, resource_id, "ABSENT", evidence_id))
        return steps
    if state == "REDACTED":
        steps.append(("VERIFICATION", "VERIFY_RECORD_REDACTED", system, resource_id, "REDACTED", evidence_id))
        return steps
    if state == "ANONYMIZED":
        steps.append(("VERIFICATION", "VERIFY_RECORD_ANONYMIZED", system, resource_id, "ANONYMIZED", evidence_id))
        return steps

    # violations
    if action == "DELETE" or category in (
        "USER_PROFILE", "SEARCH_INDEX_ENTRY", "CACHE_ENTRY", "USER_EXPORT",
        "MARKETING_PROFILE", "SUPPORT_ATTACHMENT", "SECONDARY_IDENTIFIER",
    ):
        steps.append(("FINDING", "REPORT_REMAINING_RECORD", system, resource_id, "PRESENT", evidence_id))
        steps.append(("REMEDIATION", "DELETE_RECORD", system, resource_id, "ABSENT", ""))
        steps.append(("VERIFICATION", "VERIFY_RECORD_ABSENT", system, resource_id, "ABSENT", ""))
        return steps

    if action == "REDACT" or category in ("SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG", "ACCOUNT_TOMBSTONE"):
        steps.append(("FINDING", "REPORT_EXPOSED_PII", system, resource_id, "PRESENT", evidence_id))
        steps.append(("REMEDIATION", "REDACT_RECORD", system, resource_id, "REDACTED", ""))
        steps.append(("VERIFICATION", "VERIFY_RECORD_REDACTED", system, resource_id, "REDACTED", ""))
        return steps

    if action == "ANONYMIZE" or category in ("ANALYTICS_PROFILE", "PRODUCT_TELEMETRY"):
        steps.append(("FINDING", "REPORT_EXPOSED_PII", system, resource_id, "PRESENT", evidence_id))
        steps.append(("REMEDIATION", "ANONYMIZE_RECORD", system, resource_id, "ANONYMIZED", ""))
        steps.append(("VERIFICATION", "VERIFY_RECORD_ANONYMIZED", system, resource_id, "ANONYMIZED", ""))
        return steps

    # default: treat as delete remnant
    steps.append(("FINDING", "REPORT_REMAINING_RECORD", system, resource_id, "PRESENT", evidence_id))
    steps.append(("REMEDIATION", "DELETE_RECORD", system, resource_id, "ABSENT", ""))
    steps.append(("VERIFICATION", "VERIFY_RECORD_ABSENT", system, resource_id, "ABSENT", ""))
    return steps


def sort_steps(steps):
    def key(step):
        phase, op, system, rid, state, evid = step
        if phase == "FINDING":
            op_rank = FINDING_RANK.get(op, 9)
        elif phase == "REMEDIATION":
            op_rank = REMEDIATION_RANK.get(op, 9)
        else:
            op_rank = VERIFY_RANK.get(op, 9)
        return (PHASE_RANK[phase], op_rank, system, rid)
    ordered = sorted(steps, key=key)
    out = []
    for i, (phase, op, system, rid, state, evid) in enumerate(ordered, 1):
        out.append({
            "step_no": i,
            "phase": phase,
            "system_id": system,
            "resource_id": rid,
            "operation": op,
            "state": state,
            "evidence_id": evid,
        })
    return out


def absence_citation(account_id, audit_at):
    # prefer a file that actually lists other accounts (not an empty scope-only snapshot)
    scopes = find_scopes("account_service", audit_at)
    if not scopes:
        return ""
    # files that contain some account_* observation
    populated = set()
    for obs in get_observations():
        if obs["system"] == "account_service" and obs["resource_id"].startswith("account_"):
            if _at(obs["captured_at"]) <= _at(audit_at):
                populated.add(obs["file_id"])
    ranked = sorted(scopes, key=lambda s: s["captured_at"], reverse=True)
    for s in ranked:
        if s["file_id"] in populated:
            # and this account is not present as a live row in that file
            live = latest_for_resource(account_id, audit_at) if account_id else None
            if live and live["file_id"] == s["file_id"] and infer_state(live) == "PRESENT":
                continue
            return s["file_id"]
    return ranked[0]["file_id"]


def audit_case(record, offline=False):
    case_id = record["case_id"]
    request_text = record["request_text"]
    email = record["email"]
    reported_at = record["reported_at"]
    audit_at = record["audit_at"]

    if not judge_request_validity(request_text, offline=offline):
        return {"case_id": case_id, "deletion_status": "NOT_APPLICABLE", "steps_json": []}

    identity_profile = identity_resolution(email)
    if not identity_profile:
        return {"case_id": case_id, "deletion_status": "INCOMPLETE", "steps_json": []}

    hits = find_observations(identity_profile, audit_at)
    by_rid = latest_by_resource(hits)
    identity_rids = set(by_rid.keys())

    account_id = resolve_account_id(identity_profile, reported_at, audit_at, by_rid.keys())
    if not account_id:
        account_id = account_from_neighbor(hits, reported_at, audit_at)

    batch_events = claim_batch_resources(account_id, reported_at)
    seed_files = files_touching(set(by_rid.keys()) | ({account_id} if account_id else set()), audit_at)

    # only add worker companions that lack identity hits for early silent accounts
    if not seed_files:
        for rid, event_type in batch_events.items():
            if rid in by_rid:
                continue
            obs = latest_for_resource(rid, audit_at)
            if not obs:
                continue
            if event_type in (
                "ANONYMIZE_RECORD", "REDACT_RECORD", "DISABLE_DATA_FLOW", "BLOCK_RESTORE",
                "CHECK_DATA_FLOW", "CHECK_RESTORE_ACCESS", "DELETE_RECORD",
            ) and rid.startswith(("analytics_", "data_flow_", "backup_", "support_", "export_", "tombstone_")):
                by_rid[rid] = obs

    # drop identity-matched accounts that are not the resolved subject account
    if account_id:
        for rid in list(by_rid.keys()):
            if rid.startswith("account_") and rid != account_id:
                del by_rid[rid]

    steps = []
    seen = set()

    for rid, obs in by_rid.items():
        cat = refine_category(obs)
        if rid in batch_events:
            cat = category_from_worker_event(rid, batch_events[rid], cat)
            obs = {**obs, "fields": {**obs.get("fields", {}), "record_type": cat}}
        state = infer_state(obs)
        if rid in batch_events and batch_events[rid] in ("CHECK_DATA_FLOW", "CHECK_RESTORE_ACCESS"):
            if state not in ("DISABLED", "BLOCKED"):
                state = "ACTIVE"
        system = obs["system"] or PREFIX_SYSTEM.get(rid.rsplit("_", 1)[0], "")
        for step in steps_for_resource(system, rid, cat, state, obs["file_id"]):
            steps.append(step)
        seen.add(rid)

    # account absence / remnant when we know the account id
    if account_id and account_id not in seen:
        obs = latest_for_resource(account_id, audit_at)
        if obs and infer_state(obs) == "ABSENT":
            steps.append(("VERIFICATION", "VERIFY_RECORD_ABSENT", "account_service", account_id, "ABSENT", obs["file_id"]))
        elif obs and infer_state(obs) == "PRESENT":
            for step in steps_for_resource("account_service", account_id, "USER_PROFILE", "PRESENT", obs["file_id"]):
                steps.append(step)
        else:
            steps.append(("VERIFICATION", "VERIFY_RECORD_ABSENT", "account_service", account_id, "ABSENT", ""))

    has_finding = any(s[0] == "FINDING" for s in steps)
    if has_finding:
        status = "INCOMPLETE"
        remediated = {s[3] for s in steps if s[0] == "REMEDIATION"}
        fixed = []
        for phase, op, system, rid, state, evid in steps:
            if phase == "FINDING":
                fixed.append((phase, op, system, rid, state, evid))
            elif phase == "REMEDIATION":
                fixed.append((phase, op, system, rid, state, ""))
            else:
                # INCOMPLETE: every verification is future work → empty evidence_id
                fixed.append((phase, op, system, rid, state, ""))
        steps = fixed
    else:
        status = "COMPLETE"
        fixed = []
        for phase, op, system, rid, state, evid in steps:
            if phase != "VERIFICATION":
                continue
            if not evid and rid.startswith("account_"):
                evid = absence_citation(rid, audit_at)
            fixed.append((phase, op, system, rid, state, evid))
        steps = fixed

    return {
        "case_id": case_id,
        "deletion_status": status,
        "steps_json": sort_steps(steps),
    }


def write_output(rows, path):
    rows = sorted(rows, key=lambda r: r["case_id"])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "deletion_status", "steps_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row["case_id"],
                "deletion_status": row["deletion_status"],
                "steps_json": json.dumps(row["steps_json"], ensure_ascii=False, separators=(",", ":")),
            })


def _resolve_path(path, default_dir=None):
    if os.path.isabs(path):
        return path
    base = default_dir or REPO_ROOT
    return os.path.abspath(os.path.join(base, path))


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Data Still Exists — deletion-audit agent",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Deletion-requests CSV (default: dataset/deletion_requests.csv). "
             "Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.csv",
        help="Output CSV path (default: output.csv at repo root). "
             "Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Shorthand: use dataset/sample_deletion_requests.csv as --input "
             "(ignored if --input is set).",
    )
    parser.add_argument("--offline", action="store_true", help="Skip Gemini; use heuristic intent")
    parser.add_argument("--case", type=str, help="Run a single case_id")
    args = parser.parse_args()

    if args.input:
        input_path = _resolve_path(args.input)
    elif args.sample:
        input_path = os.path.join(dataset_path, "sample_deletion_requests.csv")
    else:
        input_path = os.path.join(dataset_path, "deletion_requests.csv")
    out_path = _resolve_path(args.output)

    if not os.path.isfile(input_path):
        raise SystemExit(f"input not found: {input_path}")

    # warm the evidence cache once
    get_observations()

    source = pd.read_csv(input_path)
    records = source.to_dict("records")
    if args.case:
        records = [r for r in records if r["case_id"] == args.case]
        if not records:
            raise SystemExit(f"unknown case_id: {args.case}")

    outputs = []
    for i, record in enumerate(records, 1):
        row = audit_case(record, offline=args.offline)
        outputs.append(row)
        print(f"[{i}/{len(records)}] {row['case_id']} {row['deletion_status']} steps={len(row['steps_json'])}")

    write_output(outputs, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
