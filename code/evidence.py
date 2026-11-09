import json
import os
import re
import sqlite3
import warnings

import pandas as pd
from pypdf import PdfReader


# dataset lives one level up from code/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(REPO_ROOT, 'dataset')

# evidence manifest (file_id <-> path/captured_at)
evidence_files = pd.read_csv(os.path.join(dataset_path, 'evidence_files.csv'))

# only used for LINK_RESOURCE_IDENTITY rows (maps printed references -> resource_id)
system_activity = pd.read_csv(os.path.join(dataset_path, 'system_activity.csv'))

# observations are built once and cached; delete this file to force a rebuild
CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache', 'observations.json')

# resource_id prefix -> canonical system / data_category (dataset conventions)
PREFIX_SYSTEM = {
    "account": "account_service",
    "search": "search_service",
    "support": "support_platform",
    "analytics": "analytics_platform",
    "export": "export_service",
    "data_flow": "integration_service",
    "backup": "backup_service",
    "audit_record": "security_service",
    "tombstone": "identity_service",
    "marketing": "marketing_platform",
    "object": "object_storage",
    "legal_hold": "legal_service",
    "cache": "cache_service",
    "email_log": "email_delivery_service",
    "telemetry": "telemetry_service",
}
PREFIX_CATEGORY = {
    "account": "USER_PROFILE",
    "search": "SEARCH_INDEX_ENTRY",
    "support": "SUPPORT_MESSAGE",
    "analytics": "ANALYTICS_PROFILE",
    "export": "USER_EXPORT",
    "data_flow": "DATA_FLOW",
    "backup": "BACKUP_ARCHIVE",
    "audit_record": "SECURITY_AUDIT_LOG",
    "tombstone": "ACCOUNT_TOMBSTONE",
    "marketing": "MARKETING_PROFILE",
    "object": "SUPPORT_ATTACHMENT",
    "legal_hold": "LEGAL_HOLD_RECORD",
    "cache": "CACHE_ENTRY",
    "email_log": "EMAIL_DELIVERY_LOG",
    "telemetry": "PRODUCT_TELEMETRY",
}

# SQL table -> system (replica tables carry origin_system in the row instead)
SQL_TABLE_SYSTEM = {
    "customer_profiles": "account_service",
    "customer_profiles_primary": "account_service",
    "customer_profiles_failover_replica": "account_service",
    "customer_search_documents": "search_service",
    "search_documents_primary": "search_service",
    "customer_analytics_profiles": "analytics_platform",
    "support_records": "support_platform",
    "customer_exports": "export_service",
    "connector_subscriptions": "integration_service",
    "backup_restore_entries": "backup_service",
    "security_audit_records": "security_service",
}

RESOURCE_RE = re.compile(
    r"\b(?:account|search|support|analytics|export|data_flow|backup|audit_record|tombstone|"
    r"marketing|object|legal_hold|cache|email_log|telemetry)_[A-Za-z0-9]+\b"
)
LOG_KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("((?:[^"\\]|\\.)*)"|(\S+))')

# only these fields are used to match a person (never raw document dumps — those caused false hits)
IDENT_FIELDS = (
    "email", "phone", "email_hash", "user_id", "customer_reference", "subject_reference",
    "subject_ref", "correlation_key", "lookup_key", "match_key", "subject_key",
    "requester_email", "requester_phone", "owner_email", "owner_phone", "customer_email",
    "matched_value", "full_name", "name", "display_name", "requester_name",
    "customer_email",
)
PHONE_FIELDS = ("phone", "requester_phone", "owner_phone", "customer_phone")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
USER_RE = re.compile(r"\buser_\d+\b", re.I)

_observations = None
_scopes = None
_by_resource = None
_ocr_engine = None


def _phone_digits(value):
    return re.sub(r"\D+", "", str(value or ""))


def _prefix(resource_id):
    return str(resource_id or "").rsplit("_", 1)[0]


def _system_for(resource_id, hint=None):
    return PREFIX_SYSTEM.get(_prefix(resource_id), hint)


def _flatten(value, out=None):
    # one-level merge of nested dicts (document/profile/metrics) so every field is searchable
    if out is None:
        out = {}
    for k, v in value.items():
        if isinstance(v, dict):
            _flatten(v, out)
        else:
            out[k] = v
    return out


def _observation(meta, resource_id, system, fields, observed_at=None):
    fields = {k: v for k, v in fields.items() if v is not None and v != ""}
    # match text = identifier fields only (plus resource_id)
    id_parts = [str(resource_id or "")]
    for key in IDENT_FIELDS:
        val = fields.get(key)
        if val is not None and str(val).lower() not in ("", "null", "none"):
            id_parts.append(str(val))
    text = " ".join(id_parts)
    digits = " ".join(_phone_digits(fields.get(f)) for f in PHONE_FIELDS if fields.get(f))
    return {
        "file_id": meta["file_id"],
        "file_path": meta["file_path"],
        "origin": meta["origin"],
        "captured_at": meta["captured_at"],
        "observed_at": observed_at or meta["captured_at"],
        "system": system,
        "resource_id": resource_id or "",
        "category": PREFIX_CATEGORY.get(_prefix(resource_id), ""),
        "fields": fields,
        "text_l": text.lower(),
        "digits": digits,
    }


# --- parsers -------------------------------------------------------------

def _parse_sql(path, meta):
    con = sqlite3.connect(":memory:")
    con.executescript(open(path, encoding="utf-8").read())
    con.row_factory = sqlite3.Row
    out = []
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        if table == "snapshot_scope":
            # declares which systems were checked; a file can be only this (= nothing found)
            for row in con.execute("SELECT origin_system FROM snapshot_scope"):
                out.append(_observation(meta, "", row["origin_system"], {"scope_only": True}))
            continue
        for row in con.execute(f'SELECT * FROM "{table}"'):
            row = dict(row)
            if "record_json" in row:
                # replica/primary tables store the record as JSON
                fields = _flatten(json.loads(row.pop("record_json")))
                fields["collection"] = row.get("collection_name")
                system = _system_for(fields.get("resource_id"), row.get("origin_system"))
            else:
                fields = row
                system = _system_for(row.get("resource_id"), SQL_TABLE_SYSTEM.get(table))
            fields["table"] = table
            out.append(_observation(meta, fields.get("resource_id"), system, fields, row.get("updated_at")))
    return out


def _parse_json(path, meta):
    data = json.load(open(path, encoding="utf-8"))
    out = []

    # walk any of the 4 JSON shapes: a record is any dict with a resource_id
    def walk(node, system=None, when=None):
        if isinstance(node, dict):
            system = node.get("system") or node.get("data_store") or system
            when = node.get("recorded_at") or node.get("observed_at") or node.get("captured_at") \
                or node.get("exported_at") or when
            if "resource_id" in node:
                fields = _flatten(node)
                out.append(_observation(meta, fields["resource_id"], _system_for(fields["resource_id"], system), fields, when))
                return
            for key, value in node.items():
                # service_exports is keyed by system name
                walk(value, key if key in PREFIX_SYSTEM.values() else system, when)
        elif isinstance(node, list):
            for item in node:
                walk(item, system, when)

    walk(data)
    return out


def _log_fields(rest):
    if rest.lstrip().startswith("{"):
        return json.loads(rest[rest.index("{"):])
    # cloud runtime wraps the pairs in message="..." with escaped quotes
    rest = re.sub(r'message="(.*)"$', r"\1", rest.replace('\\"', '"'))
    return {k: (q if q is not None and q != "" else bare) for k, _, q, bare in LOG_KV_RE.findall(rest)}


def _parse_log(path, meta):
    # Merge every line that shares a trace_id so profile_lookup_started
    # (lookup_key=email) attaches to profile_lookup_completed (resource_id=account_*).
    by_trace = {}
    orphan = []  # lines with a resource_id but no trace
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        # keep identity-bearing starts even when resource_id is absent
        if (
            "resource_id" not in line
            and "lookup_key" not in line
            and "correlation_key" not in line
            and "subject_key" not in line
            and "subject_ref" not in line
            and "subject_reference" not in line
        ):
            continue
        when, rest = line.split(" ", 1)
        fields = _log_fields(rest)
        service = fields.get("service") or fields.get("function") or fields.get("logger")
        trace = fields.get("trace_id") or fields.get("requestId")
        if " | " in rest[:40]:
            parts = [p.strip() for p in rest.split(" | ")]
            service, trace, fields["event"] = parts[2], parts[3], parts[4]
        elif rest.startswith("<"):
            parts = rest.split(" ", 5)
            service, trace = parts[3], parts[4]
        if not trace:
            if fields.get("resource_id"):
                orphan.append({"fields": fields, "service": service, "when": when})
            continue
        entry = by_trace.setdefault(trace, {"fields": {}, "service": service, "when": when})
        entry["fields"].update({k: v for k, v in fields.items() if v is not None and v != ""})
        entry["when"] = when
        if service:
            entry["service"] = service

    out = []
    for entry in list(by_trace.values()) + orphan:
        f = entry["fields"]
        rid = f.get("resource_id")
        if not rid:
            continue
        # promote identity-bearing keys into matchable fields
        for key in ("lookup_key", "subject_key", "subject_ref", "subject_reference", "match_key"):
            val = str(f.get(key) or "")
            if not val:
                continue
            if "@" in val and "email" not in f:
                f["email"] = val
            elif val.lower().startswith("user_") and "user_id" not in f:
                f["user_id"] = val
            elif len(val) == 64 and all(c in "0123456789abcdef" for c in val.lower()) and "email_hash" not in f:
                f["email_hash"] = val
            if val.lower().startswith("user_") and "subject_reference" not in f:
                f["subject_reference"] = val
        corr = str(f.get("correlation_key") or "")
        if corr.startswith("user_") and "user_id" not in f:
            f["user_id"] = corr
        elif "@" in corr and "email" not in f:
            f["email"] = corr
        elif len(corr) == 64 and "email_hash" not in f:
            f["email_hash"] = corr
        system = _system_for(rid, f.get("system") or entry["service"])
        out.append(_observation(meta, rid, system, f, entry["when"]))
    return out


def _parse_txt(path, meta):
    out = []
    for line in open(path, encoding="utf-8"):
        parts = [p.strip() for p in line.lstrip("- ").split("|")]
        if len(parts) == 6 and parts[0][:4].isdigit():
            when, system, rid, record_type, result, subject = parts
            fields = {"resource_id": rid, "record_type": record_type, "result": result, "subject_reference": subject}
            out.append(_observation(meta, rid, _system_for(rid, system), fields, when))
    return out


def _document_observations(meta, text, aliases, link_owners=None):
    # PDFs/screenshots: pull out structured ids; aliases map printed refs -> resource_id
    emails = EMAIL_RE.findall(text)
    users = USER_RE.findall(text)
    found = {}
    for rid in set(RESOURCE_RE.findall(text)):
        found[rid] = {"resource_id": rid}
    for alias, (rid, system) in aliases.items():
        if alias in text:
            found.setdefault(rid, {"resource_id": rid})
            found[rid]["visible_reference"] = alias
            found[rid]["_system_hint"] = system
    # if a LINK row already named the owner, attach that email/user_id to the resource
    link_owners = link_owners or {}
    out = []
    for rid, fields in found.items():
        owners = link_owners.get(rid, {})
        if owners.get("email"):
            fields["email"] = owners["email"]
        if owners.get("user_id"):
            fields["user_id"] = owners["user_id"]
        # single-subject docs (support attachment): keep extracted contact fields
        if len(emails) == 1 and "email" not in fields:
            fields["email"] = emails[0]
        if len(users) == 1 and "user_id" not in fields:
            fields["user_id"] = users[0]
        # PDFs under documents/ with a support_* id are attachments to delete
        if rid.startswith("support_") and meta["file_path"].startswith("evidence/documents/"):
            fields["record_type"] = "SUPPORT_ATTACHMENT"
        system = _system_for(rid, fields.pop("_system_hint", None))
        out.append(_observation(meta, rid, system, fields))
    if not out and (emails or users):
        # no resource id printed — still keep contacts so identity search can find the file
        fields = {}
        if len(emails) == 1:
            fields["email"] = emails[0]
        if len(users) == 1:
            fields["user_id"] = users[0]
        out.append(_observation(meta, "", meta.get("origin"), fields))
    return out


def _parse_pdf(path, meta, aliases, link_owners=None):
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return _document_observations(meta, text, aliases, link_owners)


def get_ocr_engine():
    # single shared PaddleOCR instance (model load is expensive)
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_engine


def _extract_image_text(abs_path):
    results = get_ocr_engine().predict(abs_path)
    lines = []
    for res in results or []:
        texts = res.get("rec_texts") if isinstance(res, dict) else getattr(res, "rec_texts", None)
        if texts is None and hasattr(res, "get"):
            texts = res.get("rec_texts")
        if texts:
            lines.extend(str(t) for t in texts if t)
    return "\n".join(lines)


def _parse_image(path, meta, aliases, link_owners=None):
    # OCR is optional: a broken/missing paddleocr install must not stop the other 83 files
    try:
        text = _extract_image_text(path)
    except Exception as exc:
        warnings.warn(f"{meta['file_id']}: OCR unavailable, screenshot skipped ({type(exc).__name__})")
        return []
    return _document_observations(meta, text, aliases, link_owners)


# --- build / query -------------------------------------------------------

def build_link_aliases():
    # "linked export_1842 (NST-2026-0718-1842) to ..." -> alias map + owner email/user_id per resource
    aliases = {}
    owners = {}
    links = system_activity[system_activity["event_type"] == "LINK_RESOURCE_IDENTITY"]
    for _, row in links.iterrows():
        details = str(row["details"])
        m = re.search(r"\(([^)]+)\)", details)
        if m and m.group(1) != row["resource_id"]:
            aliases[m.group(1)] = (row["resource_id"], row["target_system"])
        email = (EMAIL_RE.findall(details) or [None])[0]
        user = (USER_RE.findall(details) or [None])[0]
        owners[row["resource_id"]] = {
            "email": email.lower() if email else None,
            "user_id": user.lower() if user else None,
            "system": row["target_system"],
        }
    return aliases, owners


def build_observations():
    aliases, link_owners = build_link_aliases()
    observations = []
    scopes = []  # which systems each file covers (used to prove absence)
    for _, meta in evidence_files.iterrows():
        path = os.path.join(dataset_path, meta["file_path"])
        ext = os.path.splitext(path)[1].lower()
        if ext == ".sql":
            found = _parse_sql(path, meta)
        elif ext == ".json":
            found = _parse_json(path, meta)
        elif ext == ".log":
            found = _parse_log(path, meta)
        elif ext == ".txt":
            found = _parse_txt(path, meta)
        elif ext == ".pdf":
            found = _parse_pdf(path, meta, aliases, link_owners)
        elif ext == ".png":
            found = _parse_image(path, meta, aliases, link_owners)
        else:
            found = []
        for system in {o["system"] for o in found if o["system"]}:
            scopes.append({"file_id": meta["file_id"], "captured_at": meta["captured_at"], "system": system})
        observations.extend(o for o in found if not o["fields"].get("scope_only"))
    return observations, scopes


def _load():
    global _observations, _scopes, _by_resource
    if _observations is None:
        if os.path.exists(CACHE_PATH):
            cached = json.load(open(CACHE_PATH, encoding="utf-8"))
            _observations, _scopes = cached["observations"], cached["scopes"]
        else:
            _observations, _scopes = build_observations()
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            json.dump({"observations": _observations, "scopes": _scopes},
                      open(CACHE_PATH, "w", encoding="utf-8"), default=str)
        _by_resource = {}
        for obs in _observations:
            if obs["resource_id"]:
                _by_resource.setdefault(obs["resource_id"], []).append(obs)


def get_observations():
    _load()
    return _observations


def get_scopes():
    _load()
    return _scopes


def _token_pattern(tokens):
    # whole-token match so user_18 does not hit user_180 and a.rao@x.com does not hit ma.rao@x.com
    alternation = "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return re.compile(rf"(?<![\w.+-])(?:{alternation})(?![\w.+-])")


def find_observations(identity_profile, audit_at):
    # every observation that still shows this person at/before audit_at
    if not identity_profile:
        return []
    audit_ts = pd.Timestamp(audit_at)
    pattern = _token_pattern(identity_profile["tokens"])
    phones = identity_profile["phone_digits"]
    matched = []
    for obs in get_observations():
        if pd.Timestamp(obs["captured_at"]) > audit_ts or pd.Timestamp(obs["observed_at"]) > audit_ts:
            continue
        hits = sorted(set(pattern.findall(obs["text_l"])))
        hits += [p for p in phones if p in obs["digits"]]
        if hits:
            matched.append({**obs, "matched_identifiers": hits})
    return matched


def latest_for_resource(resource_id, audit_at):
    # newest observation for this resource_id at/before audit_at (even with no identity match)
    _load()
    audit_ts = pd.Timestamp(audit_at)
    rows = [
        o for o in _by_resource.get(resource_id, [])
        if pd.Timestamp(o["captured_at"]) <= audit_ts and pd.Timestamp(o["observed_at"]) <= audit_ts
    ]
    if not rows:
        return None
    return max(rows, key=lambda o: (o["observed_at"], o["captured_at"]))


def find_scopes(system, audit_at):
    # files that checked this system at/before audit_at (needed to prove a record is ABSENT)
    audit_ts = pd.Timestamp(audit_at)
    return [s for s in get_scopes() if s["system"] == system and pd.Timestamp(s["captured_at"]) <= audit_ts]
