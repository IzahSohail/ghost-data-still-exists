from datetime import datetime
import json
import os
import re
import sys
import unicodedata

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google import genai
from paddleocr import PaddleOCR


load_dotenv()

MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# dataset lives one level up from code/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(REPO_ROOT, 'dataset')

# read deletion_requests.csv
deletion_requests = pd.read_csv(os.path.join(dataset_path, 'deletion_requests.csv'))

# read user_identities.csv
user_identities = pd.read_csv(os.path.join(dataset_path, 'user_identities.csv'))

# read user_audit_log.csv
user_audit_log = pd.read_csv(os.path.join(dataset_path, 'user_audit_log.csv'))

# evidence manifest (file_id <-> path/captured_at)
evidence_files = pd.read_csv(os.path.join(dataset_path, 'evidence_files.csv'))

# Built once: OCR each image, then every case just does string matching.
_image_text_index = None
_ocr_engine = None

def judge_request_validity(request_text):
    prompt = (
        "Classify this deletion request. Reply with ONLY YES or NO.\n"
        "YES = The user requests account deletion, personal-data deletion, erasure," 
        "removal, or the equivalent in any language.\n"
        "NO = the person is requesting access, correction, export, portability, marketing opt-out, "
        "or any other non-deletion request.\n\n"
        f"Request:\n{request_text}"
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    answer = (response.text or "").strip().upper()
    return answer.startswith("YES")

def identity_resolution(email):
    email_l = str(email).strip().lower()

    # 1) current snapshot match
    match = user_identities[user_identities["email"].str.lower() == email_l]
    user_id = match.iloc[0]["user_id"] if not match.empty else None

    # 2) request email may only appear in audit history
    if user_id is None:
        email_changes = user_audit_log[user_audit_log["field_name"] == "email"]
        for _, row in email_changes.iterrows():
            old = str(row["old_value"] or "").lower()
            new = str(row["new_value"] or "").lower()
            if email_l in (old, new):
                user_id = row["user_id"]
                break

    if user_id is None:
        return None

    profile = user_identities[user_identities["user_id"] == user_id].iloc[0]
    emails = {profile["email"]}
    phones = {profile["phone"]}
    email_hashes = {profile["email_hash"]}

    history = user_audit_log[user_audit_log["user_id"] == user_id]
    for _, row in history.iterrows():
        for value in (row["old_value"], row["new_value"]):
            if pd.isna(value) or value == "":
                continue
            if row["field_name"] == "email":
                emails.add(value)
            elif row["field_name"] == "phone":
                phones.add(value)
            elif row["field_name"] == "email_hash":
                email_hashes.add(value)

    return {
        "user_id": user_id,
        "name": profile["name"],
        "emails": list(emails),
        "phones": list(phones),
        "email_hashes": list(email_hashes),
    }

def _phone_digits(value):
    return re.sub(r"\D+", "", str(value or ""))


def get_ocr_engine():
    """Single shared PaddleOCR instance (model load is expensive)."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_engine


def _extract_image_text(abs_path):
    """OCR one image; used only while building the image index."""
    results = get_ocr_engine().predict(abs_path)
    lines = []
    for res in results or []:
        texts = None
        if isinstance(res, dict):
            texts = res.get("rec_texts")
        else:
            texts = getattr(res, "rec_texts", None)
            if texts is None and hasattr(res, "get"):
                texts = res.get("rec_texts")
        if texts:
            lines.extend(str(t) for t in texts if t)
    return "\n".join(lines)


def build_image_text_index():
    """OCR each image once. Returns list of searchable evidence records."""
    rows = evidence_files[evidence_files["file_path"].str.startswith("evidence/images/")]
    index = []
    for _, row in rows.iterrows():
        abs_path = os.path.join(dataset_path, row["file_path"])
        text = _extract_image_text(abs_path)
        index.append(
            {
                "file_id": row["file_id"],
                "file_path": row["file_path"],
                "origin": row["origin"],
                "captured_at": row["captured_at"],
                "text": text,
                "text_l": text.lower(),
                "digits": _phone_digits(text),
            }
        )
    return index


def get_image_text_index():
    global _image_text_index
    if _image_text_index is None:
        _image_text_index = build_image_text_index()
    return _image_text_index


def verify_decision(identity_profile, audit_at, running_evidence=None):
    """
    Collect evidence that still shows this user at/before audit_at.
    Step 1 (current): image screenshots only.
    Returns the running evidence list for later LLM + deletion_rules judgment.
    """
    if running_evidence is None:
        running_evidence = []
    if not identity_profile:
        return running_evidence

    audit_ts = pd.Timestamp(audit_at)

    # identifiers to hunt for in extracted screenshot text
    needles = []
    needles.append(("user_id", identity_profile["user_id"]))
    needles.append(("name", identity_profile["name"]))
    for email in identity_profile.get("emails", []):
        needles.append(("email", email))
    for phone in identity_profile.get("phones", []):
        needles.append(("phone", phone))
    for h in identity_profile.get("email_hashes", []):
        needles.append(("email_hash", h))

    for item in get_image_text_index():
        if pd.Timestamp(item["captured_at"]) > audit_ts:
            continue

        matched = []
        for kind, value in needles:
            if not value or (isinstance(value, float) and pd.isna(value)):
                continue
            if kind == "phone":
                digits = _phone_digits(value)
                if len(digits) >= 7 and digits in item["digits"]:
                    matched.append({"kind": kind, "value": value})
            else:
                if str(value).lower() in item["text_l"]:
                    matched.append({"kind": kind, "value": value})

        if matched:
            running_evidence.append(
                {
                    "source": "image",
                    "file_id": item["file_id"],
                    "file_path": item["file_path"],
                    "origin": item["origin"],
                    "captured_at": item["captured_at"],
                    "matched_identifiers": matched,
                    "text": item["text"],
                }
            )

    return running_evidence

# for every record in deletion_requests, use "request_text" field to send to llm to determine if this request qualifies as a deletion request
for record in deletion_requests:
    case_id = record["case_id"]
    request_text = record["request_text"]
    email = record["email"]
    requested_at = record["requested_at"]
    status = record["reported_status"]
    reported_at = record["reported_at"]
    audit_at = record["audit_at"]

    # judge request validity with llm 
    is_valid = judge_request_validity(request_text)

    # if request is not valid, prepare json output for this record
    if not is_valid:
        output = {
            case_id: case_id,
            deletion_status: "NOT_APPLICABLE",
            steps_json: [],
        }

    # if request is valid, prepare json output for this record
    if is_valid:
        identity_profile = identity_resolution(email)
        running_evidence = verify_decision(identity_profile, audit_at)
