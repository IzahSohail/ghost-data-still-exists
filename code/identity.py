import hashlib
import os
import re

import pandas as pd


# dataset lives one level up from code/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(REPO_ROOT, 'dataset')

# read user_identities.csv
user_identities = pd.read_csv(os.path.join(dataset_path, 'user_identities.csv'))

# read user_audit_log.csv
user_audit_log = pd.read_csv(os.path.join(dataset_path, 'user_audit_log.csv'))


def _phone_digits(value):
    return re.sub(r"\D+", "", str(value or ""))


def _clean(value):
    return None if pd.isna(value) or value == "" else str(value).strip()


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
    emails = {profile["email"].lower()}
    phones = {profile["phone"]}
    email_hashes = {profile["email_hash"]}

    history = user_audit_log[user_audit_log["user_id"] == user_id]
    for _, row in history.iterrows():
        for value in (_clean(row["old_value"]), _clean(row["new_value"])):
            if value is None:
                continue
            if row["field_name"] == "email":
                emails.add(value.lower())
            elif row["field_name"] == "phone":
                phones.add(value)
            elif row["field_name"] == "email_hash":
                email_hashes.add(value)

    # old emails may only survive as their hash in downstream systems
    email_hashes |= {hashlib.sha256(e.encode()).hexdigest() for e in emails}

    # tokens are lowercase substrings to hunt for in evidence; phones are matched on digits only
    tokens = {user_id.lower()} | emails | {h.lower() for h in email_hashes}
    phone_digits = {_phone_digits(p) for p in phones if len(_phone_digits(p)) >= 7}

    return {
        "user_id": user_id,
        "name": profile["name"],
        "emails": sorted(emails),
        "phones": sorted(phones),
        "email_hashes": sorted(email_hashes),
        "tokens": sorted(tokens),
        "phone_digits": sorted(phone_digits),
    }
