"""
Synthetic data generator for DSAR episodes.

Case 1 and Case 2 episodes are generated deterministically from a single seed so
observations, hidden state, and grader ground truth remain in sync.
"""

from __future__ import annotations

import random
import re
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    ACCOUNT_MANAGER_NOTES,
    ACCOUNT_REFERENCE_PREFIXES,
    BILLING_DISPUTE_REASONS,
    CANCELLATION_REASONS,
    CASE1_DSAR_TEMPLATE,
    CASE3_ACTION_DISCLOSE,
    CASE3_ACTION_ESCALATE,
    CASE3_ACTION_EXCLUDE,
    CASE3_ACTION_PARTIAL_REDACT,
    CASE3_DSAR_TEMPLATE,
    CASE3_DISTRACTOR_TEMPLATES,
    CASE3_INTERNAL_HR_CODES,
    BOT_TEMPLATES,
    COLLEAGUE_NAMES,
    CASE2_DSAR_TEMPLATE,
    CASE2_PROPORTIONATE_METHODS,
    CASE2_SENTENCE_LABEL_INTERNAL,
    CASE2_SENTENCE_LABEL_PII,
    CASE2_SENTENCE_LABEL_REQUESTER,
    CASE2_VERIFICATION_THRESHOLD,
    CITIES_WITH_POSTCODES,
    EMPLOYEE_NAMES,
    EMAIL_DOMAINS,
    FIELD_GROUND_TRUTH,
    FIELD_METADATA,
    HEALTH_CONDITIONS,
    HEALTH_TRAP_TEMPLATES,
    LEAD_SOURCE_TAGS,
    MANAGER_NAMES,
    MANAGER_PERF_TEMPLATES,
    MARKETING_PREFERENCES,
    PIPELINE_NAMES,
    PRODUCT_NAMES,
    PR_PHRASES,
    PROJECT_NAMES,
    REFERRAL_CREDIT_BALANCES,
    REQUESTER_NAMES,
    REQUESTER_CLEAN_TEMPLATES,
    SALARY_PHRASES,
    SHARD_ROUTING_KEYS,
    SUBSCRIPTION_PLANS,
    SUPPORT_AGENT_NAMES,
    SUPPORT_PHONE_NUMBERS,
    TECH_SUPPORT_ISSUES,
    THREAD_REPLY_TEMPLATES,
    WORK_EMAIL_DOMAINS,
)


def _make_email(name: str, rng: random.Random) -> str:
    parts = name.lower().replace("'", "").replace("-", " ").split()
    return f"{'.'.join(parts)}@{rng.choice(EMAIL_DOMAINS)}"


def _make_work_email(name: str, rng: random.Random) -> str:
    parts = name.lower().replace("'", "").replace("-", " ").split()
    first = parts[0]
    last = parts[-1]
    patterns = [
        f"{first[0]}.{last}",
        f"{first}.{last}",
        f"{first[0]}{last}",
        f"{last}.{first[0]}",
    ]
    return f"{rng.choice(patterns)}@{rng.choice(WORK_EMAIL_DOMAINS)}"


def _make_payment_history(rng: random.Random) -> List[Dict[str, Any]]:
    count = rng.randint(2, 4)
    base_date = date(2023, 1, 1)
    entries = []
    for index in range(count):
        pay_date = base_date + timedelta(days=30 * index + rng.randint(0, 5))
        amount = round(rng.choice([9.99, 19.99, 29.99, 49.99, 79.99]), 2)
        entries.append({"date": pay_date.isoformat(), "amount": amount})
    return entries


def _make_usage_summary(rng: random.Random) -> str:
    templates = [
        "logged_in_{n}_times_last_month",
        "used_export_feature_{n}_times",
        "created_{n}_projects",
        "invited_{n}_team_members",
        "generated_{n}_reports",
    ]
    return rng.choice(templates).format(n=rng.randint(2, 50))


def _make_subscription_start(rng: random.Random) -> str:
    start = date(2021, 1, 1)
    days_range = (date(2023, 12, 31) - start).days
    return (start + timedelta(days=rng.randint(0, days_range))).isoformat()


def _make_request_date(rng: random.Random) -> str:
    start = date(2026, 1, 1)
    days_range = 119
    return (start + timedelta(days=rng.randint(0, days_range))).isoformat()


def _build_field_item(field_id: str, value: Any) -> Dict[str, Any]:
    metadata = FIELD_METADATA.get(field_id)
    if metadata is None:
        return {
            "field_id": field_id,
            "field_name": field_id.replace("_", " ").title(),
            "field_value": value,
            "source_silo": "unknown",
            "datatype": "unknown",
            "field_description": "No description available.",
        }

    field_name, source_silo, datatype, description = metadata
    return {
        "field_id": field_id,
        "field_name": field_name,
        "field_value": value,
        "source_silo": source_silo,
        "datatype": datatype,
        "field_description": description,
    }


def _build_structured_case_values(
    full_name: str,
    email: str,
    billing_address: str,
    support_ticket_ids: List[str],
    rng: random.Random,
) -> Dict[str, Any]:
    values = {
        "full_name": full_name,
        "email": email,
        "billing_address": billing_address,
        "subscription_plan": rng.choice(SUBSCRIPTION_PLANS),
        "subscription_start_date": _make_subscription_start(rng),
        "payment_history": _make_payment_history(rng),
        "referral_credit_balance": round(rng.choice(REFERRAL_CREDIT_BALANCES), 2),
        "marketing_preferences": rng.choice(MARKETING_PREFERENCES),
        "product_usage_summary": _make_usage_summary(rng),
        "support_ticket_ids": support_ticket_ids,
        "customer_health_score": round(rng.uniform(0.0, 100.0), 1),
        "risk_score": round(rng.uniform(0.0, 1.0), 2),
        "churn_probability": round(rng.uniform(0.0, 1.0), 2),
        "lead_source_tag": rng.choice(LEAD_SOURCE_TAGS),
        "shard_routing_key": rng.choice(SHARD_ROUTING_KEYS),
        "account_manager_notes": rng.choice(ACCOUNT_MANAGER_NOTES),
        "campaign_cpa": round(rng.uniform(8.0, 45.0), 2),
    }
    return values


def _build_customer_record(raw_values: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    customer_record = [_build_field_item(field_id, value) for field_id, value in raw_values.items()]
    rng.shuffle(customer_record)
    return customer_record


def _variant_postcode(postcode: str, rng: random.Random) -> str:
    variants = [postcode, postcode.replace(" ", ""), postcode.lower(), postcode.upper()]
    return rng.choice(variants)


def _mask_email(email: str) -> str:
    local, domain = email.split("@", 1)
    masked_local = "*" if len(local) <= 1 else local[0] + ("*" * max(4, len(local) - 1))
    return f"{masked_local}@{domain}"


def _mask_address(address: str) -> str:
    parts = [part.strip() for part in address.split(",")]
    if len(parts) < 3:
        return address
    street, city, postcode = parts[0], parts[1], parts[2]
    street_num = street.split()[0]
    masked_postcode = postcode[:3] + "***" if len(postcode) >= 3 else "***"
    return f"{street_num} ****, {city}, {masked_postcode}"


def _normalize_name(name: str) -> str:
    return " ".join(part for part in name.lower().replace(".", "").split() if len(part) > 1)


def _compute_identity_confidence(
    submitted_identity: Dict[str, Any],
    internal_identity_full: Dict[str, Any],
    rng: random.Random,
) -> float:
    submitted_name = _normalize_name(submitted_identity["full_name"])
    internal_name = _normalize_name(internal_identity_full["full_name"])
    submitted_address = submitted_identity["billing_address"]
    internal_address = internal_identity_full["billing_address"]

    submitted_city = submitted_address.split(",")[1].strip()
    internal_city = internal_address.split(",")[1].strip()
    submitted_street = submitted_address.split(",")[0].strip()
    internal_street = internal_address.split(",")[0].strip()
    submitted_postcode = submitted_address.split(",")[2].strip().replace(" ", "").lower()
    internal_postcode = internal_address.split(",")[2].strip().replace(" ", "").lower()

    score = 0.05
    if submitted_name.split()[0] == internal_name.split()[0]:
        score += 0.10
    if submitted_name.split()[-1] == internal_name.split()[-1]:
        score += 0.15
    if submitted_city == internal_city:
        score += 0.08
    if submitted_street.split()[0] == internal_street.split()[0]:
        score += 0.06
    if submitted_postcode[:3] == internal_postcode[:3]:
        score += 0.05
    score += rng.uniform(-0.02, 0.02)
    return round(min(0.55, max(0.35, score)), 2)


def _make_support_email(agent_name: str) -> str:
    parts = agent_name.lower().split()
    return f"{parts[0]}.{parts[-1]}@support.techcorp.com"


def _customer_name_variation(base_name: str, rng: random.Random) -> str:
    parts = base_name.split()
    first = parts[0]
    last = parts[-1]
    normalized_last = last.replace("'", "")
    variants = [
        base_name,
        f"{first} {rng.choice(['A.', 'B.', 'C.', 'R.', 'S.'])} {last}",
        f"{first[0]}. {last}",
        f"{first}-{last}",
        f"{first} {normalized_last}",
        f"{first[0]}. {normalized_last}",
    ]
    return rng.choice(variants)


def _submitted_email_variation(internal_email: str, name: str, rng: random.Random) -> str:
    local_part, _ = internal_email.split("@", 1)
    parts = name.lower().replace("'", "").replace("-", " ").split()
    first = parts[0]
    last = parts[-1]
    normalized_last = last.replace("'", "")
    local_variants = [
        local_part,
        f"{first}.{last}",
        f"{first[0]}.{last}",
        f"{first[0]}{normalized_last}",
        f"{normalized_last}.{first[0]}",
    ]
    return f"{rng.choice(local_variants)}@{rng.choice(EMAIL_DOMAINS)}"


def _near_miss_street_variation(street: str, rng: random.Random) -> str:
    parts = street.split()
    if not parts:
        return street
    try:
        number = int(parts[0])
    except ValueError:
        return street

    street_body = " ".join(parts[1:])
    variants = [
        street,
        f"{number + 1} {street_body}",
        f"{max(1, number - 1)} {street_body}",
        street.replace(" St", " Street"),
        street.replace(" Street", " St"),
    ]
    return rng.choice(variants)


def _address_variation(street: str, city: str, postcode: str, rng: random.Random, family: str) -> str:
    postcode_variant = _variant_postcode(postcode, rng)
    street_variant = _near_miss_street_variation(street, rng)
    if family == "registered_postcode":
        return f"{street_variant}, {city}, {postcode_variant}"
    if family == "account_reference":
        return f"{street_variant} Apt {rng.choice([2, 3, 5, 7, 9, 11, 12])}, {city}, {postcode_variant}"
    if family == "transaction_date":
        return f"{street_variant}, {city}, {postcode_variant}"
    return f"{street_variant}, {city}, {postcode_variant}"


def _pick_identity_family(rng: random.Random) -> str:
    return rng.choice(sorted(CASE2_PROPORTIONATE_METHODS))


def _pick_competing_identity_family(correct_family: str, rng: random.Random) -> str:
    candidates = sorted(CASE2_PROPORTIONATE_METHODS - {correct_family})
    return rng.choice(candidates)


def _identity_support_level(
    method: str,
    *,
    correct_family: str,
    competing_family: str,
) -> str:
    if method == correct_family:
        return "strong"
    if method == competing_family:
        return "plausible"
    return "background"


def _pick_levelled_note(
    options_by_level: Dict[str, List[str]],
    level: str,
    rng: random.Random,
) -> str:
    return rng.choice(options_by_level[level])


def _build_identity_scenario(
    base_name: str,
    city: str,
    street: str,
    postcode: str,
    rng: random.Random,
) -> Dict[str, Any]:
    family = _pick_identity_family(rng)
    competing_family = _pick_competing_identity_family(family, rng)
    internal_full_name = base_name
    submitted_full_name = _customer_name_variation(base_name, rng)
    internal_email = _make_work_email(internal_full_name, rng)
    submitted_email = _make_email(internal_full_name, rng)
    internal_billing_address = f"{street}, {city}, {postcode}"
    submitted_billing_address = _address_variation(street, city, postcode, rng, family)
    account_reference = f"{rng.choice(ACCOUNT_REFERENCE_PREFIXES)}-{rng.randint(100000, 999999)}"
    payment_history = _make_payment_history(rng)
    recent_transaction_date = payment_history[-1]["date"]
    renewal_window = recent_transaction_date[:7]
    postcode_prefix = postcode.replace(" ", "")[:3]
    account_prefix = account_reference[:4]

    submitted_identity = {
        "full_name": submitted_full_name,
        "email": submitted_email,
        "billing_address": submitted_billing_address,
    }
    internal_identity_full = {
        "full_name": internal_full_name,
        "email": internal_email,
        "billing_address": internal_billing_address,
        "registered_postcode": postcode,
        "account_reference": account_reference,
        "recent_transaction_date": recent_transaction_date,
    }

    account_reference_level = _identity_support_level(
        "account_reference",
        correct_family=family,
        competing_family=competing_family,
    )
    transaction_date_level = _identity_support_level(
        "transaction_date",
        correct_family=family,
        competing_family=competing_family,
    )
    registered_postcode_level = _identity_support_level(
        "registered_postcode",
        correct_family=family,
        competing_family=competing_family,
    )

    billing_review_note = _pick_levelled_note(
        {
            "strong": [
                (
                    f"The billing review thread says the requester previously quoted the "
                    f"account marker {account_prefix} while checking service continuity."
                ),
                (
                    f"The active account-status review still matches the historical billing "
                    f"marker {account_prefix} the requester had used before."
                ),
            ],
            "plausible": [
                (
                    f"The billing reconciliation path still keeps the historical account "
                    f"marker {account_prefix} in view for this case."
                ),
                (
                    f"A prior billing review note leaves the account marker {account_prefix} "
                    f"visible beside the current renewal discussion."
                ),
            ],
            "background": [
                (
                    f"Archived billing material still shows the account marker {account_prefix} "
                    f"alongside the current subscription context."
                ),
                (
                    f"The older billing entity continues to carry marker {account_prefix} in "
                    f"the record summary."
                ),
            ],
        },
        account_reference_level,
        rng,
    )
    renewal_timing_note = _pick_levelled_note(
        {
            "strong": [
                (
                    f"The renewal window under review still lines up with the {renewal_window} "
                    f"billing event the requester referred to."
                ),
                (
                    f"The billing event summary keeps pointing back to the {renewal_window} "
                    f"renewal timing raised during the service continuity review."
                ),
            ],
            "plausible": [
                (
                    f"The current account-status review still sits in the {renewal_window} "
                    f"renewal window for reconciliation."
                ),
                (
                    f"The billing event summary keeps the {renewal_window} renewal period in "
                    f"scope while the case is reviewed."
                ),
            ],
            "background": [
                (
                    f"A historical billing summary still lists activity around the "
                    f"{renewal_window} renewal window."
                ),
                (
                    f"The archived renewal view keeps {renewal_window} visible in the "
                    f"account timeline."
                ),
            ],
        },
        transaction_date_level,
        rng,
    )
    workspace_location_note = _pick_levelled_note(
        {
            "strong": [
                (
                    f"The workspace-state review still anchors the account to {city} with "
                    f"postcode prefix {postcode_prefix}."
                ),
                (
                    f"The managed review state still shows the same {city} billing location "
                    f"and postcode prefix {postcode_prefix} on the workspace record."
                ),
            ],
            "plausible": [
                (
                    f"The workspace-state note still points to {city} and postcode prefix "
                    f"{postcode_prefix} for the old setup."
                ),
                (
                    f"The review team still sees the same {city} location trail with postcode "
                    f"prefix {postcode_prefix} on the workspace history."
                ),
            ],
            "background": [
                (
                    f"Older workspace material still references {city} with postcode prefix "
                    f"{postcode_prefix}."
                ),
                (
                    f"The historical workspace snapshot preserves the {city} location marker "
                    f"and postcode prefix {postcode_prefix}."
                ),
            ],
        },
        registered_postcode_level,
        rng,
    )
    workspace_ownership_notes: Dict[str, Dict[str, List[str]]] = {
        "account_reference": {
            "strong": [
                (
                    f"Old-workspace ownership notes say the requester previously controlled "
                    f"the team workspace tied to billing marker {account_prefix}."
                ),
                (
                    f"The prior workspace owner trail still lines up with the billing marker "
                    f"{account_prefix} used on the account."
                ),
            ],
            "plausible": [
                (
                    f"Workspace ownership history still keeps billing marker {account_prefix} "
                    f"near the managed review state."
                ),
                (
                    f"The old workspace record still carries billing marker {account_prefix} "
                    f"next to the current review path."
                ),
            ],
            "background": [
                "Old-workspace ownership notes still reflect a prior business-admin handoff.",
                "The older workspace trail still shows a historical ownership change.",
            ],
        },
        "transaction_date": {
            "strong": [
                (
                    f"Old-workspace ownership notes say the requester lost the former work "
                    f"inbox just after the {renewal_window} renewal window."
                ),
                (
                    f"The workspace ownership trail still ties the access break to the "
                    f"{renewal_window} renewal period."
                ),
            ],
            "plausible": [
                (
                    f"The old-workspace notes still keep the {renewal_window} renewal window "
                    f"in view during the review."
                ),
                (
                    f"Workspace ownership history still references the {renewal_window} review "
                    f"window on the old team setup."
                ),
            ],
            "background": [
                "The old workspace still shows a previous handoff between business admins.",
                "Workspace ownership history still mentions a former admin transition.",
            ],
        },
        "registered_postcode": {
            "strong": [
                (
                    f"The old-workspace ownership note still maps the prior owner to the "
                    f"{city} location trail used on the billing profile."
                ),
                (
                    f"Workspace ownership history still follows the same {city} location "
                    f"marker used on the current account."
                ),
            ],
            "plausible": [
                (
                    f"The previous workspace owner trail still carries the {city} location "
                    f"context through the review."
                ),
                (
                    f"The old workspace remains tied to the same {city} location summary in "
                    f"the managed review state."
                ),
            ],
            "background": [
                "The old workspace still shows a historical ownership change.",
                "Workspace ownership history still records a prior admin handoff.",
            ],
        },
    }
    ownership_method = family if family != "registered_postcode" else competing_family
    workspace_ownership_level = _identity_support_level(
        ownership_method,
        correct_family=family,
        competing_family=competing_family,
    )
    workspace_ownership_note = _pick_levelled_note(
        workspace_ownership_notes[ownership_method],
        workspace_ownership_level,
        rng,
    )

    masked_billing = {
        "full_name": internal_full_name,
        "billing_address": _mask_address(internal_billing_address),
        "subscription_plan_hint": rng.choice(SUBSCRIPTION_PLANS),
        "billing_review_note": billing_review_note,
        "billing_event_summary": renewal_timing_note,
    }
    masked_crm = {
        "full_name": internal_full_name,
        "email": _mask_email(internal_email),
        "known_name_variant": submitted_full_name,
        "workspace_location_note": workspace_location_note,
        "workspace_ownership_note": workspace_ownership_note,
    }

    support_by_method = {
        "account_reference": {
            "billing": account_reference_level,
            "crm": (
                workspace_ownership_level
                if ownership_method == "account_reference"
                else "background"
            ),
        },
        "transaction_date": {
            "billing": transaction_date_level,
            "crm": (
                workspace_ownership_level
                if ownership_method == "transaction_date"
                else "background"
            ),
        },
        "registered_postcode": {
            "billing": registered_postcode_level,
            "crm": registered_postcode_level,
        },
    }

    return {
        "correct_verification_method": family,
        "competing_verification_method": competing_family,
        "submitted_identity": submitted_identity,
        "internal_identity_full": internal_identity_full,
        "internal_identity_masked": {"billing": masked_billing, "crm": masked_crm},
        "starting_identity_confidence": _compute_identity_confidence(
            submitted_identity=submitted_identity,
            internal_identity_full=internal_identity_full,
            rng=rng,
        ),
        "recent_transaction_date": recent_transaction_date,
        "account_reference": account_reference,
        "identity_ambiguity": {
            "plausible_methods": [family, competing_family],
            "support_by_method": support_by_method,
        },
    }


def _build_sentence_chunk(
    kind: str,
    *entries: Tuple[str, str],
    pair_id: str | None = None,
) -> Dict[str, Any]:
    chunk = {"kind": kind, "entries": list(entries)}
    if pair_id is not None:
        chunk["pair_id"] = pair_id
    return chunk


def _shuffle_sentence_chunks(
    sentence_chunks: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    shuffled = list(sentence_chunks)
    rng.shuffle(shuffled)
    return shuffled


def _build_ticket_message(
    speaker: str,
    message_index: int,
    sentence_chunks: List[Dict[str, Any]],
    next_sentence_index: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[int, str], int, List[Dict[str, Any]]]:
    message_ground_truth: Dict[int, str] = {}
    sentences = []
    sentence_texts = []
    chunk_metadata: List[Dict[str, Any]] = []
    ordered_chunks = (
        _shuffle_sentence_chunks(sentence_chunks, rng)
        if speaker == "support"
        else list(sentence_chunks)
    )

    for chunk_index, chunk in enumerate(ordered_chunks):
        chunk_metadata.append(
            {
                "speaker": speaker,
                "message_index": message_index,
                "chunk_index": chunk_index,
                "kind": chunk["kind"],
                "size": len(chunk["entries"]),
                "pair_id": chunk.get("pair_id"),
            }
        )
        for label, text in chunk["entries"]:
            sentences.append(
                {
                    "sentence_index": next_sentence_index,
                    "speaker": speaker,
                    "text": text,
                }
            )
            message_ground_truth[next_sentence_index] = label
            sentence_texts.append(text)
            next_sentence_index += 1

    return {
        "message_index": message_index,
        "speaker": speaker,
        "text": " ".join(sentence_texts),
        "sentences": sentences,
    }, message_ground_truth, next_sentence_index, chunk_metadata


def _pick(ticket_pool: List[str], rng: random.Random) -> str:
    return rng.choice(ticket_pool)


def _technical_support_message_specs(requester_name: str, rng: random.Random) -> List[Tuple[str, List[Dict[str, Any]]]]:
    agent_name = rng.choice(SUPPORT_AGENT_NAMES)
    issue = rng.choice(TECH_SUPPORT_ISSUES)
    product = rng.choice(PRODUCT_NAMES)
    phone = rng.choice(SUPPORT_PHONE_NUMBERS)

    customer_openers = [
        f"I am unable to {issue} in {product}.",
        f"I keep hitting an error whenever I try to {issue}.",
        f"The issue happens every time I attempt to {issue} in {product}.",
    ]
    customer_details = [
        "The issue started after yesterday's update.",
        "This only started happening this week.",
        "It worked before, so I think something changed recently.",
    ]
    support_requester_pairs = [
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    f"The team review of the workspace state is still tied to the email "
                    f"mismatch showing on the {product} account record tied to this support request."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "That review is about restoring your access path and service continuity "
                    "for the affected workspace while the current state is reconciled."
                ),
            ),
        ),
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    f"The account-status review still follows the same service reference on "
                    f"the {product} workspace that remains attached to this customer issue."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "It is checking whether the current workspace state can continue under the "
                    "same access path while the failure is reviewed."
                ),
            ),
        ),
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    f"The current review with the platform team still treats your workspace "
                    f"state as active even though the email path for {issue} keeps failing on the account."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "That means your stored workspace content and account status remain "
                    "available while the active path is brought back into a workable state."
                ),
            ),
        ),
    ]
    support_pii = [
        (
            f"The direct number on file for {agent_name} remains {phone} if one reviewer "
            f"needs to stay with this case."
        ),
        (
            f"The current reviewer record still lists {agent_name} on {phone} for any direct "
            f"follow-up."
        ),
        (
            f"The support profile keeps {agent_name} attached to the direct line {phone} for "
            f"continuity."
        ),
    ]
    support_internal = [
        (
            "The case now sits within a managed review window on the service continuation "
            "protocol used for platform reconciliation on the workspace state."
        ),
        (
            "This thread currently remains inside a managed review state on the platform "
            "reconciliation pathway covering the same workspace state."
        ),
        (
            "The workspace issue sits in a managed review window that follows the service "
            "continuation protocol for the active workspace state."
        ),
        (
            "The workspace issue sits in a managed review window that follows the service "
            "continuation protocol for the active workspace state."
        ),
    ]
    chosen_support_requester_pair = _pick(support_requester_pairs, rng)
    return [
        (
            "customer",
            [
                _build_sentence_chunk(
                    "customer_request",
                    (CASE2_SENTENCE_LABEL_REQUESTER, _pick(customer_openers, rng)),
                ),
                _build_sentence_chunk(
                    "customer_context",
                    (CASE2_SENTENCE_LABEL_REQUESTER, _pick(customer_details, rng)),
                ),
            ],
        ),
        (
            "support",
            [
                _build_sentence_chunk(
                    "support_requester_context_pair",
                    *chosen_support_requester_pair,
                ),
                _build_sentence_chunk(
                    "support_pii_contact",
                    (CASE2_SENTENCE_LABEL_PII, _pick(support_pii, rng)),
                ),
                _build_sentence_chunk(
                    "support_internal_process",
                    (CASE2_SENTENCE_LABEL_INTERNAL, _pick(support_internal, rng)),
                ),
            ],
        ),
    ]


def _billing_dispute_message_specs(requester_name: str, rng: random.Random) -> List[Tuple[str, List[Dict[str, Any]]]]:
    agent_name = rng.choice(SUPPORT_AGENT_NAMES)
    support_email = _make_support_email(agent_name)
    reason = rng.choice(BILLING_DISPUTE_REASONS)

    customer_openers = [
        f"I am questioning {reason} on my account.",
        f"I need help understanding {reason} on my latest invoice.",
        f"I am disputing {reason} that appeared on my statement.",
    ]
    customer_details = [
        "I do not think this amount matches what I expected to be billed.",
        "This does not align with the charge I thought I had authorised.",
        "I expected a different billing outcome for this period.",
    ]
    support_requester_pairs = [
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "The billing review is still matching the invoice reference against the "
                    "active renewal window on the account record for this dispute."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "That check is about whether your subscription should receive a credit "
                    "under the current service state while access continues unchanged."
                ),
            ),
        ),
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "The account-status review still treats this as a customer balance issue "
                    "against the same billing reference on the active subscription record."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "If an adjustment is due, it would return to the same account balance "
                    "already attached to the service state under review for this invoice dispute."
                ),
            ),
        ),
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "The review team still sees the disputed invoice under the same service "
                    "reference in the continuity window on the subscription record for this case."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "That means the check only affects the charge outcome and not the rest of "
                    "the subscription history or the current access state."
                ),
            ),
        ),
    ]
    support_pii = [
        (
            f"The direct mailbox on file for {agent_name} remains {support_email} if this "
            f"review has to stay with one person."
        ),
        (
            f"The reviewer record still shows {support_email} for {agent_name} on this "
            f"billing case."
        ),
        (
            f"The current case profile keeps {agent_name} linked to {support_email} for "
            f"direct follow-up."
        ),
    ]
    support_internal = [
        (
            "The account remains inside a managed review state on the reconciliation pathway "
            "used for disputed renewals on the active billing state."
        ),
        (
            "The case still sits within a managed review window under the billing "
            "reconciliation pathway covering the current billing state."
        ),
        (
            "This dispute remains parked in the managed review state used by the billing "
            "reconciliation pathway for the active billing state."
        ),
    ]
    chosen_support_requester_pair = _pick(support_requester_pairs, rng)
    return [
        (
            "customer",
            [
                _build_sentence_chunk(
                    "customer_request",
                    (CASE2_SENTENCE_LABEL_REQUESTER, _pick(customer_openers, rng)),
                ),
                _build_sentence_chunk(
                    "customer_context",
                    (CASE2_SENTENCE_LABEL_REQUESTER, _pick(customer_details, rng)),
                ),
            ],
        ),
        (
            "support",
            [
                _build_sentence_chunk(
                    "support_requester_context_pair",
                    *chosen_support_requester_pair,
                ),
                _build_sentence_chunk(
                    "support_pii_contact",
                    (CASE2_SENTENCE_LABEL_PII, _pick(support_pii, rng)),
                ),
                _build_sentence_chunk(
                    "support_internal_process",
                    (CASE2_SENTENCE_LABEL_INTERNAL, _pick(support_internal, rng)),
                ),
            ],
        ),
    ]


def _cancellation_message_specs(requester_name: str, rng: random.Random) -> List[Tuple[str, List[Dict[str, Any]]]]:
    account_manager = rng.choice(SUPPORT_AGENT_NAMES)
    manager_email = _make_support_email(account_manager)
    manager_phone = rng.choice(SUPPORT_PHONE_NUMBERS)
    reason = rng.choice(CANCELLATION_REASONS)

    customer_openers = [
        f"I would like to cancel because we are {reason}.",
        f"I want to cancel the subscription because we are {reason}.",
        f"We need to cancel as we are {reason}.",
    ]
    customer_details = [
        "Please let me know what happens to my subscription at the end of the current term.",
        "I need to know when access ends once the cancellation is processed.",
        "Please confirm the timing of the cancellation on my current contract.",
    ]
    support_requester_pairs = [
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "The service continuity review still follows the same renewal reference "
                    "on the current contract record for this cancellation request."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "That review only governs when access ends and keeps the current "
                    "workspace state available through the paid term while the contract state is reviewed."
                ),
            ),
        ),
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "The account-status review still shows your cancellation under the same "
                    "renewal reference in the active window on the account record rather than an immediate shutdown path."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "That means the service continues through the paid period and your "
                    "workspace content stays available through that period."
                ),
            ),
        ),
        (
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "The review team is still treating this as a service continuity request "
                    "under the same contract reference on the current subscription record."
                ),
            ),
            (
                CASE2_SENTENCE_LABEL_REQUESTER,
                (
                    "It affects renewal timing on your contract but does not remove your "
                    "existing access before the paid term closes."
                ),
            ),
        ),
    ]
    support_pii_1 = [
        (
            f"The direct phone record for {account_manager} remains {manager_phone} if one "
            f"reviewer has to carry this through."
        ),
        (
            f"The contact profile still lists {account_manager} on {manager_phone} for manual "
            f"follow-up."
        ),
        (
            f"The current service record keeps {account_manager} linked to {manager_phone} for "
            f"continuity."
        ),
    ]
    support_pii_2 = [
        (
            f"The direct mailbox on file for {account_manager} remains {manager_email} for "
            f"follow-up that has to stay with the same reviewer."
        ),
        (
            f"The reviewer profile still ties {account_manager} to {manager_email} on this "
            f"cancellation case."
        ),
        (
            f"The account record keeps {manager_email} as the mailbox attached to "
            f"{account_manager} for continuity."
        ),
    ]
    support_internal = [
        (
            "The case sits within a managed review window on the service continuation "
            "protocol used for retention handling on the active contract state."
        ),
        (
            "This cancellation remains in a managed review state on the retention-side "
            "service continuation pathway covering the current contract state."
        ),
        (
            "The current cancellation thread remains parked in the managed review window "
            "used by the service continuation protocol for the active contract state."
        ),
    ]
    chosen_support_requester_pair = _pick(support_requester_pairs, rng)

    return [
        (
            "customer",
            [
                _build_sentence_chunk(
                    "customer_request",
                    (CASE2_SENTENCE_LABEL_REQUESTER, _pick(customer_openers, rng)),
                ),
                _build_sentence_chunk(
                    "customer_context",
                    (CASE2_SENTENCE_LABEL_REQUESTER, _pick(customer_details, rng)),
                ),
            ],
        ),
        (
            "support",
            [
                _build_sentence_chunk(
                    "support_requester_context_pair",
                    *chosen_support_requester_pair,
                ),
                _build_sentence_chunk(
                    "support_pii_phone",
                    (CASE2_SENTENCE_LABEL_PII, _pick(support_pii_1, rng)),
                ),
                _build_sentence_chunk(
                    "support_pii_email",
                    (CASE2_SENTENCE_LABEL_PII, _pick(support_pii_2, rng)),
                ),
                _build_sentence_chunk(
                    "support_internal_process",
                    (CASE2_SENTENCE_LABEL_INTERNAL, _pick(support_internal, rng)),
                ),
            ],
        ),
    ]


def _build_ticket_from_specs(
    ticket_id: str,
    category: str,
    message_specs: List[Tuple[str, List[Dict[str, Any]]]],
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    next_sentence_index = 0
    messages = []
    ticket_ground_truth: Dict[int, str] = {}
    chunk_metadata: List[Dict[str, Any]] = []

    for message_index, (speaker, sentence_chunks) in enumerate(message_specs):
        message, message_truth, next_sentence_index, message_chunk_metadata = _build_ticket_message(
            speaker=speaker,
            message_index=message_index,
            sentence_chunks=sentence_chunks,
            next_sentence_index=next_sentence_index,
            rng=rng,
        )
        messages.append(message)
        ticket_ground_truth.update(message_truth)
        chunk_metadata.extend(message_chunk_metadata)

    return {
        "ticket_id": ticket_id,
        "category": category,
        "messages": messages,
        "chunk_metadata": chunk_metadata,
    }, ticket_ground_truth


def _generate_technical_support_ticket(
    ticket_id: str,
    requester_name: str,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    return _build_ticket_from_specs(ticket_id, "technical_support", _technical_support_message_specs(requester_name, rng), rng)


def _generate_billing_dispute_ticket(
    ticket_id: str,
    requester_name: str,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    return _build_ticket_from_specs(ticket_id, "billing_dispute", _billing_dispute_message_specs(requester_name, rng), rng)


def _generate_cancellation_ticket(
    ticket_id: str,
    requester_name: str,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    return _build_ticket_from_specs(ticket_id, "cancellation_request", _cancellation_message_specs(requester_name, rng), rng)


def generate_case1_episode(
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str], str]:
    rng = random.Random(seed)

    full_name = rng.choice(REQUESTER_NAMES)
    email = _make_email(full_name, rng)
    city, postcode, street = rng.choice(CITIES_WITH_POSTCODES)
    billing_address = f"{street}, {city}, {postcode}"
    support_ticket_ids = [f"TKT-{rng.randint(1000, 9999)}" for _ in range(2)]

    raw_values = _build_structured_case_values(
        full_name=full_name,
        email=email,
        billing_address=billing_address,
        support_ticket_ids=support_ticket_ids,
        rng=rng,
    )
    customer_record = _build_customer_record(raw_values, rng)
    ground_truth = dict(FIELD_GROUND_TRUTH)
    request_date = _make_request_date(rng)
    dsar_text = CASE1_DSAR_TEMPLATE.format(
        email=email,
        full_name=full_name,
        request_date=request_date,
    )
    return customer_record, raw_values, ground_truth, dsar_text


def generate_case2_episode(seed: Optional[int] = None) -> Dict[str, Any]:
    rng = random.Random(seed)

    base_name = rng.choice(REQUESTER_NAMES)
    city, postcode, street = rng.choice(CITIES_WITH_POSTCODES)
    identity = _build_identity_scenario(base_name, city, street, postcode, rng)

    ticket_builders = [
        _generate_technical_support_ticket,
        _generate_billing_dispute_ticket,
        _generate_cancellation_ticket,
    ]
    tickets: List[Dict[str, Any]] = []
    ticket_ground_truth: Dict[str, Dict[int, str]] = {}
    support_ticket_ids: List[str] = []

    for builder in ticket_builders:
        ticket_id = f"TKT-{rng.randint(1000, 9999)}"
        while ticket_id in support_ticket_ids:
            ticket_id = f"TKT-{rng.randint(1000, 9999)}"
        ticket, truth = builder(ticket_id, identity["internal_identity_full"]["full_name"], rng)
        tickets.append(ticket)
        ticket_ground_truth[ticket_id] = truth
        support_ticket_ids.append(ticket_id)

    rng.shuffle(tickets)

    raw_values = _build_structured_case_values(
        full_name=identity["internal_identity_full"]["full_name"],
        email=identity["internal_identity_full"]["email"],
        billing_address=identity["internal_identity_full"]["billing_address"],
        support_ticket_ids=support_ticket_ids,
        rng=rng,
    )
    customer_record = _build_customer_record(raw_values, rng)
    ground_truth = dict(FIELD_GROUND_TRUTH)
    request_date = _make_request_date(rng)
    dsar_text = CASE2_DSAR_TEMPLATE.format(
        submitted_name=identity["submitted_identity"]["full_name"],
        submitted_email=identity["submitted_identity"]["email"],
        submitted_address=identity["submitted_identity"]["billing_address"],
        request_date=request_date,
    )

    return {
        "customer_record": customer_record,
        "values_lookup": raw_values,
        "ground_truth": ground_truth,
        "dsar_text": dsar_text,
        "submitted_identity": identity["submitted_identity"],
        "internal_identity_full": identity["internal_identity_full"],
        "internal_identity_masked": identity["internal_identity_masked"],
        "starting_identity_confidence": identity["starting_identity_confidence"],
        "verification_threshold": CASE2_VERIFICATION_THRESHOLD,
        "correct_verification_method": identity["correct_verification_method"],
        "competing_verification_method": identity["competing_verification_method"],
        "identity_ambiguity": identity["identity_ambiguity"],
        "tickets": tickets,
        "ticket_ground_truth": ticket_ground_truth,
    }


def _case3_sentence_items(text: str) -> List[Dict[str, Any]]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return [{"sentence_idx": idx, "text": part} for idx, part in enumerate(parts)]


def _random_slack_user_id(rng: random.Random) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "U" + "".join(rng.choice(alphabet) for _ in range(8))


def _unique_slack_user_ids(rng: random.Random, count: int = 4) -> List[str]:
    ids: List[str] = []
    while len(ids) < count:
        candidate = _random_slack_user_id(rng)
        if candidate not in ids:
            ids.append(candidate)
    return ids


def _build_slack_message(
    msg_id: str,
    user_id: str,
    text: str,
    ts: str,
    *,
    thread_ts: Optional[str] = None,
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    msg = {
        "msg_id": msg_id,
        "user": user_id,
        "text": text,
        "ts": ts,
        "sentences": _case3_sentence_items(text),
    }
    if thread_ts is not None:
        msg["thread_ts"] = thread_ts
    if subtype is not None:
        msg["subtype"] = subtype
    return msg


def _random_case3_message_id(rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "msg_" + "".join(rng.choice(alphabet) for _ in range(10))


def _generate_case3_users_json(
    requester_name: str,
    manager_name: str,
    colleague_name: str,
    rng: random.Random,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    bot_name = f"{rng.choice(PIPELINE_NAMES)}Bot"
    role_names = {
        "requester": requester_name,
        "manager": manager_name,
        "colleague": colleague_name,
        "bot": bot_name,
    }
    user_ids = _unique_slack_user_ids(rng, count=4)
    rng.shuffle(user_ids)
    role_to_user_id = {
        role: user_id for role, user_id in zip(role_names.keys(), user_ids)
    }

    users_json: Dict[str, Dict[str, Any]] = {}
    for role, user_id in role_to_user_id.items():
        name = role_names[role]
        users_json[user_id] = {
            "real_name": name,
            "display_name": name.split()[0].lower(),
        }
        if role == "bot":
            users_json[user_id]["is_bot"] = True
    return users_json, role_to_user_id


def _generate_case3_timestamps(rng: random.Random) -> Dict[str, str]:
    base_ts = 1700000000 + rng.randint(0, 100000)
    clean_offset = rng.randint(10, 120)
    thread_offset = clean_offset + rng.randint(11, 90)

    used_offsets = {clean_offset, thread_offset}
    other_keys = ["health", "mixed", "bot", "perf"]
    offsets: Dict[str, int] = {"clean": clean_offset, "thread": thread_offset}
    for key in other_keys:
        offset = rng.randint(1, 420)
        while offset in used_offsets:
            offset = rng.randint(1, 420)
        used_offsets.add(offset)
        offsets[key] = offset

    return {key: str(base_ts + offset) for key, offset in offsets.items()}


def _case3_distractor_count() -> int:
    raw = os.environ.get("DSAR_CASE3_DISTRACTORS", "").strip()
    if not raw:
        return 0
    try:
        return max(0, min(4, int(raw)))
    except ValueError:
        return 0


def generate_case3_episode(seed: Optional[int] = None) -> Dict[str, Any]:
    rng = random.Random(seed)

    requester_name = rng.choice(EMPLOYEE_NAMES)
    manager_name = rng.choice(MANAGER_NAMES)
    colleague_name = rng.choice(COLLEAGUE_NAMES)
    project = rng.choice(PROJECT_NAMES)
    project_lower = project.lower()
    health_condition = rng.choice(HEALTH_CONDITIONS)
    build_num = rng.randint(1000, 9999)
    env = rng.choice(["eu-west-2", "us-east-1", "ap-southeast-1"])
    requester_first = requester_name.split()[0]
    request_date = _make_request_date(rng)
    employee_id = f"EMP-{rng.randint(1000, 9999)}"
    team_channel = f"{project_lower}-team"
    other_project = rng.choice([candidate for candidate in PROJECT_NAMES if candidate != project])
    users_json, role_to_user_id = _generate_case3_users_json(
        requester_name=requester_name,
        manager_name=manager_name,
        colleague_name=colleague_name,
        rng=rng,
    )
    timestamps = _generate_case3_timestamps(rng)

    health_text = rng.choice(HEALTH_TRAP_TEMPLATES).format(
        requester=requester_first,
        project=project,
        health_condition=health_condition,
    )
    clean_text = rng.choice(REQUESTER_CLEAN_TEMPLATES).format(
        project=project,
        project_lower=project_lower,
    )
    pr_phrase = rng.choice(PR_PHRASES).format(project=project)
    salary_phrase = rng.choice(SALARY_PHRASES)
    mixed_text = f"Hey {requester_first}, {pr_phrase}? Also, {salary_phrase}."
    thread_text = rng.choice(THREAD_REPLY_TEMPLATES)
    bot_text = rng.choice(BOT_TEMPLATES).format(
        project=project,
        build_num=build_num,
        region=env,
        env=env,
        timestamp=timestamps["bot"],
    )
    perf_text = (
        rng.choice(MANAGER_PERF_TEMPLATES).format(requester=requester_first)
        + f" Ref {rng.choice(CASE3_INTERNAL_HR_CODES)}."
    )

    msg_ids = {
        "health": _random_case3_message_id(rng),
        "clean": _random_case3_message_id(rng),
        "mixed": _random_case3_message_id(rng),
        "thread": _random_case3_message_id(rng),
        "bot": _random_case3_message_id(rng),
        "perf": _random_case3_message_id(rng),
    }
    while len(set(msg_ids.values())) < len(msg_ids):
        for key, value in list(msg_ids.items()):
            if list(msg_ids.values()).count(value) > 1:
                msg_ids[key] = _random_case3_message_id(rng)

    messages = [
        _build_slack_message(
            msg_ids["health"],
            role_to_user_id["manager"],
            health_text,
            timestamps["health"],
        ),
        _build_slack_message(
            msg_ids["clean"],
            role_to_user_id["requester"],
            clean_text,
            timestamps["clean"],
        ),
        _build_slack_message(
            msg_ids["mixed"],
            role_to_user_id["colleague"],
            mixed_text,
            timestamps["mixed"],
        ),
        _build_slack_message(
            msg_ids["thread"],
            role_to_user_id["manager"],
            thread_text,
            timestamps["thread"],
            thread_ts=timestamps["clean"],
        ),
        _build_slack_message(
            msg_ids["bot"],
            role_to_user_id["bot"],
            bot_text,
            timestamps["bot"],
            subtype="bot_message",
        ),
        _build_slack_message(
            msg_ids["perf"],
            role_to_user_id["manager"],
            perf_text,
            timestamps["perf"],
        ),
    ]
    distractor_count = _case3_distractor_count()
    for index in range(distractor_count):
        distractor_id = _random_case3_message_id(rng)
        while distractor_id in msg_ids.values():
            distractor_id = _random_case3_message_id(rng)
        distractor_text = rng.choice(CASE3_DISTRACTOR_TEMPLATES).format(
            colleague=colleague_name.split()[0],
            manager=manager_name.split()[0],
            other_project=other_project,
        )
        distractor_user = rng.choice(
            [role_to_user_id["manager"], role_to_user_id["colleague"]]
        )
        distractor_ts = str(int(max(timestamps.values(), key=int)) + 50 + index * 17)
        messages.append(
            _build_slack_message(
                distractor_id,
                distractor_user,
                distractor_text,
                distractor_ts,
            )
        )
    rng.shuffle(messages)

    ground_truth = {
        msg_ids["health"]: {
            "action": CASE3_ACTION_ESCALATE,
            "is_special_category": True,
            "sentence_ground_truth": None,
        },
        msg_ids["clean"]: {
            "action": CASE3_ACTION_DISCLOSE,
            "is_special_category": False,
            "sentence_ground_truth": None,
        },
        msg_ids["mixed"]: {
            "action": CASE3_ACTION_PARTIAL_REDACT,
            "is_special_category": False,
            "sentence_ground_truth": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
            },
            "kept_sentence_idx": 0,
            "redacted_sentence_idx": 1,
        },
        msg_ids["thread"]: {
            "action": CASE3_ACTION_DISCLOSE,
            "is_special_category": False,
            "thread_parent_id": msg_ids["clean"],
            "sentence_ground_truth": None,
        },
        msg_ids["bot"]: {
            "action": CASE3_ACTION_EXCLUDE,
            "is_special_category": False,
            "sentence_ground_truth": None,
        },
        msg_ids["perf"]: {
            "action": CASE3_ACTION_DISCLOSE,
            "is_special_category": False,
            "sentence_ground_truth": None,
        },
    }
    for message in messages:
        if message["msg_id"] not in ground_truth:
            ground_truth[message["msg_id"]] = {
                "action": CASE3_ACTION_EXCLUDE,
                "is_special_category": False,
                "sentence_ground_truth": None,
            }

    dsar_text = CASE3_DSAR_TEMPLATE.format(
        request_date=request_date,
        requester_name=requester_name,
        requester_username=requester_first.lower(),
        employee_id=employee_id,
        team_channel=team_channel,
    )

    return {
        "messages": messages,
        "users_json": users_json,
        "ground_truth": ground_truth,
        "dsar_text": dsar_text,
        "requester_user_id": role_to_user_id["requester"],
        "requester_name": requester_name,
        "special_category_message_ids": [msg_ids["health"]],
        "thread_parent_id": msg_ids["clean"],
        "thread_reply_id": msg_ids["thread"],
        "bot_message_id": msg_ids["bot"],
        "mixed_sentence_message_id": msg_ids["mixed"],
        "employee_id": employee_id,
        "team_channel": team_channel,
    }
