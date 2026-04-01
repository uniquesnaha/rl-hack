"""
Synthetic data generator for DSAR episodes.

Each call to generate_case1_episode() produces:
  1. A customer_record (list of FieldItem dicts) — what the agent sees
  2. A ground_truth (dict) — what the grader uses (hidden from agent)
  3. A dsar_text (str) — the DSAR request letter

Both record and ground truth are produced in the SAME function call
so they can never drift out of sync.
"""

import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    ACCOUNT_MANAGER_NOTES,
    CASE1_DSAR_TEMPLATE,
    CITIES_WITH_POSTCODES,
    EMAIL_DOMAINS,
    FIELD_GROUND_TRUTH,
    FIELD_METADATA,
    LEAD_SOURCE_TAGS,
    MARKETING_PREFERENCES,
    PROFIT_TIERS,
    REQUESTER_NAMES,
    SHARD_ROUTING_KEYS,
    SUBSCRIPTION_PLANS,
)


def _make_email(name: str, rng: random.Random) -> str:
    """Generate email from name: 'Sarah Mitchell' → 'sarah.mitchell@gmail.com'."""
    parts = name.lower().split()
    local = ".".join(parts)
    domain = rng.choice(EMAIL_DOMAINS)
    return f"{local}@{domain}"


def _make_payment_history(rng: random.Random) -> list:
    """Generate 2-4 fake payment entries."""
    count = rng.randint(2, 4)
    base_date = date(2023, 1, 1)
    entries = []
    for i in range(count):
        pay_date = base_date + timedelta(days=30 * i + rng.randint(0, 5))
        amount = round(rng.choice([9.99, 19.99, 29.99, 49.99, 79.99]), 2)
        entries.append({"date": pay_date.isoformat(), "amount": amount})
    return entries


def _make_usage_summary(rng: random.Random) -> str:
    """Generate a product usage summary string."""
    templates = [
        "logged_in_{n}_times_last_month",
        "used_export_feature_{n}_times",
        "created_{n}_projects",
        "invited_{n}_team_members",
        "generated_{n}_reports",
    ]
    template = rng.choice(templates)
    n = rng.randint(2, 50)
    return template.format(n=n)


def _make_ticket_ids(rng: random.Random) -> list:
    """Generate 2 fake support ticket IDs."""
    return [f"TKT-{rng.randint(1000, 9999)}" for _ in range(2)]


def _make_subscription_start(rng: random.Random) -> str:
    """Random date between 2021-01-01 and 2023-12-31."""
    start = date(2021, 1, 1)
    days_range = (date(2023, 12, 31) - start).days
    return (start + timedelta(days=rng.randint(0, days_range))).isoformat()


def _build_field_item(field_id: str, value: Any) -> Dict[str, Any]:
    """Build a FieldItem dict from a field_id and its generated value.

    Uses FIELD_METADATA to enrich with display name, source, datatype,
    and description.
    """
    meta = FIELD_METADATA.get(field_id)
    if meta is None:
        return {
            "field_id": field_id,
            "field_name": field_id.replace("_", " ").title(),
            "field_value": value,
            "source_silo": "unknown",
            "datatype": "unknown",
            "field_description": "No description available.",
        }

    display_name, source_silo, datatype, description = meta
    return {
        "field_id": field_id,
        "field_name": display_name,
        "field_value": value,
        "source_silo": source_silo,
        "datatype": datatype,
        "field_description": description,
    }


def generate_case1_episode(
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str], str]:
    """Generate one complete Case 1 episode.

    Args:
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (customer_record, values_lookup, ground_truth, dsar_request_text):
        - customer_record: list of FieldItem dicts (16 items: 9 requester + 7 internal)
        - values_lookup: flat dict of {field_id: value} for grader/draft_response use
        - ground_truth: dict mapping field_id to 'REQUESTER_DATA' or 'INTERNAL_ONLY'
        - dsar_request_text: the DSAR ticket text shown to the agent
    """
    rng = random.Random(seed)

    # ── Requester identity ────────────────────────────────────────────────
    full_name = rng.choice(REQUESTER_NAMES)
    email = _make_email(full_name, rng)
    city, postcode, street = rng.choice(CITIES_WITH_POSTCODES)
    billing_address = f"{street}, {city}, {postcode}"

    # ── Build raw values: 9 REQUESTER_DATA fields ─────────────────────────
    raw_values: Dict[str, Any] = {
        "full_name": full_name,
        "email": email,
        "billing_address": billing_address,
        "subscription_plan": rng.choice(SUBSCRIPTION_PLANS),
        "subscription_start_date": _make_subscription_start(rng),
        "payment_history": _make_payment_history(rng),
        "marketing_preferences": rng.choice(MARKETING_PREFERENCES),
        "product_usage_summary": _make_usage_summary(rng),
        "support_ticket_ids": _make_ticket_ids(rng),
    }

    # ── Build raw values: 7 INTERNAL_ONLY fields ─────────────────────────
    raw_values.update({
        "risk_score": round(rng.uniform(0.0, 1.0), 2),
        "churn_probability": round(rng.uniform(0.0, 1.0), 2),
        "profit_tier": rng.choice(PROFIT_TIERS),
        "shard_routing_key": rng.choice(SHARD_ROUTING_KEYS),
        "account_manager_notes": rng.choice(ACCOUNT_MANAGER_NOTES),
        "lead_source_tag": rng.choice(LEAD_SOURCE_TAGS),
        "campaign_cpa": round(rng.uniform(8.0, 45.0), 2),
    })

    # ── Build rich FieldItem list ─────────────────────────────────────────
    customer_record = []
    for field_id, value in raw_values.items():
        customer_record.append(_build_field_item(field_id, value))

    # Shuffle the field order so agents can't rely on position
    rng.shuffle(customer_record)

    # ── Ground truth: same keys, classification labels ────────────────────
    ground_truth = dict(FIELD_GROUND_TRUTH)  # copy the fixed mapping

    # ── DSAR request text ─────────────────────────────────────────────────
    request_date = date.today().isoformat()
    dsar_text = CASE1_DSAR_TEMPLATE.format(
        email=email,
        full_name=full_name,
        request_date=request_date,
    )

    return customer_record, raw_values, ground_truth, dsar_text
