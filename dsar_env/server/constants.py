"""
Synthetic data pools for DSAR environment.

All randomisation comes from selecting values from these pools.
Field names are fixed — only values change per episode.
This ensures the RL agent must learn semantic meaning, not memorise values.
"""

# ─── Pool 1: Requester names ──────────────────────────────────────────────────
REQUESTER_NAMES = [
    "Sarah Mitchell", "James Okafor", "Priya Sharma", "Carlos Reyes",
    "Emma Johansson", "David Nguyen", "Fatima Al-Hassan", "Liam O'Brien",
    "Yuki Tanaka", "Aisha Patel", "Marco Rossi", "Chloe Dubois",
    "Raj Mehta", "Olivia Brown", "Hassan Yilmaz", "Sofia Andersson",
    "Chen Wei", "Isabella Garcia", "Kwame Asante", "Maria Kowalski",
    "Thomas Fischer", "Anna Petrova", "Lucas Santos", "Mia Thompson",
    "Benjamin Lee",
]

# ─── Pool 2: Email domains ───────────────────────────────────────────────────
EMAIL_DOMAINS = [
    "gmail.com", "outlook.com", "yahoo.com", "protonmail.com",
    "icloud.com", "hotmail.com", "fastmail.com", "zoho.com",
]

# ─── Pool 3: Subscription plan names ─────────────────────────────────────────
SUBSCRIPTION_PLANS = [
    "Pro Monthly", "Starter Annual", "Business Plus",
    "Enterprise Tier", "Developer Free", "Team Premium",
]

# ─── Pool 4: Cities with postcodes ───────────────────────────────────────────
CITIES_WITH_POSTCODES = [
    ("London", "EC1A 1BB", "14 Aldersgate St"),
    ("Manchester", "M1 1AE", "27 Piccadilly Gardens"),
    ("Edinburgh", "EH1 1YZ", "5 Princes St"),
    ("Birmingham", "B1 1BB", "33 New St"),
    ("Bristol", "BS1 5TR", "8 Baldwin St"),
    ("New York", "10001", "142 W 36th St"),
    ("San Francisco", "94102", "501 Market St"),
    ("Chicago", "60601", "233 S Wacker Dr"),
    ("Austin", "73301", "1100 Congress Ave"),
    ("Seattle", "98101", "1201 3rd Ave"),
    ("Toronto", "M5H 2N2", "100 King St W"),
    ("Berlin", "10115", "Friedrichstraße 43"),
    ("Amsterdam", "1012 JS", "Damrak 70"),
    ("Sydney", "2000", "1 Macquarie Pl"),
    ("Dublin", "D02 YX88", "25 St Stephen's Green"),
]

# ─── Pool 5: Account manager note templates ──────────────────────────────────
ACCOUNT_MANAGER_NOTES = [
    "Price-sensitive customer, avoid premium upsell. Prefers email contact only.",
    "High-touch account, requires quarterly check-ins with CSM. Renewal risk Q4.",
    "Flagged for churn risk, enrolled in retention campaign RC-2024-Q2.",
    "Technical buyer — route product feedback directly to engineering liaison.",
    "Expansion opportunity: team grew 3x last quarter. Push Enterprise upgrade.",
]

# ─── Pool 6: Shard routing key formats ───────────────────────────────────────
SHARD_ROUTING_KEYS = [
    "eu-west-2-db4", "us-east-1-db7", "ap-southeast-1-db2",
    "eu-central-1-db1", "us-west-2-db3", "ap-northeast-1-db5",
    "eu-north-1-db6", "sa-east-1-db8",
]

# ─── Pool 7: Lead source tags ────────────────────────────────────────────────
LEAD_SOURCE_TAGS = [
    "google_ads_q3", "organic_search", "referral_program",
    "linkedin_campaign", "partner_channel", "conference_lead",
]

# ─── Pool 8: Profit tier values ──────────────────────────────────────────────
PROFIT_TIERS = ["low-margin", "mid-tier", "high-value"]

# ─── Pool 9: Marketing preferences ───────────────────────────────────────────
MARKETING_PREFERENCES = ["email_opted_in", "all_opted_out", "sms_only"]

# ─── Field classification ground truth ────────────────────────────────────────
# This mapping NEVER changes. Only values change per episode.
# The agent must learn to classify by field semantics, not memorise.

REQUESTER_DATA_FIELDS = [
    "full_name",
    "email",
    "billing_address",
    "subscription_plan",
    "subscription_start_date",
    "payment_history",
    "marketing_preferences",
    "product_usage_summary",
    "support_ticket_ids",
]

INTERNAL_ONLY_FIELDS = [
    "risk_score",
    "churn_probability",
    "profit_tier",
    "shard_routing_key",
    "account_manager_notes",
    "lead_source_tag",
    "campaign_cpa",
]

# Convenience: full ground truth mapping
FIELD_GROUND_TRUTH = {}
for _f in REQUESTER_DATA_FIELDS:
    FIELD_GROUND_TRUTH[_f] = "REQUESTER_DATA"
for _f in INTERNAL_ONLY_FIELDS:
    FIELD_GROUND_TRUTH[_f] = "INTERNAL_ONLY"


# ─── Rich field metadata ─────────────────────────────────────────────────────
# Each field has: (field_name_display, source_silo, datatype, description)
# Descriptions are factual but do NOT reveal whether the field should be
# disclosed or withheld — that judgment is the agent's job.

FIELD_METADATA = {
    # REQUESTER_DATA fields
    "full_name": (
        "Full Name",
        "crm",
        "personal_identifier",
        "The individual's full legal name as registered in the system.",
    ),
    "email": (
        "Email Address",
        "crm",
        "personal_identifier",
        "Primary email address associated with the individual's account.",
    ),
    "billing_address": (
        "Billing Address",
        "billing",
        "personal_identifier",
        "Postal address used for billing and invoicing purposes.",
    ),
    "subscription_plan": (
        "Subscription Plan",
        "billing",
        "financial",
        "The name of the subscription tier the individual is enrolled in.",
    ),
    "subscription_start_date": (
        "Subscription Start Date",
        "billing",
        "financial",
        "The date when the individual first subscribed to the service.",
    ),
    "payment_history": (
        "Payment History",
        "billing",
        "financial",
        "Record of payments made by the individual, including dates and amounts.",
    ),
    "marketing_preferences": (
        "Marketing Preferences",
        "crm",
        "behavioral",
        "The individual's stated communication and marketing opt-in/out preferences.",
    ),
    "product_usage_summary": (
        "Product Usage Summary",
        "crm",
        "behavioral",
        "Aggregated summary of how the individual has used the product features.",
    ),
    "support_ticket_ids": (
        "Support Ticket IDs",
        "crm",
        "personal_identifier",
        "Reference IDs for support tickets filed by the individual.",
    ),

    # INTERNAL_ONLY fields
    "risk_score": (
        "Risk Score",
        "crm",
        "analytical",
        "A numerical score computed from account behaviour, payment patterns, and engagement signals.",
    ),
    "churn_probability": (
        "Churn Probability",
        "crm",
        "analytical",
        "A model-predicted probability score that the account will not renew in the next quarter.",
    ),
    "profit_tier": (
        "Profit Tier",
        "billing",
        "analytical",
        "A revenue-tier classification assigned to the account based on contract value and margin.",
    ),
    "shard_routing_key": (
        "Shard Routing Key",
        "billing",
        "infrastructure",
        "A database partition key used by the storage layer to route read/write operations for this account.",
    ),
    "account_manager_notes": (
        "Account Manager Notes",
        "crm",
        "operational",
        "Free-text notes written by the assigned account manager about strategy, renewal risk, and engagement history.",
    ),
    "lead_source_tag": (
        "Lead Source Tag",
        "crm",
        "operational",
        "A tag recorded at account creation indicating which acquisition channel or campaign sourced this customer.",
    ),
    "campaign_cpa": (
        "Campaign CPA",
        "crm",
        "analytical",
        "The cost-per-acquisition figure from the paid marketing campaign attributed to this account at signup.",
    ),
}


# ─── DSAR ticket template for Case 1 ─────────────────────────────────────────
CASE1_DSAR_TEMPLATE = (
    "Subject: Data Subject Access Request\n"
    "From: {email}\n"
    "Date: {request_date}\n\n"
    "Dear Data Protection Officer,\n\n"
    "Under Article 15 of the UK GDPR, I am writing to request access to all "
    "personal data your organisation holds about me.\n\n"
    "My name is {full_name} and I am a current subscriber to your service. "
    "Please provide a complete copy of my personal data within the statutory "
    "30-day period.\n\n"
    "Kind regards,\n"
    "{full_name}"
)

# ─── Valid silos for Case 1 ───────────────────────────────────────────────────
CASE1_VALID_SILOS = {"billing", "crm"}

# ─── Episode parameters ──────────────────────────────────────────────────────
MAX_STEPS = 30
FREE_STEPS = 10
STEP_COST = 0.01
