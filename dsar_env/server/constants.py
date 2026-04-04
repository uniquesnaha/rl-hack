"""
Synthetic data pools and metadata for the DSAR environment.

Field names are fixed while field values vary per episode. This forces the
agent to learn semantics rather than memorize exact values.
"""

# Case 1 requester names
REQUESTER_NAMES = [
    "Sarah Mitchell",
    "James Okafor",
    "Priya Sharma",
    "Carlos Reyes",
    "Emma Johansson",
    "David Nguyen",
    "Fatima Al-Hassan",
    "Liam O'Brien",
    "Yuki Tanaka",
    "Aisha Patel",
    "Marco Rossi",
    "Chloe Dubois",
    "Raj Mehta",
    "Olivia Brown",
    "Hassan Yilmaz",
    "Sofia Andersson",
    "Chen Wei",
    "Isabella Garcia",
    "Kwame Asante",
    "Maria Kowalski",
    "Thomas Fischer",
    "Anna Petrova",
    "Lucas Santos",
    "Mia Thompson",
    "Benjamin Lee",
]

EMAIL_DOMAINS = [
    "gmail.com",
    "outlook.com",
    "yahoo.com",
    "protonmail.com",
    "icloud.com",
    "hotmail.com",
    "fastmail.com",
    "zoho.com",
]

SUBSCRIPTION_PLANS = [
    "Pro Monthly",
    "Starter Annual",
    "Business Plus",
    "Enterprise Tier",
    "Developer Free",
    "Team Premium",
]

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
    ("Berlin", "10115", "Friedrichstrasse 43"),
    ("Amsterdam", "1012 JS", "Damrak 70"),
    ("Sydney", "2000", "1 Macquarie Pl"),
    ("Dublin", "D02 YX88", "25 St Stephen's Green"),
]

ACCOUNT_MANAGER_NOTES = [
    "Price-sensitive customer, avoid premium upsell. Prefers email contact only.",
    "High-touch account, requires quarterly check-ins with CSM. Renewal risk Q4.",
    "Flagged for churn risk, enrolled in retention campaign RC-2024-Q2.",
    "Technical buyer - route product feedback directly to engineering liaison.",
    "Expansion opportunity: team grew 3x last quarter. Push Enterprise upgrade.",
]

SHARD_ROUTING_KEYS = [
    "eu-west-2-db4",
    "us-east-1-db7",
    "ap-southeast-1-db2",
    "eu-central-1-db1",
    "us-west-2-db3",
    "ap-northeast-1-db5",
    "eu-north-1-db6",
    "sa-east-1-db8",
]

LEAD_SOURCE_TAGS = [
    "google_ads_q3",
    "organic_search",
    "referral_program",
    "linkedin_campaign",
    "partner_channel",
    "conference_lead",
]

PROFIT_TIERS = ["low-margin", "mid-tier", "high-value"]
MARKETING_PREFERENCES = ["email_opted_in", "all_opted_out", "sms_only"]
REFERRAL_CREDIT_BALANCES = [0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 75.0, 120.0]

REQUESTER_DATA_FIELDS = [
    "full_name",
    "email",
    "billing_address",
    "subscription_plan",
    "subscription_start_date",
    "payment_history",
    "referral_credit_balance",
    "marketing_preferences",
    "product_usage_summary",
    "support_ticket_ids",
]

INTERNAL_ONLY_FIELDS = [
    "customer_health_score",
    "engagement_index",
    "lifetime_value_estimate",
    "profit_tier",
    "shard_routing_key",
    "account_manager_notes",
    "campaign_cpa",
]

FIELD_GROUND_TRUTH = {}
for _field in REQUESTER_DATA_FIELDS:
    FIELD_GROUND_TRUTH[_field] = "REQUESTER_DATA"
for _field in INTERNAL_ONLY_FIELDS:
    FIELD_GROUND_TRUTH[_field] = "INTERNAL_ONLY"

FIELD_METADATA = {
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
    "referral_credit_balance": (
        "Referral Credit Balance",
        "billing",
        "financial",
        "The outstanding credit balance accrued through the referral programme, applicable to future billing cycles.",
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
    "customer_health_score": (
        "Customer Health Score",
        "crm",
        "analytical",
        "A composite score reflecting account activity, payment consistency, and feature adoption over the trailing 90 days.",
    ),
    "engagement_index": (
        "Engagement Index",
        "crm",
        "analytical",
        "A normalised index derived from recent product activity, support interactions, and feature utilisation patterns.",
    ),
    "lifetime_value_estimate": (
        "Lifetime Value Estimate",
        "billing",
        "analytical",
        "A projected revenue figure calculated from the account's current plan, usage patterns, and historical retention data.",
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
    "campaign_cpa": (
        "Campaign CPA",
        "crm",
        "analytical",
        "The cost-per-acquisition figure from the paid marketing campaign attributed to this account at signup.",
    ),
}

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

CASE1_VALID_SILOS = {"billing", "crm"}

MAX_STEPS = 30
FREE_STEPS = 10
STEP_COST = 0.01

# Case 2 constants
CASE2_VALID_SILOS = CASE1_VALID_SILOS
CASE2_FREE_STEPS = 15
CASE2_VERIFICATION_THRESHOLD = 0.80

CASE2_PROPORTIONATE_METHODS = {
    "transaction_date",
    "account_reference",
    "registered_postcode",
}
CASE2_DISPROPORTIONATE_METHODS = {
    "passport_copy",
    "photo_id",
}
CASE2_VERIFICATION_METHODS = (
    CASE2_PROPORTIONATE_METHODS | CASE2_DISPROPORTIONATE_METHODS
)

CASE2_SENTENCE_LABEL_REQUESTER = "REQUESTER_DATA"
CASE2_SENTENCE_LABEL_PII = "THIRD_PARTY_PII"
CASE2_SENTENCE_LABEL_INTERNAL = "INTERNAL_NOTE"

WORK_EMAIL_DOMAINS = [
    "techcorp.com",
    "acmeworks.io",
    "northstar.ai",
    "bluegrid.co",
    "atlascloud.dev",
]

SUPPORT_AGENT_NAMES = [
    "Maya Collins",
    "Ethan Brooks",
    "Nina Kapoor",
    "Daniel Foster",
    "Ava Martinez",
    "Oliver Grant",
    "Sana Rahman",
    "Leo Bennett",
]

SUPPORT_PHONE_NUMBERS = [
    "07700 900123",
    "07700 900456",
    "07700 900789",
    "020 7946 0123",
    "020 7946 0456",
    "+44 20 7946 0789",
]

INTERNAL_ESCALATION_CODES = [
    "ESC-PLAT-1042",
    "ESC-BILL-2088",
    "ESC-RET-3091",
    "ESC-SUP-4175",
    "ESC-OPS-5220",
]

BILLING_SYSTEM_FLAGS = [
    "FLAG-BILLING-REVIEW",
    "FLAG-DISPUTE-HOLD",
    "FLAG-MANUAL-REFUND-CHECK",
    "FLAG-CARD-RETRY-WATCH",
]

RETENTION_CAMPAIGN_CODES = [
    "RET-Q2-WINBACK",
    "RET-LAPSE-SAVE-10",
    "RET-ANNUAL-UPGRADE-PUSH",
    "RET-LOYALTY-CREDIT",
]

PRODUCT_NAMES = [
    "Insights Dashboard",
    "Usage Export",
    "Team Workspace",
    "Billing Console",
    "Automation Studio",
]

TECH_SUPPORT_ISSUES = [
    "export the monthly usage report",
    "open the billing dashboard",
    "complete the CSV download",
    "load the workspace settings page",
    "access the admin panel",
]

BILLING_DISPUTE_REASONS = [
    "an annual renewal charge",
    "a duplicate invoice",
    "a failed refund",
    "a late cancellation fee",
]

CANCELLATION_REASONS = [
    "moving to another tool",
    "closing the project",
    "ending the team subscription",
    "reducing software spend",
]

ACCOUNT_REFERENCE_PREFIXES = ["ACCT", "SUB", "BILL", "CRM"]

CASE2_DSAR_TEMPLATE = (
    "Subject: Data Subject Access Request\n"
    "From: {submitted_email}\n"
    "Date: {request_date}\n\n"
    "Dear Privacy Team,\n\n"
    "I am requesting access to all personal data you hold about me under "
    "Article 15 of the UK GDPR.\n\n"
    "My details are:\n"
    "- Name: {submitted_name}\n"
    "- Email: {submitted_email}\n"
    "- Address: {submitted_address}\n\n"
    "I no longer have access to the work email previously associated with my "
    "account, so I am contacting you from my current personal email address.\n\n"
    "Kind regards,\n"
    "{submitted_name}"
)
