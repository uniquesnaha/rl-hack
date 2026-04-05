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
    "risk_score",
    "churn_probability",
    "lead_source_tag",
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
        "The credit balance currently available on the individual's account and applied to future invoices.",
    ),
    "marketing_preferences": (
        "Marketing Preferences",
        "crm",
        "behavioral",
        "The individual's own marketing communication choices and opt-in or opt-out preferences.",
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
        "Reference IDs for support cases opened by the individual on their account.",
    ),
    "customer_health_score": (
        "Customer Health Score",
        "crm",
        "analytical",
        "A company-generated score used by account teams to monitor account stability, payment consistency, and feature adoption over the trailing 90 days.",
    ),
    "risk_score": (
        "Risk Score",
        "crm",
        "analytical",
        "A company-generated risk indicator used by service teams to prioritise account review and intervention.",
    ),
    "churn_probability": (
        "Churn Probability",
        "crm",
        "analytical",
        "A company-generated probability estimate used by retention teams to forecast account cancellation risk.",
    ),
    "lead_source_tag": (
        "Lead Source Tag",
        "crm",
        "marketing_attribution",
        "A marketing attribution label used by marketing teams to measure campaign source and internal acquisition reporting.",
    ),
    "shard_routing_key": (
        "Shard Routing Key",
        "billing",
        "infrastructure",
        "A database partition key used by the storage layer to route read/write operations for this account in backend systems.",
    ),
    "account_manager_notes": (
        "Account Manager Notes",
        "crm",
        "operational",
        "Free-text notes written by the assigned account manager for account strategy, renewal risk, and engagement planning.",
    ),
    "campaign_cpa": (
        "Campaign CPA",
        "crm",
        "analytical",
        "A cost-per-acquisition figure used for campaign performance reporting and marketing analysis.",
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

# Case 3 constants
CASE3_MAX_STEPS = 40
CASE3_FREE_STEPS = 16

CASE3_ACTION_DISCLOSE = "disclose"
CASE3_ACTION_PARTIAL_REDACT = "partial_redact"
CASE3_ACTION_EXCLUDE = "exclude"
CASE3_ACTION_ESCALATE = "escalate"

CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA = "special_category_health_data"
CASE3_REASON_CODE_MIXED_SENSITIVE_THIRD_PARTY_DATA = "mixed_sensitive_third_party_data"
CASE3_REASON_CODE_REQUIRES_HUMAN_BALANCING = "requires_human_balancing"
CASE3_REASON_CODES = {
    CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
    CASE3_REASON_CODE_MIXED_SENSITIVE_THIRD_PARTY_DATA,
    CASE3_REASON_CODE_REQUIRES_HUMAN_BALANCING,
}

EMPLOYEE_NAMES = [
    "Alex Chen",
    "Jordan Murphy",
    "Priya Nair",
    "Sam Okafor",
    "Taylor Brooks",
    "Morgan Silva",
    "Casey Zhang",
    "Riley Hassan",
]

MANAGER_NAMES = [
    "Sarah Mitchell",
    "David Okonkwo",
    "Emma Johansson",
    "James Reyes",
    "Fatima Al-Hassan",
    "Liam O'Brien",
    "Yuki Tanaka",
    "Aisha Patel",
]

COLLEAGUE_NAMES = [
    "Carlos Reyes",
    "Sophie Dubois",
    "Raj Mehta",
    "Olivia Brown",
    "Hassan Yilmaz",
    "Sofia Andersson",
    "Chen Wei",
    "Isabella Garcia",
]

PROJECT_NAMES = [
    "Phoenix",
    "Atlas",
    "Orion",
    "Nexus",
    "Helios",
    "Titan",
]

HEALTH_CONDITIONS = [
    "anxiety",
    "stress-related illness",
    "burnout",
    "depression",
    "mental health difficulties",
    "exhaustion managing this situation",
]

PIPELINE_NAMES = [
    "Jenkins",
    "GitHub Actions",
    "CircleCI",
    "ArgoCD",
    "Terraform",
]

CASE3_INTERNAL_HR_CODES = [
    "HR-PERF-2024-Q3",
    "PERF-TRACK-441",
    "HR-ESC-2290",
    "REVIEW-FLAG-88",
    "CONDUCT-REF-124",
]

SALARY_PHRASES = [
    "my salary review is next week and I'm nervous about it",
    "my pay review is coming up and I'm honestly anxious",
    "I have a compensation review next week that I'm worried about",
    "my annual review is next week and I'm not sure how it will go",
    "my compensation discussion is coming up and it has me on edge",
    "I'm quietly stressing about my pay conversation next week",
]

PR_PHRASES = [
    "can you take a look at my PR for the {project} service",
    "could you review my pull request for {project} when you have time",
    "would you mind looking over my PR for {project}",
    "can you check my PR for the {project} module",
    "can you sanity-check my changes for {project}",
    "when you get a minute, can you review the {project} patch",
]

HEALTH_TRAP_TEMPLATES = [
    "{requester} missed the {project} delivery again this sprint. "
    "I've honestly been struggling with {health_condition} myself trying to manage this situation.",
    "{requester} hasn't hit the {project} targets again. "
    "Honestly, dealing with my own {health_condition} while handling this has been difficult.",
    "Another missed deadline from {requester} on {project}. "
    "I want to flag that my own {health_condition} is making this harder to manage than usual.",
    "{requester}'s output on {project} is below what we expected. "
    "I should also note I've been managing some {health_condition} that is affecting my capacity.",
    "{requester} is still behind on the {project} milestones. "
    "I'm dealing with some {health_condition} and it is affecting how I am handling this.",
    "We need to talk about {requester}'s delivery issues on {project}. "
    "I have my own {health_condition} in the background and it is making this harder to navigate.",
]

REQUESTER_CLEAN_TEMPLATES = [
    "Just pushed the {project} pipeline fix. Config is live at /deploy/prod/{project_lower}-v2.",
    "Pushed my fix for the {project} service. Pipeline should be green now.",
    "The {project} build issue is resolved. Deployed to staging, running smoke tests now.",
    "Fixed the {project} integration bug. PR merged, deployment running.",
]

THREAD_REPLY_TEMPLATES = [
    "Looks clean, merging to main.",
    "Approved. Deploying now.",
    "Good to go from my side.",
    "Reviewed it. Shipping this one.",
]

BOT_TEMPLATES = [
    "Deployment successful: {project}-v{build_num} deployed to {region} at {timestamp}",
    "Build #{build_num} passed. {project} deployed to {env}.",
    "Pipeline complete: {project} v{build_num} - all checks green.",
    "Deploy finished: {project}-v{build_num} is live in {env}.",
]

MANAGER_PERF_TEMPLATES = [
    "Flagging {requester}'s Q3 performance for HR review. Output consistently below target.",
    "HR flag: {requester}'s performance this quarter needs formal review.",
    "Escalating {requester}'s performance metrics to HR. Pattern of missed targets.",
    "{requester}'s output this quarter is formally flagged for performance review.",
    "We should bring {requester}'s quarter into the HR review discussion.",
    "{requester}'s quarter needs a formal HR discussion before we close it out.",
    "I think {requester}'s recent quarter belongs in the next HR review cycle.",
]

CASE3_DISTRACTOR_TEMPLATES = [
    "{colleague} asked if we can move the {other_project} rollout to next week.",
    "Reminder: the infra review for {other_project} is still on Thursday afternoon.",
    "{manager} said the budget note for {other_project} can wait until next sprint.",
    "The {other_project} checklist is done. Nothing else needed on that thread.",
]

CASE3_ESCALATION_HEALTH_KEYWORDS = [
    "health",
    "anxiety",
    "medical",
    "burnout",
    "depression",
    "stress",
    "mental health",
    "special health",
]

CASE3_ESCALATION_LEGAL_KEYWORDS = [
    "article 9",
    "special category",
    "special-category",
    "special category data",
    "sensitive personal data",
]

CASE3_REASON_CODE_BONUS = 0.10
CASE3_ESCALATION_KEYWORD_BONUS = 0.05
CASE3_CALIBRATION_BONUS = 0.05

CASE3_DSAR_TEMPLATE = (
    "Subject: Data Subject Access Request\n"
    "From: counsel@employment-law.example\n"
    "Date: {request_date}\n\n"
    "Dear Privacy Team,\n\n"
    "I act for {requester_name}. IT has already surfaced the attached candidate "
    "Slack messages from the broader export as potentially responsive to this request.\n\n"
    "Please review and disclose all candidate Slack messages from the last 12 months "
    "that relate to {requester_name}, username {requester_username}, employee ID "
    "{employee_id}, or the #{team_channel} team channel, subject to any lawful "
    "redactions or escalations required under data protection law.\n\n"
    "Kind regards,\n"
    "Employment Counsel"
)
