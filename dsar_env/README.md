# DSAR-OpenEnv: GDPR Data Subject Access Request Compliance Environment

A reinforcement learning environment for training AI agents to process **Data Subject Access Requests (DSARs)** under GDPR/UK GDPR compliance. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Motivation

Data Subject Access Requests are **legally mandated** under GDPR Article 15 in 30+ countries. Every organisation that processes personal data must respond within 30 days.

- **Cost**: Manual DSAR processing costs **$1,500+ per request** ([source](https://secureprivacy.ai))
- **Volume**: DSAR volumes are growing **43% year-over-year** as data awareness increases
- **Risk**: Incorrect handling leads to regulatory fines (up to 4% of global revenue) and legal liability
- **Complexity**: Each DSAR requires classifying dozens of data fields as personal data (must disclose) vs internal business data (must withhold), verifying requester identity, and redacting third-party information

This environment trains RL agents to automate the most operationally complex parts of DSAR compliance.

## Tasks

| Task | Difficulty | Description | Baseline LLM Score |
|------|-----------|-------------|-------------------|
| `task_easy` | Easy | Clean consumer request — classify 16 fields as personal vs internal data | 0.65–0.75 |
| `task_medium` | Medium | Mismatched identity verification + support ticket redaction at sentence level | 0.45–0.55 |
| `task_hard` | Hard | Weaponised employee DSAR on Slack export with mixed-ownership messages | 0.20–0.35 |

### Task 1: Clean Consumer Request (Easy)

A customer submits a straightforward DSAR. The agent receives a merged record with 16 fields from billing and CRM databases — 9 fields are personal data (name, email, payment history, etc.) and 7 are internal business data (risk scores, churn probability, infrastructure keys, etc.). The agent must classify each field correctly.

**Skills tested**: Semantic understanding of data field types, distinguishing personal from operational data.

### Task 2: Mismatched Identity + Support Tickets (Medium)

A former customer submits a DSAR from a different email than their account. The agent must first verify identity through proportionate means (not requesting passport/photo ID), then process support ticket transcripts at sentence granularity, redacting staff PII while preserving the customer's content.

**Skills tested**: Identity verification proportionality, sentence-level PII detection, sequential decision-making.

### Task 3: Weaponised Employee DSAR on Slack (Hard)

A former employee's lawyer submits a broad DSAR requesting all Slack messages mentioning them. The agent processes a Slack JSON export with aliased user IDs, threaded messages, bot messages, and sentences containing mixed ownership (requester data + colleague salary info in one sentence; manager performance feedback + manager's health disclosure in another).

**Skills tested**: User ID resolution, thread context tracking, Article 9 special-category escalation, sentence-level ownership splitting.

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `query_silo` | `silo_name`: `"billing"` or `"crm"` | Query a data silo |
| `classify_field` | `field_id`: field name, `decision`: `"disclose"` or `"withhold"` | Classify a field |
| `compile_response` | *(none)* | Finalize and submit response |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `dsar_request` | string | The DSAR request letter |
| `merged_record` | dict | All data fields and their values |
| `available_actions` | list[str] | Available action types |
| `silo_results` | list[str] | Silos queried so far |
| `identity_verified` | bool | Whether identity has been confirmed |
| `draft_response` | dict | Fields selected for disclosure |
| `audit_trail` | list[str] | Action log |
| `deadline_pressure` | float | 1.0 (fresh) → 0.0 (deadline) |
| `steps_remaining` | int | Steps left in episode (max 30) |
| `classified_fields` | list[str] | Fields already classified |
| `done` | bool | Whether episode has ended |
| `reward` | float | Reward from last action |

## Reward Design

### Step-Level Rewards (immediate, per action)

| Action | Result | Reward |
|--------|--------|--------|
| Query valid silo (first time) | New data accessed | +0.05 |
| Query silo (redundant) | Wasted step | -0.05 |
| Query invalid silo | Hallucinated silo | -0.05 |
| Classify correctly (disclose personal) | Correct | +0.10 |
| Classify correctly (withhold internal) | Correct | +0.10 |
| **Leak internal data** (disclose internal) | **Data breach** | **-0.30** |
| Withhold personal data wrongly | Compliance failure | -0.15 |
| Steps beyond free allowance (10) | Deadline pressure | -0.01/step |

### Terminal Score (on compile_response)

```
F1 = harmonic_mean(precision, recall)
privacy_penalty = leaked_count × 0.30 × (1 + leaked_count × 0.50)
silo_bonus = max(0, 0.05 - unnecessary_queries × 0.02)
score = clamp(F1 - privacy_penalty + silo_bonus, 0.0, 1.0)
```

The **non-linear privacy penalty** makes each additional leak progressively more expensive:
- 1 leak → 0.45 penalty
- 2 leaks → 1.20 penalty (floors score to 0.0)
- 3 leaks → 2.25 penalty

## Setup & Usage

### Install

```bash
pip install openenv-core[core]
cd dsar_env
pip install -e .
```

### Run Locally

```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline agent
API_BASE_URL=https://api.groq.com/openai/v1 \
MODEL_NAME=llama-3.3-70b-versatile \
HF_TOKEN=your_key \
python inference.py
```

### Run with Docker

```bash
docker build -t dsar-env:latest .
docker run -p 8000:8000 dsar-env:latest
```

### Deploy to HuggingFace Spaces

```bash
openenv push --repo-id your-username/dsar-env
```

## Baseline Scores

| Task | Expected Range | Description |
|------|---------------|-------------|
| `task_easy` | 0.65–0.75 | LLM handles obvious fields but leaks 1-2 operational metrics |
| `task_medium` | 0.45–0.55 | Verification usually OK but sentence-level redaction is imprecise |
| `task_hard` | 0.20–0.35 | Thread resolution fails, Article 9 escalation missed |

## For RL Researchers

This environment is designed for RL post-training. Wrap it with a Gymnasium adapter:

```python
from dsar_env import DSAREnv

# Connect to running server
with DSAREnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="task_easy")
    
    while not obs.observation.done:
        action = your_policy(obs)  # Your RL agent
        obs = env.step(action)
    
    print(f"Score: {obs.reward}")
```

Compatible with: **TRL**, **torchforge**, **SkyRL**, **ART**, **Oumi**.

## License

BSD 3-Clause License
