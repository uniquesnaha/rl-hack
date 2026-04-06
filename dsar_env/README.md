---
title: DSAR Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
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
| `task_easy` | Easy | Clean consumer request — classify 17 fields as personal vs internal data | 0.45–0.60 |
| `task_medium` | Medium | Mismatched identity verification + support ticket redaction at sentence level | 0.45–0.55 |
| `task_hard` | Hard | Candidate-set Slack DSAR triage with thread resolution, mixed ownership, and Article 9 escalation | Pending re-measurement |

### Task 1: Clean Consumer Request (Easy)

A customer submits a straightforward DSAR. The agent receives a merged record with 17 fields from billing and CRM databases — 10 fields are personal data (name, email, payment history, referral credits, etc.) and 7 are internal business data (customer health score, engagement index, lifetime value estimate, infrastructure keys, etc.). The agent must classify each field correctly.

**Skills tested**: Semantic understanding of data field types, distinguishing personal from operational data.

### Task 2: Mismatched Identity + Support Tickets (Medium)

A former customer submits a DSAR from a different email than their account. The agent must first verify identity through proportionate means (not requesting passport/photo ID), then process support ticket transcripts at sentence granularity, redacting staff PII while preserving the customer's content.

**Skills tested**: Identity verification proportionality, sentence-level PII detection, sequential decision-making.

### Task 3: Weaponised Employee DSAR on Slack (Hard)

A former employee's lawyer submits a broad workplace-dispute DSAR. IT has already surfaced a candidate set of six potentially responsive Slack messages from the broader export. The agent must triage those candidate messages using a Slack JSON export with aliased user IDs, threaded replies, bot messages, and mixed-ownership sentences (requester data + colleague salary info in one message; manager performance feedback + manager health disclosure in another).

**Skills tested**: User ID resolution, thread context tracking, Article 9 special-category escalation, sentence-level ownership splitting.

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `query_silo` | `silo_name`: `"billing"` or `"crm"` | Query a data silo |
| `classify_field` | `field_id`: field name, `decision`: `"disclose"` or `"withhold"` | Classify a field |
| `verify_identity` | `verification_method` | Proportionate identity verification for `task_medium` |
| `redact_span` | `ticket_id`, `sentence_index`, `decision` | Sentence redaction for `task_medium` |
| `process_message` | `msg_id`, `action_label` | Case 3 message-level triage: disclose, partial_redact, exclude, or escalate |
| `redact_sentence` | `msg_id`, `sentence_index`, `decision` | Case 3 sentence-level keep/redact decision for mixed-ownership Slack messages |
| `escalate_with_reason` | `msg_id`, `reason` | Case 3 legal justification for an escalated message |
| `compile_response` | *(none)* | Finalize and submit response |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `dsar_request` | string | The DSAR request letter |
| `merged_record` | dict | Currently revealed data fields and their values |
| `available_actions` | list[str] | Available action types |
| `silo_results` | list[str] | Silos queried so far |
| `identity_verified` | bool | Whether identity has been confirmed |
| `draft_response` | dict | Fields selected for disclosure |
| `audit_trail` | list[str] | Action log |
| `deadline_pressure` | float | 1.0 (fresh) → 0.0 (deadline) |
| `steps_remaining` | int | Steps left in episode (max 30) |
| `classified_fields` | list[str] | Fields already classified |
| `tickets` | list | Case 2 support-ticket corpus after verification |
| `processed_sentences` | dict | Case 2 sentence decisions |
| `slack_export` | list | Case 3 candidate Slack messages with stable sentence indices |
| `users_json` | dict | Case 3 visible Slack user mapping |
| `processed_messages` | dict | Case 3 message decisions |
| `escalation_log` | dict | Case 3 escalation reasons |
| `messages_pending` | list[str] | Case 3 message IDs still awaiting triage |
| `sentences_pending` | dict | Case 3 unresolved sentence indices |
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
privacy_penalty = leaked_count × 0.30 × (1 + leaked_count × 0.45)
compliance_score = clamp(F1 - privacy_penalty, 0.0, 1.0)
step_efficiency = clamp((MAX_STEPS - steps_used) / (MAX_STEPS - MIN_STEPS_CASE1), 0.0, 1.0)
silo_efficiency = max(0, 1.0 - 0.5 * missing_required_silos - 0.25 * extra_silos)
efficiency_score = 0.5 * step_efficiency + 0.5 * silo_efficiency
score = 0.0 if compliance_score == 0 else clamp(0.9 * compliance_score + 0.1 * efficiency_score, 0.0, 1.0)
```

The **non-linear privacy penalty** makes each additional leak progressively more expensive:
- 1 leak → 0.435 penalty
- 2 leaks → 1.14 penalty (floors score to 0.0)
- 3 leaks → 2.115 penalty

This formula keeps **compliance accuracy primary**, while still explicitly rewarding step efficiency and correct silo usage.

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
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
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

Baseline score bands should be re-measured end-to-end with the current
environment dependencies before final submission. The environment now supports
all three tasks, including the implemented Case 3 hard task.

| Task | Expected Range | Description |
|------|---------------|-------------|
| `task_easy` | 0.45–0.60 | LLM handles obvious fields but still leaks or misses some operational metrics |
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
