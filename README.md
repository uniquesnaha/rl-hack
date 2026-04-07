# DSAR-OpenEnv: Privacy Compliance RL Environment

DSAR-OpenEnv is a real-world OpenEnv benchmark for **Data Subject Access Request (DSAR)** handling under GDPR / UK GDPR style privacy operations. It is designed for training and evaluating agents on a workflow that compliance, legal operations, privacy engineering, and support teams actually perform: deciding what data to disclose, what to withhold, when to verify identity, how to redact third-party data, and when to escalate high-risk material.

The deployable OpenEnv project lives in [`dsar_env`](d:/rl-hack/dsar_env).

This is not a toy workflow. The environment models practical compliance work where the agent must balance:

- requester access rights
- internal-only operational data
- third-party privacy protection
- special-category escalation
- deadline pressure and completion discipline

## Why This Environment Matters

DSAR handling is a strong RL benchmark domain because it combines:

- structured data access decisions
- partial observability
- sequential workflow gates
- asymmetric costs for privacy mistakes
- realistic catastrophic failure modes

The environment is useful both as a benchmark and as a training curriculum for privacy/compliance agents.

## Tasks

| Task | Difficulty | What the agent must do | Main skills tested |
| --- | --- | --- | --- |
| `task_easy` | Easy | Query billing + CRM silos and classify 17 customer fields as disclose vs withhold | field semantics, basic disclosure policy |
| `task_medium` | Medium | Verify identity proportionately, then redact support tickets sentence by sentence | verification strategy, sentence-level privacy filtering |
| `task_hard` | Hard | Triage six candidate Slack messages from a workplace-dispute DSAR | thread reasoning, mixed ownership, bot exclusion, escalation judgment |

### Task 1: Consumer Record Classification

The agent receives a straightforward customer DSAR and must inspect structured fields drawn from billing and CRM systems. Some fields are clearly requester-owned data; others are internal operational or analytical fields that should not be disclosed.

Representative requester-owned fields include:

- `full_name`
- `email`
- `billing_address`
- `payment_history`
- `support_ticket_ids`
- `referral_credit_balance`

Representative internal-only fields include:

- `customer_health_score`
- `risk_score`
- `churn_probability`
- `lead_source_tag`
- `shard_routing_key`
- `account_manager_notes`
- `campaign_cpa`

### Task 2: Identity Verification + Ticket Redaction

The requester provides near-match identity details rather than a perfect account match. The agent must:

1. query the relevant silos for masked evidence
2. choose a **proportionate** verification method
3. unlock the support-ticket corpus
4. redact staff PII and internal-only content while preserving requester-owned text

This task is designed as a sequential workflow rather than a one-shot classification problem.

### Task 3: Slack DSAR Triage

The hard task models a workplace-dispute DSAR where IT has already surfaced a six-message candidate set from a broader Slack export. The agent must decide, for each candidate message, whether to:

- disclose it
- send it to sentence-level redaction
- exclude it
- escalate it with a structured reason code and free-text justification

The candidate set contains:

- a requester-authored technical message that is still disclose-worthy
- a mixed-ownership message requiring sentence-level splitting
- a bot/system deployment message that should be excluded
- a thread reply that depends on parent-message context
- a requester-entitled HR/performance message
- a special-category health trap that must be escalated rather than disclosed

## Action Space

| Action | Parameters | Used in | Description |
| --- | --- | --- | --- |
| `query_silo` | `silo_name` | Easy, Medium | Query `billing` or `crm` |
| `classify_field` | `field_id`, `decision` | Easy | Mark a structured field as `disclose` or `withhold` |
| `verify_identity` | `verification_method` | Medium | Attempt a proportionate or disproportionate verification method |
| `redact_span` | `ticket_id`, `sentence_index`, `decision` | Medium | Keep or redact one ticket sentence |
| `process_message` | `msg_id`, `action_label` | Hard | Choose `disclose`, `partial_redact`, `exclude`, or `escalate` |
| `redact_sentence` | `msg_id`, `sentence_index`, `decision` | Hard | Keep or redact one sentence in a mixed-ownership Slack message |
| `escalate_with_reason` | `msg_id`, `reason_code`, `reason` | Hard | Supply structured + free-text escalation justification |
| `compile_response` | none | All | Finalize the episode |

## Observation Space

Common fields across tasks:

| Field | Type | Description |
| --- | --- | --- |
| `episode_id` | string | stable episode identifier |
| `task_id` | string | current task |
| `dsar_request` | string | request text shown to the agent |
| `available_actions` | list[str] | action types currently valid |
| `draft_response` | dict | current output under construction |
| `audit_trail` | list | ordered step history |
| `deadline_pressure` | float | time pressure signal |
| `steps_remaining` | int | steps left in episode |
| `compile_ready` | bool | whether `compile_response` is currently valid |
| `terminal_details` | dict | task-specific terminal metrics on completion |

Task-specific fields:

- Easy
- `customer_record`
- `silo_results`
- `classified_fields`
- Medium
- `phase`
- `identity_confidence`
- `identity_threshold`
- `submitted_identity`
- `internal_identity`
- `tickets`
- `processed_sentences`
- `pending_sentence_count`
- `total_sentence_count`
- `completion_coverage`
- Hard
- `slack_export`
- `users_json`
- `processed_messages`
- `escalation_log`
- `escalation_reason_codes`
- `messages_pending`
- `sentences_pending`

## Reward Design

### Easy

- positive reward for correct disclose/withhold decisions
- negative reward for withholding requester-owned data
- stronger negative reward for leaking internal-only data
- terminal score based on disclosure precision/recall, privacy penalties, and light efficiency terms

### Medium

- positive reward for querying useful silos
- positive reward for proportionate verification
- negative reward for disproportionate verification
- sentence-level reward for keep/redact correctness
- terminal score combines identity quality, redaction quality, and completion coverage

### Hard

- positive reward for correct top-level message routing
- positive reward for correct sentence-level splitting
- reward bonus for structured escalation reasons
- terminal score combines:
- top-level message accuracy
- sentence redaction quality
- escalation quality
- calibration penalties for especially bad top-level decisions

## Design Decisions and Calibration Notes

### Hard task ruggedness

`task_hard` is intentionally rugged. Episodes that mishandle the most important compliance decisions can score very low or clip to `0.0`. This is deliberate: the benchmark is meant to distinguish mild mistakes from severe regulatory failures.

### Bimodal hard-task behavior

The hard task often produces a bimodal baseline pattern. Some trajectories land in a modest hard-performance band; others collapse to `0.0` after a catastrophic or compounded failure pattern. This is a feature of the environment design, not a symptom of flat grading.

### Catastrophic vs compounded failure

Hard zeros can happen in two ways:

- direct catastrophic failure, especially on special-category handling
- multiple major compliance-action errors in the same episode, such as failing escalation, disclosing a bot/system message, excluding requester-entitled data, or over-redacting mixed ownership content

### Hosted-model variance vs environment determinism

Environment-side generation and grading are deterministic for a fixed seed. However, hosted LLM providers can still introduce rerun variance even at `temperature=0.0`. In practice this means:

- fixed seeds keep the **episode** stable
- the external hosted model may still vary slightly in **behavior**

### Strict compile gate

The compile gate requires the agent to finish all pending work before submission. This is intentional and models a real compliance workflow: a DSAR response is not valid if the triage/redaction/escalation process is incomplete.

## Baseline Configuration

The baseline script is [`dsar_env/inference.py`](d:/rl-hack/dsar_env/inference.py) and uses the OpenAI client against an OpenAI-compatible endpoint.

Representative default per-task seeds:

```text
task_easy: 7
task_medium: 3
task_hard: 31
```

Representative hosted-Qwen baseline behavior is:

| Task | Typical band |
| --- | --- |
| `task_easy` | around `0.95` |
| `task_medium` | roughly `0.43` to `0.56` |
| `task_hard` | roughly `0.0` to `0.30`, with occasional low-end collapse on severe failures |

## Local Setup

### Install

```bash
cd dsar_env
pip install -e .
```

### Run the environment server locally

```bash
cd dsar_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the baseline locally

```bash
cd dsar_env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct:fastest
HF_TOKEN=your_token_here
DSAR_ENV_URL=http://localhost:8000
python inference.py
```

## Hugging Face Space Deployment

The project is designed to run as a Docker-based HF Space from the contents of [`dsar_env`](d:/rl-hack/dsar_env).

For remote baseline runs against your deployed Space, set:

```bash
DSAR_ENV_URL=https://<your-space>.hf.space
```

The inference script then targets the deployed environment instead of a local uvicorn instance.

## Docker

```bash
cd dsar_env
docker build -t dsar-env:latest .
docker run -p 8000:8000 dsar-env:latest
```

## Validation

Before submitting, verify:

1. your HF Space responds to `POST /reset`
2. `docker build` succeeds
3. `openenv validate` succeeds
4. `python inference.py` completes and emits only structured `[START]`, `[STEP]`, and `[END]` logs on stdout

## Repository Layout

```text
rl-hack/
  README.md
  dsar_env/
    inference.py
    models.py
    client.py
    openenv.yaml
    Dockerfile
    server/
      app.py
      constants.py
      generator.py
      grader.py
      dsar_environment.py
    tests/
```

## Why This Is Novel

This environment is not a generic "privacy" toy task. It combines:

- structured disclosure policy
- workflow gating
- sentence-level redaction
- thread-aware message interpretation
- legal escalation with structured reason codes

That combination makes it both practically useful and scientifically interesting for RL-style post-training.

## License

BSD 3-Clause License
