---
title: Dsar Env Environment Server
emoji: 🔊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AutoDSAR

AutoDSAR is a deterministic OpenEnv environment for training agents on safety-critical privacy operations around Data Subject Access Requests (DSARs).

The current environment includes five tasks:
- `task_easy`: structured customer-record disclosure classification
- `task_medium`: identity verification plus sentence-level support-ticket redaction
- `task_adversarial_identity`: adversarial identity review with spoof detection
- `task_hard`: Slack compliance triage with special-category escalation
- `task_breach_embedded`: compact DSAR review with embedded breach detection and ordered notification workflow

## Why this is an RL environment

AutoDSAR is sequential, partially observable, and safety-critical:
- the agent must gather evidence before acting
- the correct next action depends on hidden episode state
- wrong actions can worsen compliance state and trigger recovery work
- reward and safety cost are tracked separately
- workflow state is exposed explicitly for training and debugging

## Tasks

### `task_easy`
Query `billing` and `crm`, reveal the structured customer record, then classify each field as `disclose` or `withhold`.

### `task_medium`
Perform proportionate identity verification first, then redact support-ticket sentences one by one while preserving requester-owned content and removing third-party/internal content.

### `task_adversarial_identity`
Review a suspicious identity request, query both silos, then either verify a genuine requester proportionately or flag a spoofed requester with a concrete reason.

### `task_hard`
Triage a candidate Slack export using `disclose`, `partial_redact`, `exclude`, or `escalate`, with special-category health data acting as the main legal trap.

### `task_breach_embedded`
Handle a compact DSAR record while detecting whether the request also contains a breach signal. If a breach is present, the agent must flag it and notify regulator then requester before compile.

## Action space

- `query_silo`
- `classify_field`
- `verify_identity`
- `flag_adversarial`
- `flag_breach_signal`
- `notify_regulator`
- `notify_requester`
- `redact_span`
- `process_message`
- `redact_sentence`
- `escalate_with_reason`
- `file_remediation_note`
- `justify_verification_method`
- `file_redaction_remediation`
- `compile_response`

## Observation highlights

All tasks expose:
- `episode_id`
- `task_id`
- `available_actions`
- `audit_trail`
- `steps_remaining`
- `compile_ready`
- `current_compliance_state`
- `workflow_state`
- `step_safety_cost`
- `episode_safety_cost`
- `constraint_events`

Task-specific fields:
- easy: `customer_record`, `classified_fields`, `silo_results`
- medium: `phase`, `identity_confidence`, `tickets`, `processed_sentences`
- adversarial identity: `submitted_identity`, `internal_identity`, `identity_confidence`
- hard: `slack_export`, `users_json`, `processed_messages`, `messages_pending`, `sentences_pending`
- breach embedded: `breach_detected`, `regulator_notified`, `requester_notified`, `breach_scope_fields`, `breach_signal_context`

## Reward and safety

AutoDSAR keeps task reward and safety cost separate:
- `reward` captures progress and task quality
- `step_safety_cost` captures immediate privacy/compliance harm
- `episode_safety_cost` accumulates harm across the episode

Examples of safety events:
- internal data leak
- disproportionate verification
- false-positive rejection
- identity spoof accepted
- false breach report
- breach signal missed
- requester notice missed
- third-party disclosure
- special-category near miss
- special-category disclosure
- unsafe compile

## Training-oriented features

The environment includes:
- reactive compliance-risk states
- workflow-state IDs
- deterministic difficulty tiers
- procedural scenario variants
- diagnosis/reasoning reward
- optional potential-based shaping
- optional trajectory export for imitation and offline RL

Trajectory export can be enabled with:
- `DSAR_EXPORT_TRAJECTORIES=true`
- `DSAR_TRAJECTORY_EXPORT_PATH=dsar_trajectories.jsonl`

## Quickstart

### Install

```bash
cd dsar_env
pip install -e .[dev]
```

### Run the environment server

```bash
cd dsar_env
uvicorn server.app:app --host 127.0.0.1 --port 8010
```

### Run the baseline client

```bash
cd dsar_env
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct:fastest \
HF_TOKEN=your_hf_token \
DSAR_ENV_URL=http://localhost:8010 \
DSAR_TASKS=task_easy,task_medium,task_adversarial_identity,task_hard,task_breach_embedded \
python inference.py
```

## Useful environment variables

- `DSAR_TASKS`
- `DSAR_TASK_SEEDS`
- `DSAR_TRACE`
- `DSAR_ENABLE_POTENTIAL_SHAPING`
- `DSAR_EXPORT_TRAJECTORIES`
- `DSAR_TRAJECTORY_EXPORT_PATH`

## Project structure

```text
rl-hack/
|-- README.md
`-- dsar_env/
    |-- inference.py
    |-- models.py
    |-- openenv.yaml
    |-- server/
    |   |-- app.py
    |   |-- constants.py
    |   |-- dsar_environment.py
    |   |-- generator.py
    |   `-- grader.py
    `-- tests/
        |-- test_adversarial_identity.py
        |-- test_breach_embedded.py
        |-- test_case1.py
        |-- test_case2.py
        |-- test_case3.py
        |-- test_inference_helpers.py
        |-- test_phase2_enhancements.py
        |-- test_phase3_training_upgrades.py
        `-- test_reactive_states.py
```
