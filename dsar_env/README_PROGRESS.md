# DSAR-OpenEnv Progress README

This file is the current project status README.

The existing `README.md` is being kept as an older sample/reference document.
If you prepare a final submission, this progress README should be the one you
polish and promote into the main submission-facing README.

## Current Status

- `task_easy` is fully implemented and tested.
- `task_medium` is now implemented as a two-phase identity verification plus support-ticket redaction task.
- `task_hard` is now implemented as candidate-set Slack compliance triage with thread resolution, sentence-level redaction, bot exclusion, and calibrated escalation.
- `inference.py` now supports all three tasks via the `DSAR_TASKS` environment variable and defaults to running all three.

## What Works Today

- OpenEnv-compatible FastAPI environment server
- Root-level `openenv.yaml`
- Root-level `Dockerfile`
- Deterministic Case 1 generator and grader
- Deterministic Case 2 generator and grader
- Deterministic Case 3 Slack-triage generator and grader
- Progressive field reveal through `query_silo`
- Two-phase Case 2 flow with identity confidence and ticket redaction
- Case 3 message-level triage with sentence-level Slack redaction and escalation reasons
- Typed action and observation models
- Baseline inference loop using an OpenAI-compatible client
- Hugging Face router-style default configuration in `inference.py`
- Optional `EPISODE_SEED` for reproducible local debugging

## Current Task Design

`task_easy` models a clean consumer DSAR with 17 fields:

- 10 requester-disclosable fields
- 7 internal-only fields

The action loop is:

1. `query_silo billing`
2. `query_silo crm`
3. `classify_field <field_id> <disclose|withhold>`
4. `compile_response`

The grader combines:

- field classification quality
- leak penalties
- step efficiency
- silo efficiency

`task_medium` models an identity-mismatch DSAR followed by support-ticket redaction:

- Phase 1: query masked billing/CRM identity evidence and verify proportionately
- Phase 2: review sentence-level ticket content with `redact_span`
- Terminal score: `0.30 * identity_score + 0.70 * redaction_score`

`task_hard` models Slack candidate-set compliance triage for a workplace-dispute DSAR:

- Candidate Slack messages are already surfaced from the broader export
- Agent must resolve aliased user IDs, follow `thread_ts`, redact mixed-ownership sentences, exclude bot output, and escalate special-category entanglement with a legally meaningful reason
- Terminal score combines normalized message accuracy, sentence-redaction quality, and escalation quality with explicit privacy-breach penalties

## Important Honesty Notes

- The repository now has real implementations for all 3 tasks.
- Baseline score ranges should be re-measured end-to-end with the current environment dependencies before any final submission README claims exact numbers.

## Local Run

Start the environment server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run the current baseline:

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct:fastest \
HF_TOKEN=hf_your_token \
python inference.py
```

Optional reproducible run:

```bash
EPISODE_SEED=42 python inference.py
```

Run all three tasks together:

```bash
DSAR_TASKS=task_easy,task_medium,task_hard python inference.py
```

## Recommended Next Steps

1. Re-measure end-to-end baseline scores for all three tasks in a fully provisioned environment.
2. Align `inference.py` stdout with the exact hackathon logging contract if required for submission.
3. Replace the sample `README.md` with a final submission README once deployment validation is complete.
