# DSAR-OpenEnv Progress README

This file is the current project status README.

The existing `README.md` is being kept as an older sample/reference document.
If you prepare a final submission, this progress README should be the one you
polish and promote into the main submission-facing README.

## Current Status

- `task_easy` is fully implemented and tested.
- `task_medium` is now implemented as a two-phase identity verification plus support-ticket redaction task.
- `task_hard` is scaffolded only and currently falls back to the Case 1 generator.
- `inference.py` defaults to `task_easy`, but can run medium as well via the `DSAR_TASKS` environment variable.

## What Works Today

- OpenEnv-compatible FastAPI environment server
- Root-level `openenv.yaml`
- Root-level `Dockerfile`
- Deterministic Case 1 generator and grader
- Deterministic Case 2 generator and grader
- Progressive field reveal through `query_silo`
- Two-phase Case 2 flow with identity confidence and ticket redaction
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

## Important Honesty Notes

- The repository is not yet fully submission-ready for a 3-task claim.
- `task_medium` is real, but `task_hard` is still a planned task direction rather than a finished task.
- Any document that describes all 3 tasks as fully implemented is currently ahead
  of the code.

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

Run easy + medium together:

```bash
DSAR_TASKS=task_easy,task_medium python inference.py
```

## Recommended Next Steps

1. Implement a real `task_hard` generator, observation flow, and grading path.
2. Expand `inference.py` default task selection once all required tasks are real.
3. Replace the sample `README.md` with a final honest submission README.
