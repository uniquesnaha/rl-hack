#!/usr/bin/env python3
"""
DSAR Environment - Baseline Inference Script.

This script demonstrates the DSAR environment by running an LLM agent
against the currently enabled task set and reporting scores.

Environment variables:
    API_BASE_URL   - Optional LLM endpoint override.
                     Default: https://router.huggingface.co/v1
    MODEL_NAME     - Optional model override.
                     Default: Qwen/Qwen2.5-72B-Instruct:fastest
    OPENAI_API_KEY - Primary API key for direct OpenAI usage
    HF_TOKEN       - Optional fallback for Hugging Face router usage
    EPISODE_SEED   - Optional fixed seed for reproducible environment resets
    DSAR_TASKS     - Comma-separated task list override (default: task_easy)
    DSAR_TRACE     - Set to 1/true/on to print detailed debug logs
    DSAR_MULTI_SEED - Comma-separated seed list for calibration runs

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct:fastest \
    HF_TOKEN=your_hf_token \
    python inference.py
"""

import json
import os
import re
import sys
import time

from openai import OpenAI


def _parse_optional_int(raw_value: str | None) -> int | None:
    """Parse an optional integer environment variable."""
    if raw_value in (None, ""):
        return None
    return int(raw_value)


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = 0.0  # keep deterministic when a fixed seed is provided
MAX_TOKENS = 512
MAX_STEPS = 30
EPISODE_SEED = _parse_optional_int(os.environ.get("EPISODE_SEED"))
TASK_IDS = [task.strip() for task in os.environ.get("DSAR_TASKS", "task_easy").split(",") if task.strip()]
TRACE_ENABLED = os.environ.get("DSAR_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
MULTI_SEED_VALUES = [
    seed.strip()
    for seed in os.environ.get("DSAR_MULTI_SEED", "").split(",")
    if seed.strip()
]

# OpenAI-compatible client pointing at the configured LLM provider.
def _select_api_key(base_url: str) -> str:
    lowered = base_url.lower()
    if "huggingface.co" in lowered:
        return HF_TOKEN
    return OPENAI_API_KEY or HF_TOKEN


client = OpenAI(
    base_url=API_BASE_URL,
    api_key=_select_api_key(API_BASE_URL),
)

SYSTEM_PROMPT = """Process the DSAR by emitting exactly one valid action per turn.

Available actions:
- query_silo billing
- query_silo crm
- classify_field <field_id> disclose
- classify_field <field_id> withhold
- verify_identity <transaction_date|account_reference|registered_postcode|passport_copy|photo_id>
- redact_span <ticket_id> <sentence_index> <keep|redact>
- compile_response

Rules:
- For task_easy, query both silos before classifying fields.
- For task_medium identity, use a proportionate verification method.
- For task_medium redaction, process each sentence once and only compile when all are done.
- Keep sentences about the requester's issue, account, access, subscription, invoice outcome, cancellation timing, or workspace.
- Redact staff contact details and internal workflow/process notes.
- Never repeat the same silo query or sentence decision.
- Respond with only the action text."""

FALLBACK_ACTION = "compile_response"


def trace(title: str, payload: object | None = None) -> None:
    """Print optional verbose debug information."""
    if not TRACE_ENABLED:
        return

    print(f"\n[TRACE] {title}")
    if payload is None:
        return

    if isinstance(payload, str):
        print(payload)
        return

    try:
        print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
    except TypeError:
        print(str(payload))


def merged_metadata(step_data: dict, observation: dict) -> dict:
    """Merge top-level and observation metadata for robust terminal reporting."""
    top = step_data.get("metadata", {}) if isinstance(step_data, dict) else {}
    inner = observation.get("metadata", {}) if isinstance(observation, dict) else {}
    merged = {}
    if isinstance(top, dict):
        merged.update(top)
    if isinstance(inner, dict):
        merged.update(inner)
    return merged


def parse_model_action(response_text: str) -> dict:
    """Parse the LLM's text response into a structured action dict."""
    text = response_text.strip()

    match = re.search(r"query_silo\s+(\w+)", text, re.IGNORECASE)
    if match:
        return {
            "action_type": "query_silo",
            "silo_name": match.group(1).lower(),
        }

    match = re.search(
        r"classify_field\s+([\w_]+)\s+(disclose|withhold)",
        text,
        re.IGNORECASE,
    )
    if match:
        return {
            "action_type": "classify_field",
            "field_id": match.group(1).lower(),
            "decision": match.group(2).lower(),
        }

    match = re.search(
        r"verify_identity\s+(transaction_date|account_reference|registered_postcode|passport_copy|photo_id)",
        text,
        re.IGNORECASE,
    )
    if match:
        return {
            "action_type": "verify_identity",
            "verification_method": match.group(1).lower(),
        }

    match = re.search(
        r"redact_span\s+([A-Za-z0-9_-]+)\s+(\d+)\s+(keep|redact)",
        text,
        re.IGNORECASE,
    )
    if match:
        return {
            "action_type": "redact_span",
            "ticket_id": match.group(1),
            "sentence_index": int(match.group(2)),
            "decision": match.group(3).lower(),
        }

    if re.search(r"compile_response", text, re.IGNORECASE):
        return {"action_type": "compile_response"}

    print(f"  [WARN] Could not parse action from: {text[:120]}...")
    return {"action_type": "compile_response"}


def format_observation(obs: dict) -> str:
    """Format an observation dict into a readable string for the LLM."""
    parts = []
    phase = obs.get("phase", "classification")
    task_id = obs.get("task_id", "task_easy")

    steps_remaining = obs.get("steps_remaining", "?")
    classified = set(obs.get("classified_fields", []))
    silos_queried = obs.get("silo_results", [])

    parts.append(f"Task: {task_id}")
    parts.append(f"Phase: {phase}")
    parts.append(f"Steps remaining: {steps_remaining}")
    if silos_queried:
        parts.append(f"Silos already queried: {', '.join(silos_queried)}")
    else:
        parts.append("Silos already queried: NONE - you must query billing and crm first")

    missing_silos = [silo for silo in ("billing", "crm") if silo not in silos_queried]
    if missing_silos:
        parts.append(
            "Required next step: query the remaining silo(s) before any "
            f"classification: {', '.join(missing_silos)}"
        )

    if classified:
        parts.append(
            "\nALREADY CLASSIFIED "
            f"({len(classified)} fields - DO NOT re-classify these):\n  "
            + ", ".join(sorted(classified))
        )

    if obs.get("dsar_request"):
        parts.append(f"\n=== DSAR REQUEST ===\n{obs['dsar_request']}")

    if task_id == "task_medium" and phase == "identity":
        parts.append(
            "\n=== IDENTITY VERIFICATION ===\n"
            f"Confidence: {obs.get('identity_confidence', 0.0):.2f} / "
            f"{obs.get('identity_threshold', 0.0):.2f}"
        )
        submitted_identity = obs.get("submitted_identity", {})
        internal_identity = obs.get("internal_identity", {})
        if submitted_identity:
            parts.append(f"Submitted identity: {json.dumps(submitted_identity, default=str)}")
        if internal_identity:
            parts.append(f"Visible internal identity evidence: {json.dumps(internal_identity, default=str)}")
        parts.append(
            "Available verification methods: transaction_date, account_reference, "
            "registered_postcode, passport_copy, photo_id"
        )

    if task_id == "task_medium" and phase == "redaction":
        parts.append("\n=== REDACTION TICKETS ===")
        processed = obs.get("processed_sentences", {})
        tickets = obs.get("tickets", [])
        pending_count = 0
        total_count = obs.get("total_sentence_count", 0)
        completion_coverage = obs.get("completion_coverage", 0.0)
        compile_ready = obs.get("compile_ready", False)
        parts.append(
            f"Redaction progress: processed {total_count - obs.get('pending_sentence_count', 0)} / {total_count} "
            f"(coverage={completion_coverage:.2f})"
        )
        parts.append(f"Compile ready: {str(compile_ready).lower()}")
        parts.append(
            "Redaction reminder: keep requester issue/account/access/subscription explanations; "
            "redact staff contact details and internal workflow/process notes."
        )
        for ticket in tickets:
            ticket_id = ticket.get("ticket_id", "unknown")
            ticket_category = ticket.get("category", "unknown")
            parts.append(f"Ticket {ticket_id} ({ticket_category})")
            for message in ticket.get("messages", []):
                speaker = message.get("speaker", "unknown")
                for sentence in message.get("sentences", []):
                    sentence_index = sentence.get("sentence_index", -1)
                    decision = processed.get(ticket_id, {}).get(str(sentence_index))
                    if decision is None:
                        decision = processed.get(ticket_id, {}).get(sentence_index)
                    if decision is None:
                        pending_count += 1
                        decision_display = "PENDING"
                    else:
                        decision_display = f"DONE:{decision}"
                    parts.append(
                        f"  [{ticket_id}:{sentence_index}] ({speaker}) {sentence.get('text', '')} -> {decision_display}"
                    )
        parts.append(f"Pending sentence decisions: {pending_count}")
        if pending_count == 0 and tickets:
            parts.append("All sentences processed. Call compile_response now.")
        elif pending_count > 0:
            parts.append(
                "DO NOT call compile_response yet. You must finish every PENDING sentence first."
            )

    customer_record = obs.get("customer_record", [])
    pending_fields = []

    for item in customer_record:
        if isinstance(item, dict):
            field_id = item.get("field_id", "unknown")
        else:
            field_id = getattr(item, "field_id", "unknown")

        if field_id not in classified:
            pending_fields.append(item)

    if task_id == "task_easy" and pending_fields:
        parts.append(
            f"\n=== PENDING FIELDS - CLASSIFY THESE ({len(pending_fields)} remaining) ==="
        )
        for item in pending_fields:
            if isinstance(item, dict):
                field_id = item.get("field_id", "unknown")
                field_name = item.get("field_name", field_id)
                field_value = item.get("field_value", "N/A")
                field_description = item.get("field_description", "")
            else:
                field_id = getattr(item, "field_id", "unknown")
                field_name = getattr(item, "field_name", field_id)
                field_value = getattr(item, "field_value", "N/A")
                field_description = getattr(item, "field_description", "")

            value_str = (
                json.dumps(field_value, default=str)
                if not isinstance(field_value, str)
                else field_value
            )
            parts.append(
                f"  [{field_id}]  {field_name}\n"
                f"    Value: {value_str}\n"
                f"    {field_description}"
            )
    elif task_id == "task_easy" and customer_record:
        parts.append("\nAll fields classified. Call compile_response now.")
    elif task_id == "task_easy":
        parts.append("\n=== NO FIELDS REVEALED YET ===")
        parts.append(
            "Query billing and crm to reveal the customer record before classifying anything."
        )

    if obs.get("error"):
        parts.append(f"\nLast action error: {obs['error']}")
        if "All ticket sentences must be processed" in str(obs["error"]):
            parts.append(
                "Your previous compile failed because sentences are still pending. "
                "Resume redact_span decisions before trying compile_response again."
            )

    return "\n".join(parts)


def run_episode(env_url: str, task_id: str) -> float:
    """Run one episode against the environment and return the final score."""
    import requests

    print(f"\n{'=' * 60}")
    print(f"Starting episode: {task_id}")
    print(f"{'=' * 60}")

    reset_payload = {"task_id": task_id}
    if EPISODE_SEED is not None:
        reset_payload["seed"] = EPISODE_SEED

    trace("RESET REQUEST", {"url": f"{env_url}/reset", "json": reset_payload})

    reset_resp = requests.post(
        f"{env_url}/reset",
        json=reset_payload,
        timeout=30,
    )
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()
    trace("RESET RESPONSE", reset_data)

    observation = reset_data.get("observation", reset_data)
    episode_id = observation.get("episode_id", "")
    if not episode_id:
        episode_id = observation.get("metadata", {}).get("episode_id", "")
    done = reset_data.get("done", observation.get("done", False))

    record = observation.get("customer_record", [])
    print(f"Episode ID: {episode_id}")
    if EPISODE_SEED is not None:
        print(f"Seed: {EPISODE_SEED}")
    else:
        print("Seed: random")
    print(f"Fields in record: {len(record)}")
    trace("INITIAL OBSERVATION", observation)

    history = []
    total_reward = 0.0
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        obs_text = format_observation(observation)
        user_prompt = f"Step {step}/{MAX_STEPS}.\n\n{obs_text}"
        if history:
            user_prompt += "\n\nAction history (last 5):\n" + "\n".join(history[-5:])
        trace("FORMATTED OBSERVATION", obs_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        trace("MODEL MESSAGES", messages)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
            trace("RAW MODEL RESPONSE", response_text)
        except Exception as exc:
            print(f"  [ERROR] LLM request failed: {exc}. Using fallback.")
            response_text = FALLBACK_ACTION
            trace("RAW MODEL RESPONSE", {"error": str(exc), "fallback": response_text})

        action_dict = parse_model_action(response_text)
        action_dict["metadata"] = {"episode_id": episode_id}
        trace("PARSED ACTION", action_dict)
        action_str = f"{action_dict['action_type']}"
        if action_dict.get("silo_name"):
            action_str += f" {action_dict['silo_name']}"
        if action_dict.get("field_id"):
            action_str += f" {action_dict['field_id']} {action_dict.get('decision', '')}"
        if action_dict.get("verification_method"):
            action_str += f" {action_dict['verification_method']}"
        if action_dict.get("ticket_id") is not None:
            action_str += (
                f" {action_dict['ticket_id']} "
                f"{action_dict.get('sentence_index', '')} "
                f"{action_dict.get('decision', '')}"
            )

        print(f"  Step {step}: {action_str}")
        trace("STEP REQUEST", {"url": f"{env_url}/step", "json": {"action": action_dict}})

        step_resp = requests.post(
            f"{env_url}/step",
            json={"action": action_dict},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()
        trace("STEP RESPONSE", step_data)

        observation = step_data.get("observation", step_data)
        meta = merged_metadata(step_data, observation)
        reward = step_data.get("reward", observation.get("reward", 0.0)) or 0.0
        done = step_data.get("done", observation.get("done", False))
        total_reward += reward

        error_flag = f" ERROR: {observation.get('error')}" if observation.get("error") else ""
        history.append(f"Step {step}: {action_str} -> reward {reward:+.4f}{error_flag}")
        print(f"    Reward: {reward:+.4f} | Done: {done}{error_flag}")
        trace("UPDATED OBSERVATION", observation)

        if done:
            terminal_score = meta.get("terminal_score")
            if terminal_score is None:
                terminal_score = 0.0 if observation.get("error") else reward
            final_score = terminal_score
            if TRACE_ENABLED:
                trace(
                    "TERMINAL METRICS",
                    {
                        key: meta.get(key)
                        for key in [
                            "identity_score",
                            "phase1_reward_sum",
                            "kept_precision",
                            "kept_recall",
                            "redaction_f1",
                            "leakage_rate",
                            "redaction_score",
                            "leaked_pii_count",
                            "total_pii_sentences",
                            "completion_coverage",
                            "termination_reason",
                            "terminal_score",
                        ]
                        if key in meta
                    },
                )
            print("\n  Episode complete!")
            print(f"  Terminal score: {final_score:.4f}")
            print(f"  Cumulative step reward: {total_reward:.4f}")
            print(f"  Steps used: {step}")
            break
    else:
        print(f"  Reached max steps ({MAX_STEPS}).")
        final_score = observation.get("metadata", {}).get("terminal_score", 0.0)

    return final_score


def main() -> None:
    """Run baseline inference across the currently implemented task set."""
    env_url = os.environ.get("DSAR_ENV_URL", "http://localhost:8000")

    print(f"DSAR Environment URL: {env_url}")
    print(f"LLM API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"{'=' * 60}")

    import requests

    try:
        health = requests.get(f"{env_url}/health", timeout=10)
        health.raise_for_status()
        print(f"Environment health: {health.json()}")
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {env_url}: {exc}")
        print("Make sure the DSAR environment server is running.")
        sys.exit(1)

    tasks = TASK_IDS
    scores = {}

    start_time = time.time()

    if MULTI_SEED_VALUES:
        original_seed = os.environ.get("EPISODE_SEED")
        for task_id in tasks:
            task_scores = []
            for seed_value in MULTI_SEED_VALUES:
                os.environ["EPISODE_SEED"] = seed_value
                global EPISODE_SEED
                EPISODE_SEED = _parse_optional_int(seed_value)
                try:
                    score = run_episode(env_url, task_id)
                    task_scores.append(score)
                except Exception as exc:
                    print(f"\n  ERROR running {task_id} seed {seed_value}: {exc}")
                    task_scores.append(0.0)
            if original_seed is None:
                os.environ.pop("EPISODE_SEED", None)
            else:
                os.environ["EPISODE_SEED"] = original_seed
            EPISODE_SEED = _parse_optional_int(original_seed)
            scores[task_id] = task_scores

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("MULTI-SEED SCORES")
        print(f"{'=' * 60}")
        for task_id, task_scores in scores.items():
            mean_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
            print(
                f"  {task_id}: mean={mean_score:.4f} "
                f"min={min(task_scores):.4f} max={max(task_scores):.4f} "
                f"seeds={','.join(MULTI_SEED_VALUES)}"
            )
        print(f"\n  Total runtime: {elapsed:.1f}s")
        print(f"{'=' * 60}")
        return

    for task_id in tasks:
        try:
            score = run_episode(env_url, task_id)
            scores[task_id] = score
        except Exception as exc:
            print(f"\n  ERROR running {task_id}: {exc}")
            scores[task_id] = 0.0

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("BASELINE SCORES")
    print(f"{'=' * 60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
