#!/usr/bin/env python3
"""
DSAR Environment — Baseline Inference Script.

This script demonstrates the DSAR environment by running an LLM agent
against all three tasks and reporting scores.

Environment variables required:
    API_BASE_URL  — The API endpoint for the LLM (e.g., https://api.groq.com/openai/v1)
    MODEL_NAME    — The model to use (e.g., llama-3.3-70b-versatile)
    HF_TOKEN      — HuggingFace / API key (also used as OPENAI_API_KEY)

Usage:
    API_BASE_URL=https://api.groq.com/openai/v1 \
    MODEL_NAME=llama-3.3-70b-versatile \
    HF_TOKEN=your_key \
    python inference.py
"""

import json
import os
import re
import sys
import time

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TEMPERATURE = 0.0   # must be 0.0 for reproducible hackathon scores
MAX_TOKENS = 512
MAX_STEPS = 30

# OpenAI-compatible client pointing at the configured LLM provider
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
)

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a GDPR compliance agent processing a Data Subject Access Request (DSAR).

A customer has requested all personal data held about them under Article 15 of the UK GDPR. You have access to the company's database records which contain a mix of:
- PERSONAL DATA belonging to the customer (must be disclosed)
- INTERNAL BUSINESS DATA belonging to the company (must NOT be disclosed)

Your job is to:
1. First query BOTH data silos: query_silo billing, then query_silo crm
2. Classify every PENDING field — one per turn
3. Once all fields are classified, call compile_response

Available actions (respond with EXACTLY ONE per turn):
  query_silo billing          — Query the billing database
  query_silo crm              — Query the CRM database
  classify_field <field_id> disclose   — Mark a field as personal data to include
  classify_field <field_id> withhold   — Mark a field as internal data to exclude
  compile_response            — Finalize and submit your classification

CRITICAL RULES:
- NEVER classify a field that appears in the ALREADY CLASSIFIED list
- NEVER query the same silo twice
- Only call compile_response when ALL PENDING fields have been classified
- Respond with ONLY the action — no explanation text

Use each field's name, value, and description to decide whether it belongs
to the customer (disclose) or is the company's internal business data (withhold).
Personal data belonging to the customer must be disclosed.
Internal operational, analytical, or infrastructure data must be withheld."""

# ─── Fallback action ──────────────────────────────────────────────────────────
FALLBACK_ACTION = "compile_response"


def parse_model_action(response_text: str) -> dict:
    """Parse the LLM's text response into a structured action dict.

    Returns a dict with action_type and optional parameters.
    Uses deterministic string parsing for reproducible baseline scores.
    """
    text = response_text.strip()

    # Pattern: query_silo <name>
    match = re.search(r"query_silo\s+(\w+)", text, re.IGNORECASE)
    if match:
        return {
            "action_type": "query_silo",
            "silo_name": match.group(1).lower(),
        }

    # Pattern: classify_field <field_id> <decision>
    match = re.search(
        r"classify_field\s+([\w_]+)\s+(disclose|withhold)", text, re.IGNORECASE
    )
    if match:
        return {
            "action_type": "classify_field",
            "field_id": match.group(1).lower(),
            "decision": match.group(2).lower(),
        }

    # Pattern: compile_response
    if re.search(r"compile_response", text, re.IGNORECASE):
        return {"action_type": "compile_response"}

    # Fallback — compile what we have so far
    print(f"  [WARN] Could not parse action from: {text[:120]}...")
    return {"action_type": "compile_response"}


def format_observation(obs: dict) -> str:
    """Format an observation dict into a readable string for the LLM.

    Splits fields into PENDING (not yet classified) and ALREADY DONE so the
    LLM can't accidentally re-classify a field it already handled.
    """
    parts = []

    # ── Step status summary at the top ───────────────────────────────────
    steps_remaining = obs.get("steps_remaining", "?")
    classified = set(obs.get("classified_fields", []))
    silos_queried = obs.get("silo_results", [])

    parts.append(f"Steps remaining: {steps_remaining}")
    if silos_queried:
        parts.append(f"Silos already queried: {', '.join(silos_queried)}")
    else:
        parts.append("Silos already queried: NONE — you must query billing and crm first")

    if classified:
        parts.append(f"\n⚠ ALREADY CLASSIFIED ({len(classified)} fields — DO NOT re-classify these):\n  {', '.join(sorted(classified))}")

    # ── DSAR request ─────────────────────────────────────────────────────
    if obs.get("dsar_request"):
        parts.append(f"\n=== DSAR REQUEST ===\n{obs['dsar_request']}")

    # ── Split record into PENDING vs DONE ────────────────────────────────
    customer_record = obs.get("customer_record", [])
    pending_fields = []
    done_fields = []

    for item in customer_record:
        if isinstance(item, dict):
            fid = item.get("field_id", "unknown")
        else:
            fid = getattr(item, "field_id", "unknown")
        if fid in classified:
            done_fields.append(fid)
        else:
            pending_fields.append(item)

    # Show PENDING fields with full metadata (these need action)
    if pending_fields:
        parts.append(f"\n=== PENDING FIELDS — CLASSIFY THESE ({len(pending_fields)} remaining) ===")
        for item in pending_fields:
            if isinstance(item, dict):
                fid = item.get("field_id", "unknown")
                fname = item.get("field_name", fid)
                fval = item.get("field_value", "N/A")
                source = item.get("source_silo", "unknown")
                dtype = item.get("datatype", "unknown")
                desc = item.get("field_description", "")
            else:
                fid = getattr(item, "field_id", "unknown")
                fname = getattr(item, "field_name", fid)
                fval = getattr(item, "field_value", "N/A")
                source = getattr(item, "source_silo", "unknown")
                dtype = getattr(item, "datatype", "unknown")
                desc = getattr(item, "field_description", "")

            val_str = json.dumps(fval, default=str) if not isinstance(fval, str) else fval
            parts.append(
                f"  [{fid}]  {fname}\n"
                f"    Value: {val_str}\n"
                f"    {desc}"
            )
    else:
        parts.append("\n✓ All fields classified. Call compile_response now.")

    if obs.get("error"):
        parts.append(f"\n⚠ Last action error: {obs['error']}")

    return "\n".join(parts)


def run_episode(env_url: str, task_id: str) -> float:
    """Run one episode against the environment and return the final score.

    Args:
        env_url: Base URL of the running DSAR environment.
        task_id: Task to run (task_easy, task_medium, task_hard).

    Returns:
        Final episode score (0.0 to 1.0).
    """
    import requests

    print(f"\n{'='*60}")
    print(f"Starting episode: {task_id}")
    print(f"{'='*60}")

    # ── Reset the environment ─────────────────────────────────────────────
    reset_resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id, "seed": 42},   # fixed seed → reproducible field values
        timeout=30,
    )
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()

    observation = reset_data.get("observation", reset_data)
    episode_id = observation.get("episode_id", "")
    if not episode_id:
        episode_id = observation.get("metadata", {}).get("episode_id", "")
    done = reset_data.get("done", observation.get("done", False))

    record = observation.get("customer_record", [])
    print(f"Episode ID: {episode_id}")
    print(f"Fields in record: {len(record)}")

    history = []
    total_reward = 0.0
    final_score = 0.0

    # ── Episode loop ──────────────────────────────────────────────────────
    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        # Format observation for the LLM
        obs_text = format_observation(observation)

        # Build message context with recent history
        user_prompt = f"Step {step}/{MAX_STEPS}.\n\n{obs_text}"
        if history:
            user_prompt += f"\n\nAction history (last 5):\n" + "\n".join(history[-5:])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # ── Call LLM ──────────────────────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [ERROR] LLM request failed: {exc}. Using fallback.")
            response_text = FALLBACK_ACTION

        # ── Parse action ──────────────────────────────────────────────────
        action_dict = parse_model_action(response_text)
        action_dict["metadata"] = {"episode_id": episode_id}
        action_str = f"{action_dict['action_type']}"
        if action_dict.get("silo_name"):
            action_str += f" {action_dict['silo_name']}"
        if action_dict.get("field_id"):
            action_str += f" {action_dict['field_id']} {action_dict.get('decision', '')}"

        print(f"  Step {step}: {action_str}")

        # ── Send action to environment ────────────────────────────────────
        step_resp = requests.post(
            f"{env_url}/step",
            json={"action": action_dict},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()

        observation = step_data.get("observation", step_data)
        reward = step_data.get("reward", observation.get("reward", 0.0)) or 0.0
        done = step_data.get("done", observation.get("done", False))
        total_reward += reward

        error_flag = f" ERROR: {observation.get('error')}" if observation.get("error") else ""
        history.append(f"Step {step}: {action_str} -> reward {reward:+.4f}{error_flag}")
        print(f"    Reward: {reward:+.4f} | Done: {done}{error_flag}")

        if done:
            terminal_score = observation.get("metadata", {}).get("terminal_score", reward)
            final_score = terminal_score
            print(f"\n  Episode complete!")
            print(f"  Terminal score: {final_score:.4f}")
            print(f"  Cumulative step reward: {total_reward:.4f}")
            print(f"  Steps used: {step}")
            break
    else:
        print(f"  Reached max steps ({MAX_STEPS}).")
        final_score = observation.get("metadata", {}).get("terminal_score", 0.0)

    return final_score


def main():
    """Run baseline inference across all three tasks."""
    env_url = os.environ.get("DSAR_ENV_URL", "http://localhost:8000")

    print(f"DSAR Environment URL: {env_url}")
    print(f"LLM API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}")

    # ── Test environment health ───────────────────────────────────────────
    import requests
    try:
        health = requests.get(f"{env_url}/health", timeout=10)
        health.raise_for_status()
        print(f"Environment health: {health.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {env_url}: {e}")
        print("Make sure the DSAR environment server is running.")
        sys.exit(1)

    # ── Run only easy task ────────────────────────────────────────────────
    tasks = ["task_easy"]
    scores = {}

    start_time = time.time()

    for task_id in tasks:
        try:
            score = run_episode(env_url, task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"\n  ERROR running {task_id}: {e}")
            scores[task_id] = 0.0

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
