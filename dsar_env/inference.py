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
    DSAR_TASK_SEEDS - Optional per-task seed map, e.g. task_easy:7,task_medium:3,task_hard:15

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


def _parse_task_seed_map(raw_value: str | None) -> dict[str, int]:
    """Parse DSAR_TASK_SEEDS like 'task_easy:7,task_medium:3'."""
    if raw_value in (None, ""):
        return {}

    parsed: dict[str, int] = {}
    for entry in raw_value.split(","):
        item = entry.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Invalid DSAR_TASK_SEEDS entry "
                f"'{item}'. Expected format task_id:seed."
            )
        task_id, seed_value = item.split(":", 1)
        task_id = task_id.strip()
        seed_value = seed_value.strip()
        if not task_id or not seed_value:
            raise ValueError(
                "Invalid DSAR_TASK_SEEDS entry "
                f"'{item}'. Expected format task_id:seed."
            )
        parsed[task_id] = int(seed_value)
    return parsed


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:fastest")
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = 0.0  # keep deterministic when a fixed seed is provided
MAX_TOKENS = 512
MAX_STEPS = 30
EPISODE_SEED = _parse_optional_int(os.environ.get("EPISODE_SEED"))
TASK_SEED_MAP = _parse_task_seed_map(os.environ.get("DSAR_TASK_SEEDS"))
TASK_IDS = [task.strip() for task in os.environ.get("DSAR_TASKS", "task_easy,task_medium,task_hard").split(",") if task.strip()]
TRACE_ENABLED = os.environ.get("DSAR_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
INFERENCE_MODE = os.environ.get("DSAR_INFERENCE_MODE", "raw").strip().lower()
CASE3_HEURISTIC_REQUESTED = (
    os.environ.get("DSAR_DEBUG_CASE3_HEURISTIC", os.environ.get("DSAR_USE_CASE3_HEURISTIC", ""))
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
CASE3_HEURISTIC_ENABLED = INFERENCE_MODE == "debug" and CASE3_HEURISTIC_REQUESTED
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
- process_message <msg_id> <disclose|partial_redact|exclude|escalate>
- redact_sentence <msg_id> <sentence_index> <keep|redact>
- escalate_with_reason <msg_id> <reason_code> :: <reason>
- compile_response

Rules:
- For task_easy, query both silos before classifying fields.
- For task_medium identity, use a proportionate verification method.
- For task_medium redaction, process each sentence once and only compile when all are done.
- For task_hard, never use query_silo, classify_field, verify_identity, or redact_span.
- For task_hard, use the observation state to determine which action type is currently valid.
- Never repeat the same silo query or sentence decision.
- Respond with only the action text."""

FALLBACK_ACTION = "compile_response"
MODEL_ACTION_MAX_RETRIES = 3

CASE3_HEALTH_TERMS = (
    "anxiety",
    "stress-related illness",
    "burnout",
    "depression",
    "mental health",
    "medical",
    "health",
)
CASE3_PERFORMANCE_TERMS = (
    "performance review",
    "hr flag",
    "output this quarter",
    "missed the",
    "targets again",
    "below target",
)
CASE3_TECHNICAL_TERMS = (
    "fix",
    "build issue",
    "deployment",
    "deploy",
    "pipeline",
    "merged",
    "integration bug",
    "smoke tests",
)
CASE3_PR_TERMS = (
    "pr",
    "pull request",
    "review my",
    "look at my",
    "looking over my",
)
CASE3_COMPENSATION_TERMS = (
    "salary review",
    "pay review",
    "compensation review",
    "annual review",
)


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
    terminal_details = observation.get("terminal_details", {}) if isinstance(observation, dict) else {}
    if isinstance(terminal_details, dict):
        merged.update(terminal_details)
    return merged


def _available_actions(obs: dict) -> list[str]:
    actions = obs.get("available_actions", []) or []
    return [str(action) for action in actions]


def _action_type_allowed(action_dict: dict, available_actions: list[str]) -> bool:
    return action_dict.get("action_type") in set(available_actions)


def _action_validation_message(available_actions: list[str], invalid_action_type: str) -> str:
    allowed = ", ".join(available_actions)
    return (
        f"Your previous action type '{invalid_action_type}' is invalid for the current state. "
        f"Respond again with exactly one action whose action type is one of: {allowed}."
    )


def _case3_pending_escalation_ids(obs: dict) -> list[str]:
    processed_messages = _case3_processed_messages(obs)
    escalation_log = obs.get("escalation_log", {}) or {}
    escalation_reason_codes = obs.get("escalation_reason_codes", {}) or {}
    return sorted(
        msg_id
        for msg_id, decision in processed_messages.items()
        if decision.get("action") == "escalate"
        and (msg_id not in escalation_log or msg_id not in escalation_reason_codes)
    )


def _action_params_allowed(action_dict: dict, obs: dict) -> bool:
    if obs.get("task_id") != "task_hard":
        return True

    action_type = action_dict.get("action_type")
    if action_type == "process_message":
        return action_dict.get("msg_id") in set(obs.get("messages_pending", []) or [])
    if action_type == "redact_sentence":
        msg_id = action_dict.get("msg_id")
        sentence_index = action_dict.get("sentence_index")
        pending = obs.get("sentences_pending", {}) or {}
        return msg_id in pending and sentence_index in set(pending.get(msg_id, []))
    if action_type == "escalate_with_reason":
        return action_dict.get("msg_id") in set(_case3_pending_escalation_ids(obs))
    if action_type == "compile_response":
        return bool(obs.get("compile_ready"))
    return True


def _action_parameter_validation_message(obs: dict, action_dict: dict) -> str:
    action_type = action_dict.get("action_type", "unknown")
    if obs.get("task_id") != "task_hard":
        return (
            f"Your previous action parameters for '{action_type}' were invalid for the current state. "
            "Respond again with exactly one valid action."
        )

    if action_type == "process_message":
        pending_messages = ", ".join(obs.get("messages_pending", []) or [])
        return (
            f"Your previous process_message target was invalid for the current state. "
            f"Respond again using one msg_id from messages_pending: {pending_messages}."
        )
    if action_type == "redact_sentence":
        pending = obs.get("sentences_pending", {}) or {}
        return (
            f"Your previous redact_sentence target was invalid for the current state. "
            f"Respond again using one exact msg_id/sentence_index pair from sentences_pending: "
            f"{json.dumps(pending, default=str)}."
        )
    if action_type == "escalate_with_reason":
        pending_escalations = ", ".join(_case3_pending_escalation_ids(obs))
        return (
            f"Your previous escalate_with_reason target was invalid for the current state. "
            f"Respond again using one msg_id from pending escalations: {pending_escalations}."
        )
    if action_type == "compile_response":
        return (
            "compile_response is not valid yet. Respond again with exactly one valid non-terminal action "
            "for the current state."
        )
    return (
        f"Your previous action parameters for '{action_type}' were invalid for the current state. "
        "Respond again with exactly one valid action."
    )


def _case3_processed_messages(obs: dict) -> dict:
    return obs.get("processed_messages", {}) or {}


def _case3_users(obs: dict) -> dict:
    return obs.get("users_json", {}) or {}


def _extract_terminal_metrics(meta: dict, observation: dict) -> dict:
    preferred_keys = [
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
        "c1_message_accuracy",
        "c2_sentence_redaction",
        "c3_escalation_quality",
        "n_pii_breaches",
        "privacy_penalty",
        "task3_score",
        "failure_summary",
        "message_diagnostics",
        "incorrect_message_ids",
        "constraint_violated",
        "schema_gate",
    ]
    terminal_metrics = {key: meta.get(key) for key in preferred_keys if key in meta}
    if terminal_metrics:
        return terminal_metrics

    terminal_details = observation.get("terminal_details", {}) if isinstance(observation, dict) else {}
    if isinstance(terminal_details, dict) and terminal_details:
        return {key: terminal_details.get(key) for key in preferred_keys if key in terminal_details}

    metadata = observation.get("metadata", {}) if isinstance(observation, dict) else {}
    if isinstance(metadata, dict) and metadata:
        extracted = {key: metadata.get(key) for key in preferred_keys if key in metadata}
        if extracted:
            return extracted

    return {}


def _case3_requester_user_id(obs: dict) -> str | None:
    users = _case3_users(obs)
    requester_name = ""
    dsar_request = obs.get("dsar_request", "")
    match = re.search(r"I act for ([^.]+)\.", dsar_request)
    if match:
        requester_name = match.group(1).strip()
    for user_id, payload in users.items():
        if payload.get("real_name") == requester_name:
            return user_id
    return None


def _case3_slack_messages(obs: dict) -> list[dict]:
    normalized = []
    for message in obs.get("slack_export", []) or []:
        if isinstance(message, dict):
            normalized.append(message)
        else:
            normalized.append(
                {
                    "msg_id": getattr(message, "msg_id", ""),
                    "user": getattr(message, "user", ""),
                    "text": getattr(message, "text", ""),
                    "ts": getattr(message, "ts", ""),
                    "thread_ts": getattr(message, "thread_ts", None),
                    "subtype": getattr(message, "subtype", None),
                    "sentences": [
                        {
                            "sentence_idx": getattr(sentence, "sentence_idx", -1),
                            "text": getattr(sentence, "text", ""),
                        }
                        for sentence in getattr(message, "sentences", [])
                    ],
                }
            )
    return normalized


def _case3_parent_map(obs: dict) -> dict[str, dict]:
    messages = _case3_slack_messages(obs)
    by_ts = {message.get("ts"): message for message in messages}
    return by_ts


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _is_case3_bot_message(message: dict, users_json: dict) -> bool:
    if message.get("subtype") == "bot_message":
        return True
    return bool(users_json.get(message.get("user", ""), {}).get("is_bot"))


def _is_case3_mixed_message(message: dict) -> bool:
    sentences = message.get("sentences", [])
    if len(sentences) != 2:
        return False
    first = sentences[0].get("text", "")
    second = sentences[1].get("text", "")
    return _contains_any(first, CASE3_PR_TERMS) and _contains_any(second, CASE3_COMPENSATION_TERMS)


def _is_case3_health_trap(message: dict, requester_display_name: str) -> bool:
    sentences = message.get("sentences", [])
    if len(sentences) != 2:
        return False
    first = sentences[0].get("text", "")
    second = sentences[1].get("text", "")
    mentions_requester = requester_display_name and requester_display_name.lower() in first.lower()
    return mentions_requester and _contains_any(first, CASE3_PERFORMANCE_TERMS) and _contains_any(second, CASE3_HEALTH_TERMS)


def _is_case3_requester_technical(message: dict, requester_user_id: str | None) -> bool:
    return message.get("user") == requester_user_id and _contains_any(message.get("text", ""), CASE3_TECHNICAL_TERMS)


def _is_case3_thread_reply_disclose(message: dict, obs: dict) -> bool:
    thread_ts = message.get("thread_ts")
    if not thread_ts:
        return False
    parent = _case3_parent_map(obs).get(thread_ts)
    if not parent:
        return False
    requester_user_id = _case3_requester_user_id(obs)
    return _is_case3_requester_technical(parent, requester_user_id)


def _is_case3_clean_manager_performance(message: dict, requester_display_name: str) -> bool:
    if message.get("thread_ts"):
        return False
    if _contains_any(message.get("text", ""), CASE3_HEALTH_TERMS):
        return False
    lowered = message.get("text", "").lower()
    return requester_display_name.lower() in lowered and (
        "performance review" in lowered or "hr flag" in lowered or "output this quarter" in lowered
    )


def choose_task_hard_action(obs: dict) -> dict | None:
    if obs.get("task_id") != "task_hard":
        return None

    processed_messages = _case3_processed_messages(obs)
    users_json = _case3_users(obs)
    messages = _case3_slack_messages(obs)
    requester_user_id = _case3_requester_user_id(obs)
    requester_payload = users_json.get(requester_user_id or "", {})
    requester_display_name = requester_payload.get("display_name", "")
    messages_pending = obs.get("messages_pending", []) or []
    sentences_pending = obs.get("sentences_pending", {}) or {}
    escalation_log = obs.get("escalation_log", {}) or {}
    escalation_reason_codes = obs.get("escalation_reason_codes", {}) or {}

    pending_message_map = {message["msg_id"]: message for message in messages if message.get("msg_id") in messages_pending}
    if messages_pending:
        priority_order = []
        for msg_id in messages_pending:
            message = pending_message_map[msg_id]
            if _is_case3_bot_message(message, users_json):
                priority_order.append((0, msg_id, "exclude"))
            elif _is_case3_health_trap(message, requester_display_name):
                priority_order.append((1, msg_id, "escalate"))
            elif _is_case3_mixed_message(message):
                priority_order.append((2, msg_id, "partial_redact"))
            elif _is_case3_thread_reply_disclose(message, obs):
                priority_order.append((3, msg_id, "disclose"))
            elif _is_case3_requester_technical(message, requester_user_id):
                priority_order.append((4, msg_id, "disclose"))
            elif _is_case3_clean_manager_performance(message, requester_display_name):
                priority_order.append((5, msg_id, "disclose"))
            else:
                priority_order.append((6, msg_id, "disclose"))
        priority_order.sort()
        _, msg_id, action_label = priority_order[0]
        return {
            "action_type": "process_message",
            "msg_id": msg_id,
            "action_label": action_label,
        }

    if sentences_pending:
        msg_id = sorted(sentences_pending.keys())[0]
        sentence_idx = sorted(sentences_pending[msg_id])[0]
        decision = "keep" if sentence_idx == 0 else "redact"
        return {
            "action_type": "redact_sentence",
            "msg_id": msg_id,
            "sentence_index": sentence_idx,
            "decision": decision,
        }

    unresolved_escalations = sorted(
        msg_id
        for msg_id, decision in processed_messages.items()
        if decision.get("action") == "escalate"
        and (msg_id not in escalation_log or msg_id not in escalation_reason_codes)
    )
    if unresolved_escalations:
        msg_id = unresolved_escalations[0]
        return {
            "action_type": "escalate_with_reason",
            "msg_id": msg_id,
            "reason_code": "special_category_health_data",
            "reason": "Escalate because this contains stress-related health data and article 9 special category information.",
        }

    if obs.get("compile_ready"):
        return {"action_type": "compile_response"}

    return None


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

    match = re.search(
        r"process_message\s+([A-Za-z0-9_-]+)\s+(disclose|partial_redact|exclude|escalate)",
        text,
        re.IGNORECASE,
    )
    if match:
        return {
            "action_type": "process_message",
            "msg_id": match.group(1),
            "action_label": match.group(2).lower(),
        }

    match = re.search(
        r"redact_sentence\s+([A-Za-z0-9_-]+)\s+(\d+)\s+(keep|redact)",
        text,
        re.IGNORECASE,
    )
    if match:
        return {
            "action_type": "redact_sentence",
            "msg_id": match.group(1),
            "sentence_index": int(match.group(2)),
            "decision": match.group(3).lower(),
        }

    match = re.search(
        r"escalate_with_reason\s+([A-Za-z0-9_-]+)\s+([A-Za-z0-9_]+)\s+::\s+(.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return {
            "action_type": "escalate_with_reason",
            "msg_id": match.group(1),
            "reason_code": match.group(2),
            "reason": match.group(3).strip(),
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
    if task_id in {"task_easy", "task_medium"}:
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

    if task_id == "task_hard":
        parts.append("\n=== CASE 3 SLACK TRIAGE ===")
        parts.append(
            "These six Slack messages are the candidate responsive set already surfaced from the broader export."
        )
        users_json = obs.get("users_json", {})
        if users_json:
            parts.append(f"Users JSON: {json.dumps(users_json, default=str)}")
        processed_messages = obs.get("processed_messages", {})
        escalation_log = obs.get("escalation_log", {})
        messages_pending = obs.get("messages_pending", [])
        sentences_pending = obs.get("sentences_pending", {})
        compile_ready = obs.get("compile_ready", False)
        parts.append(f"Messages pending: {', '.join(messages_pending) if messages_pending else 'NONE'}")
        parts.append(f"Compile ready: {str(compile_ready).lower()}")
        if sentences_pending:
            parts.append(f"Sentence decisions pending: {json.dumps(sentences_pending, default=str)}")
        unresolved_escalations = [
            msg_id
            for msg_id, decision in processed_messages.items()
            if decision.get("action") == "escalate"
            and (
                msg_id not in escalation_log
                or msg_id not in obs.get("escalation_reason_codes", {})
            )
        ]
        if unresolved_escalations:
            parts.append(
                "Escalation reasons pending: " + ", ".join(sorted(unresolved_escalations))
            )
        parts.append(
            "Reason codes: "
            "special_category_health_data, mixed_sensitive_third_party_data, requires_human_balancing"
        )
        for message in obs.get("slack_export", []):
            if isinstance(message, dict):
                msg_id = message.get("msg_id", "unknown")
                user = message.get("user", "unknown")
                text = message.get("text", "")
                ts = message.get("ts", "")
                thread_ts = message.get("thread_ts")
                subtype = message.get("subtype")
                sentences = message.get("sentences", [])
            else:
                msg_id = getattr(message, "msg_id", "unknown")
                user = getattr(message, "user", "unknown")
                text = getattr(message, "text", "")
                ts = getattr(message, "ts", "")
                thread_ts = getattr(message, "thread_ts", None)
                subtype = getattr(message, "subtype", None)
                sentences = getattr(message, "sentences", [])
            decision = processed_messages.get(msg_id)
            decision_text = "PENDING" if decision is None else json.dumps(decision, default=str)
            reason_text = escalation_log.get(msg_id)
            reason_code = obs.get("escalation_reason_codes", {}).get(msg_id)
            message_action = decision.get("action") if isinstance(decision, dict) else None
            header = f"Message {msg_id} | user={user} | ts={ts}"
            if thread_ts:
                header += f" | thread_ts={thread_ts}"
            if subtype:
                header += f" | subtype={subtype}"
            parts.append(header)
            parts.append(f"  Text: {text}")
            for sentence in sentences:
                if isinstance(sentence, dict):
                    idx = sentence.get("sentence_idx", -1)
                    sent_text = sentence.get("text", "")
                else:
                    idx = getattr(sentence, "sentence_idx", -1)
                    sent_text = getattr(sentence, "text", "")
                sentence_decision_map = processed_messages.get(msg_id, {}).get("sentence_decisions", {})
                sentence_status = sentence_decision_map.get(idx)
                if sentence_status is None:
                    sentence_status = sentence_decision_map.get(str(idx))
                if sentence_status is not None:
                    sentence_display = sentence_status
                elif message_action == "partial_redact":
                    sentence_display = "PENDING"
                elif message_action in {"disclose", "exclude", "escalate"}:
                    sentence_display = "N/A"
                else:
                    sentence_display = "AWAITING_MESSAGE_DECISION"
                parts.append(f"  [{idx}] {sent_text} -> {sentence_display}")
            parts.append(f"  Message decision: {decision_text}")
            if reason_code:
                parts.append(f"  Escalation reason_code: {reason_code}")
            if reason_text:
                parts.append(f"  Escalation reason: {reason_text}")

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


def run_episode(env_url: str, task_id: str, episode_seed: int | None = None) -> dict:
    """Run one episode against the environment and return score plus debug details."""
    import requests

    print(f"\n{'=' * 60}")
    print(f"Starting episode: {task_id}")
    print(f"{'=' * 60}")

    reset_payload = {"task_id": task_id}
    if episode_seed is not None:
        reset_payload["seed"] = episode_seed

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
    episode_max_steps = max(MAX_STEPS, int(observation.get("steps_remaining", MAX_STEPS)))

    record = observation.get("customer_record", [])
    print(f"Episode ID: {episode_id}")
    if episode_seed is not None:
        print(f"Seed: {episode_seed}")
    else:
        print("Seed: random")
    print(f"Fields in record: {len(record)}")
    trace("INITIAL OBSERVATION", observation)

    history = []
    total_reward = 0.0
    final_score = 0.0
    terminal_metrics: dict = {}

    for step in range(1, episode_max_steps + 1):
        if done:
            break

        obs_text = format_observation(observation)
        user_prompt = f"Step {step}/{episode_max_steps}.\n\n{obs_text}"
        if history:
            user_prompt += "\n\nAction history (last 5):\n" + "\n".join(history[-5:])
        trace("FORMATTED OBSERVATION", obs_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        trace("MODEL MESSAGES", messages)

        heuristic_action = choose_task_hard_action(observation) if CASE3_HEURISTIC_ENABLED else None
        if heuristic_action is not None:
            action_dict = heuristic_action
            trace("RAW MODEL RESPONSE", {"action_source": "heuristic", "heuristic_action": action_dict})
        else:
            available_actions = _available_actions(observation)
            retry_messages = list(messages)
            action_dict = {"action_type": FALLBACK_ACTION}
            for attempt in range(1, MODEL_ACTION_MAX_RETRIES + 1):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=retry_messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    trace(
                        "RAW MODEL RESPONSE",
                        {
                            "attempt": attempt,
                            "response": response_text,
                            "available_actions": available_actions,
                        },
                    )
                except Exception as exc:
                    print(f"  [ERROR] LLM request failed: {exc}. Using fallback.")
                    response_text = FALLBACK_ACTION
                    trace(
                        "RAW MODEL RESPONSE",
                        {
                            "attempt": attempt,
                            "error": str(exc),
                            "fallback": response_text,
                            "available_actions": available_actions,
                        },
                    )

                action_dict = parse_model_action(response_text)
                if _action_type_allowed(action_dict, available_actions) and _action_params_allowed(action_dict, observation):
                    break

                invalid_action_type = action_dict.get("action_type", "unknown")
                if not _action_type_allowed(action_dict, available_actions):
                    correction_message = _action_validation_message(available_actions, invalid_action_type)
                    trace_payload = {
                        "attempt": attempt,
                        "invalid_action_type": invalid_action_type,
                        "available_actions": available_actions,
                    }
                else:
                    correction_message = _action_parameter_validation_message(observation, action_dict)
                    trace_payload = {
                        "attempt": attempt,
                        "invalid_action_type": invalid_action_type,
                        "invalid_action": action_dict,
                        "available_actions": available_actions,
                    }
                retry_messages = retry_messages + [
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": correction_message},
                ]
                trace("ACTION VALIDATION RETRY", trace_payload)
            else:
                raise RuntimeError(
                    "Model did not produce a valid action type after "
                    f"{MODEL_ACTION_MAX_RETRIES} attempts. Allowed: {available_actions}"
                )
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
        if action_dict.get("msg_id") is not None:
            action_str += f" {action_dict['msg_id']}"
        if action_dict.get("action_label") is not None:
            action_str += f" {action_dict['action_label']}"
        elif action_dict["action_type"] == "redact_sentence":
            action_str += (
                f" {action_dict.get('sentence_index', '')} "
                f"{action_dict.get('decision', '')}"
            )
        if action_dict.get("reason") is not None:
            action_str += f" {action_dict.get('reason_code', '')} :: {action_dict['reason']}"

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
            terminal_metrics = _extract_terminal_metrics(meta, observation)
            if TRACE_ENABLED:
                trace("TERMINAL METRICS", terminal_metrics)
            print("\n  Episode complete!")
            print(f"  Terminal score: {final_score:.4f}")
            print(f"  Cumulative step reward: {total_reward:.4f}")
            print(f"  Steps used: {step}")
            break
    else:
        print(f"  Reached max steps ({episode_max_steps}).")
        final_score = observation.get("metadata", {}).get("terminal_score", 0.0)
        terminal_metrics = _extract_terminal_metrics(observation.get("metadata", {}), observation)

    return {
        "task_id": task_id,
        "score": final_score,
        "history": history,
        "terminal_metrics": terminal_metrics,
    }


def main() -> None:
    """Run baseline inference across the currently implemented task set."""
    env_url = os.environ.get("DSAR_ENV_URL", "http://localhost:8000")
    if INFERENCE_MODE not in {"raw", "debug"}:
        print(f"ERROR: Unsupported DSAR_INFERENCE_MODE '{INFERENCE_MODE}'. Use raw or debug.")
        sys.exit(1)
    if CASE3_HEURISTIC_REQUESTED and INFERENCE_MODE != "debug":
        print(
            "ERROR: Case 3 heuristic assistance requires DSAR_INFERENCE_MODE=debug. "
            "Raw benchmark mode cannot use heuristic assistance."
        )
        sys.exit(1)

    print(f"DSAR Environment URL: {env_url}")
    print(f"LLM API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    if TASK_SEED_MAP:
        print(f"Task Seeds: {TASK_SEED_MAP}")
    elif EPISODE_SEED is not None:
        print(f"Seed: {EPISODE_SEED}")
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
        for task_id in tasks:
            task_scores = []
            for seed_value in MULTI_SEED_VALUES:
                try:
                    result = run_episode(env_url, task_id, episode_seed=_parse_optional_int(seed_value))
                    task_scores.append(result)
                except Exception as exc:
                    print(f"\n  ERROR running {task_id} seed {seed_value}: {exc}")
                    task_scores.append(
                        {
                            "task_id": task_id,
                            "score": 0.0,
                            "history": [f"ERROR: {exc}"],
                            "terminal_metrics": {},
                            "seed": seed_value,
                        }
                    )
            scores[task_id] = task_scores

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("MULTI-SEED SCORES")
        print(f"{'=' * 60}")
        for task_id, task_scores in scores.items():
            numeric_scores = [result["score"] for result in task_scores]
            mean_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
            print(
                f"  {task_id}: mean={mean_score:.4f} "
                f"min={min(numeric_scores):.4f} max={max(numeric_scores):.4f} "
                f"seeds={','.join(MULTI_SEED_VALUES)}"
            )
            for seed_value, result in zip(MULTI_SEED_VALUES, task_scores):
                metrics = result.get("terminal_metrics", {})
                print(
                    f"    seed={seed_value} score={result['score']:.4f} "
                    f"metrics={json.dumps(metrics, default=str)}"
                )
                failure_summary = metrics.get("failure_summary", [])
                if failure_summary:
                    print("    failure_summary:")
                    for line in failure_summary[:6]:
                        print(f"      {line}")
                if result["score"] <= 0.10 and result.get("history"):
                    print("    last_actions:")
                    for line in result["history"][-5:]:
                        print(f"      {line}")
        print(f"\n  Total runtime: {elapsed:.1f}s")
        print(f"{'=' * 60}")
        return

    for task_id in tasks:
        try:
            episode_seed = TASK_SEED_MAP.get(task_id, EPISODE_SEED)
            result = run_episode(env_url, task_id, episode_seed=episode_seed)
            scores[task_id] = result["score"]
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
