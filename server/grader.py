"""
Deterministic grading logic for the DSAR environment.

Case 1 and Case 2 use separate scoring helpers so both tasks remain
deterministic and independently testable.
"""

from __future__ import annotations

import os

from collections import OrderedDict

from typing import Any, Dict, List, Optional, Set

from .constants import (
    CASE1_VALID_SILOS,
    CASE5_COMPACT_INTERNAL_FIELDS,
    CASE5_COMPACT_REQUESTER_FIELDS,
    CASE5_FREE_STEPS,
    CASE5_TASK_ID,
    CASE4_FREE_STEPS,
    CASE4_TASK_ID,
    CASE3_ACTION_DISCLOSE,
    CASE3_ACTION_ESCALATE,
    CASE3_ACTION_EXCLUDE,
    CASE3_ACTION_PARTIAL_REDACT,
    CASE3_CALIBRATION_BONUS,
    CASE3_ESCALATION_HEALTH_KEYWORDS,
    CASE3_ESCALATION_KEYWORD_BONUS,
    CASE3_ESCALATION_LEGAL_KEYWORDS,
    CASE3_FREE_STEPS,
    CASE3_REASON_CODES,
    CASE3_REASON_CODE_BONUS,
    CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
    CASE2_DISPROPORTIONATE_METHODS,
    CASE2_FREE_STEPS,
    CASE2_PROPORTIONATE_METHODS,
    CASE2_SENTENCE_LABEL_INTERNAL,
    CASE2_SENTENCE_LABEL_PII,
    CASE2_SENTENCE_LABEL_REQUESTER,
    CASE2_VALID_SILOS,
    FREE_STEPS,
    INTERNAL_ONLY_FIELDS,
    MAX_STEPS,
    REQUESTER_DATA_FIELDS,
    STEP_COST,
)


CONSTRAINT_LEAK_LIMIT = 3
MIN_STEPS_CASE1 = len(REQUESTER_DATA_FIELDS) + len(INTERNAL_ONLY_FIELDS) + len(CASE1_VALID_SILOS) + 1
MIN_VALID_TASK_SCORE = 0.002
MAX_VALID_TASK_SCORE = 0.998


def clamp_task_score(value: float) -> float:
    """Clamp final task scores into the open interval (0, 1)."""
    return round(max(MIN_VALID_TASK_SCORE, min(MAX_VALID_TASK_SCORE, value)), 4)


PROCESS_MILESTONES = {
    "both_silos_queried": 0.05,
    "halfway_classified": 0.05,
    "all_fields_classified": 0.08,
    "identity_phase_silo_queried": 0.03,
    "identity_verified_first_attempt": 0.12,
    "redaction_phase_started": 0.05,
    "redaction_halfway": 0.05,
    "redaction_complete": 0.10,
    "adversarial_evidence_gathered": 0.05,
    "adversarial_verified_genuine": 0.10,
    "adversarial_flagged_correctly": 0.12,
    "breach_signal_detected": 0.12,
    "regulator_notified_on_time": 0.10,
    "requester_notified": 0.08,
    "first_message_triaged": 0.03,
    "health_trap_identified": 0.12,
    "mixed_message_identified": 0.05,
    "all_messages_triaged": 0.08,
}

_MILESTONES_FIRED: OrderedDict[str, None] = OrderedDict()
MAX_MILESTONE_KEYS = 4096
POTENTIAL_GAMMA = 0.99

_TRUTHY = {"1", "true", "yes", "on"}
_EASY_REMEDIATION_KEYWORDS = {
    "internal",
    "business",
    "commercial",
    "sensitive",
    "mistaken disclosure",
    "infrastructure",
    "routing key",
}
_MEDIUM_VERIFICATION_KEYWORDS = {
    "proportionate",
    "identity",
    "verification",
    "necessary",
    "excessive",
    "passport",
    "photo id",
}
_MEDIUM_REDACTION_KEYWORDS = {
    "third-party",
    "third party",
    "staff",
    "employee",
    "pii",
    "unsafe",
    "redaction",
}
_ADVERSARIAL_FLAG_KEYWORDS = {
    "spoof",
    "impersonation",
    "mismatch",
    "inconsistent",
    "postcode",
    "transaction",
    "name",
    "urgency",
}
_ADVERSARIAL_REMEDIATION_KEYWORDS = {
    "premature",
    "evidence",
    "verification",
    "false positive",
    "review",
    "identity",
    "proportionate",
}
_BREACH_FLAG_KEYWORDS = {
    "breach",
    "phishing",
    "exposed",
    "unauthorised",
    "incident",
    "forum",
    "reference",
}
_BREACH_NOTIFY_KEYWORDS = {
    "regulator",
    "article 33",
    "supervisory",
    "notify",
    "data subject",
    "article 34",
    "breach",
}
_HARD_ESCALATION_KEYWORDS = {
    "special category",
    "article 9",
    "health",
    "sensitive",
    "third-party",
    "third party",
    "human review",
    "balancing",
}
_HARD_REMEDIATION_KEYWORDS = {
    "bot",
    "system",
    "requester-entitled",
    "requester entitled",
    "misroute",
    "remediation",
    "disclosure error",
}
_DIAGNOSIS_ACTIONS = {
    "task_easy": {"file_remediation_note"},
    "task_medium": {"justify_verification_method", "file_redaction_remediation"},
    CASE4_TASK_ID: {"flag_adversarial", "file_remediation_note"},
    CASE5_TASK_ID: {"flag_breach_signal", "notify_regulator", "notify_requester", "file_remediation_note"},
    "task_hard": {"escalate_with_reason", "file_remediation_note"},
}
_DIAGNOSIS_TERMINAL_WEIGHTS = {
    "task_easy": 0.05,
    "task_medium": 0.07,
    CASE4_TASK_ID: 0.08,
    CASE5_TASK_ID: 0.08,
    "task_hard": 0.10,
}


def fire_milestone(episode_id: str, milestone: str) -> float:
    key = f"{episode_id}:{milestone}"
    if key in _MILESTONES_FIRED:
        return 0.0
    _MILESTONES_FIRED[key] = None
    while len(_MILESTONES_FIRED) > MAX_MILESTONE_KEYS:
        _MILESTONES_FIRED.popitem(last=False)
    return PROCESS_MILESTONES.get(milestone, 0.0)


def clear_episode_milestones(episode_id: str) -> None:
    prefix = f"{episode_id}:"
    stale_keys = [key for key in _MILESTONES_FIRED.keys() if key.startswith(prefix)]
    for key in stale_keys:
        _MILESTONES_FIRED.pop(key, None)


def quadratic_progress_score(correct_steps: int, optimal_steps: int) -> float:
    if optimal_steps <= 0:
        return clamp_task_score(0.0)
    ratio = min(1.0, max(0.0, correct_steps / optimal_steps))
    return clamp_task_score(ratio**2)


def compute_trap_avoidance_score(worsened_count: int) -> float:
    return clamp_task_score(max(0.0, 0.10 - 0.03 * worsened_count))


def blend_reactive_terminal_score(core_score: float, progress_score: float, trap_avoidance_score: float) -> float:
    if core_score <= 0.01:
        return clamp_task_score(core_score)
    return clamp_task_score(0.80 * core_score + 0.10 * progress_score + 0.10 * trap_avoidance_score)


def _matches_keywords(reason: str, keywords: Set[str]) -> int:
    lowered = reason.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def diagnosis_applicable(task_id: str, action_type: str) -> bool:
    return action_type in _DIAGNOSIS_ACTIONS.get(task_id, set())


def compute_diagnosis_quality(
    task_id: str,
    action_type: str,
    reason: Optional[str],
    reason_code: Optional[str] = None,
) -> Optional[float]:
    if not diagnosis_applicable(task_id, action_type):
        return None

    text = (reason or "").strip()
    if not text:
        return 0.0

    if task_id == "task_easy":
        matches = _matches_keywords(text, _EASY_REMEDIATION_KEYWORDS)
        return min(1.0, 0.35 * matches)

    if task_id == "task_medium" and action_type == "justify_verification_method":
        matches = _matches_keywords(text, _MEDIUM_VERIFICATION_KEYWORDS)
        return min(1.0, 0.30 * matches)

    if task_id == "task_medium" and action_type == "file_redaction_remediation":
        matches = _matches_keywords(text, _MEDIUM_REDACTION_KEYWORDS)
        return min(1.0, 0.30 * matches)

    if task_id == CASE4_TASK_ID and action_type == "flag_adversarial":
        matches = _matches_keywords(text, _ADVERSARIAL_FLAG_KEYWORDS)
        return min(1.0, 0.25 * matches)

    if task_id == CASE4_TASK_ID and action_type == "file_remediation_note":
        matches = _matches_keywords(text, _ADVERSARIAL_REMEDIATION_KEYWORDS)
        return min(1.0, 0.30 * matches)

    if task_id == CASE5_TASK_ID and action_type == "flag_breach_signal":
        matches = _matches_keywords(text, _BREACH_FLAG_KEYWORDS)
        return min(1.0, 0.25 * matches)

    if task_id == CASE5_TASK_ID and action_type in {"notify_regulator", "notify_requester"}:
        matches = _matches_keywords(text, _BREACH_NOTIFY_KEYWORDS)
        return min(1.0, 0.22 * matches)

    if task_id == CASE5_TASK_ID and action_type == "file_remediation_note":
        matches = _matches_keywords(text, _BREACH_FLAG_KEYWORDS | _BREACH_NOTIFY_KEYWORDS)
        return min(1.0, 0.20 * matches)

    if task_id == "task_hard" and action_type == "file_remediation_note":
        matches = _matches_keywords(text, _HARD_REMEDIATION_KEYWORDS)
        return min(1.0, 0.35 * matches)

    if task_id == "task_hard" and action_type == "escalate_with_reason":
        matches = _matches_keywords(text, _HARD_ESCALATION_KEYWORDS)
        quality = min(0.8, 0.20 * matches)
        if reason_code == CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA:
            quality += 0.20
        return min(1.0, quality)

    return 0.0


def compute_diagnosis_step_bonus(
    task_id: str,
    action_type: str,
    reason: Optional[str],
    reason_code: Optional[str] = None,
) -> float:
    quality = compute_diagnosis_quality(task_id, action_type, reason, reason_code)
    if quality is None or quality <= 0.0:
        return 0.0
    scale = 0.04 if action_type == "escalate_with_reason" else 0.03
    return round(scale * quality, 4)


def compute_diagnosis_terminal_score(diagnosis_scores: List[float]) -> Optional[float]:
    if not diagnosis_scores:
        return None
    avg_score = sum(diagnosis_scores) / len(diagnosis_scores)
    return clamp_task_score(avg_score)


def blend_diagnosis_terminal_score(task_id: str, base_score: float, diagnosis_score: Optional[float]) -> float:
    if diagnosis_score is None or base_score <= 0.01:
        return clamp_task_score(base_score)
    weight = _DIAGNOSIS_TERMINAL_WEIGHTS.get(task_id, 0.0)
    if weight <= 0.0:
        return clamp_task_score(base_score)
    return clamp_task_score((1.0 - weight) * base_score + weight * diagnosis_score)


def is_potential_shaping_enabled() -> bool:
    return os.environ.get("DSAR_ENABLE_POTENTIAL_SHAPING", "").strip().lower() in _TRUTHY


def _risk_drag(state: str) -> float:
    return {
        "clean": 0.0,
        "risk_elevated": -0.08,
        "regulatory_alert": -0.16,
        "enforcement": -0.24,
    }.get(state, 0.0)


def _field_classification_accuracy(episode: Any) -> float:
    classified_fields = getattr(episode, "classified_fields", set())
    if not classified_fields:
        return 0.0
    draft_response = getattr(episode, "draft_response", {})
    ground_truth = getattr(episode, "ground_truth", {})
    correct = 0
    for field_id in classified_fields:
        predicted = "REQUESTER_DATA" if field_id in draft_response else "INTERNAL_ONLY"
        if ground_truth.get(field_id) == predicted:
            correct += 1
    return correct / max(1, len(classified_fields))


def compute_potential(episode: Any) -> float:
    if not is_potential_shaping_enabled():
        return 0.0

    task_id = getattr(episode, "task_id", "")
    risk_drag = _risk_drag(getattr(episode, "compliance_risk_state", "clean"))

    if task_id == "task_easy":
        total = len(getattr(episode, "ground_truth", {}))
        classified = len(getattr(episode, "classified_fields", set()))
        queried = len(getattr(episode, "queried_silos", set()))
        progress = classified / max(1, total)
        accuracy = _field_classification_accuracy(episode)
        evidence = queried / max(1, len(CASE1_VALID_SILOS))
        return 0.40 * progress * accuracy + 0.15 * evidence + risk_drag

    if task_id == "task_medium":
        if getattr(episode, "phase", "") == "identity":
            queried = len(getattr(episode, "queried_silos", set())) / max(1, len(CASE2_VALID_SILOS))
            confidence = min(
                1.0,
                getattr(episode, "identity_confidence", 0.0) / max(0.01, getattr(episode, "verification_threshold", 1.0)),
            )
            verified = 0.25 if getattr(episode, "verification_succeeded", False) else 0.0
            safe_attempt = 0.10 if getattr(episode, "proportionate_attempt_count", 0) > 0 else 0.0
            return 0.15 * queried + 0.25 * confidence + verified + safe_attempt + risk_drag
        processed = sum(len(v) for v in getattr(episode, "processed_sentences", {}).values())
        total = sum(len(v) for v in getattr(episode, "ticket_ground_truth", {}).values())
        progress = processed / max(1, total)
        verified = 0.20 if getattr(episode, "verification_succeeded", False) else 0.0
        leak_penalty = min(0.10, 0.05 * getattr(episode, "leaked_pii_sentences", 0))
        return 0.45 * progress + verified - leak_penalty + risk_drag

    if task_id == CASE4_TASK_ID:
        queried = len(getattr(episode, "queried_silos", set())) / max(1, len(CASE2_VALID_SILOS))
        confidence = min(
            1.0,
            getattr(episode, "identity_confidence", 0.0) / max(0.01, getattr(episode, "verification_threshold", 1.0)),
        )
        resolved = 0.30 if (
            getattr(episode, "verification_succeeded", False)
            or getattr(episode, "adversarial_flagged", False)
        ) else 0.0
        return 0.20 * queried + 0.20 * confidence + resolved + risk_drag

    if task_id == CASE5_TASK_ID:
        total_fields = len(getattr(episode, "ground_truth", {}))
        queried = len(getattr(episode, "queried_silos", set())) / max(1, len(CASE1_VALID_SILOS))
        field_progress = len(getattr(episode, "classified_fields", set())) / max(1, total_fields)
        field_accuracy = _field_classification_accuracy(episode)
        if getattr(episode, "has_breach", False):
            workflow_progress = (
                (
                    int(getattr(episode, "breach_detected", False))
                    + int(getattr(episode, "regulator_notified", False))
                    + int(getattr(episode, "requester_notified", False))
                )
                / 3.0
            ) ** 2
        else:
            workflow_progress = 0.45 if getattr(episode, "false_breach_reported", False) else 1.0
        return 0.15 * queried + 0.35 * field_progress * field_accuracy + 0.35 * workflow_progress + risk_drag

    if task_id == "task_hard":
        processed = len(getattr(episode, "processed_messages", {}))
        total = len(getattr(episode, "slack_export", []))
        progress = processed / max(1, total)
        escalated_correctly = 0.0
        special_ids = getattr(episode, "special_category_message_ids", [])
        if special_ids:
            sc_id = special_ids[0]
            if getattr(episode, "processed_messages", {}).get(sc_id, {}).get("action") == CASE3_ACTION_ESCALATE:
                escalated_correctly = 0.20
                if sc_id in getattr(episode, "escalation_reason_codes", {}):
                    escalated_correctly += 0.10
        return 0.45 * progress + escalated_correctly + risk_drag

    return 0.0


def compute_potential_shaping_delta(phi_before: float, phi_after: float) -> float:
    if not is_potential_shaping_enabled():
        return 0.0
    return round(POTENTIAL_GAMMA * phi_after - phi_before, 4)


def compute_step_reward(
    action_type: str,
    *,
    silo_name: str | None = None,
    field_id: str | None = None,
    decision: str | None = None,
    ground_truth: Dict[str, str],
    queried_silos: Set[str],
    classified_fields: Set[str],
    visible_field_ids: Set[str] | None = None,
    step_number: int,
    task_id: str = "task_easy",
) -> float:
    """Compute the immediate reward for a Case 1 step action."""
    reward = 0.0

    if action_type == "query_silo":
        if silo_name not in CASE1_VALID_SILOS:
            reward = -0.05
        elif silo_name in queried_silos:
            reward = -0.05
        else:
            reward = 0.05

    elif action_type == "classify_field":
        if field_id is None or field_id not in ground_truth:
            reward = -0.05
        elif visible_field_ids is not None and field_id not in visible_field_ids:
            reward = -0.05
        elif field_id in classified_fields:
            reward = -0.05
        else:
            true_label = ground_truth[field_id]
            if true_label == "REQUESTER_DATA" and decision == "disclose":
                reward = 0.10
            elif true_label == "INTERNAL_ONLY" and decision == "withhold":
                reward = 0.10
            elif true_label == "INTERNAL_ONLY" and decision == "disclose":
                reward = -0.30
            elif true_label == "REQUESTER_DATA" and decision == "withhold":
                reward = -0.15
            else:
                reward = -0.05

    elif action_type == "compile_response":
        reward = 0.0

    if step_number > FREE_STEPS:
        reward -= STEP_COST

    return round(reward, 4)


def compute_terminal_score(
    draft_response: Dict[str, Any],
    ground_truth: Dict[str, str],
    queried_silos: Set[str],
    steps_used: int = 0,
    task_id: str = "task_easy",
) -> float:
    """Compute the Case 1 terminal score."""
    if not isinstance(draft_response, dict):
        return clamp_task_score(0.0)
    if not all(isinstance(key, str) for key in draft_response.keys()):
        return clamp_task_score(0.0)

    agent_disclosed = set(draft_response.keys())
    requester_fields = set(REQUESTER_DATA_FIELDS)
    internal_fields = set(INTERNAL_ONLY_FIELDS)

    correctly_disclosed = agent_disclosed & requester_fields
    leaked = agent_disclosed & internal_fields
    leaked_count = len(leaked)

    precision = len(correctly_disclosed) / len(agent_disclosed) if agent_disclosed else 0.0
    recall = len(correctly_disclosed) / len(requester_fields) if requester_fields else 0.0

    if precision + recall == 0:
        field_f1 = 0.0
    else:
        field_f1 = 2.0 * precision * recall / (precision + recall)

    privacy_penalty = leaked_count * 0.30 * (1.0 + leaked_count * 0.45)

    required_silos = CASE1_VALID_SILOS
    missing_required_silos = len(required_silos - queried_silos)
    extra_silos = len(queried_silos - required_silos)

    compliance_score = max(0.0, min(1.0, field_f1 - privacy_penalty))
    step_efficiency = max(
        0.0,
        min(1.0, (MAX_STEPS - steps_used) / (MAX_STEPS - MIN_STEPS_CASE1)),
    )
    silo_efficiency = max(0.0, 1.0 - 0.5 * missing_required_silos - 0.25 * extra_silos)
    efficiency_score = 0.5 * step_efficiency + 0.5 * silo_efficiency

    if compliance_score == 0.0:
        return clamp_task_score(0.0)

    return clamp_task_score(0.9 * compliance_score + 0.1 * efficiency_score)


def compute_step_reward_case2(
    action_type: str,
    *,
    phase: str,
    step_number: int,
    queried_silos: Set[str],
    silo_name: str | None = None,
    verification_method: str | None = None,
    correct_verification_method: str | None = None,
    verification_attempts: Set[str] | None = None,
    ticket_id: str | None = None,
    sentence_index: int | None = None,
    decision: str | None = None,
    ticket_ground_truth: Dict[str, Dict[int, str]] | None = None,
    processed_sentences: Dict[str, Dict[int, str]] | None = None,
    identity_verified: bool = False,
    all_sentences_processed: bool = True,
    blocked_compile_attempts: int = 0,
) -> float:
    """Compute Case 2 step rewards without changing the v1 reward table."""
    reward = 0.0
    verification_attempts = verification_attempts or set()
    ticket_ground_truth = ticket_ground_truth or {}
    processed_sentences = processed_sentences or {}

    if action_type == "query_silo":
        if phase not in {"identity", "redaction"}:
            reward = -0.05
        elif silo_name not in CASE2_VALID_SILOS:
            reward = -0.05
        elif silo_name in queried_silos:
            reward = -0.05
        else:
            reward = 0.05

    elif action_type == "verify_identity":
        if phase != "identity":
            reward = -0.05
        elif verification_method is None:
            reward = -0.05
        elif verification_method in verification_attempts:
            reward = -0.05
        elif verification_method in CASE2_DISPROPORTIONATE_METHODS:
            reward = -0.20
        elif verification_method in CASE2_PROPORTIONATE_METHODS:
            if verification_method == correct_verification_method:
                reward = 0.20 if verification_method == "registered_postcode" else 0.25
            else:
                reward = 0.10
        else:
            reward = -0.05

    elif action_type == "redact_span":
        if phase != "redaction":
            reward = -0.05
        elif ticket_id is None or sentence_index is None or decision is None:
            reward = -0.05
        elif ticket_id not in ticket_ground_truth:
            reward = -0.05
        elif sentence_index not in ticket_ground_truth[ticket_id]:
            reward = -0.05
        elif ticket_id in processed_sentences and sentence_index in processed_sentences[ticket_id]:
            reward = -0.05
        else:
            label = ticket_ground_truth[ticket_id][sentence_index]
            if label == CASE2_SENTENCE_LABEL_REQUESTER and decision == "keep":
                reward = 0.10
            elif label == CASE2_SENTENCE_LABEL_PII and decision == "redact":
                reward = 0.12
            elif label == CASE2_SENTENCE_LABEL_INTERNAL and decision == "redact":
                reward = 0.10
            elif label == CASE2_SENTENCE_LABEL_PII and decision == "keep":
                reward = -0.30
            elif label == CASE2_SENTENCE_LABEL_INTERNAL and decision == "keep":
                reward = -0.15
            elif label == CASE2_SENTENCE_LABEL_REQUESTER and decision == "redact":
                reward = -0.20
            else:
                reward = -0.05

    elif action_type == "compile_response":
        if phase == "identity" and not identity_verified:
            reward = -0.50
        elif phase == "redaction" and not all_sentences_processed:
            if blocked_compile_attempts <= 1:
                reward = -0.05
            elif blocked_compile_attempts == 2:
                reward = -0.08
            else:
                reward = -0.12
        else:
            reward = 0.0

    else:
        reward = -0.05

    if step_number > CASE2_FREE_STEPS:
        reward -= STEP_COST

    return round(reward, 4)


def compute_terminal_score_case2_details(
    processed_sentences: Dict[str, Dict[int, str]],
    ticket_ground_truth: Dict[str, Dict[int, str]],
    phase1_reward_sum: float,
    verification_succeeded: bool = True,
    proportionate_attempt_count: int = 1,
    disproportionate_used: bool = False,
    completed_all_sentences: bool = True,
    termination_reason: str = "compile_response",
) -> Dict[str, Any]:
    """Compute Case 2 terminal score and expose calibration/debug details."""
    if not isinstance(processed_sentences, dict):
        return {
            "schema_gate": clamp_task_score(0.0),
            "task2_score": clamp_task_score(0.0),
            "completion_coverage": clamp_task_score(0.0),
            "termination_reason": termination_reason,
        }
    if not isinstance(ticket_ground_truth, dict):
        return {
            "schema_gate": clamp_task_score(0.0),
            "task2_score": clamp_task_score(0.0),
            "completion_coverage": clamp_task_score(0.0),
            "termination_reason": termination_reason,
        }

    allowed_decisions = {"keep", "redact"}
    for ticket_id, sentence_decisions in processed_sentences.items():
        if ticket_id not in ticket_ground_truth:
            return {
                "schema_gate": clamp_task_score(0.0),
                "task2_score": clamp_task_score(0.0),
                "completion_coverage": clamp_task_score(0.0),
                "termination_reason": termination_reason,
            }
        if not isinstance(sentence_decisions, dict):
            return {
                "schema_gate": clamp_task_score(0.0),
                "task2_score": clamp_task_score(0.0),
                "completion_coverage": clamp_task_score(0.0),
                "termination_reason": termination_reason,
            }
        for sentence_index, decision in sentence_decisions.items():
            if sentence_index not in ticket_ground_truth[ticket_id]:
                return {
                    "schema_gate": clamp_task_score(0.0),
                    "task2_score": clamp_task_score(0.0),
                    "completion_coverage": clamp_task_score(0.0),
                    "termination_reason": termination_reason,
                }
            if decision not in allowed_decisions:
                return {
                    "schema_gate": clamp_task_score(0.0),
                    "task2_score": clamp_task_score(0.0),
                    "completion_coverage": clamp_task_score(0.0),
                    "termination_reason": termination_reason,
                }

    correctly_kept = 0
    total_kept_by_agent = 0
    total_requester_sentences = 0
    leaked_pii_count = 0
    total_pii_sentences = 0
    total_internal_sentences = 0
    correctly_redacted_unsafe = 0
    processed_count = 0
    total_sentence_count = 0

    for ticket_id, sentence_labels in ticket_ground_truth.items():
        for sentence_index, label in sentence_labels.items():
            total_sentence_count += 1
            if label == CASE2_SENTENCE_LABEL_REQUESTER:
                total_requester_sentences += 1
            elif label == CASE2_SENTENCE_LABEL_PII:
                total_pii_sentences += 1
            elif label == CASE2_SENTENCE_LABEL_INTERNAL:
                total_internal_sentences += 1

            decision = processed_sentences.get(ticket_id, {}).get(sentence_index)
            if decision in allowed_decisions:
                processed_count += 1
            if decision == "keep":
                total_kept_by_agent += 1
                if label == CASE2_SENTENCE_LABEL_REQUESTER:
                    correctly_kept += 1
                elif label == CASE2_SENTENCE_LABEL_PII:
                    leaked_pii_count += 1
            elif decision == "redact" and label in {
                CASE2_SENTENCE_LABEL_PII,
                CASE2_SENTENCE_LABEL_INTERNAL,
            }:
                correctly_redacted_unsafe += 1

    completion_coverage = processed_count / total_sentence_count if total_sentence_count > 0 else 0.0

    kept_precision = correctly_kept / total_kept_by_agent if total_kept_by_agent > 0 else 0.0
    kept_recall = correctly_kept / total_requester_sentences if total_requester_sentences > 0 else 0.0
    requester_keep_recall = kept_recall

    if kept_precision + kept_recall == 0:
        redaction_f1 = 0.0
    else:
        redaction_f1 = (2.0 * kept_precision * kept_recall) / (kept_precision + kept_recall)

    leakage_rate = leaked_pii_count / total_pii_sentences if total_pii_sentences > 0 else 0.0
    total_unsafe_sentences = total_pii_sentences + total_internal_sentences
    unsafe_redaction_recall = (
        correctly_redacted_unsafe / total_unsafe_sentences
        if total_unsafe_sentences > 0
        else 0.0
    )
    redaction_score = max(
        0.0,
        min(
            1.0,
            (0.70 * redaction_f1)
            + (0.08 * unsafe_redaction_recall)
            + (0.12 * requester_keep_recall)
            - (leakage_rate * 0.65),
        ),
    )
    if not verification_succeeded:
        identity_score = 0.0
    else:
        if proportionate_attempt_count <= 1:
            identity_score = 0.65
        elif proportionate_attempt_count == 2:
            identity_score = 0.38
        else:
            identity_score = 0.18
        if disproportionate_used:
            identity_score -= 0.20
        identity_score = max(0.0, min(1.0, identity_score))
    pre_completion_score = max(0.0, min(1.0, 0.30 * identity_score + 0.70 * redaction_score))
    task2_score = clamp_task_score(completion_coverage * pre_completion_score)

    return {
        "schema_gate": clamp_task_score(1.0),
        "identity_score": clamp_task_score(identity_score),
        "phase1_reward_sum": round(phase1_reward_sum, 4),
        "kept_precision": clamp_task_score(kept_precision),
        "kept_recall": clamp_task_score(kept_recall),
        "redaction_f1": clamp_task_score(redaction_f1),
        "requester_keep_recall": clamp_task_score(requester_keep_recall),
        "unsafe_redaction_recall": clamp_task_score(unsafe_redaction_recall),
        "leakage_rate": clamp_task_score(leakage_rate),
        "redaction_score": clamp_task_score(redaction_score),
        "pre_completion_score": clamp_task_score(pre_completion_score),
        "leaked_pii_count": leaked_pii_count,
        "total_pii_sentences": total_pii_sentences,
        "total_internal_sentences": total_internal_sentences,
        "total_requester_sentences": total_requester_sentences,
        "completion_coverage": clamp_task_score(completion_coverage),
        "termination_reason": termination_reason,
        "task2_score": round(task2_score, 4),
    }


def compute_terminal_score_case2(
    processed_sentences: Dict[str, Dict[int, str]],
    ticket_ground_truth: Dict[str, Dict[int, str]],
    phase1_reward_sum: float,
    verification_succeeded: bool = True,
    proportionate_attempt_count: int = 1,
    disproportionate_used: bool = False,
    completed_all_sentences: bool = True,
    termination_reason: str = "compile_response",
) -> float:
    """Compute the Case 2 terminal score."""
    return compute_terminal_score_case2_details(
        processed_sentences=processed_sentences,
        ticket_ground_truth=ticket_ground_truth,
        phase1_reward_sum=phase1_reward_sum,
        verification_succeeded=verification_succeeded,
        proportionate_attempt_count=proportionate_attempt_count,
        disproportionate_used=disproportionate_used,
        completed_all_sentences=completed_all_sentences,
        termination_reason=termination_reason,
    )["task2_score"]


def compute_step_reward_case4(
    *,
    action_type: str,
    step_number: int,
    queried_silos: Set[str],
    silo_name: str | None = None,
    verification_method: str | None = None,
    verification_attempts: Set[str] | None = None,
    correct_verification_method: str | None = None,
    spoofed_supported_methods: Set[str] | None = None,
    is_adversarial: bool = False,
    evidence_complete: bool = False,
    compile_ready: bool = False,
) -> float:
    reward = 0.0
    verification_attempts = verification_attempts or set()
    spoofed_supported_methods = spoofed_supported_methods or set()

    if action_type == "query_silo":
        if silo_name not in CASE2_VALID_SILOS:
            reward = -0.05
        elif silo_name in queried_silos:
            reward = -0.05
        else:
            reward = 0.05
    elif action_type == "verify_identity":
        if verification_method is None:
            reward = -0.05
        elif verification_method in verification_attempts:
            reward = -0.05
        elif not evidence_complete:
            reward = -0.05
        elif verification_method in CASE2_DISPROPORTIONATE_METHODS:
            reward = -0.20
        elif not is_adversarial and verification_method == correct_verification_method:
            reward = 0.20 if verification_method == "registered_postcode" else 0.25
        elif is_adversarial and verification_method in spoofed_supported_methods:
            reward = 0.05
        elif verification_method in CASE2_PROPORTIONATE_METHODS:
            reward = 0.05
        else:
            reward = -0.05
    elif action_type == "flag_adversarial":
        if not evidence_complete:
            reward = -0.05
        elif is_adversarial:
            reward = 0.20
        else:
            reward = -0.12
    elif action_type == "file_remediation_note":
        reward = 0.05
    elif action_type == "compile_response":
        reward = 0.0 if compile_ready else -0.05
    else:
        reward = -0.05

    if step_number > CASE4_FREE_STEPS:
        reward -= STEP_COST

    return round(reward, 4)


def compute_terminal_score_case4_details(
    *,
    is_adversarial: bool,
    adversarial_flagged: bool,
    verification_succeeded: bool,
    queried_silos: Set[str],
    disproportionate_used: bool,
    termination_reason: str,
) -> Dict[str, Any]:
    evidence_discipline = 1.0 if queried_silos == CASE2_VALID_SILOS else 0.5
    proportionality = 0.4 if disproportionate_used else 1.0

    if is_adversarial and adversarial_flagged:
        resolution_accuracy = 1.0
        false_positive_penalty = 0.0
        false_negative_penalty = 0.0
        incorrect_resolution_type = None
    elif (not is_adversarial) and verification_succeeded:
        resolution_accuracy = 1.0
        false_positive_penalty = 0.0
        false_negative_penalty = 0.0
        incorrect_resolution_type = None
    elif (not is_adversarial) and adversarial_flagged:
        resolution_accuracy = 0.4
        false_positive_penalty = 0.15
        false_negative_penalty = 0.0
        incorrect_resolution_type = "false_positive_rejection"
    elif is_adversarial and verification_succeeded:
        resolution_accuracy = 0.40
        false_positive_penalty = 0.0
        false_negative_penalty = 0.16
        incorrect_resolution_type = "spoof_accepted"
    else:
        resolution_accuracy = 0.15
        false_positive_penalty = 0.0
        false_negative_penalty = 0.15 if is_adversarial else 0.0
        incorrect_resolution_type = "unresolved_timeout" if termination_reason != "compile_response" else "unresolved_state"

    raw = (
        0.65 * resolution_accuracy
        + 0.20 * evidence_discipline
        + 0.15 * proportionality
        - false_positive_penalty
        - false_negative_penalty
    )
    task4_score = clamp_task_score(max(0.0, raw))

    return {
        "schema_gate": clamp_task_score(1.0),
        "resolution_accuracy": clamp_task_score(resolution_accuracy),
        "evidence_discipline": clamp_task_score(evidence_discipline),
        "proportionality": clamp_task_score(proportionality),
        "false_positive_penalty": clamp_task_score(false_positive_penalty),
        "false_negative_penalty": clamp_task_score(false_negative_penalty),
        "task4_score": task4_score,
        "termination_reason": termination_reason,
        "incorrect_resolution_type": incorrect_resolution_type,
        "is_adversarial_episode": is_adversarial,
    }


def compute_terminal_score_case4(
    *,
    is_adversarial: bool,
    adversarial_flagged: bool,
    verification_succeeded: bool,
    queried_silos: Set[str],
    disproportionate_used: bool,
    termination_reason: str,
) -> float:
    return compute_terminal_score_case4_details(
        is_adversarial=is_adversarial,
        adversarial_flagged=adversarial_flagged,
        verification_succeeded=verification_succeeded,
        queried_silos=queried_silos,
        disproportionate_used=disproportionate_used,
        termination_reason=termination_reason,
    )["task4_score"]


def compute_step_reward_case5(
    *,
    action_type: str,
    step_number: int,
    ground_truth: Dict[str, str],
    queried_silos: Set[str],
    classified_fields: Set[str],
    visible_field_ids: Set[str] | None = None,
    silo_name: str | None = None,
    field_id: str | None = None,
    decision: str | None = None,
    has_breach: bool = False,
    evidence_complete: bool = False,
    breach_detected: bool = False,
    regulator_notified: bool = False,
    requester_notified: bool = False,
    compile_ready: bool = False,
) -> float:
    reward = 0.0

    if action_type == "query_silo":
        if silo_name not in CASE1_VALID_SILOS:
            reward = -0.05
        elif silo_name in queried_silos:
            reward = -0.05
        else:
            reward = 0.05

    elif action_type == "classify_field":
        if field_id is None or field_id not in ground_truth:
            reward = -0.05
        elif visible_field_ids is not None and field_id not in visible_field_ids:
            reward = -0.05
        elif field_id in classified_fields:
            reward = -0.05
        else:
            true_label = ground_truth[field_id]
            if true_label == "REQUESTER_DATA" and decision == "disclose":
                reward = 0.10
            elif true_label == "INTERNAL_ONLY" and decision == "withhold":
                reward = 0.10
            elif true_label == "INTERNAL_ONLY" and decision == "disclose":
                reward = -0.30
            elif true_label == "REQUESTER_DATA" and decision == "withhold":
                reward = -0.15
            else:
                reward = -0.05

    elif action_type == "flag_breach_signal":
        if not evidence_complete:
            reward = -0.05
        elif has_breach:
            reward = 0.15
        else:
            reward = -0.10

    elif action_type == "notify_regulator":
        if not breach_detected or regulator_notified:
            reward = -0.08 if not breach_detected else -0.05
        else:
            reward = 0.12

    elif action_type == "notify_requester":
        if not regulator_notified or requester_notified:
            reward = -0.08 if not regulator_notified else -0.05
        else:
            reward = 0.08

    elif action_type == "file_remediation_note":
        reward = 0.05

    elif action_type == "compile_response":
        reward = 0.0 if compile_ready else -0.05

    else:
        reward = -0.05

    if step_number > CASE5_FREE_STEPS:
        reward -= STEP_COST

    return round(reward, 4)


def _compute_case5_field_score(
    draft_response: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> float:
    if not isinstance(draft_response, dict):
        return 0.0

    requester_fields = set(CASE5_COMPACT_REQUESTER_FIELDS)
    internal_fields = set(CASE5_COMPACT_INTERNAL_FIELDS)
    relevant_ground_truth = {
        field_id: label
        for field_id, label in ground_truth.items()
        if field_id in requester_fields | internal_fields
    }
    if not relevant_ground_truth:
        return 0.0

    agent_disclosed = {field_id for field_id in draft_response if field_id in relevant_ground_truth}
    correctly_disclosed = agent_disclosed & requester_fields
    leaked = agent_disclosed & internal_fields

    precision = len(correctly_disclosed) / len(agent_disclosed) if agent_disclosed else 0.0
    recall = len(correctly_disclosed) / len(requester_fields) if requester_fields else 0.0
    field_f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    privacy_penalty = len(leaked) * 0.22
    return max(0.0, min(1.0, field_f1 - privacy_penalty))


def _count_case5_internal_leaks(
    draft_response: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> int:
    if not isinstance(draft_response, dict):
        return 0
    internal_fields = set(CASE5_COMPACT_INTERNAL_FIELDS)
    relevant_ground_truth = {
        field_id: label
        for field_id, label in ground_truth.items()
        if field_id in internal_fields
    }
    agent_disclosed = {field_id for field_id in draft_response if field_id in relevant_ground_truth}
    return len(agent_disclosed & internal_fields)


def compute_terminal_score_case5_details(
    *,
    draft_response: Dict[str, Any],
    ground_truth: Dict[str, str],
    has_breach: bool,
    breach_detected: bool,
    breach_detection_step: Optional[int] = None,
    breach_flag_after_full_review: bool = False,
    regulator_notified: bool,
    requester_notified: bool,
    false_breach_reported: bool,
    notification_order_violated: bool,
    termination_reason: str,
) -> Dict[str, Any]:
    field_score = _compute_case5_field_score(draft_response, ground_truth)
    leaked_internal_count = _count_case5_internal_leaks(draft_response, ground_truth)
    field_leak_penalty = min(0.45, 0.45 * leaked_internal_count)
    breach_timing_discipline = 1.0

    if has_breach:
        breach_detection = 1.0 if breach_detected else 0.10
        if regulator_notified and requester_notified:
            notification_completeness = 1.0
            incorrect_resolution_type = None
        elif regulator_notified or requester_notified:
            notification_completeness = 0.25
            incorrect_resolution_type = "partial_breach_notification"
        else:
            notification_completeness = 0.0
            incorrect_resolution_type = "breach_signal_missed" if not breach_detected else "notifications_missing"

        workflow_completion = (
            (
                int(breach_detected)
                + int(regulator_notified)
                + int(requester_notified)
            )
            / 3.0
        ) ** 2
        if breach_detected:
            if breach_flag_after_full_review:
                breach_timing_discipline = 0.15
            elif breach_detection_step is not None and breach_detection_step >= 8:
                breach_timing_discipline = 0.45
            elif breach_detection_step is not None and breach_detection_step >= 6:
                breach_timing_discipline = 0.70
        workflow_completion *= breach_timing_discipline
        completion_coverage = field_score * (0.35 + 0.65 * workflow_completion)
    else:
        if false_breach_reported or regulator_notified or requester_notified:
            breach_detection = 0.45
            notification_completeness = 0.35 if (regulator_notified or requester_notified) else 0.60
            completion_coverage = 0.65 * field_score
            incorrect_resolution_type = "false_breach_report"
        else:
            breach_detection = 1.0
            notification_completeness = 1.0
            completion_coverage = field_score
            incorrect_resolution_type = None

    proportionality_discipline = max(
        0.20,
        1.0
        - (0.35 if false_breach_reported else 0.0)
        - (0.35 if notification_order_violated else 0.0),
    )

    raw = (
        0.50 * field_score
        + 0.20 * breach_detection
        + 0.20 * notification_completeness
        + 0.10 * proportionality_discipline
        - field_leak_penalty
    )
    task5_score = clamp_task_score(max(0.0, raw * completion_coverage))

    return {
        "schema_gate": clamp_task_score(1.0),
        "field_score": clamp_task_score(field_score),
        "field_leak_penalty": clamp_task_score(field_leak_penalty),
        "breach_timing_discipline": clamp_task_score(breach_timing_discipline),
        "breach_detection": clamp_task_score(breach_detection),
        "notification_completeness": clamp_task_score(notification_completeness),
        "proportionality_discipline": clamp_task_score(proportionality_discipline),
        "completion_coverage": clamp_task_score(completion_coverage),
        "task5_score": task5_score,
        "has_breach_episode": has_breach,
        "breach_detected": breach_detected,
        "regulator_notified": regulator_notified,
        "requester_notified": requester_notified,
        "termination_reason": termination_reason,
        "incorrect_resolution_type": incorrect_resolution_type,
    }


def compute_step_reward_case3(
    action_type: str,
    *,
    step_number: int,
    msg_id: str | None = None,
    action_label: str | None = None,
    sentence_index: int | None = None,
    decision: str | None = None,
    reason: str | None = None,
    reason_code: str | None = None,
    ground_truth: Dict[str, Dict[str, Any]] | None = None,
    processed_messages: Dict[str, Dict[str, Any]] | None = None,
    escalation_log: Dict[str, str] | None = None,
    escalation_reason_codes: Dict[str, str] | None = None,
) -> float:
    reward = 0.0
    ground_truth = ground_truth or {}
    processed_messages = processed_messages or {}
    escalation_log = escalation_log or {}
    escalation_reason_codes = escalation_reason_codes or {}

    if action_type == "process_message":
        if msg_id is None or action_label is None or msg_id not in ground_truth:
            reward = -0.05
        elif msg_id in processed_messages:
            reward = -0.05
        else:
            true_action = ground_truth[msg_id]["action"]
            is_special = ground_truth[msg_id].get("is_special_category", False)
            if is_special and action_label == CASE3_ACTION_ESCALATE:
                reward = 0.15
            elif is_special and action_label == CASE3_ACTION_DISCLOSE:
                reward = -0.20
            elif is_special and action_label == CASE3_ACTION_PARTIAL_REDACT:
                reward = -0.10
            elif is_special and action_label == CASE3_ACTION_EXCLUDE:
                reward = -0.15
            elif true_action == CASE3_ACTION_DISCLOSE and action_label == CASE3_ACTION_DISCLOSE:
                reward = 0.05
            elif true_action == CASE3_ACTION_PARTIAL_REDACT and action_label == CASE3_ACTION_PARTIAL_REDACT:
                reward = 0.05
            elif true_action == CASE3_ACTION_EXCLUDE and action_label == CASE3_ACTION_EXCLUDE:
                reward = 0.05
            elif action_label == CASE3_ACTION_ESCALATE and true_action != CASE3_ACTION_ESCALATE:
                reward = -0.08
            elif true_action == CASE3_ACTION_DISCLOSE and action_label == CASE3_ACTION_EXCLUDE:
                reward = -0.10
            elif true_action == CASE3_ACTION_PARTIAL_REDACT and action_label == CASE3_ACTION_DISCLOSE:
                reward = -0.15
            elif true_action == CASE3_ACTION_PARTIAL_REDACT and action_label == CASE3_ACTION_EXCLUDE:
                reward = -0.10
            else:
                reward = -0.05

    elif action_type == "redact_sentence":
        if msg_id is None or sentence_index is None or decision is None:
            reward = -0.05
        elif processed_messages.get(msg_id, {}).get("action") != CASE3_ACTION_PARTIAL_REDACT:
            reward = -0.05
        elif not ground_truth.get(msg_id, {}).get("sentence_ground_truth"):
            reward = -0.05
        elif sentence_index in processed_messages.get(msg_id, {}).get("sentence_decisions", {}):
            reward = -0.05
        else:
            sentence_truth = ground_truth[msg_id]["sentence_ground_truth"]
            true_label = sentence_truth.get(sentence_index)
            if true_label == CASE2_SENTENCE_LABEL_REQUESTER and decision == "keep":
                reward = 0.15
            elif true_label == CASE2_SENTENCE_LABEL_PII and decision == "redact":
                reward = 0.15
            elif true_label == CASE2_SENTENCE_LABEL_REQUESTER and decision == "redact":
                reward = -0.20
            elif true_label == CASE2_SENTENCE_LABEL_PII and decision == "keep":
                reward = -0.25
            else:
                reward = -0.05

    elif action_type == "escalate_with_reason":
        if msg_id is None or reason is None or reason_code is None:
            reward = -0.05
        elif processed_messages.get(msg_id, {}).get("action") != CASE3_ACTION_ESCALATE:
            reward = -0.05
        elif msg_id in escalation_log or msg_id in escalation_reason_codes:
            reward = -0.05
        else:
            lowered = reason.lower()
            has_health = any(term in lowered for term in CASE3_ESCALATION_HEALTH_KEYWORDS)
            has_legal = any(term in lowered for term in CASE3_ESCALATION_LEGAL_KEYWORDS)
            if reason_code == CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA:
                reward += CASE3_REASON_CODE_BONUS
            if has_health and has_legal:
                reward += CASE3_ESCALATION_KEYWORD_BONUS

    elif action_type == "compile_response":
        reward = 0.0
    else:
        reward = -0.05

    if step_number > CASE3_FREE_STEPS:
        reward -= STEP_COST
    return round(reward, 4)


def _schema_gate_case3(agent_output: Dict[str, Any]) -> bool:
    if not isinstance(agent_output, dict):
        return False
    required_keys = {"processed_messages", "escalation_log", "escalation_reason_codes"}
    if not required_keys.issubset(agent_output.keys()):
        return False
    processed_messages = agent_output.get("processed_messages")
    escalation_log = agent_output.get("escalation_log")
    escalation_reason_codes = agent_output.get("escalation_reason_codes")
    if (
        not isinstance(processed_messages, dict)
        or not isinstance(escalation_log, dict)
        or not isinstance(escalation_reason_codes, dict)
    ):
        return False
    valid_actions = {
        CASE3_ACTION_DISCLOSE,
        CASE3_ACTION_PARTIAL_REDACT,
        CASE3_ACTION_EXCLUDE,
        CASE3_ACTION_ESCALATE,
    }
    for msg_id, decision in processed_messages.items():
        if not isinstance(msg_id, str) or not isinstance(decision, dict):
            return False
        if decision.get("action") not in valid_actions:
            return False
        sentence_decisions = decision.get("sentence_decisions")
        if sentence_decisions is not None and not isinstance(sentence_decisions, dict):
            return False
    for msg_id, reason_code in escalation_reason_codes.items():
        if not isinstance(msg_id, str) or reason_code not in CASE3_REASON_CODES:
            return False
    return True


def _count_pii_breaches_case3(
    agent_processed: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> int:
    breaches = 0
    for msg_id, gt in ground_truth.items():
        agent_decision = agent_processed.get(msg_id, {})
        action = agent_decision.get("action")
        if gt.get("is_special_category") and action == CASE3_ACTION_DISCLOSE:
            breaches += 1
            continue
        if gt.get("action") == CASE3_ACTION_PARTIAL_REDACT:
            if action == CASE3_ACTION_DISCLOSE:
                breaches += 1
                continue
            if action == CASE3_ACTION_PARTIAL_REDACT:
                sentence_decisions = agent_decision.get("sentence_decisions", {})
                if sentence_decisions.get(1) == "keep":
                    breaches += 1
    return breaches


def _compute_c1_case3(
    agent_processed: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> float:
    per_message_scores = []
    max_possible = 0.40
    for msg_id, gt in ground_truth.items():
        agent_action = agent_processed.get(msg_id, {}).get("action")
        correct_action = gt["action"]
        is_special = gt.get("is_special_category", False)

        if is_special and agent_action == CASE3_ACTION_DISCLOSE:
            per_message_scores.append(-0.20)
        elif is_special and agent_action == CASE3_ACTION_ESCALATE:
            per_message_scores.append(0.15)
        elif agent_action == correct_action:
            per_message_scores.append(0.05)
        elif agent_action == CASE3_ACTION_ESCALATE and correct_action != CASE3_ACTION_ESCALATE:
            per_message_scores.append(-0.08)
        elif correct_action == CASE3_ACTION_DISCLOSE and agent_action == CASE3_ACTION_EXCLUDE:
            per_message_scores.append(-0.10)
        elif correct_action == CASE3_ACTION_PARTIAL_REDACT and agent_action == CASE3_ACTION_DISCLOSE:
            per_message_scores.append(-0.15)
        else:
            per_message_scores.append(-0.05)

    raw = sum(per_message_scores) / max_possible
    return clamp_task_score(raw)


def _compute_c2_case3(
    agent_processed: Dict[str, Dict[str, Any]],
    mixed_sentence_msg_id: str,
) -> float:
    agent_decision = agent_processed.get(mixed_sentence_msg_id, {})
    if agent_decision.get("action") != CASE3_ACTION_PARTIAL_REDACT:
        return clamp_task_score(0.0)

    sentence_decisions = agent_decision.get("sentence_decisions", {})
    score = 0.0
    if sentence_decisions.get(0) == "keep":
        score += 0.15
    if sentence_decisions.get(1) == "redact":
        score += 0.15
    if sentence_decisions.get(1) == "keep":
        score -= 0.30
    if sentence_decisions.get(0) == "redact":
        score -= 0.20

    return clamp_task_score(score / 0.30)


def _compute_c3_case3(
    agent_processed: Dict[str, Dict[str, Any]],
    escalation_log: Dict[str, str],
    escalation_reason_codes: Dict[str, str],
    ground_truth: Dict[str, Dict[str, Any]],
    special_category_msg_id: str,
) -> float:
    agent_action = agent_processed.get(special_category_msg_id, {}).get("action")
    if agent_action == CASE3_ACTION_ESCALATE:
        base = 0.20
    elif agent_action == CASE3_ACTION_DISCLOSE:
        base = -0.30
    else:
        base = -0.15

    reason = escalation_log.get(special_category_msg_id, "").lower()
    reason_code = escalation_reason_codes.get(special_category_msg_id)
    reason_code_bonus = (
        CASE3_REASON_CODE_BONUS
        if reason_code == CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA
        else 0.0
    )
    has_health = any(term in reason for term in CASE3_ESCALATION_HEALTH_KEYWORDS)
    has_legal = any(term in reason for term in CASE3_ESCALATION_LEGAL_KEYWORDS)
    keyword_bonus = CASE3_ESCALATION_KEYWORD_BONUS if has_health and has_legal else 0.0

    escalated_ids = {
        msg_id for msg_id, decision in agent_processed.items()
        if decision.get("action") == CASE3_ACTION_ESCALATE
    }
    correct_escalation_ids = {
        msg_id for msg_id, gt in ground_truth.items()
        if gt["action"] == CASE3_ACTION_ESCALATE
    }
    unnecessary_escalations = escalated_ids - correct_escalation_ids
    lazy_penalty = len(unnecessary_escalations) * -0.08
    calibration_bonus = (
        CASE3_CALIBRATION_BONUS
        if escalated_ids == correct_escalation_ids and agent_action == CASE3_ACTION_ESCALATE
        else 0.0
    )

    raw = base + reason_code_bonus + keyword_bonus + lazy_penalty + calibration_bonus
    return clamp_task_score(raw / 0.40)


def _case3_failure_explanations(
    agent_processed: Dict[str, Dict[str, Any]],
    escalation_log: Dict[str, str],
    escalation_reason_codes: Dict[str, str],
    ground_truth: Dict[str, Dict[str, Any]],
) -> tuple[list[str], Dict[str, Dict[str, Any]], list[str]]:
    failure_summary: list[str] = []
    message_diagnostics: Dict[str, Dict[str, Any]] = {}
    incorrect_message_ids: list[str] = []

    for msg_id, gt in ground_truth.items():
        actual = agent_processed.get(msg_id, {}).get("action")
        expected = gt["action"]
        notes: list[str] = []
        if actual != expected:
            incorrect_message_ids.append(msg_id)
            if gt.get("is_special_category") and actual != CASE3_ACTION_ESCALATE:
                notes.append("special-category health trap not escalated")
            elif expected == CASE3_ACTION_PARTIAL_REDACT and actual == CASE3_ACTION_DISCLOSE:
                notes.append("mixed-ownership message fully disclosed")
            elif expected == CASE3_ACTION_PARTIAL_REDACT and actual != CASE3_ACTION_PARTIAL_REDACT:
                notes.append("mixed-ownership message not sent to sentence-level redaction")
            elif expected == CASE3_ACTION_EXCLUDE and actual == CASE3_ACTION_DISCLOSE:
                notes.append("bot/system output disclosed")
            elif expected == CASE3_ACTION_DISCLOSE and actual == CASE3_ACTION_ESCALATE:
                notes.append("unnecessary escalation on requester-entitled data")
            elif expected == CASE3_ACTION_DISCLOSE and actual == CASE3_ACTION_EXCLUDE:
                notes.append("requester-entitled data excluded")
            else:
                notes.append(f"expected {expected} but got {actual}")

        if expected == CASE3_ACTION_PARTIAL_REDACT and actual == CASE3_ACTION_PARTIAL_REDACT:
            sentence_truth = gt.get("sentence_ground_truth", {})
            sentence_decisions = agent_processed.get(msg_id, {}).get("sentence_decisions", {})
            for sentence_idx, label in sentence_truth.items():
                actual_decision = sentence_decisions.get(sentence_idx)
                if actual_decision is None:
                    actual_decision = sentence_decisions.get(str(sentence_idx))
                if label == CASE2_SENTENCE_LABEL_REQUESTER and actual_decision != "keep":
                    notes.append(f"sentence {sentence_idx} requester data not kept")
                elif label == CASE2_SENTENCE_LABEL_PII and actual_decision != "redact":
                    notes.append(f"sentence {sentence_idx} third-party data not redacted")

        if expected == CASE3_ACTION_ESCALATE:
            if msg_id not in escalation_reason_codes:
                notes.append("missing escalation reason_code")
            if msg_id not in escalation_log:
                notes.append("missing escalation reason text")

        if notes:
            message_diagnostics[msg_id] = {
                "expected_action": expected,
                "actual_action": actual,
                "notes": notes,
            }
            for note in notes:
                failure_summary.append(f"{msg_id}: {note}")

    return failure_summary, message_diagnostics, incorrect_message_ids


def compute_terminal_score_case3(
    agent_output: Dict[str, Any],
    ground_truth: Dict[str, Dict[str, Any]],
    special_category_msg_ids: List[str],
    mixed_sentence_msg_id: str,
) -> Dict[str, Any]:
    default_details = {
        "schema_gate": clamp_task_score(0.0),
        "c1_message_accuracy": clamp_task_score(0.0),
        "c2_sentence_redaction": clamp_task_score(0.0),
        "c3_escalation_quality": clamp_task_score(0.0),
        "calibration_penalty": clamp_task_score(0.0),
        "n_pii_breaches": 0,
        "privacy_penalty": clamp_task_score(0.0),
        "task3_score": clamp_task_score(0.0),
        "termination_reason": "unknown",
        "failure_summary": [],
        "message_diagnostics": {},
        "incorrect_message_ids": [],
    }
    if not _schema_gate_case3(agent_output):
        details = dict(default_details)
        details["termination_reason"] = "schema_gate_failed"
        details["failure_summary"] = ["schema gate failed"]
        return details

    agent_processed = agent_output["processed_messages"]
    escalation_log = agent_output.get("escalation_log", {})
    escalation_reason_codes = agent_output.get("escalation_reason_codes", {})
    failure_summary, message_diagnostics, incorrect_message_ids = _case3_failure_explanations(
        agent_processed,
        escalation_log,
        escalation_reason_codes,
        ground_truth,
    )

    for sc_id in special_category_msg_ids:
        if agent_processed.get(sc_id, {}).get("action") == CASE3_ACTION_DISCLOSE:
            details = dict(default_details)
            details.update(
                {
                    "schema_gate": clamp_task_score(1.0),
                    "constraint_violated": True,
                    "termination_reason": "special_category_disclosed",
                    "n_pii_breaches": max(1, _count_pii_breaches_case3(agent_processed, ground_truth)),
                    "privacy_penalty": clamp_task_score(0.45),
                    "failure_summary": failure_summary or [f"{sc_id}: special-category health trap disclosed"],
                    "message_diagnostics": message_diagnostics,
                    "incorrect_message_ids": incorrect_message_ids,
                }
            )
            return details

    c1 = _compute_c1_case3(agent_processed, ground_truth)
    c2 = _compute_c2_case3(agent_processed, mixed_sentence_msg_id)
    c3 = _compute_c3_case3(
        agent_processed,
        escalation_log,
        escalation_reason_codes,
        ground_truth,
        special_category_msg_ids[0],
    )
    calibration_penalty = 0.0
    for diagnostic in message_diagnostics.values():
        expected = diagnostic.get("expected_action")
        actual = diagnostic.get("actual_action")
        if expected == CASE3_ACTION_DISCLOSE and actual == CASE3_ACTION_PARTIAL_REDACT:
            calibration_penalty += 0.10
        elif expected == CASE3_ACTION_DISCLOSE and actual == CASE3_ACTION_ESCALATE:
            calibration_penalty += 0.10
        elif expected == CASE3_ACTION_DISCLOSE and actual == CASE3_ACTION_EXCLUDE:
            calibration_penalty += 0.12
        elif expected == CASE3_ACTION_EXCLUDE and actual == CASE3_ACTION_DISCLOSE:
            calibration_penalty += 0.14

    n_breaches = _count_pii_breaches_case3(agent_processed, ground_truth)
    privacy_penalty = n_breaches * 0.30 * (1.0 + n_breaches * 0.50)
    mixed_redaction_penalty = 0.0
    if c2 <= 0.10:
        mixed_redaction_penalty = 0.25
    elif c2 < 0.50:
        mixed_redaction_penalty = 0.10

    raw = (
        0.35 * c1
        + 0.35 * c2
        + 0.30 * c3
        - privacy_penalty
        - calibration_penalty
        - mixed_redaction_penalty
    )
    task3_score = clamp_task_score(raw)

    return {
        "schema_gate": clamp_task_score(1.0),
        "c1_message_accuracy": clamp_task_score(c1),
        "c2_sentence_redaction": clamp_task_score(c2),
        "c3_escalation_quality": clamp_task_score(c3),
        "calibration_penalty": clamp_task_score(calibration_penalty),
        "mixed_redaction_penalty": clamp_task_score(mixed_redaction_penalty),
        "n_pii_breaches": n_breaches,
        "privacy_penalty": clamp_task_score(privacy_penalty),
        "task3_score": round(task3_score, 4),
        "termination_reason": "compile_response",
        "failure_summary": failure_summary,
        "message_diagnostics": message_diagnostics,
        "incorrect_message_ids": incorrect_message_ids,
    }
