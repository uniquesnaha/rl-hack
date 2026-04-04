"""
Deterministic grading logic for the DSAR environment.

Case 1 and Case 2 use separate scoring helpers so both tasks remain
deterministic and independently testable.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from .constants import (
    CASE1_VALID_SILOS,
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


CONSTRAINT_LEAK_LIMIT = 2
MIN_STEPS_CASE1 = len(REQUESTER_DATA_FIELDS) + len(INTERNAL_ONLY_FIELDS) + len(CASE1_VALID_SILOS) + 1


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
        return 0.0
    if not all(isinstance(key, str) for key in draft_response.keys()):
        return 0.0

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
        return 0.0

    return round(max(0.0, min(1.0, 0.9 * compliance_score + 0.1 * efficiency_score)), 4)


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
            "schema_gate": 0.0,
            "task2_score": 0.0,
            "completion_coverage": 0.0,
            "termination_reason": termination_reason,
        }
    if not isinstance(ticket_ground_truth, dict):
        return {
            "schema_gate": 0.0,
            "task2_score": 0.0,
            "completion_coverage": 0.0,
            "termination_reason": termination_reason,
        }

    allowed_decisions = {"keep", "redact"}
    for ticket_id, sentence_decisions in processed_sentences.items():
        if ticket_id not in ticket_ground_truth:
            return {
                "schema_gate": 0.0,
                "task2_score": 0.0,
                "completion_coverage": 0.0,
                "termination_reason": termination_reason,
            }
        if not isinstance(sentence_decisions, dict):
            return {
                "schema_gate": 0.0,
                "task2_score": 0.0,
                "completion_coverage": 0.0,
                "termination_reason": termination_reason,
            }
        for sentence_index, decision in sentence_decisions.items():
            if sentence_index not in ticket_ground_truth[ticket_id]:
                return {
                    "schema_gate": 0.0,
                    "task2_score": 0.0,
                    "completion_coverage": 0.0,
                    "termination_reason": termination_reason,
                }
            if decision not in allowed_decisions:
                return {
                    "schema_gate": 0.0,
                    "task2_score": 0.0,
                    "completion_coverage": 0.0,
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
            (0.75 * redaction_f1)
            + (0.10 * unsafe_redaction_recall)
            + (0.15 * requester_keep_recall)
            - (leakage_rate * 0.45),
        ),
    )
    if not verification_succeeded:
        identity_score = 0.0
    else:
        if proportionate_attempt_count <= 1:
            identity_score = 0.85
        elif proportionate_attempt_count == 2:
            identity_score = 0.55
        else:
            identity_score = 0.30
        if disproportionate_used:
            identity_score -= 0.20
        identity_score = max(0.0, min(1.0, identity_score))
    pre_completion_score = max(0.0, min(1.0, 0.30 * identity_score + 0.70 * redaction_score))
    task2_score = max(0.0, min(1.0, completion_coverage * pre_completion_score))

    return {
        "schema_gate": 1.0,
        "identity_score": round(identity_score, 4),
        "phase1_reward_sum": round(phase1_reward_sum, 4),
        "kept_precision": round(kept_precision, 4),
        "kept_recall": round(kept_recall, 4),
        "redaction_f1": round(redaction_f1, 4),
        "requester_keep_recall": round(requester_keep_recall, 4),
        "unsafe_redaction_recall": round(unsafe_redaction_recall, 4),
        "leakage_rate": round(leakage_rate, 4),
        "redaction_score": round(redaction_score, 4),
        "pre_completion_score": round(pre_completion_score, 4),
        "leaked_pii_count": leaked_pii_count,
        "total_pii_sentences": total_pii_sentences,
        "total_internal_sentences": total_internal_sentences,
        "total_requester_sentences": total_requester_sentences,
        "completion_coverage": round(completion_coverage, 4),
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
