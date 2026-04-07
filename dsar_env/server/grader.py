"""
Deterministic grading logic for the DSAR environment.

Case 1 and Case 2 use separate scoring helpers so both tasks remain
deterministic and independently testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .constants import (
    CASE1_VALID_SILOS,
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
TASK_SCORE_EPS = 0.0001


def clamp_task_score(value: float) -> float:
    """Clamp final task scores into the open interval (0, 1)."""
    return round(max(TASK_SCORE_EPS, min(1.0 - TASK_SCORE_EPS, value)), 4)


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
            "schema_gate": 0.0,
            "task2_score": clamp_task_score(0.0),
            "completion_coverage": 0.0,
            "termination_reason": termination_reason,
        }
    if not isinstance(ticket_ground_truth, dict):
        return {
            "schema_gate": 0.0,
            "task2_score": clamp_task_score(0.0),
            "completion_coverage": 0.0,
            "termination_reason": termination_reason,
        }

    allowed_decisions = {"keep", "redact"}
    for ticket_id, sentence_decisions in processed_sentences.items():
        if ticket_id not in ticket_ground_truth:
            return {
                "schema_gate": 0.0,
                "task2_score": clamp_task_score(0.0),
                "completion_coverage": 0.0,
                "termination_reason": termination_reason,
            }
        if not isinstance(sentence_decisions, dict):
            return {
                "schema_gate": 0.0,
                "task2_score": clamp_task_score(0.0),
                "completion_coverage": 0.0,
                "termination_reason": termination_reason,
            }
        for sentence_index, decision in sentence_decisions.items():
            if sentence_index not in ticket_ground_truth[ticket_id]:
                return {
                    "schema_gate": 0.0,
                    "task2_score": clamp_task_score(0.0),
                    "completion_coverage": 0.0,
                    "termination_reason": termination_reason,
                }
            if decision not in allowed_decisions:
                return {
                    "schema_gate": 0.0,
                    "task2_score": clamp_task_score(0.0),
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
    return round(max(0.0, min(1.0, raw)), 4)


def _compute_c2_case3(
    agent_processed: Dict[str, Dict[str, Any]],
    mixed_sentence_msg_id: str,
) -> float:
    agent_decision = agent_processed.get(mixed_sentence_msg_id, {})
    if agent_decision.get("action") != CASE3_ACTION_PARTIAL_REDACT:
        return 0.0

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

    return round(max(0.0, min(1.0, score / 0.30)), 4)


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
    return round(max(0.0, min(1.0, raw / 0.40)), 4)


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
        "schema_gate": 0.0,
        "c1_message_accuracy": 0.0,
        "c2_sentence_redaction": 0.0,
        "c3_escalation_quality": 0.0,
        "calibration_penalty": 0.0,
        "n_pii_breaches": 0,
        "privacy_penalty": 0.0,
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
                    "schema_gate": 1.0,
                    "constraint_violated": True,
                    "termination_reason": "special_category_disclosed",
                    "n_pii_breaches": max(1, _count_pii_breaches_case3(agent_processed, ground_truth)),
                    "privacy_penalty": 0.45,
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
    raw = 0.40 * c1 + 0.30 * c2 + 0.30 * c3 - privacy_penalty - calibration_penalty
    task3_score = clamp_task_score(raw)

    return {
        "schema_gate": 1.0,
        "c1_message_accuracy": c1,
        "c2_sentence_redaction": c2,
        "c3_escalation_quality": c3,
        "calibration_penalty": round(calibration_penalty, 4),
        "n_pii_breaches": n_breaches,
        "privacy_penalty": round(privacy_penalty, 4),
        "task3_score": round(task3_score, 4),
        "termination_reason": "compile_response",
        "failure_summary": failure_summary,
        "message_diagnostics": message_diagnostics,
        "incorrect_message_ids": incorrect_message_ids,
    }
