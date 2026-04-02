"""
Deterministic grading logic for DSAR environment.

Two independent scoring functions:
  1. compute_step_reward() — fires on every step() call
  2. compute_terminal_score() — fires when compile_response is called

Terminal formula (frozen spec — Case 1):
    schema_gate = 1 if draft_response is a valid {str: any} dict, else 0
    field_f1    = (2 × precision × recall) / (precision + recall)
    privacy_penalty = n × 0.30 × (1 + n × 0.45)   ← non-linear, intentionally steep
    compliance_score = clamp(field_f1 − privacy_penalty, 0.0, 1.0)
    step_efficiency = clamp((MAX_STEPS − steps_used) / (MAX_STEPS − MIN_STEPS_CASE1), 0.0, 1.0)
    silo_efficiency = max(0, 1.0 − 0.5 × missing_required_silos − 0.25 × extra_silos)
    efficiency_score = 0.5 × step_efficiency + 0.5 × silo_efficiency
    task1_score = 0.0 if compliance_score == 0 else clamp(0.9 × compliance_score + 0.1 × efficiency_score, 0.0, 1.0)

The non-linear privacy penalty (quadratic growth) is critical:
    1 leak  → penalty = 0.435  (one error cannot be offset by three correct)
    2 leaks → penalty = 1.14   (still floors to 0.0 — catastrophic leakage)
    3 leaks → episode terminates immediately via constraint_violated flag

The step-level reward asymmetry (-0.30 for leaks vs +0.10 for correct)
is preserved unchanged from the frozen spec.

All arithmetic is pure Python — no ML, no NLP, fully deterministic.
"""

from typing import Any, Dict, Set

from .constants import (
    CASE1_VALID_SILOS,
    FREE_STEPS,
    INTERNAL_ONLY_FIELDS,
    MAX_STEPS,
    REQUESTER_DATA_FIELDS,
    STEP_COST,
)

# Maximum leaks before constraint_violated triggers episode termination
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
    """Compute the immediate reward for a single step action.

    NOTE: queried_silos and classified_fields must be PRE-MUTATION snapshots
    (taken before the action modifies the episode state) so the grader
    correctly evaluates the action as if it hasn't happened yet.

    Args:
        action_type: One of 'query_silo', 'classify_field', 'compile_response'
        silo_name: For query_silo actions
        field_id: For classify_field actions
        decision: For classify_field actions ('disclose' or 'withhold')
        ground_truth: Hidden field classification mapping
        queried_silos: PRE-MUTATION set of already-queried silo names
        classified_fields: PRE-MUTATION set of already-classified field names
        visible_field_ids: PRE-MUTATION set of field IDs currently visible to the agent.
            When provided, classifying unseen fields is treated as invalid.
        step_number: Current step number (1-indexed)
        task_id: Task identifier

    Returns:
        Immediate reward as float.
    """
    reward = 0.0

    if action_type == "query_silo":
        if silo_name not in CASE1_VALID_SILOS:
            reward = -0.05   # Invalid/hallucinated silo name
        elif silo_name in queried_silos:
            reward = -0.05   # Redundant query
        else:
            reward = 0.05    # Good: querying a relevant data source

    elif action_type == "classify_field":
        if field_id is None or field_id not in ground_truth:
            reward = -0.05   # Non-existent field
        elif visible_field_ids is not None and field_id not in visible_field_ids:
            reward = -0.05   # Cannot classify data that has not been revealed yet
        elif field_id in classified_fields:
            reward = -0.05   # Redundant classification
        else:
            true_label = ground_truth[field_id]
            if true_label == "REQUESTER_DATA" and decision == "disclose":
                reward = 0.10      # Correct: customer gets their data
            elif true_label == "INTERNAL_ONLY" and decision == "withhold":
                reward = 0.10      # Correct: internal data protected
            elif true_label == "INTERNAL_ONLY" and decision == "disclose":
                reward = -0.30     # LEAK: internal data exposed (3x penalty)
            elif true_label == "REQUESTER_DATA" and decision == "withhold":
                reward = -0.15     # Wrong: customer's entitlement denied
            else:
                reward = -0.05     # Invalid decision value

    elif action_type == "compile_response":
        reward = 0.0   # Terminal grader handles scoring separately

    # Step cost beyond the free allowance (steps 1-10 are free)
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
    """Compute the final episode score when compile_response is called.

    Uses the FROZEN SPEC formula:
        schema_gate     = binary: 1 if draft_response is valid, else 0
        field_f1        = (2 × P × R) / (P + R)
        privacy_penalty = n × 0.30 × (1 + n × 0.45)   [non-linear]
        compliance_score = clamp(field_f1 − privacy_penalty, 0.0, 1.0)
        step_efficiency = clamp((MAX_STEPS - steps_used) / (MAX_STEPS - MIN_STEPS_CASE1), 0.0, 1.0)
        silo_efficiency = max(0, 1.0 − 0.5 × missing_required_silos − 0.25 × extra_silos)
        efficiency_score = 0.5 × step_efficiency + 0.5 × silo_efficiency
        score = 0.0 if compliance_score == 0 else clamp(0.9 × compliance_score + 0.1 × efficiency_score, 0.0, 1.0)

    The schema gate is the FIRST check — malformed output returns 0.0 immediately.
    This is required because the hackathon grader must handle bad agent output
    without crashing, and must not reward agents for producing junk.

    Args:
        draft_response: Dict of {field_id: value} the agent chose to disclose.
        ground_truth: Hidden mapping of {field_id: 'REQUESTER_DATA' | 'INTERNAL_ONLY'}.
        queried_silos: Set of silo names the agent queried during the episode.
        steps_used: Number of steps the agent consumed.
        task_id: Task identifier.

    Returns:
        Final score clamped to [0.0, 1.0], rounded to 4 decimal places.
        Returns 0.0 immediately if schema gate fails.
    """
    # ── Step 1: HARD SCHEMA GATE (must be first) ──────────────────────────
    # Returns 0.0 immediately on any malformed output — no partial scoring.
    if not isinstance(draft_response, dict):
        return 0.0
    if not all(isinstance(k, str) for k in draft_response.keys()):
        return 0.0

    # ── Step 2: Define field sets ─────────────────────────────────────────
    agent_disclosed = set(draft_response.keys())
    requester_fields = set(REQUESTER_DATA_FIELDS)
    internal_fields = set(INTERNAL_ONLY_FIELDS)

    correctly_disclosed = agent_disclosed & requester_fields
    leaked = agent_disclosed & internal_fields
    leaked_count = len(leaked)

    # ── Step 3: Precision and recall ──────────────────────────────────────
    if len(agent_disclosed) == 0:
        precision = 0.0
    else:
        precision = len(correctly_disclosed) / len(agent_disclosed)

    if len(requester_fields) == 0:
        recall = 0.0
    else:
        recall = len(correctly_disclosed) / len(requester_fields)

    # ── Step 4: F1 score ──────────────────────────────────────────────────
    if precision + recall == 0:
        field_f1 = 0.0
    else:
        field_f1 = 2.0 * precision * recall / (precision + recall)

    # ── Step 5: Non-linear privacy penalty ───────────────────────────────
    # n × 0.30 × (1 + n × 0.45) — quadratic growth so leaks cannot be offset
    # 1 leak  → 0.435  (meaningful)
    # 2 leaks → 1.14   (floors to 0.0)
    # 3 leaks → 2.115  (well beyond floor, but episode terminates before this)
    n = leaked_count
    privacy_penalty = n * 0.30 * (1.0 + n * 0.45)

    # ── Step 6: Efficiency components ─────────────────────────────────────
    # Explicitly score whether the agent queried the required silos and
    # how economically it used the episode budget.
    required_silos = CASE1_VALID_SILOS
    missing_required_silos = len(required_silos - queried_silos)
    extra_silos = len(queried_silos - required_silos)

    # ── Step 7: Assemble final score ──────────────────────────────────────
    # Compliance stays primary; efficiency provides a small secondary signal.
    compliance_score = max(0.0, min(1.0, field_f1 - privacy_penalty))
    step_efficiency = max(
        0.0,
        min(1.0, (MAX_STEPS - steps_used) / (MAX_STEPS - MIN_STEPS_CASE1)),
    )
    silo_efficiency = max(0.0, 1.0 - 0.5 * missing_required_silos - 0.25 * extra_silos)
    efficiency_score = 0.5 * step_efficiency + 0.5 * silo_efficiency

    if compliance_score == 0.0:
        return 0.0

    final_score = max(
        0.0,
        min(1.0, 0.9 * compliance_score + 0.1 * efficiency_score),
    )

    return round(final_score, 4)
