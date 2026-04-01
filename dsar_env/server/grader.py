"""
Deterministic grading logic for DSAR environment.

Two independent scoring functions:
  1. compute_step_reward() — fires on every step() call
  2. compute_terminal_score() — fires when compile_response is called

Terminal formula (frozen spec — Case 1):
    schema_gate = 1 if draft_response is a valid {str: any} dict, else 0
    field_f1    = (2 × precision × recall) / (precision + recall)
    privacy_penalty = n × 0.30 × (1 + n × 0.50)   ← non-linear, intentionally steep
    silo_bonus  = max(0, 0.05 − unnecessary_queries × 0.02)
    task1_score = schema_gate × clamp(field_f1 − privacy_penalty + silo_bonus, 0.0, 1.0)

The non-linear privacy penalty (cubic growth) is critical:
    1 leak  → penalty = 0.45  (one error cannot be offset by three correct)
    2 leaks → penalty = 1.20  (floors to 0.0 — catastrophic leakage)
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


def compute_step_reward(
    action_type: str,
    *,
    silo_name: str | None = None,
    field_id: str | None = None,
    decision: str | None = None,
    ground_truth: Dict[str, str],
    queried_silos: Set[str],
    classified_fields: Set[str],
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
        privacy_penalty = n × 0.30 × (1 + n × 0.50)   [non-linear]
        silo_bonus      = max(0, 0.05 − unnecessary × 0.02)
        score = schema_gate × clamp(field_f1 − privacy_penalty + silo_bonus, 0.0, 1.0)

    The schema gate is the FIRST check — malformed output returns 0.0 immediately.
    This is required because the hackathon grader must handle bad agent output
    without crashing, and must not reward agents for producing junk.

    Args:
        draft_response: Dict of {field_id: value} the agent chose to disclose.
        ground_truth: Hidden mapping of {field_id: 'REQUESTER_DATA' | 'INTERNAL_ONLY'}.
        queried_silos: Set of silo names the agent queried during the episode.
        steps_used: Number of steps the agent consumed (unused in formula, kept for API compat).
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
    # n × 0.30 × (1 + n × 0.50) — cubic growth so leaks cannot be offset
    # 1 leak  → 0.45   (meaningful)
    # 2 leaks → 1.20   (floors to 0.0)
    # 3 leaks → 2.25   (well beyond floor, but episode terminates before this)
    n = leaked_count
    privacy_penalty = n * 0.30 * (1.0 + n * 0.50)

    # ── Step 6: Silo bonus ────────────────────────────────────────────────
    # Small positive reward for querying exactly the right silos.
    # Penalises unnecessary queries (hallucinated or extra silos).
    required_silos = CASE1_VALID_SILOS
    unnecessary = len(queried_silos - required_silos)
    silo_bonus = max(0.0, 0.05 - unnecessary * 0.02)

    # ── Step 7: Assemble final score ──────────────────────────────────────
    # schema_gate=1 (passed above) × clamp(F1 - penalty + bonus, 0.0, 1.0)
    raw_score = field_f1 - privacy_penalty + silo_bonus
    final_score = max(0.0, min(1.0, raw_score))

    return round(final_score, 4)
