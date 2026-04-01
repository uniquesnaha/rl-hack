"""
Unit tests for DSAR Environment — Case 1.

Tests cover:
  - Generator produces correct field counts and rich FieldItem dicts
  - Grader step rewards match frozen spec
  - Grader terminal scoring with FROZEN FORMULA:
      task1_score = schema_gate × clamp(F1 - privacy_penalty + silo_bonus, 0.0, 1.0)
      privacy_penalty = n × 0.30 × (1 + n × 0.50)  [non-linear]
      silo_bonus = max(0, 0.05 - unnecessary × 0.02)
  - constraint_violated triggers on >2 leaks
  - Score always in [0.0, 1.0]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.generator import generate_case1_episode
from server.grader import compute_step_reward, compute_terminal_score
from server.constants import (
    REQUESTER_DATA_FIELDS,
    INTERNAL_ONLY_FIELDS,
    FIELD_GROUND_TRUTH,
    FIELD_METADATA,
)


class TestGenerator:
    """Tests for the synthetic data generator."""

    def test_field_count(self):
        record, values, gt, text = generate_case1_episode(seed=42)
        assert len(record) == 16, f"Expected 16 FieldItems, got {len(record)}"
        assert len(values) == 16
        assert len(gt) == 16

    def test_requester_fields_present(self):
        record, values, gt, _ = generate_case1_episode(seed=42)
        for field in REQUESTER_DATA_FIELDS:
            assert field in values, f"Missing requester field: {field}"
            assert gt[field] == "REQUESTER_DATA"

    def test_internal_fields_present(self):
        record, values, gt, _ = generate_case1_episode(seed=42)
        for field in INTERNAL_ONLY_FIELDS:
            assert field in values, f"Missing internal field: {field}"
            assert gt[field] == "INTERNAL_ONLY"

    def test_reproducibility(self):
        r1, v1, g1, t1 = generate_case1_episode(seed=123)
        r2, v2, g2, t2 = generate_case1_episode(seed=123)
        assert v1 == v2, "Same seed should produce identical values"
        assert g1 == g2, "Same seed should produce identical ground truth"

    def test_different_seeds_different_data(self):
        _, v1, _, _ = generate_case1_episode(seed=1)
        _, v2, _, _ = generate_case1_episode(seed=2)
        assert v1["full_name"] != v2["full_name"] or v1["email"] != v2["email"]

    def test_dsar_text_contains_name(self):
        _, values, _, text = generate_case1_episode(seed=42)
        assert values["full_name"] in text

    def test_payment_history_is_list(self):
        _, values, _, _ = generate_case1_episode(seed=42)
        assert isinstance(values["payment_history"], list)
        assert len(values["payment_history"]) >= 2

    def test_risk_score_range(self):
        _, values, _, _ = generate_case1_episode(seed=42)
        assert 0.0 <= values["risk_score"] <= 1.0

    def test_field_items_have_rich_metadata(self):
        """Each FieldItem dict must have all 6 required keys."""
        record, _, _, _ = generate_case1_episode(seed=42)
        required_keys = {"field_id", "field_name", "field_value", "source_silo", "datatype", "field_description"}
        for item in record:
            assert isinstance(item, dict)
            missing = required_keys - set(item.keys())
            assert not missing, f"FieldItem {item.get('field_id')} missing keys: {missing}"

    def test_field_items_have_correct_sources(self):
        record, _, _, _ = generate_case1_episode(seed=42)
        valid_sources = {"billing", "crm"}
        for item in record:
            assert item["source_silo"] in valid_sources

    def test_all_fields_have_metadata(self):
        for field_id in FIELD_GROUND_TRUTH:
            assert field_id in FIELD_METADATA, f"Missing FIELD_METADATA for: {field_id}"


class TestStepRewards:
    """Tests for step-level reward computation (frozen spec values)."""

    def test_valid_silo_query(self):
        r = compute_step_reward(
            "query_silo", silo_name="billing",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=1,
        )
        assert r == 0.05

    def test_redundant_silo_query(self):
        r = compute_step_reward(
            "query_silo", silo_name="billing",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos={"billing"}, classified_fields=set(), step_number=2,
        )
        assert r == -0.05

    def test_invalid_silo_query(self):
        r = compute_step_reward(
            "query_silo", silo_name="hr_system",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=1,
        )
        assert r == -0.05

    def test_correct_disclose(self):
        r = compute_step_reward(
            "classify_field", field_id="full_name", decision="disclose",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=1,
        )
        assert r == 0.10

    def test_correct_withhold(self):
        r = compute_step_reward(
            "classify_field", field_id="risk_score", decision="withhold",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=1,
        )
        assert r == 0.10

    def test_leak_penalty(self):
        r = compute_step_reward(
            "classify_field", field_id="churn_probability", decision="disclose",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=1,
        )
        assert r == -0.30

    def test_wrongly_withheld(self):
        r = compute_step_reward(
            "classify_field", field_id="email", decision="withhold",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=1,
        )
        assert r == -0.15

    def test_step_cost_after_free(self):
        r = compute_step_reward(
            "query_silo", silo_name="billing",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=11,
        )
        assert r == 0.04  # 0.05 - 0.01

    def test_step_cost_within_free(self):
        r = compute_step_reward(
            "query_silo", silo_name="billing",
            ground_truth=FIELD_GROUND_TRUTH,
            queried_silos=set(), classified_fields=set(), step_number=10,
        )
        assert r == 0.05


class TestTerminalScoring:
    """Tests for terminal grader — FROZEN SPEC formula:
       task1_score = schema_gate × clamp(F1 - privacy_penalty + silo_bonus, 0.0, 1.0)
       privacy_penalty = n × 0.30 × (1 + n × 0.50)
       silo_bonus = max(0, 0.05 - unnecessary × 0.02)
    """

    def test_perfect_classification(self):
        """All 9 requester fields disclosed, 0 leaks → F1=1.0, penalty=0, bonus=0.05 → 1.0"""
        draft = {f: "value" for f in REQUESTER_DATA_FIELDS}
        score = compute_terminal_score(draft, FIELD_GROUND_TRUTH, {"billing", "crm"}, steps_used=12)
        # score = clamp(1.0 - 0 + 0.05, 0, 1) = 1.0
        assert score == 1.0, f"Perfect classification should score 1.0, got {score}"

    def test_one_leak_penalty(self):
        """1 leak: privacy_penalty = 1 × 0.30 × 1.5 = 0.45 — meaningful, not trivial."""
        draft = {f: "value" for f in REQUESTER_DATA_FIELDS}
        draft["churn_probability"] = 0.5  # 1 internal field leaked
        score = compute_terminal_score(draft, FIELD_GROUND_TRUTH, {"billing", "crm"}, steps_used=14)
        # precision = 9/10 = 0.9, recall = 1.0, F1 = 2*0.9/(1.9) ≈ 0.947
        # penalty = 0.45, bonus = 0.05
        # score = clamp(0.947 - 0.45 + 0.05, 0, 1) = 0.547
        assert 0.50 <= score <= 0.60, f"One leak should score ~0.547, got {score}"

    def test_one_leak_penalty_is_significant(self):
        """One leak must cost ~0.45 — NOT the trivial 0.021 of the G1 formula."""
        draft = {f: "value" for f in REQUESTER_DATA_FIELDS}
        draft["churn_probability"] = 0.5
        score_clean = compute_terminal_score(
            {f: "v" for f in REQUESTER_DATA_FIELDS}, FIELD_GROUND_TRUTH, {"billing", "crm"}
        )
        score_leaked = compute_terminal_score(draft, FIELD_GROUND_TRUTH, {"billing", "crm"})
        penalty_impact = score_clean - score_leaked
        assert penalty_impact >= 0.40, f"One leak should cost at least 0.40 points, cost {penalty_impact:.3f}"

    def test_two_leaks_floors_to_zero(self):
        """2 leaks: penalty = 2×0.30×2.0 = 1.20 — floors score to 0.0."""
        draft = {f: "value" for f in REQUESTER_DATA_FIELDS}
        draft["churn_probability"] = 0.5
        draft["risk_score"] = 0.7
        score = compute_terminal_score(draft, FIELD_GROUND_TRUTH, {"billing", "crm"}, steps_used=16)
        # penalty = 1.20 — way above any F1, so raw < 0 → floored to 0.0
        assert score == 0.0, f"Two leaks should floor to 0.0, got {score}"

    def test_empty_draft(self):
        """Empty draft → F1=0, penalty=0, bonus=0.05 → score=0.05"""
        score = compute_terminal_score({}, FIELD_GROUND_TRUTH, {"billing", "crm"}, steps_used=5)
        assert abs(score - 0.05) < 0.001, f"Empty draft should give 0.05 (silo bonus only), got {score}"

    def test_empty_draft_no_silos(self):
        """Empty draft, no silos queried → bonus = 0.05 (no unnecessary), F1=0 → 0.05"""
        score = compute_terminal_score({}, FIELD_GROUND_TRUTH, set(), steps_used=5)
        # no unnecessary queries, so bonus still 0.05
        assert abs(score - 0.05) < 0.001, f"Expected 0.05, got {score}"

    def test_disclose_everything_floors_to_zero(self):
        """Disclosing all 16 fields: massive leak penalty obliterates F1."""
        all_fields = list(REQUESTER_DATA_FIELDS) + list(INTERNAL_ONLY_FIELDS)
        draft = {f: "value" for f in all_fields}
        score = compute_terminal_score(draft, FIELD_GROUND_TRUTH, {"billing", "crm"}, steps_used=18)
        # penalty = 7×0.30×(1+7×0.5) = 7×0.30×4.5 = 9.45 — catastrophically above any F1
        assert score == 0.0, f"Disclosing everything should floor to 0.0, got {score}"

    def test_unnecessary_silo_reduces_bonus(self):
        """Extra silo queries reduce silo_bonus by 0.02 per unnecessary silo.

        Must use a partial draft (F1 < 1.0) so the difference isn't
        masked by clamping at 1.0.
        """
        # 8 of 9 requester fields → F1 ≈ 0.94 (well below 1.0 so clamping doesn't hide the diff)
        partial_draft = {f: "v" for i, f in enumerate(REQUESTER_DATA_FIELDS) if i < 8}
        score_clean = compute_terminal_score(partial_draft, FIELD_GROUND_TRUTH, {"billing", "crm"})
        score_extra = compute_terminal_score(partial_draft, FIELD_GROUND_TRUTH, {"billing", "crm", "hr"})
        # extra 1 unnecessary → bonus = max(0, 0.05 - 0.02) = 0.03 vs 0.05
        assert score_clean > score_extra, (
            f"Unnecessary silo query should reduce score: clean={score_clean}, extra={score_extra}"
        )
        diff = round(score_clean - score_extra, 4)
        assert abs(diff - 0.02) < 0.001, f"One unnecessary query should cost 0.02, cost {diff}"

    def test_schema_gate_invalid_keys(self):
        """Non-string keys must fail schema gate → 0.0."""
        score = compute_terminal_score({123: "value"}, FIELD_GROUND_TRUTH, set())
        assert score == 0.0

    def test_schema_gate_not_dict(self):
        """Non-dict draft must fail schema gate → 0.0."""
        score = compute_terminal_score("not a dict", FIELD_GROUND_TRUTH, set())
        assert score == 0.0

    def test_schema_gate_none(self):
        """None draft must fail schema gate → 0.0."""
        score = compute_terminal_score(None, FIELD_GROUND_TRUTH, set())
        assert score == 0.0

    def test_score_always_in_range(self):
        """Score must always be in [0.0, 1.0] regardless of inputs."""
        import random
        rng = random.Random(42)
        all_fields = list(REQUESTER_DATA_FIELDS) + list(INTERNAL_ONLY_FIELDS)
        for _ in range(200):
            sample_size = rng.randint(0, len(all_fields))
            sampled = rng.sample(all_fields, sample_size)
            draft = {f: "v" for f in sampled}
            n_silos = rng.randint(0, 4)
            silos = set(rng.sample(["billing", "crm", "hr", "marketing", "analytics"], min(n_silos, 5)))
            steps = rng.randint(1, 30)
            score = compute_terminal_score(draft, FIELD_GROUND_TRUTH, silos, steps_used=steps)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_non_linear_penalty_growth(self):
        """Verify penalty grows non-linearly: 2 leaks cost MORE than 2 × (1 leak cost)."""
        # 1 leak cost
        draft_1 = {f: "v" for f in REQUESTER_DATA_FIELDS}
        draft_1["churn_probability"] = 0.5
        s1 = compute_terminal_score(draft_1, FIELD_GROUND_TRUTH, {"billing", "crm"})

        # 2 leaks cost
        draft_2 = {f: "v" for f in REQUESTER_DATA_FIELDS}
        draft_2["churn_probability"] = 0.5
        draft_2["risk_score"] = 0.7
        s2 = compute_terminal_score(draft_2, FIELD_GROUND_TRUTH, {"billing", "crm"})

        # Perfect score
        s_perfect = 1.0

        drop_1 = s_perfect - s1  # ~0.45
        drop_2 = s_perfect - 0.0  # 1.00 (floored)

        assert drop_2 > 2 * drop_1, (
            f"Non-linear: 2-leak drop ({drop_2:.3f}) should be > 2× 1-leak drop ({2*drop_1:.3f})"
        )

    def test_baseline_target_range(self):
        """Typical baseline LLM: 8/9 correct, 1 leak → ~0.45-0.60 with frozen formula."""
        draft = {}
        for i, f in enumerate(REQUESTER_DATA_FIELDS):
            if i < 8:
                draft[f] = "value"
        draft["churn_probability"] = 0.5  # 1 leak
        score = compute_terminal_score(draft, FIELD_GROUND_TRUTH, {"billing", "crm"}, steps_used=14)
        # precision=8/9=0.889, recall=8/9=0.889, F1=0.889
        # penalty = 0.45, bonus = 0.05
        # score = clamp(0.889 - 0.45 + 0.05, 0, 1) = 0.489
        assert 0.40 <= score <= 0.60, f"Baseline scenario should be ~0.49, got {score}"


if __name__ == "__main__":
    test_classes = [TestGenerator, TestStepRewards, TestTerminalScoring]
    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        methods = sorted([m for m in dir(instance) if m.startswith("test_")])
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: EXCEPTION {type(e).__name__}: {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed}/{total} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(1 if failed > 0 else 0)
