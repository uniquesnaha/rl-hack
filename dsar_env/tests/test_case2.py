"""Unit tests for DSAR Environment - Case 2."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.constants import (
    CASE2_PROPORTIONATE_METHODS,
    CASE2_SENTENCE_LABEL_INTERNAL,
    CASE2_SENTENCE_LABEL_PII,
    CASE2_SENTENCE_LABEL_REQUESTER,
    CASE2_VERIFICATION_THRESHOLD,
)
from server.dsar_environment import DSAREnvironment, _EPISODES
from server.generator import generate_case2_episode
from server.grader import (
    compute_step_reward_case2,
    compute_terminal_score_case2,
    compute_terminal_score_case2_details,
)


def _count_ticket_sentences(ticket: dict) -> int:
    return sum(len(message["sentences"]) for message in ticket["messages"])


def _support_requester_sentence_count(ticket: dict, ticket_ground_truth: dict) -> int:
    count = 0
    for message in ticket["messages"]:
        if message["speaker"] != "support":
            continue
        for sentence in message["sentences"]:
            if ticket_ground_truth[sentence["sentence_index"]] == CASE2_SENTENCE_LABEL_REQUESTER:
                count += 1
    return count


def _label_counts(ticket: dict, ticket_ground_truth: dict) -> dict:
    counts = {
        CASE2_SENTENCE_LABEL_REQUESTER: 0,
        CASE2_SENTENCE_LABEL_PII: 0,
        CASE2_SENTENCE_LABEL_INTERNAL: 0,
    }
    for message in ticket["messages"]:
        for sentence in message["sentences"]:
            counts[ticket_ground_truth[sentence["sentence_index"]]] += 1
    return counts


LEGACY_REDACT_PATTERNS = [
    "@",
    "contact ",
    "reach ",
    "reach me",
    "call me",
    "call ",
    "phone",
    "email",
    "escalat",
    "reference ",
    "flag ",
    "internal ",
    "campaign ",
    "code ",
]

EXPECTED_TICKET_LABEL_COUNTS = {
    "technical_support": {
        CASE2_SENTENCE_LABEL_REQUESTER: 4,
        CASE2_SENTENCE_LABEL_PII: 1,
        CASE2_SENTENCE_LABEL_INTERNAL: 1,
    },
    "billing_dispute": {
        CASE2_SENTENCE_LABEL_REQUESTER: 4,
        CASE2_SENTENCE_LABEL_PII: 1,
        CASE2_SENTENCE_LABEL_INTERNAL: 1,
    },
    "cancellation_request": {
        CASE2_SENTENCE_LABEL_REQUESTER: 4,
        CASE2_SENTENCE_LABEL_PII: 2,
        CASE2_SENTENCE_LABEL_INTERNAL: 1,
    },
}


def _matches_legacy_redact_pattern(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in LEGACY_REDACT_PATTERNS) or any(char.isdigit() for char in text)


def _normalize_name(text: str) -> str:
    return " ".join(
        part
        for part in text.lower().replace(".", "").replace("-", " ").replace("'", "").split()
        if len(part) > 1
    )


def _normalize_local_part(text: str) -> set[str]:
    cleaned = text.lower().replace(".", " ").replace("-", " ").replace("_", " ")
    return {part for part in cleaned.split() if part}


def _heuristic_support_redact_policy(bundle: dict) -> dict:
    """Approximate a baseline that keeps customer sentences and redacts support sentences."""
    processed = {}
    for ticket in bundle["tickets"]:
        ticket_id = ticket["ticket_id"]
        processed[ticket_id] = {}
        for message in ticket["messages"]:
            for sentence in message["sentences"]:
                decision = "keep" if sentence["speaker"] == "customer" else "redact"
                processed[ticket_id][sentence["sentence_index"]] = decision
    return processed


def _baseline_proxy_case2_policy(bundle: dict) -> tuple[dict, int]:
    """
    Approximate a medium-task baseline:
    - one wrong-but-proportionate verification before success
    - keep customer sentences
    - redact support sentences
    - leak exactly one support PII sentence
    """
    processed = _heuristic_support_redact_policy(bundle)
    leak_target = None
    for ticket in bundle["tickets"]:
        ticket_id = ticket["ticket_id"]
        for message in ticket["messages"]:
            for sentence in message["sentences"]:
                label = bundle["ticket_ground_truth"][ticket_id][sentence["sentence_index"]]
                if label == CASE2_SENTENCE_LABEL_PII:
                    leak_target = (ticket_id, sentence["sentence_index"])
                    break
            if leak_target:
                break
        if leak_target:
            break

    if leak_target:
        processed[leak_target[0]][leak_target[1]] = "keep"

    return processed, 2


class TestCase2Generator:
    def test_case2_reproducibility(self):
        assert generate_case2_episode(seed=42) == generate_case2_episode(seed=42)

    def test_case2_confidence_range(self):
        bundle = generate_case2_episode(seed=42)
        assert 0.35 <= bundle["starting_identity_confidence"] <= 0.55
        assert bundle["verification_threshold"] == CASE2_VERIFICATION_THRESHOLD

    def test_case2_generates_three_tickets(self):
        bundle = generate_case2_episode(seed=42)
        assert len(bundle["tickets"]) == 3
        assert len(bundle["ticket_ground_truth"]) == 3

    def test_case2_ticket_labels_valid(self):
        bundle = generate_case2_episode(seed=42)
        valid = {
            CASE2_SENTENCE_LABEL_REQUESTER,
            CASE2_SENTENCE_LABEL_PII,
            CASE2_SENTENCE_LABEL_INTERNAL,
        }
        for labels in bundle["ticket_ground_truth"].values():
            assert labels
            assert set(labels.values()).issubset(valid)

    def test_each_ticket_has_2_to_4_turns_and_6_to_8_sentences(self):
        bundle = generate_case2_episode(seed=42)
        for ticket in bundle["tickets"]:
            assert 2 <= len(ticket["messages"]) <= 4
            assert 6 <= _count_ticket_sentences(ticket) <= 8

    def test_each_ticket_has_support_origin_requester_sentence(self):
        bundle = generate_case2_episode(seed=42)
        for ticket in bundle["tickets"]:
            truth = bundle["ticket_ground_truth"][ticket["ticket_id"]]
            assert _support_requester_sentence_count(ticket, truth) >= 1

    def test_each_ticket_has_exact_label_composition_by_category(self):
        bundle = generate_case2_episode(seed=42)
        for ticket in bundle["tickets"]:
            truth = bundle["ticket_ground_truth"][ticket["ticket_id"]]
            counts = _label_counts(ticket, truth)
            assert counts == EXPECTED_TICKET_LABEL_COUNTS[ticket["category"]]

    def test_each_ticket_has_internal_note(self):
        bundle = generate_case2_episode(seed=42)
        for ticket_id, labels in bundle["ticket_ground_truth"].items():
            assert CASE2_SENTENCE_LABEL_INTERNAL in set(labels.values())

    def test_at_least_two_tickets_have_pii(self):
        bundle = generate_case2_episode(seed=42)
        tickets_with_pii = 0
        for labels in bundle["ticket_ground_truth"].values():
            if CASE2_SENTENCE_LABEL_PII in set(labels.values()):
                tickets_with_pii += 1
        assert tickets_with_pii >= 2

    def test_at_least_one_ticket_has_two_pii_sentences(self):
        bundle = generate_case2_episode(seed=42)
        max_pii = 0
        for ticket in bundle["tickets"]:
            truth = bundle["ticket_ground_truth"][ticket["ticket_id"]]
            counts = _label_counts(ticket, truth)
            max_pii = max(max_pii, counts[CASE2_SENTENCE_LABEL_PII])
        assert max_pii >= 2

    def test_identity_ambiguity_includes_competing_proportionate_methods(self):
        for seed in range(1, 11):
            bundle = generate_case2_episode(seed=seed)
            ambiguity = bundle["identity_ambiguity"]
            plausible_methods = ambiguity["plausible_methods"]
            assert len(plausible_methods) == 2
            assert len(set(plausible_methods)) == 2
            assert bundle["correct_verification_method"] in plausible_methods
            assert bundle["competing_verification_method"] in plausible_methods
            assert set(plausible_methods).issubset(CASE2_PROPORTIONATE_METHODS)

            billing = bundle["internal_identity_masked"]["billing"]
            crm = bundle["internal_identity_masked"]["crm"]
            assert "billing_review_note" in billing
            assert "billing_event_summary" in billing
            assert "workspace_location_note" in crm
            assert "workspace_ownership_note" in crm

    def test_each_ticket_contains_context_dependent_support_pair(self):
        for seed in range(1, 11):
            bundle = generate_case2_episode(seed=seed)
            for ticket in bundle["tickets"]:
                assert any(
                    meta["speaker"] == "support"
                    and meta["kind"] == "support_requester_context_pair"
                    and meta["size"] == 2
                    for meta in ticket["chunk_metadata"]
                )

    def test_each_ticket_has_support_requester_sentence_that_old_heuristic_would_redact(self):
        for seed in range(1, 11):
            bundle = generate_case2_episode(seed=seed)
            for ticket in bundle["tickets"]:
                truth = bundle["ticket_ground_truth"][ticket["ticket_id"]]
                support_requester_sentences = [
                    sentence["text"]
                    for message in ticket["messages"]
                    if message["speaker"] == "support"
                    for sentence in message["sentences"]
                    if truth[sentence["sentence_index"]] == CASE2_SENTENCE_LABEL_REQUESTER
                ]
                assert any(_matches_legacy_redact_pattern(text) for text in support_requester_sentences)

    def test_each_ticket_has_unsafe_sentence_without_legacy_pattern(self):
        for seed in range(1, 11):
            bundle = generate_case2_episode(seed=seed)
            for ticket in bundle["tickets"]:
                truth = bundle["ticket_ground_truth"][ticket["ticket_id"]]
                unsafe_sentences = [
                    sentence["text"]
                    for message in ticket["messages"]
                    for sentence in message["sentences"]
                    if truth[sentence["sentence_index"]] in {
                        CASE2_SENTENCE_LABEL_PII,
                        CASE2_SENTENCE_LABEL_INTERNAL,
                    }
                ]
                assert any(not _matches_legacy_redact_pattern(text) for text in unsafe_sentences)


class TestCase2StepRewards:
    def test_wrong_but_proportionate_verification_partial_credit(self):
        reward = compute_step_reward_case2(
            action_type="verify_identity",
            phase="identity",
            step_number=1,
            queried_silos=set(),
            verification_method="registered_postcode",
            correct_verification_method="transaction_date",
            verification_attempts=set(),
        )
        assert reward == 0.10

    def test_disproportionate_verification_penalty(self):
        reward = compute_step_reward_case2(
            action_type="verify_identity",
            phase="identity",
            step_number=1,
            queried_silos=set(),
            verification_method="passport_copy",
            correct_verification_method="transaction_date",
            verification_attempts=set(),
        )
        assert reward == -0.20

    def test_sentence_keep_requester_reward(self):
        reward = compute_step_reward_case2(
            action_type="redact_span",
            phase="redaction",
            step_number=1,
            queried_silos={"billing"},
            ticket_id="T1",
            sentence_index=0,
            decision="keep",
            ticket_ground_truth={"T1": {0: CASE2_SENTENCE_LABEL_REQUESTER}},
            processed_sentences={},
            identity_verified=True,
        )
        assert reward == 0.10

    def test_sentence_keep_pii_leak_penalty(self):
        reward = compute_step_reward_case2(
            action_type="redact_span",
            phase="redaction",
            step_number=1,
            queried_silos={"billing"},
            ticket_id="T1",
            sentence_index=0,
            decision="keep",
            ticket_ground_truth={"T1": {0: CASE2_SENTENCE_LABEL_PII}},
            processed_sentences={},
            identity_verified=True,
        )
        assert reward == -0.30

    def test_blocked_compile_first_penalty(self):
        reward = compute_step_reward_case2(
            action_type="compile_response",
            phase="redaction",
            step_number=1,
            queried_silos={"billing", "crm"},
            identity_verified=True,
            all_sentences_processed=False,
            blocked_compile_attempts=1,
        )
        assert reward == -0.05

    def test_blocked_compile_second_penalty(self):
        reward = compute_step_reward_case2(
            action_type="compile_response",
            phase="redaction",
            step_number=1,
            queried_silos={"billing", "crm"},
            identity_verified=True,
            all_sentences_processed=False,
            blocked_compile_attempts=2,
        )
        assert reward == -0.08

    def test_blocked_compile_third_penalty(self):
        reward = compute_step_reward_case2(
            action_type="compile_response",
            phase="redaction",
            step_number=1,
            queried_silos={"billing", "crm"},
            identity_verified=True,
            all_sentences_processed=False,
            blocked_compile_attempts=3,
        )
        assert reward == -0.12


class TestCase2TerminalScoring:
    def test_perfect_case2_terminal_score(self):
        processed = {"T1": {0: "keep", 1: "redact", 2: "redact"}}
        truth = {
            "T1": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
                2: CASE2_SENTENCE_LABEL_INTERNAL,
            }
        }
        score = compute_terminal_score_case2(
            processed,
            truth,
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert 0.94 <= score <= 0.96

    def test_case2_terminal_score_pii_leak_is_much_lower(self):
        processed = {"T1": {0: "keep", 1: "keep", 2: "redact"}}
        truth = {
            "T1": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
                2: CASE2_SENTENCE_LABEL_INTERNAL,
            }
        }
        score = compute_terminal_score_case2(
            processed,
            truth,
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert score < 0.50

    def test_case2_terminal_schema_gate(self):
        score = compute_terminal_score_case2(
            {"T404": {0: "keep"}},
            {"T1": {0: CASE2_SENTENCE_LABEL_REQUESTER}},
            phase1_reward_sum=0.25,
        )
        assert score == 0.0

    def test_terminal_details_include_debug_submetrics(self):
        processed = {"T1": {0: "keep", 1: "redact", 2: "redact"}}
        truth = {
            "T1": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
                2: CASE2_SENTENCE_LABEL_INTERNAL,
            }
        }
        details = compute_terminal_score_case2_details(processed, truth, phase1_reward_sum=0.35)
        assert details["schema_gate"] == 1.0
        assert "identity_score" in details
        assert "redaction_f1" in details
        assert "kept_precision" in details
        assert "kept_recall" in details
        assert "leakage_rate" in details
        assert details["task2_score"] == compute_terminal_score_case2(processed, truth, 0.35)

    def test_incomplete_redaction_returns_partial_score(self):
        processed = {"T1": {0: "keep"}}
        truth = {
            "T1": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
            }
        }
        score = compute_terminal_score_case2(
            processed_sentences=processed,
            ticket_ground_truth=truth,
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=False,
            termination_reason="incomplete_redaction_timeout",
        )
        assert 0.44 <= score <= 0.45

    def test_identity_score_not_inflated_by_multiple_proportionate_attempts(self):
        processed = {"T1": {0: "keep", 1: "redact", 2: "redact"}}
        truth = {
            "T1": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
                2: CASE2_SENTENCE_LABEL_INTERNAL,
            }
        }
        first_try = compute_terminal_score_case2(
            processed,
            truth,
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        second_try = compute_terminal_score_case2(
            processed,
            truth,
            phase1_reward_sum=0.45,
            verification_succeeded=True,
            proportionate_attempt_count=2,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        third_try = compute_terminal_score_case2(
            processed,
            truth,
            phase1_reward_sum=0.55,
            verification_succeeded=True,
            proportionate_attempt_count=3,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert second_try < first_try
        assert third_try < second_try

    def test_incomplete_details_expose_partial_progress(self):
        processed = {"T1": {0: "keep"}}
        truth = {
            "T1": {
                0: CASE2_SENTENCE_LABEL_REQUESTER,
                1: CASE2_SENTENCE_LABEL_PII,
            }
        }
        details = compute_terminal_score_case2_details(
            processed_sentences=processed,
            ticket_ground_truth=truth,
            phase1_reward_sum=0.45,
            verification_succeeded=True,
            proportionate_attempt_count=2,
            disproportionate_used=False,
            completed_all_sentences=False,
            termination_reason="incomplete_redaction_timeout",
        )
        assert 0.0 < details["completion_coverage"] < 1.0
        assert details["termination_reason"] == "incomplete_redaction_timeout"
        assert details["task2_score"] > 0.0

    def test_all_redact_first_try_identity_is_mediocre(self):
        bundle = generate_case2_episode(seed=1)
        processed = {
            ticket["ticket_id"]: {
                sentence["sentence_index"]: "redact"
                for message in ticket["messages"]
                for sentence in message["sentences"]
            }
            for ticket in bundle["tickets"]
        }
        score = compute_terminal_score_case2(
            processed_sentences=processed,
            ticket_ground_truth=bundle["ticket_ground_truth"],
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert 0.30 <= score <= 0.35

    def test_all_redact_second_try_identity_is_worse(self):
        bundle = generate_case2_episode(seed=1)
        processed = {
            ticket["ticket_id"]: {
                sentence["sentence_index"]: "redact"
                for message in ticket["messages"]
                for sentence in message["sentences"]
            }
            for ticket in bundle["tickets"]
        }
        score = compute_terminal_score_case2(
            processed_sentences=processed,
            ticket_ground_truth=bundle["ticket_ground_truth"],
            phase1_reward_sum=0.45,
            verification_succeeded=True,
            proportionate_attempt_count=2,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert 0.20 <= score <= 0.27

    def test_all_redact_third_try_identity_is_low(self):
        bundle = generate_case2_episode(seed=1)
        processed = {
            ticket["ticket_id"]: {
                sentence["sentence_index"]: "redact"
                for message in ticket["messages"]
                for sentence in message["sentences"]
            }
            for ticket in bundle["tickets"]
        }
        score = compute_terminal_score_case2(
            processed_sentences=processed,
            ticket_ground_truth=bundle["ticket_ground_truth"],
            phase1_reward_sum=0.55,
            verification_succeeded=True,
            proportionate_attempt_count=3,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert score < 0.22

    def test_semantically_correct_policy_beats_all_redact_by_margin(self):
        bundle = generate_case2_episode(seed=1)
        perfect = {}
        all_redact = {}
        for ticket in bundle["tickets"]:
            ticket_id = ticket["ticket_id"]
            perfect[ticket_id] = {}
            all_redact[ticket_id] = {}
            for message in ticket["messages"]:
                for sentence in message["sentences"]:
                    sentence_index = sentence["sentence_index"]
                    label = bundle["ticket_ground_truth"][ticket_id][sentence_index]
                    perfect[ticket_id][sentence_index] = "keep" if label == CASE2_SENTENCE_LABEL_REQUESTER else "redact"
                    all_redact[ticket_id][sentence_index] = "redact"

        perfect_score = compute_terminal_score_case2(
            processed_sentences=perfect,
            ticket_ground_truth=bundle["ticket_ground_truth"],
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        exploit_score = compute_terminal_score_case2(
            processed_sentences=all_redact,
            ticket_ground_truth=bundle["ticket_ground_truth"],
            phase1_reward_sum=0.35,
            verification_succeeded=True,
            proportionate_attempt_count=1,
            disproportionate_used=False,
            completed_all_sentences=True,
        )
        assert perfect_score - exploit_score >= 0.20

    def test_case2_baseline_proxy_mean_score_in_medium_band(self):
        scores = []
        for seed in range(1, 11):
            bundle = generate_case2_episode(seed=seed)
            processed, attempt_count = _baseline_proxy_case2_policy(bundle)
            score = compute_terminal_score_case2(
                processed_sentences=processed,
                ticket_ground_truth=bundle["ticket_ground_truth"],
                phase1_reward_sum=0.45,
                verification_succeeded=True,
                proportionate_attempt_count=attempt_count,
                disproportionate_used=False,
                completed_all_sentences=True,
            )
            scores.append(score)
        mean_score = sum(scores) / len(scores)
        assert 0.40 <= mean_score <= 0.55, f"Expected mean medium-band score, got {mean_score:.4f}"


class TestCase2EnvironmentFlow:
    def test_reset_task_medium_starts_in_identity_phase(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        assert obs.phase == "identity"
        assert obs.identity_verified is False
        assert obs.tickets == []
        assert obs.customer_record == []

    def test_query_billing_reveals_masked_identity_only(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id

        result = env.step(
            {
                "action_type": "query_silo",
                "silo_name": "billing",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.phase == "identity"
        assert result.customer_record == []
        assert result.tickets == []
        assert "billing_address" in result.internal_identity

    def test_correct_verification_unlocks_redaction(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        correct = _EPISODES[ep_id].correct_verification_method

        result = env.step(
            {
                "action_type": "verify_identity",
                "verification_method": correct,
                "metadata": {"episode_id": ep_id},
            }
        )
        expected = 0.20 if correct == "registered_postcode" else 0.25
        assert result.phase == "redaction"
        assert result.identity_verified is True
        assert len(result.tickets) == 3
        assert result.reward == expected

    def test_wrong_proportionate_verification_stays_in_identity(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        correct = _EPISODES[ep_id].correct_verification_method
        wrong = next(method for method in {"transaction_date", "account_reference", "registered_postcode"} if method != correct)

        result = env.step(
            {
                "action_type": "verify_identity",
                "verification_method": wrong,
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.phase == "identity"
        assert result.reward == 0.10

    def test_redact_span_in_identity_phase_penalized(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        result = env.step(
            {
                "action_type": "redact_span",
                "ticket_id": "TKT-1",
                "sentence_index": 0,
                "decision": "redact",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.reward == -0.05
        assert result.error is not None

    def test_compile_before_verification_hard_penalty(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        result = env.step(
            {
                "action_type": "compile_response",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is True
        assert result.reward == -0.50
        assert result.metadata["terminal_score"] == 0.0

    def test_compile_blocked_until_all_sentences_processed(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        correct = _EPISODES[ep_id].correct_verification_method
        obs = env.step(
            {
                "action_type": "verify_identity",
                "verification_method": correct,
                "metadata": {"episode_id": ep_id},
            }
        )
        result = env.step(
            {
                "action_type": "compile_response",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is False
        assert result.reward == -0.05
        assert result.error is not None

    def test_terminal_metadata_exposes_case2_metrics(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        correct = _EPISODES[ep_id].correct_verification_method
        obs = env.step(
            {
                "action_type": "verify_identity",
                "verification_method": correct,
                "metadata": {"episode_id": ep_id},
            }
        )
        for ticket in obs.tickets:
            for message in ticket.messages:
                for sentence in message.sentences:
                    label = _EPISODES[ep_id].ticket_ground_truth[ticket.ticket_id][sentence.sentence_index]
                    decision = "keep" if label == CASE2_SENTENCE_LABEL_REQUESTER else "redact"
                    obs = env.step(
                        {
                            "action_type": "redact_span",
                            "ticket_id": ticket.ticket_id,
                            "sentence_index": sentence.sentence_index,
                            "decision": decision,
                            "metadata": {"episode_id": ep_id},
                        }
                    )

        final_obs = env.step(
            {
                "action_type": "compile_response",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert final_obs.done is True
        for key in [
            "identity_score",
            "redaction_f1",
            "kept_precision",
            "kept_recall",
            "leakage_rate",
            "redaction_score",
            "leaked_pii_count",
            "total_pii_sentences",
            "completion_coverage",
            "termination_reason",
        ]:
            assert key in final_obs.metadata

    def test_timeout_with_incomplete_redaction_scores_partial_and_honest(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_medium")
        ep_id = obs.episode_id
        correct = _EPISODES[ep_id].correct_verification_method
        env.step(
            {
                "action_type": "verify_identity",
                "verification_method": correct,
                "metadata": {"episode_id": ep_id},
            }
        )

        episode = _EPISODES[ep_id]
        episode.step_count = 29
        env._state.step_count = 29
        result = env.step(
            {
                "action_type": "compile_response",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is True
        assert result.metadata["terminal_score"] == 0.0
        assert result.metadata["termination_reason"] == "incomplete_redaction_timeout"
        assert result.metadata["completion_coverage"] == 0.0
