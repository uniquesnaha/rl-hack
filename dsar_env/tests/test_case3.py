"""Unit tests for DSAR Environment - Case 3."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    _extract_terminal_metrics,
    _action_params_allowed,
    _action_parameter_validation_message,
    _action_type_allowed,
    _action_validation_message,
    _available_actions,
    choose_task_hard_action,
    format_observation,
)
from server.constants import (
    CASE2_SENTENCE_LABEL_PII,
    CASE2_SENTENCE_LABEL_REQUESTER,
    CASE3_ACTION_DISCLOSE,
    CASE3_ACTION_ESCALATE,
    CASE3_ACTION_EXCLUDE,
    CASE3_ACTION_PARTIAL_REDACT,
    CASE3_ESCALATION_KEYWORD_BONUS,
    CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
    PROJECT_NAMES,
)
from server.dsar_environment import DSAREnvironment, _EPISODES
from server.generator import generate_case3_episode
from server.grader import (
    _compute_c1_case3,
    _compute_c2_case3,
    _compute_c3_case3,
    _count_pii_breaches_case3,
    _schema_gate_case3,
    compute_terminal_score_case3,
)


def _perfect_case3_output(bundle: dict) -> dict:
    processed_messages = {}
    escalation_log = {}
    escalation_reason_codes = {}
    for msg_id, gt in bundle["ground_truth"].items():
        entry = {"action": gt["action"]}
        if gt["action"] == CASE3_ACTION_PARTIAL_REDACT:
            entry["sentence_decisions"] = {0: "keep", 1: "redact"}
        processed_messages[msg_id] = entry
        if gt["action"] == CASE3_ACTION_ESCALATE:
            escalation_log[msg_id] = (
                "Escalate because this includes anxiety health data and article 9 "
                "special category information."
            )
            escalation_reason_codes[msg_id] = CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA
    return {
        "processed_messages": processed_messages,
        "escalation_log": escalation_log,
        "escalation_reason_codes": escalation_reason_codes,
    }


def _base_case3_observation(bundle: dict) -> dict:
    return {
        "task_id": "task_hard",
        "phase": "triage",
        "steps_remaining": 40,
        "dsar_request": bundle["dsar_text"],
        "slack_export": bundle["messages"],
        "users_json": bundle["users_json"],
        "processed_messages": {},
        "escalation_log": {},
        "escalation_reason_codes": {},
        "messages_pending": sorted(bundle["ground_truth"].keys()),
        "sentences_pending": {},
        "compile_ready": False,
        "classified_fields": [],
        "silo_results": [],
    }


class TestCase3Generator:
    def test_case3_reproducibility(self):
        assert generate_case3_episode(seed=42) == generate_case3_episode(seed=42)

    def test_case3_generates_exactly_six_messages(self):
        bundle = generate_case3_episode(seed=42)
        assert len(bundle["messages"]) == 6
        assert len(bundle["ground_truth"]) == 6
        assert len(bundle["special_category_message_ids"]) == 1

    def test_case3_optional_distractors_are_off_by_default(self):
        previous = os.environ.pop("DSAR_CASE3_DISTRACTORS", None)
        try:
            bundle = generate_case3_episode(seed=42)
            assert len(bundle["messages"]) == 6
        finally:
            if previous is not None:
                os.environ["DSAR_CASE3_DISTRACTORS"] = previous

    def test_case3_optional_distractors_expand_message_set(self):
        previous = os.environ.get("DSAR_CASE3_DISTRACTORS")
        os.environ["DSAR_CASE3_DISTRACTORS"] = "2"
        try:
            bundle = generate_case3_episode(seed=42)
            assert len(bundle["messages"]) == 8
            assert len(bundle["ground_truth"]) == 8
        finally:
            if previous is None:
                os.environ.pop("DSAR_CASE3_DISTRACTORS", None)
            else:
                os.environ["DSAR_CASE3_DISTRACTORS"] = previous

    def test_case3_user_ids_vary_across_seeds(self):
        user_id_sets = {
            tuple(sorted(generate_case3_episode(seed=seed)["users_json"].keys()))
            for seed in range(1, 6)
        }
        assert len(user_id_sets) > 1

    def test_case3_users_json_does_not_expose_role(self):
        bundle = generate_case3_episode(seed=42)
        for user in bundle["users_json"].values():
            assert "role" not in user
            assert "real_name" in user
            assert "display_name" in user

    def test_case3_messages_expose_stable_sentence_arrays(self):
        bundle = generate_case3_episode(seed=42)
        for message in bundle["messages"]:
            sentence_indices = [sentence["sentence_idx"] for sentence in message["sentences"]]
            assert sentence_indices == list(range(len(message["sentences"])))
            assert all(sentence["text"] for sentence in message["sentences"])

    def test_case3_mixed_message_has_exactly_two_sentences(self):
        bundle = generate_case3_episode(seed=42)
        mixed_id = bundle["mixed_sentence_message_id"]
        mixed_message = next(message for message in bundle["messages"] if message["msg_id"] == mixed_id)
        assert len(mixed_message["sentences"]) == 2

    def test_case3_irregular_timestamps_and_independent_msg_ids(self):
        bundle = generate_case3_episode(seed=42)
        timestamps = sorted(int(message["ts"]) for message in bundle["messages"])
        gaps = [timestamps[idx + 1] - timestamps[idx] for idx in range(len(timestamps) - 1)]
        assert len(set(gaps)) > 1
        for message in bundle["messages"]:
            assert message["ts"] not in message["msg_id"]

    def test_case3_thread_reply_points_to_parent_timestamp(self):
        bundle = generate_case3_episode(seed=42)
        parent = next(message for message in bundle["messages"] if message["msg_id"] == bundle["thread_parent_id"])
        reply = next(message for message in bundle["messages"] if message["msg_id"] == bundle["thread_reply_id"])
        requester_first = bundle["requester_name"].split()[0].lower()
        assert reply["thread_ts"] == parent["ts"]
        assert reply["text"] in {
            "Looks clean, merging to main.",
            "Approved. Deploying now.",
            "Good to go from my side.",
            "Reviewed it. Shipping this one.",
        }
        assert requester_first not in reply["text"].lower()
        assert all(project.lower() not in reply["text"].lower() for project in PROJECT_NAMES)


class TestCase3Grader:
    def test_schema_gate_rejects_missing_keys(self):
        assert _schema_gate_case3({"processed_messages": {}}) is False

    def test_perfect_output_normalizes_all_components_to_one(self):
        bundle = generate_case3_episode(seed=42)
        output = _perfect_case3_output(bundle)
        processed = output["processed_messages"]
        escalation_log = output["escalation_log"]
        escalation_reason_codes = output["escalation_reason_codes"]

        assert _compute_c1_case3(processed, bundle["ground_truth"]) == 1.0
        assert _compute_c2_case3(processed, bundle["mixed_sentence_message_id"]) == 1.0
        assert (
            _compute_c3_case3(
                processed,
                escalation_log,
                escalation_reason_codes,
                bundle["ground_truth"],
                bundle["special_category_message_ids"][0],
            )
            == 1.0
        )

        details = compute_terminal_score_case3(
            output,
            bundle["ground_truth"],
            bundle["special_category_message_ids"],
            bundle["mixed_sentence_message_id"],
        )
        assert details["task3_score"] == 1.0

    def test_breach_counting_only_counts_real_disclosure_harm(self):
        bundle = generate_case3_episode(seed=42)
        mixed_id = bundle["mixed_sentence_message_id"]
        special_id = bundle["special_category_message_ids"][0]
        processed = {
            special_id: {"action": CASE3_ACTION_DISCLOSE},
            mixed_id: {"action": CASE3_ACTION_PARTIAL_REDACT, "sentence_decisions": {0: "keep", 1: "keep"}},
        }
        assert _count_pii_breaches_case3(processed, bundle["ground_truth"]) == 2

    def test_keyword_bonus_requires_health_and_legal_terms(self):
        bundle = generate_case3_episode(seed=42)
        special_id = bundle["special_category_message_ids"][0]
        processed = {special_id: {"action": CASE3_ACTION_ESCALATE}}
        weak_reason = {special_id: "Escalate because of anxiety in this message."}
        reason_codes = {special_id: CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA}
        strong_reason = {
            special_id: "Escalate because of anxiety health data and article 9 special category content."
        }
        assert (
            _compute_c3_case3(processed, weak_reason, reason_codes, bundle["ground_truth"], special_id)
            < _compute_c3_case3(processed, strong_reason, reason_codes, bundle["ground_truth"], special_id)
        )

    def test_lazy_escalation_penalty_reduces_c3(self):
        bundle = generate_case3_episode(seed=42)
        perfect = _perfect_case3_output(bundle)
        extra_id = next(
            msg_id
            for msg_id, gt in bundle["ground_truth"].items()
            if gt["action"] != CASE3_ACTION_ESCALATE
        )
        imperfect = _perfect_case3_output(bundle)
        imperfect["processed_messages"][extra_id] = {"action": CASE3_ACTION_ESCALATE}
        assert (
            _compute_c3_case3(
                imperfect["processed_messages"],
                imperfect["escalation_log"],
                imperfect["escalation_reason_codes"],
                bundle["ground_truth"],
                bundle["special_category_message_ids"][0],
            )
            < _compute_c3_case3(
                perfect["processed_messages"],
                perfect["escalation_log"],
                perfect["escalation_reason_codes"],
                bundle["ground_truth"],
                bundle["special_category_message_ids"][0],
            )
        )

    def test_reason_code_drives_c3_more_than_keyword_bonus(self):
        bundle = generate_case3_episode(seed=42)
        special_id = bundle["special_category_message_ids"][0]
        processed = {special_id: {"action": CASE3_ACTION_ESCALATE}}
        strong_reason = {
            special_id: "Escalate because of anxiety health data and article 9 special category content."
        }
        wrong_reason_code = {special_id: "requires_human_balancing"}
        right_reason_code = {special_id: CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA}
        assert (
            _compute_c3_case3(processed, strong_reason, right_reason_code, bundle["ground_truth"], special_id)
            > _compute_c3_case3(processed, strong_reason, wrong_reason_code, bundle["ground_truth"], special_id)
        )

    def test_keyword_bonus_stays_small(self):
        assert CASE3_ESCALATION_KEYWORD_BONUS == 0.05

    def test_special_category_disclosure_forces_zero_terminal_score(self):
        bundle = generate_case3_episode(seed=42)
        output = _perfect_case3_output(bundle)
        special_id = bundle["special_category_message_ids"][0]
        output["processed_messages"][special_id] = {"action": CASE3_ACTION_DISCLOSE}
        details = compute_terminal_score_case3(
            output,
            bundle["ground_truth"],
            bundle["special_category_message_ids"],
            bundle["mixed_sentence_message_id"],
        )
        assert details["task3_score"] == 0.0
        assert details["termination_reason"] == "special_category_disclosed"


class TestCase3EnvironmentFlow:
    def test_reset_task_hard_returns_case3_observation_shape(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        assert obs.phase == "triage"
        assert len(obs.slack_export) == 6
        assert obs.users_json
        assert obs.available_actions == ["process_message"]

    def test_special_category_disclosure_terminates_immediately(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        special_id = _EPISODES[ep_id].special_category_message_ids[0]
        result = env.step(
            {
                "action_type": "process_message",
                "msg_id": special_id,
                "action_label": CASE3_ACTION_DISCLOSE,
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is True
        assert result.metadata["terminal_score"] == 0.0
        assert result.metadata["termination_reason"] == "special_category_disclosed"

    def test_redact_sentence_invalid_before_partial_redact(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        mixed_id = _EPISODES[ep_id].mixed_sentence_message_id
        result = env.step(
            {
                "action_type": "redact_sentence",
                "msg_id": mixed_id,
                "sentence_index": 0,
                "decision": "keep",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is False
        assert result.error is not None

    def test_redact_sentence_rejected_on_non_sentence_actionable_message(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        episode = _EPISODES[ep_id]
        non_mixed_id = next(
            msg_id
            for msg_id, gt in episode.ground_truth.items()
            if gt["action"] == CASE3_ACTION_DISCLOSE
        )
        env.step(
            {
                "action_type": "process_message",
                "msg_id": non_mixed_id,
                "action_label": CASE3_ACTION_PARTIAL_REDACT,
                "metadata": {"episode_id": ep_id},
            }
        )
        result = env.step(
            {
                "action_type": "redact_sentence",
                "msg_id": non_mixed_id,
                "sentence_index": 0,
                "decision": "keep",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is False
        assert "does not support sentence-level redaction" in str(result.error)

    def test_escalate_with_reason_invalid_before_escalate(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        special_id = _EPISODES[ep_id].special_category_message_ids[0]
        result = env.step(
            {
                "action_type": "escalate_with_reason",
                "msg_id": special_id,
                "reason_code": CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
                "reason": "Contains anxiety health data and article 9 special category information.",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert result.done is False
        assert result.error is not None

    def test_compile_blocked_when_decisions_missing(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        episode = _EPISODES[ep_id]

        for msg_id, gt in episode.ground_truth.items():
            action = gt["action"]
            obs = env.step(
                {
                    "action_type": "process_message",
                    "msg_id": msg_id,
                    "action_label": action,
                    "metadata": {"episode_id": ep_id},
                }
            )
            if action == CASE3_ACTION_ESCALATE:
                obs = env.step(
                    {
                        "action_type": "escalate_with_reason",
                        "msg_id": msg_id,
                        "reason_code": CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
                        "reason": "Contains anxiety health data and article 9 special category information.",
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

    def test_available_actions_progress_by_case3_stage(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        episode = _EPISODES[ep_id]
        mixed_id = episode.mixed_sentence_message_id
        special_id = episode.special_category_message_ids[0]

        for msg_id, gt in episode.ground_truth.items():
            obs = env.step(
                {
                    "action_type": "process_message",
                    "msg_id": msg_id,
                    "action_label": gt["action"],
                    "metadata": {"episode_id": ep_id},
                }
            )

        assert obs.available_actions == ["redact_sentence"]

        obs = env.step(
            {
                "action_type": "redact_sentence",
                "msg_id": mixed_id,
                "sentence_index": 0,
                "decision": "keep",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert obs.available_actions == ["redact_sentence"]

        obs = env.step(
            {
                "action_type": "redact_sentence",
                "msg_id": mixed_id,
                "sentence_index": 1,
                "decision": "redact",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert obs.available_actions == ["escalate_with_reason"]

        obs = env.step(
            {
                "action_type": "escalate_with_reason",
                "msg_id": special_id,
                "reason_code": CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
                "reason": "Contains anxiety health data and article 9 special category information.",
                "metadata": {"episode_id": ep_id},
            }
        )
        assert obs.available_actions == ["compile_response"]

    def test_golden_trajectory_yields_high_score(self):
        env = DSAREnvironment()
        obs = env.reset(seed=42, task_id="task_hard")
        ep_id = obs.episode_id
        episode = _EPISODES[ep_id]

        for msg_id, gt in episode.ground_truth.items():
            obs = env.step(
                {
                    "action_type": "process_message",
                    "msg_id": msg_id,
                    "action_label": gt["action"],
                    "metadata": {"episode_id": ep_id},
                }
            )
            if gt["action"] == CASE3_ACTION_PARTIAL_REDACT:
                obs = env.step(
                    {
                        "action_type": "redact_sentence",
                        "msg_id": msg_id,
                        "sentence_index": 0,
                        "decision": "keep",
                        "metadata": {"episode_id": ep_id},
                    }
                )
                obs = env.step(
                    {
                        "action_type": "redact_sentence",
                        "msg_id": msg_id,
                        "sentence_index": 1,
                        "decision": "redact",
                        "metadata": {"episode_id": ep_id},
                    }
                )
            elif gt["action"] == CASE3_ACTION_ESCALATE:
                obs = env.step(
                    {
                        "action_type": "escalate_with_reason",
                        "msg_id": msg_id,
                        "reason_code": CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA,
                        "reason": (
                            "Escalate because this contains anxiety health data and "
                            "article 9 special category information."
                        ),
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
        assert final_obs.metadata["task3_score"] >= 0.95
        assert final_obs.metadata["c1_message_accuracy"] == 1.0
        assert final_obs.metadata["c2_sentence_redaction"] == 1.0
        assert final_obs.metadata["c3_escalation_quality"] == 1.0


class TestCase3InferenceHelpers:
    def test_action_type_allowed_only_accepts_current_family(self):
        obs = {"available_actions": ["redact_sentence"]}
        assert _available_actions(obs) == ["redact_sentence"]
        assert _action_type_allowed({"action_type": "redact_sentence"}, ["redact_sentence"]) is True
        assert _action_type_allowed({"action_type": "process_message"}, ["redact_sentence"]) is False

    def test_action_validation_message_names_allowed_actions(self):
        message = _action_validation_message(["redact_sentence"], "process_message")
        assert "process_message" in message
        assert "redact_sentence" in message

    def test_action_params_allowed_rejects_wrong_pending_sentence_target(self):
        obs = {
            "task_id": "task_hard",
            "sentences_pending": {"msg_a": [1]},
        }
        assert _action_params_allowed(
            {
                "action_type": "redact_sentence",
                "msg_id": "msg_a",
                "sentence_index": 1,
            },
            obs,
        ) is True
        assert _action_params_allowed(
            {
                "action_type": "redact_sentence",
                "msg_id": "msg_a",
                "sentence_index": 0,
            },
            obs,
        ) is False

    def test_action_params_allowed_rejects_wrong_pending_escalation_target(self):
        obs = {
            "task_id": "task_hard",
            "processed_messages": {
                "msg_a": {"action": "escalate"},
                "msg_b": {"action": "disclose"},
            },
            "escalation_log": {},
            "escalation_reason_codes": {},
        }
        assert _action_params_allowed(
            {"action_type": "escalate_with_reason", "msg_id": "msg_a"},
            obs,
        ) is True
        assert _action_params_allowed(
            {"action_type": "escalate_with_reason", "msg_id": "msg_b"},
            obs,
        ) is False

    def test_parameter_validation_message_mentions_current_pending_targets(self):
        obs = {
            "task_id": "task_hard",
            "sentences_pending": {"msg_a": [1]},
            "processed_messages": {"msg_a": {"action": "escalate"}},
            "escalation_log": {},
            "escalation_reason_codes": {},
        }
        redact_message = _action_parameter_validation_message(
            obs,
            {"action_type": "redact_sentence", "msg_id": "msg_a", "sentence_index": 0},
        )
        assert "msg_a" in redact_message
        assert "1" in redact_message

    def test_formatter_marks_non_partial_redact_sentences_as_not_applicable(self):
        bundle = generate_case3_episode(seed=42)
        obs = _base_case3_observation(bundle)
        bot_id = bundle["bot_message_id"]
        obs["processed_messages"] = {
            bot_id: {"action": CASE3_ACTION_EXCLUDE, "sentence_decisions": {}}
        }
        obs["messages_pending"] = [msg_id for msg_id in obs["messages_pending"] if msg_id != bot_id]
        formatted = format_observation(obs)
        assert "N/A" in formatted
        assert bot_id in formatted

    def test_heuristic_prioritizes_process_message_then_sentence_then_escalation_then_compile(self):
        bundle = generate_case3_episode(seed=42)
        obs = _base_case3_observation(bundle)
        first_action = choose_task_hard_action(obs)
        assert first_action is not None
        assert first_action == {
            "action_type": "process_message",
            "msg_id": bundle["bot_message_id"],
            "action_label": CASE3_ACTION_EXCLUDE,
        }

        mixed_id = bundle["mixed_sentence_message_id"]
        special_id = bundle["special_category_message_ids"][0]
        obs["processed_messages"] = {
            msg_id: {"action": gt["action"], "sentence_decisions": {}}
            for msg_id, gt in bundle["ground_truth"].items()
        }
        obs["processed_messages"][mixed_id]["sentence_decisions"] = {0: "keep"}
        obs["messages_pending"] = []
        obs["sentences_pending"] = {mixed_id: [1]}
        second_action = choose_task_hard_action(obs)
        assert second_action == {
            "action_type": "redact_sentence",
            "msg_id": mixed_id,
            "sentence_index": 1,
            "decision": "redact",
        }

        obs["processed_messages"][mixed_id]["sentence_decisions"] = {0: "keep", 1: "redact"}
        obs["sentences_pending"] = {}
        third_action = choose_task_hard_action(obs)
        assert third_action["action_type"] == "escalate_with_reason"
        assert third_action["msg_id"] == special_id

        obs["escalation_log"] = {
            special_id: "Escalate because this contains stress-related health data and article 9 special category information."
        }
        obs["escalation_reason_codes"] = {special_id: CASE3_REASON_CODE_SPECIAL_CATEGORY_HEALTH_DATA}
        obs["compile_ready"] = True
        fourth_action = choose_task_hard_action(obs)
        assert fourth_action == {"action_type": "compile_response"}

    def test_terminal_metrics_include_case3_scores(self):
        bundle = generate_case3_episode(seed=42)
        output = _perfect_case3_output(bundle)
        details = compute_terminal_score_case3(
            output,
            bundle["ground_truth"],
            bundle["special_category_message_ids"],
            bundle["mixed_sentence_message_id"],
        )
        for key in [
            "c1_message_accuracy",
            "c2_sentence_redaction",
            "c3_escalation_quality",
            "n_pii_breaches",
            "privacy_penalty",
            "task3_score",
        ]:
            assert key in details

    def test_extract_terminal_metrics_falls_back_to_terminal_details(self):
        observation = {
            "terminal_details": {
                "task3_score": 0.0,
                "c1_message_accuracy": 0.1,
                "termination_reason": "compile_response",
            }
        }
        extracted = _extract_terminal_metrics({}, observation)
        assert extracted["task3_score"] == 0.0
        assert extracted["c1_message_accuracy"] == 0.1
        assert extracted["termination_reason"] == "compile_response"
