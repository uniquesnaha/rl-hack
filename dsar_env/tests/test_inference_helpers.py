"""Focused tests for inference parsing and state validation helpers."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    _action_params_allowed,
    _case1_pending_field_ids,
    _case2_pending_sentence_targets,
    parse_model_action,
)


class TestInferenceParsing:
    def test_parse_classify_field_accepts_bracketed_field_id(self):
        parsed = parse_model_action("classify_field [support_ticket_ids] disclose")
        assert parsed == {
            "action_type": "classify_field",
            "field_id": "support_ticket_ids",
            "decision": "disclose",
        }

    def test_parse_model_action_returns_parse_error_for_unknown_shape(self):
        parsed = parse_model_action("nonsense action")
        assert parsed["action_type"] == "__parse_error__"


class TestInferenceStateValidation:
    def test_case1_pending_field_ids_excludes_classified_fields(self):
        obs = {
            "customer_record": [
                {"field_id": "full_name"},
                {"field_id": "email"},
            ],
            "classified_fields": ["full_name"],
        }
        assert _case1_pending_field_ids(obs) == ["email"]

    def test_action_params_allowed_for_task_easy_classify_field(self):
        obs = {
            "task_id": "task_easy",
            "customer_record": [{"field_id": "email"}],
            "classified_fields": [],
        }
        assert _action_params_allowed(
            {"action_type": "classify_field", "field_id": "email", "decision": "disclose"},
            obs,
        ) is True
        assert _action_params_allowed(
            {"action_type": "classify_field", "field_id": "full_name", "decision": "disclose"},
            obs,
        ) is False

    def test_case2_pending_sentence_targets_reads_unprocessed_sentences(self):
        obs = {
            "processed_sentences": {"TKT-1": {0: "keep"}},
            "tickets": [
                {
                    "ticket_id": "TKT-1",
                    "messages": [
                        {
                            "sentences": [
                                {"sentence_index": 0, "text": "A"},
                                {"sentence_index": 1, "text": "B"},
                            ]
                        }
                    ],
                }
            ],
        }
        assert _case2_pending_sentence_targets(obs) == {"TKT-1": [1]}

    def test_action_params_allowed_for_task_medium_redact_span(self):
        obs = {
            "task_id": "task_medium",
            "phase": "redaction",
            "processed_sentences": {},
            "tickets": [
                {
                    "ticket_id": "TKT-1",
                    "messages": [
                        {"sentences": [{"sentence_index": 0, "text": "A"}]}
                    ],
                }
            ],
        }
        assert _action_params_allowed(
            {
                "action_type": "redact_span",
                "ticket_id": "TKT-1",
                "sentence_index": 0,
                "decision": "keep",
            },
            obs,
        ) is True
        assert _action_params_allowed(
            {
                "action_type": "redact_span",
                "ticket_id": "TKT-1",
                "sentence_index": 1,
                "decision": "keep",
            },
            obs,
        ) is False
