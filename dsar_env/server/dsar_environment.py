"""Core DSAR environment logic."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

from dsar_env.models import (
    AuditEntry,
    DSARAction,
    DSARObservation,
    FieldItem,
    SlackMessageItem,
    SlackSentenceItem,
    TicketItem,
    TicketMessageItem,
    TicketSentenceItem,
)
from .constants import (
    CASE1_VALID_SILOS,
    CASE2_VALID_SILOS,
    CASE3_ACTION_ESCALATE,
    CASE3_ACTION_PARTIAL_REDACT,
    CASE3_MAX_STEPS,
    CASE3_REASON_CODES,
    MAX_STEPS,
)
from .generator import generate_case1_episode, generate_case2_episode, generate_case3_episode
from .grader import (
    clamp_task_score,
    compute_step_reward,
    compute_step_reward_case2,
    compute_step_reward_case3,
    compute_terminal_score,
    compute_terminal_score_case2,
    compute_terminal_score_case2_details,
    compute_terminal_score_case3,
)


@dataclass
class EpisodeData:
    episode_id: str
    task_id: str
    customer_record: List[Dict[str, Any]]
    values_lookup: Dict[str, Any]
    ground_truth: Dict[str, Any]
    dsar_text: str
    queried_silos: set = dc_field(default_factory=set)
    visible_field_ids: set = dc_field(default_factory=set)
    classified_fields: set = dc_field(default_factory=set)
    draft_response: Dict[str, Any] = dc_field(default_factory=dict)
    audit_trail: List[AuditEntry] = dc_field(default_factory=list)
    step_count: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    leaked_count: int = 0
    constraint_violated: bool = False
    phase: str = "classification"
    identity_confidence: float = 1.0
    verification_threshold: float = 1.0
    correct_verification_method: str = ""
    verification_attempted: bool = False
    verification_succeeded: bool = False
    verification_attempts: set = dc_field(default_factory=set)
    proportionate_attempt_count: int = 0
    disproportionate_used: bool = False
    submitted_identity: Dict[str, Any] = dc_field(default_factory=dict)
    internal_identity_full: Dict[str, Any] = dc_field(default_factory=dict)
    internal_identity_visible: Dict[str, Any] = dc_field(default_factory=dict)
    internal_identity_masked_by_silo: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    tickets: List[Dict[str, Any]] = dc_field(default_factory=list)
    ticket_ground_truth: Dict[str, Dict[int, str]] = dc_field(default_factory=dict)
    processed_sentences: Dict[str, Dict[int, str]] = dc_field(default_factory=dict)
    phase1_reward_sum: float = 0.0
    leaked_pii_sentences: int = 0
    blocked_compile_attempts: int = 0
    slack_export: List[Dict[str, Any]] = dc_field(default_factory=list)
    users_json: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    processed_messages: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    escalation_log: Dict[str, str] = dc_field(default_factory=dict)
    escalation_reason_codes: Dict[str, str] = dc_field(default_factory=dict)
    special_category_message_ids: List[str] = dc_field(default_factory=list)
    mixed_sentence_message_id: str = ""
    thread_parent_id: str = ""
    thread_reply_id: str = ""
    bot_message_id: str = ""


_EPISODES: Dict[str, EpisodeData] = {}


def _cleanup_old_episodes(max_episodes: int = 100) -> None:
    if len(_EPISODES) > max_episodes:
        for key in list(_EPISODES.keys())[: len(_EPISODES) - max_episodes]:
            del _EPISODES[key]


def _visible_field_items(episode: EpisodeData) -> List[FieldItem]:
    return [
        FieldItem(**item)
        for item in episode.customer_record
        if item["field_id"] in episode.visible_field_ids
    ]


def _ticket_items(tickets: List[Dict[str, Any]]) -> List[TicketItem]:
    out: List[TicketItem] = []
    for ticket in tickets:
        msgs = []
        for message in ticket.get("messages", []):
            msgs.append(
                TicketMessageItem(
                    message_index=message["message_index"],
                    speaker=message["speaker"],
                    text=message["text"],
                    sentences=[TicketSentenceItem(**s) for s in message.get("sentences", [])],
                )
            )
        out.append(TicketItem(ticket_id=ticket["ticket_id"], category=ticket["category"], messages=msgs))
    return out


def _slack_items(messages: List[Dict[str, Any]]) -> List[SlackMessageItem]:
    out: List[SlackMessageItem] = []
    for message in messages:
        out.append(
            SlackMessageItem(
                msg_id=message["msg_id"],
                user=message["user"],
                text=message["text"],
                ts=message["ts"],
                thread_ts=message.get("thread_ts"),
                subtype=message.get("subtype"),
                sentences=[SlackSentenceItem(**s) for s in message.get("sentences", [])],
            )
        )
    return out


def _sorted_values(values: set) -> List[str]:
    return sorted(str(v) for v in values)


def _reveal_fields_for_silo(episode: EpisodeData, silo_name: str) -> int:
    newly = {
        item["field_id"]
        for item in episode.customer_record
        if item["source_silo"] in {silo_name, "both"}
    } - episode.visible_field_ids
    episode.visible_field_ids.update(newly)
    return len(newly)


def _all_case2_sentences_processed(episode: EpisodeData) -> bool:
    for ticket_id, labels in episode.ticket_ground_truth.items():
        processed = episode.processed_sentences.get(ticket_id, {})
        for sentence_index in labels:
            if sentence_index not in processed:
                return False
    return True


def _case2_progress(episode: EpisodeData) -> tuple[int, int, float]:
    """Return processed count, total count, and completion coverage."""
    total_count = sum(len(labels) for labels in episode.ticket_ground_truth.values())
    processed_count = sum(len(sentence_map) for sentence_map in episode.processed_sentences.values())
    coverage = processed_count / total_count if total_count > 0 else 0.0
    return processed_count, total_count, coverage


def _case3_pending_messages(episode: EpisodeData) -> List[str]:
    return sorted(
        message["msg_id"]
        for message in episode.slack_export
        if message["msg_id"] not in episode.processed_messages
    )


def _case3_sentences_pending(episode: EpisodeData) -> Dict[str, List[int]]:
    pending: Dict[str, List[int]] = {}
    for message in episode.slack_export:
        msg_id = message["msg_id"]
        decision = episode.processed_messages.get(msg_id, {})
        sentence_ground_truth = episode.ground_truth.get(msg_id, {}).get("sentence_ground_truth")
        if decision.get("action") != CASE3_ACTION_PARTIAL_REDACT or not sentence_ground_truth:
            continue
        sentence_decisions = decision.get("sentence_decisions", {})
        unresolved = [
            sentence["sentence_idx"]
            for sentence in message.get("sentences", [])
            if sentence["sentence_idx"] in sentence_ground_truth
            and sentence["sentence_idx"] not in sentence_decisions
        ]
        if unresolved:
            pending[msg_id] = unresolved
    return pending


def _all_case3_messages_processed(episode: EpisodeData) -> bool:
    return len(episode.processed_messages) == len(episode.slack_export)


def _all_case3_escalations_completed(episode: EpisodeData) -> bool:
    for msg_id, decision in episode.processed_messages.items():
        if (
            decision.get("action") == CASE3_ACTION_ESCALATE
            and (
                msg_id not in episode.escalation_log
                or msg_id not in episode.escalation_reason_codes
            )
        ):
            return False
    return True


def _case3_compile_ready(episode: EpisodeData) -> bool:
    return (
        _all_case3_messages_processed(episode)
        and not _case3_sentences_pending(episode)
        and _all_case3_escalations_completed(episode)
    )


def _max_steps_for_episode(episode: EpisodeData) -> int:
    return CASE3_MAX_STEPS if episode.task_id == "task_hard" else MAX_STEPS


class DSAREnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_episode_id: Optional[str] = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: str = "task_easy", **kwargs: Any) -> Observation:
        _cleanup_old_episodes()
        ep_id = episode_id or str(uuid4())
        self._current_episode_id = ep_id
        self._state = State(episode_id=ep_id, step_count=0)

        if task_id == "task_medium":
            bundle = generate_case2_episode(seed=seed)
            episode = EpisodeData(
                episode_id=ep_id,
                task_id=task_id,
                customer_record=bundle["customer_record"],
                values_lookup=bundle["values_lookup"],
                ground_truth=bundle["ground_truth"],
                dsar_text=bundle["dsar_text"],
                phase="identity",
                identity_confidence=bundle["starting_identity_confidence"],
                verification_threshold=bundle["verification_threshold"],
                correct_verification_method=bundle["correct_verification_method"],
                submitted_identity=bundle["submitted_identity"],
                internal_identity_full=bundle["internal_identity_full"],
                internal_identity_masked_by_silo=bundle["internal_identity_masked"],
                tickets=bundle["tickets"],
                ticket_ground_truth=bundle["ticket_ground_truth"],
            )
        elif task_id == "task_hard":
            bundle = generate_case3_episode(seed=seed)
            episode = EpisodeData(
                episode_id=ep_id,
                task_id=task_id,
                customer_record=[],
                values_lookup={},
                ground_truth=bundle["ground_truth"],
                dsar_text=bundle["dsar_text"],
                phase="triage",
                slack_export=bundle["messages"],
                users_json=bundle["users_json"],
                special_category_message_ids=bundle["special_category_message_ids"],
                mixed_sentence_message_id=bundle["mixed_sentence_message_id"],
                thread_parent_id=bundle["thread_parent_id"],
                thread_reply_id=bundle["thread_reply_id"],
                bot_message_id=bundle["bot_message_id"],
            )
        else:
            customer_record, values_lookup, ground_truth, dsar_text = generate_case1_episode(seed=seed)
            episode = EpisodeData(
                episode_id=ep_id,
                task_id=task_id if task_id == "task_easy" else "task_easy",
                customer_record=customer_record,
                values_lookup=values_lookup,
                ground_truth=ground_truth,
                dsar_text=dsar_text,
                phase="classification",
            )

        _EPISODES[ep_id] = episode
        return self._build_observation(episode, reward=0.0, error=None)

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if isinstance(action, DSARAction):
            dsar_action = action
        elif isinstance(action, dict):
            dsar_action = DSARAction(**action)
        elif hasattr(action, "model_dump"):
            dsar_action = DSARAction(**action.model_dump())
        else:
            dsar_action = DSARAction(
                action_type=getattr(action, "action_type", "compile_response"),
                silo_name=getattr(action, "silo_name", None),
                field_id=getattr(action, "field_id", None),
                decision=getattr(action, "decision", None),
                verification_method=getattr(action, "verification_method", None),
                ticket_id=getattr(action, "ticket_id", None),
                sentence_index=getattr(action, "sentence_index", None),
                msg_id=getattr(action, "msg_id", None),
                action_label=getattr(action, "action_label", None),
                reason=getattr(action, "reason", None),
                reason_code=getattr(action, "reason_code", None),
                metadata=getattr(action, "metadata", {}),
            )

        ep_id = dsar_action.metadata.get("episode_id") or self._current_episode_id
        if ep_id is None or ep_id not in _EPISODES:
            return DSARObservation(done=True, reward=0.0, metadata={"error": "No active episode. Call reset() first."}, episode_id=ep_id or "", error="No active episode. Call reset() first.")

        episode = _EPISODES[ep_id]
        if episode.done:
            return self._build_observation(episode, reward=0.0, error="Episode already finished.", extra_metadata={"error": "Episode already finished."}, done=True)

        episode.step_count += 1
        self._state.step_count = episode.step_count
        if episode.task_id == "task_medium":
            return self._step_case2(episode, dsar_action)
        if episode.task_id == "task_hard":
            return self._step_case3(episode, dsar_action)
        return self._step_case1(episode, dsar_action)

    def _step_case1(self, episode: EpisodeData, action: DSARAction) -> Observation:
        pre_queried = frozenset(episode.queried_silos)
        pre_visible = frozenset(episode.visible_field_ids)
        pre_classified = frozenset(episode.classified_fields)
        error = None
        desc = f"Unknown action type: '{action.action_type}'"

        if action.action_type == "query_silo":
            if action.silo_name and action.silo_name not in episode.queried_silos and action.silo_name in CASE1_VALID_SILOS:
                episode.queried_silos.add(action.silo_name)
                count = _reveal_fields_for_silo(episode, action.silo_name)
                desc = f"Successfully queried silo '{action.silo_name}' and revealed {count} field(s)"
            elif action.silo_name in episode.queried_silos:
                error = f"Silo '{action.silo_name}' already queried this episode."
                desc = f"Redundant query: silo '{action.silo_name}' already queried"
            else:
                error = f"Invalid silo name: '{action.silo_name}'. Valid silos: billing, crm"
                desc = f"Invalid silo: '{action.silo_name}'"
        elif action.action_type == "classify_field":
            fid = action.field_id
            if fid and fid in episode.ground_truth and fid not in episode.visible_field_ids:
                error = f"Field '{fid}' is not visible yet. Query the relevant silo first."
                desc = f"Attempted to classify unrevealed field '{fid}'"
            elif fid and fid in episode.ground_truth and fid not in episode.classified_fields:
                episode.classified_fields.add(fid)
                if action.decision == "disclose":
                    episode.draft_response[fid] = episode.values_lookup.get(fid)
                    if episode.ground_truth[fid] == "INTERNAL_ONLY":
                        episode.leaked_count += 1
                        desc = f"LEAK: classified internal field '{fid}' as disclose (leaked_count={episode.leaked_count})"
                    else:
                        desc = f"Correctly disclosed personal data field '{fid}'"
                else:
                    desc = (
                        f"Incorrectly withheld personal data field '{fid}'"
                        if episode.ground_truth[fid] == "REQUESTER_DATA"
                        else f"Correctly withheld internal field '{fid}'"
                    )
            elif fid and fid in episode.classified_fields:
                error = f"Field '{fid}' already classified this episode."
                desc = f"Redundant classification: '{fid}' already done"
            elif fid and fid not in episode.ground_truth:
                error = f"Field '{fid}' does not exist in the record."
                desc = f"Unknown field: '{fid}'"
            else:
                error = "classify_field requires field_id and decision parameters."
                desc = "Invalid classify_field call - missing parameters"
        elif action.action_type != "compile_response":
            error = f"Unknown action type: '{action.action_type}'. Use: query_silo, classify_field, compile_response"

        step_reward = compute_step_reward(
            action_type=action.action_type,
            silo_name=action.silo_name,
            field_id=action.field_id,
            decision=action.decision,
            ground_truth=episode.ground_truth,
            queried_silos=pre_queried,
            classified_fields=pre_classified,
            visible_field_ids=pre_visible,
            step_number=episode.step_count,
            task_id=episode.task_id,
        )
        episode.cumulative_reward += step_reward
        episode.audit_trail.append(AuditEntry(step=episode.step_count, action=action.action_type, description=desc, reward=step_reward))

        if action.action_type == "compile_response":
            terminal = compute_terminal_score(episode.draft_response, episode.ground_truth, episode.queried_silos, steps_used=episode.step_count, task_id=episode.task_id)
            episode.done = True
            episode.cumulative_reward += terminal
            episode.audit_trail.append(AuditEntry(step=episode.step_count, action="compile_response", description=f"Response compiled. Terminal score: {terminal:.4f}", reward=terminal))
            return self._build_observation(episode, reward=terminal, error=None, done=True, extra_metadata={"terminal_score": terminal, "cumulative_reward": round(episode.cumulative_reward, 4), "fields_classified": len(episode.classified_fields), "fields_leaked": episode.leaked_count, "steps_used": episode.step_count})

        if MAX_STEPS - episode.step_count <= 0 and not episode.done:
            terminal = compute_terminal_score(episode.draft_response, episode.ground_truth, episode.queried_silos, steps_used=episode.step_count, task_id=episode.task_id)
            episode.done = True
            episode.cumulative_reward += terminal
            return self._build_observation(episode, reward=step_reward + terminal, error=error, done=True, extra_metadata={"terminal_score": terminal, "cumulative_reward": round(episode.cumulative_reward, 4)})

        return self._build_observation(episode, reward=step_reward, error=error)

    def _step_case2(self, episode: EpisodeData, action: DSARAction) -> Observation:
        pre_phase = episode.phase
        pre_queried = frozenset(episode.queried_silos)
        pre_attempts = frozenset(episode.verification_attempts)
        pre_processed = {k: dict(v) for k, v in episode.processed_sentences.items()}
        pre_verified = episode.verification_succeeded
        error = None
        desc = f"Unknown action type: '{action.action_type}'"

        if action.action_type == "query_silo":
            if action.silo_name and action.silo_name not in episode.queried_silos and action.silo_name in CASE2_VALID_SILOS:
                episode.queried_silos.add(action.silo_name)
                if episode.phase == "identity":
                    episode.internal_identity_visible.update(episode.internal_identity_masked_by_silo.get(action.silo_name, {}))
                    desc = f"Queried silo '{action.silo_name}' and revealed masked identity evidence"
                else:
                    count = _reveal_fields_for_silo(episode, action.silo_name)
                    desc = f"Queried silo '{action.silo_name}' and revealed {count} full field(s) after verification"
            elif action.silo_name in episode.queried_silos:
                error = f"Silo '{action.silo_name}' already queried this episode."
                desc = f"Redundant query: silo '{action.silo_name}' already queried"
            else:
                error = f"Invalid silo name: '{action.silo_name}'. Valid silos: billing, crm"
                desc = f"Invalid silo: '{action.silo_name}'"
        elif action.action_type == "verify_identity":
            method = action.verification_method
            if episode.phase != "identity":
                error = "verify_identity is only valid during the identity phase."
                desc = f"Attempted verify_identity during phase '{episode.phase}'"
            elif method is None:
                error = "verify_identity requires a verification_method."
                desc = "Missing verification_method"
            elif method not in {"transaction_date", "account_reference", "registered_postcode", "passport_copy", "photo_id"}:
                error = f"Unknown verification method: '{method}'."
                desc = f"Invalid verification method '{method}'"
            elif method in episode.verification_attempts:
                error = f"Verification method '{method}' already used."
                desc = f"Repeated verification method '{method}'"
            else:
                episode.verification_attempted = True
                episode.verification_attempts.add(method)
                if method in {"transaction_date", "account_reference", "registered_postcode"}:
                    episode.proportionate_attempt_count += 1
                if method in {"passport_copy", "photo_id"}:
                    episode.disproportionate_used = True
                if method == episode.correct_verification_method:
                    episode.identity_confidence = 0.90
                    episode.verification_succeeded = True
                    episode.phase = "redaction"
                    episode.internal_identity_visible = dict(episode.internal_identity_full)
                    for silo_name in episode.queried_silos:
                        _reveal_fields_for_silo(episode, silo_name)
                    desc = f"Verification succeeded with method '{method}'. Redaction phase unlocked."
                else:
                    desc = f"Verification method '{method}' attempted without crossing the threshold"
        elif action.action_type == "redact_span":
            if episode.phase != "redaction":
                error = "redact_span is only valid during the redaction phase."
                desc = f"Attempted redact_span during phase '{episode.phase}'"
            elif action.ticket_id is None or action.sentence_index is None or action.decision is None:
                error = "redact_span requires ticket_id, sentence_index, and decision."
                desc = "Missing redact_span parameters"
            elif action.ticket_id not in episode.ticket_ground_truth:
                error = f"Ticket '{action.ticket_id}' does not exist."
                desc = f"Invalid ticket id '{action.ticket_id}'"
            elif action.sentence_index not in episode.ticket_ground_truth[action.ticket_id]:
                error = f"Sentence index {action.sentence_index} does not exist in ticket '{action.ticket_id}'."
                desc = f"Invalid sentence index {action.sentence_index} for ticket '{action.ticket_id}'"
            elif action.sentence_index in episode.processed_sentences.get(action.ticket_id, {}):
                error = f"Sentence {action.sentence_index} in ticket '{action.ticket_id}' already processed."
                desc = f"Repeated redact_span on {action.ticket_id}:{action.sentence_index}"
            elif action.decision not in {"keep", "redact"}:
                error = "redact_span decision must be 'keep' or 'redact'."
                desc = f"Invalid redaction decision '{action.decision}'"
            else:
                episode.processed_sentences.setdefault(action.ticket_id, {})[action.sentence_index] = action.decision
                if episode.ticket_ground_truth[action.ticket_id][action.sentence_index] == "THIRD_PARTY_PII" and action.decision == "keep":
                    episode.leaked_pii_sentences += 1
                desc = f"Processed sentence {action.sentence_index} in ticket '{action.ticket_id}' with decision '{action.decision}'"
        elif action.action_type == "compile_response":
            if episode.phase == "identity" and episode.identity_confidence < episode.verification_threshold:
                step_reward = compute_step_reward_case2(action_type="compile_response", phase=pre_phase, step_number=episode.step_count, queried_silos=pre_queried, verification_method=action.verification_method, correct_verification_method=episode.correct_verification_method, verification_attempts=pre_attempts, ticket_id=action.ticket_id, sentence_index=action.sentence_index, decision=action.decision, ticket_ground_truth=episode.ticket_ground_truth, processed_sentences=pre_processed, identity_verified=pre_verified, all_sentences_processed=False, blocked_compile_attempts=episode.blocked_compile_attempts)
                episode.phase1_reward_sum += step_reward
                episode.cumulative_reward += step_reward
                episode.done = True
                episode.audit_trail.append(AuditEntry(step=episode.step_count, action="compile_response", description="compile_response called before identity threshold was reached", reward=step_reward))
                return self._build_observation(episode, reward=step_reward, error="Cannot compile response before identity confidence reaches the threshold.", done=True, extra_metadata={"terminal_score": clamp_task_score(0.0), "steps_used": episode.step_count})
            if episode.phase == "redaction" and not _all_case2_sentences_processed(episode):
                episode.blocked_compile_attempts += 1
                error = "All ticket sentences must be processed before compile_response."
                desc = "compile_response blocked because some sentences remain unprocessed"
            elif episode.phase == "redaction":
                details = compute_terminal_score_case2_details(
                    episode.processed_sentences,
                    episode.ticket_ground_truth,
                    episode.phase1_reward_sum,
                    verification_succeeded=episode.verification_succeeded,
                    proportionate_attempt_count=episode.proportionate_attempt_count,
                    disproportionate_used=episode.disproportionate_used,
                    completed_all_sentences=True,
                    termination_reason="compile_response",
                )
                terminal = details["task2_score"]
                episode.done = True
                episode.cumulative_reward += terminal
                episode.audit_trail.append(AuditEntry(step=episode.step_count, action="compile_response", description=f"Case 2 response compiled. Terminal score: {terminal:.4f}", reward=terminal))
                return self._build_observation(
                    episode,
                    reward=terminal,
                    error=None,
                    done=True,
                    extra_metadata={
                        "terminal_score": terminal,
                        "cumulative_reward": round(episode.cumulative_reward, 4),
                        "steps_used": episode.step_count,
                        **details,
                    },
                )
            else:
                error = f"compile_response is not valid during phase '{episode.phase}'."
                desc = f"Invalid compile_response during phase '{episode.phase}'"
        else:
            error = "Unknown action type for Case 2. Use: query_silo, verify_identity, redact_span, compile_response"

        step_reward = compute_step_reward_case2(
            action_type=action.action_type,
            phase=pre_phase,
            step_number=episode.step_count,
            queried_silos=pre_queried,
            silo_name=action.silo_name,
            verification_method=action.verification_method,
            correct_verification_method=episode.correct_verification_method,
            verification_attempts=pre_attempts,
            ticket_id=action.ticket_id,
            sentence_index=action.sentence_index,
            decision=action.decision,
            ticket_ground_truth=episode.ticket_ground_truth,
            processed_sentences=pre_processed,
            identity_verified=pre_verified,
            all_sentences_processed=_all_case2_sentences_processed(episode),
            blocked_compile_attempts=episode.blocked_compile_attempts,
        )
        episode.cumulative_reward += step_reward
        if pre_phase == "identity":
            episode.phase1_reward_sum += step_reward
        episode.audit_trail.append(AuditEntry(step=episode.step_count, action=action.action_type, description=desc, reward=step_reward))

        if MAX_STEPS - episode.step_count <= 0 and not episode.done:
            details = (
                compute_terminal_score_case2_details(
                    episode.processed_sentences,
                    episode.ticket_ground_truth,
                    episode.phase1_reward_sum,
                    verification_succeeded=episode.verification_succeeded,
                    proportionate_attempt_count=episode.proportionate_attempt_count,
                    disproportionate_used=episode.disproportionate_used,
                    completed_all_sentences=_all_case2_sentences_processed(episode),
                    termination_reason=(
                        "max_steps_complete"
                        if _all_case2_sentences_processed(episode)
                        else "incomplete_redaction_timeout"
                    ),
                )
                if episode.phase == "redaction"
                else {
                    "task2_score": clamp_task_score(0.0),
                    "completion_coverage": 0.0,
                    "termination_reason": "max_steps_identity_timeout",
                }
            )
            terminal = details["task2_score"]
            episode.done = True
            episode.cumulative_reward += terminal
            return self._build_observation(
                episode,
                reward=step_reward + terminal,
                error=error,
                done=True,
                extra_metadata={
                    "terminal_score": terminal,
                    "cumulative_reward": round(episode.cumulative_reward, 4),
                    "steps_used": episode.step_count,
                    **details,
                },
            )

        return self._build_observation(episode, reward=step_reward, error=error)

    def _step_case3(self, episode: EpisodeData, action: DSARAction) -> Observation:
        pre_processed = {
            msg_id: {
                "action": decision.get("action"),
                "sentence_decisions": dict(decision.get("sentence_decisions", {})),
            }
            for msg_id, decision in episode.processed_messages.items()
        }
        pre_escalation_log = dict(episode.escalation_log)
        pre_escalation_reason_codes = dict(episode.escalation_reason_codes)
        error = None
        desc = f"Unknown action type: '{action.action_type}'"
        step_reward_override: Optional[float] = None

        if action.action_type == "process_message":
            msg_id = action.msg_id
            action_label = action.action_label
            valid_actions = {"disclose", "partial_redact", "exclude", "escalate"}
            if msg_id is None or action_label is None:
                error = "process_message requires msg_id and action_label."
                desc = "Missing process_message parameters"
            elif msg_id not in episode.ground_truth:
                error = f"Message '{msg_id}' does not exist."
                desc = f"Invalid message id '{msg_id}'"
            elif msg_id in episode.processed_messages:
                error = f"Message '{msg_id}' already processed."
                desc = f"Repeated process_message on '{msg_id}'"
            elif action_label not in valid_actions:
                error = f"Invalid action_label '{action_label}'."
                desc = f"Invalid Case 3 action label '{action_label}'"
            else:
                episode.processed_messages[msg_id] = {"action": action_label, "sentence_decisions": {}}
                desc = f"Processed message '{msg_id}' with action '{action_label}'"
        elif action.action_type == "redact_sentence":
            msg_id = action.msg_id
            sentence_index = action.sentence_index
            decision = action.decision
            if msg_id is None or sentence_index is None or decision is None:
                error = "redact_sentence requires msg_id, sentence_index, and decision."
                desc = "Missing redact_sentence parameters"
            elif msg_id not in episode.processed_messages:
                error = f"Message '{msg_id}' has not been processed yet."
                desc = f"redact_sentence before process_message on '{msg_id}'"
            elif episode.processed_messages[msg_id].get("action") != CASE3_ACTION_PARTIAL_REDACT:
                error = f"Message '{msg_id}' is not in partial_redact state."
                desc = f"redact_sentence attempted on non-partial-redact message '{msg_id}'"
            elif decision not in {"keep", "redact"}:
                error = "redact_sentence decision must be 'keep' or 'redact'."
                desc = f"Invalid redact_sentence decision '{decision}'"
            else:
                sentence_ground_truth = episode.ground_truth.get(msg_id, {}).get("sentence_ground_truth")
                if not sentence_ground_truth:
                    error = f"Message '{msg_id}' does not support sentence-level redaction."
                    desc = f"redact_sentence attempted on non-sentence-actionable message '{msg_id}'"
                    sentence_ground_truth = None
                message = next((msg for msg in episode.slack_export if msg["msg_id"] == msg_id), None)
                visible_sentence_indices = {
                    sentence["sentence_idx"] for sentence in (message or {}).get("sentences", [])
                }
                if error is not None:
                    pass
                elif sentence_index not in visible_sentence_indices or sentence_index not in sentence_ground_truth:
                    error = f"Sentence index {sentence_index} does not exist in message '{msg_id}'."
                    desc = f"Invalid sentence index {sentence_index} for '{msg_id}'"
                elif sentence_index in episode.processed_messages[msg_id].get("sentence_decisions", {}):
                    error = f"Sentence {sentence_index} in message '{msg_id}' already processed."
                    desc = f"Repeated redact_sentence on '{msg_id}:{sentence_index}'"
                else:
                    episode.processed_messages[msg_id].setdefault("sentence_decisions", {})[sentence_index] = decision
                    desc = f"Processed sentence {sentence_index} in message '{msg_id}' with decision '{decision}'"
        elif action.action_type == "escalate_with_reason":
            msg_id = action.msg_id
            reason = action.reason
            reason_code = action.reason_code
            if msg_id is None or reason is None or reason_code is None:
                error = "escalate_with_reason requires msg_id, reason_code, and reason."
                desc = "Missing escalate_with_reason parameters"
            elif episode.processed_messages.get(msg_id, {}).get("action") != CASE3_ACTION_ESCALATE:
                error = f"Message '{msg_id}' was not processed as escalate."
                desc = f"escalate_with_reason attempted on non-escalated message '{msg_id}'"
            elif msg_id in episode.escalation_log or msg_id in episode.escalation_reason_codes:
                error = f"Message '{msg_id}' already has an escalation reason."
                desc = f"Repeated escalation reason for '{msg_id}'"
            elif reason_code not in CASE3_REASON_CODES:
                error = f"Invalid reason_code '{reason_code}'."
                desc = f"Invalid escalation reason_code '{reason_code}'"
            else:
                episode.escalation_log[msg_id] = reason
                episode.escalation_reason_codes[msg_id] = reason_code
                desc = f"Recorded escalation reason for '{msg_id}' with reason_code '{reason_code}'"
        elif action.action_type == "compile_response":
            if not _all_case3_messages_processed(episode):
                error = "All Slack messages must be processed before compile_response."
                desc = "compile_response blocked because messages remain unprocessed"
                step_reward_override = -0.05
            elif _case3_sentences_pending(episode):
                error = "All sentence-level redactions must be completed before compile_response."
                desc = "compile_response blocked because sentence decisions remain pending"
                step_reward_override = -0.05
            elif not _all_case3_escalations_completed(episode):
                error = "All escalated messages must include a reason before compile_response."
                desc = "compile_response blocked because escalation reasons remain pending"
                step_reward_override = -0.05
            else:
                details = compute_terminal_score_case3(
                    {
                        "processed_messages": episode.processed_messages,
                        "escalation_log": episode.escalation_log,
                        "escalation_reason_codes": episode.escalation_reason_codes,
                    },
                    episode.ground_truth,
                    episode.special_category_message_ids,
                    episode.mixed_sentence_message_id,
                )
                terminal = details["task3_score"]
                episode.done = True
                episode.cumulative_reward += terminal
                episode.audit_trail.append(
                    AuditEntry(
                        step=episode.step_count,
                        action="compile_response",
                        description=f"Case 3 response compiled. Terminal score: {terminal:.4f}",
                        reward=terminal,
                    )
                )
                return self._build_observation(
                    episode,
                    reward=terminal,
                    error=None,
                    done=True,
                    extra_metadata={
                        "terminal_score": terminal,
                        "cumulative_reward": round(episode.cumulative_reward, 4),
                        "steps_used": episode.step_count,
                        **details,
                    },
                )
        else:
            error = (
                "Unknown action type for Case 3. Use: process_message, redact_sentence, "
                "escalate_with_reason, compile_response"
            )

        step_reward = compute_step_reward_case3(
            action_type=action.action_type,
            step_number=episode.step_count,
            msg_id=action.msg_id,
            action_label=action.action_label,
            sentence_index=action.sentence_index,
            decision=action.decision,
            reason=action.reason,
            reason_code=action.reason_code,
            ground_truth=episode.ground_truth,
            processed_messages=pre_processed,
            escalation_log=pre_escalation_log,
            escalation_reason_codes=pre_escalation_reason_codes,
        )
        if step_reward_override is not None:
            step_reward = step_reward_override
        episode.cumulative_reward += step_reward
        episode.audit_trail.append(
            AuditEntry(step=episode.step_count, action=action.action_type, description=desc, reward=step_reward)
        )

        if (
            action.action_type == "process_message"
            and action.msg_id in episode.special_category_message_ids
            and action.action_label == "disclose"
            and not episode.done
        ):
            episode.constraint_violated = True
            episode.done = True
            details = compute_terminal_score_case3(
                {
                    "processed_messages": episode.processed_messages,
                    "escalation_log": episode.escalation_log,
                    "escalation_reason_codes": episode.escalation_reason_codes,
                },
                episode.ground_truth,
                episode.special_category_message_ids,
                episode.mixed_sentence_message_id,
            )
            return self._build_observation(
                episode,
                reward=step_reward,
                error="CONSTRAINT VIOLATED: special-category health data was disclosed.",
                done=True,
                extra_metadata={
                    "terminal_score": clamp_task_score(0.0),
                    "steps_used": episode.step_count,
                    **details,
                },
            )

        if _max_steps_for_episode(episode) - episode.step_count <= 0 and not episode.done:
            details = compute_terminal_score_case3(
                {
                    "processed_messages": episode.processed_messages,
                    "escalation_log": episode.escalation_log,
                    "escalation_reason_codes": episode.escalation_reason_codes,
                },
                episode.ground_truth,
                episode.special_category_message_ids,
                episode.mixed_sentence_message_id,
            )
            terminal = details["task3_score"]
            episode.done = True
            episode.cumulative_reward += terminal
            return self._build_observation(
                episode,
                reward=step_reward + terminal,
                error=error,
                done=True,
                extra_metadata={
                    "terminal_score": terminal,
                    "cumulative_reward": round(episode.cumulative_reward, 4),
                    "steps_used": episode.step_count,
                    **details,
                },
            )

        return self._build_observation(episode, reward=step_reward, error=error)

    def _build_observation(self, episode: EpisodeData, reward: float, error: Optional[str], extra_metadata: Optional[Dict[str, Any]] = None, done: Optional[bool] = None) -> DSARObservation:
        extra_metadata = extra_metadata or {}
        max_steps = _max_steps_for_episode(episode)
        steps_remaining = max(0, max_steps - episode.step_count)
        metadata = {"episode_id": episode.episode_id, "task_id": episode.task_id, "step_count": episode.step_count, "cumulative_reward": round(episode.cumulative_reward, 4)}
        metadata.update(extra_metadata)
        if episode.task_id == "task_medium":
            actions = ["query_silo", "verify_identity", "compile_response"] if episode.phase == "identity" else ["query_silo", "redact_span", "compile_response"]
            processed_count, total_count, coverage = _case2_progress(episode)
            pending_count = max(0, total_count - processed_count)
            compile_ready = episode.phase == "redaction" and pending_count == 0 and total_count > 0
            if episode.phase == "redaction":
                actions = ["query_silo", "redact_span"] + (["compile_response"] if compile_ready else [])
            return DSARObservation(
                done=episode.done if done is None else done,
                reward=reward,
                metadata=metadata,
                episode_id=episode.episode_id,
                task_id=episode.task_id,
                dsar_request=episode.dsar_text,
                customer_record=_visible_field_items(episode),
                available_actions=actions,
                silo_results=_sorted_values(episode.queried_silos),
                identity_verified=episode.verification_succeeded,
                draft_response={"processed_sentences": episode.processed_sentences} if episode.processed_sentences else {},
                audit_trail=episode.audit_trail,
                deadline_pressure=max(0.0, steps_remaining / max_steps),
                steps_remaining=steps_remaining,
                classified_fields=[],
                constraint_violated=False,
                error=error,
                phase=episode.phase,
                identity_confidence=episode.identity_confidence,
                identity_threshold=episode.verification_threshold,
                submitted_identity=episode.submitted_identity,
                internal_identity=episode.internal_identity_visible,
                tickets=_ticket_items(episode.tickets) if episode.verification_succeeded else [],
                processed_sentences=episode.processed_sentences,
                pending_sentence_count=pending_count,
                total_sentence_count=total_count,
                completion_coverage=coverage,
                compile_ready=compile_ready,
                terminal_details=extra_metadata,
            )
        if episode.task_id == "task_hard":
            pending_messages = _case3_pending_messages(episode)
            sentences_pending = _case3_sentences_pending(episode)
            compile_ready = _case3_compile_ready(episode)
            unresolved_escalations = sorted(
                msg_id
                for msg_id, decision in episode.processed_messages.items()
                if decision.get("action") == CASE3_ACTION_ESCALATE
                and (
                    msg_id not in episode.escalation_log
                    or msg_id not in episode.escalation_reason_codes
                )
            )
            if pending_messages:
                actions = ["process_message"]
            elif sentences_pending:
                actions = ["redact_sentence"]
            elif unresolved_escalations:
                actions = ["escalate_with_reason"]
            elif compile_ready:
                actions = ["compile_response"]
            else:
                actions = ["process_message", "redact_sentence", "escalate_with_reason"]
            return DSARObservation(
                done=episode.done if done is None else done,
                reward=reward,
                metadata=metadata,
                episode_id=episode.episode_id,
                task_id=episode.task_id,
                dsar_request=episode.dsar_text,
                customer_record=[],
                available_actions=actions,
                silo_results=[],
                identity_verified=True,
                draft_response={"processed_messages": episode.processed_messages},
                audit_trail=episode.audit_trail,
                deadline_pressure=max(0.0, steps_remaining / max_steps),
                steps_remaining=steps_remaining,
                classified_fields=[],
                constraint_violated=episode.constraint_violated,
                error=error,
                phase="triage",
                identity_confidence=1.0,
                identity_threshold=1.0,
                submitted_identity={},
                internal_identity={},
                tickets=[],
                processed_sentences={},
                pending_sentence_count=0,
                total_sentence_count=0,
                completion_coverage=1.0 if compile_ready else 0.0,
                compile_ready=compile_ready,
                slack_export=_slack_items(episode.slack_export),
                users_json=episode.users_json,
                processed_messages=episode.processed_messages,
                escalation_log=episode.escalation_log,
                escalation_reason_codes=episode.escalation_reason_codes,
                messages_pending=pending_messages,
                sentences_pending=sentences_pending,
                terminal_details=extra_metadata,
            )
        return DSARObservation(
            done=episode.done if done is None else done,
            reward=reward,
            metadata=metadata,
            episode_id=episode.episode_id,
            task_id=episode.task_id,
            dsar_request=episode.dsar_text,
            customer_record=_visible_field_items(episode),
            available_actions=["query_silo", "classify_field", "compile_response"],
            silo_results=_sorted_values(episode.queried_silos),
            identity_verified=True,
            draft_response=episode.draft_response,
            audit_trail=episode.audit_trail,
            deadline_pressure=max(0.0, steps_remaining / max_steps),
            steps_remaining=steps_remaining,
            classified_fields=_sorted_values(episode.classified_fields),
            constraint_violated=episode.constraint_violated,
            error=error,
            phase="classification",
            identity_confidence=1.0,
            identity_threshold=1.0,
            submitted_identity={},
            internal_identity={},
            tickets=[],
            processed_sentences={},
            pending_sentence_count=0,
            total_sentence_count=0,
            completion_coverage=0.0,
            compile_ready=True,
            terminal_details=extra_metadata,
            slack_export=[],
            users_json={},
            processed_messages={},
            escalation_log={},
            escalation_reason_codes={},
            messages_pending=[],
            sentences_pending={},
        )

    @property
    def state(self) -> State:
        return self._state
