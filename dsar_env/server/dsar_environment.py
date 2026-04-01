"""
DSAR Environment — core environment logic.

Subclasses OpenEnv's Environment base class.
Implements reset(), step(), and state property.

Uses a module-level _EPISODES dict for state persistence across HTTP requests,
since the OpenEnv HTTPEnvServer creates a new environment instance per request.

Version note: The environment returns Observation subclasses with `done` and
`reward` set. If a future OpenEnv release changes the step/reset return contract,
update the return types here accordingly.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from dataclasses import dataclass, field as dc_field

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

from dsar_env.models import DSARAction, DSARObservation, FieldItem, AuditEntry
from .generator import generate_case1_episode
from .grader import compute_step_reward, compute_terminal_score
from .constants import CASE1_VALID_SILOS, MAX_STEPS


# ─── Episode data stored in module-level dict ─────────────────────────────────

@dataclass
class EpisodeData:
    """Internal episode state — hidden from the agent, used by grader."""
    episode_id: str
    task_id: str
    customer_record: List[Dict[str, Any]]  # List of FieldItem dicts
    values_lookup: Dict[str, Any]           # Flat {field_id: value} for grader
    ground_truth: Dict[str, str]            # {field_id: 'REQUESTER_DATA'|'INTERNAL_ONLY'}
    dsar_text: str
    queried_silos: set = dc_field(default_factory=set)
    classified_fields: set = dc_field(default_factory=set)
    draft_response: Dict[str, Any] = dc_field(default_factory=dict)
    audit_trail: list = dc_field(default_factory=list)  # List[AuditEntry]
    step_count: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    leaked_count: int = 0
    constraint_violated: bool = False


# Module-level persistent episode store — keyed by UUID
_EPISODES: Dict[str, EpisodeData] = {}


def _cleanup_old_episodes(max_episodes: int = 100) -> None:
    """Remove oldest episodes if store grows too large (memory safety for HF Space)."""
    if len(_EPISODES) > max_episodes:
        keys = list(_EPISODES.keys())
        for k in keys[: len(keys) - max_episodes]:
            del _EPISODES[k]


class DSAREnvironment(Environment):
    """DSAR Compliance Environment.

    Implements a GDPR Data Subject Access Request processing environment
    with three difficulty levels via task_id parameter on reset():
      - task_easy:   Clean consumer request — field-level classification
      - task_medium:  Mismatched identity + support ticket redaction
      - task_hard:   Weaponised employee DSAR on Slack export
    """

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_episode_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task_easy",
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment and start a new episode.

        Args:
            seed: Optional random seed for reproducible episodes.
            episode_id: Optional custom episode ID.
            task_id: Task to run — 'task_easy', 'task_medium', or 'task_hard'.

        Returns:
            DSARObservation with the initial state.
        """
        _cleanup_old_episodes()

        ep_id = episode_id or str(uuid4())
        self._current_episode_id = ep_id
        self._state = State(episode_id=ep_id, step_count=0)

        # Generate fresh synthetic data
        if task_id == "task_easy":
            customer_record, values_lookup, ground_truth, dsar_text = generate_case1_episode(seed=seed)
        elif task_id == "task_medium":
            # TODO: implement Case 2 generator (uses Case 1 for now)
            customer_record, values_lookup, ground_truth, dsar_text = generate_case1_episode(seed=seed)
        elif task_id == "task_hard":
            # TODO: implement Case 3 generator (uses Case 1 for now)
            customer_record, values_lookup, ground_truth, dsar_text = generate_case1_episode(seed=seed)
        else:
            customer_record, values_lookup, ground_truth, dsar_text = generate_case1_episode(seed=seed)

        # Store episode state
        episode = EpisodeData(
            episode_id=ep_id,
            task_id=task_id,
            customer_record=customer_record,
            values_lookup=values_lookup,
            ground_truth=ground_truth,
            dsar_text=dsar_text,
        )
        _EPISODES[ep_id] = episode

        # Build initial observation — customer_record as FieldItem models
        field_items = [FieldItem(**fi) for fi in customer_record]

        return DSARObservation(
            done=False,
            reward=0.0,
            metadata={"episode_id": ep_id, "task_id": task_id},
            episode_id=ep_id,
            task_id=task_id,
            dsar_request=dsar_text,
            customer_record=field_items,
            available_actions=["query_silo", "classify_field", "compile_response"],
            silo_results=[],
            identity_verified=True,
            draft_response={},
            audit_trail=[],
            deadline_pressure=1.0,
            steps_remaining=MAX_STEPS,
            classified_fields=[],
            constraint_violated=False,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one step in the environment.

        Args:
            action: DSARAction with action_type and parameters.
            timeout_s: Optional timeout (unused).

        Returns:
            DSARObservation with updated state, reward, and done flag.
        """
        # ── Parse the action ──────────────────────────────────────────────
        if isinstance(action, DSARAction):
            dsar_action = action
        elif isinstance(action, dict):
            dsar_action = DSARAction(**action)
        elif hasattr(action, "model_dump"):
            data = action.model_dump()
            dsar_action = DSARAction(**data)
        else:
            dsar_action = DSARAction(
                action_type=getattr(action, "action_type", "compile_response"),
                silo_name=getattr(action, "silo_name", None),
                field_id=getattr(action, "field_id", None),
                decision=getattr(action, "decision", None),
                metadata=getattr(action, "metadata", {}),
            )

        # ── Resolve episode ID ────────────────────────────────────────────
        ep_id = (
            dsar_action.metadata.get("episode_id")
            or self._current_episode_id
        )

        if ep_id is None or ep_id not in _EPISODES:
            return DSARObservation(
                done=True,
                reward=0.0,
                metadata={"error": "No active episode. Call reset() first."},
                episode_id=ep_id or "",
                constraint_violated=False,
                error="No active episode. Call reset() first.",
            )

        episode = _EPISODES[ep_id]

        if episode.done:
            field_items = [FieldItem(**fi) for fi in episode.customer_record]
            return DSARObservation(
                done=True,
                reward=0.0,
                metadata={"episode_id": ep_id, "error": "Episode already finished."},
                episode_id=ep_id,
                task_id=episode.task_id,
                dsar_request=episode.dsar_text,
                customer_record=field_items,
                draft_response=episode.draft_response,
                audit_trail=episode.audit_trail,
                classified_fields=list(episode.classified_fields),
                silo_results=list(episode.queried_silos),
                steps_remaining=0,
                deadline_pressure=0.0,
                error="Episode already finished.",
            )

        # ── Increment step counter ────────────────────────────────────────
        episode.step_count += 1
        self._state.step_count = episode.step_count

        action_type = dsar_action.action_type
        error_msg = None

        # ── Snapshot pre-mutation state for reward calculation ────────────
        # IMPORTANT: We snapshot BEFORE mutation so the grader sees the
        # state at the time the action was taken, not after. This prevents
        # every valid action looking "redundant" to the reward function.
        pre_queried_silos = frozenset(episode.queried_silos)
        pre_classified_fields = frozenset(episode.classified_fields)

        # ── Process: query_silo ───────────────────────────────────────────
        if action_type == "query_silo":
            silo = dsar_action.silo_name
            if silo and silo not in episode.queried_silos and silo in CASE1_VALID_SILOS:
                episode.queried_silos.add(silo)
                desc = f"Successfully queried silo '{silo}'"
            elif silo and silo in episode.queried_silos:
                error_msg = f"Silo '{silo}' already queried this episode."
                desc = f"Redundant query: silo '{silo}' already queried"
            else:
                error_msg = f"Invalid silo name: '{silo}'. Valid silos: billing, crm"
                desc = f"Invalid silo: '{silo}'"

        # ── Process: classify_field ───────────────────────────────────────
        elif action_type == "classify_field":
            fid = dsar_action.field_id
            dec = dsar_action.decision

            if fid and fid in episode.ground_truth and fid not in episode.classified_fields:
                episode.classified_fields.add(fid)
                if dec == "disclose":
                    episode.draft_response[fid] = episode.values_lookup.get(fid)
                    if episode.ground_truth[fid] == "INTERNAL_ONLY":
                        episode.leaked_count += 1
                        desc = f"LEAK: classified internal field '{fid}' as disclose (leaked_count={episode.leaked_count})"
                    else:
                        desc = f"Correctly disclosed personal data field '{fid}'"
                else:
                    if episode.ground_truth[fid] == "REQUESTER_DATA":
                        desc = f"Incorrectly withheld personal data field '{fid}'"
                    else:
                        desc = f"Correctly withheld internal field '{fid}'"
            elif fid and fid in episode.classified_fields:
                error_msg = f"Field '{fid}' already classified this episode."
                desc = f"Redundant classification: '{fid}' already done"
            elif fid and fid not in episode.ground_truth:
                error_msg = f"Field '{fid}' does not exist in the record."
                desc = f"Unknown field: '{fid}'"
            else:
                error_msg = "classify_field requires field_id and decision parameters."
                desc = "Invalid classify_field call — missing parameters"
        else:
            desc = f"Unknown action type: '{action_type}'"

        # ── Compute step reward (using PRE-mutation snapshots) ────────────
        step_reward = compute_step_reward(
            action_type=action_type,
            silo_name=dsar_action.silo_name,
            field_id=dsar_action.field_id,
            decision=dsar_action.decision,
            ground_truth=episode.ground_truth,
            queried_silos=pre_queried_silos,
            classified_fields=pre_classified_fields,
            step_number=episode.step_count,
            task_id=episode.task_id,
        )
        episode.cumulative_reward += step_reward

        # ── Append AuditEntry (USP: structured transparent audit trail) ───
        episode.audit_trail.append(AuditEntry(
            step=episode.step_count,
            action=action_type,
            description=desc if "desc" in dir() else action_type,
            reward=step_reward,
        ))

        # ── Constraint gate: >2 leaks → immediate termination ─────────────
        if episode.leaked_count > 2 and not episode.done:
            episode.constraint_violated = True
            episode.done = True
            episode.audit_trail.append(AuditEntry(
                step=episode.step_count,
                action="constraint_violated",
                description=f"Episode terminated: leaked {episode.leaked_count} internal fields (limit is 2)",
                reward=0.0,
            ))
            field_items = [FieldItem(**fi) for fi in episode.customer_record]
            steps_remaining = max(0, MAX_STEPS - episode.step_count)
            return DSARObservation(
                done=True,
                reward=step_reward,
                metadata={
                    "episode_id": ep_id,
                    "task_id": episode.task_id,
                    "terminal_score": 0.0,
                    "constraint_violated": True,
                    "fields_leaked": episode.leaked_count,
                    "steps_used": episode.step_count,
                },
                episode_id=ep_id,
                task_id=episode.task_id,
                dsar_request=episode.dsar_text,
                customer_record=field_items,
                draft_response=episode.draft_response,
                audit_trail=episode.audit_trail,
                classified_fields=list(episode.classified_fields),
                silo_results=list(episode.queried_silos),
                steps_remaining=steps_remaining,
                deadline_pressure=steps_remaining / MAX_STEPS,
                constraint_violated=True,
                error="CONSTRAINT VIOLATED: Leaked more than 2 internal fields. Episode terminated.",
            )

        # ── Process: compile_response ─────────────────────────────────────
        if action_type == "compile_response":
            terminal_score = compute_terminal_score(
                draft_response=episode.draft_response,
                ground_truth=episode.ground_truth,
                queried_silos=episode.queried_silos,
                steps_used=episode.step_count,
                task_id=episode.task_id,
            )
            episode.done = True
            episode.audit_trail.append(AuditEntry(
                step=episode.step_count,
                action="compile_response",
                description=f"Response compiled. Terminal score: {terminal_score:.4f}",
                reward=terminal_score,
            ))
            episode.cumulative_reward += terminal_score

            field_items = [FieldItem(**fi) for fi in episode.customer_record]
            steps_remaining = max(0, MAX_STEPS - episode.step_count)
            return DSARObservation(
                done=True,
                reward=terminal_score,
                metadata={
                    "episode_id": ep_id,
                    "task_id": episode.task_id,
                    "terminal_score": terminal_score,
                    "cumulative_reward": round(episode.cumulative_reward, 4),
                    "fields_classified": len(episode.classified_fields),
                    "fields_leaked": episode.leaked_count,
                    "steps_used": episode.step_count,
                },
                episode_id=ep_id,
                task_id=episode.task_id,
                dsar_request=episode.dsar_text,
                customer_record=field_items,
                draft_response=episode.draft_response,
                audit_trail=episode.audit_trail,
                classified_fields=list(episode.classified_fields),
                silo_results=list(episode.queried_silos),
                steps_remaining=steps_remaining,
                deadline_pressure=steps_remaining / MAX_STEPS,
                constraint_violated=episode.constraint_violated,
            )
        elif action_type not in ["query_silo", "classify_field"]:
            error_msg = f"Unknown action type: '{action_type}'. Use: query_silo, classify_field, compile_response"

        # ── Check auto-termination at step limit ─────────────────────────
        steps_remaining = MAX_STEPS - episode.step_count
        is_done = steps_remaining <= 0

        if is_done and not episode.done:
            terminal_score = compute_terminal_score(
                draft_response=episode.draft_response,
                ground_truth=episode.ground_truth,
                queried_silos=episode.queried_silos,
                steps_used=episode.step_count,
                task_id=episode.task_id,
            )
            episode.done = True
            episode.cumulative_reward += terminal_score
            episode.audit_trail.append(AuditEntry(
                step=episode.step_count,
                action="auto_terminate",
                description=f"Auto-terminated at max steps. Terminal: {terminal_score:.4f}",
                reward=terminal_score,
            ))
            step_reward += terminal_score

        # ── Build observation ─────────────────────────────────────────────
        field_items = [FieldItem(**fi) for fi in episode.customer_record]
        return DSARObservation(
            done=is_done or episode.done,
            reward=step_reward,
            metadata={
                "episode_id": ep_id,
                "task_id": episode.task_id,
                "step_count": episode.step_count,
                "cumulative_reward": round(episode.cumulative_reward, 4),
            },
            episode_id=ep_id,
            task_id=episode.task_id,
            dsar_request=episode.dsar_text,
            customer_record=field_items,
            available_actions=["query_silo", "classify_field", "compile_response"],
            silo_results=list(episode.queried_silos),
            identity_verified=True,
            draft_response=episode.draft_response,
            audit_trail=episode.audit_trail,
            deadline_pressure=max(0.0, steps_remaining / MAX_STEPS),
            steps_remaining=max(0, steps_remaining),
            classified_fields=list(episode.classified_fields),
            constraint_violated=episode.constraint_violated,
            error=error_msg,
        )

    @property
    def state(self) -> State:
        """Get the current environment state for debugging/validation.

        Used by openenv validate and debug tooling — never called by inference.py.
        """
        return self._state
