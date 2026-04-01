"""
DSAR Compliance Environment — Pydantic models for actions, observations, and field items.

Action and Observation inherit from OpenEnv's Pydantic BaseModel types.
The Observation base class already provides `done`, `reward`, and `metadata` fields.

Note: The environment returns Observation subclasses with `done` and `reward` set.
The framework's HTTPEnvServer serializes these into StepResponse/ResetResponse.
If a future OpenEnv version changes this contract, the environment logic in
dsar_environment.py should be updated to match — treat this as version-sensitive.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


# ─── AuditEntry: per-step action log ────────────────────────────────────────────

class AuditEntry(BaseModel):
    """A single entry in the episode audit trail.

    Every step appends one AuditEntry with a plain-English description of
    what happened — one of the confirmed USPs for judging transparency.
    """

    step: int = Field(..., description="Step number (1-indexed) when this entry was recorded")
    action: str = Field(..., description="Action type: query_silo, classify_field, compile_response")
    description: str = Field(..., description="Plain-English description of what happened")
    reward: float = Field(default=0.0, description="Immediate reward received for this action")


# ─── FieldItem: rich per-field metadata ───────────────────────────────────────

class FieldItem(BaseModel):
    """A single field in the customer record with rich metadata.

    This structured representation provides the agent with semantic context
    about each field — not just the raw column name and value, but also
    where it came from, what type of data it is, and a human-readable
    description. This makes the classification task genuinely semantic
    rather than simple column-name guessing.
    """

    field_id: str = Field(
        ...,
        description="Unique identifier for this field (e.g., 'full_name', 'risk_score')",
    )
    field_name: str = Field(
        ...,
        description="Human-readable name (e.g., 'Full Name', 'Risk Score')",
    )
    field_value: Any = Field(
        ...,
        description="The actual value of this field",
    )
    source_silo: str = Field(
        ...,
        description="Which data silo this field came from ('billing', 'crm', 'both')",
    )
    datatype: str = Field(
        ...,
        description="Data type category ('personal_identifier', 'financial', 'behavioral', 'operational', 'infrastructure', 'analytical')",
    )
    field_description: str = Field(
        ...,
        description="Human-readable description of what this field represents",
    )


# ─── DSARAction ───────────────────────────────────────────────────────────────

class DSARAction(Action):
    """Action sent by the agent to the DSAR environment.

    Three action types are supported:
    - query_silo: Query a data silo for records (billing, crm)
    - classify_field: Classify a field as disclose or withhold
    - compile_response: Finalize the response and trigger terminal grading
    """

    action_type: str = Field(
        ...,
        description="Type of action: 'query_silo', 'classify_field', or 'compile_response'",
    )
    silo_name: Optional[str] = Field(
        default=None,
        description="For query_silo: name of the silo to query ('billing' or 'crm')",
    )
    field_id: Optional[str] = Field(
        default=None,
        description="For classify_field: the field_id from the customer record",
    )
    decision: Optional[str] = Field(
        default=None,
        description="For classify_field: 'disclose' or 'withhold'",
    )


# ─── DSARObservation ──────────────────────────────────────────────────────────

class DSARObservation(Observation):
    """Observation returned by the DSAR environment.

    Inherits from Observation which provides:
    - done: bool (whether episode has terminated)
    - reward: float | None (reward signal from last action)
    - metadata: Dict[str, Any] (additional metadata)
    """

    episode_id: str = Field(
        default="",
        description="The unique UUID for this episode",
    )
    task_id: str = Field(
        default="task_easy",
        description="Current task identifier",
    )
    dsar_request: str = Field(
        default="",
        description="The DSAR ticket text describing the request",
    )
    customer_record: List[FieldItem] = Field(
        default_factory=list,
        description="Structured list of all fields in the data subject's record, with metadata",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: ["query_silo", "classify_field", "compile_response"],
        description="List of available action types",
    )
    silo_results: List[str] = Field(
        default_factory=list,
        description="Names of silos that have been queried this episode",
    )
    identity_verified: bool = Field(
        default=True,
        description="Whether the requester's identity has been verified",
    )
    draft_response: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fields the agent has chosen to disclose so far",
    )
    audit_trail: List[AuditEntry] = Field(
        default_factory=list,
        description="Ordered log of AuditEntry objects — one per step, with plain-English descriptions",
    )
    deadline_pressure: float = Field(
        default=1.0,
        description="Normalised time pressure: 1.0 (start) → 0.0 (deadline). Calculated as steps_remaining/max_steps",
    )
    steps_remaining: int = Field(
        default=30,
        description="Number of steps remaining in this episode",
    )
    classified_fields: List[str] = Field(
        default_factory=list,
        description="Field IDs that have already been classified this episode",
    )
    constraint_violated: bool = Field(
        default=False,
        description="True when agent has leaked more than 2 internal fields — triggers immediate episode termination",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid",
    )
