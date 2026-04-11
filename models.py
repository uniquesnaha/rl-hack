"""
DSAR Compliance Environment - Pydantic models for actions, observations, and
ticket/field items.

Action and Observation inherit from OpenEnv's Pydantic BaseModel types.
The Observation base class already provides `done`, `reward`, and `metadata`
fields.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class AuditEntry(BaseModel):
    """A single entry in the episode audit trail."""

    step: int = Field(..., description="Step number (1-indexed) when this entry was recorded")
    action: str = Field(..., description="Action type executed at this step")
    description: str = Field(..., description="Plain-English description of what happened")
    reward: float = Field(default=0.0, description="Immediate reward received for this action")


class FieldItem(BaseModel):
    """A single structured field in the customer record."""

    field_id: str = Field(
        ...,
        description="Unique identifier for this field (for example 'full_name')",
    )
    field_name: str = Field(
        ...,
        description="Human-readable name of the field",
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
        description="Semantic datatype category for the field",
    )
    field_description: str = Field(
        ...,
        description="Human-readable description of what this field represents",
    )


class TicketSentenceItem(BaseModel):
    """A single sentence within a support ticket thread."""

    sentence_index: int = Field(..., description="Sentence index unique within the ticket")
    speaker: str = Field(..., description="Speaker role: customer or support")
    text: str = Field(..., description="Sentence text shown to the agent")


class TicketMessageItem(BaseModel):
    """A single message within a support ticket thread."""

    message_index: int = Field(..., description="Message index unique within the ticket")
    speaker: str = Field(..., description="Speaker role: customer or support")
    text: str = Field(..., description="Full message text")
    sentences: List[TicketSentenceItem] = Field(
        default_factory=list,
        description="Sentence-indexed fragments for sentence-level redaction decisions",
    )


class TicketItem(BaseModel):
    """A support ticket thread visible during Case 2 redaction."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    category: str = Field(..., description="Ticket archetype/category")
    messages: List[TicketMessageItem] = Field(
        default_factory=list,
        description="Ordered support-ticket messages with sentence indexing",
    )


class SlackSentenceItem(BaseModel):
    """A single sentence fragment within a Slack message."""

    sentence_idx: int = Field(..., description="Sentence index unique within the Slack message")
    text: str = Field(..., description="Sentence text shown to the agent")


class SlackMessageItem(BaseModel):
    """A single Slack export message visible during Case 3 triage."""

    msg_id: str = Field(..., description="Stable message identifier for processing actions")
    user: str = Field(..., description="Slack user ID for the message author")
    text: str = Field(..., description="Full Slack message text")
    ts: str = Field(..., description="Slack-style timestamp string")
    thread_ts: Optional[str] = Field(
        default=None,
        description="Parent thread timestamp if this message is part of a thread",
    )
    subtype: Optional[str] = Field(
        default=None,
        description="Slack message subtype, e.g. bot_message",
    )
    sentences: List[SlackSentenceItem] = Field(
        default_factory=list,
        description="Deterministic sentence fragments for sentence-level redaction decisions",
    )


class ConstraintEventItem(BaseModel):
    """A structured safety / compliance cost event emitted by the environment."""

    step: int = Field(..., description="Step number when the safety event was triggered")
    event_type: str = Field(..., description="Stable safety-cost event type identifier")
    cost: float = Field(..., description="Safety cost assigned to this event")
    message: str = Field(..., description="Human-readable explanation of the safety event")


class DSARAction(Action):
    """Action sent by the agent to the DSAR environment."""

    action_type: str = Field(
        ...,
        description=(
            "Type of action: 'query_silo', 'classify_field', "
            "'verify_identity', 'redact_span', 'process_message', "
            "'redact_sentence', 'escalate_with_reason', "
            "'flag_adversarial', 'flag_breach_signal', 'notify_regulator', "
            "'notify_requester', 'file_remediation_note', 'justify_verification_method', "
            "'file_redaction_remediation', or 'compile_response'"
        ),
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
        description=(
            "For classify_field: 'disclose' or 'withhold'. "
            "For redact_span: 'keep' or 'redact'."
        ),
    )
    verification_method: Optional[str] = Field(
        default=None,
        description=(
            "For verify_identity: one of transaction_date, account_reference, "
            "registered_postcode, passport_copy, or photo_id"
        ),
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="For redact_span: ticket identifier",
    )
    sentence_index: Optional[int] = Field(
        default=None,
        description="For redact_span: sentence index within the ticket",
    )
    msg_id: Optional[str] = Field(
        default=None,
        description="For process_message/redact_sentence/escalate_with_reason: Slack message ID",
    )
    action_label: Optional[str] = Field(
        default=None,
        description=(
            "For process_message: one of disclose, partial_redact, exclude, or escalate"
        ),
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "For escalate_with_reason: written legal justification. "
            "For remediation/justification actions: short explanation or note text."
        ),
    )
    reason_code: Optional[str] = Field(
        default=None,
        description=(
            "For escalate_with_reason: structured escalation code such as "
            "special_category_health_data"
        ),
    )


class DSARObservation(Observation):
    """Observation returned by the DSAR environment."""

    episode_id: str = Field(default="", description="The unique UUID for this episode")
    task_id: str = Field(default="task_easy", description="Current task identifier")
    dsar_request: str = Field(default="", description="The DSAR ticket text describing the request")
    customer_record: List[FieldItem] = Field(
        default_factory=list,
        description="Structured list of currently visible fields in the data subject's record",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: ["query_silo", "classify_field", "compile_response"],
        description="List of available action types at this point in the episode",
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
        description="Current draft disclosure/redaction output for the episode",
    )
    audit_trail: List[AuditEntry] = Field(
        default_factory=list,
        description="Ordered log of audit entries, one per step",
    )
    deadline_pressure: float = Field(
        default=1.0,
        description="Normalized time pressure: 1.0 at start to 0.0 at deadline",
    )
    steps_remaining: int = Field(
        default=30,
        description="Number of steps remaining in this episode",
    )
    classified_fields: List[str] = Field(
        default_factory=list,
        description="Case 1 field IDs that have already been classified",
    )
    constraint_violated: bool = Field(
        default=False,
        description="True when a hard environment constraint was violated",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid",
    )

    # Case 2 specific fields
    phase: str = Field(
        default="classification",
        description="Current episode phase: classification, identity, or redaction",
    )
    identity_confidence: float = Field(
        default=1.0,
        description="Current identity confidence score used in Case 2 verification",
    )
    identity_threshold: float = Field(
        default=1.0,
        description="Identity confidence threshold required to unlock redaction",
    )
    submitted_identity: Dict[str, Any] = Field(
        default_factory=dict,
        description="Identity details supplied by the requester",
    )
    internal_identity: Dict[str, Any] = Field(
        default_factory=dict,
        description="Internal identity evidence currently visible to the agent",
    )
    tickets: List[TicketItem] = Field(
        default_factory=list,
        description="Support ticket corpus visible during Case 2 redaction",
    )
    processed_sentences: Dict[str, Dict[int, str]] = Field(
        default_factory=dict,
        description="Sentence-level Case 2 decisions keyed by ticket and sentence index",
    )
    pending_sentence_count: int = Field(
        default=0,
        description="Number of Case 2 ticket sentences still awaiting a keep/redact decision",
    )
    total_sentence_count: int = Field(
        default=0,
        description="Total number of Case 2 ticket sentences in the current episode",
    )
    completion_coverage: float = Field(
        default=0.0,
        description="Fraction of Case 2 ticket sentences already processed",
    )
    compile_ready: bool = Field(
        default=False,
        description="Whether the episode has satisfied the requirements to call compile_response",
    )
    terminal_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Terminal metrics and failure summaries for completed episodes",
    )

    # Case 3 specific fields
    slack_export: List[SlackMessageItem] = Field(
        default_factory=list,
        description="Candidate Slack messages surfaced for Case 3 compliance triage",
    )
    users_json: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Visible Slack user mapping keyed by user ID",
    )
    processed_messages: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Case 3 message-level decisions keyed by Slack message ID",
    )
    escalation_log: Dict[str, str] = Field(
        default_factory=dict,
        description="Case 3 escalation reasons keyed by Slack message ID",
    )
    escalation_reason_codes: Dict[str, str] = Field(
        default_factory=dict,
        description="Case 3 structured escalation reason codes keyed by Slack message ID",
    )
    messages_pending: List[str] = Field(
        default_factory=list,
        description="Case 3 message IDs that still need a process_message decision",
    )
    sentences_pending: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Case 3 unresolved sentence indices for messages awaiting sentence decisions",
    )
    breach_detected: bool = Field(
        default=False,
        description="Whether the embedded breach concern has been flagged correctly in the current episode",
    )
    regulator_notified: bool = Field(
        default=False,
        description="Whether the regulator notification step has been completed",
    )
    requester_notified: bool = Field(
        default=False,
        description="Whether the data-subject notification step has been completed",
    )
    breach_scope_fields: List[str] = Field(
        default_factory=list,
        description="Fields believed to be in scope of the breach, visible after a correct breach flag",
    )
    breach_signal_context: Optional[str] = Field(
        default=None,
        description="Visible breach concern text surfaced from the DSAR when applicable",
    )
    current_compliance_state: str = Field(
        default="clean",
        description="Current reactive compliance risk state for the episode",
    )
    required_followup_action: Optional[str] = Field(
        default=None,
        description="Action required before compile_response becomes valid again",
    )
    worsened_transitions: int = Field(
        default=0,
        description="Number of times the episode transitioned into a worse compliance state",
    )
    recovery_actions_taken: int = Field(
        default=0,
        description="Number of successful recovery actions taken in the episode",
    )
    last_action_outcome: str = Field(
        default="no_effect",
        description="Outcome class of the last action: progress, worsened, recovery, or no_effect",
    )
    state_change_message: Optional[str] = Field(
        default=None,
        description="Human-readable message describing the latest compliance state transition",
    )
    workflow_state: str = Field(
        default="classification",
        description="Explicit workflow-state identifier derived from the task phase and recovery status",
    )
    step_safety_cost: float = Field(
        default=0.0,
        description="Safety cost generated by the most recent action",
    )
    episode_safety_cost: float = Field(
        default=0.0,
        description="Cumulative safety cost accrued over the episode",
    )
    constraint_events: List[ConstraintEventItem] = Field(
        default_factory=list,
        description="Structured list of safety / compliance constraint events triggered so far",
    )
