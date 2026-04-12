"""Custom Gradio UI for the AutoDSAR Hugging Face Space."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import gradio as gr


TASKS: Dict[str, Dict[str, str]] = {
    "task_easy": {
        "label": "Easy - structured disclosure",
        "mission": "Query billing and CRM, then decide which customer fields can be disclosed.",
        "starter": '{"action_type":"query_silo","silo_name":"billing"}',
    },
    "task_medium": {
        "label": "Medium - identity and ticket redaction",
        "mission": "Verify identity proportionately, then redact support-ticket sentences.",
        "starter": '{"action_type":"query_silo","silo_name":"crm"}',
    },
    "task_adversarial_identity": {
        "label": "Adversarial - spoof review",
        "mission": "Gather evidence and decide whether to verify a genuine requester or flag spoofing.",
        "starter": '{"action_type":"query_silo","silo_name":"crm"}',
    },
    "task_hard": {
        "label": "Hard - Slack compliance triage",
        "mission": "Process Slack messages while escalating special-category health traps.",
        "starter": '{"action_type":"process_message","msg_id":"<message_id>","action_label":"exclude"}',
    },
    "task_breach_embedded": {
        "label": "Breach - DSAR plus notification workflow",
        "mission": "Detect hidden breach signals, notify regulator, notify requester, then compile safely.",
        "starter": '{"action_type":"flag_breach_signal","reason":"The request contains a potential unauthorized disclosure signal."}',
    },
}

DIFFICULTIES = ["low", "medium", "high"]

CSS = """
.gradio-container {
  color: #202124;
  background:
    linear-gradient(180deg, rgba(255,255,255,.94), rgba(247,250,249,.96)),
    url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1800&q=80");
  background-size: cover;
  background-position: center top;
  background-attachment: fixed;
}
.autodsar-hero {
  background: rgba(255, 255, 255, .92);
  border: 1px solid rgba(32,33,36,.18);
  border-radius: 8px;
  padding: 28px;
}
.autodsar-kicker {
  color: #0f766e;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}
.autodsar-hero h1 {
  margin: 8px 0 10px;
  color: #202124;
}
.autodsar-hero p {
  color: #3f3f46;
  max-width: 920px;
}
.autodsar-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 18px;
}
.autodsar-metric {
  border-left: 4px solid #e11d48;
  background: #ffffff;
  border-radius: 8px;
  padding: 12px;
}
.autodsar-metric strong {
  display: block;
  color: #111827;
}
.autodsar-metric span {
  color: #52525b;
}
.autodsar-note {
  background: #f0fdfa;
  border: 1px solid #99f6e4;
  border-radius: 8px;
  padding: 12px 14px;
}
button.primary {
  background: #0f766e !important;
  border-radius: 8px !important;
}
textarea, input, select {
  border-radius: 8px !important;
}
@media (max-width: 760px) {
  .autodsar-strip {
    grid-template-columns: 1fr;
  }
}
"""


def _pretty(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def _task_id(label: str) -> str:
    for task_id, info in TASKS.items():
        if info["label"] == label:
            return task_id
    return "task_easy"


def _summary(obs: Dict[str, Any]) -> str:
    fields = {
        "episode_id": obs.get("episode_id"),
        "task_id": obs.get("task_id"),
        "workflow_state": obs.get("workflow_state"),
        "compliance_state": obs.get("current_compliance_state"),
        "compile_ready": obs.get("compile_ready"),
        "steps_remaining": obs.get("steps_remaining"),
        "step_safety_cost": obs.get("step_safety_cost"),
        "episode_safety_cost": obs.get("episode_safety_cost"),
        "last_action_outcome": obs.get("last_action_outcome"),
        "required_followup_action": obs.get("required_followup_action"),
        "available_actions": obs.get("available_actions", []),
    }
    return _pretty(fields)


def _audit(obs: Dict[str, Any]) -> str:
    trail: List[Dict[str, Any]] = obs.get("audit_trail") or []
    events: List[Dict[str, Any]] = obs.get("constraint_events") or []
    return _pretty({"audit_trail": trail[-8:], "constraint_events": events[-8:]})


def _visible_payload(obs: Dict[str, Any]) -> str:
    keys = [
        "dsar_request",
        "customer_record",
        "silo_results",
        "tickets",
        "slack_export",
        "messages_pending",
        "breach_signal_context",
        "breach_scope_fields",
        "draft_response",
        "terminal_details",
        "error",
    ]
    return _pretty({key: obs.get(key) for key in keys if obs.get(key) not in (None, [], {})})


def _task_markdown(label: str) -> str:
    info = TASKS[_task_id(label)]
    return f"""
### Mission
{info["mission"]}

Starter action:

```json
{info["starter"]}
```
"""


def build_autodsar_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Return a custom Gradio Blocks app for OpenEnv's custom tab."""

    task_labels = [info["label"] for info in TASKS.values()]

    async def reset(task_label: str, difficulty: str, seed: int):
        payload = {
            "task_id": _task_id(task_label),
            "difficulty_tier": difficulty,
            "seed": int(seed),
        }
        result = await web_manager.reset_environment(payload)
        obs = result.get("observation", {})
        return (
            _task_markdown(task_label),
            _summary(obs),
            _visible_payload(obs),
            _audit(obs),
            TASKS[_task_id(task_label)]["starter"],
        )

    async def step(action_json: str):
        try:
            action = json.loads(action_json)
        except json.JSONDecodeError as exc:
            error = {"error": f"Invalid JSON: {exc}"}
            return _pretty(error), _pretty({}), _pretty(error)
        result = await web_manager.step_environment(action)
        obs = result.get("observation", {})
        return _summary(obs), _visible_payload(obs), _audit(obs)

    with gr.Blocks(title="AutoDSAR") as demo:
        gr.HTML(
            f"""
            <style>{CSS}</style>
            <section class="autodsar-hero">
              <div class="autodsar-kicker">Privacy operations RL benchmark</div>
              <h1>AutoDSAR</h1>
              <p>
                Train agents on the hard parts of data-subject access work: evidence gathering,
                proportional identity review, redaction, special-category escalation, breach
                detection, ordered notifications, and recovery after unsafe moves.
              </p>
              <div class="autodsar-strip">
                <div class="autodsar-metric"><strong>5 tasks</strong><span>Structured disclosure to breach response.</span></div>
                <div class="autodsar-metric"><strong>Reactive state</strong><span>Unsafe actions change the workflow.</span></div>
                <div class="autodsar-metric"><strong>Safety cost</strong><span>Reward and compliance harm are separate.</span></div>
                <div class="autodsar-metric"><strong>Reproducible</strong><span>Seeds, tiers, graders, and trajectories.</span></div>
              </div>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(task_labels, value=task_labels[0], label="Task")
                difficulty = gr.Dropdown(DIFFICULTIES, value="medium", label="Difficulty tier")
                seed = gr.Number(value=42, precision=0, label="Seed")
                reset_button = gr.Button("Start episode", variant="primary")
                mission = gr.Markdown(_task_markdown(task_labels[0]), elem_classes=["autodsar-note"])
                action_json = gr.Code(
                    value=TASKS["task_easy"]["starter"],
                    language="json",
                    label="Action JSON",
                    lines=8,
                )
                step_button = gr.Button("Run action", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Code(label="Workflow and safety state", language="json", lines=14)
                payload = gr.Code(label="Visible observation payload", language="json", lines=18)
                audit = gr.Code(label="Audit trail and constraint events", language="json", lines=12)

        gr.Markdown(
            """
### What to try
Start with `task_easy`, run `query_silo` for `billing` and `crm`, then classify fields as `disclose` or `withhold`.
For the harder tasks, watch `workflow_state`, `current_compliance_state`, `required_followup_action`,
`step_safety_cost`, and `constraint_events`; those are the signals that make AutoDSAR more than a flat classifier.
            """
        )

        task.change(_task_markdown, inputs=task, outputs=mission)
        reset_button.click(
            reset,
            inputs=[task, difficulty, seed],
            outputs=[mission, summary, payload, audit, action_json],
        )
        step_button.click(step, inputs=action_json, outputs=[summary, payload, audit])

    return demo
