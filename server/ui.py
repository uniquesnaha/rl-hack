"""Custom Gradio UI for the AutoDSAR Hugging Face Space."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import RedirectResponse
from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Action, Environment, Observation
from openenv.core.env_server.web_interface import (
    WebInterfaceManager,
    get_quick_start_markdown,
    load_environment_metadata,
)


TASKS: Dict[str, Dict[str, str]] = {
    "task_easy": {
        "label": "Easy - structured disclosure",
        "short": "Structured field disclosure",
        "mission": "Query billing and CRM, then classify each field as disclose or withhold.",
        "starter": '{"action_type":"query_silo","silo_name":"billing"}',
        "next": '{"action_type":"query_silo","silo_name":"crm"}',
        "risk": "Internal-only fields hide inside ordinary customer records.",
    },
    "task_medium": {
        "label": "Medium - identity and ticket redaction",
        "short": "Identity plus redaction",
        "mission": "Verify identity proportionately, then redact support-ticket sentences.",
        "starter": '{"action_type":"query_silo","silo_name":"crm"}',
        "next": '{"action_type":"justify_verification_method","verification_method":"account_reference","reason":"Use proportionate account evidence before ticket redaction."}',
        "risk": "Over-verification and third-party support-ticket leaks both count against the agent.",
    },
    "task_adversarial_identity": {
        "label": "Adversarial - spoof review",
        "short": "Spoof-resistant identity",
        "mission": "Gather evidence, verify genuine requesters, or flag adversarial behavior.",
        "starter": '{"action_type":"query_silo","silo_name":"crm"}',
        "next": '{"action_type":"flag_adversarial","reason":"Evidence is inconsistent with the requester identity."}',
        "risk": "Near-miss identities, urgency pressure, stale evidence, and borrowed details.",
    },
    "task_hard": {
        "label": "Hard - Slack compliance triage",
        "short": "Slack triage and escalation",
        "mission": "Process Slack messages, partial-redact where needed, and escalate health traps.",
        "starter": '{"action_type":"process_message","msg_id":"<message_id>","action_label":"exclude"}',
        "next": '{"action_type":"escalate_with_reason","msg_id":"<message_id>","reason_code":"special_category_health_data","reason":"The message contains special-category health data requiring legal review."}',
        "risk": "Special-category health content is mixed with ordinary operational chat.",
    },
    "task_breach_embedded": {
        "label": "Breach - DSAR plus notification workflow",
        "short": "Hidden breach response",
        "mission": "Detect a hidden breach signal, notify regulator, notify requester, then compile.",
        "starter": '{"action_type":"flag_breach_signal","reason":"The request contains a possible unauthorized disclosure signal."}',
        "next": '{"action_type":"notify_regulator","reason":"Breach signal confirmed; regulator notice must precede requester notice."}',
        "risk": "Late breach detection and out-of-order notifications reduce the terminal score.",
    },
}

DIFFICULTIES = ["low", "medium", "high"]

CSS = """
.gradio-container {
  color: #f8fafc !important;
  background:
    linear-gradient(180deg, rgba(3,7,18,.92), rgba(15,23,42,.96)),
    url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1800&q=80");
  background-size: cover;
  background-position: center top;
  background-attachment: fixed;
}
.gradio-container, .gradio-container * {
  color-scheme: dark;
}
.gradio-container label,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container p,
.gradio-container li,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: #f8fafc !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container .cm-editor,
.gradio-container .cm-scroller,
.gradio-container .cm-content,
.gradio-container pre,
.gradio-container code {
  background: #020617 !important;
  color: #e2e8f0 !important;
  border-color: #334155 !important;
}
.autodsar-shell {
  max-width: 1180px;
  margin: 0 auto;
}
.autodsar-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 0 18px;
}
.autodsar-brand {
  font-weight: 900;
  font-size: 30px;
  color: #f8fafc;
}
.autodsar-badge {
  display: inline-block;
  border: 1px solid #2dd4bf;
  border-radius: 8px;
  padding: 5px 9px;
  background: rgba(20,184,166,.14);
  color: #5eead4;
  font-size: 13px;
  font-weight: 800;
}
.autodsar-hero {
  min-height: 460px;
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, .85fr);
  gap: 24px;
  align-items: center;
  padding: 30px 0 34px;
}
.autodsar-copy {
  background: rgba(15,23,42,.9);
  border: 1px solid rgba(148,163,184,.38);
  border-radius: 8px;
  padding: 34px;
  box-shadow: 0 22px 70px rgba(2,6,23,.48);
}
.autodsar-kicker {
  color: #5eead4;
  font-weight: 900;
  letter-spacing: 0;
  text-transform: uppercase;
}
.autodsar-copy h1 {
  margin: 10px 0 12px;
  color: #f8fafc;
  font-size: 68px;
  line-height: 1.02;
}
.autodsar-copy p, .autodsar-copy li {
  color: #cbd5e1;
  font-size: 20px;
}
.autodsar-board {
  background: rgba(2,6,23,.88);
  border: 1px solid rgba(148,163,184,.38);
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 22px 70px rgba(2,6,23,.52);
}
.autodsar-ladder {
  display: grid;
  gap: 10px;
}
.autodsar-step {
  display: grid;
  grid-template-columns: 38px 1fr 86px;
  align-items: center;
  gap: 10px;
  padding: 12px;
  background: rgba(15,23,42,.92);
  border: 1px solid #334155;
  border-radius: 8px;
}
.autodsar-dot {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  background: #0f766e;
  color: #ffffff;
  font-weight: 900;
}
.autodsar-step:nth-child(2) .autodsar-dot { background: #2563eb; }
.autodsar-step:nth-child(3) .autodsar-dot { background: #dc2626; }
.autodsar-step:nth-child(4) .autodsar-dot { background: #ca8a04; }
.autodsar-step:nth-child(5) .autodsar-dot { background: #7c3aed; }
.autodsar-step strong {
  display: block;
  color: #f8fafc;
}
.autodsar-step span {
  color: #94a3b8;
  font-size: 13px;
}
.autodsar-risk {
  color: #fb7185;
  font-weight: 800;
  text-align: right;
}
.autodsar-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.autodsar-metric {
  border-left: 4px solid #fb7185;
  background: rgba(15,23,42,.9);
  border-top: 1px solid #334155;
  border-right: 1px solid #334155;
  border-bottom: 1px solid #334155;
  border-radius: 8px;
  padding: 14px;
}
.autodsar-metric strong {
  display: block;
  color: #f8fafc;
  font-size: 20px;
}
.autodsar-metric span {
  color: #cbd5e1;
  font-size: 13px;
}
.autodsar-band {
  background: rgba(15,23,42,.92);
  border: 1px solid rgba(148,163,184,.38);
  border-radius: 8px;
  padding: 22px;
  margin: 14px 0;
}
.autodsar-chart {
  display: grid;
  gap: 10px;
}
.autodsar-bar {
  display: grid;
  grid-template-columns: 172px 1fr 72px;
  gap: 10px;
  align-items: center;
  color: #cbd5e1;
}
.autodsar-fill {
  height: 14px;
  border-radius: 8px;
  background: linear-gradient(90deg, #2dd4bf, #60a5fa, #fb7185);
}
.autodsar-note {
  background: rgba(20,184,166,.14);
  border: 1px solid #2dd4bf;
  border-radius: 8px;
  padding: 12px 14px;
}
.autodsar-alert {
  background: rgba(251,146,60,.13);
  border: 1px solid #fb923c;
  border-radius: 8px;
  padding: 12px 14px;
}
button.primary {
  background: #14b8a6 !important;
  color: #04111d !important;
  font-weight: 900 !important;
  border-radius: 8px !important;
}
button.secondary {
  background: #1e293b !important;
  color: #f8fafc !important;
  border: 1px solid #475569 !important;
  border-radius: 8px !important;
}
textarea, input, select {
  border-radius: 8px !important;
}
@media (max-width: 900px) {
  .autodsar-hero, .autodsar-strip {
    grid-template-columns: 1fr;
  }
  .autodsar-copy h1 {
    font-size: 44px;
  }
  .autodsar-step, .autodsar-bar {
    grid-template-columns: 1fr;
  }
  .autodsar-risk {
    text-align: left;
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
        "difficulty_tier": obs.get("metadata", {}).get("difficulty_tier")
        if isinstance(obs.get("metadata"), dict)
        else None,
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
    return _pretty({"audit_trail": trail[-10:], "constraint_events": events[-10:]})


def _visible_payload(obs: Dict[str, Any]) -> str:
    keys = [
        "dsar_request",
        "customer_record",
        "silo_results",
        "submitted_identity",
        "internal_identity",
        "tickets",
        "processed_sentences",
        "slack_export",
        "users_json",
        "messages_pending",
        "sentences_pending",
        "breach_signal_context",
        "breach_scope_fields",
        "regulator_notified",
        "requester_notified",
        "draft_response",
        "terminal_details",
        "error",
    ]
    return _pretty({key: obs.get(key) for key in keys if obs.get(key) not in (None, [], {})})


def _task_markdown(label: str) -> str:
    task_id = _task_id(label)
    info = TASKS[task_id]
    return f"""
### {info["short"]}
{info["mission"]}

Risk focus: {info["risk"]}

Starter action:

```json
{info["starter"]}
```
"""


def _task_template(label: str, template_kind: str) -> str:
    info = TASKS[_task_id(label)]
    return info["next"] if template_kind == "Next useful action" else info["starter"]


def _home_html() -> str:
    ladder = "".join(
        f"""
        <div class="autodsar-step">
          <div class="autodsar-dot">{idx}</div>
          <div><strong>{info["short"]}</strong><span>{task_id}</span></div>
          <div class="autodsar-risk">{info["risk"].split()[0]}</div>
        </div>
        """
        for idx, (task_id, info) in enumerate(TASKS.items(), start=1)
    )
    return f"""
    <style>{CSS}</style>
    <div class="autodsar-shell">
      <div class="autodsar-nav">
        <div class="autodsar-brand">AutoDSAR</div>
        <div class="autodsar-badge">OpenEnv privacy RL</div>
      </div>
      <section class="autodsar-hero">
        <div class="autodsar-copy">
          <div class="autodsar-kicker">Data-subject access request benchmark</div>
          <h1>Train agents to handle privacy work without cutting corners.</h1>
          <p>
            AutoDSAR turns DSAR operations into sequential RL tasks: evidence gathering,
            proportional identity review, redaction, escalation, breach notification, and
            recovery after unsafe moves.
          </p>
          <ul>
            <li>Five deterministic tasks from structured disclosure to breach response.</li>
            <li>Separate task reward and compliance safety cost.</li>
            <li>Workflow states and compile gates that force process discipline.</li>
          </ul>
        </div>
        <div class="autodsar-board">
          <h2>Benchmark ladder</h2>
          <div class="autodsar-ladder">{ladder}</div>
        </div>
      </section>
      <section class="autodsar-strip">
        <div class="autodsar-metric"><strong>5</strong><span>task families</span></div>
        <div class="autodsar-metric"><strong>3</strong><span>difficulty tiers</span></div>
        <div class="autodsar-metric"><strong>2</strong><span>reward and safety channels</span></div>
        <div class="autodsar-metric"><strong>1</strong><span>ordered breach workflow</span></div>
      </section>
    </div>
    """


def _guide_html() -> str:
    return f"""
    <style>{CSS}</style>
    <div class="autodsar-shell">
      <div class="autodsar-band">
        <h1>Benchmark guide</h1>
        <p>
          AutoDSAR is not a one-shot classifier. The agent must complete a workflow while
          minimizing safety cost. Watch the fields below while you train or debug a policy.
        </p>
      </div>
      <div class="autodsar-band">
        <h2>Difficulty and hidden-state pressure</h2>
        <div class="autodsar-chart">
          <div class="autodsar-bar"><strong>task_easy</strong><div class="autodsar-fill" style="width: 28%"></div><span>fields</span></div>
          <div class="autodsar-bar"><strong>task_medium</strong><div class="autodsar-fill" style="width: 48%"></div><span>identity</span></div>
          <div class="autodsar-bar"><strong>adversarial_identity</strong><div class="autodsar-fill" style="width: 66%"></div><span>spoofing</span></div>
          <div class="autodsar-bar"><strong>task_hard</strong><div class="autodsar-fill" style="width: 78%"></div><span>triage</span></div>
          <div class="autodsar-bar"><strong>breach_embedded</strong><div class="autodsar-fill" style="width: 92%"></div><span>breach</span></div>
        </div>
      </div>
      <div class="autodsar-band">
        <h2>Signals to track</h2>
        <p><code>workflow_state</code> tells you where the process is. <code>current_compliance_state</code> tells you whether the agent made the situation worse. <code>required_followup_action</code> tells you how to recover. <code>step_safety_cost</code> and <code>episode_safety_cost</code> separate compliance harm from reward.</p>
      </div>
      <div class="autodsar-band">
        <h2>Breach task order</h2>
        <p><code>flag_breach_signal</code> -> <code>notify_regulator</code> -> <code>notify_requester</code> -> <code>compile_response</code>. Late detection or leaked internal fields can still hurt the terminal score.</p>
      </div>
    </div>
    """


def build_autodsar_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Return the custom AutoDSAR Gradio app."""

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
        with gr.Tabs(selected="home"):
            with gr.Tab("Home", id="home"):
                gr.HTML(_home_html())
                gr.Markdown(
                    """
Use the tabs above to open the live training workbench or the benchmark guide. The workbench is connected to the same OpenEnv reset and step endpoints used by the API.
                    """,
                    elem_classes=["autodsar-alert"],
                )

            with gr.Tab("Training Workbench", id="training"):
                gr.HTML(
                    f"""
                    <style>{CSS}</style>
                    <div class="autodsar-shell">
                      <div class="autodsar-nav">
                        <div class="autodsar-brand">Training Workbench</div>
                        <div class="autodsar-badge">Live OpenEnv episode</div>
                      </div>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configure episode")
                        task = gr.Dropdown(task_labels, value=task_labels[0], label="Task")
                        difficulty = gr.Dropdown(DIFFICULTIES, value="medium", label="Difficulty tier")
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        reset_button = gr.Button("Start episode", variant="primary")
                        mission = gr.Markdown(_task_markdown(task_labels[0]), elem_classes=["autodsar-note"])
                        template_kind = gr.Radio(
                            ["Starter action", "Next useful action"],
                            value="Starter action",
                            label="Action template",
                        )
                        action_json = gr.Code(
                            value=TASKS["task_easy"]["starter"],
                            language="json",
                            label="Action JSON",
                            lines=9,
                        )
                        step_button = gr.Button("Run action", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### Live state")
                        summary = gr.Code(label="Workflow and safety state", language="json", lines=14)
                        payload = gr.Code(label="Visible observation payload", language="json", lines=20)
                        audit = gr.Code(label="Audit trail and constraint events", language="json", lines=12)

                gr.Markdown(
                    """
### Training loop
Start an episode, run one action at a time, and watch the safety fields. A good policy should improve task progress without pushing `current_compliance_state` into elevated risk or accumulating avoidable `episode_safety_cost`.
                    """,
                    elem_classes=["autodsar-alert"],
                )

            with gr.Tab("Benchmark Guide", id="guide"):
                gr.HTML(_guide_html())

        task.change(_task_markdown, inputs=task, outputs=mission)
        template_kind.change(_task_template, inputs=[task, template_kind], outputs=action_json)
        task.change(_task_template, inputs=[task, template_kind], outputs=action_json)
        reset_button.click(
            reset,
            inputs=[task, difficulty, seed],
            outputs=[mission, summary, payload, audit, action_json],
        )
        step_button.click(step, inputs=action_json, outputs=[summary, payload, audit])

    return demo


def create_autodsar_web_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[Any] = None,
) -> FastAPI:
    """Create the FastAPI app with AutoDSAR as the primary Gradio UI."""

    app = create_fastapi_app(
        env, action_cls, observation_cls, max_concurrent_envs, concurrency_config
    )
    metadata = load_environment_metadata(env, env_name)
    web_manager = WebInterfaceManager(env, action_cls, observation_cls, metadata)

    @app.get("/", include_in_schema=False)
    async def web_root():
        return RedirectResponse(url="/web/")

    @app.get("/web", include_in_schema=False)
    async def web_root_no_slash():
        return RedirectResponse(url="/web/")

    @app.get("/web/metadata")
    async def web_metadata():
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        if "message" in request:
            message = request["message"]
            if hasattr(web_manager.env, "message_to_action"):
                action = web_manager.env.message_to_action(message)
                if hasattr(action, "tokens"):
                    action_data = {"tokens": action.tokens.tolist()}
                else:
                    action_data = action.model_dump(exclude={"metadata"})
            else:
                action_data = {"message": message}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    gradio_blocks = build_autodsar_ui(
        web_manager,
        action_fields={},
        metadata=metadata,
        is_chat_env=False,
        title=metadata.name,
        quick_start_md=get_quick_start_markdown(metadata, action_cls, observation_cls),
    )
    return gr.mount_gradio_app(
        app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )
