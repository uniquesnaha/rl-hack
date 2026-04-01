# How RL Training Works with the DSAR Environment

## 1. The Core RL Loop (Universal)

Every RL system in history — from Atari games to ChatGPT — uses THIS loop:

```
┌─────────────────────────────────────────────────────────┐
│                   THE RL LOOP                           │
│                                                         │
│   Environment ──observation──► Agent                    │
│   Environment ◄──action──────── Agent                   │
│   Environment ──reward────────► Agent                   │
│   (repeat until done=True)                              │
│                                                         │
│   Agent's goal: maximise total reward over time         │
└─────────────────────────────────────────────────────────┘
```

In your DSAR environment:
- **Environment** = the FastAPI server (`uvicorn server.app:app`)
- **Agent** = whatever program sends actions to `/step`
- **Observation** = the 16 FieldItems (customer data record)
- **Action** = `query_silo`, `classify_field`, `compile_response`
- **Reward** = +0.10 (correct), -0.30 (leak), etc.

---

## 2. What You've Built vs What a Trained Agent Is

Right now, `inference.py` uses **Llama-3.3-70b-versatile** as the agent.
That is a zero-shot LLM agent — it reads the fields and guesses.

A **trained RL agent** is completely different. It starts knowing nothing
and learns purely from reward signals — like a baby learning to walk
by falling and getting up, not by reading a book about walking.

```
Zero-shot LLM agent (what you have now):
  Input: "classify_field churn_probability"
  LLM reads description → applies language understanding → outputs "withhold"
  Score: ~0.99 (already very capable from training)

Trained RL agent (what future researchers will build):
  Episode 1:    random guesses → score 0.12  (terrible)
  Episode 100:  some patterns → score 0.45
  Episode 1000: learned "risk/churn/profit = withhold" → score 0.78
  Episode 5000: near-optimal policy → score 0.95+
```

---

## 3. How Case 1 Training Actually Works (Step by Step)

### Phase A: Collect Experience

The RL training loop runs thousands of episodes against your server:

```python
# Pseudocode of what an RL training library does
for episode in range(10000):
    # Reset: get fresh episode (different seed each time)
    obs = env.reset(seed=episode)
    done = False
    episode_data = []

    while not done:
        # Agent picks an action (random at first, smarter later)
        action = agent.policy(obs)

        # Send to YOUR server
        next_obs, reward, done = env.step(action)

        # Store the experience tuple
        episode_data.append((obs, action, reward, next_obs, done))
        obs = next_obs

    # After episode: update the agent's brain using the rewards
    agent.learn(episode_data)
```

Your server at `http://localhost:8000` is the environment. 
The RL library calls `/reset` and `/step` in this loop.

### Phase B: The Agent Learns

After collecting episodes, the agent updates its "policy" — its brain
that maps observations to actions. With PPO (the most common algorithm):

```
Reward +0.10 for "classify_field full_name disclose"
→ Agent learns: when you see a field with "name" in the description,
                the word "disclose" is the right token to output

Reward -0.30 for "classify_field risk_score disclose"
→ Agent learns: when you see a "score" field with probability/numerical value,
                never say "disclose"

Reward -0.01 per step beyond step 10
→ Agent learns: stop querying redundant silos, classify fields faster
```

The agent gets no instructions. It discovers these rules purely from the
+/- signals coming back from YOUR grader.

---

## 4. How Your Specific Code Enables This

### 4a. The `_EPISODES` dict (state persistence)

```python
# dsar_environment.py line ~53
_EPISODES: Dict[str, EpisodeData] = {}
```

This is critical for RL training. An RL library runs **thousands of episodes
in parallel** on different threads/processes:

```
Thread 1: episode abc-123 → step 5 → "classify_field email"
Thread 2: episode def-456 → step 2 → "query_silo billing"
Thread 3: episode ghi-789 → step 8 → "compile_response"
```

All hitting `/step` at the same time. The `_EPISODES` dict keyed by
`episode_id` is what keeps them from interfering with each other.
Without this, Thread 2's state would corrupt Thread 1's episode.

### 4b. The Grader (the reward function IS the teacher)

```python
# grader.py — this IS the training signal
def compute_step_reward(...) -> float:
    if true_label == "INTERNAL_ONLY" and decision == "disclose":
        reward = -0.30     # LEAK: agent learns never to do this

def compute_terminal_score(...) -> float:
    privacy_penalty = n * 0.30 * (1 + n * 0.50)  # non-linear
    # 1 leak:  -0.45  (painful but survivable)
    # 2 leaks: -1.20  (catastrophic, floors to 0)
```

The non-linear penalty is **intentional RL design**. It creates a
curriculum:
- Early agent: leaks multiple fields, gets 0.0 → strong gradient signal
- Mid agent: leaks 1 field, gets ~0.50 → still incentivised to improve
- Good agent: zero leaks, gets 0.95+ → pushes F1 higher

### 4c. `deadline_pressure` (the time signal)

```python
deadline_pressure = steps_remaining / MAX_STEPS  # 1.0 → 0.0
```

This normalised value tells the agent "you're running out of steps."
A smart agent learns to call `compile_response` when this approaches 0
rather than waiting — exactly the kind of implicit constraint RL learns
that hard-coded rules cannot capture.

### 4d. `constraint_violated` (the hard safety gate)

```python
# dsar_environment.py
if episode.leaked_count > 2 and not episode.done:
    episode.constraint_violated = True
    episode.done = True   # immediate termination
    return obs with reward=step_reward (current step's small penalty)
```

This teaches the agent that privacy violations have HARD consequences —
the episode simply ends. An RL agent that learns in an environment with
this gate will develop a **risk-averse policy**: even one leak is
painful (-0.45), two leaks are catastrophic (score 0.0), three leaks
end the game immediately.

---

## 5. How This Aligns with the Hackathon

The hackathon is from **Meta's OpenEnv** initiative. The goal is to:

> Build real-world RL environments that sit between toy games (Atari)
> and impossible problems (solve all of chemistry).
> The environments should be deployable, serve real domains, and
> provide meaningful RL training signals.

Your DSAR environment hits every criterion:

| Criterion | How DSAR env satisfies it |
|-----------|--------------------------|
| **Real domain** | GDPR compliance is a real £2bn industry problem |
| **Deployable** | FastAPI server with OpenEnv spec → runs anywhere |
| **Meaningful rewards** | Non-linear privacy penalty, step costs, constraint gates |
| **Difficulty progression** | task_easy → task_medium → task_hard |
| **RL training signal** | Dense rewards every step, clear terminal grade |
| **Baseline agent** | inference.py shows how an LLM interacts with it |

The hackathon judges evaluate the **environment**, not how smart your
agent is. They want to see:
1. Can an agent learn from this? (yes — reward signals are dense and informative)
2. Is the task meaningful? (yes — GDPR classification is real work)
3. Is the difficulty real? (needs task_medium + task_hard to confirm)

---

## 6. The Three-Phase Future Training Pipeline

Once your environment is deployed, researchers would train agents in
three phases:

### Phase 1: Zero-shot Baseline (what you have now)
```
inference.py → LLM reads fields → makes decisions → score ~0.85-0.99
```
Acts as the upper bound. Shows what a strong pre-trained LLM can do
without any environment-specific training.

### Phase 2: Supervised Fine-tuning (SFT Warm-start)
```
Generate 10,000 episode trajectories where the LLM is correct
→ Fine-tune a smaller model (e.g., Llama-3-8b) on these demonstrations
→ Smaller model now knows the task format
→ score goes from ~0.40 → ~0.70
```

### Phase 3: RL Post-training (RLVR / PPO / GRPO)
```
Run the SFT model against your environment for 100k+ episodes
→ Model receives reward signals from your grader
→ Policy gradient updates: increase probability of +reward actions
→ After training: score goes from ~0.70 → ~0.92+
→ Model learns things the SFT data never showed it:
   - "risk_score withhold" even when description is ambiguous
   - Call compile_response before running out of steps
   - Never re-classify a field (saves steps for step efficiency)
```

The key insight: **the RL training finds behaviours that pure
language understanding cannot**. An LLM might guess that
`risk_score` should be withheld. An RL-trained agent KNOWS it,
because the -0.30 signal 10,000 times burned it into the weights.

---

## 7. The Gymnasium Wrapper (plug-in point for RL libraries)

Standard RL libraries (Stable Baselines 3, TRL, OpenRLHF, veRL) expect
a Gym-compatible environment. Here's the wrapper that makes your
FastAPI server compatible with all of them:

```python
import gymnasium as gym
import requests

class DSARGymEnv(gym.Env):
    def __init__(self, server_url="http://localhost:8000", task_id="task_easy"):
        self.server_url = server_url
        self.task_id = task_id
        self.episode_id = None

    def reset(self, seed=None):
        r = requests.post(f"{self.server_url}/reset",
                          json={"task_id": self.task_id, "seed": seed})
        obs = r.json()
        self.episode_id = obs["observation"]["episode_id"]
        return obs, {}

    def step(self, action: dict):
        action["metadata"] = {"episode_id": self.episode_id}
        r = requests.post(f"{self.server_url}/step", json={"action": action})
        data = r.json()
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        return obs, reward, done, False, {}
```

A researcher can now do:
```python
from stable_baselines3 import PPO
env = DSARGymEnv(task_id="task_easy")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

That's it — your environment becomes a first-class RL training target
for any standard RL library in the world.

---

## Summary

```
YOUR DSAR ENVIRONMENT
        │
        ├── For the hackathon:
        │     Evaluated as an environment design
        │     Judges check: deployable, real domain, meaningful rewards,
        │                   3 difficulty levels, reproducible baseline
        │
        └── For future RL research:
              Acts like Atari/CartPole but for language + compliance
              Any RL algorithm can train against it via HTTP or Gym wrapper
              Training produces agents that learn GDPR rules from rewards,
              not from instruction — the core premise of RLVR (Reinforcement
              Learning from Verified Rewards)
```
