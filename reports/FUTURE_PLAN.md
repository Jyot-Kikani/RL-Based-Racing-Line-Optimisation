# F1 Racing Line Optimisation — Complete Future Plan

> Current status: Stage 1 training underway, ~30k steps, agent already
> completing full 5000-step laps. `ep_rew_mean` climbing steadily.

---

## Phase 1 — Finish Stage 1 (you are here)

**Goal:** Solidify survival behaviour before switching reward modes.

### When to stop Stage 1

Watch these two signals in tensorboard (`logs/stage1`):

| Signal | Stop condition |
|---|---|
| `rollout/ep_rew_mean` | Plateaus for >20k consecutive steps |
| `eval/mean_ep_length` | Stays at 5000 consistently across 3+ evals |

Given the current trajectory (5000-step laps already at 20k steps), expect
plateau around **50k–80k steps**. Running the full 300k is unnecessary —
Stage 1 has already succeeded earlier than expected.

### Before moving on — verify via rollout

```bash
python rollout.py --model models/stage1/best_model --episodes 3 --no-save
```

Watch for:
- Car navigates both hairpins without crashing
- Speed on straights is near MAX_SPEED
- No wandering or oscillation on straights

If the car still crashes occasionally at hairpins, let Stage 1 run another
20k steps and check again.

---

## Phase 2 — Stage 2: Time Attack on Drag Strip

**Goal:** Same track, new objective — go *fast*, not just survive.

### What changes
- `REWARD_MODE` switches to `"laptime"`
- Loads Stage 1 `best_model.zip` weights AND replay buffer
- Learning rate drops to `1e-4` (fine-tune, don't overwrite)

### Run it
```bash
python train.py --stage 2
```

### What to watch

| Metric | Expected behaviour |
|---|---|
| `ep_rew_mean` | Initial drop is normal — laptime reward is scaled differently |
| `ep_len_mean` | Should stay near 5000 — agent must not forget how to survive |
| `eval/mean_reward` | Should climb steadily as agent goes faster |

**Red flag:** If `ep_len_mean` drops below 2000 within the first 50k steps,
the agent is forgetting survival behaviour. Fix: reduce `SPEED_WEIGHT` in
`config.py` to make the speed reward less aggressive, or restart Stage 2
from Stage 1 weights with a lower LR (`5e-5`).

### What the agent should learn
- Brake *earlier* into hairpins (anticipatory, not reactive)
- Hit the apex of the U-turns to minimise distance
- Hold full throttle longer on straights
- The optimal line through the U-turn is the inside arc — this is the
  racing line principle that transfers to Monza

### Stop condition
`eval/mean_reward` plateaus for 30k+ steps, OR 400k total steps reached.

---

## Phase 3 — Stage 3: Transfer to Monza

**Goal:** Apply everything learned on the drag strip to a full F1 circuit.

### What changes
- `TRACK_FILE` switches to `monza.csv`
- Loads Stage 2 `best_model.zip` weights only (no replay buffer — different track)
- Learning rate drops to `5e-5` — very conservative to preserve cornering knowledge

### Run it
```bash
python train.py --stage 3
```

### What to expect
Monza has 11 corners of varying complexity: chicanes, high-speed sweepers,
and the Parabolica. The agent transfers the *physics* it learned (when to
brake, how to hold a turn) but must relearn the *geometry* of each corner.

Expect early Stage 3 to look like early Stage 1 — crashes at unfamiliar
corners. This is normal. The difference is that Stage 3 should converge
*much faster* than a cold-start would, because the critic already assigns
correct values to "approaching corner at high speed" states.

**Realistic timeline:** Meaningful laps on Monza by 200k–300k steps.
Full convergence by 600k steps.

### Monza-specific concerns
- The two Lesmo corners are taken at high speed — the agent may
  initially over-brake here
- The Ascari chicane requires two quick direction changes — test this
  specifically in rollout
- The Parabolica is a long decreasing-radius corner — hardest corner
  for a reactive sensor-based agent

### Stop condition
Agent completes consistent full laps of Monza without crashing,
and `eval/mean_reward` plateaus.

---

## Phase 4 — Evaluation & Baseline Comparison

This phase produces the actual results for your report. Run everything
*after* Stage 3 is complete.

### Step 1: Run the PID baseline

The PID controller follows the centerline at constant throttle. This is
your "dumb driver" reference point.

```bash
python baseline.py
```

This outputs `outputs/trajectories/trajectory_baseline.csv` with lap time,
speed profile, and trajectory coordinates.

**Record:** Mean lap time across 5 laps, mean speed, off-track rate.

### Step 2: Run the RL agent rollout

```bash
python rollout.py --model models/stage3/best_model --episodes 5
```

This saves `outputs/trajectories/trajectory_ep1.csv` through `ep5.csv`.

**Record:** Same metrics as PID for direct comparison.

### Step 3: Generate all plots

```bash
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
```

**Plots produced:**
- `outputs/plots/racing_line_heatmap.png` — the learned racing line
  coloured by speed (red = slow, green = fast). This is your main
  deliverable visual.
- `outputs/plots/speed_profile.png` — speed vs frame, shows braking zones

### Step 4: Overlay comparison plot

Write a short script (or add to `visualize.py`) that overlays three lines
on the same track:

1. **Gray dashed** — centerline (reference)
2. **Blue** — PID trajectory
3. **Green→Red heatmap** — RL agent trajectory

This single image is the most compelling result in your report — it shows
visually that the RL agent has found a different (and better) line than the
naive centerline follower.

### Step 5: Compile results table

| Metric | PID baseline | RL agent (Stage 3) | Improvement |
|---|---|---|---|
| Mean lap time (s) | | | |
| Mean speed (km/h) | | | |
| Min speed at hairpins (km/h) | | | |
| Lap completion rate | | | |
| Off-track incidents | | | |

---

## Phase 5 — Report & Writeup

Structure your report around this narrative arc:

```
Problem → Environment → Curriculum → Results → Analysis
```

### Suggested sections

**1. Introduction**
- Why racing line optimisation is a hard problem
- Why RL is a natural fit (sequential decisions, delayed reward)
- What this project demonstrates

**2. Environment Design**
- Bicycle kinematic model with grip limits and drag
- 7-ray LiDAR sensor suite
- Reward function design choices (why checkpoint first, why laptime second)
- Problems encountered and how they were solved (rolling start, anti-stall,
  is_on_track fix)

**3. Curriculum Learning Strategy**
- Why cold-start on Monza fails
- Three-stage progression and the reasoning behind each
- Stage handoff mechanism (weight transfer, replay buffer reuse)

**4. Results**
- Training curves (reward and episode length vs timesteps) — screenshot
  from tensorboard for all 3 stages
- Racing line heatmap on Monza
- Comparison table: RL vs PID
- Overlay plot: RL line vs PID line vs centerline

**5. Analysis**
- Where the agent found non-obvious lines (apex cutting, late braking)
- Where it still struggles (complex chicanes, decreasing-radius corners)
- What the reward curve shape tells us about the learning process

**6. Conclusion**
- What was achieved vs what was planned
- Limitations (sensor-based vs model-based, single circuit)
- Future work (lookahead observation, multi-circuit generalisation)

---

## Contingency Plans

### If Stage 3 struggles to transfer to Monza

Add an intermediate stage on the `test_oval.csv` (already in your repo).
The oval introduces varied corner radii without Monza's complexity.

```
drag strip → oval → Monza
```

Modify `train.py` to add Stage 2.5:
```python
2.5: {
    "track":       "data/tracks/test_oval.csv",
    "reward_mode": "laptime",
    "steps":       200_000,
    "lr":          1e-4,
}
```

### If laptime reward causes catastrophic forgetting in Stage 2

Reduce `SPEED_WEIGHT` from `0.01` to `0.005`. This halves the speed
incentive so the agent doesn't throw away survival behaviour to chase speed.

### If Monza lap times are worse than PID

This is actually fine for the report — explain *why* (PID has perfect
centerline knowledge hardcoded, the RL agent only has sensors), and focus
the comparison on the *racing line shape* rather than raw lap time. The
interesting result is that the RL agent finds a geometrically different
line, not necessarily a faster one at Stage 3 convergence.

### If training is too slow on your machine

Add `n_envs=4` parallel environments to Stage 1 (safe because it's the
first stage with no weight loading):

```python
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(RacingEnv, n_envs=4)
```

This runs 4 environments simultaneously and roughly 3–4× training throughput.
Note: `EvalCallback` still needs a single env, not a VecEnv.

---

## Complete Timeline Summary

| Phase | What | Command | Done when |
|---|---|---|---|
| 1 | Finish Stage 1 | `train.py --stage 1` | `ep_len_mean` plateaus at 5000 |
| 2 | Stage 2 time attack | `train.py --stage 2` | `eval_reward` plateaus |
| 3 | Stage 3 Monza | `train.py --stage 3` | consistent laps, no crashes |
| 4a | PID baseline | `baseline.py` | CSV saved |
| 4b | RL rollout | `rollout.py` | 5 trajectory CSVs saved |
| 4c | Generate plots | `visualize.py` | heatmap + speed profile saved |
| 4d | Overlay plot | custom script | comparison image saved |
| 5 | Write report | — | submitted |

**Estimated remaining training compute** (at ~17 fps / 13 fps observed):
- Stage 1 remaining (~30k steps): ~30 min
- Stage 2 (400k steps): ~8–9 hours
- Stage 3 (600k steps): ~12–13 hours
- Total: ~22 hours of training time

Run stages overnight. Each stage saves `best_model.zip` continuously so
you can stop and resume at any time without losing progress.
