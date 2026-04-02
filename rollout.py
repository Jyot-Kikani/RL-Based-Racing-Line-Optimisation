# ── rollout.py ────────────────────────────────────────────────────────────────
# Load a trained model, run N laps, export trajectory CSV.
# Run: python rollout.py --model models/best_model
#
# Output CSV columns: frame, x, y, heading, speed, reward

import argparse
import csv
import os
import numpy as np
from stable_baselines3 import SAC
from env.race import RacingEnv
from config import TRAJECTORY_DIR


def run_rollout(model_path: str, n_episodes: int = 3, render: bool = True):
    model  = SAC.load(model_path)
    mode   = "human" if render else None
    env    = RacingEnv(render_mode=mode)

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        frame     = 0
        rows      = []
        total_rew = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rows.append([
                frame,
                round(env.car.x, 4),
                round(env.car.y, 4),
                round(env.car.heading, 4),
                round(env.car.speed, 4),
                round(reward, 4),
            ])
            total_rew += reward
            frame += 1

            if render:
                env.render()

        print(f"Episode {ep+1}: {frame} steps, total reward {total_rew:.2f}")

        os.makedirs(TRAJECTORY_DIR, exist_ok=True)
        out_path = os.path.join(TRAJECTORY_DIR, f"trajectory_ep{ep+1}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "heading", "speed", "reward"])
            writer.writerows(rows)
        print(f"  Saved → {out_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/best_model")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    run_rollout(args.model, args.episodes, render=not args.no_render)