# ── baseline.py ───────────────────────────────────────────────────────────────
# PID controller that follows the track centerline.
# Used as a performance baseline to compare against the RL agent.
# Run: python baseline.py
#
# Outputs a trajectory CSV to outputs/trajectories/trajectory_baseline.csv

import numpy as np
import csv
import os
from env.race import RacingEnv
from config import TRAJECTORY_DIR


class PIDController:
    """Steers proportionally to signed cross-track error from centerline."""

    def __init__(self, kp: float = 0.8, kd: float = 0.1):
        self.kp        = kp
        self.kd        = kd
        self._prev_err = 0.0

    def compute_steer(self, car_x, car_y, car_heading, track) -> float:
        idx   = track.nearest_waypoint(car_x, car_y)
        wp    = track.centerline[idx]
        nwp   = track.centerline[(idx + 1) % len(track.centerline)]

        # Heading toward next waypoint
        target_angle = np.arctan2(nwp[1] - wp[1], nwp[0] - wp[0])
        err           = target_angle - car_heading

        # Wrap to [-π, π]
        err = (err + np.pi) % (2 * np.pi) - np.pi

        steer        = self.kp * err + self.kd * (err - self._prev_err)
        self._prev_err = err
        return float(np.clip(steer, -1.0, 1.0))


def run_baseline(n_episodes: int = 5, render: bool = False):
    env = RacingEnv(render_mode="human" if render else None)
    pid = PIDController()
    lap_times = []

    for ep in range(n_episodes):
        obs, _   = env.reset()
        pid._prev_err = 0.0
        done     = False
        frame    = 0
        rows     = []

        while not done:
            steer = pid.compute_steer(
                env.car.x, env.car.y, env.car.heading, env.track
            )
            action = np.array([steer, 0.6], dtype=np.float32)  # constant throttle
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rows.append([frame, round(env.car.x, 4), round(env.car.y, 4),
                         round(env.car.heading, 4), round(env.car.speed, 4),
                         round(reward, 4)])
            frame += 1
            if render:
                env.render()

        lap_times.append(frame)
        print(f"Baseline ep {ep+1}: {frame} steps ({frame * 0.05:.1f}s)")

    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    out = os.path.join(TRAJECTORY_DIR, "trajectory_baseline.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "heading", "speed", "reward"])
        writer.writerows(rows)
    print(f"Saved → {out}")
    print(f"\nBaseline mean lap: {np.mean(lap_times) * 0.05:.2f}s "
          f"(±{np.std(lap_times) * 0.05:.2f}s)")
    env.close()


if __name__ == "__main__":
    run_baseline(render=False)