# ── env/reward.py ─────────────────────────────────────────────────────────────
# Two reward modes, selected by config.REWARD_MODE.
#
# checkpoint → +1 per new waypoint passed, -10 if off track.
# laptime    → progress * speed * dt, -10 if off track.

from config import (REWARD_MODE, CHECKPOINT_REWARD,
                    OFF_TRACK_PENALTY, SPEED_WEIGHT, STEP_PENALTY, DT)


def compute_reward(
    on_track: bool,
    new_waypoints: int,
    speed: float,
    prev_progress: float,
    curr_progress: float,
) -> float:
    if not on_track:
        return OFF_TRACK_PENALTY

    reward = -STEP_PENALTY

    if REWARD_MODE == "checkpoint":
        return reward + float(new_waypoints) * CHECKPOINT_REWARD

    elif REWARD_MODE == "laptime":
        progress_delta = curr_progress - prev_progress
        # Handle wrap-around at lap completion (progress jumps from ~1.0 → ~0.0)
        # A genuine lap completion: delta is large-negative (e.g. -0.98).
        # Backward motion is small-negative (e.g. -0.002).
        # Threshold: if delta < -0.5 it must be a lap wrap, not going backward.
        if progress_delta < -0.5:
            progress_delta = (1.0 - prev_progress) + curr_progress  # true forward delta
        elif progress_delta < 0:
            progress_delta = 0.0  # genuinely going backward — no reward
        return reward + (progress_delta * speed * SPEED_WEIGHT)

    return reward