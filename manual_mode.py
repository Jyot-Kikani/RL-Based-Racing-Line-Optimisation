# ── manual_mode.py ────────────────────────────────────────────────────────────
# Drive the environment yourself to verify physics and rendering.
# Run: python manual_mode.py
#
# Controls:
#   UP    → accelerate
#   DOWN  → brake
#   LEFT  → steer left
#   RIGHT → steer right
#   R     → reset
#   ESC   → quit

import pygame
import numpy as np
from env.race import RacingEnv
from config import FPS


current_steer = 0.0

def get_human_action(keys) -> np.ndarray:
    global current_steer
    target_steer = 0.0
    accel = 0.0
    if keys[pygame.K_LEFT]:  target_steer =  1.0
    if keys[pygame.K_RIGHT]: target_steer = -1.0
    if keys[pygame.K_UP]:    accel =  1.0
    if keys[pygame.K_DOWN]:  accel = -1.0
    
    # Smooth steering control for less discrete jumps
    STEER_SPEED = 0.04
    RETURN_SPEED = 0.20  # Fast return to center

    if target_steer == 0:
        # Auto-center when keys are released
        if current_steer > 0:
            current_steer = max(0.0, current_steer - RETURN_SPEED)
        elif current_steer < 0:
            current_steer = min(0.0, current_steer + RETURN_SPEED)
    elif target_steer > 0:
        if current_steer < 0:
            current_steer = min(0.0, current_steer + RETURN_SPEED) # Fast counter-steer
        else:
            current_steer = min(current_steer + STEER_SPEED, target_steer)
    elif target_steer < 0:
        if current_steer > 0:
            current_steer = max(0.0, current_steer - RETURN_SPEED) # Fast counter-steer
        else:
            current_steer = max(current_steer - STEER_SPEED, target_steer)

    return np.array([current_steer, accel], dtype=np.float32)


def main():
    env = RacingEnv(render_mode="human")
    obs, _ = env.reset()
    clock  = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, _ = env.reset()

        keys   = pygame.key.get_pressed()
        action = get_human_action(keys)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode ended — {info}")
            obs, _ = env.reset()

        clock.tick(FPS)

    env.close()


if __name__ == "__main__":
    main()