# ── train.py ──────────────────────────────────────────────────────────────────
# Run: python train.py
# Monitor: tensorboard --logdir logs/

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env.race import RacingEnv
from config import (TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES,
                    LEARNING_RATE, BATCH_SIZE, BUFFER_SIZE,
                    MODEL_DIR, LOG_DIR)


def main():
    env      = Monitor(RacingEnv())
    eval_env = Monitor(RacingEnv())

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR + "checkpoints/",
        name_prefix="sac_racing",
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb, checkpoint_cb],
    )
    model.save(MODEL_DIR + "sac_final")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()