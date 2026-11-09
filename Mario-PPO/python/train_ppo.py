from pathlib import Path

from stable_baselines3 import PPO
from mario_env import MarioEnv


def main():
    project_dir = Path(__file__).resolve().parents[1]
    model_dir = project_dir / "models"
    log_dir = project_dir / "logs"

    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Für das erste Training ohne Spielfenster, damit es schneller läuft.
    env = MarioEnv(timer=20, render=False)

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            verbose=1,
            tensorboard_log=str(log_dir),
        )

        model.learn(total_timesteps=10_000)
        model.save(str(model_dir / "mario_ppo_baseline"))
        print("Modell gespeichert.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
