from pathlib import Path

from stable_baselines3 import PPO
from mario_env import MarioEnv


def main():
    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "models" / "mario_ppo_baseline"

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_path}. Zuerst train_ppo.py ausführen."
        )

    env = MarioEnv(timer=20, render=True)
    model = PPO.load(str(model_path))

    try:
        observation, info = env.reset()

        while True:
            action, _state = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Spiel beendet:", info)
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
