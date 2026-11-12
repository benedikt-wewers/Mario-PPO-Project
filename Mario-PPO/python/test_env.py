from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mario_env import MarioEnv


def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"

    env = DummyVecEnv([lambda: MarioEnv(timer=20, render=True)])
    env = VecNormalize.load(str(model_dir / "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_dir / "mario_ppo_baseline"))

    try:
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0]:
                print("Spiel beendet:", info[0])
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()