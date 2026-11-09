from mario_env import MarioEnv


def main():
    env = MarioEnv(timer=20, render=True)

    try:
        observation, info = env.reset(seed=42)
        print("Gymnasium-Umgebung gestartet.")
        print("Shape der Observation:", observation.shape)
        print("Action Space:", env.action_space)
        print("Start-Fortschritt:", info["completion"])

        for step in range(2000):
            # Zufällige Aktionen dienen nur zum Test der Schnittstelle.
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            if step % 25 == 0:
                print(
                    f"Schritt {step:4d} | Aktion {action:2d} | "
                    f"Reward {reward:7.3f} | "
                    f"Fortschritt {info.get('completion', 0):.3f}"
                )

            if terminated or truncated:
                print("Episode beendet:", info)
                break

        print("Test abgeschlossen.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
