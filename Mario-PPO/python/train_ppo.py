from pathlib import Path
from collections import deque
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mario_env import MarioEnv


class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, report_every=500):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.report_every = report_every
        self.last_episode = 0
        self.recent_episodes = deque(maxlen=10)
        self.stage = 0
        self.stage_names = (
            "Vorankommen ohne unnötige Sprünge",
            "Weiterkommen und Hindernisse gezielt behandeln",
            "Level zuverlässig abschließen",
            "Level wiederholt gewinnen und unnötige Sprünge vermeiden",
        )

    def _on_step(self):
        episode_steps = self.training_env.get_attr("episode_steps")[0]
        episode_reward = self.training_env.get_attr("episode_reward")[0]
        completion = self.training_env.get_attr("last_completion")[0]

        if self.num_timesteps % self.report_every == 0:
            progress = self.num_timesteps / self.total_timesteps * 100
            print(
                f"Training {self.num_timesteps:,}/{self.total_timesteps:,} "
                f"({progress:.1f}%) | Stufe {self.stage + 1}/4 | aktueller Episodenschritt "
                f"{episode_steps} | Level {completion:.1%} | "
                f"Episodenreward {episode_reward:+.2f}",
                flush=True,
            )

        episode_infos = self.locals.get("infos", [])
        for info in episode_infos:
            episode = info.get("episode")
            if episode:
                self.last_episode += 1
                self.recent_episodes.append(info)
                details = info.get("episode", {})
                print(
                    f"\n=== Lernepisode {self.last_episode} abgeschlossen ===\n"
                    f"Stufe {self.stage + 1}/4: {self.stage_names[self.stage]}\n"
                    f"Level-Fortschritt: {info.get('completion', 0):.1%} | "
                    f"Status: {info.get('game_status', 'Zeitlimit/Tod')}\n"
                    f"Sprünge: {info.get('jumps', 0)} | unnötig: "
                    f"{info.get('unneeded_jumps', 0)} "
                    f"({info.get('unneeded_jump_rate', 0):.0%}) | "
                    f"Schaden: {info.get('hurts', 0)}\n"
                    f"Gesamtreward: {info.get('episode_reward', details.get('r', 0)):+.2f}\n"
                    f"Endschritt: "
                    f"Aktion {MarioEnv.action_name(info.get('action', 0))} | "
                    f"{self._format_details(info.get('reward_details', []))} | "
                    f"Abschlussreward {info.get('reward', 0):+.2f}\n"
                    f"Ereignisbilanz: {self._format_event_summary(info.get('event_summary', {}), info.get('event_point_summary', {}))}\n"
                    f"===============================",
                    flush=True,
                )
                self._maybe_promote()
        return True

    def _maybe_promote(self):
        if self.stage >= 3 or len(self.recent_episodes) < self.recent_episodes.maxlen:
            return

        episodes = list(self.recent_episodes)
        if self.stage == 0:
            mastered = sum(
                item.get("completion", 0) >= 0.15
                and item.get("unneeded_jump_rate", 0) <= 0.50
                for item in episodes
            )
            criterion = "mindestens 15% Level-Fortschritt und höchstens 50% unnötige Sprünge"
        elif self.stage == 1:
            mastered = sum(
                item.get("completion", 0) >= 0.40
                and item.get("unneeded_jump_rate", 0) <= 0.35
                for item in episodes
            )
            criterion = "mindestens 40% Fortschritt und höchstens 35% unnötige Sprünge"
        elif self.stage == 2:
            mastered = sum(
                item.get("won", False)
                and item.get("unneeded_jump_rate", 0) <= 0.25
                for item in episodes
            )
            criterion = "Sieg und höchstens 25% unnötige Sprünge"
        else:
            mastered = sum(
                item.get("won", False)
                and item.get("unneeded_jump_rate", 0) <= 0.10
                and item.get("hurts", 0) <= 1
                for item in episodes
            )
            criterion = "Sieg, höchstens 10% unnötige Sprünge und maximal ein Treffer"

        if mastered < 8:
            print(
                f"Curriculum bleibt auf Stufe {self.stage + 1}: "
                f"{mastered}/10 Episoden erfüllt ({criterion}; benötigt 8/10).",
                flush=True,
            )
            return

        previous = self.stage
        self.stage += 1
        self.training_env.env_method("set_curriculum_stage", self.stage)
        model_dir = Path(self.training_env.get_attr("ppo_dir")[0]) / "models"
        stage_model = model_dir / f"mario_ppo_stage_{self.stage}"
        self.model.save(str(stage_model))
        self.training_env.save(str(model_dir / f"vecnormalize_stage_{self.stage}.pkl"))
        (model_dir / "curriculum.json").write_text(
            json.dumps({"stage": self.stage, "episodes_seen": self.last_episode}, indent=2),
            encoding="utf-8",
        )
        print(
            f"\n*** STUFE GESCHAFFT: {previous + 1} -> {self.stage + 1} ***\n"
            f"8 von 10 letzten Episoden erfüllten das Ziel. "
            f"Das bestehende Modell lernt jetzt weiter: {self.stage_names[self.stage]}\n"
            f"Stufenmodell gespeichert: {stage_model}.zip",
            flush=True,
        )
        self.recent_episodes.clear()

    @staticmethod
    def _format_details(details):
        if not details:
            return "keine Einzelereignisse"
        return ", ".join(f"{name} {points:+.2f}" for name, points in details)

    @staticmethod
    def _format_event_summary(summary, point_summary):
        if not summary:
            return "keine"
        return ", ".join(
            f"{name}: {count}x ({point_summary.get(name, 0):+.2f})"
            for name, count in summary.items()
        )

    def _on_training_end(self):
        print()


def main():
    project_dir = Path(__file__).resolve().parents[1]
    model_dir = project_dir / "models"
    log_dir = project_dir / "logs"
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(MarioEnv(timer=20, render=True))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # etwas Exploration
        verbose=1,
        tensorboard_log=str(log_dir),
    )
    callbacks = [
        CheckpointCallback(
            save_freq=50_000, save_path=str(model_dir), name_prefix="mario_ppo"
        ),
        TrainingProgressCallback(total_timesteps=1_000_000),
    ]

    try:
        model.learn(total_timesteps=1_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("Training abgebrochen, speichere Zwischenstand.")
    finally:
        model.save(str(model_dir / "mario_ppo_baseline"))
        env.save(str(model_dir / "vecnormalize.pkl"))
        env.close()


if __name__ == "__main__":
    main()
