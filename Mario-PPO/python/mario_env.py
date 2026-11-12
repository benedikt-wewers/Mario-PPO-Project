from pathlib import Path
import socket
import subprocess
from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarioEnv(gym.Env):
    """
    Gymnasium-Umgebung für das originale Mario-AI-Framework.

    Beobachtung:
      13 skalare Werte + 256 Werte aus dem Mario-Grid = 269 Werte.

    Aktionen:
      0 bis 11, entsprechend der Zuordnung in MarioPpoAgent.java.
    """

    metadata = {"render_modes": []}

    def __init__(self, timer=20, port=5555, render=True):
        super().__init__()

        self.timer = timer
        self.port = port
        self.render_game = render

        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(269,),
            dtype=np.float32,
        )

        self.ppo_dir = Path(__file__).resolve().parents[1]
        self.project_dir = self.ppo_dir.parent
        self.framework_dir = self.project_dir / "Mario-AI-Framework"
        self.framework_out = self.framework_dir / "out"
        self.java_out = self.ppo_dir / "out"
        self.level_path = self.framework_dir / "levels" / "original" / "lvl-1.txt"

        self.server = None
        self.connection = None
        self.reader = None
        self.process = None
        self.last_observation = None
        self.last_completion = 0.0
        self.last_coins = 0.0
        self.last_mushrooms = 0.0
        self.last_flowers = 0.0
        self.last_bricks = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_events = Counter()
        self.episode_event_points = Counter()
        self.episode_jumps = 0
        self.episode_unneeded_jumps = 0
        self.episode_hurts = 0
        self.curriculum_stage = 0
        self.last_reward_details = []
        self.episode_done = False

    def _read_line(self):
        line = self.reader.readline()
        if line == "":
            raise RuntimeError(
                "Java hat die Verbindung geschlossen, ohne ein Ergebnis zu senden."
            )
        return line.strip()

    def _parse_observation(self, line):
        parts = line.split(",")
        if parts[0] != "OBS":
            raise RuntimeError(f"OBS-Nachricht erwartet, erhalten: {line[:100]}")

        normalized = [
            "1" if value.lower() == "true"
            else "0" if value.lower() == "false"
            else value
            for value in parts[1:]
            ]

        values = np.asarray(normalized, dtype=np.float32)
        if values.size != 269:
            raise RuntimeError(
                f"Falsche Beobachtungsgroesse: {values.size}, erwartet: 269."
            )
        return values

    def _start_java(self):
        if not self.framework_out.exists():
            raise FileNotFoundError(
                f"Framework-Klassen nicht gefunden: {self.framework_out}"
            )
        if not self.java_out.exists():
            raise FileNotFoundError(
                f"Eigene Java-Klassen nicht gefunden: {self.java_out}"
            )
        if not (self.java_out / "MarioPpoAgent.class").exists():
            raise FileNotFoundError("MarioPpoAgent.class fehlt; Java-Bridge zuerst kompilieren.")
        if not (self.framework_out / "engine" / "core" / "MarioForwardModel.class").exists():
            raise FileNotFoundError("Mario-AI-Framework-Klassen fehlen; Framework zuerst kompilieren.")
        if not self.level_path.exists():
            raise FileNotFoundError(f"Level nicht gefunden: {self.level_path}")

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", self.port))
        self.server.listen(1)
        self.server.settimeout(30)

        classpath = f"{self.framework_out};{self.java_out}"
        command = [
            "java",
            f"-Dmario.record.root={self.ppo_dir / 'recordings'}",
            "-cp",
            classpath,
            "MarioPpoRunner",
            str(self.level_path),
            str(self.timer),
            str(self.port),
            str(self.render_game).lower(),
        ]

        self.process = subprocess.Popen(
            command,
            cwd=str(self.framework_dir),
        )

        try:
            self.connection, address = self.server.accept()
            self.connection.settimeout(60)
            self.reader = self.connection.makefile(
                "r", encoding="utf-8", newline="\n"
            )
        except Exception:
            self.close()
            raise

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()

        self.episode_done = False
        self.last_observation = None
        self.last_completion = 0.0
        self.last_coins = 0.0
        self.last_mushrooms = 0.0
        self.last_flowers = 0.0
        self.last_bricks = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_events.clear()
        self.episode_event_points.clear()
        self.episode_jumps = 0
        self.episode_unneeded_jumps = 0
        self.episode_hurts = 0
        self.last_reward_details = []

        self._start_java()

        first_line = self._read_line()
        observation = self._parse_observation(first_line)

        self.last_observation = observation
        self.last_completion = float(observation[6])
        self.last_coins = float(observation[9])
        self.last_mushrooms = float(observation[10])
        self.last_flowers = float(observation[11])
        self.last_bricks = float(observation[12])

        return observation, {"completion": self.last_completion}

    def step(self, action):
        if self.connection is None:
            raise RuntimeError("Keine aktive Java-Verbindung. Zuerst reset() aufrufen.")
        if self.episode_done:
            raise RuntimeError("Episode beendet. Vor dem nächsten Schritt reset() aufrufen.")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Ungueltige Aktion: {action}")

        jump_action = action in (3, 4, 5, 6, 9, 10)
        jump_context = self._jump_context(self.last_observation) if jump_action else False

        self.connection.sendall(f"{action}\n".encode("utf-8"))
        event_line = self._read_line()
        if not event_line.startswith("EVT"):
            raise RuntimeError(f"EVT-Nachricht erwartet, erhalten: {event_line[:100]}")
        events = []
        for item in event_line.split(",")[1:]:
            if item:
                event_type, event_param = (int(value) for value in item.split(":"))
                events.append((event_type, event_param))
        line = self._read_line()

        if line.startswith("OBS,"):
            observation = self._parse_observation(line)
            completion = float(observation[6])

            progress_reward = (completion - self.last_completion) * 100.0
            step_penalty = -0.01
            event_rewards = []
            event_names = []
            for name, increase, points in (
                ("Münze gesammelt", observation[9] - self.last_coins, 2.0),
                ("Pilz gesammelt", observation[10] - self.last_mushrooms, 5.0),
                ("Feuerblume gesammelt", observation[11] - self.last_flowers, 5.0),
                ("Ziegel zerstört", observation[12] - self.last_bricks, 0.5),
            ):
                for _ in range(max(0, int(increase))):
                    event_names.append(name)
                    event_rewards.append((name, points))
                    self.episode_events[name] += 1
                    self.episode_event_points[name] += points

            for event_type, event_param in events:
                name, points = self._event_reward(event_type, event_param)
                if name:
                    event_names.append(name)
                    if points:
                        event_rewards.append((name, points))
                        self.episode_events[name] += 1
                        self.episode_event_points[name] += points
                    if event_type == 9:
                        self.episode_hurts += 1

            did_jump = jump_action and any(event_type == 6 for event_type, _ in events)
            if did_jump:
                self.episode_jumps += 1
                if not jump_context:
                    self.episode_unneeded_jumps += 1
                    penalty = (0.10, 0.20, 0.30, 0.40)[self.curriculum_stage]
                    event_rewards.append(("Sprung ohne Hindernis/Gegner", -penalty))
                    event_names.append("Sprung ohne Hindernis/Gegner")
                    self.episode_events["Sprung ohne Hindernis/Gegner"] += 1
                    self.episode_event_points["Sprung ohne Hindernis/Gegner"] -= penalty
                else:
                    self.episode_events["Situationsgerechter Sprung"] += 1

            reward = progress_reward + step_penalty + sum(points for _, points in event_rewards)
            self.episode_reward += reward
            self.episode_steps += 1
            self.last_reward_details = []
            if progress_reward:
                self.last_reward_details.append(("Level-Fortschritt", progress_reward))
            self.last_reward_details.append(("Zeitkosten", step_penalty))
            self.last_reward_details.extend(event_rewards)

            self.last_completion = completion
            self.last_coins = float(observation[9])
            self.last_mushrooms = float(observation[10])
            self.last_flowers = float(observation[11])
            self.last_bricks = float(observation[12])
            self.last_observation = observation

            return observation, float(reward), False, False, {
                "completion": completion,
                "reward": reward,
                "progress_reward": progress_reward,
                "step_penalty": step_penalty,
                "event_rewards": event_rewards,
                "events": event_names,
                "action": action,
                "episode_reward": self.episode_reward,
                "episode_steps": self.episode_steps,
                "stage": self.curriculum_stage,
            }

        if line.startswith("RESULT,"):
            parts = line.split(",", 2)
            completion = float(parts[1])
            status = parts[2].strip().upper()

            won = status == "WIN"
            reward = 100.0 if won else -100.0
            self.episode_done = True
            self.last_reward_details = [("Level gewonnen" if won else f"Episode {status}", reward)]
            event_rewards = []
            event_names = []
            if jump_action and any(event_type == 6 for event_type, _ in events):
                self.episode_jumps += 1
                if not jump_context:
                    self.episode_unneeded_jumps += 1
                    penalty = (0.10, 0.20, 0.30, 0.40)[self.curriculum_stage]
                    reward -= penalty
                    self.last_reward_details.append(("Sprung ohne Hindernis/Gegner", -penalty))
                    self.episode_events["Sprung ohne Hindernis/Gegner"] += 1
                    self.episode_event_points["Sprung ohne Hindernis/Gegner"] -= penalty
                else:
                    self.episode_events["Situationsgerechter Sprung"] += 1
            for event_type, event_param in events:
                name, points = self._event_reward(event_type, event_param)
                if name:
                    event_names.append(name)
                    if points:
                        event_rewards.append((name, points))
                        self.episode_events[name] += 1
                        self.episode_event_points[name] += points
            reward += sum(points for _, points in event_rewards)
            self.episode_reward += reward
            self.last_reward_details.extend(event_rewards)

            info = {
                "completion": completion,
                "game_status": status,
                "won": won,
                "reward": reward,
                "episode_reward": self.episode_reward,
                "episode_steps": self.episode_steps,
                "event_rewards": event_rewards,
                "events": event_names,
                "action": action,
                "event_summary": dict(self.episode_events),
                "event_point_summary": dict(self.episode_event_points),
                "reward_details": self.last_reward_details,
                "stage": self.curriculum_stage,
                "jumps": self.episode_jumps,
                "unneeded_jumps": self.episode_unneeded_jumps,
                "unneeded_jump_rate": (
                    self.episode_unneeded_jumps / self.episode_jumps
                    if self.episode_jumps else 0.0
                ),
                "hurts": self.episode_hurts,
            }

            return self.last_observation, reward, True, False, info

        raise RuntimeError(f"Unbekannte Java-Nachricht: {line[:100]}")

    @staticmethod
    def _event_reward(event_type, event_param):
        # Framework EventType values: BUMP=1, kills=2..5, JUMP=6, LAND=7,
        # COLLECT=8, HURT=9. Params use MarioForwardModel/SpriteType values.
        if event_type == 1 and event_param == 24:  # OBS_QUESTION_BLOCK
            return "Frageblock getroffen", 0.25
        if event_type == 1 and event_param == 22:  # OBS_BRICK
            return "Ziegelblock getroffen", 0.0
        if event_type in (2, 3, 4, 5):
            return "Gegner besiegt", 3.0
        if event_type == 9:
            return "Schaden erlitten", -10.0
        if event_type == 6:
            return "Sprung", 0.0
        if event_type == 7:
            return "Landung", 0.0
        return None, 0.0

    @staticmethod
    def _jump_context(observation):
        """Return whether the near-ahead grid contains a block or enemy."""
        if observation is None or len(observation) != 269:
            return False
        grid = observation[13:].reshape((16, 16))  # Java grid is [x][y].
        ahead = grid[9:13, 4:11]
        return bool(np.any((ahead >= 2) & (ahead <= 15)) or np.any(ahead >= 17))

    def set_curriculum_stage(self, stage):
        self.curriculum_stage = max(0, min(3, int(stage)))

    @staticmethod
    def action_name(action):
        names = (
            "nichts", "rechts", "rechts+schnell", "rechts+Sprung",
            "rechts+schnell+Sprung", "Sprung", "schnell+Sprung",
            "links", "links+schnell", "links+Sprung",
            "links+schnell+Sprung", "schnell",
        )
        return names[int(action)]

    def close(self):
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None

        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None

        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass
            self.server = None

        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
