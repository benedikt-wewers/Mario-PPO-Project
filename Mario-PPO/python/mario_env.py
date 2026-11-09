from pathlib import Path
import socket
import subprocess

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarioEnv(gym.Env):
    """
    Gymnasium-Umgebung für das originale Mario-AI-Framework.

    Beobachtung:
      9 skalare Werte + 256 Werte aus dem Mario-Grid = 265 Werte.

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
            shape=(265,),
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
        if values.size != 265:
            raise RuntimeError(
                f"Falsche Beobachtungsgroesse: {values.size}, erwartet: 265."
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

        self._start_java()

        first_line = self._read_line()
        observation = self._parse_observation(first_line)

        self.last_observation = observation
        self.last_completion = float(observation[7])

        return observation, {"completion": self.last_completion}

    def step(self, action):
        if self.connection is None:
            raise RuntimeError("Keine aktive Java-Verbindung. Zuerst reset() aufrufen.")
        if self.episode_done:
            raise RuntimeError("Episode beendet. Vor dem nächsten Schritt reset() aufrufen.")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Ungueltige Aktion: {action}")

        self.connection.sendall(f"{action}\n".encode("utf-8"))
        line = self._read_line()

        if line.startswith("OBS,"):
            observation = self._parse_observation(line)
            completion = float(observation[7])

            # Einfacher erster Reward: Fortschritt nach vorne wird belohnt.
            reward = (completion - self.last_completion) * 100.0 - 0.01

            self.last_completion = completion
            self.last_observation = observation

            return observation, float(reward), False, False, {
                "completion": completion
            }

        if line.startswith("RESULT,"):
            parts = line.split(",", 2)
            completion = float(parts[1])
            status = parts[2].strip().upper()

            won = status == "WIN"
            reward = 100.0 if won else -100.0
            self.episode_done = True

            info = {
                "completion": completion,
                "game_status": status,
                "won": won,
            }

            return self.last_observation, reward, True, False, info

        raise RuntimeError(f"Unbekannte Java-Nachricht: {line[:100]}")

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
