"""Create an animated GIF from a recorded Mario training episode."""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Turn saved Mario PNG frames into a timelapse GIF."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Playback rate for the GIF (default: 15 fps).",
    )
    args = parser.parse_args()

    recordings_dir = Path(__file__).resolve().parents[1] / "recordings"
    frames = sorted(recordings_dir.glob("*-frame-*.png"))
    if not frames:
        raise SystemExit(f"Keine Aufnahmebilder gefunden in: {recordings_dir}")

    output = recordings_dir / "training-timelapse.gif"
    images = [Image.open(frame).convert("RGB") for frame in frames]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / max(1, args.fps))),
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()
    print(f"Timelapse erstellt: {output}")


if __name__ == "__main__":
    main()
