"""Render local review frames from the explicit joint demonstration selection."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ur5e_real.adapters.robotwin_pi05.process_data import audit_selection, load_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes, _ = audit_selection(args.data_root, load_selection(args.selection))
    canvas = Image.new("RGB", (6 * 240, len(episodes) * 2 * 200), "white")
    draw = ImageDraw.Draw(canvas)
    for row, episode in enumerate(episodes):
        closed = int(np.flatnonzero(episode.gripper > 0.5)[0])
        opened = int(np.flatnonzero(np.diff(episode.gripper) < 0)[-1] + 1)
        indices = [
            0,
            closed,
            closed + int(np.argmax(episode.tcp[closed:opened, 2])),
            opened,
            min(opened + 2, len(episode.times) - 1),
            len(episode.times) - 2,
        ]
        for col, (label, i) in enumerate(zip(("start", "close", "highest", "open", "open+0.2s", "last obs"), indices)):
            for offset, (camera, paths) in enumerate((("head", episode.head), ("wrist", episode.wrist))):
                x, y = col * 240, (row * 2 + offset) * 200
                with Image.open(paths[i]) as original:
                    canvas.paste(original.resize((240, 180)), (x, y + 20))
                draw.text((x + 3, y + 3), f"{episode.run_id[-6:]} {camera} {label}", fill="black")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
