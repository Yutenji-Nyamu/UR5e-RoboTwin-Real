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
    columns = 3 * episodes[0].audit["gripper_cycles"] + 3
    canvas = Image.new("RGB", (columns * 240, len(episodes) * 2 * 200), "white")
    draw = ImageDraw.Draw(canvas)
    for row, episode in enumerate(episodes):
        closes = np.flatnonzero(np.diff(episode.gripper, prepend=0) > 0)
        opens = np.flatnonzero(np.diff(episode.gripper, prepend=0) < 0)
        expected = episode.audit["gripper_cycles"]
        if len(closes) != expected or len(opens) != expected:
            raise ValueError(f"{episode.run_id}: resampled labels lost a gripper cycle")
        points = [("start", 0)]
        for cycle, (closed, opened) in enumerate(zip(closes, opens), 1):
            points.extend(
                [
                    (f"close {cycle}", int(closed)),
                    (f"highest {cycle}", int(closed + np.argmax(episode.tcp[closed:opened, 2]))),
                    (f"open {cycle}", int(opened)),
                ]
            )
        points.extend(
            [
                (
                    "last cmd open+0.2s",
                    min(
                        int(
                            np.searchsorted(
                                episode.times, episode.audit["gripper_events"][-1]["controller_time_s"] + 0.2 - 1e-6
                            )
                        ),
                        len(episode.times) - 1,
                    ),
                ),
                ("last obs", len(episode.times) - 2),
            ]
        )
        for col, (label, i) in enumerate(points):
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
