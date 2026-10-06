#!/usr/bin/env python3
"""Build alternative ICL demo maps for the Zero-WAM demo-sensitivity probe.

  official.py   the released mapping (copied)
  swap_task.py  each unseen task gets the official demo of ANOTHER unseen task (cyclic shift by 3)
  seen_task.py  each unseen task gets a demo of a seen RoboTwin task with unrelated semantics (fixed picks)

Paths point into ``human_data/robotwin/<run>/samples/<sample>/generated_video_kling-v3.mp4``; the client resolves
the precomputed latent under ``--icl_latent_root`` from the sample name, so the mp4 itself is not needed.
"""
from pathlib import Path
import re

ZW = Path("/m2v_intern_v3/danglingwei/codes/Zero-WAM_260923")
LATENTS = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/datas/HumanGen/human_latents/robotwin")
OUT = Path(__file__).resolve().parent / "maps"
RUN = "run_robotwin_20260728_013720_robotwin_n1000"
UNSEEN = ["place_object_scale", "stamp_seal", "open_microwave", "move_stapler_pad", "stack_blocks_three", "place_bread_basket", "place_empty_cup"]
# unrelated seen-task demos (sample names are stable in the released run; matched by prefix keyword)
SEEN_KEYWORDS = ["Shake_the_bottle", "Click_the_alarm_clock", "hammer", "Open_the_laptop", "lift_the_pot", "dustbin", "Hang_the_mug"]


def header():
    return ('"""Auto-generated ICL demo map (probe)."""\nfrom pathlib import Path\n'
            f'_ROOT = Path("{ZW}") / "data" / "HumanGen" / "human_data" / "robotwin" / "{RUN}" / "samples"\n'
            'def _video(s):\n    return str(_ROOT / s / "generated_video_kling-v3.mp4")\n')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = (ZW / "evaluation/robotwin/robotwin_icl_human_videos.py").read_text()
    official = dict(re.findall(r'"([a-z_]+)": \[_video\(\s*"([^"]+)"\s*\)\]', src))
    assert set(official) == set(UNSEEN), official.keys()
    (OUT / "official.py").write_text(src)
    shifted = {t: official[UNSEEN[(i + 3) % len(UNSEEN)]] for i, t in enumerate(UNSEEN)}
    body = "ROBOTWIN_ICL_HUMAN_VIDEOS = {\n" + "".join(f'    "{t}": [_video("{s}")],  # demo of {[k for k, v in official.items() if v == s][0]}\n' for t, s in shifted.items()) + "}\n"
    (OUT / "swap_task.py").write_text(header() + body)
    samples = sorted(p.name for p in (LATENTS / RUN / "samples").iterdir()) if (LATENTS / RUN / "samples").exists() else []
    picks = []
    for kw in SEEN_KEYWORDS:
        hit = [s for s in samples if kw.lower() in s.lower()]
        picks.append(hit[0] if hit else None)
    if all(picks):
        body = "ROBOTWIN_ICL_HUMAN_VIDEOS = {\n" + "".join(f'    "{t}": [_video("{s}")],\n' for t, s in zip(UNSEEN, picks)) + "}\n"
        (OUT / "seen_task.py").write_text(header() + body)
    else:
        print("seen-task samples not all found yet:", picks)
    print("maps written to", OUT)
    for t in UNSEEN:
        print(f"  {t:22s} official={official[t][:40]}...  swap={shifted[t][:40]}...")


if __name__ == "__main__":
    main()
