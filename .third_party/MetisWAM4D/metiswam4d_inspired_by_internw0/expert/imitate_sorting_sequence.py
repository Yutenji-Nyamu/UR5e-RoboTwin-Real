"""imitate_sorting_sequence: a support Franka first replays a demonstration that puts aim0..aim4 into basket1 in some
order; the two X5 arms must then put the matching objects t* into basket0 (x = -0.4, left) in the same order, with
the aims left in basket1, and finish with both arms at their initial pose (step limit 1600).  Moving an X5 arm more
than 0.3 m from its start while the support arm is away from its own start, or putting an object into basket0 out of
order, fails the episode.

The order is the env's ``target_label`` (read from the support-arm trajectory file, the ground truth the reward
uses).  The expert holds both arms still until the support arm's action queue has drained (about 500 steps), then
for each object in order: the left arm picks it (``grasping.pick``, only grasps whose translated copy reaches the
drop above basket0) and drops it 2 cm above the basket rim, centred on the basket.  Objects the left arm cannot
take top-down are relayed by the right arm to the table centre (the layout's object-free area) first; the relay of the next
object runs while the left arm carries the current one to the basket.
"""
from __future__ import annotations

import os

import numpy as np

from metiswam4d_inspired_by_internw0.expert import grasping, memory_skills, skills
from metiswam4d_inspired_by_internw0.expert.skills import TABLE_Z

if os.environ.get("METIS_EXPERT_RELOAD") == "1":
    import importlib
    memory_skills = importlib.reload(memory_skills)

LEFT, RIGHT = 0, 1
BASKET = "basket0"
RELAY = np.array([0.0, -0.175])          # centre of the layout's prohibited (object-free) area
RIM_GAP = 0.02
WAIT_LIMIT = 900
PACE = 1.3


def basket_drop(sc: skills.Scene):
    pose, corners = sc.object_pose(BASKET), sc.object_corners(BASKET)
    return pose[:2].copy(), float(corners[:, 2].max()) + RIM_GAP


def basket_check(sc: skills.Scene, label: str):
    xy, bottom = basket_drop(sc)
    return memory_skills.reachable_after(sc, memory_skills.target_delta(sc, label, xy, bottom))


def left_reaches(sc: skills.Scene, label: str) -> bool:
    ranked, _ = grasping.plan_grasps(sc, label, arms=(LEFT,), tilts=(0.0,), check=basket_check(sc, label))
    return bool(ranked)


def in_basket(sc: skills.Scene, label: str) -> bool:
    rm = sc.env.reward_manager
    return rm.call_func_parser(rm.is_A_in_B(label_A=label, label_B=BASKET), 0) >= 1


def relay(sc: skills.Scene, label: str):
    """Right arm: ``label`` to the table centre, then home; returns True when it rests there."""
    arm = yield from memory_skills.transfer(sc, label, RELAY, TABLE_Z + memory_skills.DROP_GAP, 0.06, 0.06,
                                            arms=(RIGHT,), attempts=2, reserve=150, pace=PACE, close=(4, 4))
    yield from sc.go_home((RIGHT,))
    return arm is not None


def left_pick(sc: skills.Scene, label: str):
    xy, bottom = basket_drop(sc)
    sc.remember(label)
    lift = max(0.10, bottom + 0.03 - sc.object_corners(label)[:, 2].min())
    arm = None
    for tilts in ((0.0,), grasping.TILTS):
        arm = yield from grasping.pick(sc, label, lift, arms=(LEFT,), tries=2, reserve=80, tilts=tilts,
                                       check=basket_check(sc, label), min_lift=0.08, pace=PACE)
        if arm is not None or "no reachable grasp" not in sc.notes[-1]:
            break
    return arm is not None


def drop(sc: skills.Scene, label: str):
    xy, bottom = basket_drop(sc)
    ok = yield from memory_skills.place_at(sc, LEFT, label, xy, bottom)
    yield from sc.hold(3)
    return ok


def episode(sc: skills.Scene):
    env = sc.env
    order = [env.target_label[i][0] for i in range(5)]
    sc.note(f"order {order}")
    waited = 0
    while waited < WAIT_LIMIT and not (env.query_support_times[0] > 0 and len(env.support_arm_action[0]) == 0):
        yield from sc.hold(1)
        waited += 1
    yield from sc.hold(5)
    sc.note(f"support arm done after {waited} steps")
    relayed = set()
    for k, label in enumerate(order):
        for attempt in range(3):
            if label not in relayed and not left_reaches(sc, label):
                sc.note(f"{label}: relay through the table centre")
                ok = yield from relay(sc, label)
                if not ok:
                    sc.note("expert gave up")
                    return
                relayed.add(label)
            ok = yield from left_pick(sc, label)
            if not ok:
                relayed.discard(label)
                continue
            nxt = order[k + 1] if k + 1 < len(order) else None
            if nxt is not None and nxt not in relayed and not left_reaches(sc, nxt):
                sc.note(f"{nxt}: relay through the table centre while {label} goes to the basket")
                ok, done = yield from memory_skills.parallel(sc, drop(sc, label),
                                                             memory_skills.after(sc, 4, relay(sc, nxt)))
                if done:
                    relayed.add(nxt)
            else:
                ok = yield from drop(sc, label)
            inside = in_basket(sc, label)
            sc.note(f"{label}: dropped, in basket {inside}")
            if inside:
                break
            relayed.discard(label)
        else:
            sc.note("expert gave up")
            return
    yield from sc.go_home()
    yield from sc.hold(8)
    sc.note("home")
