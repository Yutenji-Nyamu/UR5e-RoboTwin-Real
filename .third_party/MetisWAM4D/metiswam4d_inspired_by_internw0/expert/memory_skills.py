"""Primitives shared by the Memory-task experts: spring-button presses, pick-and-place of a block onto a target pose,
both arms moving at once.

Buttons (``SpringButton``): the reward reads the ``press`` functional joint's ratio ``(q - lower) / (upper - lower)``
once per action; a press counts when it goes from above 0.95 to below 0.5 and the ordered checks need one action
below 0.5, then one above 0.9.  The expert presses with closed fingers, straight down on the cap centre, along a
joint-interpolated stroke between a hover and a press pose that are solved once per button, and holds each phase
until the ratio confirms it.
"""
from __future__ import annotations

import math

import numpy as np

from metiswam4d_inspired_by_internw0.expert import grasping, skills
from metiswam4d_inspired_by_internw0.expert.skills import ARM_SLICES, CLOSED, GRIP_INDEX, OPEN, pose7

TIP = grasping.TIP
HOVER = 0.015              # closed finger tips above the cap at the hover pose (m)
PRESS_EXTRA = 0.004        # tips below the fully pressed cap at the press pose (m)
STROKE = 4                 # actions per stroke (down or up)
DROP_GAP = 0.012           # held block bottom above the support at release (finger tips reach below the block)
RETREAT = 0.06


# -- buttons ---------------------------------------------------------------------------------------------------------
def press_joint(sc: skills.Scene, label: str):
    name = sc.instance(label)
    meta = sc.layout.get_instance_metadata(inst_name=name, env_idx=0)
    joints = meta["passive"]["functional"]["press"]["parent_joint"]
    joint = joints if isinstance(joints, str) else joints[0]
    return sc.layout.get_scene_object(env_idx=0, inst_name=name), joint


def button_ratio(sc: skills.Scene, label: str) -> float:
    inst, joint = press_joint(sc, label)
    info = inst.get_joint_info(joint)
    return float((info["position"] - info["lower"]) / (info["upper"] - info["lower"]))


def button_travel(sc: skills.Scene, label: str) -> float:
    inst, joint = press_joint(sc, label)
    info = inst.get_joint_info(joint)
    return float(abs(info["upper"] - info["lower"]))


def button_top(sc: skills.Scene, label: str) -> np.ndarray:
    """Cap centre: the pose's xy, the top of the bounding box."""
    pose, corners = sc.object_pose(label), sc.object_corners(label)
    return np.array([pose[0], pose[1], corners[:, 2].max()])


def topdown_at(sc: skills.Scene, arm: int, tip, seed=None, yaws: int = 24):
    """Straight-down ``link6`` pose with the closed finger tips at ``tip`` and its IK solution, the yaw closest to
    ``seed`` (default: the arm's initial joints) in joint space; None when unreachable."""
    seed = sc.home[arm] if seed is None else np.asarray(seed, float)
    best = None
    for k in range(yaws):
        rot = skills.topdown_rotation(2 * math.pi * k / yaws)
        pose = pose7(np.asarray(tip, float) - rot[:, 0] * TIP, rot)
        q = sc.ik(arm, pose, seed=seed)
        if q is None:
            continue
        cost = float(np.abs(q - seed).sum())
        if best is None or cost < best[2]:
            best = (pose, q, cost)
    return None if best is None else best[:2]


def button_poses(sc: skills.Scene, arm: int, label: str):
    """(hover pose, hover joints, press joints) of ``arm`` over ``label``; None when unreachable."""
    top = button_top(sc, label)
    hover = topdown_at(sc, arm, top + [0, 0, HOVER])
    if hover is None:
        return None
    press = top.copy()
    press[2] -= button_travel(sc, label) + PRESS_EXTRA
    q_press = sc.ik(arm, pose7(press - skills.rotation_of(hover[0])[:, 0] * TIP, skills.rotation_of(hover[0])),
                    seed=hover[1])
    if q_press is None:
        return None
    return hover[0], hover[1], q_press


def stroke(sc: skills.Scene, arm: int, goal, steps: int = STROKE):
    start = sc.cmd[ARM_SLICES[arm]].copy()
    for k in range(1, steps + 1):
        sc.cmd[ARM_SLICES[arm]] = start + (np.asarray(goal, float) - start) * k / steps
        yield sc.cmd.copy()


def press(sc: skills.Scene, arm: int, label: str, times: int = 1, poses=None):
    """Press ``label`` ``times`` times with ``arm`` (gripper closed, starting anywhere: the arm first moves to the
    hover pose); returns the number of confirmed presses (ratio below 0.5, then back above 0.95)."""
    poses = poses or button_poses(sc, arm, label)
    if poses is None:
        sc.note(f"{label}: no press pose for arm {arm}")
        return 0
    hover, q_hover, q_press = poses
    if sc.cmd[GRIP_INDEX[arm]] > 0.5:
        yield from sc.gripper(arm, CLOSED, steps=3, settle=0)
    if np.abs(sc.cmd[ARM_SLICES[arm]] - q_hover).max() > 0.02:
        path = sc.plan_joint(arm, q_hover)
        if path is None:
            path = np.stack([sc.cmd[ARM_SLICES[arm]], q_hover])
        yield from sc.follow(arm, path)
        yield from sc.hold(1)
    done = 0
    for _ in range(times):
        deeper = q_press
        yield from stroke(sc, arm, deeper)
        for extra in range(4):
            yield from sc.hold(1)
            if button_ratio(sc, label) < 0.3:
                break
            if extra == 1:                     # not down yet: push 4 mm further along the approach
                pose = sc.fk_pose(arm, deeper)
                pose[2] -= 0.004
                nxt = sc.ik(arm, pose, seed=deeper)
                if nxt is not None:
                    deeper = nxt
                    yield from stroke(sc, arm, deeper, steps=1)
        down = button_ratio(sc, label)
        yield from sc.hold(1)
        yield from stroke(sc, arm, q_hover)
        for _ in range(8):
            if button_ratio(sc, label) > 0.96:
                break
            yield from sc.hold(1)
        up = button_ratio(sc, label)
        if down < 0.5 and up > 0.95:
            done += 1
        else:
            sc.note(f"{label}: press not confirmed (down ratio {down:.2f}, up ratio {up:.2f})")
        yield from sc.hold(1)
    return done


# -- pick and place --------------------------------------------------------------------------------------------------
def target_delta(sc: skills.Scene, label: str, xy, bottom_z: float) -> np.ndarray:
    """Translation that brings ``label``'s pose origin to ``xy`` and its bottom to ``bottom_z``."""
    pose, corners = sc.object_pose(label), sc.object_corners(label)
    return np.array([xy[0] - pose[0], xy[1] - pose[1], bottom_z - corners[:, 2].min()])


def reachable_after(sc: skills.Scene, delta: np.ndarray):
    """Grasp filter: the grasp translated by ``delta`` is reachable 3 cm above and at the drop height."""
    def check(arm, grasp, q_grasp):
        drop = grasp.copy()
        drop[:3] += delta
        above = drop.copy()
        above[2] += 0.03
        q = sc.ik(arm, above, seed=q_grasp)
        return q is not None and sc.ik(arm, drop, seed=q) is not None
    return check


def place_at(sc: skills.Scene, arm: int, label: str, xy, bottom_z: float, carry_z: float | None = None):
    """Put the object held by ``arm`` with its pose origin at ``xy`` and its bottom at ``bottom_z`` (same gripper
    orientation, straight Cartesian lines: at ``carry_z`` gripper height if given, to 3 cm above the drop, down),
    release and back off upwards; returns False when the drop is unreachable."""
    ee = sc.fk_pose(arm, sc.cmd[ARM_SLICES[arm]])
    drop = ee.copy()
    drop[:3] += target_delta(sc, label, xy, bottom_z)
    above = drop.copy()
    above[2] += 0.03
    if carry_z is not None:
        above[2] = max(above[2], carry_z)
    tilted = skills.rotation_of(ee)[2, 0] > -0.95
    vmax = 0.025 if tilted else 0.04
    ok = yield from sc.move_line(arm, above, vmax=vmax, pieces=16, max_jump=0.25)
    if not ok:
        ok = yield from sc.move_to(arm, above, vmax=vmax)
    if not ok:
        sc.note(f"{label}: above-drop pose unreachable")
        return False
    yield from sc.hold(1)
    ok = yield from sc.move_line(arm, drop, vmax=0.03, pieces=4)
    if not ok:
        sc.note(f"{label}: drop pose unreachable")
        return False
    yield from sc.hold(2)
    yield from sc.gripper(arm, OPEN, steps=3, settle=2)
    back = drop.copy()
    back[2] += RETREAT
    ok = yield from sc.move_line(arm, back, vmax=0.06)
    if not ok:
        yield from sc.go_home((arm,))
    return True


def transfer(sc: skills.Scene, label: str, xy, bottom_z: float, lift: float, xy_tol: float, prefer_arm=None,
             arms=(0, 1), attempts: int = 3, reserve: int = 60, pace: float = 1.4, close=(4, 3)):
    """Pick ``label`` with an arm that also reaches the drop, place it at ``xy`` / ``bottom_z``; re-pick while the
    resting xy error exceeds ``xy_tol``.  Returns the arm that placed it (None on failure)."""
    for attempt in range(attempts):
        sc.remember(label)
        arm = None
        for tilts in ((0.0,), grasping.TILTS):       # top-down grasps carry blocks without slipping; tilted if none
            arm = yield from grasping.pick(sc, label, lift, arms=arms, prefer_arm=prefer_arm, tries=2,
                                           reserve=reserve, tilts=tilts,
                                           check=reachable_after(sc, target_delta(sc, label, xy, bottom_z)),
                                           pace=pace, close=close)
            if arm is not None or "no reachable grasp" not in sc.notes[-1]:
                break
        if arm is None:
            return None
        ok = yield from place_at(sc, arm, label, xy, bottom_z)
        if not ok:
            return None
        yield from sc.hold(2)
        pose = sc.object_pose(label)
        err = float(np.hypot(pose[0] - xy[0], pose[1] - xy[1]))
        sc.note(f"{label}: placed with arm {arm}, xy error {err:.4f} m")
        if err <= xy_tol:
            return arm
    return None


# -- both arms -------------------------------------------------------------------------------------------------------
def parallel(sc: skills.Scene, *generators):
    """Run single-arm primitives (each writes only its own arm / gripper entries of ``sc.cmd``) side by side, one
    action per step; a finished one holds.  Returns their return values."""
    results = [None] * len(generators)
    live = list(range(len(generators)))
    while live:
        for i in list(live):
            try:
                next(generators[i])
            except StopIteration as stop:
                results[i] = stop.value
                live.remove(i)
        if live:
            yield sc.cmd.copy()
    return results


def after(sc: skills.Scene, steps: int, generator):
    """``generator`` delayed by ``steps`` held actions (for ``parallel``)."""
    for _ in range(steps):
        yield None
    result = yield from generator
    return result


def joints_together(sc: skills.Scene, goals: dict):
    """Both arms to joint goals ``{arm: q}`` at the same time (cuRobo joint plans, interpolation fallback)."""
    paths = []
    for arm, q in goals.items():
        path = sc.plan_joint(arm, q)
        if path is None:
            path = np.stack([sc.cmd[ARM_SLICES[arm]], np.asarray(q, float)])
        paths.append((arm, path))
    if len(paths) == 1:
        yield from sc.follow(*paths[0])
    elif paths:
        yield from sc.follow(paths[0][0], paths[0][1], other=paths[1])
