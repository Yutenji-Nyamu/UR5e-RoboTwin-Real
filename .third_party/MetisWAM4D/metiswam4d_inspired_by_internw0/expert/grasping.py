"""Grasp sampling on the object's ground-truth surface points and the pick primitive of the scripted experts.

Gripper model in the grasp frame (columns of the ``link6`` rotation): ``a`` approach (= link6 +x), ``y`` closing axis,
``z = a x y``; the grasp centre sits ``PAD_CENTRE`` m along ``a`` from ``link6`` (X5 finger meshes: tips at 0.158,
palm face at 0.084, pads taper to about +-1 cm across the closing plane).  For a candidate approach line the centre is
placed ``DEPTH`` below the highest surface point inside the closing column (lifted when the tips would touch the
table), re-centred between the outermost pad contacts, and accepted when

  * the pad zone holds surface (``MIN_CONTACT`` points) with a closing width in [4 mm, ``MAX_WIDTH``],
  * no surface lies in the open fingers' columns (anything up-stream of the tips: the fingers come down along ``a``),
  * no surface lies inside the opening above the palm face.

Approach lines: a 1.5 cm grid over the object's footprint x 12 finger yaws, straight down; then 25 / 45 deg tilted
away from each arm's root (fingers horizontal).  Geometric cost = distance to the footprint centroid + width + tilt;
the cheapest are checked with IK (both finger signs, both arms) and ranked by joint distance from the current command.
"""
from __future__ import annotations

import math

import numpy as np

from metiswam4d_inspired_by_internw0.expert import skills
from metiswam4d_inspired_by_internw0.expert.skills import ARM_SLICES, CLOSED, OPEN, TABLE_Z, VMAX, pose7

PAD_CENTRE = 0.135         # link6 x of the grasp centre (pad middle)
TIP = 0.158
PALM = 0.084
OPEN_HALF = 0.0437         # inner pad face at full opening (m from the centre)
PAD_UP, PAD_DOWN = 0.020, 0.023
PAD_HALF_W = 0.010
FINGER_THICK = 0.020
FINGER_HALF_W = 0.020
DEPTH = 0.015              # grasp centre below the local top along the approach (m)
TIP_CLEAR = 0.003          # finger tips above the table (m)
MIN_CONTACT = 4
MAX_WIDTH = 0.080
CONTACT_BAND = 0.004       # surface within this of the outermost pad contact counts as that pad's contact patch
CONTACT_AREA = 0.02 * 0.03 # pad patch area whose full coverage scores quality 1 (m^2)
MIN_QUALITY = 0.08
GRID = 0.015
YAWS = 12
TILTS = (0.0, math.radians(25), math.radians(45))
AZIMUTHS = tuple(math.radians(d) for d in (0, -15, 15, -30, 30))   # tilted approach directions around the radial
PRE_BACKOFF = 0.09
EMPTY_GRIP = 0.002         # finger joint (m) below which the gripper closed on nothing
IK_TOPDOWN, IK_TILTED = 8, 4   # IK-checked grasps per approach group (top-down; per arm and tilt)


def grasp_rotation(approach, finger) -> np.ndarray:
    x = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(finger, float) - x * (np.asarray(finger, float) @ x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def evaluate(points: np.ndarray, starts: np.ndarray, rot: np.ndarray):
    """Approach lines through ``starts`` [C, 3] (on the table plane) with grasp frame ``rot`` ->
    (centres [C, 3], closing widths [C], valid [C])."""
    a, y, z = rot[:, 0], rot[:, 1], rot[:, 2]
    d = points[None] - starts[:, None]
    s, v, w = -(d @ a), d @ y, d @ z
    column = (np.abs(v) < OPEN_HALF) & (np.abs(w) <= PAD_HALF_W)
    top = np.where(column, s, -np.inf).max(axis=1)
    found = np.isfinite(top)
    centres = starts - a[None] * np.where(found, top - DEPTH, 0.0)[:, None]
    low = centres[:, 2] + a[2] * (TIP - PAD_CENTRE) - abs(z[2]) * FINGER_HALF_W - abs(y[2]) * OPEN_HALF
    raise_by = np.maximum(0.0, TABLE_Z + TIP_CLEAR - low) / max(-a[2], 1e-3)
    centres = centres - a[None] * raise_by[:, None]

    def pads(c):
        d = points[None] - c[:, None]
        u, v, w = d @ a, d @ y, d @ z
        inner = (u >= -PAD_UP) & (u <= PAD_DOWN) & (np.abs(w) <= PAD_HALF_W) & (np.abs(v) < OPEN_HALF)
        lo = np.where(inner, v, np.inf).min(axis=1)
        hi = np.where(inner, v, -np.inf).max(axis=1)
        return u, v, w, inner.sum(axis=1), lo, hi

    _, _, _, count, lo, hi = pads(centres)
    centres = centres + y[None] * np.where(count > 0, 0.5 * (lo + hi), 0.0)[:, None]
    u, v, w, count, lo, hi = pads(centres)
    width = np.where(count > 0, hi - lo, 0.0)
    inner = (u >= -PAD_UP) & (u <= PAD_DOWN) & (np.abs(w) <= PAD_HALF_W) & (np.abs(v) < OPEN_HALF)
    contact = np.minimum((inner & (v <= lo[:, None] + CONTACT_BAND)).sum(axis=1),
                         (inner & (v >= hi[:, None] - CONTACT_BAND)).sum(axis=1))
    av = np.abs(v)
    finger = (av >= OPEN_HALF - 0.002) & (av <= OPEN_HALF + FINGER_THICK) & (np.abs(w) <= FINGER_HALF_W) \
        & (u <= TIP - PAD_CENTRE)
    palm = (av < OPEN_HALF) & (np.abs(w) <= 0.03) & (u < PALM - PAD_CENTRE)
    valid = found & (count >= MIN_CONTACT) & (width >= 0.004) & (width <= MAX_WIDTH) \
        & ~finger.any(axis=1) & ~palm.any(axis=1)
    return centres, width, contact, valid


def sample(sc: skills.Scene, points: np.ndarray, density: float, arms, tilts) -> list[dict]:
    """Geometrically valid grasps, cheapest first (``{rot, centre, width, quality, tilt, arms, group}``); quality =
    the smaller pad's contact patch relative to ``CONTACT_AREA`` of surface."""
    xy = points[:, :2]
    centroid = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    cells = np.concatenate([centroid[None], np.unique(np.round(xy / GRID), axis=0) * GRID])
    starts = np.concatenate([cells, np.full((len(cells), 1), TABLE_Z)], axis=1)
    lines = []
    for k in range(YAWS):
        yaw = math.pi * k / YAWS
        lines.append((grasp_rotation([0, 0, -1], [math.cos(yaw), math.sin(yaw), 0]), 0.0, tuple(arms)))
    for arm in arms:
        base = np.asarray(sc.robots[arm].entity_origin_pose[:3], float)
        heading = math.atan2(centroid[1] - base[1], centroid[0] - base[0])
        for tilt in tilts:
            if tilt == 0:
                continue
            for offset in AZIMUTHS:
                radial = np.array([math.cos(heading + offset), math.sin(heading + offset), 0.0])
                approach = np.array([0.0, 0.0, -math.cos(tilt)]) + radial * math.sin(tilt)
                lines.append((grasp_rotation(approach, [-radial[1], radial[0], 0.0]), tilt, (arm,)))
    out = []
    for rot, tilt, owners in lines:
        group = "top" if tilt == 0 else f"arm{owners[0]}_tilt{math.degrees(tilt):.0f}"
        with np.errstate(invalid="ignore"):
            centres, width, contact, valid = evaluate(points, starts, rot)
        quality = np.minimum(1.0, contact / (density * CONTACT_AREA))
        seen = set()
        for i in np.flatnonzero(valid & (quality >= MIN_QUALITY)):
            key = tuple(np.round(centres[i] / 0.005).astype(int))
            if key in seen:
                continue
            seen.add(key)
            cost = 4.0 * float(np.linalg.norm(centres[i, :2] - centroid)) + 1.0 * (1.0 - float(quality[i])) \
                + 0.5 * tilt
            out.append({"rot": rot, "centre": centres[i], "width": float(width[i]), "quality": float(quality[i]),
                        "tilt": tilt, "arms": owners, "group": group, "geo": cost})
    out.sort(key=lambda g: g["geo"])
    return out


def lift_height(sc: skills.Scene, arm: int, grasp, q_grasp, lifts):
    """First height of ``lifts`` whose straight-up pose is reachable from the grasp (None if none is)."""
    for h in lifts:
        up = grasp.copy()
        up[2] += h
        if sc.ik(arm, up, seed=q_grasp) is not None:
            return h
    return None


def plan_grasps(sc: skills.Scene, label: str, arms=(0, 1), prefer_arm=None, tilts=TILTS, exclude=(), check=None,
                lifts=()):
    """Reachable grasps of ``label`` (pre-grasp, grasp and lifted IK; ``check(arm, grasp_pose, q_grasp)`` must hold
    too), cheapest first.  The best ``IK_TOPDOWN`` top-down grasps and ``IK_TILTED`` of every tilted approach are
    checked; ``exclude`` = centres tried before; ``lifts`` = acceptable lift heights, preferred first."""
    grasps = sample(sc, sc.object_points(label), sc.point_density(label), arms, tilts)
    grasps = [g for g in grasps if all(np.linalg.norm(g["centre"] - e) > 0.01 for e in exclude)]
    ranked, budget = [], {}
    rejected = sc.rejected = {"pre": 0, "grasp": 0, "check": 0, "lift": 0}
    for g in grasps:
        limit = IK_TOPDOWN if g["group"] == "top" else IK_TILTED
        if budget.get(g["group"], 0) >= limit:
            continue
        budget[g["group"]] = budget.get(g["group"], 0) + 1
        flips = (g["rot"], g["rot"] * np.array([1.0, -1.0, -1.0])) if g["tilt"] == 0 else (g["rot"],)
        for arm in g["arms"]:
            for rot in flips:
                link6 = g["centre"] - rot[:, 0] * PAD_CENTRE
                grasp, pre = pose7(link6, rot), pose7(link6 - rot[:, 0] * PRE_BACKOFF, rot)
                q_pre = sc.ik(arm, pre)
                if q_pre is None:
                    rejected["pre"] += 1
                    continue
                q_grasp = sc.ik(arm, grasp, seed=q_pre)
                if q_grasp is None:
                    rejected["grasp"] += 1
                    continue
                if check is not None and not check(arm, grasp, q_grasp):
                    rejected["check"] += 1
                    continue
                lift = lift_height(sc, arm, grasp, q_grasp, lifts) if lifts else 0.0
                if lift is None:
                    rejected["lift"] += 1
                    continue
                cost = g["geo"] + 0.3 * float(np.abs(q_pre - sc.cmd[ARM_SLICES[arm]]).sum())
                if prefer_arm is not None and arm != prefer_arm:
                    cost += 0.6
                ranked.append({**g, "arm": arm, "grasp": grasp, "pre": pre, "q_pre": q_pre, "q_grasp": q_grasp,
                               "lift": lift, "cost": cost})
    ranked.sort(key=lambda g: g["cost"])
    return ranked, len(grasps)


def pick(sc: skills.Scene, label: str, lift: float, arms=(0, 1), prefer_arm=None, tries: int = 3, reserve: int = 45,
         tilts=TILTS, check=None, min_lift=None, pace: float = 1.0, close=(5, 5)):
    """Grasp ``label`` and raise the gripper ``lift`` m (``min_lift`` where the full lift is out of reach); returns
    the holding arm (None after ``tries`` misses, when no grasp is reachable or fewer than ``reserve`` steps remain).
    A miss re-opens, backs off and re-plans from the object's current pose, skipping the centres already tried.
    ``check`` filters grasps (see ``plan_grasps``); ``pace`` scales the descent / lift speeds, ``close`` = gripper
    (closing, settling) steps."""
    if label not in sc.initial:
        sc.remember(label)
    tried = []
    while len(tried) < tries and sc.steps_left >= reserve:
        order, sampled = plan_grasps(sc, label, arms, prefer_arm, tilts, exclude=tried, check=check,
                                     lifts=(lift,) if min_lift is None else (lift, min_lift))
        moved = None
        for g in order[:4]:
            ok = yield from sc.move_to(g["arm"], g["pre"], grip=OPEN)
            if ok:
                moved = g
                break
        if moved is None:
            sc.note(f"{label}: no reachable grasp ({len(order)} ranked of {sampled} sampled, IK rejections "
                    f"{getattr(sc, 'rejected', {})})")
            return None
        g, arm = moved, moved["arm"]
        tried.append(g["centre"])
        sc.note(f"{label}: try {len(tried)} arm {arm} tilt {math.degrees(g['tilt']):.0f} width {g['width']:.3f} "
                f"centre {np.round(g['centre'], 3).tolist()} ({len(order)} ranked of {sampled} sampled)")
        yield from sc.hold(1)
        ok = yield from sc.move_line(arm, g["grasp"], vmax=0.03 * pace)
        if not ok:
            yield from sc.follow(arm, np.stack([sc.cmd[ARM_SLICES[arm]], g["q_grasp"]]), vmax=0.03 * pace)
        yield from sc.hold(2)
        err = np.linalg.norm(sc.ee_pose(arm)[:3] - g["grasp"][:3])
        yield from sc.gripper(arm, CLOSED, steps=close[0], settle=close[1])
        finger = sc.gripper_q(arm)
        if finger < EMPTY_GRIP:
            sc.note(f"{label}: empty grasp (finger {finger:.4f}, link6 error {err:.4f} m)")
            yield from sc.gripper(arm, OPEN, steps=3, settle=1)
            yield from sc.move_line(arm, g["pre"], vmax=VMAX)
            continue
        up = g["grasp"].copy()
        up[2] += g["lift"]
        ok = yield from sc.move_line(arm, up, vmax=0.035 * pace, pieces=5)
        if not ok:
            sc.note(f"{label}: no lift path")
            return None
        yield from sc.hold(3)
        rise, finger = sc.lift_of(label), sc.gripper_q(arm)
        if rise > 0.5 * g["lift"] and finger >= EMPTY_GRIP:
            sc.note(f"{label}: lifted {rise:.3f} m (finger {finger:.4f})")
            return arm
        sc.note(f"{label}: slipped (rise {rise:.3f} m, finger {finger:.4f}, link6 error {err:.4f} m)")
        yield from sc.gripper(arm, OPEN, steps=3, settle=3)
        yield from sc.move_line(arm, g["pre"], vmax=VMAX)
    return None
