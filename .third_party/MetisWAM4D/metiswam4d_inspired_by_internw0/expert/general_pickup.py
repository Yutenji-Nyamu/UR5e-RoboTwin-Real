"""general_pickup: lift the object labelled ``target`` by more than 10 cm (step limit 200).

``grasping.pick`` with a 15 cm lift: grasps sampled on the target's ground-truth surface points (top-down first,
tilted approaches for far objects), both arms, the arm on the target's side preferred; a miss re-opens and tries the
next grasp.  The environment ends the episode as soon as the lift check passes.
"""
from __future__ import annotations

import numpy as np

from metiswam4d_inspired_by_internw0.expert import grasping, skills
from metiswam4d_inspired_by_internw0.expert.skills import CLOSED, OPEN

LABEL = "target"
LIFT = 0.15
MIN_LIFT = 0.125


def episode(sc: skills.Scene):
    sc.remember(LABEL)
    x = float(sc.object_corners(LABEL)[:, 0].mean())
    arm = yield from grasping.pick(sc, LABEL, LIFT, prefer_arm=0 if x < 0 else 1, min_lift=MIN_LIFT)
    if arm is None:
        sc.note("expert gave up")
        return
    yield from sc.hold(10)
    sc.note(f"held, lift {sc.lift_of(LABEL):.3f} m")


def probe(sc: skills.Scene):
    """Frame / gripper geometry checks printed at the start of a development episode."""
    for arm in range(2):
        q = sc.joints(arm)
        ee = sc.ee_pose(arm)
        fk = sc.fk_pose(arm, q)
        ik = sc.ik(arm, ee, seed=q)
        sc.note(f"arm {arm} root {list(sc.robots[arm].entity_origin_pose)} q {np.round(q, 3).tolist()} ee "
                f"{np.round(ee, 4).tolist()} fk-ee {np.round(fk - ee, 4).tolist()} ik-q "
                f"{None if ik is None else np.round(ik - q, 3).tolist()}")
    for value in (CLOSED, OPEN):
        yield from sc.gripper(0, value, steps=4, settle=8)
        ee = sc.ee_pose(0)
        r = skills.rotation_of(ee)
        rel = {}
        for link in ("link7", "link8"):
            p = np.asarray(sc.rm.get_link_pose(sc.robots[0], link, env_idx_list=[0], is_relative=True)[0], float)
            rel[link] = np.round(r.T @ (p[:3] - ee[:3]), 4).tolist()
        sc.note(f"gripper cmd {value} finger q {sc.gripper_q(0):.4f} link7/8 in link6 {rel} "
                f"bounds {finger_bounds(sc, 0)}")


def finger_bounds(sc: skills.Scene, arm: int):
    """Finger geometry extent in the link6 frame (USD world bounds of link7 / link8)."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        ee = sc.ee_pose(arm)
        r = skills.rotation_of(ee)
        origin = np.asarray(sc.rm.scene.env_origins[0].detach().cpu().numpy(), float)
        out = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/envs/env_0/robot{arm}")):
            if prim.GetName() in ("link7", "link8"):
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                pts = skills.box_corners(np.concatenate([np.array(box.GetMin()), np.array(box.GetMax())]))
                loc = (pts - origin - ee[:3]) @ r
                out[prim.GetName()] = [np.round(loc.min(0), 4).tolist(), np.round(loc.max(0), 4).tolist()]
        return out
    except Exception as error:  # noqa: BLE001 - diagnostics only
        return f"{type(error).__name__}: {error}"
