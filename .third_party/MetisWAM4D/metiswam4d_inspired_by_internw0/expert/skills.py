"""Scene access, planning and motion primitives of the scripted experts (inside the Isaac client, env 0).

Frames: every pose is ``[x y z qw qx qy qz]`` in the env-relative world frame (``get_real_endpose(is_relative=True)``,
``layout_manager.get_instance_pose``).  Arm poses are ``link6`` poses; the fingers point along ``link6`` +x (the
approach axis; tips at 0.158 m, palm face at 0.084 m) and close along ``link6`` y (inner pad faces at +-q, q = finger
joint, 0.0437 open).  The shared cuRobo planner of the two X5 arms takes ``link6`` targets in this frame plus the
arm's root pose.

Commands are the 14-D absolute joint vector ``[left arm 6, left gripper, right arm 6, right gripper]`` (gripper 1 =
open, 0 = closed) of ``joint_action.joints_to_action_dicts``.  An arm path from cuRobo (4 ms samples) is re-timed to
the 25 Hz action rate with a trapezoidal profile bounded by ``VMAX`` rad per action.
"""
from __future__ import annotations

import math

import numpy as np
import transforms3d as t3d

TABLE_Z = 0.765            # table top (env frame, m); the robot roots stand on it
VMAX = 0.06                # rad per action (1.5 rad/s at 25 Hz)
RAMP = 4                   # actions of acceleration / deceleration
OPEN, CLOSED = 1.0, 0.0
ARM_SLICES = (slice(0, 6), slice(7, 13))
GRIP_INDEX = (6, 13)


def pose7(position, rotation) -> np.ndarray:
    return np.concatenate([np.asarray(position, float).reshape(3), t3d.quaternions.mat2quat(rotation)])


def rotation_of(pose) -> np.ndarray:
    return t3d.quaternions.quat2mat(np.asarray(pose, float)[3:7])


def topdown_rotation(theta: float) -> np.ndarray:
    """``link6`` orientation pointing straight down with the finger axis (link6 y) at yaw ``theta``."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(theta), math.sin(theta), 0.0])
    return np.stack([x, y, np.cross(x, y)], axis=1)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def retime(path: np.ndarray, vmax: float = VMAX, ramp: int = RAMP) -> np.ndarray:
    """Dense joint path ``[N, 6]`` -> per-action targets with a trapezoidal speed profile (last = path end)."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return path[-1:].copy()
    seg = np.abs(np.diff(path, axis=0)).max(axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = s[-1]
    if length < 1e-4:
        return path[-1:].copy()
    n = int(math.ceil(length / vmax)) + ramp
    k = np.arange(n)
    w = np.minimum(1.0, np.minimum((k + 1) / ramp, (n - k) / ramp))
    target = length * np.cumsum(w) / w.sum()
    out = np.stack([np.interp(target, s, path[:, j]) for j in range(path.shape[1])], axis=1)
    out[-1] = path[-1]
    return out


def box_corners(bounds) -> np.ndarray:
    lo, hi = np.asarray(bounds[:3], float), np.asarray(bounds[3:6], float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def surface_points(obj, count: int, seed: int = 0):
    """Area-weighted surface samples of every USD mesh under a scene object's prim, in the frame of its root prim
    (rotation + translation of the root's world transform, scale kept in the points) -> (points, samples per m^2)."""
    import omni.usd
    from pxr import Usd, UsdGeom
    root = obj.stage.GetPrimAtPath(obj._prim_path)
    m_root = np.array(omni.usd.get_world_transform_matrix(root), float)        # row-vector convention
    u, _, vt = np.linalg.svd(m_root[:3, :3])
    r_root, t_root = u @ vt, m_root[3, :3]
    tris = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        index = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            continue
        m = np.array(omni.usd.get_world_transform_matrix(prim), float)
        p = np.asarray(points, float) @ m[:3, :3] + m[3, :3]
        p = (p - t_root) @ r_root.T
        index, start = np.asarray(index), 0
        for n in np.asarray(counts):
            face = index[start:start + n]
            start += n
            for k in range(1, n - 1):
                tris.append(p[[face[0], face[k], face[k + 1]]])
    if not tris:
        raise RuntimeError(f"no mesh under {obj._prim_path}")
    tris = np.stack(tris)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=count, p=area / area.sum())
    a, b = rng.random(count), rng.random(count)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    t = tris[pick]
    samples = t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0])
    return samples, count / float(area.sum())


class Scene:
    """Ground-truth state, planning and the running joint command of one episode."""

    def __init__(self, env):
        self.env = env
        self.rm = env.robot_manager
        self.layout = env.scene_manager.layout_manager
        self.robots = [r for r in self.rm.robot_list if r.type == "target"]   # [left, right]
        self.planner = [self.rm.planner[r.robot_name] for r in self.robots]
        self.home = [self.joints(i) for i in range(2)]
        self.cmd = np.zeros(14)
        for i in range(2):
            self.cmd[ARM_SLICES[i]] = self.home[i]
            self.cmd[GRIP_INDEX[i]] = OPEN
        self.notes: list[str] = []
        self.initial = {}
        self._points = {}

    # -- state ---------------------------------------------------------------------------------------------------
    @property
    def steps_left(self) -> int:
        return int(self.env.step_lim) - int(self.env.take_action_cnt[0])

    def note(self, text: str) -> None:
        self.notes.append(f"{int(self.env.take_action_cnt[0])}:{text}")
        print(f"[expert] step {int(self.env.take_action_cnt[0])}: {text}", flush=True)

    def joints(self, arm: int) -> np.ndarray:
        return np.asarray(self.rm.get_joint(self.robots[arm], env_idx_list=[0])[0], float)

    def ee_pose(self, arm: int) -> np.ndarray:
        return np.asarray(self.rm.get_real_endpose(self.robots[arm], env_idx_list=[0], is_relative=True)[0], float)

    def gripper_q(self, arm: int) -> float:
        return float(np.asarray(self.rm.get_end_effector_real_val(self.robots[arm], env_idx_list=[0])[0])[0])

    def instance(self, label: str) -> str:
        name = self.layout.get_instance_name(env_idx=0, label=label)
        if name is None:
            raise KeyError(f"no instance labelled {label}")
        return name

    def object_pose(self, label: str) -> np.ndarray:
        pos, rot = self.layout.get_instance_pose(env_idx=0, inst_name=self.instance(label))
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        rot = rot.detach().cpu().numpy() if hasattr(rot, "detach") else np.asarray(rot)
        return np.concatenate([np.asarray(pos, float).reshape(3), np.asarray(rot, float).reshape(4)])

    def object_corners(self, label: str) -> np.ndarray:
        """World corners ``[8, 3]`` of the object's oriented bounding box at its current pose."""
        name = self.instance(label)
        local = self.layout.get_instance_bbox_vertices(inst_name=name, env_idx=0)
        if isinstance(local, tuple):                    # rigid.get_bbox(): (pos, ori, [min xyz, max xyz] local)
            local = box_corners(local[2])
        local = np.asarray(local, float).reshape(-1, 3)
        pose = self.object_pose(label)
        return local @ rotation_of(pose).T + pose[:3]

    def object_points(self, label: str, count: int = 2500) -> np.ndarray:
        """World surface points ``[count, 3]`` of the object at its current pose (area-weighted samples of every
        mesh under its prim, cached in the object's root frame)."""
        if label not in self._points:
            self._points[label] = surface_points(self.layout.get_scene_object(env_idx=0, inst_name=self.instance(label)),
                                                 count)
        pose = self.object_pose(label)
        return self._points[label][0] @ rotation_of(pose).T + pose[:3]

    def point_density(self, label: str) -> float:
        """Surface samples per m^2 of ``object_points(label)``."""
        self.object_points(label)
        return self._points[label][1]

    def remember(self, label: str) -> None:
        self.initial[label] = self.object_pose(label)

    def lift_of(self, label: str) -> float:
        return float(self.object_pose(label)[2] - self.initial[label][2])

    # -- planning ------------------------------------------------------------------------------------------------
    def ik(self, arm: int, pose, seed=None):
        seed = self.cmd[ARM_SLICES[arm]] if seed is None else seed
        result = self.planner[arm].solve_ik_to_joint(np.asarray(seed, np.float32), list(map(float, pose)),
                                                     real_robot_pose=list(self.robots[arm].entity_origin_pose))
        if result.get("status") != "Success":
            return None
        return np.asarray(result["joint_value"], float).reshape(6)

    def plan(self, arm: int, pose, start=None):
        """Collision-aware cuRobo path (dense ``[N, 6]``) from ``start`` (default: the commanded joints) to ``pose``."""
        start = self.cmd[ARM_SLICES[arm]] if start is None else start
        try:
            result = self.planner[arm].plan_path(np.asarray(start, np.float32), list(map(float, pose)),
                                                 real_robot_pose=list(self.robots[arm].entity_origin_pose))
        except Exception as error:  # noqa: BLE001 - a planner failure is a failed candidate
            self.note(f"plan_path raised {type(error).__name__}: {error}")
            return None
        if result.get("status") != "Success":
            return None
        return np.concatenate([np.asarray(start, float)[None], np.asarray(result["position"], float)])

    def plan_joint(self, arm: int, goal, start=None):
        start = self.cmd[ARM_SLICES[arm]] if start is None else start
        try:
            result = self.planner[arm].plan_joint(np.asarray(start, np.float32), np.asarray(goal, np.float32))
        except Exception as error:  # noqa: BLE001
            self.note(f"plan_joint raised {type(error).__name__}: {error}")
            return None
        if result.get("status") != "Success":
            return None
        return np.concatenate([np.asarray(start, float)[None], np.asarray(result["position"], float)])

    def line(self, arm: int, start_q, goal_pose, pieces: int = 4, max_jump: float = 0.5):
        """Joint path along a short straight Cartesian segment: IK at ``pieces`` points, each seeded by the previous
        solution; None when a solution jumps (branch change)."""
        start_pose = self.fk_pose(arm, start_q)
        if start_pose is None:
            return None
        rot = rotation_of(goal_pose)
        q = np.asarray(start_q, float)
        path = [q]
        for k in range(1, pieces + 1):
            p = start_pose[:3] + (np.asarray(goal_pose[:3]) - start_pose[:3]) * k / pieces
            nxt = self.ik(arm, pose7(p, rot), seed=q)
            if nxt is None or np.abs(nxt - q).max() > max_jump:
                return None
            path.append(nxt)
            q = nxt
        return np.stack(path)

    def fk_pose(self, arm: int, q):
        """``link6`` pose of joint vector ``q`` (cuRobo kinematics, mapped back to the env frame)."""
        import torch
        from curobo.types import JointState
        planner = self.planner[arm]
        full = planner._build_cspace_joint_values(np.asarray(q, np.float32))
        js = JointState.from_position(torch.as_tensor(full, dtype=torch.float32, device=planner.device_cfg.device)
                                      .reshape(1, -1), joint_names=planner.cspace_joint_names)
        kin = planner.motion_planner.compute_kinematics(js)
        tool = kin.tool_poses.to_dict()[planner.ee_link]
        p = tool.position.detach().cpu().numpy().reshape(-1)[:3] - np.asarray(planner.frame_bias, float)
        quat = tool.quaternion.detach().cpu().numpy().reshape(-1)[:4]
        base = np.asarray(self.robots[arm].entity_origin_pose, float)
        rb = rotation_of(base)
        return pose7(rb @ p + base[:3], rb @ t3d.quaternions.quat2mat(quat))

    # -- motion --------------------------------------------------------------------------------------------------
    def follow(self, arm: int, path, vmax: float = VMAX, grip=None, other=None):
        """Yield commands moving ``arm`` along ``path`` (re-timed); ``other`` = optional (arm, path) moved at the same
        time (the shorter one holds its end)."""
        mine = retime(path, vmax)
        theirs = retime(other[1], vmax) if other is not None else None
        n = max(len(mine), len(theirs) if theirs is not None else 0)
        for k in range(n):
            self.cmd[ARM_SLICES[arm]] = mine[min(k, len(mine) - 1)]
            if theirs is not None:
                self.cmd[ARM_SLICES[other[0]]] = theirs[min(k, len(theirs) - 1)]
            if grip is not None:
                self.cmd[GRIP_INDEX[arm]] = grip
            yield self.cmd.copy()

    def hold(self, steps: int):
        for _ in range(steps):
            yield self.cmd.copy()

    def gripper(self, arm: int, value: float, steps: int = 6, settle: int = 4):
        start = self.cmd[GRIP_INDEX[arm]]
        for k in range(1, steps + 1):
            self.cmd[GRIP_INDEX[arm]] = start + (value - start) * k / steps
            yield self.cmd.copy()
        yield from self.hold(settle)

    def move_to(self, arm: int, pose, vmax: float = VMAX, grip=None):
        """cuRobo path to ``pose``; fallbacks: cuRobo joint-space plan to its IK solution, then joint interpolation
        (joint change < 2 rad).  Returns False when unreachable."""
        path = self.plan(arm, pose)
        if path is None:
            q = self.ik(arm, pose)
            path = self.plan_joint(arm, q) if q is not None else None
            if path is None and q is not None and np.abs(q - self.cmd[ARM_SLICES[arm]]).max() < 2.0:
                self.note(f"arm {arm}: cuRobo plans failed, joint interpolation")
                path = np.stack([self.cmd[ARM_SLICES[arm]], q])
            if path is None:
                self.note(f"arm {arm}: unreachable ({'no IK' if q is None else 'no plan'})")
        if path is None:
            return False
        travel = np.abs(np.diff(path, axis=0)).sum(axis=0)
        self.note(f"arm {arm}: move {len(retime(path, vmax))} steps, joint travel {np.round(travel, 2).tolist()}, "
                  f"net {np.round(path[-1] - path[0], 2).tolist()}")
        yield from self.follow(arm, path, vmax=vmax, grip=grip)
        return True

    def move_line(self, arm: int, pose, vmax: float = 0.03, pieces: int = 4, max_jump: float = 0.5):
        """Straight Cartesian segment to ``pose``; fallbacks: cuRobo path, joint interpolation to the goal's IK
        solution (joint change < 1.5 rad).  Returns False when unreachable."""
        start = self.cmd[ARM_SLICES[arm]]
        path = self.line(arm, start, pose, pieces=pieces, max_jump=max_jump)
        if path is None:
            path = self.plan(arm, pose)
        if path is None:
            q = self.ik(arm, pose, seed=start)
            if q is not None and np.abs(q - start).max() < 1.5:
                path = np.stack([start, q])
        if path is None:
            return False
        yield from self.follow(arm, path, vmax=vmax)
        return True

    def go_home(self, arms=(0, 1)):
        """Both (or the given) arms back to their initial joints at the same time."""
        paths = []
        for arm in arms:
            path = self.plan_joint(arm, self.home[arm])
            if path is None:
                path = np.stack([self.cmd[ARM_SLICES[arm]], self.home[arm]])
            paths.append((arm, path))
        if len(paths) == 1:
            yield from self.follow(*paths[0])
        else:
            yield from self.follow(paths[0][0], paths[0][1], other=paths[1])

