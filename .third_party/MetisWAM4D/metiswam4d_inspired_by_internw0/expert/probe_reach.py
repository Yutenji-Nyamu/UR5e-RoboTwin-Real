"""Offline reachability map of the X5 arm with RoboDojo's CuroboPlanner (no Isaac): IK success of grasp orientations
(top-down at several finger yaws, tilted approaches) over table positions, for the right arm.

    source <RoboDojo>/scripts/activate_robodojo.sh
    PYTHONPATH=<project>:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python -m metiswam4d_inspired_by_internw0.expert.probe_reach
"""
import math
import os
import sys

import numpy as np

ROOT = os.environ["ROBODOJO_ROOT"]
sys.path[:0] = [ROOT]
os.chdir(ROOT)


def main():
    from env.planner_manager.curobo_planner import CuroboPlanner
    from metiswam4d_inspired_by_internw0.expert.grasping import PAD_CENTRE, grasp_rotation
    from metiswam4d_inspired_by_internw0.expert.skills import pose7
    joints = [f"joint{i}" for i in range(1, 7)]
    root = [0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707]
    planner = CuroboPlanner(robot_origin_pose=[0, 0, 0, 1, 0, 0, 0], active_joints_name=joints, all_joints=joints,
                            dt=0.004, yml_path=os.path.join(ROOT, "Assets/Robots/x5/curobo.yml"),
                            table_height=0.74 - root[2])
    q0 = np.zeros(6, np.float32)
    home = np.array([0.3005, -0.3523, 0.9215])
    rot_home = grasp_rotation([0, 1, 0], [1, 0, 0])
    r = planner.solve_ik_to_joint(q0, list(pose7(home, rot_home)), real_robot_pose=root)
    print("home-like pose", r["status"], np.round(r.get("joint_value", []), 3))
    for z in (0.80, 0.85, 0.90):
        for name, approach, finger in (("down y0", [0, 0, -1], [1, 0, 0]), ("down y90", [0, 0, -1], [0, 1, 0]),
                                       ("tilt45 fwd", [0, math.sin(0.785), -math.cos(0.785)], [1, 0, 0]),
                                       ("tilt25 fwd", [0, math.sin(0.436), -math.cos(0.436)], [1, 0, 0]),
                                       ("horiz fwd", [0, 1, 0], [1, 0, 0])):
            rot = grasp_rotation(approach, finger)
            row = []
            for y in (-0.25, -0.15, -0.05, 0.05):
                for x in (0.1, 0.3, 0.45):
                    centre = np.array([x, y, z])
                    ok = planner.solve_ik_to_joint(q0, list(pose7(centre - rot[:, 0] * PAD_CENTRE, rot)),
                                                   real_robot_pose=root)["status"] == "Success"
                    row.append("#" if ok else ".")
            print(f"z {z:.2f} {name:11s} " + "".join(row))


if __name__ == "__main__":
    main()
