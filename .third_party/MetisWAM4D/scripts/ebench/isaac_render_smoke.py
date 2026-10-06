"""Headless Isaac Sim 4.1 boot + rendered RGB / depth frames, to check Vulkan / RTX and render settings on a node.

GPU selection goes through Isaac's own settings with all GPUs visible: CUDA_VISIBLE_DEVICES only restricts CUDA
while Vulkan still enumerates every GPU, which breaks RTX's Vulkan-CUDA interop.

Prints a noise measure (std of the Laplacian over a flat ground patch) so render settings can be compared:

    source scripts/ebench/isaac_env.sh && /usr/local/ebench/genmanip-venv/bin/python \
        scripts/ebench/isaac_render_smoke.py --gpu 1 --out /tmp/isaac_smoke.png [--setting /rtx/post/aa/op=1 ...]
"""

import argparse
import json
import time

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--out", default="/tmp/isaac_smoke.png")
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--setting", action="append", default=[], help="carb setting KEY=JSON_VALUE, applied after boot")
parser.add_argument("--hold", type=float, default=0.0, help="keep stepping with rendering this many seconds after")
args = parser.parse_args()

t0 = time.time()
app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": args.gpu, "physics_gpu": args.gpu})
print(f"STAGE app_ready {time.time() - t0:.1f}s", flush=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid  # noqa: E402
from omni.isaac.core.utils.prims import create_prim  # noqa: E402
from omni.isaac.sensor import Camera  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

settings = carb.settings.get_settings()
for item in args.setting:
    key, value = item.split("=", 1)
    settings.set(key, json.loads(value))
for key in ("/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/pathtracing/spp",
            "/rtx/pathtracing/optixDenoiser/enabled"):
    print(f"SETTING {key} = {settings.get(key)}", flush=True)

world = World()
# A local box as ground: add_default_ground_plane() fetches its USD from the remote asset root (no network here).
world.scene.add(FixedCuboid(prim_path="/World/ground", position=np.array([0, 0, -0.05]),
                            scale=np.array([5.0, 5.0, 0.1]), color=np.array([0.5, 0.5, 0.5])))
create_prim("/World/light", "DomeLight", attributes={"inputs:intensity": 1000.0})
world.scene.add(DynamicCuboid(prim_path="/World/cube", position=np.array([0, 0, 0.5]), size=0.3,
                              color=np.array([1.0, 0.2, 0.2])))
camera = Camera(prim_path="/World/cam", position=np.array([1.5, 0.0, 1.0]), resolution=(640, 480),
                orientation=np.array([0.0, 0.2588, 0.0, -0.9659]))
world.reset()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
print(f"STAGE scene_ready {time.time() - t0:.1f}s", flush=True)
t1 = time.time()
for _ in range(args.frames):
    world.step(render=True)
step_ms = (time.time() - t1) / args.frames * 1000
rgb = camera.get_rgba()[:, :, :3]
depth = camera.get_depth()
Image.fromarray(rgb).save(args.out)
gray = rgb.astype(np.float32).mean(-1)
noise = ndimage.laplace(gray[400:470, 20:200]).std()
valid = np.isfinite(depth)
print(f"RENDER_OK {time.time() - t0:.1f}s step={step_ms:.0f}ms rgb_mean={rgb.mean():.1f} noise={noise:.2f} "
      f"depth_valid={valid.mean():.2f}", flush=True)
t_hold = time.time()
while time.time() - t_hold < args.hold:
    world.step(render=True)
if args.hold:
    print(f"HOLD_OK {args.hold:.0f}s", flush=True)
app.close()
