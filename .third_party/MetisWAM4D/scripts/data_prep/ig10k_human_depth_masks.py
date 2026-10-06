#!/usr/bin/env python
"""IG-10K human demos: per-frame DA3 depth + SAM 3.1 hand/arm & object masks.

Runs on the 288p ego stream produced by `ig10k_human_resize.py` (512x288 @ 30 fps), so the
extracted depth/masks are already at the training resolution and index 1:1 with the frames
the model will see.

MUST be run with /usr/bin/python3.10 -- the vendored pycocotools `_mask` extension is
cpython-310 and DA3 needs `moviepy.editor`, which only exists in the moviepy 1.x installed
there.  See scripts/data_prep/launch_ig10k_depth_masks.sh.

Models (all local, machine is air-gapped; same paths JanusTrack4d used for the imperfect set):
  DA3     depth_anything_3.api.DepthAnything3 <- .../DepthAnything3_260406/src
          weights .../files/depth_study_models/da3nested-v11
  SAM 3.1 sam3.model_builder.build_sam3_predictor(version='sam3.1')
          weights .../files/depth_study_models/sam3.1/sam3.1_multiplex.pt

Unit of work = one LeRobot episode (a timestamp range inside a shared mp4).  Tracking is
reset per episode; a session is never propagated across an episode boundary.

Output, mirroring the source layout:
  {out_root}/{subset}/{task_dir}/depth/ep_{i:05d}.mkv   12-bit depth, FFV1 (verified lossless,
                                            all-intra so per-frame seeking still works)
  {out_root}/{subset}/{task_dir}/depth_index.json       per episode: frames, scale, offset,
                                            quant_bits -> metres = value * scale + offset
  {out_root}/{subset}/{task_dir}/masks.h5   ep_{i:05d}: uint8 packbits (T,2,288,64), gzip
                                            channel 0 = arm+hand, 1 = manipulated objects
                                            unpack: np.unpackbits(bits,axis=-1,count=512,
                                                                  bitorder='little')
Masks use the same packbits/'little' convention as preprocess/robotwin_imperfect.py so the
existing unpack_mask() helper works unchanged.

Storage was measured on real DA3 output: masks 0.8 kB/frame; depth 12-bit FFV1 46.6 kB/frame
vs 124 kB/frame for uint16+gzip9+shuffle in h5, i.e. ~21 GB rather than ~57 GB over the set.
12 bits spans each episode's own depth range, so the quantisation step stays far below DA3's
own error (<0.1 mm on a 0.4 m range, ~1 mm on a 4 m range).

Object prompts come from IG-10K's own per-task annotation
`meta/mask_labels/{dir}/observation.images.{ego}_mask/global.json`; hand/arm prompts are fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

FILES = Path("/m2v_intern_v3/danglingwei/ytech_milm_intern_danglingwei/files")
DA3_SRC = "/m2v_intern_v3/danglingwei/m2v_intern2_danglingwei/codes/DepthAnything3_260406/src"
DA3_WEIGHTS = FILES / "depth_study_models/da3nested-v11"
SAM_CKPT = FILES / "depth_study_models/sam3.1/sam3.1_multiplex.pt"
SAM_PATHS = [str(FILES / "sam3_agibot"), str(FILES / "sam31_dependencies")]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ig10k_profiles as PROF  # noqa: E402

# Set by apply_profile(); defaults are the finished human set (src == out: single training root
# holding the 288p videos, the copied LeRobot tables and the annotations).
PROFILE = PROF.profile("human")
SRC_ROOT = str(PROFILE["out_root"])
OUT_ROOT = SRC_ROOT
SUBSETS = PROFILE["subsets"]

# "human arm" alone already covers both arms *and* the hands/fingers: measured 6819 px/frame vs
# 6844 for ("human hand","human arm") with visually identical overlays, at 1/3 the cost.
# SAM cost is linear in prompt count, so the second prompt bought nothing.
# The robot profile swaps this for ("robot arm",).
HAND_PROMPTS = PROFILE["arm_prompts"]
TARGET_H = 288  # matches ig10k_human_resize.py


def apply_profile(name: str) -> dict:
    """Switch the module to another IG-10K profile (subsets, roots, arm prompt)."""
    global PROFILE, SRC_ROOT, OUT_ROOT, SUBSETS, HAND_PROMPTS, DET_SCORE_THRESH
    PROFILE = PROF.profile(name)
    SRC_ROOT = OUT_ROOT = str(PROFILE["out_root"])
    SUBSETS = PROFILE["subsets"]
    HAND_PROMPTS = PROFILE["arm_prompts"]
    if "DET_SCORE_THRESH" not in os.environ:
        DET_SCORE_THRESH = float(PROFILE["det_thresh"])
    return PROFILE


def ego_width(ep: dict) -> int:
    """288p width of the ego view, derived from the mask stream's native aspect ratio."""
    return round(int(ep["mask_w"]) * TARGET_H / int(ep["mask_h"]) / 2) * 2
DA3_PROCESS_RES = 504
DA3_RES_METHOD = "upper_bound_resize"
DA3_STRIDE = 4  # frames between views inside one DA3 window (wider baseline, as in JanusTrack4d)
# 16 views is ~20% faster than 8, but DA3 activations dominate peak memory and t3's GPUs are
# shared with another tenant holding ~45 GB, so 8 keeps us inside the remaining headroom.
# DA3 is only ~11% of total cost, so this is ~2% overall.
DA3_MAX_VIEWS = int(os.environ.get("DA3_MAX_VIEWS", "8"))
DEPTH_BITS = 12


# --------------------------------------------------------------------------- shared helpers
def pack_mask(mask):
    return np.packbits(mask, axis=-1, bitorder="little")


def unpack_mask(bits, width):
    return np.unpackbits(bits, axis=-1, count=width, bitorder="little").astype(bool)


def log(msg, fh=None):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def ego_key(info: dict) -> str:
    ks = [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
    ego = [k for k in ks if "zed" in k and not k.endswith(("_depth", "_mask"))]
    if len(ego) != 1:
        raise RuntimeError(f"expected exactly one ego stream, got {ego}")
    return ego[0]


def clean_label(s) -> str:
    """Strip WordPiece leftovers from IG-10K's own labels.

    Some mask_labels entries were written straight out of a tokenizer, e.g. '##er cotton swabs'
    or '##er small beaker'. Fed to SAM as-is they match nothing, so the whole propagation pass
    is wasted. '##er' is a continuation piece whose head word is lost, so drop the piece
    entirely ('##er cotton swabs' -> 'cotton swabs') rather than keeping a bare 'er'.
    """
    if not s:
        return ""
    words = [w for w in str(s).split() if w and not w.startswith("##")]
    return " ".join(words).strip()


def object_prompts(task_dir: Path, name: str, ego: str, out_dir: Path | None = None) -> list[str]:
    """SAM text prompts for the manipulated objects.

    `imitator_human_v1` ships per-task noun phrases in meta/mask_labels; the 16
    `imitator_human_v1_levels` dirs ship none (only a Chinese task string), so those fall back
    to the Qwen-grounded prompts cached by the `prompts` stage.  Returns [] only when neither
    source exists -- callers must treat that as an error, never as "no objects".
    """
    p = task_dir / "meta" / "mask_labels" / name / f"observation.images.{ego}_mask" / "global.json"
    if p.exists():
        labels = json.loads(p.read_text())
        # keys are stringified mask ids; keep annotation order, drop duplicates
        got = list(dict.fromkeys(clean_label(v) for v in labels.values() if clean_label(v)))
        if got:
            return got
    if out_dir is not None:
        cache = out_dir / "sam_prompts.json"
        if cache.exists():
            return list(json.loads(cache.read_text())["sam_prompts"])
    return []


def shipped_mask_key(info: dict, ego: str) -> str:
    """IG-10K's own Grounded-SAM-2 mask stream for the ego view, if it shipped one.

    The 53 `imitator_human_v1` dirs have `{ego}_mask` (h264 gbrp, pixel value == instance id,
    names in meta/mask_labels/.../global.json) -- strictly better than re-running SAM, which in a
    side-by-side missed one of the three annotated objects entirely. The 16 `_levels` dirs have
    no mask stream, so those still need SAM.
    """
    key = f"observation.images.{ego}_mask"
    return f"{ego}_mask" if key in info["features"] else ""


def read_shipped_mask(task_dir: Path, mask_key: str, ep: dict, hw, n: int) -> np.ndarray:
    """Decode this episode's slice of the shipped mask stream as an instance-id map (T,H,W)."""
    import cv2

    h, w = hw
    src = (task_dir / "videos" / f"observation.images.{mask_key}"
           / f"chunk-{ep['mask_chunk']:03d}" / f"file-{ep['mask_file']:03d}.mp4")
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
         "-ss", f"{ep['mask_from_s']:.6f}", "-to", f"{ep['mask_to_s']:.6f}", "-i", str(src),
         "-vf", "fps=30", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"mask decode rc={r.returncode}: {r.stderr.decode()[:300]}")
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    sh = int(ep["mask_h"])
    sw = int(ep["mask_w"])
    a = a.reshape(-1, sh, sw, 3)[..., 0]  # gbrp with value == instance id, channels identical
    if len(a) != n:
        raise RuntimeError(f"mask frames {len(a)} != rgb frames {n}")
    # NEAREST: these are label ids, interpolation would invent instances that do not exist
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_NEAREST) for f in a])


def mano_columns(task_dir: Path) -> list[str]:
    """MANO hand columns in the episode data parquet, if IG-10K shipped any.

    `imitator_human_v1` carries 40 of them (left/right x 4 views); the 16
    `imitator_human_v1_levels` dirs carry none, same gap as meta/mask_labels.  Where MANO exists
    there is no point paying SAM for an arm mask -- the hand pose is already annotated.
    """
    import pyarrow.parquet as pq

    files = sorted((task_dir / "data").glob("chunk-*/*.parquet"))
    if not files:
        return []
    cols = pq.read_schema(files[0]).names
    return [c for c in cols if ".mano_" in c or "pred_keypoints_3d" in c]


def task_instruction(task_dir: Path) -> str:
    import pyarrow.parquet as pq

    t = pq.read_table(task_dir / "meta" / "tasks.parquet")
    for col in ("task", "__index_level_0__"):
        if col in t.column_names:
            vals = [v for v in t[col].to_pylist() if v and str(v) not in ("robot", "human")]
            if vals:
                return str(vals[0])
    # robot dirs carry no task text in tasks.parquet (just "robot"); use the dataset's own
    # English description (intent ; layout ; per-arm steps) from _meta_extra/task_desc
    desc = json.loads(Path(PROFILE["desc"]).read_text()).get(task_dir.name)
    if desc:
        return str(desc[0] if isinstance(desc, list) else desc)
    return ""


def human_counterpart_labels(task_dir_name: str) -> list[str]:
    """Object names IG-10K annotated on the matching human demo, for robot L0/L1 dirs.

    L0/L1 keep the human demo's objects (only the layout changes at L1), so those verified
    names are good extra candidates; L2/L3 swap objects and get nothing from here.  The
    screening step keeps only prompts that actually fire.
    """
    import re

    m = re.match(r"robot_(H\d+)_L([01])", task_dir_name)
    if not m:
        return []
    h = f"human_{m.group(1)}"
    p = PROF.RAW_ROOT / "imitator_human_v1" / h / "meta" / "mask_labels" / h
    hits = sorted(p.glob("observation.images.*_mask/global.json"))
    if not hits:
        return []
    return [clean_label(v) for v in json.loads(hits[0].read_text()).values() if clean_label(v)]


def episodes_of(task_dir: Path, ego: str):
    """Yield per-episode dicts: index, video path, frame range (derived from LeRobot timestamps)."""
    import pyarrow.parquet as pq

    key = f"videos/observation.images.{ego}"
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    mkey_name = shipped_mask_key(info, ego)
    mkey = f"videos/observation.images.{mkey_name}" if mkey_name else ""
    mshape = (info["features"][f"observation.images.{mkey_name}"]["shape"] if mkey_name else None)
    rows = []
    for f in sorted((task_dir / "meta" / "episodes").glob("chunk-*/*.parquet")):
        t = pq.read_table(f)
        cols = {c: t[c].to_pylist() for c in t.column_names}
        for i in range(len(cols["episode_index"])):
            r = {
                "episode": int(cols["episode_index"][i]),
                "length": int(cols["length"][i]),
                "video": task_dir / "videos" / f"observation.images.{ego}"
                         / f"chunk-{int(cols[f'{key}/chunk_index'][i]):03d}"
                         / f"file-{int(cols[f'{key}/file_index'][i]):03d}.mp4",
                "from_s": float(cols[f"{key}/from_timestamp"][i]),
                "to_s": float(cols[f"{key}/to_timestamp"][i]),
            }
            if mkey and f"{mkey}/from_timestamp" in cols:
                # the mask stream has its own timestamps; never reuse the RGB ones
                r.update(mask_chunk=int(cols[f"{mkey}/chunk_index"][i]),
                         mask_file=int(cols[f"{mkey}/file_index"][i]),
                         mask_from_s=float(cols[f"{mkey}/from_timestamp"][i]),
                         mask_to_s=float(cols[f"{mkey}/to_timestamp"][i]),
                         mask_h=mshape[1], mask_w=mshape[2])
            rows.append(r)
    rows.sort(key=lambda r: r["episode"])
    return rows


def cut_episode(ep: dict, dst: Path, fps: int = 30) -> int:
    """Copy one episode's frame range out of the shared mp4. Returns frame count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # re-encode (not -c copy) so the cut lands exactly on the requested frames
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{ep['from_s']:.6f}", "-to", f"{ep['to_s']:.6f}", "-i", str(ep["video"]),
         "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg cut rc={r.returncode}: {r.stderr.strip()[:300]}")
    return read_frames(dst).shape[0]


def read_frames(path: Path) -> np.ndarray:
    """Decode a whole (short) clip to uint8 RGB (T,H,W,3)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return np.stack(out)


# --------------------------------------------------------------------------- DA3
_DA3 = {}


def da3_model():
    if "m" not in _DA3:
        if DA3_SRC not in sys.path:
            sys.path.insert(0, DA3_SRC)
        import torch
        from depth_anything_3.api import DepthAnything3
        _DA3["m"] = DepthAnything3.from_pretrained(str(DA3_WEIGHTS)).cuda().eval()
        _DA3["torch"] = torch
    return _DA3["m"], _DA3["torch"]


def da3_depth(rgb: np.ndarray) -> np.ndarray:
    """Monocular/any-view DA3 depth in metres for every frame of one episode, float32 (T,H,W).

    No GT intrinsics/extrinsics exist for this data, so we let DA3 infer pose itself and feed
    stride-DA3_STRIDE windows: each pass covers the whole episode with a wide baseline between
    views, and the DA3_STRIDE passes together cover every frame exactly once.
    """
    import cv2

    model, torch = da3_model()
    n, h, w = rgb.shape[:3]
    depth = np.zeros((n, h, w), np.float32)
    filled = np.zeros(n, bool)
    for phase in range(DA3_STRIDE):
        idx_all = np.arange(phase, n, DA3_STRIDE)
        for s in range(0, len(idx_all), DA3_MAX_VIEWS):
            idx = idx_all[s:s + DA3_MAX_VIEWS]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model.inference(list(rgb[idx]), process_res=DA3_PROCESS_RES,
                                       process_res_method=DA3_RES_METHOD)
            z = np.asarray(pred.depth, dtype=np.float32)
            for j, t in enumerate(idx):
                depth[t] = cv2.resize(z[j], (w, h), interpolation=cv2.INTER_LINEAR)
                filled[t] = True
            del pred
    torch.cuda.empty_cache()
    if not filled.all():
        raise RuntimeError(f"DA3 left {int((~filled).sum())} frames empty")
    if not np.isfinite(depth).all() or not (depth > 0).all():
        raise RuntimeError("invalid DA3 depth (non-finite or non-positive)")
    return depth


def quantize_depth(depth: np.ndarray, bits: int = DEPTH_BITS):
    """float32 metres -> uint16 holding `bits` levels + (scale, offset)."""
    lo = float(depth.min())
    hi = float(depth.max())
    levels = 2 ** bits - 1
    scale = max((hi - lo) / levels, 1e-12)
    q = np.clip(np.rint((depth - lo) / scale), 0, levels).astype(np.uint16)
    return q, scale, lo


def write_depth_ffv1(q: np.ndarray, dst: Path):
    """Store quantised depth as all-intra FFV1 (lossless, ~2.7x smaller than h5+gzip)."""
    n, h, w = q.shape
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mkv")  # keep .mkv last so ffmpeg picks the muxer
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:",
         "-pix_fmt", "gray16le", "-c:v", "ffv1", "-level", "3", str(tmp)],
        input=np.ascontiguousarray(q).tobytes(), capture_output=True, timeout=1800,
    )
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffv1 encode rc={p.returncode}: {p.stderr.decode()[:300]}")
    os.replace(tmp, dst)


def read_depth_ffv1(path: Path, hw, scale: float, offset: float) -> np.ndarray:
    """Decode an FFV1 depth file back to metres (float32). Kept here as the reference reader."""
    h, w = hw
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                        "-pix_fmt", "gray16le", "-"], capture_output=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"ffv1 decode rc={p.returncode}: {p.stderr.decode()[:300]}")
    q = np.frombuffer(p.stdout, dtype=np.uint16).reshape(-1, h, w)
    return q.astype(np.float32) * scale + offset


# --------------------------------------------------------------------------- SAM 3.1
_SAM = {}


def sam_predictor():
    if "p" not in _SAM:
        for p in SAM_PATHS:
            if p not in sys.path:
                sys.path.insert(0, p)
        from sam3.model_builder import build_sam3_predictor
        _SAM["p"] = build_sam3_predictor(checkpoint_path=str(SAM_CKPT), version="sam3.1",
                                         use_fa3=False, compile=False,
                                         async_loading_frames=False,
                                         max_num_objects=SAM_MAX_OBJECTS)
    return _SAM["p"]


# 0.3 was used for the first pass; the 2026-09-24 repair of the 8 weak `_levels` dirs ran at 0.15
# (small objects such as a 250 px carrot or a red juice box at 288p only fire below 0.3)
DET_SCORE_THRESH = float(os.environ.get("DET_SCORE_THRESH", "0.3"))
SCREEN_FRAMES = 16
# The batched-grounding batch is the peak allocation in the detection phase (it OOM'd inside the
# segmentation head at 16). 4 keeps two workers per 80 GB card alive; utilisation stays pinned
# because the other worker covers the gaps.
GROUNDING_BATCH = int(os.environ.get("GROUNDING_BATCH", "4"))
# we only ever union masks, so tracking dozens of instances buys nothing but memory
SAM_MAX_OBJECTS = int(os.environ.get("SAM_MAX_OBJECTS", "8"))
# Where IG-10K ships MANO we reuse it instead of segmenting the arm; set FORCE_ARM=1 to override.
FORCE_ARM = os.environ.get("FORCE_ARM", "") == "1"
# Prefer IG-10K's own mask stream over SAM where it exists; FORCE_SAM_MASKS=1 overrides.
FORCE_SAM_MASKS = os.environ.get("FORCE_SAM_MASKS", "") == "1"


def sam_detect(video: Path, n: int, hw, prompts: list[str], frames=None, extra=None) -> dict:
    """Per-frame SAM 3.1 detection, no tracker. Returns {prompt: bool (len(frames),H,W)}.

    Much cheaper than propagation because there is no memory attention and the backbone
    features are reused across prompts via the session feature_cache.  Measured against the
    tracked path: fine for the arm (IoU 0.976, every frame hit) but it loses the small objects
    in ~45% of frames, so it is used only for the arm channel and for screening.
    """
    import torch

    h, w = hw
    idx = list(range(n)) if frames is None else list(frames)
    # extra = (prompts, frames) evaluated in the same session so they reuse the backbone cache
    jobs = [(p, idx) for p in prompts]
    if extra is not None:
        jobs += [(p, list(extra[1])) for p in extra[0]]
    predictor = sam_predictor()
    model = predictor.model
    model.score_threshold_detection = DET_SCORE_THRESH
    model.suppress_det_close_to_boundary = False
    out = {}
    sid = predictor.handle_request(dict(type="start_session", resource_path=str(video),
                                       offload_video_to_cpu=True))["session_id"]
    try:
        for prompt, idx in jobs:
            predictor.handle_request(dict(type="reset_session", session_id=sid))
            state = predictor._get_session(sid)["state"]
            state["input_batch"].find_text_batch[0] = prompt
            with torch.inference_mode():  # text_ids were allocated under inference mode
                for inp in state["input_batch"].find_inputs:
                    inp.text_ids[...] = model.TEXT_ID_FOR_TEXT
            cur = np.zeros((len(idx), h, w), bool)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for j, t in enumerate(idx):
                    det, keep = model.run_backbone_and_detection(
                        frame_idx=t, num_frames=n, input_batch=state["input_batch"],
                        geometric_prompt=state["constants"]["empty_geometric_prompt"],
                        feature_cache=state["feature_cache"], reverse=False,
                        use_batched_grounding=True,
                        batched_grounding_batch_size=GROUNDING_BATCH)
                    m = det["mask"][keep]
                    if len(m):
                        u = m.amax(dim=0).float()[None, None]
                        cur[j] = (torch.nn.functional.interpolate(
                            u, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0
                        ).cpu().numpy()
            out[prompt] = cur
    finally:
        predictor.handle_request(dict(type="close_session", session_id=sid))
    return out


def _first_hits(det: dict, prompts: list[str], frames: list[int]) -> dict:
    """{prompt: first sampled frame index where detection fired}, dropping prompts that never did."""
    out = {}
    for p in prompts:
        m = det.get(p)
        if m is None:
            continue
        hit = np.nonzero(m.reshape(len(frames), -1).any(1))[0]
        if len(hit):
            out[p] = int(frames[hit[0]])
    return out


def screen_only(video: Path, n: int, hw, objs: list[str]) -> dict:
    """Object screen without the arm pass -> {prompt: first frame it was seen in}."""
    if not objs:
        return {}
    frames = np.linspace(0, n - 1, min(SCREEN_FRAMES, n)).round().astype(int).tolist()
    det = sam_detect(video, n, hw, objs, frames=frames)
    return _first_hits(det, objs, frames)


def detect_arm_and_screen(video: Path, n: int, hw, objs: list[str]):
    """One detection session for the arm *and* the object screen, so the backbone runs once.

    The session-level feature_cache is what makes extra prompts cheap, so the arm mask and the
    "does this object appear at all" test must share a single session -- two sessions would pay
    the backbone twice.  Returns (arm mask over all frames, object prompts that actually fire).
    """
    frames = np.linspace(0, n - 1, min(SCREEN_FRAMES, n)).round().astype(int).tolist()
    # ask for every frame for the arm, and only the screen frames for the objects
    det = sam_detect(video, n, hw, list(HAND_PROMPTS), frames=None, extra=(objs, frames))
    arm = np.zeros((n,) + tuple(hw), bool)
    for p in HAND_PROMPTS:
        arm |= det[p]
    return arm, _first_hits(det, objs, frames)


def sam_masks(video: Path, n: int, hw, prompts) -> np.ndarray:
    """Union of the tracked binary masks for `prompts` over the clip -> bool (T,H,W).

    `prompts` is {text: start_frame} (from the screen) or a list (start at frame 0).  The
    prompt is placed on the first frame the screen actually saw the object in and propagated
    in BOTH directions from there.  Prompting at frame 0 with forward-only propagation was
    measured identical -- but only when the object is visible at frame 0; for objects the
    person brings into view later it silently produced 46% empty episodes on `_levels`.
    """
    h, w = hw
    predictor = sam_predictor()
    acc = np.zeros((n, h, w), bool)
    if not prompts:
        return acc
    starts = prompts if isinstance(prompts, dict) else {p: 0 for p in prompts}
    sid = predictor.handle_request(dict(type="start_session", resource_path=str(video),
                                        offload_video_to_cpu=True))["session_id"]
    def one_pass(prompt: str, start: int) -> np.ndarray:
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        predictor.handle_request(dict(type="add_prompt", session_id=sid,
                                      frame_index=start, text=prompt))
        cur = np.zeros((n, h, w), bool)
        seen = np.zeros(n, bool)
        direction = "forward" if start == 0 else "both"
        for resp in predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=sid,
                     propagation_direction=direction, start_frame_index=start)):
            t = resp["frame_index"]
            if t >= n:
                continue
            m = np.asarray(resp["outputs"]["out_binary_masks"], bool)
            if m.size:
                cur[t] |= m.reshape(-1, h, w).any(0)
            seen[t] = True
        if not seen.all():
            raise RuntimeError(f"SAM skipped {int((~seen).sum())} frames")
        return cur

    try:
        for prompt, start in starts.items():
            start = int(min(max(start, 0), n - 1))
            # Mid-clip prompting is outside what this SAM build was exercised on (KeyError on
            # the frame index, or "No points" when the tracker's own threshold sees nothing at
            # that frame). Fall back to the known-good frame-0 forward pass, and never let one
            # prompt take the whole task dir down with it.
            attempts = [(start, "both")] if start else []
            attempts.append((0, "forward"))
            for k, (s, _) in enumerate(attempts):
                try:
                    acc |= one_pass(prompt, s)
                    break
                except torch_oom():
                    raise
                except Exception as e:  # noqa: BLE001
                    if k == len(attempts) - 1:
                        print(f"WARN prompt '{prompt}' skipped after {len(attempts)} attempts: "
                              f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                    else:
                        print(f"WARN prompt '{prompt}' at frame {s} failed "
                              f"({type(e).__name__}), retrying from frame 0", flush=True)
    finally:
        predictor.handle_request(dict(type="close_session", session_id=sid))
    return acc


# --------------------------------------------------------------------------- Qwen prompt grounding
QWEN_PATH = "/m2v_intern_v3/danglingwei/m2v_intern_public_models/Qwen/Qwen3.5-9B"
_QWEN = {}


def qwen_model():
    if "m" not in _QWEN:
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        _QWEN["p"] = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True)
        _QWEN["m"] = Qwen3_5ForConditionalGeneration.from_pretrained(
            QWEN_PATH, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True).eval()
    return _QWEN["p"], _QWEN["m"]


QWEN_PROMPT_HUMAN = (
    "These frames are a first-person recording of a human performing the task below. "
    "Identify the physical objects the person actually manipulates (holds, picks up, pushes, "
    "cuts, pours or hands over). Exclude the human body, hands, table, floor and background, "
    "and exclude stationary receptacles unless they are actually moved. ")
QWEN_PROMPT_ROBOT = (
    "These frames are a fixed-camera recording of a dual-arm robot performing the task below. "
    "Identify the physical objects the robot actually manipulates (holds, picks up, pushes, "
    "cuts, pours or hands over). Exclude the robot arms and grippers, table, floor and background, "
    "and exclude stationary receptacles unless they are actually moved. ")
QWEN_PROMPT_TAIL = (
    'Reply with JSON only: {"sam_prompts": ["short ENGLISH visual noun phrase"]}. '
    "For EACH manipulated object give 2-3 alternative phrasings (with and without colour, "
    "a generic category word, and the literal translation of the task's own noun), because "
    "the downstream segmenter is picky about wording and a later step keeps whichever phrasing "
    "actually matches. Trust the task text over your colour guess from the frames: e.g. "
    "蜜桃果汁 is 'peach juice', not orange juice. Each phrase must be a concrete noun phrase with "
    "no verbs, e.g. 'yellow plate', 'plate', 'peach juice box', 'juice box'. The task may be "
    "written in Chinese; always answer in English.\nTask: ")


def ground_prompts(src_root: Path, out_root: Path, sub: str, d: str, scratch: Path, logfh):
    """Ask Qwen3.5 for SAM prompts when IG-10K ships no mask_labels (the `_levels` dirs)."""
    import torch
    from PIL import Image

    td = src_root / sub / d
    info = json.loads((td / "meta" / "info.json").read_text())
    ego = ego_key(info)
    out_dir = out_root / sub / d
    out_dir.mkdir(parents=True, exist_ok=True)
    instruction = task_instruction(td)
    eps = episodes_of(td, ego)
    clip = scratch / f"prompt_{sub}_{d}.mp4"
    try:
        cut_episode(eps[0], clip)
        rgb = read_frames(clip)
    finally:
        clip.unlink(missing_ok=True)
    idx = np.linspace(0, len(rgb) - 1, 16).round().astype(int)
    processor, model = qwen_model()
    content = [{"type": "image", "image": Image.fromarray(rgb[t])} for t in idx]
    head = QWEN_PROMPT_ROBOT if PROFILE["name"] == "robot" else QWEN_PROMPT_HUMAN
    content.append({"type": "text", "text": head + QWEN_PROMPT_TAIL + instruction})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=600, do_sample=False)
    answer = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0]
    parsed = json.loads(answer[answer.index("{"):answer.rindex("}") + 1])
    prompts = [s.strip() for s in parsed.get("sam_prompts", []) if s and s.strip()]
    extra = human_counterpart_labels(d) if PROFILE["name"] == "robot" else []
    prompts = list(dict.fromkeys(prompts + extra))
    if not prompts:
        raise RuntimeError(f"{sub}/{d}: Qwen returned no prompts; answer={answer[:300]}")
    (out_dir / "sam_prompts.json").write_text(json.dumps(
        {"sam_prompts": prompts, "instruction": instruction,
         "source": "Qwen3.5-9B grounding" + (" + human-demo mask labels" if extra else ""),
         "human_labels": extra, "frames_used": idx.tolist(), "raw_answer": answer}, indent=1, ensure_ascii=False))
    log(f"PROMPTS {sub}/{d}: {instruction} -> {prompts}", logfh)
    return prompts


def cmd_prompts(a):
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    (out_root / "_logs").mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    logfh = open(out_root / "_logs" / f"prompts.{host}.log", "a")
    scratch = Path(a.scratch) / f"prompts_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    todo = []
    for sub, d in task_dirs(src_root):
        td = src_root / sub / d
        info = json.loads((td / "meta" / "info.json").read_text())
        has_labels = (td / "meta" / "mask_labels").is_dir()
        if has_labels and not a.force:
            continue  # shipped labels; nothing to ground
        if a.force or not object_prompts(td, d, ego_key(info), out_root / sub / d):
            todo.append((sub, d))
    log(f"grounding prompts for {len(todo)} dirs without mask_labels", logfh)
    try:
        for sub, d in todo:
            try:
                ground_prompts(src_root, out_root, sub, d, scratch, logfh)
            except Exception:  # noqa: BLE001
                log(f"ERROR prompts {sub}/{d}:\n{traceback.format_exc()}", logfh)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    log("prompts stage done", logfh)


# --------------------------------------------------------------------------- per-task-dir work
def task_dirs(src_root: Path):
    out = []
    for sub in SUBSETS:
        for d in sorted(os.listdir(src_root / sub)):
            p = src_root / sub / d
            if (p / "meta" / "info.json").is_file() and PROF.allowed(PROFILE, d):
                out.append((sub, d))
    return out


def paths_for(out_root: Path, sub: str, d: str, shard=None):
    """Output paths; with shard=(k, n) the masks/depth index go to per-shard files (see merge-shards)."""
    base = out_root / sub / d
    tag = f".shard{shard[0]}of{shard[1]}" if shard else ""
    return {"depth": base / f"depth_index{tag}.json", "depth_dir": base / "depth",
            "masks": base / f"masks{tag}.h5",
            "lock": out_root / "_locks" / f"{sub}__{d}", "base": base}


def process_task_dir(src_root: Path, out_root: Path, sub: str, d: str, stages, scratch: Path, logfh,
                     shard=None):
    """One task dir; shard=(k, n) processes only episodes k, k+n, ... so a slow dir can be spread
    over several GPUs (the SAM-tracked dirs run at 0.3-0.5 fps and take 8-16 h alone)."""
    import h5py

    td = src_root / sub / d
    info = json.loads((td / "meta" / "info.json").read_text())
    ego = ego_key(info)
    eps = episodes_of(td, ego)
    if shard is not None:
        eps = eps[shard[0]::shard[1]]
    p = paths_for(out_root, sub, d, shard)
    p["base"].mkdir(parents=True, exist_ok=True)
    mano = mano_columns(td) if not FORCE_ARM else []
    want_arm = not mano
    mkey = "" if FORCE_SAM_MASKS else shipped_mask_key(info, ego)
    shipped = {"task_dir": td, "key": mkey} if mkey else None
    if shipped:
        labels = json.loads(
            (td / "meta" / "mask_labels" / d / f"observation.images.{mkey}" / "global.json"
             ).read_text())
        objs = [clean_label(v) for v in labels.values()]
        src_desc = f"shipped {mkey} stream"
    else:
        objs = object_prompts(td, d, ego, p["base"])
        src_desc = "sam3.1"
        if "masks" in stages and not objs:
            raise RuntimeError(
                f"{sub}/{d}: no mask stream and no object prompts (no meta/mask_labels, no "
                f"sam_prompts.json). Run the `prompts` stage first -- writing an empty object "
                f"channel would look like 'this task manipulates nothing'.")
    log(f"{sub}/{d}: ego={ego} episodes={len(eps)} objects={objs} via {src_desc}; "
        f"mano_cols={len(mano)} arm={'sam3.1' if want_arm else 'skipped (MANO)'}", logfh)

    mask_file = None
    depth_index = {"model": "da3nested-v11", "unit": "metre", "quant_bits": DEPTH_BITS,
                   "codec": "ffv1 gray16le (lossless)", "ego_key": ego,
                   "encoding": "metre = value * scale + offset",
                   "process_res": DA3_PROCESS_RES, "stride": DA3_STRIDE,
                   "camera_conditioned": False, "episodes": {}}
    if "masks" in stages:
        mask_file = h5py.File(p["masks"].with_suffix(".h5.tmp"), "w")
        if shipped:
            inst = {k: clean_label(v) for k, v in json.loads(
                (td / "meta" / "mask_labels" / d
                 / f"observation.images.{mkey}" / "global.json").read_text()).items()}
            if want_arm:
                # robot subset: shipped ids for the objects, SAM 3.1 detection for the arm,
                # written into the same id map under a reserved id (readers: arm = ids == arm_id)
                if any(int(k) >= PROF.ARM_ID for k in inst):
                    raise RuntimeError(f"{sub}/{d}: shipped instance id collides with ARM_ID")
                inst[str(PROF.ARM_ID)] = " | ".join(HAND_PROMPTS)
            mask_file.attrs.update(
                complete=False, ego_key=ego, format="uint8 instance-id map",
                channels="instance_ids",
                object_source=f"IG-10K shipped {mkey} stream (Grounded-SAM-2), "
                              f"nearest-downscaled to the 288p training resolution",
                arm_source=(f"sam3.1 per-frame text detection, stored as instance id {PROF.ARM_ID}"
                            if want_arm else
                            "not segmented; use the MANO hand annotation in data/*.parquet"),
                mano_columns=json.dumps(mano),
                hand_prompts=json.dumps(list(HAND_PROMPTS) if want_arm else []),
                instance_labels=json.dumps(inst, ensure_ascii=False))
            if want_arm:
                mask_file.attrs["arm_id"] = PROF.ARM_ID
        else:
            mask_file.attrs.update(
                complete=False, model="sam3.1_multiplex", bitorder="little", ego_key=ego,
                format="packbits bool channels",
                # read this: the channel list is per task dir, not fixed
                channels="arm_and_hand,manipulated_objects" if want_arm else "manipulated_objects",
                arm_source="sam3.1 text prompt" if want_arm else "not segmented; use MANO",
                object_source="sam3.1 text prompts (no mask stream shipped for this subset)",
                mano_columns=json.dumps(mano),
                objects_exclude_arm=bool(want_arm),
                hand_prompts=json.dumps(list(HAND_PROMPTS) if want_arm else []),
                object_prompts=json.dumps(objs),
                object_prompt_source="Qwen3.5 grounding (no meta/mask_labels in this subset)")
    t0 = time.time()
    n_frames = 0
    try:
        for ei, ep in enumerate(eps):
            clip = scratch / f"{sub}_{d}_ep{ep['episode']:05d}.mp4"
            name = f"ep_{ep['episode']:05d}"
            try:
                for attempt in range(3):
                    try:
                        n_frames += _do_episode(ep, clip, name, stages, depth_index, mask_file,
                                                objs, p, want_arm, shipped)
                        break
                    except torch_oom() as e:
                        if attempt == 2:
                            raise
                        # t3's GPUs are shared with another tenant, so headroom can vanish
                        # mid-episode; drop partial artefacts, free the cache and try once more
                        log(f"  OOM {sub}/{d} {name}, retrying after free: {e}", logfh)
                        # only discard what THIS run was producing: a masks-only pass must not
                        # delete depth an earlier run already finished (it did, once)
                        if "depth" in stages:
                            (p["depth_dir"] / f"{name}.mkv").unlink(missing_ok=True)
                            depth_index["episodes"].pop(name, None)
                        if mask_file is not None and name in mask_file:
                            del mask_file[name]
                        free_cuda()
            finally:
                clip.unlink(missing_ok=True)
            if (ei + 1) % 5 == 0 or ei + 1 == len(eps):
                el = time.time() - t0
                log(f"  {sub}/{d}: {ei + 1}/{len(eps)} eps, {n_frames} frames, "
                    f"{n_frames / max(el, 1e-9):.2f} fps, "
                    f"eta {el / (ei + 1) * (len(eps) - ei - 1) / 60:.1f} min", logfh)
        if mask_file is not None:
            mask_file.attrs["complete"] = True
    finally:
        if mask_file is not None:
            mask_file.close()
    if "masks" in stages:
        os.replace(p["masks"].with_suffix(".h5.tmp"), p["masks"])
    if "depth" in stages:
        depth_index["complete"] = True
        p["depth"].write_text(json.dumps(depth_index, indent=1))
    mb = (sum(f.stat().st_size for f in p["depth_dir"].glob("*.mkv")) if "depth" in stages else 0)
    mb = (mb + (p["masks"].stat().st_size if "masks" in stages else 0)) / 1e6
    log(f"DONE {sub}/{d}: {len(eps)} eps, {n_frames} frames, {(time.time() - t0) / 60:.1f} min, "
        f"{mb:.0f} MB ({mb * 1e6 / max(n_frames, 1) / 1e3:.1f} kB/frame)", logfh)
    return n_frames


def torch_oom():
    import torch
    return torch.OutOfMemoryError


def free_cuda():
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _do_episode(ep, clip, name, stages, depth_index, mask_file, objs, p, want_arm=True,
                shipped=None) -> int:
    """Cut one episode out of its shared mp4 and write its depth and/or masks."""
    # the shipped-mask path never touches the RGB, so skip the cut+decode entirely (it was the
    # whole cost: pure ffmpeg decode of the mask stream needs no GPU and no re-encode)
    need_rgb = ("depth" in stages) or ("masks" in stages and (not shipped or want_arm))
    if need_rgb:
        cut_episode(ep, clip)
        rgb = read_frames(clip)
        n, h, w = rgb.shape[:3]
    else:
        rgb = None
        n, h, w = ep["length"], TARGET_H, ego_width(ep)
    if "depth" in stages:
        q, scale, offset = quantize_depth(da3_depth(rgb))
        write_depth_ffv1(q, p["depth_dir"] / f"{name}.mkv")
        depth_index["episodes"][name] = {
            "frames": n, "height": h, "width": w, "scale": scale, "offset": offset,
            "from_s": ep["from_s"], "to_s": ep["to_s"]}
    if "masks" in stages and shipped:
        # IG-10K already ships Grounded-SAM-2 instance masks for this view, so store the id map
        # rather than paying SAM to produce a worse approximation of it
        ids = read_shipped_mask(shipped["task_dir"], shipped["key"], ep, (h, w), n)
        arm_px = -1
        if want_arm:
            # no hand annotation for this subset: detect the arm per frame and let it override
            # the object ids where they overlap (the arm occludes what it holds)
            det = sam_detect(clip, n, (h, w), list(HAND_PROMPTS))
            arm = np.zeros((n, h, w), bool)
            for pr in HAND_PROMPTS:
                arm |= det[pr]
            ids = ids.copy()
            ids[arm] = PROF.ARM_ID
            arm_px = int(arm.sum())
        ds = mask_file.create_dataset(name, data=ids, compression="gzip", compression_opts=4,
                                      chunks=(1, h, w))
        objects = (ids > 0) & (ids != PROF.ARM_ID) if want_arm else ids > 0
        ds.attrs.update(frames=n, object_px=int(objects.sum()), arm_px=arm_px,
                        instance_ids_present=sorted(int(v) for v in np.unique(ids) if v),
                        from_s=ep["from_s"], to_s=ep["to_s"])
    elif "masks" in stages:
        # arm via detection (IoU 0.976 vs tracked, every frame hit) and the object screen share
        # one session; objects still need the tracker because detection alone loses them in
        # ~45% of frames
        if want_arm:
            hands, kept = detect_arm_and_screen(clip, n, (h, w), objs)
        else:
            # MANO already describes the hands, so skip the full-video arm pass and only pay
            # for the object screen (SCREEN_FRAMES frames instead of n)
            hands, kept = None, screen_only(clip, n, (h, w), objs)
        if not kept:
            # every label was screened out: better to flag it than to store an object channel of
            # zeros, which downstream reads as "nothing is manipulated here"
            print(f"WARN no object prompt fired for {name} (tried {objs})", flush=True)
        objects = sam_masks(clip, n, (h, w), kept)
        if hands is not None:
            objects &= ~hands
            stack = np.stack((hands, objects), axis=1)
        else:
            stack = objects[:, None]
        bits = pack_mask(stack)
        ds = mask_file.create_dataset(name, data=bits, compression="gzip", compression_opts=4)
        ds.attrs.update(unpacked_shape=[n, stack.shape[1], h, w], frames=n,
                        hand_px=int(hands.sum()) if hands is not None else -1,
                        object_px=int(objects.sum()),
                        prompts_kept=json.dumps(list(kept)),
                        prompt_start_frames=json.dumps(kept),
                        from_s=ep["from_s"], to_s=ep["to_s"])
    return n


# --------------------------------------------------------------------------- commands
def cmd_run(a):
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    (out_root / "_locks").mkdir(parents=True, exist_ok=True)
    (out_root / "_logs").mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    tag = f"{host}.gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.{os.getpid()}"
    logfh = open(out_root / "_logs" / f"{tag}.log", "a")
    stages = tuple(a.stages.split(","))
    scratch = Path(a.scratch) / tag
    scratch.mkdir(parents=True, exist_ok=True)
    log(f"start {tag} stages={stages}", logfh)

    if a.only_dir:
        # one shard of one dir, no lock: the caller coordinates the shards and merges them
        sub, d = a.only_dir.split("/")
        k, n = (int(x) for x in a.ep_shard.split("/")) if a.ep_shard else (0, 1)
        try:
            frames = process_task_dir(src_root, out_root, sub, d, stages, scratch, logfh, shard=(k, n))
            log(f"exit {tag}: shard {k}/{n} of {sub}/{d} frames={frames}", logfh)
        except Exception:  # noqa: BLE001
            log(f"ERROR shard {k}/{n} {sub}/{d}:\n{traceback.format_exc()}", logfh)
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return

    dirs = task_dirs(src_root)
    if a.shuffle:  # spread GPUs over different task dirs instead of all racing for the first
        np.random.default_rng(abs(hash(tag)) % (2 ** 32)).shuffle(dirs)
    n_done = n_frames = 0
    try:
        for sub, d in dirs:
            p = paths_for(out_root, sub, d)
            # only redo the stages that are actually missing, so a dir whose depth already
            # landed does not pay for DA3 again when we come back for masks
            need = tuple(s for s in stages if not p[s].exists())
            if not need:
                continue
            if a.mask_source != "any":
                info = json.loads((src_root / sub / d / "meta" / "info.json").read_text())
                has = bool(shipped_mask_key(info, ego_key(info)))
                if (a.mask_source == "shipped") != has:
                    continue
            try:
                p["lock"].mkdir(parents=True)
            except FileExistsError:
                continue
            (p["lock"] / tag).touch()
            try:
                n_frames += process_task_dir(src_root, out_root, sub, d, need, scratch, logfh)
                n_done += 1
                # release on success: a stage-restricted run (e.g. CPU masks-only) must not leave
                # the dir locked against a later run that still owes it the other stage
                shutil.rmtree(p["lock"], ignore_errors=True)
            except Exception:  # noqa: BLE001
                log(f"ERROR {sub}/{d}:\n{traceback.format_exc()}", logfh)
                shutil.rmtree(p["lock"], ignore_errors=True)
                # only discard the stages this run was producing -- a masks failure must never
                # delete depth that a previous run already finished
                if "masks" in need:
                    p["masks"].with_suffix(".h5.tmp").unlink(missing_ok=True)
                if "depth" in need:
                    shutil.rmtree(p["depth_dir"], ignore_errors=True)
                    p["depth"].unlink(missing_ok=True)
                if a.stop_on_error:
                    raise
            if a.max_dirs and n_done >= a.max_dirs:
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    log(f"exit {tag}: dirs={n_done} frames={n_frames}", logfh)


def cmd_merge_shards(a):
    """Combine masks.shard*.h5 / depth_index.shard*.json of one dir into masks.h5 / depth_index.json.

    Refuses to merge unless every episode of the dir is present exactly once; shard files are
    removed afterwards so the dir looks exactly like one produced by a single worker.
    """
    import h5py

    out_root = Path(a.out_root)
    sub, d = a.only_dir.split("/")
    base = out_root / sub / d
    info = json.loads((Path(a.src_root) / sub / d / "meta" / "info.json").read_text())
    want = {f"ep_{e['episode']:05d}" for e in episodes_of(Path(a.src_root) / sub / d, ego_key(info))}
    mshards = sorted(base.glob("masks.shard*.h5"))
    dshards = sorted(base.glob("depth_index.shard*.json"))
    if not mshards or len(mshards) != len(dshards):
        raise RuntimeError(f"{sub}/{d}: {len(mshards)} mask shards vs {len(dshards)} depth shards")
    depth = None
    tmp = base / "masks.h5.tmp"
    seen = set()
    with h5py.File(tmp, "w") as out:
        for ms, ds in zip(mshards, dshards):
            di = json.loads(ds.read_text())
            if depth is None:
                depth = {k: v for k, v in di.items() if k != "episodes"}
                depth["episodes"] = {}
            depth["episodes"].update(di["episodes"])
            with h5py.File(ms) as f:
                if not out.attrs:
                    out.attrs.update(dict(f.attrs))
                for k in f:
                    if k in seen:
                        raise RuntimeError(f"{sub}/{d}: episode {k} in more than one shard")
                    f.copy(k, out)
                    seen.add(k)
        out.attrs["complete"] = True
        out.attrs["merged_from_shards"] = len(mshards)
    if seen != want or set(depth["episodes"]) != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{sub}/{d}: shards cover {len(seen)} masks / {len(depth['episodes'])} depth "
                           f"episodes, dir has {len(want)}; missing {sorted(want - seen)[:5]}")
    depth["complete"] = True
    os.replace(tmp, base / "masks.h5")
    (base / "depth_index.json").write_text(json.dumps(depth, indent=1))
    for f in mshards + dshards:
        f.unlink()
    print(f"merged {len(mshards)} shards -> {base / 'masks.h5'} ({len(seen)} episodes)")


def cmd_repair_depth(a):
    """Recompute depth for episodes listed in depth_index.json whose .mkv is missing.

    An earlier masks-only run deleted finished depth files on OOM retry (bug since fixed); the
    index still lists them.  Only those episodes are recomputed; scale/offset are refreshed.
    """
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    scratch = Path(a.scratch) / f"repair_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    logfh = open(out_root / "_logs" / f"repair_depth.{socket.gethostname().split('.')[0]}.log", "a")
    n_fixed = 0
    try:
        for idx_path in sorted(out_root.glob("*/*/depth_index.json")):
            sub, d = idx_path.parts[-3], idx_path.parts[-2]
            idx = json.loads(idx_path.read_text())
            have = {p.stem for p in (idx_path.parent / "depth").glob("*.mkv")}
            missing = [k for k in idx["episodes"] if k not in have]
            if not missing:
                continue
            td = src_root / sub / d
            info = json.loads((td / "meta" / "info.json").read_text())
            eps = {f"ep_{e['episode']:05d}": e for e in episodes_of(td, ego_key(info))}
            log(f"REPAIR {sub}/{d}: {len(missing)} missing depth episodes", logfh)
            for name in missing:
                ep = eps[name]
                clip = scratch / f"{name}.mp4"
                try:
                    cut_episode(ep, clip)
                    rgb = read_frames(clip)
                finally:
                    clip.unlink(missing_ok=True)
                n, h, w = rgb.shape[:3]
                q, scale, offset = quantize_depth(da3_depth(rgb))
                write_depth_ffv1(q, idx_path.parent / "depth" / f"{name}.mkv")
                idx["episodes"][name] = {"frames": n, "height": h, "width": w, "scale": scale,
                                         "offset": offset, "from_s": ep["from_s"], "to_s": ep["to_s"]}
                n_fixed += 1
            tmp = idx_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(idx, indent=1))
            os.replace(tmp, idx_path)
            log(f"REPAIRED {sub}/{d}", logfh)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    log(f"repair done: {n_fixed} episodes", logfh)


def cmd_status(a):
    src_root, out_root = Path(a.src_root), Path(a.out_root)
    import h5py

    dirs = task_dirs(src_root)
    done = {"depth": 0, "masks": 0}
    frames = {"depth": 0, "masks": 0}
    size = {"depth": 0, "masks": 0}
    incomplete = []
    for sub, d in dirs:
        p = paths_for(out_root, sub, d)
        if p["depth"].exists():
            done["depth"] += 1
            try:
                idx = json.loads(p["depth"].read_text())
                if not idx.get("complete"):
                    incomplete.append(str(p["depth"]))
                frames["depth"] += sum(e["frames"] for e in idx["episodes"].values())
                size["depth"] += sum(f.stat().st_size for f in p["depth_dir"].glob("*.mkv"))
            except Exception as e:  # noqa: BLE001
                incomplete.append(f"{p['depth']}: {e}")
        if p["masks"].exists():
            done["masks"] += 1
            size["masks"] += p["masks"].stat().st_size
            try:
                with h5py.File(p["masks"]) as f:
                    if not f.attrs.get("complete", False):
                        incomplete.append(str(p["masks"]))
                    frames["masks"] += sum(int(f[k].attrs["frames"]) for k in f)
            except Exception as e:  # noqa: BLE001
                incomplete.append(f"{p['masks']}: {e}")
    locks = list((out_root / "_locks").glob("*")) if (out_root / "_locks").exists() else []
    print(f"task dirs: {len(dirs)}")
    for s in ("depth", "masks"):
        kb = size[s] / max(frames[s], 1) / 1e3
        print(f"  {s:6s} dirs {done[s]}/{len(dirs)}  frames {frames[s]}  "
              f"{size[s] / 1e9:.2f} GB  {kb:.1f} kB/frame")
    print(f"locks: {len(locks)}  incomplete/unreadable: {len(incomplete)}")
    for x in incomplete[:5]:
        print("  !", x)
    for lg in sorted((out_root / "_logs").glob("*.log")):
        tail = lg.read_text().strip().splitlines()
        print(f"--- {lg.name}: {tail[-1] if tail else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROF.PROFILES), default="human")
    ap.add_argument("--src-root", default=None, help="default: the profile's root")
    ap.add_argument("--out-root", default=None, help="default: the profile's root")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--stages", default="depth,masks", help="comma list: depth,masks")
    r.add_argument("--scratch", default="/dev/shm/ig10k_anno")
    r.add_argument("--max-dirs", type=int, default=0)
    r.add_argument("--shuffle", action="store_true")
    r.add_argument("--mask-source", choices=["any", "shipped", "sam"], default="any",
                   help="restrict to dirs whose ego masks come from the shipped stream or SAM")
    r.add_argument("--workers", type=int, default=1,
                   help=">1 forks CPU workers in-process; only valid for --mask-source shipped")
    r.add_argument("--stop-on-error", action="store_true")
    r.add_argument("--only-dir", default="", help="SUB/DIR: process just this dir, no lock")
    r.add_argument("--ep-shard", default="", help="K/N with --only-dir: episodes K, K+N, ... -> shard files")
    r.set_defaults(fn=cmd_run)
    ms = sp.add_parser("merge-shards", help="combine --ep-shard outputs of one dir into masks.h5 + depth_index.json")
    ms.add_argument("--only-dir", required=True)
    ms.set_defaults(fn=cmd_merge_shards)
    q = sp.add_parser("prompts", help="Qwen-ground SAM prompts for dirs lacking mask_labels")
    q.add_argument("--scratch", default="/dev/shm/ig10k_anno")
    q.add_argument("--force", action="store_true", help="re-ground dirs that already have sam_prompts.json")
    q.set_defaults(fn=cmd_prompts)
    rd = sp.add_parser("repair-depth", help="recompute depth for index entries whose mkv is missing")
    rd.add_argument("--scratch", default="/dev/shm/ig10k_anno")
    rd.set_defaults(fn=cmd_repair_depth)
    s = sp.add_parser("status")
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    apply_profile(a.profile)
    a.src_root = a.src_root or SRC_ROOT
    a.out_root = a.out_root or OUT_ROOT
    a.fn(a)


if __name__ == "__main__":
    main()
