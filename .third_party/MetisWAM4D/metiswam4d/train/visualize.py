"""Periodic visualisation during training: roll out the asynchronous sampler on a fixed batch.

Every rank rolls out its own fixed batch (FSDP forward needs all ranks), rank 0 decodes with the
Wan VAE and writes, per sample, a frame strip (GT video / sampled video / GT Track4D / sampled
Track4D) and an action plot (GT vs sample over the active unified dimensions) to
``<output_dir>/vis/step_XXXXXXX/`` and to TensorBoard.  Latent-space errors, pixel PSNR, masked
action MSE and the sampled camera ego-motion magnitude are returned as ``vis/*`` scalars.

The decoder is any object with ``decode_latents(latents) -> uint8 [B, T, H, W, 3]`` (the RT2 online
encoder); without one, only the scalars and the action plot are produced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from metiswam4d.data.camera import camera_motion_magnitude
from metiswam4d.data.contract import SampleSpec, move_batch
from metiswam4d.sampler import AsyncSampler, SampleConditions, SampleResult
from metiswam4d.schedule import AsyncSchedule


def _psnr(a: Tensor, b: Tensor) -> float:
    mse = (a.float() - b.float()).square().mean().clamp(min=1e-10)
    return float(10.0 * torch.log10(255.0 ** 2 / mse))


def frame_strip(rows: list[Tensor], scale: int = 2, gap: int = 4) -> np.ndarray:
    """``rows``: uint8 ``[T, H, W, 3]`` tensors (any sizes) -> one uint8 canvas, one row per tensor."""
    import torch.nn.functional as F
    tiles = []
    for frames in rows:
        f = frames.permute(0, 3, 1, 2).float()
        f = F.interpolate(f, scale_factor=1.0 / scale, mode="area") if scale > 1 else f
        t, _, h, w = f.shape
        canvas = torch.full((3, h, t * (w + gap) - gap), 255.0)
        for i in range(t):
            canvas[:, :, i * (w + gap): i * (w + gap) + w] = f[i]
        tiles.append(canvas)
    width = max(t.shape[2] for t in tiles)
    height = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    out = torch.full((3, height, width), 255.0)
    y = 0
    for t in tiles:
        out[:, y: y + t.shape[1], : t.shape[2]] = t
        y += t.shape[1] + gap
    return out.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def action_plot(gt: Tensor, pred: Tensor, mask: Tensor, title: str) -> np.ndarray | None:
    """Small multiples of the active action dimensions (GT solid, sample dashed) -> uint8 RGB array."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None
    dims = torch.nonzero(mask.any(dim=0)).flatten().tolist()
    if not dims:
        return None
    cols = min(5, len(dims))
    rows = (len(dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 1.8 * rows), squeeze=False)
    for k, d in enumerate(dims):
        ax = axes[k // cols][k % cols]
        ax.plot(gt[:, d].float().cpu().numpy(), color="k", lw=1.2)
        ax.plot(pred[:, d].float().cpu().numpy(), color="tab:red", lw=1.0, ls="--")
        ax.set_title(f"dim {d}", fontsize=8)
        ax.set_ylim(-1.15, 1.15)
        ax.tick_params(labelsize=6)
    for k in range(len(dims), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def focus_panel(frames: Tensor, focus, i: int, title: str) -> np.ndarray | None:
    """Where the reader looks, for one sample at one read layer.

    ``frames``: uint8 ``[N, H, W, 3]`` head-view frames of the N transitions (background of the maps).
    Rows: RGB | object-side saliency | body-side saliency | predicted coupling c | predicted roles |
    CSIA object-side attention of each of the K queries.  Right panel: TAA aggregation weights beta[k, n]
    with anchor centres (x), and the frame transition strength s_n.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    if focus.saliency is None or focus.field is None or "track" not in focus.csia:
        return None
    n = frames.shape[0]
    s_obj = focus.saliency.object_side[i].float().cpu().numpy()          # [N, h, w]
    s_body = focus.saliency.body_side[i].float().cpu().numpy()
    coupling = focus.field.coupling[i].float().cpu().numpy()             # [N, h, w]
    role = focus.role_logits[i].float().argmax(-1).cpu().numpy()          # [N, h, w]
    alpha = focus.csia["track"].alpha_object[i].float().cpu().numpy()     # [K, N, P]
    k_all, _, p = alpha.shape
    h, w = s_obj.shape[1:]
    alpha = alpha.reshape(k_all, -1, h, w)
    beta = focus.beta["track"][i].float().cpu().numpy()                   # [K, N]
    strength = focus.saliency.frame[i].float().cpu().numpy()              # [N]
    centre = focus.anchors.centre[i].float().cpu().numpy() * n - 0.5      # frame-slot units
    width = focus.anchors.width[i].float().cpu().numpy() * n
    shown = np.linspace(0, k_all - 1, min(k_all, 8)).round().astype(int)  # CSIA rows: up to 8 queries, evenly spaced
    k = len(shown)
    rows = 5 + k
    fig = plt.figure(figsize=(1.6 * n + 4.2, 1.35 * rows))
    gs = fig.add_gridspec(rows, n + 3, width_ratios=[1] * n + [0.15, 1.6, 1.4])
    ext = (0, frames.shape[2], frames.shape[1], 0)

    def heat(ax, m, vmax=None, cmap="magma"):
        ax.imshow(frames[t], extent=ext)
        ax.imshow(m, extent=ext, cmap=cmap, alpha=0.65, interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])

    labels = ["RGB", "saliency obj", "saliency body", "coupling c", "roles (pred)"] + [f"CSIA q{j}" for j in shown]
    first_col: list = []
    for t in range(n):
        col = []
        ax = fig.add_subplot(gs[0, t]); ax.imshow(frames[t]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"n={t}", fontsize=8); col.append(ax)
        ax = fig.add_subplot(gs[1, t]); heat(ax, s_obj[t], vmax=max(s_obj.max(), 1e-6)); col.append(ax)
        ax = fig.add_subplot(gs[2, t]); heat(ax, s_body[t], vmax=max(s_body.max(), 1e-6)); col.append(ax)
        ax = fig.add_subplot(gs[3, t]); heat(ax, coupling[t], vmax=1.0, cmap="viridis"); col.append(ax)
        ax = fig.add_subplot(gs[4, t]); ax.imshow(frames[t], extent=ext)
        ax.imshow(np.ma.masked_equal(role[t], 0), extent=ext, cmap="coolwarm", alpha=0.6, vmin=1, vmax=2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); col.append(ax)
        for r, j in enumerate(shown):
            ax = fig.add_subplot(gs[5 + r, t]); heat(ax, alpha[j, t], vmax=max(alpha[j].max(), 1e-6), cmap="hot"); col.append(ax)
        if t == 0:
            first_col = col
    for ax, lab in zip(first_col, labels):
        ax.set_ylabel(lab, fontsize=7)
    ax = fig.add_subplot(gs[0:3, n + 1])
    ax.imshow(beta, cmap="Blues", aspect="auto", vmin=0, vmax=max(beta.max(), 1e-6))
    for j in range(k_all):
        ax.plot(centre[j], j, "rx", ms=3); ax.plot([centre[j] - width[j], centre[j] + width[j]], [j, j], "r-", lw=0.6)
    ax.set_title("TAA beta[k, n] (x anchor, - width)", fontsize=8); ax.set_xlabel("n", fontsize=7); ax.set_ylabel("k", fontsize=7)
    ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[3:5, n + 1])
    ax.bar(np.arange(n), strength, color="gray"); ax.set_title("frame strength s_n", fontsize=8); ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


class Visualizer:
    def __init__(self, *, raw_batch: dict | None, spec: SampleSpec, schedule: AsyncSchedule, rounds: int,
                 encoder: Any | None, device: torch.device, dtype: torch.dtype, out_dir: Path,
                 is_main: bool, tb=None, seed: int = 0, dataset=None, collate=None, fixed_index: int | None = None,
                 samples: int = 2, head_full_frame: bool = False,
                 extra_sources: list[tuple[str, Any, int]] | None = None):
        """Either a fixed ``raw_batch`` (mixture data) or ``dataset``/``collate`` (RT2): then every run
        uses the fixed window ``fixed_index`` plus ``samples - 1`` fresh random windows (seeded by the step).
        ``head_full_frame``: the whole video frame is the head view (single-view human data) instead of the
        top block of the Alpha L layout.  ``extra_sources``: further ``(name, dataset, fixed_index)`` sources sampled
        the same way, each rolled out as its own batch (sources differ in modality presence: Kling hand-track windows,
        IG-10K robot windows with actions); their scalars / files carry the source name."""
        self.raw, self.spec, self.rounds = raw_batch, spec, rounds
        self.sampler = AsyncSampler(schedule, rounds)
        self.encoder, self.device, self.dtype = encoder, device, dtype
        self.out_dir, self.is_main, self.tb, self.seed = Path(out_dir), is_main, tb, seed
        self.dataset, self.collate, self.fixed_index, self.samples = dataset, collate, fixed_index, samples
        self.head_full_frame = head_full_frame
        self.extra_sources = list(extra_sources or [])

    def _source_batch(self, dataset, fixed: int, step: int, salt: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 7919 + step + salt)
        idx = [fixed] + torch.randint(len(dataset), (max(0, self.samples - 1),), generator=g).tolist()
        return self.collate([dataset[j] for j in idx])

    def _batches(self, step: int) -> list[tuple[str, dict]]:
        if self.dataset is not None:
            batches = [("", self._source_batch(self.dataset, self.fixed_index, step, 0))]
            for k, (name, dataset, fixed) in enumerate(self.extra_sources, start=1):
                batches.append((name, self._source_batch(dataset, fixed, step, 1000003 * k)))
        else:
            batches = [("", self.raw)]
        out = []
        for name, batch in batches:
            if self.encoder is not None and "video_clean" not in batch:
                batch = self.encoder(move_batch(batch, self.device))
            out.append((name, move_batch(batch, self.device, self.dtype)))
        return out

    @torch.no_grad()
    def _focus_forward(self, model, batch: dict):
        """One denoising forward at moderate noise (video 0.5 / track 0.3 / action 0.2) -> FocusOutput per read layer."""
        from metiswam4d.model import ModelInput
        from metiswam4d.schedule import add_noise
        g = torch.Generator().manual_seed(11)
        b = batch["text_context"].shape[0]

        def noisy(x: Tensor, s: float, keep_first: bool = False) -> Tensor:
            eps = torch.randn(x.shape, generator=g).to(x)
            out, _, _ = add_noise(x, torch.full((b,), s, device=x.device), eps)
            if keep_first:
                out[:, :, :1] = x[:, :, :1]
            return out

        def sig(s: float) -> Tensor:
            return torch.full((b,), s, device=self.device)

        has_track, has_action = batch.get("track_clean") is not None, batch.get("action") is not None
        inputs = ModelInput(
            context=batch["text_context"], context_mask=batch.get("text_mask"),
            video=noisy(batch["video_clean"], 0.5, True) if batch.get("video_clean") is not None else None,
            video_sigma=sig(0.5) if batch.get("video_clean") is not None else None,
            track=noisy(batch["track_clean"], 0.3, True) if has_track else None, track_sigma=sig(0.3) if has_track else None,
            track_conditions=tuple(batch[k] for k in ("rgb_condition", "depth_condition", "mask_condition")) if has_track else None,
            condition_present=batch.get("condition_present") if has_track else None,
            camera=noisy(batch["camera_delta"], 0.3) if batch.get("camera_delta") is not None else None,
            action=noisy(batch["action"], 0.2) if has_action else None, action_sigma=sig(0.2) if has_action else None,
            proprio=batch.get("proprio"), proprio_mask=batch.get("proprio_mask"), embodiment=batch.get("embodiment"),
        )
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = model(inputs)
        return out.focus, {"progress_video": out.progress_video, "progress_body": out.progress_body,
                           "progress_action": out.progress_action}

    @torch.no_grad()
    def run(self, model, step: int) -> dict[str, float]:
        """Roll out every source batch; scalars of the primary source are ``vis/<metric>``, of the extra sources
        ``vis/<source>_<metric>``."""
        was_training = model.training
        model.eval()
        try:
            scalars: dict[str, float] = {}
            for name, batch in self._batches(step):
                part = self._run_batch(model, batch, step, name)
                scalars.update({(k if not name else k.replace("vis/", f"vis/{name}_", 1)): v for k, v in part.items()})
            return scalars
        finally:
            model.train(was_training)

    def _run_batch(self, model, batch: dict, step: int, source: str) -> dict[str, float]:
        f = self.spec.track_frames
        cond = SampleConditions(
            context=batch["text_context"], context_mask=batch.get("text_mask"),
            video_first_frame=batch["video_clean"][:, :, :1] if batch.get("video_clean") is not None else None,
            video_future_frames=f - 1,
            track_anchor=batch["track_clean"][:, :, :1] if batch.get("track_clean") is not None else None,
            track_future_frames=f - 1,
            track_conditions=tuple(batch[k] for k in ("rgb_condition", "depth_condition", "mask_condition"))
            if batch.get("track_clean") is not None else None,
            condition_present=batch.get("condition_present"),
            camera_frames=self.spec.frame_slots if batch.get("camera_delta") is not None else 0,
            action_horizon=self.spec.action_horizon if batch.get("action") is not None else 0,
            action_dim=self.spec.action_dim,
            proprio=batch.get("proprio"), proprio_mask=batch.get("proprio_mask"), embodiment=batch.get("embodiment"),
        )
        generator = torch.Generator().manual_seed(self.seed)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            result = self.sampler.sample(model, cond, generator=generator, dtype=self.dtype)
        focus, progress = self._focus_forward(model, batch) if batch.get("track_clean") is not None else ([], {})
        scalars = self._scalars(batch, result)
        for name, logits in progress.items():  # progress heads: |sigmoid(pred) - label|
            label = batch.get("progress_body" if name == "progress_action" else name)
            if logits is not None and label is not None:
                err = (torch.sigmoid(logits.float()).reshape(-1) - label.float().reshape(-1)).abs()
                valid = batch["progress_valid"].reshape(-1).bool() if batch.get("progress_valid") is not None else torch.ones_like(err, dtype=torch.bool)
                if valid.any():
                    scalars[f"vis/{name}_abs_err"] = float(err[valid].mean())
        if focus:
            last = focus[-1]
            if last.saliency is not None:
                s = last.saliency.frame.float()
                p = s / s.sum(dim=1, keepdim=True).clamp(min=1e-6)
                scalars["vis/frame_strength_entropy"] = float(-(p * (p + 1e-9).log()).sum(dim=1).mean())
            if "track" in last.beta:
                scalars["vis/taa_beta_max"] = float(last.beta["track"].float().max(dim=-1).values.mean())
            if "track" in last.csia:
                scalars["vis/csia_alpha_max"] = float(last.csia["track"].alpha_object.float().max(dim=-1).values.mean())
        if self.is_main:
            self._write(batch, result, step, scalars, focus, source)
        return scalars

    def _scalars(self, batch: dict, result: SampleResult) -> dict[str, float]:
        out: dict[str, float] = {}
        if result.video is not None:
            out["vis/video_latent_mse"] = float((result.video[:, :, 1:].float() - batch["video_clean"][:, :, 1:].float()).square().mean())
        if result.track is not None:
            out["vis/track_latent_mse"] = float((result.track[:, :, 1:].float() - batch["track_clean"][:, :, 1:].float()).square().mean())
        if result.action is not None:
            m = batch["action_mask"].float()
            err = (result.action.float() - batch["action"].float()).square() * m
            out["vis/action_mse"] = float(err.sum() / m.sum().clamp(min=1.0))
        if result.camera is not None:
            mag = camera_motion_magnitude(result.camera)
            out["vis/camera_translation"] = float(mag[..., 0].mean())
            out["vis/camera_angle_rad"] = float(mag[..., 1].mean())
        out["vis/rounds_run"] = float(result.rounds_run)
        return out

    def _write(self, batch: dict, result: SampleResult, step: int, scalars: dict[str, float], focus=(),
               source: str = "") -> None:
        import json
        from PIL import Image
        folder = self.out_dir / "vis" / f"step_{step:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        keys = batch.get("key") or [str(i) for i in range(batch["text_context"].shape[0])]
        pre = f"{source}_" if source else ""  # file / TensorBoard prefix of an extra source
        decoded: dict[str, Tensor] = {}
        if self.encoder is not None:
            pairs = {"video_gt": batch.get("video_clean"), "video": result.video,
                     "track_gt": batch.get("track_clean"), "track": result.track}
            for name, lat in pairs.items():
                if lat is not None:
                    decoded[name] = self.encoder.decode_latents(lat).cpu()
            if "video" in decoded:
                scalars["vis/video_psnr"] = _psnr(decoded["video"][:, 1:], decoded["video_gt"][:, 1:])
            if "track" in decoded:
                scalars["vis/track_psnr"] = _psnr(decoded["track"][:, 1:], decoded["track_gt"][:, 1:])
        for i, key in enumerate(keys):
            tag = str(key).replace("/", "_")
            rows = [decoded[n][i] for n in ("video_gt", "video", "track_gt", "track") if n in decoded]
            if rows:
                strip = frame_strip(rows)
                Image.fromarray(strip).save(folder / f"{pre}{i}_{tag}_frames.png")
                if self.tb is not None:
                    self.tb.add_image(f"vis/{pre}frames_{i}", strip, step, dataformats="HWC")
            if focus and focus[-1].saliency is not None:
                n_slots = focus[-1].saliency.frame.shape[1]
                if "video_gt" in decoded:
                    # Head view = top block of the L layout (2/3 height); transition n <- frame n of the 9.
                    vid = decoded["video_gt"][i]
                    head = vid[:n_slots] if self.head_full_frame else vid[:n_slots, : vid.shape[1] * 2 // 3]
                else:  # no pixel decoder: neutral background
                    head = torch.full((n_slots, 64, 80, 3), 40, dtype=torch.uint8)
                for tag_layer, fo in (("last", focus[-1]), ("first", focus[0])):
                    img = focus_panel(head, fo, i, f"{key}  step {step}  read layer {tag_layer}")
                    if img is not None:
                        Image.fromarray(img).save(folder / f"{pre}{i}_{tag}_focus_{tag_layer}.png")
                        if self.tb is not None and tag_layer == "last":
                            self.tb.add_image(f"vis/{pre}focus_{i}", img, step, dataformats="HWC")
            if result.action is not None:
                img = action_plot(batch["action"][i], result.action[i], batch["action_mask"][i],
                                  f"{key}  step {step}  (black GT, red sample)")
                if img is not None:
                    Image.fromarray(img).save(folder / f"{pre}{i}_{tag}_action.png")
                    if self.tb is not None:
                        self.tb.add_image(f"vis/{pre}action_{i}", img, step, dataformats="HWC")
        path = folder / "scalars.json"  # one file per step: sources are merged (extra sources are name-prefixed)
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged.update({(k if not source else k.replace("vis/", f"vis/{source}_", 1)): v for k, v in scalars.items()})
        path.write_text(json.dumps(merged, indent=1))


def build_visualizer(stage, *, encoder, device, dtype, rank: int, world: int, out_dir: Path,
                     is_main: bool, tb, schedule: AsyncSchedule, log: Callable = print) -> Visualizer | None:
    """Fixed per-rank batch: RT2 raw windows (encoded on the fly) or the first mixture batch."""
    tcfg = stage.training
    if not tcfg.visualize:
        return None
    n = max(1, int(tcfg.visualize_samples))
    common = dict(spec=stage.data.spec, schedule=schedule, rounds=tcfg.visualize_rounds, encoder=encoder,
                  device=device, dtype=dtype, out_dir=out_dir, is_main=is_main, tb=tb, seed=1234 + rank, samples=n)
    if stage.data.rt2 is not None:
        from metiswam4d.data.rt2 import RT2EpisodeDataset, collate_raw
        dataset = RT2EpisodeDataset(stage.data.rt2)
        g = torch.Generator().manual_seed(1234)
        fixed = int(torch.randperm(len(dataset), generator=g)[rank])
        log(f"[vis] per rank: 1 fixed window + {n - 1} random windows per checkpoint, {tcfg.visualize_rounds} sampler rounds")
        return Visualizer(raw_batch=None, dataset=dataset, collate=collate_raw, fixed_index=fixed, **common)
    if stage.data.robodojo is not None:
        # Held-out RoboDojo episodes with deterministic window starts as the fixed panel source.
        from dataclasses import replace
        from metiswam4d.data.robodojo import window_dataset
        from metiswam4d.data.rt2 import collate_raw
        rdj = stage.data.robodojo
        dataset = window_dataset(replace(rdj, split=rdj.val_split or rdj.split, fixed_windows=True))
        g = torch.Generator().manual_seed(1234)
        fixed = int(torch.randperm(len(dataset), generator=g)[rank % len(dataset)])
        log(f"[vis] per rank: 1 fixed {dataset.config.split} window + {n - 1} random per checkpoint, "
            f"{tcfg.visualize_rounds} sampler rounds")
        return Visualizer(raw_batch=None, dataset=dataset, collate=collate_raw, fixed_index=fixed, **common)
    if stage.data.human is not None:
        # Held-out IG-10K episodes (video + object/hand Track4D, all conditions present) as the fixed panel source.
        from metiswam4d.data.human.ig10k import IG10KConfig, IG10KWindowDataset
        from metiswam4d.data.human.loader import collate_human
        human = stage.data.human
        cfg = human.val or next((c.ig10k for c in human.components if c.kind == "ig10k" and not (c.ig10k or IG10KConfig()).with_action), None) \
            or IG10KConfig()
        dataset = IG10KWindowDataset(cfg)
        g = torch.Generator().manual_seed(1234)
        fixed = int(torch.randperm(len(dataset), generator=g)[rank % len(dataset)])
        extra = []
        kling_cfg = next((c.kling for c in human.components if c.kind == "kling" and c.kling.mode == "track"), None)
        if kling_cfg is not None:  # Kling hand-track windows next to the IG-10K ones (the two sources behave differently)
            from metiswam4d.data.human.kling import KlingWindowDataset
            kling = KlingWindowDataset(kling_cfg)
            extra.append(("kling", kling, int(torch.randperm(len(kling), generator=g)[rank % len(kling)])))
        robot_cfg = human.val_robot or next((c.ig10k for c in human.components if c.kind == "ig10k" and c.ig10k is not None
                                             and c.ig10k.with_action), None)
        if robot_cfg is not None:  # IG-10K robot windows: action curves + robot-side Track panels
            robot = IG10KWindowDataset(robot_cfg)
            extra.append(("robot", robot, int(torch.randperm(len(robot), generator=g)[rank % len(robot)])))
        log(f"[vis] per rank: 1 fixed IG-10K window + {n - 1} random" +
            "".join(f", 1 fixed {name} window + {n - 1} random" for name, _, _ in extra) +
            f" per checkpoint, {tcfg.visualize_rounds} rounds")
        return Visualizer(raw_batch=None, dataset=dataset, collate=collate_human, fixed_index=fixed,
                          head_full_frame=True, extra_sources=extra, **common)
    from metiswam4d.train.data import build_loader
    loader, _ = build_loader(stage.data, num_batches=1, seed=1234, rank=rank, world_size=world)
    raw = next(iter(loader))
    log(f"[vis] fixed batch per rank: {n} samples, {tcfg.visualize_rounds} sampler rounds")
    return Visualizer(raw_batch=raw, **common)


__all__ = ["Visualizer", "action_plot", "build_visualizer", "focus_panel", "frame_strip"]
