"""Training step assembly and losses.

``prepare_training_step`` turns one collated sample dict into a
:class:`~metiswam4d.model.ModelInput` (noisy blocks under an asynchronous noise
assignment, modality dropout flags) plus the regression targets.
``compute_losses`` evaluates the masked flow-matching losses of the noisy
modalities (clean modalities are conditions and get no loss), the structured
Track loss (body / object / background normalised separately), the auxiliary
sub-frame displacement and role losses of the reading interface, and the Track
condition reconstruction loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
import torch.nn.functional as F

from metiswam4d.model import ModelInput, ModelOutput
from metiswam4d.schedule import AsyncSchedule, NoiseAssignment, NoiseMixture, add_noise


@dataclass
class LossWeights:
    video: float = 1.0
    track: float = 0.2
    camera: float = 0.1     # head ego-motion tokens (Track clock)
    progress: float = 0.05  # task-progress heads (video / body), each
    action: float = 1.0
    unfold: float = 0.1
    role: float = 0.1
    condition: float = 0.01
    camera_translation: float = 1.0  # weight of the 3 translation dims inside the camera code (0 = rotation only)
    read_action: float = 0.0         # ReadActionHead: clean-action MSE from the compact tokens alone (interface probe)


@dataclass
class TrackRegionWeights:
    """Relative weights of the separately normalised Track regions."""
    body: float = 0.35
    object: float = 0.50
    background: float = 0.15


@dataclass
class DropoutConfig:
    """Dropout for action-generation samples (fractions of such samples).

    ``video_only / track_only / both``: hide the future world from the Action expert.
    ``action_text / action_proprio``: hide the Action expert's private shortcuts (task text, proprio token),
    sampled independently, so the withheld information must be read through the world interface.
    """
    video_only: float = 0.15
    track_only: float = 0.10
    both: float = 0.05
    action_text: float = 0.0
    action_proprio: float = 0.0

    def __post_init__(self) -> None:
        if min(self.video_only, self.track_only, self.both) < 0 or \
                self.video_only + self.track_only + self.both > 1.0:
            raise ValueError("dropout fractions must be non-negative and sum to at most 1")
        if not (0 <= self.action_text <= 1 and 0 <= self.action_proprio <= 1):
            raise ValueError("shortcut dropout fractions must lie in [0, 1]")


@dataclass
class TrainingStep:
    inputs: ModelInput
    assignment: NoiseAssignment
    targets: dict[str, Tensor]
    present: tuple[str, ...]


def _sample_dropout(batch: int, noisy_action: Tensor, config: DropoutConfig,
                    generator: torch.Generator | None, device) -> tuple[Tensor, Tensor]:
    u = torch.rand(batch, generator=generator).to(device)
    drop_video = u < config.video_only
    drop_track = (u >= config.video_only) & (u < config.video_only + config.track_only)
    both = (u >= config.video_only + config.track_only) & \
        (u < config.video_only + config.track_only + config.both)
    drop_video = (drop_video | both) & noisy_action
    drop_track = (drop_track | both) & noisy_action
    return drop_video, drop_track


def prepare_training_step(
    sample: dict[str, Tensor],
    *,
    schedule: AsyncSchedule,
    mixture: NoiseMixture,
    dropout: DropoutConfig | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> TrainingStep:
    """Build noisy inputs and targets from a collated sample (see ``data/contract.py``)."""
    present = tuple(m for m, key in (("video", "video_clean"), ("track", "track_clean"), ("action", "action"))
                    if sample.get(key) is not None)
    if not present:
        raise ValueError("sample has no generative modality")
    context = sample["text_context"]
    device = context.device
    batch = context.shape[0]
    local_schedule = schedule.present(present)
    assignment = mixture.sample(batch, local_schedule, generator=generator, device=device)
    targets: dict[str, Tensor] = {}
    kw: dict = dict(context=context, context_mask=sample.get("text_mask"), embodiment=sample.get("embodiment"))
    cast = (lambda x: x.to(dtype)) if dtype is not None else (lambda x: x)

    if "video" in present:
        clean = cast(sample["video_clean"])
        noisy, _, target = add_noise(clean, assignment.sigma["video"])
        noisy[:, :, :1] = clean[:, :, :1]  # clean current observation
        kw.update(video=noisy, video_sigma=assignment.sigma["video"])
        targets["video"] = target
    if "track" in present:
        clean = cast(sample["track_clean"])
        noisy, _, target = add_noise(clean, assignment.sigma["track"])
        noisy[:, :, :1] = clean[:, :, :1]  # zero-displacement anchor frame
        kw.update(track=noisy, track_sigma=assignment.sigma["track"],
                  track_conditions=tuple(cast(sample[k]) for k in ("rgb_condition", "depth_condition", "mask_condition")),
                  condition_present=sample.get("condition_present"))
        targets["track"] = target
        for key in ("track_valid", "track_role", "track_disp_frames", "track_role_frames"):
            if sample.get(key) is not None:
                targets[key] = sample[key]
        if sample.get("camera_delta") is not None:
            # Camera tokens share the Track clock: same sigma, same flow-matching path.
            clean_cam = cast(sample["camera_delta"])
            noisy_cam, _, cam_target = add_noise(clean_cam, assignment.sigma["track"])
            kw["camera"] = noisy_cam
            targets["camera"] = cam_target
            targets["camera_valid"] = sample["camera_valid"]
    if "action" in present:
        clean = cast(sample["action"])
        noisy, _, target = add_noise(clean, assignment.sigma["action"])
        kw.update(action=noisy, action_sigma=assignment.sigma["action"],
                  proprio=cast(sample["proprio"]) if sample.get("proprio") is not None else None,
                  proprio_mask=sample.get("proprio_mask"))
        targets["action"] = target
        targets["action_clean"] = clean
        if sample.get("action_mask") is not None:
            targets["action_mask"] = sample["action_mask"]
        if dropout is not None:
            noisy_action = ~assignment.is_clean("action")
            kw["drop_video"], kw["drop_track"] = _sample_dropout(batch, noisy_action, dropout, generator, device)
            if dropout.action_text > 0:
                kw["drop_action_text"] = (torch.rand(batch, generator=generator).to(device) < dropout.action_text) & noisy_action
            if dropout.action_proprio > 0:
                kw["drop_action_proprio"] = (torch.rand(batch, generator=generator).to(device) < dropout.action_proprio) & noisy_action
    for key in ("progress_video", "progress_body", "progress_valid"):
        if sample.get(key) is not None:
            targets[key] = sample[key]
    return TrainingStep(ModelInput(**kw), assignment, targets, present)


# ----------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------


def _gated_mean(per_sample: Tensor, gate: Tensor) -> Tensor:
    gate = gate.to(per_sample.dtype)
    return (per_sample * gate).sum() / gate.sum().clamp(min=1.0)


def video_loss(pred: Tensor, target: Tensor, noisy: Tensor) -> Tensor:
    err = (pred.float() - target.float()).square()[:, :, 1:]  # exclude clean first frame
    return _gated_mean(err.mean(dim=(1, 2, 3, 4)), noisy)


def action_loss(pred: Tensor, target: Tensor, noisy: Tensor, mask: Tensor | None) -> Tensor:
    err = (pred.float() - target.float()).square()
    if mask is None:
        per = err.mean(dim=(1, 2))
    else:
        m = mask.to(err.dtype)
        per = (err * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1.0)
    return _gated_mean(per, noisy)


# Alpha's per-arm EEF10 layout (xyz + rot6d + gripper) in the unified 80-D space, left 0-9 / right 34-43.
ACTION_GROUPS = {
    "xyz": (0, 1, 2, 34, 35, 36),
    "rot6d": (3, 4, 5, 6, 7, 8, 37, 38, 39, 40, 41, 42),
    "gripper": (9, 43),
}


def action_group_losses(pred: Tensor, target: Tensor, noisy: Tensor, mask: Tensor | None) -> dict[str, Tensor]:
    """Diagnostic: the action loss restricted to each EEF group (only groups with active slots in ``mask``)."""
    out: dict[str, Tensor] = {}
    base = mask if mask is not None else torch.ones_like(pred, dtype=torch.bool)
    for name, slots in ACTION_GROUPS.items():
        sel = torch.zeros(pred.shape[-1], dtype=torch.bool, device=pred.device)
        sel[list(slots)] = True
        m = base & sel
        if bool(m.any()):
            out[name] = action_loss(pred, target, noisy, m).detach()
    return out


def camera_loss(pred: Tensor, target: Tensor, noisy: Tensor, valid: Tensor | None,
                translation_weight: float = 1.0) -> Tensor:
    """Velocity MSE over the camera tokens ``[B, N, 9]``, masked by ``valid [B, N]`` and the noisy gate.
    ``translation_weight`` scales the first three code dims (0 when the trajectory's metric scale is untrusted)."""
    err = (pred.float() - target.float()).square()  # [B, N, 9]
    if translation_weight != 1.0:
        w = torch.ones(err.shape[-1], device=err.device)
        w[:3] = translation_weight
        err = err * w
    err = err.mean(dim=-1)  # [B, N]
    if valid is None:
        per = err.mean(dim=1)
        has = noisy
    else:
        m = valid.to(err.dtype)
        per = (err * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        has = noisy & (m.sum(dim=1) > 0)
    return _gated_mean(per, has)


def condition_loss(pred: Tensor, target: Tensor, present: Tensor | None) -> Tensor:
    """Condition patch reconstruction MSE ``[B, S, M * D]``; modalities absent from a sample (``present [B, M]``
    False) are excluded."""
    err = (pred.float() - target.float()).square()
    if present is None:
        return err.mean()
    b, s, _ = err.shape
    m = present.shape[1]
    err = err.reshape(b, s, m, -1).mean(dim=(1, 3))  # [B, M]
    w = present.to(err.dtype).to(err.device)
    return (err * w).sum() / w.sum().clamp(min=1.0)


def structured_track_loss(pred: Tensor, target: Tensor, noisy: Tensor, *,
                          role: Tensor | None, valid: Tensor | None,
                          weights: TrackRegionWeights) -> tuple[Tensor, dict[str, Tensor]]:
    """Region-normalised velocity MSE on future Track frames.

    ``role`` ``[B, F, H, W]`` in {0 background, 1 body, 2 object} on the latent
    grid; if absent, ``valid`` ``[B, 1, F, H, W]`` splits foreground / background.
    """
    err = (pred.float() - target.float()).square().mean(dim=1)[:, 1:]  # [B, F-1, H, W]
    b = err.shape[0]
    parts: dict[str, Tensor] = {}
    if role is not None:
        role = role[:, 1:].to(err.device)
        regions = {"background": role == 0, "body": role == 1, "object": role == 2}
    elif valid is not None:
        fg = valid[:, 0, 1:].to(err.device)
        regions = {"foreground": fg, "background": ~fg}
    else:
        regions = {"all": torch.ones_like(err, dtype=torch.bool)}
    total = torch.zeros((), dtype=err.dtype, device=err.device)
    for name, region in regions.items():
        m = region.to(err.dtype)
        per = (err * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp(min=1.0)
        has = (m.sum(dim=(1, 2, 3)) > 0) & noisy
        value = _gated_mean(per, has)
        parts[name] = value.detach()
        weight = {"body": weights.body, "object": weights.object, "background": weights.background,
                  "foreground": weights.body + weights.object, "all": 1.0}[name]
        total = total + weight * value
    return total, parts


def focus_aux_losses(output: ModelOutput, targets: dict[str, Tensor]) -> dict[str, Tensor]:
    """Averaged over read layers: L1 sub-frame displacement and role cross-entropy."""
    result: dict[str, Tensor] = {}
    disp_target = targets.get("track_disp_frames")   # [B, N, h, w, 3] normalised units
    role_target = targets.get("track_role_frames")   # [B, N, h, w] int in {0,1,2}
    if not output.focus:
        return result
    unf, role, count = 0.0, 0.0, 0
    for focus in output.focus:
        if focus.displacement is None:
            continue
        count += 1
        if disp_target is not None:
            valid = (role_target > 0) if role_target is not None else torch.ones_like(disp_target[..., 0], dtype=torch.bool)
            l1 = (focus.displacement.float() - disp_target.float()).abs().sum(dim=-1)
            unf = unf + (l1 * valid).sum() / valid.sum().clamp(min=1)
        if role_target is not None and bool((role_target >= 0).any()):   # -1 = unknown cell, not scored
            logits = focus.role_logits.float().reshape(-1, 3)
            role = role + F.cross_entropy(logits, role_target.reshape(-1).long(), ignore_index=-1)
    if count:
        if disp_target is not None:
            result["unfold"] = unf / count
        if role_target is not None:
            result["role"] = role / count
    return result


def compute_losses(output: ModelOutput, step: TrainingStep, *, weights: LossWeights,
                   track_regions: TrackRegionWeights | None = None) -> dict[str, Tensor]:
    """Weighted total plus detached components; clean modalities contribute nothing."""
    track_regions = track_regions or TrackRegionWeights()
    a, t = step.assignment, step.targets
    logs: dict[str, Tensor] = {}
    total = torch.zeros((), device=step.inputs.context.device)
    if output.video_velocity is not None:
        lv = video_loss(output.video_velocity, t["video"], ~a.is_clean("video"))
        total = total + weights.video * lv
        logs["loss/video"] = lv.detach()
    if output.track_velocity is not None:
        lt, parts = structured_track_loss(
            output.track_velocity, t["track"], ~a.is_clean("track"),
            role=t.get("track_role"), valid=t.get("track_valid"), weights=track_regions)
        total = total + weights.track * lt
        logs["loss/track"] = lt.detach()
        logs.update({f"loss/track_{k}": v for k, v in parts.items()})
        if output.condition_reconstruction is not None:
            lc = condition_loss(output.condition_reconstruction, output.condition_target, output.condition_present)
            total = total + weights.condition * lc
            logs["loss/track_condition"] = lc.detach()
        if output.camera_velocity is not None and t.get("camera") is not None:
            lcam = camera_loss(output.camera_velocity, t["camera"], ~a.is_clean("track"), t.get("camera_valid"),
                               translation_weight=weights.camera_translation)
            total = total + weights.camera * lcam
            logs["loss/camera"] = lcam.detach()
    if output.action_velocity is not None:
        noisy_action = ~a.is_clean("action")
        la = action_loss(output.action_velocity, t["action"], noisy_action, t.get("action_mask"))
        total = total + weights.action * la
        logs["loss/action"] = la.detach()
        logs.update({f"loss/action_{k}": v for k, v in
                     action_group_losses(output.action_velocity, t["action"], noisy_action, t.get("action_mask")).items()})
        # Shortcut-dropout diagnostics: action loss on samples that had to read through the world interface.
        flags = [f for f in (step.inputs.drop_action_text, step.inputs.drop_action_proprio) if f is not None]
        if flags:
            dropped = torch.stack(flags).any(dim=0).to(noisy_action.device)
            for name, sel in (("shortcut_dropped", dropped), ("shortcut_kept", ~dropped)):
                if (sel & noisy_action).any():
                    logs[f"loss/action_{name}"] = action_loss(
                        output.action_velocity, t["action"], noisy_action & sel, t.get("action_mask")).detach()
        if output.read_action is not None and weights.read_action > 0:
            # Interface probe: the compact tokens alone must predict the clean action.  Scored on noisy-action
            # samples that keep at least one world modality (dropped tokens are zero and carry nothing).
            readable = noisy_action.clone()
            drops = [d for d in (step.inputs.drop_video, step.inputs.drop_track) if d is not None]
            if len(drops) == 2:
                readable &= ~(drops[0].to(readable.device) & drops[1].to(readable.device))
            lr = action_loss(output.read_action, t["action_clean"], readable, t.get("action_mask"))
            total = total + weights.read_action * lr
            logs["loss/read_action"] = lr.detach()
    if output.read_ratio is not None:
        logs["focus/read_ratio"] = output.read_ratio.detach()
    for name, value in focus_aux_losses(output, t).items():
        # A batch without any foreground (e.g. EBench, no Track targets) yields plain 0.0 terms.
        value = torch.as_tensor(value, dtype=total.dtype, device=total.device)
        total = total + getattr(weights, name) * value
        logs[f"loss/{name}"] = value.detach()
    for name, logits, target_key in (("progress_video", output.progress_video, "progress_video"),
                                     ("progress_body", output.progress_body, "progress_body"),
                                     ("progress_action", output.progress_action, "progress_body")):
        if logits is not None and t.get(target_key) is not None:
            per = (torch.sigmoid(logits.float()).reshape(-1) - t[target_key].float().reshape(-1)).square()
            valid = t.get("progress_valid")  # samples whose window has a defined task phase (episodes, not clips)
            if valid is None:
                lp = per.mean()
            elif not bool(valid.any()):
                continue
            else:
                lp = _gated_mean(per, valid.reshape(-1).to(per.device))
            total = total + weights.progress * lp
            logs[f"loss/{name}"] = lp.detach()
    for name in step.present:
        logs[f"sigma/{name}"] = a.sigma[name].float().mean().detach()
        logs[f"clean_fraction/{name}"] = a.is_clean(name).float().mean().detach()
    if step.inputs.drop_video is not None:
        logs["dropout/video"] = step.inputs.drop_video.float().mean()
        logs["dropout/track"] = step.inputs.drop_track.float().mean()
    for name in ("drop_action_text", "drop_action_proprio"):
        flag = getattr(step.inputs, name)
        if flag is not None:
            logs[f"dropout/{name.removeprefix('drop_')}"] = flag.float().mean()
    logs["loss/total"] = total.detach()
    return {"total": total, **logs}


__all__ = [
    "ACTION_GROUPS", "DropoutConfig", "LossWeights", "TrackRegionWeights", "TrainingStep", "action_group_losses",
    "action_loss", "camera_loss",
    "compute_losses", "condition_loss", "focus_aux_losses", "prepare_training_step", "structured_track_loss", "video_loss",
]
