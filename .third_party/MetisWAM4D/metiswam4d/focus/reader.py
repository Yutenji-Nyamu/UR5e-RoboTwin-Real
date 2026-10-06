"""InteractionReader: coupling field -> saliency -> CSIA -> TAA -> compact tokens.

Runs at selected macro layers on the *current denoising state* of the world
experts' hidden tokens and returns ``K`` interaction tokens per world modality
for the Action expert, together with the auxiliary predictions (frame-level
displacement, roles) that are supervised so the coupling field is meaningful.

Video and Track may live on different token grids (multi-view T layout vs.
head view).  ``GridAlign`` resamples Track-grid saliency into the configured
window of the Video grid.  When Track is dropped for a sample (or absent), the
saliency bias is switched off and anchors fall back to uniform.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from metiswam4d.focus.coupling import (
    CouplingField, InteractionSaliency, Saliency, SpatialKernel, compute_coupling_field,
)
from metiswam4d.focus.csia import CSIA, CSIAOutput
from metiswam4d.focus.taa import TAA, Anchors
from metiswam4d.focus.unfold import DisplacementHead, RoleHead, SubFrameUnfold, frames_to_grid


@dataclass
class FocusConfig:
    d_read: int = 512
    num_queries: int = 4            # K
    kappa: int = 4                  # physical frames per latent frame
    kernel_size: int = 5
    kernel_sigma: float = 1.0
    spatial_kernel: str = "gaussian"    # gaussian | learned | adaptive  (neighbourhood of the coupling field)
    temporal_window: str = "gaussian"   # gaussian | adaptive            (TAA aggregation window)
    anchor_refine: bool = False         # learned (bounded) anchor offsets / widths; False = pure transition quantiles
    anchor_width: float = 0.15          # fixed Gaussian window width in normalised time when anchor_refine is False
    w_min: float = 0.02
    w_max: float = 0.5
    progress_dim: int = 3               # (p_video, p_body, p_video - p_body) fed to the condition encoder; 0 = off
    gamma_init: float = 0.3             # saliency log-bias strength at init (learnable, softplus)
    detach_world_hidden: bool = False   # stop reader gradients from reaching Video / Track parameters
    text_dim: int = 4096
    proprio_dim: int = 80
    read_layers: Sequence[int] | str = "track_upper_half"
    feedback_to_track: bool = False  # residual read-back of the compact tokens into Track (stage 1: trains CSIA/TAA)
    feedback_heads: int = 8
    # Video-grid window (row0, row1, col0, col1) that corresponds to the Track grid; None = whole grid.
    video_window: tuple[int, int, int, int] | None = None
    eps: float = 1e-6


class GridAlign(nn.Module):
    """Resample Track-grid maps ``[B, N, h_t, w_t]`` into the Video grid ``[B, N, h_v, w_v]``."""

    def __init__(self, window: tuple[int, int, int, int] | None):
        super().__init__()
        self.window = window

    def forward(self, maps: Tensor, video_grid: tuple[int, int]) -> Tensor:
        b, n, h, w = maps.shape
        hv, wv = video_grid
        if self.window is None:
            if (h, w) == (hv, wv):
                return maps
            return F.interpolate(maps.reshape(b * n, 1, h, w), size=(hv, wv), mode="bilinear",
                                 align_corners=False).reshape(b, n, hv, wv)
        r0, r1, c0, c1 = self.window
        if not (0 <= r0 < r1 <= hv and 0 <= c0 < c1 <= wv):
            raise ValueError(f"video window {self.window} outside grid {video_grid}")
        patch = F.interpolate(maps.reshape(b * n, 1, h, w), size=(r1 - r0, c1 - c0), mode="bilinear",
                              align_corners=False).reshape(b, n, r1 - r0, c1 - c0)
        out = torch.zeros(b, n, hv, wv, dtype=maps.dtype, device=maps.device)
        out[:, :, r0:r1, c0:c1] = patch
        return out


class ConditionEncoder(nn.Module):
    """Task text (masked mean), proprio and pooled current observation -> ``[B, d]``."""

    def __init__(self, d: int, text_dim: int, proprio_dim: int, d_video: int | None, progress_dim: int = 0):
        super().__init__()
        self.text = nn.Linear(text_dim, d)
        self.proprio = nn.Linear(proprio_dim, d)
        self.proprio_missing = nn.Parameter(torch.zeros(d))
        self.observation = nn.Linear(d_video, d) if d_video else None
        self.progress = nn.Linear(progress_dim, d) if progress_dim else None  # zero-init: no effect until learned
        if self.progress is not None:
            nn.init.zeros_(self.progress.weight)
            nn.init.zeros_(self.progress.bias)
        self.norm = nn.LayerNorm(d)

    def forward(self, context: Tensor, context_mask: Tensor | None, proprio: Tensor | None,
                proprio_mask: Tensor | None, observation_tokens: Tensor | None,
                progress: Tensor | None = None) -> Tensor:
        b = context.shape[0]
        if context_mask is None:
            pooled = context.mean(dim=1)
        else:
            m = context_mask.to(context.dtype)[..., None]
            pooled = (context * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        cond = self.text(pooled.to(self.text.weight.dtype))
        if proprio is None:
            cond = cond + self.proprio_missing
        else:
            p = self.proprio(proprio.reshape(b, -1).to(cond.dtype))
            if proprio_mask is not None:
                keep = proprio_mask.reshape(b, -1).any(dim=-1).to(cond.dtype)[:, None]
                p = p * keep + self.proprio_missing * (1 - keep)
            cond = cond + p
        if self.observation is not None and observation_tokens is not None:
            cond = cond + self.observation(observation_tokens.mean(dim=1).to(cond.dtype))
        if self.progress is not None and progress is not None:
            cond = cond + self.progress(progress.to(cond.dtype))
        return self.norm(cond)


class TrackFeedback(nn.Module):
    """Residual cross-attention from Track future tokens to the compact tokens (stage-1 read-back).

    Ungated, with a small-standard-deviation output projection: a zero gate starves every projection of
    gradient while its own gradient is sign-noise (the rt2 runs never opened it).  With ``std = 0.02 /
    sqrt(read_layers)`` the initial perturbation of the Track stream is a fraction of a percent of its norm
    and the CSIA / TAA parameters receive Track flow-matching gradient from the first step.
    """

    def __init__(self, d_track: int, d_read: int, heads: int, num_layers: int = 1):
        super().__init__()
        self.proj = nn.Linear(d_read, d_track)
        self.norm_q = nn.LayerNorm(d_track)
        self.norm_kv = nn.LayerNorm(d_track)
        self.attn = nn.MultiheadAttention(d_track, heads, batch_first=True)
        nn.init.normal_(self.attn.out_proj.weight, std=0.02 / max(1, num_layers) ** 0.5)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, track_tokens: Tensor, compact: Tensor) -> Tensor:
        kv = self.norm_kv(self.proj(compact))
        out, _ = self.attn(self.norm_q(track_tokens), kv, kv, need_weights=False)
        return out


@dataclass
class FocusOutput:
    tokens: dict[str, Tensor]                 # {"video": [B,K,d], "track": [B,K,d]}
    cond: Tensor                              # [B, d]
    displacement: Tensor | None = None        # [B, N, h, w, 3]  predicted frame-level displacement
    role_logits: Tensor | None = None         # [B, N, h, w, 3]
    field: CouplingField | None = None
    saliency: Saliency | None = None
    anchors: Anchors | None = None
    csia: dict[str, CSIAOutput] = dataclasses.field(default_factory=dict)
    beta: dict[str, Tensor] = dataclasses.field(default_factory=dict)  # [B, K, N] aggregation weights
    track_feedback: Tensor | None = None      # [B, J*h*w, d_track] residual for Track future tokens


class InteractionReader(nn.Module):
    def __init__(self, config: FocusConfig, d_track: int | None, d_video: int | None):
        super().__init__()
        if d_track is None and d_video is None:
            raise ValueError("the reader needs at least one world modality")
        self.config = config
        c = config
        d = c.d_read
        self.kernel = SpatialKernel(c.kernel_size, c.kernel_sigma, c.spatial_kernel, d_feat=d)
        self.cond = ConditionEncoder(d, c.text_dim, c.proprio_dim, d_video, c.progress_dim)
        self.unfold = nn.ModuleDict()
        self.csia = nn.ModuleDict()
        if d_track is not None:
            self.unfold["track"] = SubFrameUnfold(d_track, d, c.kappa)
            self.csia["track"] = CSIA(d, c.num_queries, d, c.gamma_init, c.eps)
            self.displacement_head = DisplacementHead(d)
            self.role_head = RoleHead(d)
            self.saliency = InteractionSaliency(d, self.kernel)
        if d_video is not None:
            self.unfold["video"] = SubFrameUnfold(d_video, d, c.kappa)
            self.csia["video"] = CSIA(d, c.num_queries, d, c.gamma_init, c.eps)
            self.align = GridAlign(c.video_window)
        self.taa = TAA(d, c.num_queries, d, c.w_min, c.w_max, c.eps, window=c.temporal_window,
                       refine=c.anchor_refine, fixed_width=c.anchor_width)
        self.pool = nn.Linear(d, d)
        self.feedback = (TrackFeedback(d_track, d, c.feedback_heads)
                         if c.feedback_to_track and d_track is not None else None)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.unfold.keys())

    def forward(
        self,
        *,
        track_hidden: Tensor | None, track_grid: tuple[int, int, int] | None,
        video_hidden: Tensor | None, video_grid: tuple[int, int, int] | None,
        context: Tensor, context_mask: Tensor | None,
        proprio: Tensor | None, proprio_mask: Tensor | None,
        drop_video: Tensor | None = None, drop_track: Tensor | None = None,
        camera_motion: Tensor | None = None, progress: Tensor | None = None,
    ) -> FocusOutput:
        """``camera_motion``: optional ``[B, N_f, 2]`` head ego-motion magnitude per transition (see model);
        ``progress``: optional ``[B, 3]`` (p_video, p_body, p_video - p_body) predicted task progress."""
        have_track = track_hidden is not None and "track" in self.unfold
        have_video = video_hidden is not None and "video" in self.unfold
        if not (have_track or have_video):
            raise ValueError("reader called without world hidden states")
        if self.config.detach_world_hidden:
            track_hidden = track_hidden.detach() if track_hidden is not None else None
            video_hidden = video_hidden.detach() if video_hidden is not None else None
        ref = track_hidden if have_track else video_hidden
        b = ref.shape[0]
        device, dtype = ref.device, ref.dtype
        keep_track = (~drop_track if drop_track is not None
                      else torch.ones(b, dtype=torch.bool, device=device)) if have_track \
            else torch.zeros(b, dtype=torch.bool, device=device)
        keep_video = (~drop_video if drop_video is not None
                      else torch.ones(b, dtype=torch.bool, device=device)) if have_video \
            else torch.zeros(b, dtype=torch.bool, device=device)

        observation = None
        if have_video:
            observation = video_hidden[:, :video_grid[1] * video_grid[2]]
        cond = self.cond(context, context_mask, proprio, proprio_mask, observation, progress)

        out = FocusOutput(tokens={}, cond=cond)
        frame_strength = None
        pooled = torch.zeros(b, self.config.d_read, dtype=dtype, device=device)
        track_features = None
        track_alpha_object = None

        if have_track:
            frames = frames_to_grid(track_hidden, track_grid, skip_frames=1)  # [B, J, h, w, d_T]
            track_features = self.unfold["track"](frames)                    # [B, N, h, w, d]
            out.displacement = self.displacement_head(track_features)
            out.role_logits = self.role_head(track_features)
            role = out.role_logits.float().softmax(dim=-1)
            kernel_feats = track_features.float() if self.config.spatial_kernel == "adaptive" else None
            out.field = compute_coupling_field(out.displacement.float(), role, self.kernel, self.config.eps,
                                               features=kernel_feats)
            cam = camera_motion.float() if camera_motion is not None else None
            out.saliency = self.saliency(out.field, cond.float(), features=kernel_feats, camera_motion=cam)
            s_obj, s_body = out.saliency.object_side.to(dtype), out.saliency.body_side.to(dtype)
            csia_t = self.csia["track"](
                track_features, cond, s_obj, s_body, out.field.coupling_vector.to(dtype),
                use_saliency=keep_track)
            out.csia["track"] = csia_t
            track_alpha_object = csia_t.alpha_object
            frame_strength = out.saliency.frame * keep_track.to(out.saliency.frame.dtype)[:, None]
            frame_strength = frame_strength + (~keep_track).to(frame_strength.dtype)[:, None]
            pooled = self.pool(track_features.mean(dim=(1, 2, 3)))
            pooled = pooled * keep_track.to(dtype)[:, None]

        if have_video:
            frames = frames_to_grid(video_hidden, video_grid, skip_frames=1)
            video_features = self.unfold["video"](frames)                    # [B, N, h_v, w_v, d]
            n = video_features.shape[1]
            if have_track:
                if out.saliency.object_side.shape[1] != n:
                    raise ValueError("Video and Track unfold to different frame counts")
                vg = (video_grid[1], video_grid[2])
                s_obj = self.align(out.saliency.object_side.to(dtype), vg)
                s_body = self.align(out.saliency.body_side.to(dtype), vg)
                # Coupling vectors on the Video grid: resample the three components.
                r = out.field.coupling_vector.to(dtype)
                r_v = torch.stack([self.align(r[..., c], vg) for c in range(3)], dim=-1)
                csia_v = self.csia["video"](video_features, cond, s_obj, s_body, r_v,
                                            object_alpha=None, use_saliency=keep_track)
            else:
                csia_v = self.csia["video"](video_features, cond, None, None, None)
            out.csia["video"] = csia_v

        if frame_strength is None:
            n = next(iter(out.csia.values())).tokens.shape[2]
            frame_strength = torch.ones(b, n, dtype=torch.float32, device=device)
        out.anchors = self.taa.anchors(frame_strength, cond, pooled)

        for name, csia_out in out.csia.items():
            queries = self.csia[name].query_vectors(cond)
            z, beta = self.taa.aggregate(csia_out.tokens, queries, out.anchors)
            keep = keep_track if name == "track" else keep_video
            out.tokens[name] = z * keep.to(z.dtype)[:, None, None]
            out.beta[name] = beta

        if self.feedback is not None and have_track:
            compact = torch.cat([out.tokens[name] for name in out.tokens], dim=1)
            future = track_hidden[:, track_grid[1] * track_grid[2]:]
            out.track_feedback = self.feedback(future, compact)
        return out

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"focus": [p for p in self.parameters() if p.requires_grad]}


__all__ = ["ConditionEncoder", "FocusConfig", "FocusOutput", "GridAlign", "InteractionReader", "TrackFeedback"]
