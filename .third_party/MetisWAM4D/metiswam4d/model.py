"""MetisWAM4D: Video / Track4D / Action experts coupled by one joint attention.

The driver runs the present experts layer by layer.  At every macro layer each
updating expert produces Q/K/V in the shared attention space; a single masked
softmax couples them under the visibility rules of :mod:`metiswam4d.attention`.
The Track expert (shallower) updates only at its mapped layers and exposes its
last K/V in between.  At the configured read layers the
:class:`~metiswam4d.focus.InteractionReader` turns the current Video / Track
hidden states into ``K`` compact interaction tokens per world modality; in the
``compact`` reading mode these are the only future-world information the Action
expert receives, through a zero-initialised gated cross-attention at the
following layers (optionally annealed from the dense reading, see
``ModelInput.dense_read_bias``).

Experts are optional: stage-1 pretraining runs ``{video, track}`` only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from metiswam4d.attention import (
    ACTION_READ_MODES, KEY_ORDER, QUERY_SEGMENTS, TokenLayout, build_visibility, joint_attention,
)
from metiswam4d.data.camera import camera_motion_magnitude
from metiswam4d.experts.local import ActionExpert, LatentExpert, TrackExpert
from metiswam4d.experts.protocol import ExpertState
from metiswam4d.focus import FocusConfig, FocusOutput, InteractionReader


@dataclass
class MetisWAM4DConfig:
    action_read: str = "compact"
    focus: FocusConfig | None = field(default_factory=FocusConfig)
    text_dim: int = 4096
    proprio_dim: int = 80
    gradient_checkpointing: bool = False
    focus_heads: int = 8          # gated cross-attention from Action to the compact tokens
    focus_head_dim: int = 64
    num_embodiments: int = 8      # embodiment token table (registry indices; see data/embodiments.py)
    # Compact mode safety floor: the dense future world keys stay visible to Action with at least this additive
    # logit bias (0 = the plain joint-attention triple expert; None = closed once the annealing ends).
    dense_read_floor: float | None = None
    # Interface regulariser: predict the clean action chunk from the compact tokens alone (0 = off).
    read_action_horizon: int = 0
    read_action_dim: int = 80

    def __post_init__(self) -> None:
        if self.action_read not in ACTION_READ_MODES:
            raise ValueError(f"action_read must be one of {ACTION_READ_MODES}")


@dataclass
class ModelInput:
    """Noisy blocks, shifted sigmas and conditions.  ``None`` = modality absent."""

    context: Tensor                          # [B, L, text_dim] raw text features
    context_mask: Tensor | None = None       # [B, L] bool
    video: Tensor | None = None              # [B, C, F_v, H, W] noisy latent (clean first frame)
    video_sigma: Tensor | None = None        # [B] shifted sigma in [0, 1]
    track: Tensor | None = None              # [B, C, F_t, H, W]
    track_sigma: Tensor | None = None
    track_conditions: tuple[Tensor, ...] | None = None  # (rgb, depth, mask) latents [B, C, 1, H, W]
    condition_present: Tensor | None = None  # [B, 3] bool  which Track conditions the sample provides (None = all)
    camera: Tensor | None = None             # [B, N_f, 9] noisy head-camera delta poses (Track clock)
    action: Tensor | None = None             # [B, H, action_dim]
    action_sigma: Tensor | None = None
    proprio: Tensor | None = None            # [B, 1, proprio_dim] or [B, proprio_dim]
    proprio_mask: Tensor | None = None       # [B, 1, proprio_dim] bool
    embodiment: Tensor | None = None         # [B] long  registry index -> one context token seen by every expert
    drop_video: Tensor | None = None         # [B] bool  modality dropout (hide future video)
    drop_track: Tensor | None = None
    drop_action_text: Tensor | None = None   # [B] bool  hide the task text from the Action expert only
    drop_action_proprio: Tensor | None = None  # [B] bool  hide the proprio token from the Action expert only
    dense_read_bias: float | None = None     # compact mode: additive logit bias on dense future keys (annealing)

    @property
    def present(self) -> tuple[str, ...]:
        names = []
        if self.video is not None:
            names.append("video")
        if self.track is not None:
            names.append("track")
        if self.action is not None:
            names.append("action")
        return tuple(names)

    @property
    def batch_size(self) -> int:
        return self.context.shape[0]


@dataclass
class ModelOutput:
    video_velocity: Tensor | None = None
    track_velocity: Tensor | None = None
    camera_velocity: Tensor | None = None    # [B, N_f, 9]
    action_velocity: Tensor | None = None
    progress_video: Tensor | None = None     # [B, 1] logits (sigmoid = fraction of the task completed)
    progress_body: Tensor | None = None
    progress_action: Tensor | None = None    # [B, 1] logits from the final Action tokens (+ text, proprio)
    read_ratio: Tensor | None = None         # scalar: mean ||compact residual|| / ||action tokens|| over read layers
    read_action: Tensor | None = None        # [B, H, A] clean action predicted from the compact tokens alone
    condition_reconstruction: Tensor | None = None
    condition_target: Tensor | None = None
    condition_present: Tensor | None = None  # [B, 3] bool: modalities to score in the reconstruction loss
    focus: list[FocusOutput] = field(default_factory=list)  # one per read layer

    def velocity(self, modality: str) -> Tensor | None:
        return getattr(self, f"{modality}_velocity")


class ProgressHead(nn.Module):
    """Mean-pooled clean tokens (+ pooled text, + proprio) -> progress logit ``[B, 1]`` (sigmoid -> [0, 1])."""

    def __init__(self, d_tokens: int, text_dim: int, proprio_dim: int = 0, hidden: int = 256):
        super().__init__()
        self.tokens = nn.Sequential(nn.LayerNorm(d_tokens), nn.Linear(d_tokens, hidden))
        self.text = nn.Linear(text_dim, hidden)
        self.proprio = nn.Linear(proprio_dim, hidden) if proprio_dim else None
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, tokens: Tensor, context: Tensor, context_mask: Tensor, proprio: Tensor | None = None) -> Tensor:
        m = context_mask.to(context.dtype)[..., None]
        text = (context * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        h = self.tokens(tokens.mean(dim=1)) + self.text(text.to(tokens.dtype))
        if self.proprio is not None and proprio is not None:
            h = h + self.proprio(proprio.reshape(proprio.shape[0], -1).to(h.dtype))
        return self.out(h)


class ReadActionHead(nn.Module):
    """Compact interaction tokens (+ reader condition) -> clean action chunk ``[B, H, A]``.

    A probe on the reading interface, trained with the action targets: the compact tokens must by themselves
    carry what the action needs, so the interface cannot go dead while the Action expert leans on its shortcuts.
    Only the reader receives its gradient; the Action expert never sees this prediction.
    """

    def __init__(self, d_read: int, num_modalities: int, horizon: int, action_dim: int, hidden: int = 1024):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.norm = nn.LayerNorm(d_read * (num_modalities + 1))
        self.net = nn.Sequential(nn.Linear(d_read * (num_modalities + 1), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * action_dim))

    def forward(self, tokens: dict[str, Tensor], cond: Tensor) -> Tensor:
        parts = [tokens[name].mean(dim=1) for name in sorted(tokens)] + [cond]
        h = torch.cat([p.to(cond.dtype) for p in parts], dim=-1)
        return self.net(self.norm(h)).view(cond.shape[0], self.horizon, self.action_dim)


class FocusCrossAttention(nn.Module):
    """Residual cross-attention from Action tokens to the compact interaction tokens.

    ``action = action + O(Attn(Q(LN(action)), K(z), V(z)))``.  No gate: a zero gate (or a zero ``O``)
    starves the projections of gradient and its own gradient is sign-noise when the action loss is
    near its floor, so the interface never opens.  ``O`` is initialised with a small standard deviation
    (``0.02 / sqrt(num_layers)``): the initial perturbation of the Action expert is a fraction of a
    percent of its hidden norm, yet every projection receives a full-size gradient from step one.
    Per-sample key masks remove dropped / absent modalities and switch the read off for clean-action
    samples; samples with no readable token receive a zero residual.
    """

    def __init__(self, d_action: int, d_read: int, heads: int, head_dim: int, num_layers: int = 1):
        super().__init__()
        width = heads * head_dim
        self.heads = heads
        self.norm = nn.LayerNorm(d_action)
        self.q = nn.Linear(d_action, width)
        self.k = nn.Linear(d_read, width)
        self.v = nn.Linear(d_read, width)
        self.o = nn.Linear(width, d_action)
        nn.init.normal_(self.o.weight, std=0.02 / max(1, num_layers) ** 0.5)
        nn.init.zeros_(self.o.bias)

    def forward(self, action: Tensor, z: Tensor, key_mask: Tensor) -> Tensor:
        """``action [B, H, d_a]``, ``z [B, Z, d_read]``, ``key_mask [B, Z]`` bool -> residual ``[B, H, d_a]``."""
        b, h_len, _ = action.shape
        any_key = key_mask.any(dim=1)
        safe_mask = key_mask.clone()
        safe_mask[~any_key, 0] = True  # avoid an all-masked softmax; the result is zeroed below
        q = self.q(self.norm(action)).view(b, h_len, self.heads, -1).transpose(1, 2)
        k = self.k(z.to(action.dtype)).view(b, z.shape[1], self.heads, -1).transpose(1, 2)
        v = self.v(z.to(action.dtype)).view(b, z.shape[1], self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask[:, None, None, :])
        out = self.o(out.transpose(1, 2).reshape(b, h_len, -1))
        return out * any_key.to(out.dtype)[:, None, None]


class MetisWAM4D(nn.Module):
    def __init__(
        self,
        *,
        video: LatentExpert | None,
        track: TrackExpert | None,
        action: ActionExpert | None,
        config: MetisWAM4DConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MetisWAM4DConfig()
        experts = {name: e for name, e in (("video", video), ("track", track), ("action", action)) if e is not None}
        if not experts:
            raise ValueError("at least one expert is required")
        heads = {e.num_heads for e in experts.values()}
        dims = {e.head_dim for e in experts.values()}
        if len(heads) != 1 or len(dims) != 1:
            raise ValueError("all experts must share the attention head layout")
        depths = {e.macro_layers for e in experts.values()}
        if len(depths) != 1:
            raise ValueError(f"experts disagree on the macro depth: {depths}")
        self.macro_layers = depths.pop()
        self.num_heads = heads.pop()
        self.video = video
        self.track = track
        self.action = action
        self.proprio_encoder = nn.Linear(self.config.proprio_dim, self.config.text_dim)
        # Embodiment token: appended to the text context (before the proprio token) so Video / Track / Action all
        # know which body acts (human hands vs. a dual-arm robot); the Track expert can then choose the covering
        # rule of its displacement field and the Action expert the action layout.  Small init: a near-zero extra
        # key barely perturbs the pretrained cross-attentions at step 0.  Kept when the Action's shortcuts (text /
        # proprio) are dropped: it is not a shortcut to the action, only the identity of the body.
        self.embodiment_embedding = nn.Embedding(self.config.num_embodiments, self.config.text_dim)
        nn.init.normal_(self.embodiment_embedding.weight, std=0.02)

        focus = self.config.focus
        self.reader: InteractionReader | None = None
        self.read_layers: tuple[int, ...] = ()
        self.focus_attention = nn.ModuleDict()  # layer index -> gated cross-attention (compact mode)
        world = video is not None or track is not None
        if focus is not None and world:
            self.reader = InteractionReader(
                focus, d_track=track.hidden_dim if track else None,
                d_video=video.hidden_dim if video else None)
            self.read_layers = self._resolve_read_layers(focus.read_layers)
            if self.reader.feedback is not None:  # one residual per read layer: scale the init accordingly
                nn.init.normal_(self.reader.feedback.attn.out_proj.weight, std=0.02 / max(1, len(self.read_layers)) ** 0.5)
            if action is not None and self.config.action_read == "compact":
                first = min(self.read_layers)
                n_read = self.macro_layers - first - 1
                for layer in range(first + 1, self.macro_layers):
                    self.focus_attention[str(layer)] = FocusCrossAttention(
                        action.hidden_dim, focus.d_read, self.config.focus_heads, self.config.focus_head_dim,
                        num_layers=n_read)
        elif self.config.action_read == "compact" and action is not None:
            raise ValueError("compact action reading requires a focus configuration and a world expert")
        self.read_action_head: ReadActionHead | None = None
        if self.reader is not None and action is not None and self.config.read_action_horizon > 0:
            self.read_action_head = ReadActionHead(focus.d_read, len(self.reader.modalities),
                                                   self.config.read_action_horizon, self.config.read_action_dim)

        # Task-progress heads on the layer-0 clean tokens (+ pooled text): p_video from the current video frame,
        # p_body from the current Track conditions / anchor and proprio.  Used as loss targets and, in the
        # reading interface, as a shared phase coordinate when video and embodiment streams are not aligned.
        self.progress_video_head = ProgressHead(video.hidden_dim, self.config.text_dim) if video is not None else None
        self.progress_body_head = ProgressHead(track.hidden_dim, self.config.text_dim, self.config.proprio_dim) \
            if track is not None else None
        # p_action from the final-layer Action tokens (the Action expert has no clean prefix; after the stack its
        # tokens have read the observation, the world interface and the proprio) -> same phase label as p_body.
        self.progress_action_head = ProgressHead(action.hidden_dim, self.config.text_dim, self.config.proprio_dim) \
            if action is not None else None

    # ------------------------------------------------------------------
    def _resolve_read_layers(self, spec: Sequence[int] | str) -> tuple[int, ...]:
        if isinstance(spec, str):
            anchor = self.track if self.track is not None else self.video
            layers = [l for l in range(self.macro_layers) if anchor.updates_at(l)]
            if spec == "track_upper_half":
                layers = [l for l in layers if l >= self.macro_layers // 2]
            elif spec == "all_track_layers":
                pass
            elif spec == "last":
                layers = layers[-1:]
            else:
                raise ValueError(f"unknown read_layers spec {spec!r}")
            return tuple(layers)
        layers = tuple(sorted(int(l) for l in spec))
        if any(l < 0 or l >= self.macro_layers for l in layers):
            raise ValueError("read layer outside the macro depth")
        return layers

    @property
    def experts(self) -> dict[str, nn.Module]:
        return {n: e for n, e in (("video", self.video), ("track", self.track), ("action", self.action)) if e is not None}

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        for name, expert in self.experts.items():
            groups[name] = [p for p in expert.parameters() if p.requires_grad]
        focus_params = list(self.focus_attention.parameters())
        if self.reader is not None:
            focus_params += list(self.reader.parameters())
        for head in (self.progress_video_head, self.progress_body_head, self.progress_action_head):
            if head is not None:  # fresh heads: same LR as the reader
                focus_params += list(head.parameters())
        focus_params += list(self.embodiment_embedding.parameters())
        if self.read_action_head is not None:
            focus_params += list(self.read_action_head.parameters())
        groups["focus"] = [p for p in focus_params if p.requires_grad]
        groups["proprio"] = [p for p in self.proprio_encoder.parameters() if p.requires_grad]
        return {k: v for k, v in groups.items() if v}

    # ------------------------------------------------------------------
    def _context(self, batch: ModelInput) -> tuple[Tensor, Tensor]:
        context = batch.context
        b, l = context.shape[:2]
        mask = batch.context_mask
        if mask is None:
            mask = torch.ones(b, l, dtype=torch.bool, device=context.device)
        mask = mask.to(device=context.device, dtype=torch.bool)
        if batch.embodiment is not None:
            token = self.embodiment_embedding(batch.embodiment.to(context.device).long())[:, None].to(context.dtype)
            context = torch.cat((context, token), dim=1)
            mask = torch.cat((mask, torch.ones(b, 1, dtype=torch.bool, device=context.device)), dim=1)
        if batch.proprio is None:
            return context, mask
        proprio = batch.proprio.reshape(b, -1).to(context.dtype)
        token = self.proprio_encoder(proprio)[:, None]
        valid = torch.ones(b, 1, dtype=torch.bool, device=context.device)
        if batch.proprio_mask is not None:
            keep = batch.proprio_mask.reshape(b, -1).any(dim=-1)
            token = token * keep.to(token.dtype)[:, None, None]
            valid = keep[:, None]
        return torch.cat((context, token), dim=1), torch.cat((mask, valid), dim=1)

    def _prepare(self, batch: ModelInput, context: Tensor, mask: Tensor) -> dict[str, ExpertState]:
        states: dict[str, ExpertState] = {}
        if batch.video is not None:
            if self.video is None:
                raise ValueError("video input given but no Video expert")
            states["video"] = self.video.prepare(
                latents=batch.video, timestep=batch.video_sigma * 1000.0, context=context, context_mask=mask)
        if batch.track is not None:
            if self.track is None:
                raise ValueError("track input given but no Track expert")
            if batch.track_conditions is None:
                raise ValueError("track input requires rgb/depth/mask conditions")
            states["track"] = self.track.prepare(
                latents=batch.track, timestep=batch.track_sigma * 1000.0,
                conditions=batch.track_conditions, context=context, context_mask=mask, camera=batch.camera,
                condition_present=batch.condition_present)
        if batch.action is not None:
            if self.action is None:
                raise ValueError("action input given but no Action expert")
            a_context, a_mask = self._action_context(batch, context, mask)
            states["action"] = self.action.prepare(
                actions=batch.action, timestep=batch.action_sigma * 1000.0, context=a_context, context_mask=a_mask)
        return states

    @staticmethod
    def _action_context(batch: ModelInput, context: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Action-only shortcut dropout: hide the task text and/or the proprio token from the Action expert.

        Dropped positions are zeroed and masked; a dropped text keeps its first position as a zero-content
        "null" key so the cross-attention never sees an all-masked row.  Other experts and the reader keep
        the full conditions, so the withheld information is only reachable through the world interface.
        """
        if batch.drop_action_text is None and batch.drop_action_proprio is None:
            return context, mask
        context, mask = context.clone(), mask.clone()
        has_proprio = batch.proprio is not None
        appended = int(has_proprio) + int(batch.embodiment is not None)  # [text | embodiment? | proprio?]
        text_len = context.shape[1] - appended
        if batch.drop_action_text is not None:
            rows = batch.drop_action_text.to(context.device)
            context[rows, :text_len] = 0
            mask[rows, :text_len] = False
            mask[rows, 0] = True
        if batch.drop_action_proprio is not None and has_proprio:
            rows = batch.drop_action_proprio.to(context.device)
            context[rows, -1] = 0
            mask[rows, -1] = False
        return context, mask

    def _clean_flags(self, batch: ModelInput) -> dict[str, Tensor | None]:
        return {
            "video": None if batch.video_sigma is None else batch.video_sigma <= 0,
            "track": None if batch.track_sigma is None else batch.track_sigma <= 0,
            "action": None if batch.action_sigma is None else batch.action_sigma <= 0,
        }

    # ------------------------------------------------------------------
    def _layer(
        self,
        layer: int,
        states: dict[str, ExpertState],
        cached_track: tuple[Tensor, Tensor] | None,
        focus_tokens: dict[str, Tensor] | None,
        batch: ModelInput,
        clean: dict[str, Tensor | None],
        context: Tensor,
        context_mask: Tensor,
        mask_cache: dict | None = None,
        progress: Tensor | None = None,
    ) -> tuple[dict[str, ExpertState], tuple[Tensor, Tensor] | None, FocusOutput | None]:
        b = batch.batch_size
        device = context.device
        qs, ks, vs, posts = [], [], [], {}
        key_len: dict[str, int] = {}
        query_len: dict[str, int] = {}
        track_kv = cached_track

        for name in ("video", "track", "action"):
            state = states.get(name)
            if state is None:
                continue
            expert = self.experts[name]
            total = state.tokens.shape[1]
            clean_count = state.clean_count
            if expert.updates_at(layer):
                q, k, v, post = expert.pre_attn(layer, state)
                posts[name] = post
                qs.append(q)
                ks.append(k)
                vs.append(v)
                if name == "track":
                    track_kv = (k, v)
            elif name == "track":
                if track_kv is None:
                    raise RuntimeError("Track must run at macro layer zero")
                ks.append(track_kv[0])
                vs.append(track_kv[1])
            else:
                raise RuntimeError(f"{name} expert must update at every macro layer")
            if name == "action":
                key_len["action"] = total
                if expert.updates_at(layer):
                    query_len["action"] = total
            else:
                key_len[f"{name}_clean"] = clean_count
                key_len[f"{name}_future"] = total - clean_count
                if expert.updates_at(layer):
                    query_len[f"{name}_clean"] = clean_count
                    query_len[f"{name}_future"] = total - clean_count

        keys = TokenLayout.from_lengths(KEY_ORDER, key_len)
        queries = TokenLayout.from_lengths(QUERY_SEGMENTS, query_len)
        dense_bias = batch.dense_read_bias if self.config.action_read == "compact" else None
        floor = self.config.dense_read_floor
        if self.config.action_read == "compact" and floor is not None:
            # Safety floor: the dense future keys never disappear from Action's joint softmax.
            dense_bias = float(floor) if dense_bias is None else max(float(dense_bias), float(floor))
        mask_key = (keys.segments, queries.segments)
        mask = mask_cache.get(mask_key) if mask_cache is not None else None
        if mask is None:
            mask = build_visibility(
                keys, queries, batch_size=b, action_read=self.config.action_read,
                drop_video=batch.drop_video, drop_track=batch.drop_track,
                clean_video=clean["video"], clean_track=clean["track"], clean_action=clean["action"],
                dense_bias=dense_bias, device=device)
            if mask_cache is not None:
                mask_cache[mask_key] = mask
        attended = joint_attention(torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1), mask,
                                   heads=self.num_heads)
        pieces = queries.split(attended)

        new_states = dict(states)
        for name, post in posts.items():
            if name == "action":
                out = pieces["action"]
            else:
                out = torch.cat((pieces[f"{name}_clean"], pieces[f"{name}_future"]), dim=1)
            new_states[name] = self.experts[name].post_attn(layer, states[name], out.contiguous(), post)

        # Compact interface: gated cross-attention from Action to the latest interaction tokens.
        if (self.config.action_read == "compact" and focus_tokens is not None and "action" in new_states
                and str(layer) in self.focus_attention):
            z_parts, mask_parts = [], []
            open_action = ~clean["action"] if clean["action"] is not None else torch.ones(b, dtype=torch.bool, device=device)
            for name, drop in (("video", batch.drop_video), ("track", batch.drop_track)):
                z = focus_tokens.get(name)
                if z is None:
                    continue
                keep = open_action if drop is None else open_action & ~drop.to(device)
                z_parts.append(z)
                mask_parts.append(keep[:, None].expand(b, z.shape[1]))
            if z_parts:
                a_state = new_states["action"]
                residual = self.focus_attention[str(layer)](
                    a_state.tokens, torch.cat(z_parts, dim=1), torch.cat(mask_parts, dim=1))
                with torch.no_grad():  # diagnostic: how much the Action reads from the compact tokens
                    ratio = residual.float().norm(dim=-1) / a_state.tokens.float().norm(dim=-1).clamp(min=1e-6)
                    self._read_ratios.append(ratio.mean())
                a_state.tokens = a_state.tokens + residual

        focus_out = None
        if self.reader is not None and layer in self.read_layers:
            ts, vs_ = new_states.get("track"), new_states.get("video")
            focus_out = self.reader(
                track_hidden=self.track.hidden(ts) if ts is not None else None,
                track_grid=ts.grid if ts is not None else None,
                video_hidden=vs_.tokens if vs_ is not None else None,
                video_grid=vs_.grid if vs_ is not None else None,
                context=batch.context, context_mask=batch.context_mask,
                proprio=batch.proprio, proprio_mask=batch.proprio_mask,
                drop_video=batch.drop_video, drop_track=batch.drop_track,
                camera_motion=self._camera_motion(batch, ts) if ts is not None else None,
                progress=progress,
            )
            if focus_out.track_feedback is not None and ts is not None:
                cond = ts.extras["condition_count"] + ts.extras["anchor_count"]
                n = focus_out.track_feedback.shape[1]  # future grid tokens only (camera tokens untouched)
                tokens = ts.tokens.clone()
                tokens[:, cond:cond + n] = tokens[:, cond:cond + n] + focus_out.track_feedback
                ts.tokens = tokens
        return new_states, track_kv, focus_out

    def _progress(self, batch: ModelInput, states: dict[str, ExpertState], context: Tensor, mask: Tensor
                  ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Progress logits from the layer-0 clean tokens; the reader receives the detached sigmoid triple."""
        p_video = p_body = None
        if self.progress_video_head is not None and "video" in states:
            vs = states["video"]
            p_video = self.progress_video_head(vs.tokens[:, :vs.clean_count], context, mask)
        if self.progress_body_head is not None and "track" in states:
            ts = states["track"]
            proprio = batch.proprio if batch.proprio is not None else None
            p_body = self.progress_body_head(ts.tokens[:, :ts.clean_count], context, mask, proprio)
        if self.reader is None or self.config.focus is None or self.config.focus.progress_dim == 0:
            return p_video, p_body, None
        b = batch.batch_size
        pv = torch.sigmoid(p_video.detach().float()) if p_video is not None else torch.zeros(b, 1, device=context.device)
        pb = torch.sigmoid(p_body.detach().float()) if p_body is not None else pv
        return p_video, p_body, torch.cat((pv, pb, pv - pb), dim=-1)

    @torch.no_grad()
    def _camera_motion(self, batch: ModelInput, track_state: ExpertState) -> Tensor | None:
        """``[B, N_f, 2]`` ego-motion magnitude (|t|, angle) from the current x0 estimate of the camera tokens.

        ``x0 = x_sigma - sigma * v`` with the camera head applied to the current layer's tokens; detached, so it
        only informs the saliency ``phi`` ("is the head moving in this transition") without training the head.
        """
        if batch.camera is None:
            return None
        velocity = self.track.camera_prediction(track_state)
        if velocity is None:
            return None
        sigma = batch.track_sigma.to(batch.camera.dtype).view(-1, 1, 1)
        x0 = batch.camera - sigma * velocity.to(batch.camera.dtype)
        return camera_motion_magnitude(x0)

    def forward(self, batch: ModelInput) -> ModelOutput:
        context, context_mask = self._context(batch)
        states = self._prepare(batch, context, context_mask)
        if not states:
            raise ValueError("no modality present in the batch")
        clean = self._clean_flags(batch)
        cached_track: tuple[Tensor, Tensor] | None = None
        focus_tokens: dict[str, Tensor] | None = None
        focus_outputs: list[FocusOutput] = []
        mask_cache: dict = {}  # visibility masks are layout-dependent only; reuse across layers
        self._read_ratios: list[Tensor] = []
        progress_video, progress_body, progress = self._progress(batch, states, context, context_mask)

        for layer in range(self.macro_layers):
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                def run(*tokens, layer=layer, states=states, cached=cached_track, focus=focus_tokens):
                    local = {name: st.shallow_copy(tok) for (name, st), tok in zip(states.items(), tokens)}
                    return self._layer(layer, local, cached, focus, batch, clean, context, context_mask, mask_cache,
                                       progress)
                states, cached_track, focus_out = checkpoint(
                    run, *(st.tokens for st in states.values()), use_reentrant=False)
            else:
                states, cached_track, focus_out = self._layer(
                    layer, states, cached_track, focus_tokens, batch, clean, context, context_mask, mask_cache,
                    progress)
            if focus_out is not None:
                focus_outputs.append(focus_out)
                focus_tokens = focus_out.tokens

        out = ModelOutput(focus=focus_outputs, progress_video=progress_video, progress_body=progress_body,
                          read_ratio=torch.stack(self._read_ratios).mean() if self._read_ratios else None)
        if "video" in states:
            out.video_velocity = self.video.finalize(states["video"])
        if "track" in states:
            out.track_velocity = self.track.finalize(states["track"])
            out.camera_velocity = self.track.camera_prediction(states["track"])
            out.condition_reconstruction, out.condition_target, out.condition_present = \
                self.track.condition_reconstruction(states["track"])
        if "action" in states:
            out.action_velocity = self.action.finalize(states["action"])
            if self.read_action_head is not None and focus_outputs and focus_outputs[-1].tokens:
                out.read_action = self.read_action_head(focus_outputs[-1].tokens, focus_outputs[-1].cond)
            if self.progress_action_head is not None:
                out.progress_action = self.progress_action_head(states["action"].tokens, context, context_mask,
                                                                batch.proprio)
        return out


__all__ = ["MetisWAM4D", "MetisWAM4DConfig", "ModelInput", "ModelOutput"]
