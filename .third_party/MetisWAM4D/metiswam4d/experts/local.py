"""Self-contained Wan-style DiT experts.

``WanStyleBlock`` reproduces the parameter names and computation order of the
Wan2.2 DiT block (``self_attn.{q,k,v,o,norm_q,norm_k}``, ``cross_attn.*``,
``norm1/2/3``, ``ffn.{0,2}``, ``modulation``) so that a Track expert can be
initialised from Wan / OpenWAM-Alpha video weights by layer mapping and width
interpolation, and so that tiny instances can stand in for the Video and
Action experts in CPU tests.  The production Video / Action experts are the
OpenWAM-Alpha modules behind ``experts/alpha.py``; all three speak the same
``ExpertAdapter`` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from metiswam4d.experts.protocol import (
    ExpertAdapter, ExpertState, macro_layer_map, merge_heads, split_heads,
)


# ----------------------------------------------------------------------------
# Building blocks (Wan-compatible)
# ----------------------------------------------------------------------------


def sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    half = dim // 2
    position = position.reshape(-1).to(torch.float64)
    freqs = torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64) / half)
    sinusoid = torch.outer(position, freqs)
    return torch.cat([sinusoid.cos(), sinusoid.sin()], dim=1)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float64), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(head_dim: int, end: int = 1024) -> tuple[Tensor, Tensor, Tensor]:
    """Wan split: temporal ``d - 2*(d//3)``, height ``d//3``, width ``d//3``."""
    return (
        precompute_freqs_cis(head_dim - 2 * (head_dim // 3), end),
        precompute_freqs_cis(head_dim // 3, end),
        precompute_freqs_cis(head_dim // 3, end),
    )


def grid_rope(freqs: tuple[Tensor, Tensor, Tensor], positions: Tensor) -> Tensor:
    """Complex ``[S, 1, head_dim/2]`` RoPE table for integer (t, h, w) positions."""
    f, h, w = freqs
    table = torch.cat((
        f[positions[:, 0]], h[positions[:, 1]], w[positions[:, 2]],
    ), dim=-1)
    return table.unsqueeze(1)


def rope_apply(x: Tensor, freqs: Tensor, num_heads: int) -> Tensor:
    """Apply a complex RoPE table ``[S, 1, D/2]`` to ``[B, S, H*D]`` (Wan numerics: float64 complex)."""
    b, s, width = x.shape
    x_heads = x.reshape(b, s, num_heads, width // num_heads)
    x_complex = torch.view_as_complex(x_heads.to(torch.float64).reshape(b, s, num_heads, -1, 2))
    if freqs.device != x.device:
        freqs = freqs.to(x.device)
    rotated = torch.view_as_real(x_complex * freqs).flatten(2)
    return rotated.to(x.dtype)


class _RopeCache:
    """Per-(grid, device) cache of complex RoPE tables (plain attribute, never a parameter)."""

    def __init__(self):
        self._tables: dict[tuple, Tensor] = {}

    def get(self, key: tuple, device: torch.device, build) -> Tensor:
        cache_key = (*key, str(device))
        table = self._tables.get(cache_key)
        if table is None:
            table = build().to(device)
            self._tables[cache_key] = table
        return table


def grid_positions(grid: tuple[int, int, int], device: torch.device | None = None) -> Tensor:
    axes = [torch.arange(n, device=device) for n in grid]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class WanSelfAttention(nn.Module):
    """Private Q/K/V/O projections into the shared attention space."""

    def __init__(self, dim: int, attention_width: int, num_heads: int, eps: float):
        super().__init__()
        if attention_width % num_heads:
            raise ValueError("attention width must be divisible by heads")
        self.num_heads = num_heads
        self.head_dim = attention_width // num_heads
        self.q = nn.Linear(dim, attention_width)
        self.k = nn.Linear(dim, attention_width)
        self.v = nn.Linear(dim, attention_width)
        self.o = nn.Linear(attention_width, dim)
        self.norm_q = RMSNorm(attention_width, eps)
        self.norm_k = RMSNorm(attention_width, eps)

    def qkv(self, x: Tensor, rope: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        if rope is not None:
            q = rope_apply(q, rope, self.num_heads)
            k = rope_apply(k, rope, self.num_heads)
        return q, k, v


class WanCrossAttention(nn.Module):
    def __init__(self, dim: int, attention_width: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, attention_width)
        self.k = nn.Linear(dim, attention_width)
        self.v = nn.Linear(dim, attention_width)
        self.o = nn.Linear(attention_width, dim)
        self.norm_q = RMSNorm(attention_width, eps)
        self.norm_k = RMSNorm(attention_width, eps)

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None) -> Tensor:
        q = split_heads(self.norm_q(self.q(x)), self.num_heads)
        k = split_heads(self.norm_k(self.k(context)), self.num_heads)
        v = split_heads(self.v(context), self.num_heads)
        mask = None
        if context_mask is not None:
            mask = context_mask[:, None, None, :].to(device=q.device, dtype=torch.bool)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(merge_heads(out))


class WanStyleBlock(nn.Module):
    """Wan DiT block split into pre- and post-attention halves."""

    def __init__(self, dim: int, attention_width: int, num_heads: int, ffn_dim: int, eps: float):
        super().__init__()
        self.self_attn = WanSelfAttention(dim, attention_width, num_heads, eps)
        self.cross_attn = WanCrossAttention(dim, attention_width, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def _chunks(self, t_mod: Tensor) -> tuple[Tensor, ...]:
        if t_mod.ndim == 4:  # [B, S, 6, dim]
            mod = self.modulation[:, None].to(t_mod) + t_mod
            return tuple(mod.unbind(dim=2))
        mod = self.modulation.to(t_mod) + t_mod  # [B, 6, dim]
        return tuple(piece for piece in mod.chunk(6, dim=1))

    def pre_attention(self, x: Tensor, t_mod: Tensor, rope: Tensor | None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._chunks(t_mod)
        hidden = self.norm1(x) * (1 + scale_msa) + shift_msa
        q, k, v = self.self_attn.qkv(hidden, rope)
        return q, k, v, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, attn_out: Tensor, post: tuple[Tensor, ...],
                       context: Tensor | None, context_mask: Tensor | None) -> Tensor:
        residual, gate_msa, shift_mlp, scale_mlp, gate_mlp = post
        x = residual + gate_msa * self.self_attn.o(attn_out)
        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context, context_mask)
        hidden = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.ffn(hidden)

    def forward(self, stage: str, *args):
        """Dispatch through ``__call__`` so FSDP hooks gather parameters for both halves."""
        if stage == "pre":
            return self.pre_attention(*args)
        if stage == "post":
            return self.post_attention(*args)
        raise ValueError(stage)


class WanHead(nn.Module):
    def __init__(self, dim: int, out_features: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_features)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 3:  # per-token [B, S, dim]
            shift, scale = (self.modulation[:, None].to(t) + t[:, :, None]).unbind(dim=2)
        else:
            shift, scale = (self.modulation.to(t) + t.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


# ----------------------------------------------------------------------------
# Core expert
# ----------------------------------------------------------------------------


@dataclass
class ExpertCoreConfig:
    dim: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 20
    num_heads: int = 24
    head_dim: int = 128
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    macro_layers: int | None = None  # depth of the joint stack; None = own depth

    @property
    def attention_width(self) -> int:
        return self.num_heads * self.head_dim


class MetisExpertCore(ExpertAdapter):
    """Blocks, time embedding, text embedding.  Subclasses add I/O stems."""

    def __init__(self, core: ExpertCoreConfig, name: str):
        super().__init__()
        self.name = name
        self.core = core
        c = core
        self.text_embedding = nn.Sequential(
            nn.Linear(c.text_dim, c.dim), nn.GELU(approximate="tanh"), nn.Linear(c.dim, c.dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(c.freq_dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(c.dim, c.dim * 6))
        self.blocks = nn.ModuleList([
            WanStyleBlock(c.dim, c.attention_width, c.num_heads, c.ffn_dim, c.eps)
            for _ in range(c.num_layers)
        ])
        macro = c.macro_layers or c.num_layers
        self._macro_map = {macro_index: block for block, macro_index in
                           enumerate(macro_layer_map(c.num_layers, macro))}
        self._macro_layers = macro

    # protocol -----------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return self.core.num_layers

    @property
    def macro_layers(self) -> int:
        return self._macro_layers

    @property
    def num_heads(self) -> int:
        return self.core.num_heads

    @property
    def head_dim(self) -> int:
        return self.core.head_dim

    @property
    def hidden_dim(self) -> int:
        return self.core.dim

    def block_index(self, layer: int) -> int | None:
        return self._macro_map.get(layer)

    # helpers -----------------------------------------------------------
    def time_features(self, timestep: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """``(time_embed [N, dim], modulation [N, 6, dim])`` for flat timesteps."""
        emb = self.time_embedding(sinusoidal_embedding_1d(self.core.freq_dim, timestep).to(dtype))
        return emb, self.time_projection(emb).unflatten(1, (6, self.core.dim))

    def embed_context(self, context: Tensor | None, context_mask: Tensor | None,
                      batch: int, device: torch.device, dtype: torch.dtype):
        if context is None:
            return None, None
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        return self.text_embedding(context.to(dtype)), context_mask.to(device=device, dtype=torch.bool)

    def pre_attn(self, layer: int, state: ExpertState):
        index = self.block_index(layer)
        if index is None:
            raise RuntimeError(f"{self.name} has no block at macro layer {layer}")
        return self.blocks[index]("pre", state.tokens, state.modulation, state.rope)

    def post_attn(self, layer: int, state: ExpertState, attn_out: Tensor, post: Any) -> ExpertState:
        index = self.block_index(layer)
        state.tokens = self.blocks[index]("post", attn_out, post, state.context, state.context_mask)
        return state


# ----------------------------------------------------------------------------
# Latent (video-like) expert
# ----------------------------------------------------------------------------


@dataclass
class LatentExpertConfig:
    core: ExpertCoreConfig = field(default_factory=ExpertCoreConfig)
    in_channels: int = 48
    out_channels: int = 48
    patch_size: tuple[int, int, int] = (1, 2, 2)
    clean_frames: int = 1  # leading latent frames that are clean conditions
    rope_max: int = 1024


class LatentExpert(MetisExpertCore):
    """Wan-like denoiser over ``[B, C, F, H, W]`` latents with a clean prefix.

    Timestep enters per token: clean prefix frames use t = 0 (TI2V convention).
    """

    def __init__(self, config: LatentExpertConfig, name: str = "video"):
        super().__init__(config.core, name)
        self.config = config
        p = config.patch_size
        self.patch_embedding = nn.Conv3d(config.in_channels, config.core.dim, p, stride=p)
        self.head = WanHead(config.core.dim, config.out_channels * math.prod(p), config.core.eps)
        # Complex tables stay plain attributes: Module.to(dtype) would drop the imaginary part.
        self.rope_tables = precompute_freqs_cis_3d(config.core.head_dim, config.rope_max)
        self._rope_cache = _RopeCache()

    def rope_for(self, positions_key: tuple, device: torch.device, positions_fn) -> Tensor:
        return self._rope_cache.get(positions_key, device, lambda: grid_rope(self.rope_tables, positions_fn()))

    def tokenize(self, latents: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        x = self.patch_embedding(latents)
        grid = tuple(int(n) for n in x.shape[-3:])
        return x.flatten(2).transpose(1, 2), grid

    def untokenize(self, x: Tensor, grid: tuple[int, int, int]) -> Tensor:
        b = x.shape[0]
        pf, ph, pw = self.config.patch_size
        f, h, w = grid
        c = self.config.out_channels
        x = x.view(b, f, h, w, pf, ph, pw, c)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, c, f * pf, h * ph, w * pw)

    def per_token_time(self, timestep: Tensor, grid: tuple[int, int, int],
                       clean_frames: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Per-token ``(head_time [B,S,dim], modulation [B,S,6,dim])``."""
        b = timestep.shape[0]
        f, h, w = grid
        per_frame = h * w
        noisy_emb, noisy_mod = self.time_features(timestep, dtype)
        clean_emb, clean_mod = self.time_features(torch.zeros_like(timestep), dtype)
        frame_emb = torch.stack([clean_emb] * clean_frames + [noisy_emb] * (f - clean_frames), dim=1)
        frame_mod = torch.stack([clean_mod] * clean_frames + [noisy_mod] * (f - clean_frames), dim=1)
        head_time = frame_emb[:, :, None].expand(b, f, per_frame, -1).reshape(b, f * per_frame, -1)
        modulation = frame_mod[:, :, None].expand(b, f, per_frame, 6, -1).reshape(b, f * per_frame, 6, -1)
        return head_time, modulation

    def prepare(self, *, latents: Tensor, timestep: Tensor, context: Tensor | None = None,
                context_mask: Tensor | None = None, **_: Any) -> ExpertState:
        tokens, grid = self.tokenize(latents)
        b, s, _ = tokens.shape
        if timestep.shape != (b,):
            raise ValueError("timestep must be [B]")
        clean = min(self.config.clean_frames, grid[0])
        head_time, modulation = self.per_token_time(timestep, grid, clean, tokens.dtype)
        rope = self.rope_for(("latent", grid), tokens.device, lambda: grid_positions(grid))
        ctx, ctx_mask = self.embed_context(context, context_mask, b, tokens.device, tokens.dtype)
        return ExpertState(
            tokens=tokens, modulation=modulation, head_time=head_time, rope=rope,
            context=ctx, context_mask=ctx_mask, clean_count=clean * grid[1] * grid[2], grid=grid,
        )

    def finalize(self, state: ExpertState) -> Tensor:
        return self.untokenize(self.head(state.tokens, state.head_time), state.grid)


# ----------------------------------------------------------------------------
# Track expert: latent expert + condition stem + role embedding
# ----------------------------------------------------------------------------


@dataclass
class TrackExpertConfig:
    core: ExpertCoreConfig = field(default_factory=ExpertCoreConfig)
    channels: int = 48
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_max: int = 1024
    condition_modalities: int = 3  # rgb, depth, mask
    camera_dim: int = 9            # per-transition camera delta pose: translation 3 + rot6d 6


class TrackExpert(LatentExpert):
    """Track4D denoiser.  Token layout ``[condition | anchor | future | camera]``.

    The condition tokens fuse the current RGB / metric depth / semantic mask
    latents (each RMS-normalised, tagged with a modality embedding); the
    anchor (first latent frame, zero displacement) and the conditions are clean
    and modulated with t = 0.  A reconstruction head regresses the normalised
    condition patches from the fused tokens (auxiliary loss).

    Optional camera tokens (one per stride transition) carry the head-camera
    ego-motion ``T_{t -> t+delta}`` and are denoised on the Track clock together
    with the displacement field; Track4D itself is ego-motion compensated
    (world motion of the surface point expressed in the camera frame at t), so
    the two are complementary rather than redundant.
    """

    def __init__(self, config: TrackExpertConfig, name: str = "track"):
        super().__init__(_latent_config_for_track(config), name)
        self.track_config = config
        c = config
        patch_volume = math.prod(c.patch_size)
        self.condition_norms = nn.ModuleList(
            [RMSNorm(c.channels, c.core.eps) for _ in range(c.condition_modalities)])
        self.condition_modality = nn.Embedding(c.condition_modalities, c.channels)
        nn.init.zeros_(self.condition_modality.weight)
        self.condition_gain = nn.Parameter(torch.ones(c.condition_modalities, c.channels))
        self.condition_fusion = nn.Conv3d(
            c.channels * c.condition_modalities, c.core.dim, c.patch_size, stride=c.patch_size)
        self.condition_mixer = nn.Sequential(
            nn.LayerNorm(c.core.dim, eps=c.core.eps), nn.Linear(c.core.dim, c.core.dim), nn.SiLU(),
            nn.Linear(c.core.dim, c.core.dim))
        nn.init.zeros_(self.condition_mixer[-1].weight)
        nn.init.zeros_(self.condition_mixer[-1].bias)
        self.condition_reconstruction_head = nn.Linear(
            c.core.dim, c.channels * c.condition_modalities * patch_volume)
        self.role_embedding = nn.Embedding(2, c.core.dim)  # 0 noisy future, 1 clean
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        # Learned stand-in latent for a condition modality that a data source does not provide (e.g. no
        # depth for KlingHumanEgo-2.5M-5000H); broadcast over the grid before the modality norm, so the fused
        # token still sees the modality tag and gain.  Absent from older checkpoints (kept fresh).
        self.condition_missing = nn.Parameter(torch.zeros(c.condition_modalities, c.channels))
        # Camera ego-motion tokens (fresh modules; absent from the v3 Track checkpoint).
        self.camera_embedding = nn.Linear(c.camera_dim, c.core.dim)
        self.camera_type = nn.Parameter(torch.randn(1, 1, c.core.dim) * 0.02)
        self.camera_head = nn.Sequential(nn.LayerNorm(c.core.dim, eps=c.core.eps), nn.Linear(c.core.dim, c.camera_dim))

    def condition_tokens(self, conditions: tuple[Tensor, ...], present: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``present [B, M]`` bool: modalities a sample provides; missing ones use ``condition_missing``."""
        normalized, fused_inputs = [], []
        for modality, (value, norm) in enumerate(zip(conditions, self.condition_norms)):
            if present is not None:
                keep = present[:, modality].to(value.dtype).view(-1, 1, 1, 1, 1)
                stand_in = self.condition_missing[modality].to(value.dtype).view(1, -1, 1, 1, 1)
                value = value * keep + stand_in * (1 - keep)
            v = norm(value.permute(0, 2, 3, 4, 1))
            normalized.append(v.permute(0, 4, 1, 2, 3))
            v = (v + self.condition_modality.weight[modality]) * self.condition_gain[modality]
            fused_inputs.append(v.permute(0, 4, 1, 2, 3))
        fused = self.condition_fusion(torch.cat(fused_inputs, dim=1)).flatten(2).transpose(1, 2)
        fused = fused + self.condition_mixer(fused)
        pf, ph, pw = self.track_config.patch_size
        targets = []
        for v in normalized:
            patches = v.unfold(2, pf, pf).unfold(3, ph, ph).unfold(4, pw, pw)
            targets.append(patches.permute(0, 2, 3, 4, 1, 5, 6, 7).flatten(1, 3).flatten(2))
        return fused, torch.cat(targets, dim=-1).detach()

    def prepare(self, *, latents: Tensor, timestep: Tensor, conditions: tuple[Tensor, ...],
                context: Tensor | None = None, context_mask: Tensor | None = None,
                camera: Tensor | None = None, condition_present: Tensor | None = None, **_: Any) -> ExpertState:
        tokens, grid = self.tokenize(latents)
        b = tokens.shape[0]
        if timestep.shape != (b,):
            raise ValueError("timestep must be [B]")
        if len(conditions) != self.track_config.condition_modalities:
            raise ValueError("unexpected number of Track conditions")
        anchor_count = grid[1] * grid[2]
        grid_count = tokens.shape[1]
        condition, condition_target = self.condition_tokens(conditions, condition_present)
        if condition.shape[1] != anchor_count:
            raise ValueError("Track conditions and latents must share the spatial grid")
        camera_count = 0
        if camera is not None:
            if camera.ndim != 3 or camera.shape[0] != b or camera.shape[-1] != self.track_config.camera_dim:
                raise ValueError(f"camera must be [B, N, {self.track_config.camera_dim}], got {tuple(camera.shape)}")
            camera_count = camera.shape[1]
            tokens = torch.cat((tokens, self.camera_embedding(camera.to(tokens.dtype)) + self.camera_type), dim=1)
        roles = torch.zeros(b, tokens.shape[1], dtype=torch.long, device=tokens.device)
        roles[:, :anchor_count] = 1
        roles[timestep <= 0] = 1  # a clean Track block is entirely a condition
        tokens = tokens + self.role_embedding(roles)

        head_time, modulation = self.per_token_time(timestep, grid, 1, tokens.dtype)
        clean_emb, clean_mod = self.time_features(torch.zeros_like(timestep), tokens.dtype)
        noisy_emb, noisy_mod = self.time_features(timestep, tokens.dtype)
        cond_count = condition.shape[1]
        head_time = torch.cat((clean_emb[:, None].expand(b, cond_count, -1), head_time,
                               noisy_emb[:, None].expand(b, camera_count, -1)), dim=1)
        modulation = torch.cat((clean_mod[:, None].expand(b, cond_count, 6, -1), modulation,
                                noisy_mod[:, None].expand(b, camera_count, 6, -1)), dim=1)

        def positions():
            track_pos = grid_positions(grid).clone()
            track_pos[:, 0] += 1
            parts = [grid_positions((1, grid[1], grid[2])), track_pos]
            if camera_count:
                # Off-grid spatial slot, temporal index beyond the latent frames (one per transition).
                cam = torch.stack((torch.arange(camera_count) + grid[0] + 1,
                                   torch.full((camera_count,), grid[1]), torch.full((camera_count,), grid[2])), dim=-1)
                parts.append(cam)
            return torch.cat(parts, dim=0)
        rope = self.rope_for(("track", grid, camera_count), tokens.device, positions)
        ctx, ctx_mask = self.embed_context(context, context_mask, b, tokens.device, tokens.dtype)
        state = ExpertState(
            tokens=torch.cat((condition, tokens), dim=1), modulation=modulation,
            head_time=head_time, rope=rope, context=ctx, context_mask=ctx_mask,
            clean_count=cond_count + anchor_count, grid=grid,
        )
        state.extras.update(condition_count=cond_count, anchor_count=anchor_count, grid_count=grid_count,
                            camera_count=camera_count, condition_tokens=condition, condition_target=condition_target,
                            condition_present=condition_present)
        return state

    def finalize(self, state: ExpertState) -> Tensor:
        cond, n = state.extras["condition_count"], state.extras["grid_count"]
        out = self.head(state.tokens[:, cond:cond + n], state.head_time[:, cond:cond + n])
        return self.untokenize(out, state.grid)

    def camera_prediction(self, state: ExpertState) -> Tensor | None:
        """Velocity prediction for the camera tokens, ``[B, N, camera_dim]`` (None without camera tokens)."""
        k = state.extras.get("camera_count", 0)
        if not k:
            return None
        return self.camera_head(state.tokens[:, -k:])

    def hidden(self, state: ExpertState) -> Tensor:
        """Anchor + future grid tokens (conditions and camera tokens excluded), ``[B, F*H*W, dim]``."""
        cond, n = state.extras["condition_count"], state.extras["grid_count"]
        return state.tokens[:, cond:cond + n]

    def condition_reconstruction(self, state: ExpertState) -> tuple[Tensor, Tensor, Tensor | None]:
        """``(prediction, target, present)``; the last two dims are laid out as ``[modality, channels * patch]``."""
        return (self.condition_reconstruction_head(state.extras["condition_tokens"]),
                state.extras["condition_target"], state.extras.get("condition_present"))


def _latent_config_for_track(config: TrackExpertConfig) -> LatentExpertConfig:
    return LatentExpertConfig(
        core=config.core, in_channels=config.channels, out_channels=config.channels,
        patch_size=config.patch_size, clean_frames=1, rope_max=config.rope_max)


# ----------------------------------------------------------------------------
# Action expert
# ----------------------------------------------------------------------------


@dataclass
class ActionExpertConfig:
    core: ExpertCoreConfig = field(default_factory=lambda: ExpertCoreConfig(num_layers=30))
    action_dim: int = 80
    max_action_len: int = 1024


class ActionExpert(MetisExpertCore):
    """Action-chunk denoiser mirroring OpenWAM-Alpha's separate Action DiT.

    Linear encoder / decoder (no AdaLN head), 1-D RoPE over action steps, one
    scalar timestep per sample; text and proprio arrive through ``context``.
    """

    def __init__(self, config: ActionExpertConfig, name: str = "action"):
        super().__init__(config.core, name)
        self.config = config
        self.action_encoder = nn.Linear(config.action_dim, config.core.dim)
        self.action_decoder = nn.Linear(config.core.dim, config.action_dim)
        self.rope_table = precompute_freqs_cis(config.core.head_dim, config.max_action_len)
        self._rope_cache = _RopeCache()

    def prepare(self, *, actions: Tensor, timestep: Tensor, context: Tensor | None = None,
                context_mask: Tensor | None = None, **_: Any) -> ExpertState:
        tokens = self.action_encoder(actions)
        b, s, _ = tokens.shape
        if timestep.shape != (b,):
            raise ValueError("timestep must be [B]")
        if s > self.config.max_action_len:
            raise ValueError("action chunk exceeds the RoPE table")
        emb, modulation = self.time_features(timestep, tokens.dtype)
        rope = self._rope_cache.get(("action", s), tokens.device, lambda: self.rope_table[:s].unsqueeze(1))
        ctx, ctx_mask = self.embed_context(context, context_mask, b, tokens.device, tokens.dtype)
        return ExpertState(tokens=tokens, modulation=modulation, head_time=emb, rope=rope,
                           context=ctx, context_mask=ctx_mask, clean_count=0, grid=None)

    def finalize(self, state: ExpertState) -> Tensor:
        return self.action_decoder(state.tokens)


__all__ = [
    "ActionExpert",
    "ActionExpertConfig",
    "ExpertCoreConfig",
    "LatentExpert",
    "LatentExpertConfig",
    "TrackExpert",
    "TrackExpertConfig",
    "WanStyleBlock",
    "grid_positions",
    "grid_rope",
    "precompute_freqs_cis",
    "precompute_freqs_cis_3d",
    "rope_apply",
    "sinusoidal_embedding_1d",
]
