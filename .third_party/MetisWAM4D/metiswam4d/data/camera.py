"""Camera ego-motion code carried by the Track expert's camera tokens.

One code per stride transition ``t -> t + delta`` (``N_f`` per window): the relative pose
``T_{t -> t+delta}`` of the camera at ``t + delta`` expressed in the camera frame at ``t``,
encoded as ``[translation / scale (3), rot6d - identity (6)]`` so that a static camera is the
zero vector (mirroring the zero-displacement Track4D of the static background).  Track4D itself
is ego-motion compensated (world motion of the surface point in the camera-``t`` axes), so the
displacement field and the camera code are complementary, not redundant.
"""
from __future__ import annotations

import torch
from torch import Tensor

CAMERA_DIM = 9
TRANSLATION_SCALE_M = 0.1          # head translation per 4 frames @30 fps is a few cm -> O(1) codes
_ROT6D_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def identity_code(*shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    """``[*shape, 9]`` codes of a static camera (all zeros)."""
    return torch.zeros(*shape, CAMERA_DIM, dtype=dtype, device=device)


def encode_camera_delta(translation: Tensor, rotation: Tensor, scale_m: float = TRANSLATION_SCALE_M) -> Tensor:
    """``translation [..., 3]`` (metres) and ``rotation [..., 3, 3]`` -> code ``[..., 9]``.

    rot6d takes the first two *columns* of ``R`` (Zhou et al. 2019).
    """
    rot6d = torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)
    ident = torch.as_tensor(_ROT6D_IDENTITY, dtype=rot6d.dtype, device=rot6d.device)
    return torch.cat((translation / scale_m, rot6d - ident), dim=-1)


def decode_camera_delta(code: Tensor, scale_m: float = TRANSLATION_SCALE_M) -> tuple[Tensor, Tensor]:
    """Code ``[..., 9]`` -> ``(translation [..., 3] metres, rotation [..., 3, 3])`` via Gram-Schmidt."""
    code = code.float()
    translation = code[..., :3] * scale_m
    ident = torch.as_tensor(_ROT6D_IDENTITY, dtype=code.dtype, device=code.device)
    six = code[..., 3:] + ident
    a1, a2 = six[..., :3], six[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return translation, torch.stack((b1, b2, b3), dim=-1)


def camera_motion_magnitude(code: Tensor) -> Tensor:
    """``[..., 9]`` -> ``[..., 2]``: translation norm (scale units) and rotation angle (radians)."""
    _, rotation = decode_camera_delta(code)
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    return torch.stack((code[..., :3].float().norm(dim=-1), angle), dim=-1)


__all__ = [
    "CAMERA_DIM", "TRANSLATION_SCALE_M", "camera_motion_magnitude", "decode_camera_delta",
    "encode_camera_delta", "identity_code",
]
