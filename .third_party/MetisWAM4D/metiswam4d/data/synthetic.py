"""Synthetic samples that honour the contract (tests and smoke runs).

Roles are drawn as two rectangular blobs (body, object) so the structured
Track loss and the role / displacement supervision have non-trivial masks; the
object displacement follows the body displacement in the second half of the
window so a coupling transition exists.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from metiswam4d.data.camera import identity_code
from metiswam4d.data.contract import SampleSpec


class SyntheticLatentDataset(Dataset):
    def __init__(self, spec: SampleSpec, length: int = 64, *, with_video: bool = True,
                 with_track: bool = True, with_action: bool = True, with_roles: bool = True,
                 with_camera: bool = True, text_len: int = 12, seed: int = 0, embodiment: int = 1):
        self.spec, self.length = spec, length
        self.with_video, self.with_track, self.with_action, self.with_roles = with_video, with_track, with_action, with_roles
        self.with_camera = with_camera
        self.text_len, self.seed, self.embodiment = text_len, seed, embodiment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + index)
        s = self.spec
        progress = torch.rand(1, generator=g)
        sample: dict = {
            "text_context": torch.randn(self.text_len - index % 3, s.text_dim, generator=g),
            "embodiment": self.embodiment,
            "key": f"synthetic/{index}",
            "progress_video": progress,
            "progress_body": progress.clone(),
        }
        if self.with_video:
            sample["video_clean"] = torch.randn(s.video_channels, s.video_frames, s.video_height, s.video_width, generator=g)
        if self.with_track:
            sample["track_clean"] = torch.randn(s.track_channels, s.track_frames, s.track_height, s.track_width, generator=g)
            for key in ("rgb_condition", "depth_condition", "mask_condition"):
                sample[key] = torch.randn(s.track_channels, 1, s.track_height, s.track_width, generator=g)
            if self.with_roles:
                role = torch.zeros(s.track_frames, s.track_height, s.track_width, dtype=torch.int8)
                h, w = s.track_height, s.track_width
                role[:, h // 4: h // 2, : w // 2] = 1
                role[:, h // 2: 3 * h // 4, w // 2:] = 2
                sample["track_role"] = role
                sample["track_valid"] = (role > 0)[None]
                th, tw = s.track_token_grid
                n = s.frame_slots
                role_frames = torch.zeros(n, th, tw, dtype=torch.int8)
                role_frames[:, th // 4: th // 2, : tw // 2] = 1
                role_frames[:, th // 2: 3 * th // 4, tw // 2:] = 2
                body = torch.randn(3, generator=g) * 0.3
                disp = torch.zeros(n, th, tw, 3)
                disp[:, role_frames[0] == 1] = body
                for t in range(n // 2, n):  # object starts moving with the body: coupling transition
                    disp[t, role_frames[0] == 2] = body
                sample["track_role_frames"] = role_frames
                sample["track_disp_frames"] = disp
            if self.with_camera:
                # Static camera (robot head / fixed rig): identity ego-motion, fully supervised.
                sample["camera_delta"] = identity_code(s.frame_slots)
                sample["camera_valid"] = torch.ones(s.frame_slots, dtype=torch.bool)
        if self.with_action:
            action = torch.zeros(s.action_horizon, s.action_dim)
            mask = torch.zeros(s.action_horizon, s.action_dim, dtype=torch.bool)
            for lo in (0, 34):
                action[:, lo:lo + 10] = torch.rand(s.action_horizon, 10, generator=g) * 2 - 1
                mask[:, lo:lo + 10] = True
            sample["action"], sample["action_mask"] = action, mask
            sample["proprio"] = action[:1].clone()
            sample["proprio_mask"] = mask[:1].clone()
        return sample


__all__ = ["SyntheticLatentDataset"]
