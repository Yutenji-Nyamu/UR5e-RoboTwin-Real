"""RoboDojo (RDJ_MetisWAM4D) loader: base-frame EEF20, window contract, unknown-role handling."""
from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from metiswam4d.data.robodojo.eef_base import (
    BASE_ROTATION, LEFT_BASE_POS, RIGHT_BASE_POS, base_to_world_eef20, world_to_base_eef20,
)
from metiswam4d.data.robodojo.episode_dataset import RDJEpisodeDataset, RDJWindowConfig, load_official_instructions
from metiswam4d.data.rt2.eef import SLOTS
from metiswam4d.data.rt2.episode_dataset import collate_raw

RDJ_ROOT = Path(RDJWindowConfig.root)
VENDOR = Path(os.environ.get("OPENWAM_VENDOR",
                             "/m2v_intern_v3/danglingwei/codes/wam_proj/JanusTrack4d_260824/janusact4d_rdj_imperfect_v1/vendor"))
needs_data = pytest.mark.skipif(not (RDJ_ROOT / "index.jsonl").exists(), reason="RDJ_MetisWAM4D not mounted")


def test_base_rotation_is_90deg_yaw():
    expected = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(BASE_ROTATION, expected, atol=1e-6)


def test_world_to_base_roundtrip_and_axes():
    rng = np.random.default_rng(0)
    e = np.zeros((5, 20), dtype=np.float32)
    for arm in (0, 10):
        e[:, arm:arm + 3] = rng.normal(size=(5, 3))
        # random rotations -> first two columns
        q = rng.normal(size=(5, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        from scipy.spatial.transform import Rotation
        m = Rotation.from_quat(q).as_matrix()
        e[:, arm + 3:arm + 6] = m[:, :, 0]
        e[:, arm + 6:arm + 9] = m[:, :, 1]
        e[:, arm + 9] = rng.uniform(0, 1, size=5)
    b = world_to_base_eef20(e)
    assert np.allclose(base_to_world_eef20(b), e, atol=1e-5)
    # base x = world y - (-0.45), base y = -(world x - x_base), base z = world z - 0.765; gripper untouched
    assert np.allclose(b[:, 0], e[:, 1] - LEFT_BASE_POS[1], atol=1e-5)
    assert np.allclose(b[:, 1], -(e[:, 0] - LEFT_BASE_POS[0]), atol=1e-5)
    assert np.allclose(b[:, 12], e[:, 12] - RIGHT_BASE_POS[2], atol=1e-5)
    assert np.allclose(b[:, [9, 19]], e[:, [9, 19]])
    # rot6d columns stay unit length and orthogonal
    assert np.allclose(np.linalg.norm(b[:, 3:6], axis=1), 1.0, atol=1e-5)
    assert np.allclose((b[:, 3:6] * b[:, 6:9]).sum(1), 0.0, atol=1e-5)


@needs_data
@pytest.mark.skipif(not (VENDOR / "openwam").exists(), reason="OpenWAM vendor code not available")
def test_base_frame_matches_openwam_reader():
    """Our eef20 (world, from the official state) -> base must equal OpenWAM's ``read_calibrated_eef20`` on the
    official episode file, which is what Alpha RoboDojo was trained and normalised on."""
    sys.path.insert(0, str(VENDOR))
    from openwam.dataloader.robodojo import read_calibrated_eef20
    from openwam.dataloader.robodojo_contract import arx_x5_calibration
    import h5py
    ep = RDJ_ROOT / "stack_blocks" / "train_4d" / "episode0"
    ours = world_to_base_eef20(h5py.File(ep / "track4d.h5")["eef20"][:64])
    ref = read_calibrated_eef20(ep / "raw.hdf5", arx_x5_calibration(), 0, 64)
    assert ours.shape == ref.shape == (64, 20)
    assert np.abs(ours - ref).max() < 2e-4, np.abs(ours - ref).max(axis=0)


@needs_data
def test_window_contract_and_action_range():
    ds = RDJEpisodeDataset(RDJWindowConfig(samples_per_episode=1, fixed_windows=True))
    assert len(ds.rows) == 3298
    item = ds[0]
    assert item["video_frames"].shape == (9, 384, 320, 3) and item["video_frames"].dtype == torch.uint8
    assert item["track_rgb"].shape == (9, 240, 320, 3)
    assert item["track_role_px"].shape == (9, 240, 320) and set(item["track_role_px"].unique().tolist()) <= {0, 1}
    assert item["track_delta"].shape == (8, 240, 320, 3)
    # unknown pixels: black in the RGB target, zero displacement target, not foreground
    unknown = ~item["track_foreground"][1:]
    assert (item["track_rgb"][1:][unknown] == 0).all()
    assert (item["track_delta"][unknown] == 0).all()
    assert item["head_depth_mm"].shape == (240, 320) and (item["head_depth_mm"] > 0).sum() > 1000
    assert (item["head_depth_mm"][~item["head_mask"]] == 0).all()
    assert item["action"].shape == (32, 80) and item["proprio"].shape == (1, 80)
    active = item["action"][:, SLOTS]
    assert active.abs().max() <= 1.05, active.abs().max()          # Alpha RoboDojo min-max; gripper exactly in [-1, 1]
    assert (item["action"][:, ~item["action_mask"][0]] == 0).all()
    assert item["embodiment"] == 1 and item["text_context"].shape[1] == 4096
    assert item["prompt"].startswith("A video recorded from a robot's point of view")
    # deterministic window for fixed_windows
    assert ds[0]["key"] == item["key"]
    batch = collate_raw([item, ds[5]])
    assert batch["video_frames"].shape[0] == 2 and batch["text_mask"].shape[0] == 2


@needs_data
def test_rgb_channel_order_table_is_warm():
    """RoboDojo's table top is red-brown: after the BGR fix the red channel must dominate blue on the head view."""
    ds = RDJEpisodeDataset(RDJWindowConfig(samples_per_episode=1, fixed_windows=True))
    head = ds[0]["head_rgb"].float()
    bottom = head[200:, :, :]   # table region at the bottom of the head view
    assert bottom[..., 0].mean() > bottom[..., 2].mean() + 20, bottom.mean(dim=(0, 1))


@needs_data
def test_instruction_index_covers_all_train_episodes():
    ds = RDJEpisodeDataset(RDJWindowConfig(samples_per_episode=1))
    index = load_official_instructions(RDJ_ROOT / "episode_instructions_official.jsonl")
    missing = [r for r in ds.rows if not index.get(f"{r['task']}/{r['variant']}/episode{r['episode']}")]
    assert not missing, missing[:3]
    val = RDJEpisodeDataset(replace(ds.config, split="val"))
    assert len(val.rows) == 102


def test_role_cross_entropy_ignores_unknown_cells():
    logits = torch.randn(1, 8, 8, 10, 3)
    target = torch.full((1, 8, 8, 10), -1, dtype=torch.int8)
    target[0, :, 2:4, 3:6] = 1
    ref = F.cross_entropy(logits[0, :, 2:4, 3:6].reshape(-1, 3), torch.ones(8 * 2 * 3, dtype=torch.long))
    got = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1).long(), ignore_index=-1)
    assert torch.allclose(ref, got)


def test_object_batch_does_not_label_robot_only_rollout_background():
    from metiswam4d.data.rt2.online_encoder import RT2OnlineEncoder, RT2EncoderConfig
    from metiswam4d.objectives import structured_track_loss, TrackRegionWeights
    encoder=object.__new__(RT2OnlineEncoder)
    encoder.config=RT2EncoderConfig(unknown_role=False)
    encoder.device=torch.device('cpu')
    encoder.depth_min,encoder.depth_max=.12,.70
    encoder.encode_pixels=lambda pixels: torch.zeros(pixels.shape[0],2,1 if pixels.shape[1]==1 else 3,16,20)
    roles=torch.zeros(2,9,240,320,dtype=torch.uint8)
    roles[:,:,40:90,40:90]=1
    roles[1,1:,120:160,120:160]=2
    raw=dict(video_frames=torch.zeros(2,9,384,320,3,dtype=torch.uint8),
             track_rgb=torch.zeros(2,9,240,320,3,dtype=torch.uint8),
             head_rgb=torch.zeros(2,240,320,3,dtype=torch.uint8),head_depth_mm=torch.zeros(2,240,320),
             head_mask=roles[:,0].bool(),track_foreground=roles.bool(),track_role_px=roles,
             track_delta=torch.zeros(2,8,240,320,3),track_unknown_role=torch.tensor([True,False]))
    encoded=encoder(raw)
    assert (encoded['track_role'][0]==-1).any() and not (encoded['track_role'][0]==0).any()
    assert (encoded['track_role'][1]==0).any() and (encoded['track_role'][1]==2).any()
    assert (encoded['track_role_frames'][0]==-1).any()
    assert (encoded['track_role_frames'][1]==0).any()
    pred=torch.ones(2,1,3,16,20);pred[0]*=100
    loss,_=structured_track_loss(pred,torch.zeros_like(pred),torch.ones(2,dtype=torch.bool),
                                 role=encoded['track_role'],valid=None,
                                 weights=TrackRegionWeights(body=0,object=0,background=1))
    assert loss.item()==pytest.approx(1.0)
