from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ur5e_real.hardware.rgbd import DualRgbdCamera


def sdk():
    rs = MagicMock()
    rs.stream.color, rs.stream.depth = "color", "depth"
    rs.format.bgr8, rs.format.z16 = "bgr8", "z16"
    rs.camera_info = SimpleNamespace(name="name", firmware_version="firmware", usb_type_descriptor="usb")
    pipes = [MagicMock(), MagicMock()]
    rs.pipeline.side_effect = pipes
    rs.align.return_value.process.side_effect = lambda frames: frames
    for pipe in pipes:
        profile = pipe.start.return_value
        profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        profile.get_device.return_value.get_info.return_value = "test"
        stream = profile.get_stream.return_value
        stream.fps.return_value, stream.format.return_value = 30, "test_format"
        stream.get_extrinsics_to.return_value = SimpleNamespace(rotation=[1.0] * 9, translation=[0.0] * 3)
        counter = iter(range(1, 30))

        def frame_set(_timeout, counter=counter):
            index = next(counter)
            frames = MagicMock()
            for kind, arr in (("color", np.zeros((4, 6, 3), np.uint8)), ("depth", np.ones((4, 6), np.uint16))):
                frame = getattr(frames, f"get_{kind}_frame").return_value
                frame.get_data.return_value = arr
                frame.get_frame_number.return_value = index
                frame.get_timestamp.return_value = index * 33.3
                frame.get_frame_timestamp_domain.return_value = "hardware_clock"
            return frames

        pipe.wait_for_frames.side_effect = frame_set
    return rs, pipes


def test_dual_rgbd_starts_both_streams_and_returns_owned_arrays():
    rs, pipes = sdk()
    with patch("ur5e_real.hardware.rgbd._rs_module", return_value=rs), \
            patch("ur5e_real.hardware.rgbd._intrinsics", return_value={"width": 6, "height": 4}):
        cam = DualRgbdCamera("h", "w", 6, 4, 30, 0)
        cam.start()
        pair = cam.read()
        assert pair.head_depth.dtype == np.uint16
        assert pair.head_depth.flags.owndata
        assert pair.metadata["head"]["depth"]["frame_number"] == 2
        assert cam.calibration["cameras"]["head"]["depth_scale_m_per_unit"] == 0.001
        cam.stop()
        for pipe in pipes:
            pipe.stop.assert_called_once()
        assert rs.config.return_value.enable_stream.call_count == 4


def test_second_camera_failure_closes_first_camera():
    rs, pipes = sdk()
    pipes[1].start.side_effect = RuntimeError("device busy")
    with patch("ur5e_real.hardware.rgbd._rs_module", return_value=rs), \
            patch("ur5e_real.hardware.rgbd._intrinsics", return_value={}):
        cam = DualRgbdCamera("h", "w", 6, 4, 30, 0)
        with pytest.raises(RuntimeError, match="device busy"):
            cam.start()
        pipes[0].stop.assert_called_once()
        assert not cam.pipelines


def test_repeated_sensor_frames_fail_instead_of_looking_fresh():
    rs, pipes = sdk()
    with patch("ur5e_real.hardware.rgbd._rs_module", return_value=rs), \
            patch("ur5e_real.hardware.rgbd._intrinsics", return_value={}):
        cam = DualRgbdCamera("h", "w", 6, 4, 30, 0)
        cam.start()
        old = pipes[0].wait_for_frames(2000)
        pipes[0].wait_for_frames.side_effect = None
        pipes[0].wait_for_frames.return_value = old
        cam.read()
        with pytest.raises(RuntimeError, match="repeated frame"):
            cam.read()
        cam.stop()
