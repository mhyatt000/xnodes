from __future__ import annotations

import numpy as np
import pytest

from xnodes.core.model_client import (
    build_policy_payload,
    extract_action_targets,
    extract_camera_images,
    GaussianConv,
    missing_camera_images,
)


def _sample_images(size: int = 8) -> dict[str, np.ndarray]:
    return {
        "low": np.ones((size, size, 3), dtype=np.uint8),
        "side": np.ones((size, size, 3), dtype=np.uint8) * 2,
        "wrist": np.ones((size, size, 3), dtype=np.uint8) * 3,
    }


def test_build_policy_payload_shapes_and_units():
    joints = np.arange(7, dtype=np.float32)
    pose_mm = np.array([1000, 2000, 3000, 0.1, 0.2, 0.3], dtype=np.float32)
    gripper = np.array([0.4], dtype=np.float32)

    payload = build_policy_payload(
        joints=joints,
        pose=pose_mm,
        gripper=gripper,
        images=_sample_images(),
        ensemble=True,
    )

    obs = payload["observation"]
    assert payload["ensemble"] is True
    assert obs["image_primary"].shape == (1, 1, 8, 8, 3)
    assert obs["image_side"].shape == (1, 1, 8, 8, 3)
    assert obs["image_left_wrist"].shape == (1, 1, 8, 8, 3)
    assert obs["proprio_single_arm"].shape == (1, 1, 14)
    np.testing.assert_allclose(obs["proprio_single_arm"][0, 0, :3], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_build_policy_payload_validates_gripper():
    with pytest.raises(ValueError, match="Gripper out of bounds"):
        build_policy_payload(
            joints=np.zeros(7, dtype=np.float32),
            pose=np.zeros(6, dtype=np.float32),
            gripper=np.array([1.2], dtype=np.float32),
            images=_sample_images(),
        )


def test_extract_camera_images_and_missing():
    data = {
        "/xgym/camera/low": np.zeros((4, 4, 3), dtype=np.uint8),
        "/xgym/camera/wrist": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    images = extract_camera_images(data)
    assert set(images) == {"low", "side", "wrist"}
    assert missing_camera_images(images) == ["side"]


def test_extract_action_targets_resolution_and_errors():
    actions = {"action": np.arange(24, dtype=np.float32).reshape(6, 4)}
    targets = extract_action_targets(actions, resolution=2)
    assert targets.shape == (3, 4)
    np.testing.assert_allclose(targets, actions["action"][::2])

    with pytest.raises(ValueError, match="resolution must be >= 1"):
        extract_action_targets(actions, resolution=0)
    with pytest.raises(KeyError, match="Missing action key"):
        extract_action_targets({}, resolution=1)


def test_gaussian_conv_rejects_even_kernel():
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        GaussianConv(kernel_size=4)


def test_gaussian_conv_batch_smoothing_matches_expected():
    conv = GaussianConv(kernel_size=3, std=1.0)
    arr = np.array([0.0, 10.0, 0.0], dtype=np.float64)

    out = conv(arr)

    a = float(np.exp(-0.5))
    expected = np.array(
        [
            (10.0 * a) / (1.0 + a),
            10.0 / (1.0 + 2.0 * a),
            (10.0 * a) / (1.0 + a),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-7)


def test_gaussian_conv_scalar_stream_matches_batch_prefix():
    seq = [1.0, 2.0, 3.0, 4.0]
    conv = GaussianConv(kernel_size=5, std=1.0)
    streamed = [conv(x) for x in seq]

    expected = []
    for i in range(1, len(seq) + 1):
        prefix = np.array(seq[:i], dtype=np.float64)
        batch_out = np.atleast_1d(GaussianConv(kernel_size=5, std=1.0)(prefix))
        expected.append(float(batch_out[-1]))

    np.testing.assert_allclose(streamed, expected, rtol=1e-7, atol=1e-7)
