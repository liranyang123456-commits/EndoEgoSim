import cv2
import numpy as np

from scripts import eval_meshrts


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_mesh_pnp_returns_current_to_previous_edge():
    rng = np.random.default_rng(4)
    current_points = rng.uniform([-0.7, -0.5, 3.0], [0.7, 0.5, 6.0], (80, 3))
    expected = np.eye(4)
    expected[:3, :3] = _rotation_y(0.08)
    expected[:3, 3] = [0.12, -0.04, 0.18]

    previous_points = (
        expected[:3, :3] @ current_points.T
    ).T + expected[:3, 3]
    K = np.array([[520.0, 0.0, 320.0], [0.0, 515.0, 256.0], [0.0, 0.0, 1.0]])
    previous_pixels = np.column_stack(
        [
            K[0, 0] * previous_points[:, 0] / previous_points[:, 2] + K[0, 2],
            K[1, 1] * previous_points[:, 1] / previous_points[:, 2] + K[1, 2],
        ]
    )

    estimated = eval_meshrts.mesh_pnp(current_points, previous_pixels, K)

    assert estimated is not None
    assert np.allclose(estimated[:3, :3], expected[:3, :3], atol=2e-3)
    assert np.allclose(estimated[:3, 3], expected[:3, 3], atol=2e-3)


def test_essential_pose_is_inverted_for_c2w_composition(monkeypatch):
    forward_R = _rotation_y(0.2)
    forward_t = np.array([[0.3], [0.1], [0.9]])
    points = np.zeros((12, 2), dtype=np.float32)

    monkeypatch.setattr(eval_meshrts, "_orb_match", lambda *_: (points, points))
    monkeypatch.setattr(
        cv2,
        "findEssentialMat",
        lambda *_args, **_kwargs: (np.eye(3), np.ones((12, 1), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        cv2,
        "recoverPose",
        lambda *_args, **_kwargs: (12, forward_R, forward_t, np.ones((12, 1))),
    )

    estimated = eval_meshrts.pairwise_essential(
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20), dtype=np.uint8),
        np.eye(3),
    )
    forward = np.eye(4)
    forward[:3, :3] = forward_R
    forward[:3, 3] = forward_t[:, 0]

    assert np.allclose(estimated, np.linalg.inv(forward))
