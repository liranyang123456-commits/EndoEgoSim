import numpy as np

from endosim.eval.align import align_trajectories
from endosim.eval.metrics import rpe
from endosim.geometry.se3 import rot_trans, so3_exp


def test_rpe_rotation_is_bounded_near_pi():
    gt = np.repeat(np.eye(4)[None], 2, axis=0)
    est = gt.copy()
    R = so3_exp(np.array([np.pi - 1e-7, 0.0, 0.0]))
    R[0, 1] += 1e-8  # emulate accumulated floating-point non-orthogonality
    est[1, :3, :3] = R

    result = rpe(est, gt)

    assert 179.99 < result["rot_deg_mean"] <= 180.0
    assert result["rot_deg_max"] <= 180.0


def test_rpe_known_translation_and_rotation():
    gt = np.repeat(np.eye(4)[None], 2, axis=0)
    est = gt.copy()
    est[1] = rot_trans(so3_exp(np.array([0.0, 0.0, np.pi / 2])),
                       np.array([3.0, 4.0, 0.0]))

    result = rpe(est, gt)

    assert np.isclose(result["trans_mm_mean"], 5.0)
    assert np.isclose(result["rot_deg_mean"], 90.0)


def test_sim3_alignment_keeps_rotation_blocks_in_so3():
    est = np.repeat(np.eye(4)[None], 3, axis=0)
    gt = est.copy()
    est[:, 0, 3] = [0.0, 1.0, 2.0]
    gt[:, 0, 3] = [0.0, 3.0, 6.0]

    aligned, (scale, _, _) = align_trajectories(est, gt, with_scale=True)

    assert np.isclose(scale, 3.0)
    assert np.allclose(
        [np.linalg.det(pose[:3, :3]) for pose in aligned],
        np.ones(3),
    )
