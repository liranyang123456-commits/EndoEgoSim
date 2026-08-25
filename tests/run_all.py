"""EndoEgoSim 总冒烟测试: 核心模块正确性 + 端到端生成 + GT闭环 + 评测。

运行: python tests/run_all.py   （约3-5分钟）
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np


def test_se3():
    from endosim.geometry.se3 import (se3_exp, se3_log, relative, inverse,
                                      quat_from_R, R_from_quat)
    rng = np.random.default_rng(0)
    for _ in range(100):
        xi = rng.normal(0, 0.5, 6)
        T = se3_exp(xi)
        assert np.allclose(se3_log(T), xi, atol=1e-8), "se3对数映射闭环失败"
        assert np.allclose(inverse(T) @ T, np.eye(4), atol=1e-9)
        Ta, Tb = se3_exp(rng.normal(0, .3, 6)), se3_exp(rng.normal(0, .3, 6))
        assert np.allclose(relative(Ta, Tb), inverse(Ta) @ Tb, atol=1e-9)
        q = quat_from_R(Ta[:3, :3])
        assert np.allclose(R_from_quat(q), Ta[:3, :3], atol=1e-9)
    print("✅ SE(3) 工具")


def test_camera():
    from endosim.render.camera import PinholeCamera
    cam = PinholeCamera.scared_like(640, 512)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-50, 50, (100, 3)); pts[:, 2] += 100
    uv, z = cam.project(pts)
    back = cam.unproject(uv, z)
    assert np.allclose(back, pts, atol=1e-6), "投影/反投影闭环失败"
    print("✅ 相机模型")


def test_alignment_metrics():
    from endosim.eval.metrics import evaluate_trajectory
    from endosim.eval.align import umeyama_alignment
    rng = np.random.default_rng(2)
    gt = np.array([se3_exp_c(np.array([0, 0, 0, 0, 0, i * 1.0])) for i in range(30)])
    # identity
    res = evaluate_trajectory(gt.copy(), gt)
    assert res["ate_se3"]["rmse"] < 1e-9
    assert res["rpe_1"]["trans_mm_mean"] < 1e-9
    # 相似变换后的轨迹: Sim3 对齐应零误差, SE3 对齐应有残差
    s_true = 2.5
    est = gt.copy()
    est[:, :3, 3] *= s_true
    s, R, t = umeyama_alignment(est[:, :3, 3], gt[:, :3, 3], with_scale=True)
    assert abs(s - 1 / s_true) < 1e-6, f"Umeyama尺度估计错误: {s} vs {1/s_true}"
    # 退化轨迹 (全点重合): 尺度必须有限, ATE = GT 绕质心的 RMSE
    ident = np.repeat(np.eye(4)[None], len(gt), axis=0)
    res_i = evaluate_trajectory(ident, gt)
    assert np.isfinite(res_i["ate_sim3"]["rmse"])
    assert res_i["ate_sim3"]["scale"] == 1.0
    print("✅ Umeyama对齐与指标")


def se3_exp_c(xi):
    from endosim.geometry.se3 import se3_exp
    return se3_exp(xi)


def test_generator_e2e():
    from endosim.config import GenConfig
    from endosim.dataset.generator import generate_sequence
    from endosim.geometry.se3 import relative
    cfg = GenConfig()
    cfg.trajectory.n_frames = (12, 16)
    cfg.deformation.prob = 0.0  # 刚性场景做几何闭环检验
    seq = generate_sequence("smoke", seed=2024, cfg=cfg, bank=None, split="train")
    # 1. 首帧单位
    assert np.allclose(seq.poses_wc[0], np.eye(4), atol=1e-9)
    # 2. 深度与mask尺寸一致
    f = seq.frames[0]
    assert f["depth"].shape == (512, 640)
    assert f["instance"].shape == (512, 640)
    assert np.isfinite(f["depth"]).mean() > 0.5
    # 3. GT深度与GT位姿几何自洽(内存级闭环, 无量化)
    cam = seq.camera
    i, j = 0, min(5, seq.n_frames - 1)
    di = seq.frames[i]["depth"].copy()
    valid = np.isfinite(di)
    ys, xs = np.nonzero(valid)
    sel = np.random.default_rng(0).choice(len(ys), 5000, replace=False)
    pc = cam.unproject(np.stack([xs[sel], ys[sel]], 1), di[ys[sel], xs[sel]])
    # 相机i系 -> 世界 -> 相机j系: T_j^-1 @ T_i
    T_ij = np.linalg.inv(seq.poses_wc[j]) @ seq.poses_wc[i]
    pc_j = pc @ T_ij[:3, :3].T + T_ij[:3, 3]
    uv_j, z_j = cam.project(pc_j)
    dj = seq.frames[j]["depth"]
    xi = np.clip(np.round(uv_j[:, 0]).astype(int), 0, 639)
    yi = np.clip(np.round(uv_j[:, 1]).astype(int), 0, 511)
    inb = ((uv_j[:, 0] >= 0) & (uv_j[:, 0] < 640)
           & (uv_j[:, 1] >= 0) & (uv_j[:, 1] < 512))
    dz = np.abs(z_j[inb] - dj[yi[inb], xi[inb]])
    same = dz < 3.0
    assert same.sum() > 500, "闭环重投影命中过少"
    # 组织刚性时残差应极小(亚像素量化级)
    if seq.meta["deformation"] is None:
        assert np.median(dz[same]) < 2.5, f"刚性场景闭环深度残差过大: {np.median(dz[same])}"
    # 4. 物体GT: T_co = T_wc^-1 @ T_wo
    for obj in seq.objects:
        T_co = np.linalg.inv(seq.poses_wc[3]) @ obj["poses_wo"][3]
        assert np.allclose(T_co, obj["poses_co"][3], atol=1e-9), "物体相对位姿真值不一致"
    print("✅ 端到端生成 + GT闭环自洽")


def test_motion_gt():
    """运动分解GT + 光流GT 验证:
    1. 静态参照像素的流 == 独立深度+位姿投影法计算的流
    2. 无形变时参照比例应为1.0
    3. 强形变时参照比例应下降
    """
    from endosim.config import GenConfig
    from endosim.dataset.generator import generate_sequence
    cfg = GenConfig()
    cfg.trajectory.n_frames = (14, 18)

    # 刚性+无物体场景: 全部组织应为静态参照
    cfg_r = GenConfig()
    cfg_r.trajectory.n_frames = (14, 18)
    cfg_r.deformation.prob = 0.0
    cfg_r.objects.n_objects_range = (0, 0)
    cfg_r.objects.marker_prob = 0.0
    seq = generate_sequence("rigid", seed=31, cfg=cfg_r, bank=None, split="train")
    refs = [m["ref_frac"] for m in seq.motion]
    assert np.mean(refs) > 0.99, f"刚性无物体场景参照比例应≈1, got {np.mean(refs)}"

    # 光流 == 独立法(深度反投影+位姿变换) on 刚性场景(精确静态)
    t = len(seq.poses_wc) // 2
    P = seq.poses_wc
    cam = seq.camera
    f0 = seq.frames[t - 1]
    m = seq.motion[t - 1]
    mask = m["mask"]
    d0 = f0["depth"]
    ys, xs = np.nonzero(np.isfinite(d0) & (mask == 1))
    pc = cam.unproject(np.stack([xs + 0.5, ys + 0.5], 1), d0[ys, xs])
    pc_w = pc @ P[t - 1][:3, :3].T + P[t - 1][:3, 3]
    T_cw = np.linalg.inv(P[t])
    pc_c = pc_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv, _ = cam.project(pc_c)
    flow_ind = uv - np.stack([xs + 0.5, ys + 0.5], 1)
    flow_gt = m["flow"][ys, xs]
    good = np.isfinite(flow_gt[:, 0])
    assert good.sum() > 100
    err = np.linalg.norm(flow_ind[good] - flow_gt[good], axis=1)
    assert err.mean() < 0.5, f"光流独立法验证失败: mean err {err.mean()}"
    # 强形变场景: 参照比例应显著下降
    cfg_d = GenConfig()
    cfg_d.trajectory.n_frames = (14, 18)
    cfg_d.deformation.prob = 1.0
    cfg_d.deformation.strength = (0.9, 1.0)
    seq = generate_sequence("deform", seed=31, cfg=cfg_d, bank=None, split="train")
    refs = [m["ref_frac"] for m in seq.motion]
    assert np.mean(refs) < 0.99, "强形变场景参照比例应<1"
    # 运动物体像素的分解标签
    print(f"✅ 运动分解GT (刚性参照比例{np.mean(refs):.2f}, "
          f"流验证误差 {err.mean():.3f}px)")


def test_sample_int_inclusive():
    from endosim.config import sample_int
    rng = np.random.default_rng(0)
    vals = {sample_int(rng, (0, 2)) for _ in range(400)}
    assert vals == {0, 1, 2}, f"闭区间采样应覆盖 0/1/2, got {vals}"
    assert sample_int(np.random.default_rng(1), (0, 0)) == 0
    print("✅ 整数闭区间采样")


def test_keyframe_retrace_traj():
    from endosim.scene.tissue import TissueTunnel
    from endosim.geometry.trajectories import generate_trajectory, motion_stats
    rng = np.random.default_rng(7)
    tunnel = TissueTunnel(rng, length=600, radius_base=32, fold_amplitude=0.12,
                          fold_count=5, n_rings=90, n_sector=36, cap_end=True)
    poses = generate_trajectory(rng, tunnel, n_frames=12, motion_type="keyframe",
                                step_mm=2.0, rot_deg=2.0, normalize_first=True,
                                keyframe_hop_mm=(20.0, 60.0))
    assert poses.shape == (12, 4, 4)
    st = motion_stats(poses)
    assert st["rot_deg_max"] <= 15.01, f"关键帧旋转超限: {st['rot_deg_max']}"
    assert st["step_mm_mean"] > 8.0, f"关键帧步长应明显大于密采样: {st['step_mm_mean']}"
    poses_r = generate_trajectory(rng, tunnel, n_frames=20, motion_type="retrace",
                                  step_mm=2.5, rot_deg=1.5, normalize_first=True)
    assert poses_r.shape == (20, 4, 4)
    # 回撤: 末段应朝起点折返 (路径总长 > 净位移)
    net = np.linalg.norm(poses_r[-1, :3, 3] - poses_r[0, :3, 3])
    assert st["path_length_mm"] > 0
    assert motion_stats(poses_r)["path_length_mm"] > net + 5.0
    print(f"✅ keyframe/retrace 轨迹 (hop均值 {st['step_mm_mean']:.1f}mm)")


def test_eval_protocol():
    from endosim.eval.protocol import (chain_window_poses, select_frame_indices,
                                       sliding_windows)
    gt = np.array([se3_exp_c(np.array([0, 0, 0, 0, 0, i * 2.0])) for i in range(40)])
    idx_u = select_frame_indices(gt, "uniform", max_frames=8)
    assert len(idx_u) == 8
    idx_c = select_frame_indices(gt, "consecutive", max_frames=10)
    assert list(idx_c) == list(range(10))
    idx_a = select_frame_indices(gt, "adaptive", max_frames=16, max_step_mm=10.0)
    hops = np.linalg.norm(gt[idx_a][1:, :3, 3] - gt[idx_a][:-1, :3, 3], axis=1)
    assert hops.max() <= 10.0 + 1e-6, f"adaptive hop 超限: {hops.max()}"
    wins = sliding_windows(20, 8, 4)
    assert int(wins[0][0]) == 0 and int(wins[-1][-1]) == 19
    packed = [(w, gt[w]) for w in wins]
    est, used = chain_window_poses(packed, with_scale=False)
    assert len(used) == 20
    err = np.linalg.norm(est[:, :3, 3] - gt[used][:, :3, 3], axis=1)
    assert err.max() < 1e-6, "同一轨迹滑窗拼接应近零误差"
    print("✅ 评测选帧/滑窗拼接协议")


def test_scared_left_view():
    from endosim.dataset.scared_full_converter import scared_left_view
    stacked = np.zeros((2048, 1280, 3), np.uint8)
    stacked[:1024] = 10
    stacked[1024:] = 200
    left = scared_left_view(stacked)
    assert left.shape == (1024, 1280, 3)
    assert left.mean() < 20
    flat = np.ones((512, 640, 3), np.uint8)
    assert scared_left_view(flat).shape == (512, 640, 3)
    print("✅ SCARED 左目裁切")


def test_appearance_optics():
    from endosim.render.appearance import circular_vignette, reinhard_color_transfer
    from endosim.render.sensor import directional_blur
    rng = np.random.default_rng(0)
    src = rng.random((32, 40, 3)).astype(np.float32)
    ref = rng.random((48, 64, 3)).astype(np.float32) * 0.25 + 0.35
    out = reinhard_color_transfer(src, ref, 0.8)
    assert out.shape == src.shape and 0.0 <= out.min() and out.max() <= 1.0
    v = circular_vignette(np.ones((32, 40, 3), np.float32), 0.5)
    assert v[0, 0].mean() < v[16, 20].mean()
    b = directional_blur(src, 5.0, np.array([1.0, 0.0]))
    assert b.shape == src.shape
    print("✅ 外观迁移/渐晕/方向模糊")


def test_n_objects_inclusive_e2e():
    from endosim.config import GenConfig
    from endosim.dataset.generator import generate_sequence
    cfg = GenConfig()
    cfg.trajectory.n_frames = (10, 12)
    cfg.deformation.prob = 0.0
    cfg.objects.n_objects_range = (2, 2)
    cfg.objects.marker_prob = 0.0
    cfg.objects.use_bop_prob = 0.0
    cfg.appearance.color_transfer_prob = 0.0
    cfg.appearance.vignette_prob = 0.0
    cfg.appearance.barrel_prob = 0.0
    seq = generate_sequence("nobj", seed=11, cfg=cfg, bank=None, split="train")
    assert seq.meta["n_objects"] == 2, f"闭区间 (2,2) 应生成 2 个物体, got {seq.meta['n_objects']}"
    print("✅ 物体数量闭区间端到端")


def main():
    test_se3()
    test_camera()
    test_alignment_metrics()
    test_sample_int_inclusive()
    test_keyframe_retrace_traj()
    test_eval_protocol()
    test_scared_left_view()
    test_appearance_optics()
    test_generator_e2e()
    test_motion_gt()
    test_n_objects_inclusive_e2e()
    print("\n🎉 全部冒烟测试通过")


if __name__ == "__main__":
    main()
