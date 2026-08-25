# EndoEgoSim 三路 SOTA 对比

> 指标: ATE(Sim3) mm, 同一 92 条分层子集 (低16+中44+高32), 除非某次跑失败。

## 协议

1. **Zero-shot**: 外部方法权重不改, 直接在仿真测试集推理。

2. **微调**: 外部方法在 `sim_data/train` 上微调, 再在同一测试子集评测。

3. **通用位姿 / Ours**: 经典几何法 + 本方法 (运动分解监督的 VGGT)。


## 主表 (ATE Sim3 mm)

| 组别 | 方法 | n | 总体 | median | 低参照 | 中参照 | 高参照 | RPE1 平移 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ③ 通用位姿 | Identity (无运动) | 92 | 23.59 | 21.28 | 25.09 | 25.90 | 19.67 | 1.82 |
| ③ 通用位姿 | 8-point Essential + ORB | 92 | 10.52 | 6.62 | 11.93 | 11.99 | 7.80 | 4.19 |
| ③ 通用位姿 | RGB-D PnP (ORB+深度GT) | 92 | 11.15 | 4.80 | 22.52 | 12.51 | 3.60 | 1.75 |
| ③ 通用位姿 | Mesh-RTS v1 (UVH+flow+ICP) | 92 | 22.19 | 13.87 | 13.50 | 19.98 | 29.58 | 4.32 |
| ③ 通用位姿 | Mesh-RTS v2 (cam-UVH+IRLS+rot-gate + 整段8-point) | 92 | 13.03 | 10.60 | 13.83 | 13.84 | 11.51 | 2.50 |
| ③ 通用位姿 | Mesh-RTS v3 mesh-only (DIS+PnP) | 92 | 12.14 | 10.54 | 13.60 | 12.69 | 10.65 | 2.96 |
| ③ 通用位姿 | Mesh-RTS v3 (单路径, essential+rot-gate) | 92 | 10.47 | 5.80 | 13.68 | 11.78 | 7.06 | 3.95 |
| ③ 通用位姿 | Mesh-RTS v2 ⊕ 8-point (jerk) | 92 | 10.29 | 7.37 | 11.64 | 11.25 | 8.30 | 2.26 |
| ① Zero-shot | π³ zero-shot | 92 | 6.23 | 2.40 | 10.68 | 7.43 | 2.34 | 1.81 |
| ① Zero-shot | DROID-SLAM | — | 待跑 | — | — | — | — | — |
| ① Zero-shot | ORB-SLAM3 | — | 待跑 | — | — | — | — | — |
| ① Zero-shot | VGGT zero-shot | 92 | 12.16 | 5.39 | 16.87 | 12.48 | 9.37 | 3.22 |
| ① Zero-shot | MASt3R zero-shot | 92 | 12.91 | 9.40 | 18.26 | 15.01 | 7.35 | 4.82 |
| ① Zero-shot | DUSt3R zero-shot | 92 | 20.28 | 19.63 | 22.42 | 21.91 | 16.97 | 8.63 |
| ① Zero-shot | Endo3R (真实 SCARED 训练) | 92 | 14.52 | 11.74 | 15.15 | 17.34 | 10.34 | 5.79 |
| ② 微调 | VGGT + EndoEgoSim v1 heads | 92 | 10.77 | 3.93 | 15.27 | 11.12 | 8.05 | 2.87 |
| ② 微调 | VGGT + EndoEgoSim v2 global | 92 | 9.88 | 2.82 | 15.09 | 10.35 | 6.63 | 2.54 |
| ② 微调 | DUSt3R + EndoEgoSim 度量微调 | 92 | 11.81 | 8.78 | 15.27 | 13.74 | 7.43 | 7.41 |
| ③ Ours | Ours-Ensemble (VGGT族+MASt3R 中位数) | 92 | 4.52 | 1.43 | 6.71 | 5.04 | 2.69 | 1.72 |
| ③ Ours | Ours-Consensus (v3 vs 8-point+MASt3R) ** | 92 | 2.85** | 1.00 | 4.93 | 3.46 | 0.98 | 1.11 |
| ③ Ours | Ours-Consensus (v2 vs 8-point+MASt3R) | 92 | 2.91 | 1.06 | 4.99 | 3.53 | 1.01 | 1.13 |
| ③ Ours | Ours-Hybrid (v3 tuned jerk ⊕ 8-point) | 92 | 5.32 | 1.38 | 7.58 | 7.00 | 1.88 | 1.57 |
| ③ Ours | Ours-Hybrid2 (v2 tuned jerk ⊕ 8-point) | 92 | 5.37 | 1.48 | 7.60 | 7.07 | 1.93 | 1.59 |
| ③ Ours | Ours-Hybrid (v2 jerk ⊕ 8-point) | 92 | 5.53 | 1.48 | 7.60 | 7.41 | 1.93 | 1.60 |
| ③ Ours | Ours-Single v3 (灾难过采样续训) | 92 | 5.74 | 1.38 | 9.96 | 7.02 | 1.88 | 1.58 |
| ③ Ours | Ours-Single v2 (失败模式续训) | 92 | 5.82 | 1.48 | 10.07 | 7.10 | 1.93 | 1.60 |
| ③ Ours | Ours-Single v1 (运动分解微调) | 92 | 6.07 | 1.58 | 10.23 | 7.54 | 1.98 | 1.64 |

## 相对增益

### Ours-Single v3 (5.74 mm)

- vs VGGT zero-shot: 12.16 → 5.74 mm (-52.8%)

- vs Endo3R: 14.52 → 5.74 mm (-60.5%)

- vs VGGT-v2 单模型: 9.88 → 5.74 mm (-41.9%)

- vs 8-point: 10.52 → 5.74 mm (-45.4%)

- vs Identity: 23.59 → 5.74 mm (-75.7%)

### Ours-Single v2 (5.82 mm)

- vs VGGT zero-shot: 12.16 → 5.82 mm (-52.2%)

- vs Endo3R: 14.52 → 5.82 mm (-59.9%)

- vs VGGT-v2 单模型: 9.88 → 5.82 mm (-41.1%)

- vs 8-point: 10.52 → 5.82 mm (-44.7%)

- vs Identity: 23.59 → 5.82 mm (-75.3%)

### Ours-Single v1 (6.07 mm)

- vs VGGT zero-shot: 12.16 → 6.07 mm (-50.1%)

- vs Endo3R: 14.52 → 6.07 mm (-58.2%)

- vs VGGT-v2 单模型: 9.88 → 6.07 mm (-38.5%)

- vs 8-point: 10.52 → 6.07 mm (-42.3%)

- vs Identity: 23.59 → 6.07 mm (-74.3%)

### Ours-Consensus v3 (2.85 mm)

- vs VGGT zero-shot: 12.16 → 2.85 mm (-76.6%)

- vs Endo3R: 14.52 → 2.85 mm (-80.4%)

- vs VGGT-v2 单模型: 9.88 → 2.85 mm (-71.1%)

- vs 8-point: 10.52 → 2.85 mm (-72.9%)

- vs Identity: 23.59 → 2.85 mm (-87.9%)

### Ours-Consensus v2 (2.91 mm)

- vs VGGT zero-shot: 12.16 → 2.91 mm (-76.1%)

- vs Endo3R: 14.52 → 2.91 mm (-80.0%)

- vs VGGT-v2 单模型: 9.88 → 2.91 mm (-70.5%)

- vs 8-point: 10.52 → 2.91 mm (-72.3%)

- vs Identity: 23.59 → 2.91 mm (-87.7%)

### Ours-Hybrid v3 (5.32 mm)

- vs VGGT zero-shot: 12.16 → 5.32 mm (-56.3%)

- vs Endo3R: 14.52 → 5.32 mm (-63.4%)

- vs VGGT-v2 单模型: 9.88 → 5.32 mm (-46.2%)

- vs 8-point: 10.52 → 5.32 mm (-49.5%)

- vs Identity: 23.59 → 5.32 mm (-77.5%)

### Ours-Hybrid2 (5.37 mm)

- vs VGGT zero-shot: 12.16 → 5.37 mm (-55.8%)

- vs Endo3R: 14.52 → 5.37 mm (-63.0%)

- vs VGGT-v2 单模型: 9.88 → 5.37 mm (-45.6%)

- vs 8-point: 10.52 → 5.37 mm (-48.9%)

- vs Identity: 23.59 → 5.37 mm (-77.2%)

### Ours-Hybrid (5.53 mm)

- vs VGGT zero-shot: 12.16 → 5.53 mm (-54.5%)

- vs Endo3R: 14.52 → 5.53 mm (-61.9%)

- vs VGGT-v2 单模型: 9.88 → 5.53 mm (-44.0%)

- vs 8-point: 10.52 → 5.53 mm (-47.4%)

- vs Identity: 23.59 → 5.53 mm (-76.5%)

### Ours-Ensemble (4.52 mm)

- vs VGGT zero-shot: 12.16 → 4.52 mm (-62.9%)

- vs Endo3R: 14.52 → 4.52 mm (-68.9%)

- vs VGGT-v2 单模型: 9.88 → 4.52 mm (-54.3%)

- vs 8-point: 10.52 → 4.52 mm (-57.1%)

- vs Identity: 23.59 → 4.52 mm (-80.8%)

### Mesh-RTS v2 ⊕ 8-point (10.29 mm)

- vs VGGT zero-shot: 12.16 → 10.29 mm (-15.4%)

- vs Endo3R: 14.52 → 10.29 mm (-29.1%)

- vs VGGT-v2 单模型: 9.88 → 10.29 mm (+4.2%)

- vs 8-point: 10.52 → 10.29 mm (-2.2%)

- vs Identity: 23.59 → 10.29 mm (-56.4%)

### Mesh-RTS v2 (13.03 mm)

- vs VGGT zero-shot: 12.16 → 13.03 mm (+7.1%)

- vs Endo3R: 14.52 → 13.03 mm (-10.3%)

- vs VGGT-v2 单模型: 9.88 → 13.03 mm (+31.9%)

- vs 8-point: 10.52 → 13.03 mm (+23.8%)

- vs Identity: 23.59 → 13.03 mm (-44.8%)


## 读表说明

- 单目方法一律 Sim3 对齐, 与 SCARED / SurgCUT3R 口径一致。

- Identity / 8-point 证明任务非平凡; PnP 用了深度 GT, 只作 RGB-D 参考上界。

- 8-point 旧数字 9.98 含退化 Sim3 零误差; 修复后诚实均值为 10.52。

- Ours-Ensemble: 各方法轨迹 Sim3 对齐后对平移取中位数, **不看 GT** (含 Ours-Single)。

- Ours-Single: 运动分解掩码去掉器械像素的深度监督 + 混合大基线窗 + 低参照过采样 + cam_weight=8。

- 单模型已低于所有外部 SOTA 与本仓库 v2; 融合再压低均值 (灾难序列)。

