"""EndoEgoSim 配置体系（dataclass + JSON 序列化 + 全局种子管理）。"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Literal


def sample_int(rng, lo_hi) -> int:
    """闭区间整数采样。numpy Generator.integers 的 high 为开区间, 这里对配置元组做闭区间。"""
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return int(rng.integers(lo, hi + 1))


def sample_uniform(rng, lo_hi) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 512
    intrinsics: Literal["scared", "fov"] = "scared"
    fov_deg: float = 90.0          # intrinsics="fov" 时使用
    jitter: float = 0.02           # 内参随机化幅度


@dataclass
class TissueConfig:
    kind: str = "mixed"                 # tunnel | organ | mixed
    organ_prob: float = 0.4             # mixed时使用真实器官几何的概率
    organ_decimate: int = 40000         # 器官网格简化面数
    organ_tex_scale_mm: tuple = (35.0, 90.0)  # 三平面纹理: 每次重复覆盖的mm数
    length_mm: tuple = (350.0, 650.0)
    radius_base_mm: tuple = (28.0, 48.0)
    fold_amplitude: tuple = (0.10, 0.28)
    fold_count: tuple = (4, 9)
    n_rings: int = 140
    n_sector: int = 56
    cap_end: bool = True


@dataclass
class TrajectoryConfig:
    # insertion 加权: 进镜是内窥镜主运动; retrace=进后退回(回环); keyframe 见 configs/keyframe.json
    motion_types: tuple = ("insertion", "insertion", "insertion", "orbital", "free", "retrace")
    n_frames: tuple = (40, 90)
    step_mm: tuple = (0.8, 5.0)
    rot_deg: tuple = (0.5, 4.0)
    tremor_prob: float = 0.35
    tremor_mm: tuple = (0.05, 0.4)
    tremor_deg: tuple = (0.05, 0.4)
    keyframe_hop_mm: tuple = (25.0, 160.0)  # keyframe 轨迹相邻关键帧平移 (对齐 SCARED 稀疏关键帧)


@dataclass
class ObjectConfig:
    n_objects_range: tuple = (0, 2)
    use_bop_prob: float = 0.5      # 用 BOP 网格 vs 程序化器械
    bop_scale: tuple = (0.15, 0.55)
    motion_models: tuple = ("insertion", "manipulation", "free", "static")
    marker_prob: float = 0.35      # 贴壁棋盘格标记（显式参照物）出现概率


@dataclass
class DeformationConfig:
    """内窥镜现实: 形变是常态（默认85%序列含形变）。"""
    prob: float = 0.85
    strength: tuple = (0.15, 1.0)


@dataclass
class AppearanceConfig:
    texture_source: Literal["real", "procedural"] = "real"
    real_texture_prob: float = 0.85
    hsv_jitter: float = 0.06
    tissue_material_jitter: float = 0.3
    # 外观级 sim-to-real: 序列级 Reinhard 色彩迁移(同一参照图, 保时序一致)
    color_transfer_prob: float = 0.45
    color_transfer_strength: tuple = (0.25, 0.70)
    # 内窥镜光学: 圆形渐晕 + 轻度桶形畸变(仅作用于 RGB, 深度/位姿仍为针孔 GT)
    vignette_prob: float = 0.80
    vignette_strength: tuple = (0.15, 0.45)
    barrel_prob: float = 0.30
    barrel_k1: tuple = (-0.12, -0.02)


@dataclass
class SensorConfig:
    shot_noise: tuple = (0.2, 0.8)
    read_noise: tuple = (0.002, 0.006)
    blur_px: tuple = (0.0, 1.5)
    exposure_jitter: tuple = (0.05, 0.2)
    wb_jitter: float = 0.04
    haze_prob: float = 0.15
    haze_strength: tuple = (0.05, 0.25)


@dataclass
class OutputConfig:
    save_color: bool = True
    save_depth: bool = True
    save_mask: bool = True
    save_object_poses: bool = True
    save_motion_gt: bool = True
    save_flow: bool = True            # 稠密光流(压缩npz, ~0.5MB/帧)
    save_motion_mask: bool = True     # 运动分解掩码(png, 几KB/帧)
    save_normals: bool = False        # 法线图(png16, 相机系, 0=无效像素)
    stereo_baseline_mm: float = 0.0   # >0 时渲染右目(沿相机+X基线), 输出 color_right/depth_right
    depth_format: Literal["png16", "tiff"] = "png16"


@dataclass
class GenConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tissue: TissueConfig = field(default_factory=TissueConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    fps: float = 10.0
    near_mm: float = 2.0
    motion_eps_mm: float = 0.2        # 静态参照判定的场景运动阈值(mm)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenConfig":
        cfg = cls()
        for sec_name in ("camera", "tissue", "trajectory", "objects",
                         "deformation", "appearance", "sensor", "output"):
            if sec_name in d:
                sec = getattr(cfg, sec_name)
                for k, v in d[sec_name].items():
                    if hasattr(sec, k):
                        setattr(sec, k, v)
        for k in ("fps", "near_mm", "motion_eps_mm"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "GenConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
