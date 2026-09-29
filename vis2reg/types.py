"""单样本张量约定；零填充点必须通过 valid 排除，不参与几何运算。"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class PointCloud:
    points: Tensor  # [N, 3]，保持物理尺度
    valid: Tensor  # [N]，bool；不能用 xyz != 0 判断有效性

    @property
    def xyz(self) -> Tensor:
        return self.points[self.valid]


@dataclass
class GeometryInput:
    source: PointCloud  # P，术前源点云
    target: PointCloud  # Q，已合并到统一参考坐标系的术中点云


@dataclass
class SimilarityTransform:
    """列向量公式 sRx+t；代码采用每行一个点。s 的估计方式待明确。"""

    rotation: Tensor  # [3, 3]
    translation: Tensor  # [3]
    scale: Tensor  # scalar

    def apply(self, points: Tensor) -> Tensor:
        return self.scale * (points @ self.rotation.T) + self.translation


@dataclass
class ViewObservation:
    target: PointCloud  # Q_v，在该视角的相机坐标系
    intrinsics: Tensor  # K_v，[3, 3]
    mask: Tensor  # M_v，[H, W]，二值
    # 工程接口：论文没有完整交代各视角外参。必须由适配器显式提供；
    # 只有确知已在同一坐标系时才能传单位阵，不能默认假定相机相同。
    reference_to_camera: Tensor  # [4, 4]，参考系 -> 当前相机系


@dataclass
class TrainingSample:
    geometry: GeometryInput
    views: tuple[ViewObservation, ...]
    patient_id: str
    gt_pose: SimilarityTransform | None = None  # 仅合成数据提供


@dataclass
class RegistrationOutput:
    warped: Tensor  # [N_valid, 3]，W_hat，在统一参考系
    displacements: Tensor  # [N_valid, 3]，Delta，在源坐标系
    pose: SimilarityTransform
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class RenderedView:
    silhouette: Tensor  # [H, W]，可微软轮廓，用于 BCE + Dice
    depth: Tensor  # [H, W]，相机 z 深度，空像素为 0
