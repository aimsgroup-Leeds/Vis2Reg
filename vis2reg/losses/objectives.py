"""论文损失结构；距离计算、邻域搜索等算法主体有意留空。"""

import torch
from torch import Tensor, nn

from ..config import LossConfig
from ..rendering import (
    DifferentiablePointRasterizer,
    reference_to_camera,
    visible_domain,
)
from ..types import RegistrationOutput, TrainingSample


def one_way_chamfer(observed: Tensor, model: Tensor) -> Tensor:
    """CD→(A,B)=mean_a min_b ||a-b||²；A 是观测，B 是模型。

    输出为平方距离；论文表格将 CD 标为 mm，但公式为平方距离，
    此接口遵循训练公式，不擅自加入平方根。空集合策略待明确。
    """
    raise NotImplementedError("待实现平方单向 Chamfer 距离及空集策略。")


def symmetric_chamfer(first: Tensor, second: Tensor) -> Tensor:
    """CD(A,B)=CD→(A,B)+CD→(B,A)，两项相加而非取平均。"""
    return one_way_chamfer(first, second) + one_way_chamfer(second, first)


def silhouette_loss(predicted: Tensor, target: Tensor, config: LossConfig) -> Tensor:
    """BCE(S,M)+DiceLoss(S,M)，两项默认权重均为 1。

    BCE reduction、Dice 平滑常数和空 mask 规则未报告，待实现时明确。
    """
    raise NotImplementedError("待实现概率轮廓的 BCE 与 Dice 损失。")


def source_neighbors(source: Tensor, k: int | None) -> Tensor:
    """从有效源点 P 构造 N(i)，返回 [N,k] 邻居索引。

    k 未报告；是否排除自身及不足 k 个点时的策略也待明确。
    所有局部正则共享该源点邻域，点顺序与位移、warped 严格对应。
    """
    raise NotImplementedError("待指定 k 并实现源点 kNN 邻域。")


def deformation_magnitude(displacements: Tensor) -> Tensor:
    """L_def=(1/N) sum_i ||Delta_i||²。"""
    raise NotImplementedError("待实现位移幅度正则。")


def displacement_smoothness(displacements: Tensor, neighbors: Tensor) -> Tensor:
    """L_smooth=mean_i mean_{j in N(i)} ||Delta_i-Delta_j||²。"""
    raise NotImplementedError("待实现源点邻域内位移平滑正则。")


def local_distance_preservation(
    source: Tensor, warped: Tensor, neighbors: Tensor
) -> Tensor:
    """L_topo=mean_i mean_j (||W_i-W_j||-||P_i-P_j||)²。

    使用最终 warped 几何，包含全局尺度 s；不改为仅对 P+Delta 计算。
    两侧点云使用一致物理单位，此项是局部距离约束，不是拓扑保证。
    """
    raise NotImplementedError("待实现源点与最终配准点的邻域边长保持项。")


class VisibilityAwareObjective(nn.Module):
    """训练专用；mask 和相机参数在此处提供，不进入配准网络。

    注入 rasterizer 以保留渲染后端边界。输出各项原始标量及加权 total，
    调度器在真实数据刚体 warm-up 时传 include_deformation=False。
    """

    def __init__(
        self, config: LossConfig, rasterizer: DifferentiablePointRasterizer
    ) -> None:
        super().__init__()
        self.config = config
        self.rasterizer = rasterizer

    def forward(
        self,
        output: RegistrationOutput,
        sample: TrainingSample,
        *,
        include_deformation: bool = True,
    ) -> dict[str, Tensor]:
        """六项权重默认依次为 0.5、1.2、1.0、0.1、0.1、0.3。"""
        if not sample.views:
            raise ValueError("自监督损失至少需要一个有效视角。")
        per_view: dict[str, list[Tensor]] = {"3d": [], "vis": [], "sil": []}
        for view in sample.views:
            warped_camera = reference_to_camera(
                output.warped, view.reference_to_camera
            )
            rendered = self.rasterizer(
                warped_camera, view.intrinsics, tuple(view.mask.shape[-2:])
            )
            visible = visible_domain(
                rendered, view, self.rasterizer.config.max_visible_points
            )
            observed = view.target.xyz
            per_view["3d"].append(one_way_chamfer(observed, warped_camera))
            per_view["vis"].append(symmetric_chamfer(visible, observed))
            per_view["sil"].append(
                silhouette_loss(rendered.silhouette, view.mask, self.config)
            )
        terms = {name: torch.stack(values).mean() for name, values in per_view.items()}
        zero = output.warped.new_zeros(())
        terms.update({"def": zero, "smooth": zero, "topo": zero})
        if include_deformation:
            source = sample.geometry.source.xyz
            neighbors = source_neighbors(source, self.config.neighborhood_k)
            terms["def"] = deformation_magnitude(output.displacements)
            terms["smooth"] = displacement_smoothness(output.displacements, neighbors)
            terms["topo"] = local_distance_preservation(source, output.warped, neighbors)
        weights = {
            "3d": self.config.lambda_3d,
            "vis": self.config.lambda_vis,
            "sil": self.config.lambda_sil,
            "def": self.config.lambda_def,
            "smooth": self.config.lambda_smooth,
            "topo": self.config.lambda_topo,
        }
        terms["total"] = sum((weights[name] * terms[name] for name in weights), zero)
        return terms
