"""GeoTransformer -> similarity -> Sinkhorn -> mutual NN 的接口。"""

from dataclasses import dataclass

from torch import Tensor, nn

from ..config import RigidConfig


@dataclass
class Correspondences:
    source_indices: Tensor  # [C]，long
    target_indices: Tensor  # [C]，long
    confidence: Tensor  # [C]，w_ij = Pi_ij


@dataclass
class MatchingOutput:
    assignment: Tensor  # [N, M]
    correspondences: Correspondences


class GeometricMatcher(nn.Module):
    def __init__(self, config: RigidConfig):
        super().__init__()
        self.config = config

    def contextualize(
        self, source: Tensor, target: Tensor, source_features: Tensor,
        target_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """GeoTransformer 几何上下文化；层数、维度、几何嵌入待补。"""
        raise NotImplementedError("论文未给 GeoTransformer 的具体网络配置")

    def sinkhorn(self, similarity: Tensor) -> Tensor:
        """A -> Pi；迭代数、未匹配点处理和归一化边际待明确。"""
        raise NotImplementedError("待实现 Sinkhorn 软分配")

    def mutual_nearest(self, assignment: Tensor) -> Correspondences:
        """同时为行、列最大值的匹配，保留 Pi_ij 作为置信度。"""
        raise NotImplementedError("待实现 mutual filtering 及空匹配策略")

    def forward(
        self, source: Tensor, target: Tensor, source_features: Tensor,
        target_features: Tensor,
    ) -> MatchingOutput:
        fp, fq = self.contextualize(source, target, source_features, target_features)
        if self.config.temperature is None:
            raise NotImplementedError("论文未指定相似度温度 tau")
        similarity = (fp @ fq.T) / self.config.temperature
        assignment = self.sinkhorn(similarity)
        return MatchingOutput(assignment, self.mutual_nearest(assignment))
