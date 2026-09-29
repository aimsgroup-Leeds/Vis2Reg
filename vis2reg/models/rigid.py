"""鲁棒刚性种子；不把离散匹配、PROSAC、ICP 默认当成可微算子。"""

from torch import Tensor, nn

from ..config import RigidConfig
from ..types import SimilarityTransform
from .matching import Correspondences


class RobustRigidInitializer(nn.Module):
    def __init__(self, config: RigidConfig):
        super().__init__()
        self.config = config

    def prosac(
        self, source: Tensor, target: Tensor, matches: Correspondences,
    ) -> list[SimilarityTransform]:
        """按匹配置信度排序，从逐步扩大的高置信前缀取最小样本生成假设。"""
        raise NotImplementedError("待补 PROSAC、退化样本检查和尺度策略")

    def rank_and_select(
        self, hypotheses: list[SimilarityTransform], source: Tensor,
        target: Tensor, matches: Correspondences,
    ) -> list[SimilarityTransform]:
        """按加权内点分数排序，返回 top-L；阈值与 L 论文未给出。"""
        raise NotImplementedError("待补加权内点评分及候选筛选")

    def refine(
        self, pose: SimilarityTransform, source: Tensor, target: Tensor,
    ) -> SimilarityTransform:
        """trimmed point-to-point + point-to-plane ICP，随后重评估。"""
        raise NotImplementedError("待补 ICP 裁剪比例、迭代策略及目标法向估计")

    def forward(
        self, source: Tensor, target: Tensor, matches: Correspondences,
    ) -> SimilarityTransform:
        # TODO: 实现前明确梯度路径；不能只接 optimizer 就声称匹配器可训练。
        hypotheses = self.prosac(source, target, matches)
        candidates = self.rank_and_select(hypotheses, source, target, matches)
        refined = [self.refine(pose, source, target) for pose in candidates]
        return self.rank_and_select(refined, source, target, matches)[0]
