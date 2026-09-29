"""论文 §2.1 / 图 2：EdgeConv 与两个不同职责的编码器。"""

from torch import Tensor, nn

from ..config import EncoderConfig


class EdgeConvEncoder(nn.Module):
    """动态 kNN + shared MLP + 邻域 max；拼接各层输出得到多尺度特征。"""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        # TODO: 依据明确的 channels、k 构建各层；图示未给通道数。

    def forward(self, points: Tensor) -> Tensor:
        """[N,3] -> [N,C]，h0=x；max_j psi([h_i || h_j-h_i])。"""
        raise NotImplementedError("待补动态邻域、EdgeConv 层与多尺度拼接")


class PairConditionedEncoder(nn.Module):
    """E_r(P,Q) 输出源点上的 F^r_{P|Q}，与匹配编码器分开。"""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """[N,3],[M,3] -> [N,C_r]；必须依赖 Q，不能退化为 E_r(P)。

        TODO: 分别提取几何上下文并融合至源点；论文未说明具体融合算子，
        不自行将 cross-attention、池化拼接等某一方案当作原文实现。
        """
        raise NotImplementedError("待明确 pair conditioning 和源点特征查询方式")
