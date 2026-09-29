"""论文 §2.1：g_phi(PE(x) concat F^r_{P|Q}(x)) -> Delta(x)。"""

import torch
from torch import Tensor, nn

from ..config import FieldConfig


class PositionalEncoding(nn.Module):
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config

    def forward(self, source: Tensor) -> Tensor:
        """[N,3] -> [N,d_pe]；频率、是否保留原坐标、尺度处理待明确。"""
        raise NotImplementedError("待明确位置编码细节；不擅自归一化坐标")


class SirenMLP(nn.Module):
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config

    def forward(self, encoded: Tensor) -> Tensor:
        """[N,d_pe+C_r] -> [N,3]；正弦激活，图 2 标注 skip @ 2,4。

        TODO: 补层宽、深度、omega、SIREN 初始化、skip 拼接与层索引约定。
        """
        raise NotImplementedError("待定义 SIREN 形变网络")


class ImplicitDeformationField(nn.Module):
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.position = PositionalEncoding(config)
        self.mlp = SirenMLP(config)

    def forward(self, source: Tensor, pair_features: Tensor) -> Tensor:
        return self.mlp(torch.cat([self.position(source), pair_features], dim=-1))
