"""论文的可微点光栅化边界；尚未选择具体渲染后端。"""

from torch import Tensor, nn

from ..config import RasterizerConfig
from ..types import RenderedView


class DifferentiablePointRasterizer(nn.Module):
    """由相机系点云生成 soft silhouette 与深度。

    论文参数 points_per_pixel=16；半径、透明度、深度合成规则和后端
    坐标约定未给出。接入后端时必须使轮廓及有效深度对输入点保留梯度，
    不可用硬二值投影或 detach 的 z-buffer 冒充可微渲染。
    """

    def __init__(self, config: RasterizerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        points_camera: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
    ) -> RenderedView:
        """输入 [N,3]、[3,3] 与 (H,W)，输出 [H,W] 的轮廓和深度。

        深度空像素必须为 0；轮廓应为 BCE/Dice 使用的 [0,1] 概率。
        正深度方向、像素原点和内参单位须与数据侧约定一致。
        """
        raise NotImplementedError(
            "待接入可微点光栅化后端，并明确点半径、遮挡与深度合成规则。"
        )
