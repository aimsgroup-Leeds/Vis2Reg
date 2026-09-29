"""训练阶段的可微渲染与可见域监督接口。"""

from .rasterizer import DifferentiablePointRasterizer
from .visibility import reference_to_camera, visible_domain

__all__ = ["DifferentiablePointRasterizer", "reference_to_camera", "visible_domain"]
