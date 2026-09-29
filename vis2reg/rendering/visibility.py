"""由论文的 U_v 与 V_v^M 定义组织可见域构造。"""

from torch import Tensor

from ..types import RenderedView, ViewObservation


def reference_to_camera(points: Tensor, transform: Tensor) -> Tensor:
    """显式约定：外参为参考系到当前视角相机系的 4×4 刚体矩阵。

    论文未交代多视角外参处理；此接口为结构假设，数据侧必须提供，
    若各视角已共享同一坐标系，则显式传入单位矩阵。
    """
    return points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]


def backproject_selected(depth: Tensor, intrinsics: Tensor, gate: Tensor) -> Tensor:
    """将 gate 内的像素按 d(u) K^-1 [u_x,u_y,1]^T 反投影为 [L,3]。

    gate 是离散支持集；选中的深度数值须保留到渲染点云的梯度。
    待明确像素中心约定，并实现空支持集策略，不能静默生成伪观测。
    """
    raise NotImplementedError("待实现 mask 内有效深度反投影及空支持集策略。")


def cap_visible_points(points: Tensor, maximum: int) -> Tensor:
    """最多保留 maximum 个点；论文 maximum=5000。

    采样方式未报告，保留为待实现接口。选点索引可离散，选中点坐标
    不应 detach；采样应同时明确随机性、点数不足与空集的处理。
    """
    raise NotImplementedError("待确定可见点采样策略；上限来自论文。")


def visible_domain(
    rendered: RenderedView,
    view: ViewObservation,
    max_visible_points: int,
) -> Tensor:
    """U_v={u | D_v(u)>0 且 M_v(u)=1}；输出相机系 V_v^M。"""
    gate = (rendered.depth > 0) & (view.mask == 1)
    points = backproject_selected(rendered.depth, view.intrinsics, gate)
    return cap_visible_points(points, max_visible_points)
