"""已报告值来自论文 §3.1；None 表示论文未明确，不能视作可运行默认值。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    max_points: int = 6000
    num_views: int = 3
    normalize: bool = False
    center: bool = False


@dataclass(frozen=True)
class EncoderConfig:
    channels: tuple[int, ...] | None = None
    neighborhood_k: int | None = None
    pair_conditioning: str | None = None


@dataclass(frozen=True)
class RigidConfig:
    temperature: float | None = None
    sinkhorn_iterations: int | None = None
    prosac_hypotheses: int | None = None
    top_hypotheses: int | None = None
    inlier_threshold: float | None = None
    icp_trim_fraction: float | None = None
    icp_iterations: int | None = None
    scale_mode: str | None = None
    gradient_policy: str | None = None


@dataclass(frozen=True)
class FieldConfig:
    positional_frequencies: int | None = None
    hidden_dim: int | None = None
    num_layers: int | None = None
    sine_omega: float | None = None
    skip_labels: tuple[int, ...] = (2, 4)  # 图 2 标注；索引约定仍待明确


@dataclass(frozen=True)
class RasterizerConfig:
    points_per_pixel: int = 16
    max_visible_points: int = 5000
    point_radius: float | None = None
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class LossConfig:
    lambda_3d: float = 0.5
    lambda_vis: float = 1.2
    lambda_sil: float = 1.0
    lambda_def: float = 0.1
    lambda_smooth: float = 0.1
    lambda_topo: float = 0.3
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    neighborhood_k: int | None = None


@dataclass(frozen=True)
class TrainConfig:
    synthetic_epochs: int | None = None
    real_epochs: int = 60
    rigid_warmup_epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    amp: bool = True
