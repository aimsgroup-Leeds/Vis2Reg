"""Input preprocessing contracts from the implementation-details paragraph."""
from typing import Protocol

import torch
from torch import Tensor

from ..types import PointCloud, TrainingSample


def resample_or_pad(points: Tensor, max_points: int = 6000) -> PointCloud:
    """Uniform subsampling is an explicit scaffold choice, not stated in paper.

    Zero padding is excluded downstream using valid; never center or normalize.
    Call independently for source, merged target and per-view target as needed.
    """
    if points.ndim != 2 or points.shape[1] != 3 or max_points < 1:
        raise ValueError("Expected [N, 3] points and positive max_points.")
    if len(points) > max_points:
        points = points[torch.randperm(len(points), device=points.device)[:max_points]]
    padded = points.new_zeros((max_points, 3))
    padded[:len(points)] = points
    valid = torch.arange(max_points, device=points.device) < len(points)
    return PointCloud(points=padded, valid=valid)


class Preprocessor(Protocol):
    """Adapter: dropout + jitter on P, statistical denoising on Q during training.

    TODO: supply dropout rate, jitter scale/clipping, denoising neighbors/threshold,
    depth-scale calibration and augmentation/pose consistency; none are specified.
    Reconstruct Q_v from DepthAnything depth, M_v and K_v, transform before merging,
    and return source/targets with valid masks. F=3 is not temporal recurrence.
    """

    def __call__(self, raw_record: object, *, training: bool) -> TrainingSample: ...
