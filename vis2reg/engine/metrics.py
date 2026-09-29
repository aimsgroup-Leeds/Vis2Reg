"""Paper metric definitions; aggregation follows supplied official five folds.

CAUTION: the paper defines squared CD but tables label CD in mm. Squared metric
coordinates give mm^2; table conversion is unresolved and must not be inferred.
"""
import torch
from torch import Tensor

from ..types import SimilarityTransform


def pose_errors(predicted: SimilarityTransform, truth: SimilarityTransform) -> dict[str, Tensor]:
    """Both translations must use millimeters; scale is not part of paper RRE/RTE."""
    cosine = ((torch.trace(predicted.rotation.T @ truth.rotation) - 1) / 2).clamp(-1, 1)
    return {"rre_degrees": torch.rad2deg(torch.acos(cosine)),
            "rte_mm": torch.linalg.vector_norm(predicted.translation - truth.translation)}


def observation_to_model_cd_mm2(observation_mm: Tensor, warped_mm: Tensor) -> Tensor:
    """One-way L_3D; supply valid points in one coordinate frame, measured in mm.

    Minimal equation translation; production nearest-neighbor backend is unbound.
    Average per-view values for the reported real-data objective.
    """
    if len(observation_mm) == 0 or len(warped_mm) == 0:
        raise ValueError("Empty point-set metric policy must be explicit.")
    return torch.cdist(observation_mm, warped_mm).square().min(dim=1).values.mean()


def silhouette_dice_percent(predicted_binary: Tensor, target_binary: Tensor) -> Tensor:
    """Threshold rendered silhouettes explicitly before calling; paper omits it."""
    if predicted_binary.dtype != torch.bool or target_binary.dtype != torch.bool:
        raise ValueError("Supply boolean silhouettes with an explicit threshold policy.")
    if predicted_binary.shape != target_binary.shape:
        raise ValueError("Silhouette shapes differ.")
    denominator = predicted_binary.sum() + target_binary.sum()
    if denominator == 0:
        raise ValueError("Both-empty Dice policy is unspecified.")
    return 200.0 * (predicted_binary & target_binary).sum() / denominator


def table_cd_mm(*args, **kwargs) -> Tensor:
    """Do not silently take sqrt: sqrt(mean(d^2)) differs from mean(d)."""
    raise NotImplementedError("Resolve paper equation/table CD-unit discrepancy first.")
