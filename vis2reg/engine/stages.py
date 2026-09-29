"""Stage 1 synthetic pose supervision -> Stage 2 real visibility supervision."""
from dataclasses import dataclass
from typing import Callable, Iterable

from torch import Tensor

from ..types import RegistrationOutput, TrainingSample


@dataclass(frozen=True)
class Stage:
    name: str
    deform: bool
    freeze_deformation: bool
    include_deformation: bool
    requires_gt_pose: bool


def synthetic_stage() -> Stage:
    """Epoch count and pose-supervision loss are not specified in the paper."""
    return Stage("synthetic_rigid", False, True, False, True)


def real_stage(epoch: int, *, epochs: int = 60, warmup_epochs: int = 20) -> Stage:
    """Zero-based epochs: 0..19 rigid-only, 20..59 joint training."""
    if not 0 <= warmup_epochs <= epochs or not 0 <= epoch < epochs:
        raise ValueError("Invalid real-training epoch/schedule.")
    joint = epoch >= warmup_epochs
    return Stage("real_joint" if joint else "real_rigid", joint, not joint, joint, False)


class Trainer:
    """Dependency-injected scaffold; freeze/unfreeze E_r and the implicit field.

    Bind AdamW(lr=3e-4, weight_decay=1e-4), cosine scheduling, AMP and batch size 1.
    Real objective: VisibilityAwareObjective(output, sample,
    include_deformation=stage.include_deformation). Synthetic loss must be supplied.
    Model call: model(sample.geometry, deform=stage.deform); masks/K stay outside.
    """

    def __init__(self, model: object, real_objective: Callable,
                 synthetic_objective: Callable[[RegistrationOutput, TrainingSample], Tensor],
                 set_deformation_frozen: Callable[[bool], None]):
        self.model = model
        self.real_objective = real_objective
        self.synthetic_objective = synthetic_objective
        self.set_deformation_frozen = set_deformation_frozen

    def run_epoch(self, samples: Iterable[TrainingSample], stage: Stage) -> None:
        """TODO: validate GT policy, freeze non-rigid branch, forward, loss, step.

        set_deformation_frozen must cover registration_encoder and field, including
        train/eval state where relevant. Disable all three deformation regularizers
        during rigid-only stages, and use real_objective(...)["total"] for backward.

        Must first resolve gradients through MNN, PROSAC and trimmed ICP. The paper
        does not explain their differentiability or surrogate losses; a plain
        optimizer loop would falsely imply that rigid modules train end-to-end.
        Do not quietly detach them or invent an estimator in this scaffold.
        """
        raise NotImplementedError("Bind the unspecified gradient and optimization policy.")
