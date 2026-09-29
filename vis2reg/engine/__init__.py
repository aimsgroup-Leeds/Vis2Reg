"""Training-stage and evaluation structure; no execution is performed."""

from .stages import Stage, real_stage, synthetic_stage, Trainer

__all__ = ["Stage", "real_stage", "synthetic_stage", "Trainer"]
