"""P2I-LReg: 21 patients, 346 keyframes; five official patient-level folds.

Synthetic data: 2500 samples/patient, split 60/20/20 within training-fold
patients only. Official fold IDs, sample IDs and file layouts need adapters.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

from torch.utils.data import Dataset

from ..types import TrainingSample


@dataclass(frozen=True)
class PatientFold:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if tuple(map(len, groups)) != (12, 4, 5):
            raise ValueError("Supply the official 12/4/5 unique patient IDs.")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Patient splits must be disjoint.")


class RealDataset(Dataset):
    """Reader must construct three local, near-static, non-temporal views.

    Each view needs mask, K, local camera cloud and explicit reference-to-camera
    transform; geometry.target is fused in that reference frame. The paper does
    not specify the cross-camera transform/fusion procedure: require an adapter.
    Return metric coordinates without centering or normalization, and preserve
    valid masks for padded points. Masks and K belong only to supervision.
    """

    def __init__(self, records: Sequence[object], reader: Callable[[object], TrainingSample],
                 patient_ids: Sequence[str]):
        self.records, self.reader = records, reader
        self.patient_ids = frozenset(patient_ids)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrainingSample:
        sample = self.reader(self.records[index])
        if sample.patient_id not in self.patient_ids:
            raise ValueError("Sample patient lies outside the selected split.")
        if sample.gt_pose is not None or len(sample.views) != 3:
            raise ValueError("Real protocol requires no GT pose and three local views.")
        return sample


class SyntheticDataset(RealDataset):
    """Pass training-fold patients, including for synthetic validation/test.

    Reader provides known pose and synthetic geometric pair. Blender generation
    settings and exact synthetic partition IDs are not specified in the paper.
    """

    def __getitem__(self, index: int) -> TrainingSample:
        sample = self.reader(self.records[index])
        if sample.patient_id not in self.patient_ids or sample.gt_pose is None:
            raise ValueError("Synthetic sample requires training-fold patient and GT pose.")
        return sample
