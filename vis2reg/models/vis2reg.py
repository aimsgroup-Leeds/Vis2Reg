"""核心数据流；训练用 mask/K 位于独立监督分支。"""

import torch
from torch import nn

from ..types import GeometryInput, RegistrationOutput
from .deformation import ImplicitDeformationField
from .encoders import EdgeConvEncoder, PairConditionedEncoder
from .matching import GeometricMatcher
from .rigid import RobustRigidInitializer


class Vis2Reg(nn.Module):
    def __init__(
        self, matching_encoder: EdgeConvEncoder,
        registration_encoder: PairConditionedEncoder, matcher: GeometricMatcher,
        rigid_initializer: RobustRigidInitializer, field: ImplicitDeformationField,
    ):
        super().__init__()
        self.matching_encoder = matching_encoder  # Siamese：P/Q 共享参数
        self.registration_encoder = registration_encoder
        self.matcher = matcher
        self.rigid_initializer = rigid_initializer
        self.field = field

    def forward(
        self, geometry: GeometryInput, *, deform: bool = True,
    ) -> RegistrationOutput:
        source, target = geometry.source.xyz, geometry.target.xyz
        fp = self.matching_encoder(source)
        fq = self.matching_encoder(target)
        matching = self.matcher(source, target, fp, fq)
        pose = self.rigid_initializer(source, target, matching.correspondences)
        if deform:
            pair_features = self.registration_encoder(source, target)
            displacement = self.field(source, pair_features)
        else:
            displacement = torch.zeros_like(source)
        # 先在源空间形变，再应用相似变换；不能调换这两个操作。
        warped = pose.apply(source + displacement)
        return RegistrationOutput(
            warped=warped, displacements=displacement, pose=pose,
            auxiliary={"assignment": matching.assignment},
        )
