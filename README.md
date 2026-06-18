# UniRegNet

This repository contains the core model implementation for the MICCAI 2026 paper:

**Vis2Reg: Visibility-Aware Landmark-Free Geometric 3D-2D Registration for Liver Laparoscopy**

UniRegNet includes the geometric and visual encoders, cross-modal fusion, matching heads, rigid pose solvers, deformation fields, and geometry utilities used by the Vis2Reg framework.

## Structure

```
UniRegNet/
├── environment.yml
└── uniregnet/
    ├── models/
    │   ├── encoders/
    │   ├── fields/
    │   ├── fusion/
    │   ├── matching/
    │   ├── renderer/
    │   ├── rigid/
    │   ├── local_matcher.py
    │   └── uniregnet.py
    └── utils/
        ├── geometry.py
        └── logger.py
```

## Usage

```python
from uniregnet import UniRegNet

model = UniRegNet(cfg)
```

The rigid branch supports direct pose regression and learned matching with weighted SVD. The matching path can optionally use local matching, OT/Sinkhorn, a GeoTransformer-style coarse-to-fine matcher, RANSAC, and ICP refinement.
