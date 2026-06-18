# Vis2Reg

This repository contains the core model implementation for the MICCAI 2026 paper:

**Vis2Reg: Visibility-Aware Landmark-Free Geometric 3D-2D Registration for Liver Laparoscopy**

Vis2Reg includes the geometric and visual encoders, cross-modal fusion, matching heads, rigid pose solvers, deformation fields, and geometry utilities used by the framework.

## Structure

```
Vis2Reg/
├── environment.yml
└── vis2reg/
    ├── models/
    │   ├── encoders/
    │   ├── fields/
    │   ├── fusion/
    │   ├── matching/
    │   ├── renderer/
    │   ├── rigid/
    │   ├── local_matcher.py
    │   └── vis2reg.py
    └── utils/
        ├── geometry.py
        └── logger.py
```

## Usage

```python
from vis2reg import Vis2Reg

model = Vis2Reg(cfg)
```

The rigid branch supports direct pose regression and learned matching with weighted SVD. The matching path can optionally use local matching, OT/Sinkhorn, a GeoTransformer-style coarse-to-fine matcher, RANSAC, and ICP refinement.
