"""Fire-mask observation operator: what every pixel-space method sees, and only that.

This is the operator implicitly assumed by the entire mask/pixel-space literature.
It projects the state field down to a burned/unburned indicator, discarding fire
intensity, wind and source strength entirely -- which is exactly why those
quantities end up confounded (Proposition P1).

Kept differentiable via a sigmoid of the signed distance so it can be used both to
synthesise observations and to invert them.
"""

from __future__ import annotations

import torch

from pyrofield.physics.levelset import burned_fraction


def soft_mask(phi: torch.Tensor, dx: float, tau_cells: float = 1.0) -> torch.Tensor:
    """Soft burned-area mask in [0, 1] from the signed distance field."""
    return burned_fraction(phi, dx, tau_cells)


def hard_mask(phi: torch.Tensor) -> torch.Tensor:
    """Binary burned-area mask (non-differentiable; for metrics and visualisation)."""
    return (phi < 0).to(phi.dtype)


def iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Intersection over union between two masks."""
    p = (pred > threshold).to(torch.bool)
    t = (target > threshold).to(torch.bool)
    inter = (p & t).sum().to(pred.dtype)
    union = (p | t).sum().to(pred.dtype)
    return inter / torch.clamp(union, min=1.0)
