"""Analytic Gaussian-plume operator: a cheap air-quality sensor as a physical constraint.

A ground PM2.5 / CO monitor reports a single number. Under the classical Gaussian
plume solution that number is

    C = Q / (2 pi U sy sz) * exp(-yc^2 / 2 sy^2)
        * [exp(-(zr - H)^2 / 2 sz^2) + exp(-(zr + H)^2 / 2 sz^2)]

summed over the emitting fire-line cells, where ``yc``/``xd`` are the crosswind and
downwind offsets from source to receptor and ``sy``, ``sz`` are Pasquill-Gifford
dispersion widths that grow with ``xd``.

The point of this operator is the ``Q / U`` in the denominator: a sensor that costs
almost nothing becomes an equation constraining absolute wind speed and absolute
source strength, which the plume image alone cannot supply (an image fixes only the
tilt ratio ``U / w_buoy``). It is fully analytic and therefore fully differentiable.
"""

from __future__ import annotations

import math

import torch

# Pasquill-Gifford style dispersion coefficients (neutral stability, open terrain).
SY_A, SY_B = 0.08, 0.90
SZ_A, SZ_B = 0.06, 0.90


def dispersion_widths(xd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crosswind and vertical plume widths at downwind distance ``xd`` (m)."""
    xs = torch.clamp(xd, min=1.0)
    return SY_A * xs**SY_B + 1.0, SZ_A * xs**SZ_B + 1.0


def concentration(
    source_xy: torch.Tensor,
    source_w: torch.Tensor,
    receptor: tuple[float, float, float],
    Q: torch.Tensor,
    U: torch.Tensor,
    theta_w: torch.Tensor,
    release_height: float = 20.0,
) -> torch.Tensor:
    """Concentration at one receptor from a set of weighted line-source cells.

    Args:
        source_xy: ``(N, 2)`` source cell centres in metres.
        source_w: ``(N,)`` non-negative emission weights (fire-line activity).
        receptor: ``(x, y, z)`` sensor position in metres.
        Q: source strength scale (scalar tensor).
        U: wind speed in m/s (scalar tensor).
        theta_w: wind direction in radians (direction the wind blows towards).
        release_height: effective plume release height in metres.

    Returns:
        Scalar tensor: concentration in arbitrary but consistent units.
    """
    xr, yr, zr = receptor
    dxv = xr - source_xy[:, 0]
    dyv = yr - source_xy[:, 1]

    cos_w, sin_w = torch.cos(theta_w), torch.sin(theta_w)
    xd = dxv * cos_w + dyv * sin_w  # downwind offset
    yc = -dxv * sin_w + dyv * cos_w  # crosswind offset

    # Only sources upwind of the receptor contribute; softened to stay differentiable.
    upwind = torch.sigmoid(xd / 25.0)

    sy, sz = dispersion_widths(xd)
    Us = torch.clamp(U, min=0.1)

    horiz = torch.exp(-0.5 * (yc / sy) ** 2)
    vert = torch.exp(-0.5 * ((zr - release_height) / sz) ** 2) + torch.exp(
        -0.5 * ((zr + release_height) / sz) ** 2
    )
    contrib = source_w * upwind * horiz * vert / (2.0 * math.pi * Us * sy * sz)
    return Q * contrib.sum()


def concentration_field(
    source_xy: torch.Tensor,
    source_w: torch.Tensor,
    receptors: list[tuple[float, float, float]],
    Q: torch.Tensor,
    U: torch.Tensor,
    theta_w: torch.Tensor,
    release_height: float = 20.0,
) -> torch.Tensor:
    """Concentrations at several receptors, returned as a ``(n_receptors,)`` tensor."""
    return torch.stack(
        [
            concentration(source_xy, source_w, r, Q, U, theta_w, release_height)
            for r in receptors
        ]
    )
