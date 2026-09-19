"""Differentiable level-set propagation of a fire front.

The fire front is the zero level set of a signed distance function ``phi``
(negative inside the burned area). It evolves under

    d(phi)/dt + R(n) * |grad phi| = 0

where ``R`` is the direction-dependent rate of spread from
:mod:`pyrofield.physics.rothermel` and ``n`` is the outward normal.

Spatial discretisation uses the Godunov upwind scheme for an outward-moving
front (R > 0), which is what keeps the front sharp instead of diffusing it.
Everything is written in plain torch ops so gradients flow back to the physical
parameters through the entire rollout.

Grid convention: ``phi`` has shape ``(ny, nx)``; axis 0 is y (rows), axis 1 is x
(columns). Distances are in metres.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Classical minmod limiter: the smaller slope, or zero across a sign change."""
    return torch.where(a * b > 0, torch.where(a.abs() < b.abs(), a, b), torch.zeros_like(a))


def _one_sided_diffs(phi: torch.Tensor, dx: float, order: int = 2):
    """Backward/forward differences in x and y with replicate padding.

    ``order=1`` is plain first-order upwind; ``order=2`` adds the ENO minmod
    correction, which cuts the grid anisotropy of a circular front from ~5.6% to
    well under 1% (see ``scripts/validate_physics.py``). The anisotropy matters
    here because a spuriously slow diagonal would be indistinguishable from a real
    wind effect, and wind is exactly the quantity this project is trying to
    identify.
    """
    if order == 1:
        p = F.pad(phi[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        d_mx = (phi - p[1:-1, :-2]) / dx
        d_px = (p[1:-1, 2:] - phi) / dx
        d_my = (phi - p[:-2, 1:-1]) / dx
        d_py = (p[2:, 1:-1] - phi) / dx
        return d_mx, d_px, d_my, d_py

    p = F.pad(phi[None, None], (2, 2, 2, 2), mode="replicate")[0, 0]
    c = p[2:-2, 2:-2]

    # --- x direction (axis 1) ---
    xm1, xm2 = p[2:-2, 1:-3], p[2:-2, 0:-4]
    xp1, xp2 = p[2:-2, 3:-1], p[2:-2, 4:]
    d2_x = (xp1 - 2 * c + xm1) / dx**2
    d2_xm = (c - 2 * xm1 + xm2) / dx**2
    d2_xp = (xp2 - 2 * xp1 + c) / dx**2
    d_mx = (c - xm1) / dx + 0.5 * dx * _minmod(d2_xm, d2_x)
    d_px = (xp1 - c) / dx - 0.5 * dx * _minmod(d2_x, d2_xp)

    # --- y direction (axis 0) ---
    ym1, ym2 = p[1:-3, 2:-2], p[0:-4, 2:-2]
    yp1, yp2 = p[3:-1, 2:-2], p[4:, 2:-2]
    d2_y = (yp1 - 2 * c + ym1) / dx**2
    d2_ym = (c - 2 * ym1 + ym2) / dx**2
    d2_yp = (yp2 - 2 * yp1 + c) / dx**2
    d_my = (c - ym1) / dx + 0.5 * dx * _minmod(d2_ym, d2_y)
    d_py = (yp1 - c) / dx - 0.5 * dx * _minmod(d2_y, d2_yp)

    return d_mx, d_px, d_my, d_py


def grad_magnitude_upwind(phi: torch.Tensor, dx: float) -> torch.Tensor:
    """Godunov ``|grad phi|`` for a front moving outward (speed > 0)."""
    d_mx, d_px, d_my, d_py = _one_sided_diffs(phi, dx)
    gx = torch.clamp(d_mx, min=0.0) ** 2 + torch.clamp(d_px, max=0.0) ** 2
    gy = torch.clamp(d_my, min=0.0) ** 2 + torch.clamp(d_py, max=0.0) ** 2
    return torch.sqrt(gx + gy + 1e-12)


def normal_angle(phi: torch.Tensor, dx: float) -> torch.Tensor:
    """Outward normal direction of the front, as an angle in radians.

    ``phi`` is negative inside, so ``grad phi`` already points outward.
    """
    d_mx, d_px, d_my, d_py = _one_sided_diffs(phi, dx)
    gx = 0.5 * (d_mx + d_px)
    gy = 0.5 * (d_my + d_py)
    return torch.atan2(gy, gx)


def reinitialize(phi: torch.Tensor, dx: float, n_iter: int = 5) -> torch.Tensor:
    """Push ``phi`` back towards a signed distance function (``|grad phi| = 1``).

    Solves a few steps of ``d(phi)/dtau = sign(phi0) * (1 - |grad phi|)``. Kept
    differentiable so it can sit inside the rollout; ``n_iter`` is small because
    we only need to stop the level-set function from degenerating, not to reach
    a perfect distance field.
    """
    phi0 = phi
    sgn = phi0 / torch.sqrt(phi0**2 + dx**2)  # smoothed sign
    dtau = 0.4 * dx
    out = phi
    for _ in range(n_iter):
        d_mx, d_px, d_my, d_py = _one_sided_diffs(out, dx)
        # Godunov scheme selected by the sign of phi0.
        pos = torch.sqrt(
            torch.clamp(d_mx, min=0.0) ** 2
            + torch.clamp(d_px, max=0.0) ** 2
            + torch.clamp(d_my, min=0.0) ** 2
            + torch.clamp(d_py, max=0.0) ** 2
            + 1e-12
        )
        neg = torch.sqrt(
            torch.clamp(d_mx, max=0.0) ** 2
            + torch.clamp(d_px, min=0.0) ** 2
            + torch.clamp(d_my, max=0.0) ** 2
            + torch.clamp(d_py, min=0.0) ** 2
            + 1e-12
        )
        grad = torch.where(phi0 > 0, pos, neg)
        out = out - dtau * sgn * (grad - 1.0)
    return out


def step(
    phi: torch.Tensor,
    speed: torch.Tensor,
    dt: float,
    dx: float,
) -> torch.Tensor:
    """One explicit upwind step of the level-set equation.

    Args:
        phi: signed distance field, ``(ny, nx)``.
        speed: outward normal speed at every cell, same shape as ``phi``.
        dt: time step in seconds.
        dx: cell size in metres.
    """
    return phi - dt * speed * grad_magnitude_upwind(phi, dx)


def init_circle(
    ny: int,
    nx: int,
    dx: float,
    center_xy: tuple[float, float],
    radius: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Signed distance field for a circular ignition of the given radius."""
    ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * dx
    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = center_xy
    return torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) - radius


def burned_fraction(phi: torch.Tensor, dx: float, tau_cells: float = 1.0) -> torch.Tensor:
    """Soft burned-area indicator in [0, 1]; the differentiable mask operator's core."""
    return torch.sigmoid(-phi / (tau_cells * dx))


def front_band(phi: torch.Tensor, dx: float, width_cells: float = 1.5) -> torch.Tensor:
    """Soft indicator of cells on the active fire line (a band around phi = 0)."""
    return torch.exp(-0.5 * (phi / (width_cells * dx)) ** 2)


def cfl_dt(max_speed: float, dx: float, safety: float = 0.4) -> float:
    """Largest stable explicit time step for the given maximum spread rate."""
    return safety * dx / max(max_speed, 1e-6)
