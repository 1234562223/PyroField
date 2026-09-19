"""Differentiable 3-D smoke transport driven by the fire line.

Smoke is emitted along the active fire line at a rate set by the source strength
``Q``, then advected by the horizontal wind plus a vertical buoyancy velocity and
removed by a first-order decay:

    d(rho)/dt + div(v rho) = S(front) - kappa rho,    v = (U cos tw, U sin tw, w_b)

Advection uses a semi-Lagrangian step (unconditionally stable, and differentiable
because the backtrace is a ``grid_sample``).

The physically important consequence, and the reason this module exists, is that
the *tilt* of the resulting plume encodes ``atan(U / w_b)`` while its *horizontal
drift between frames* encodes ``U`` in absolute terms. Those are two different
pieces of information, and the Week-0 identifiability study measures both.

Grid convention: ``rho`` has shape ``(nz, ny, nx)``.
"""

from __future__ import annotations

import torch

from pyrofield.physics.interp import trilinear


def advect_semilagrangian(
    rho: torch.Tensor,
    velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dt: float,
    dx: float,
    dz: float,
) -> torch.Tensor:
    """Backtrace every cell along a constant velocity and resample the density.

    Args:
        rho: density field ``(nz, ny, nx)``.
        velocity: ``(vx, vy, vz)`` scalar tensors in m/s.
        dt: time step (s); ``dx``/``dz``: cell sizes (m).
    """
    nz, ny, nx = rho.shape
    device, dtype = rho.device, rho.dtype
    vx, vy, vz = velocity

    zs = (torch.arange(nz, device=device, dtype=dtype) + 0.5) * dz
    ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * dx
    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * dx
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")

    # Departure points of the characteristics.
    return trilinear(rho, gx - vx * dt, gy - vy * dt, gz - vz * dt, dx, dz)


def emit(
    rho: torch.Tensor,
    front: torch.Tensor,
    Q: torch.Tensor,
    dt: float,
    n_source_levels: int = 2,
) -> torch.Tensor:
    """Inject smoke into the lowest levels wherever the fire line is active.

    Args:
        rho: density field ``(nz, ny, nx)``.
        front: soft fire-line indicator ``(ny, nx)`` in [0, 1].
        Q: source strength (scalar tensor).
        dt: time step (s).
    """
    out = rho.clone()
    add = Q * front * dt
    for k in range(min(n_source_levels, rho.shape[0])):
        out[k] = out[k] + add
    return out


def step(
    rho: torch.Tensor,
    front: torch.Tensor,
    Q: torch.Tensor,
    U: torch.Tensor,
    theta_w: torch.Tensor,
    w_buoy: torch.Tensor,
    dt: float,
    dx: float,
    dz: float,
    kappa: float = 1.0 / 900.0,
) -> torch.Tensor:
    """One emit -> advect -> decay step of the smoke field."""
    vx = U * torch.cos(theta_w)
    vy = U * torch.sin(theta_w)
    rho = emit(rho, front, Q, dt)
    rho = advect_semilagrangian(rho, (vx, vy, w_buoy), dt, dx, dz)
    return rho * float(torch.exp(torch.tensor(-kappa * dt)))


def column_optical_depth(rho: torch.Tensor, dz: float, extinction: float = 1.0) -> torch.Tensor:
    """Nadir optical depth of the smoke column, ``(ny, nx)``.

    A cheap stand-in for what a downward-looking sensor sees; the oblique camera
    view is handled by :mod:`pyrofield.models.operators.plume_render`.
    """
    return extinction * rho.sum(dim=0) * dz
