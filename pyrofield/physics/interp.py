"""Trilinear sampling written in primitive ops, so the whole model is forward-AD-able.

``torch.nn.functional.grid_sample`` has no forward-mode autodiff rule for the 3-D
case, which would force the Fisher analysis to rely on finite differences. That
matters: the fire mask is nearly binary, so its finite-difference Jacobian is
sensitive to the step size (the mask-only Cramer-Rao bound moved by 74% across a
4x range of eps), and a bound that depends on how you measure it is not a bound.

Implementing the interpolation with floor/clamp/gather/arithmetic gives forward-mode
AD the exact derivative of the discretised model, with no step size to choose.
Behaviour matches ``grid_sample(..., mode='bilinear', padding_mode='zeros',
align_corners=False)``: cell centres sit at ``(i + 0.5) * spacing`` and samples
outside the volume read as zero.
"""

from __future__ import annotations

import torch


def trilinear(
    vol: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    dx: float,
    dz: float,
) -> torch.Tensor:
    """Sample ``vol`` ``(nz, ny, nx)`` at world coordinates, zero outside the box.

    Args:
        vol: density or scalar field on the grid.
        x, y, z: world coordinates in metres, any common broadcastable shape.
        dx: horizontal cell size; dz: vertical cell size.

    Returns:
        Tensor shaped like the broadcast of ``x``, ``y``, ``z``.
    """
    nz, ny, nx = vol.shape

    fx = x / dx - 0.5
    fy = y / dx - 0.5
    fz = z / dz - 0.5

    x0f, y0f, z0f = torch.floor(fx), torch.floor(fy), torch.floor(fz)
    tx, ty, tz = fx - x0f, fy - y0f, fz - z0f
    x0, y0, z0 = x0f.long(), y0f.long(), z0f.long()

    out = torch.zeros_like(tx)
    for kz in (0, 1):
        wz = tz if kz else (1.0 - tz)
        zi = z0 + kz
        ok_z = (zi >= 0) & (zi < nz)
        zc = zi.clamp(0, nz - 1)
        for ky in (0, 1):
            wy = ty if ky else (1.0 - ty)
            yi = y0 + ky
            ok_y = (yi >= 0) & (yi < ny)
            yc = yi.clamp(0, ny - 1)
            for kx in (0, 1):
                wx = tx if kx else (1.0 - tx)
                xi = x0 + kx
                ok = ok_z & ok_y & (xi >= 0) & (xi < nx)
                xc = xi.clamp(0, nx - 1)
                out = out + (wx * wy * wz) * vol[zc, yc, xc] * ok.to(vol.dtype)
    return out
