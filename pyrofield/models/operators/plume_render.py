"""Differentiable volume rendering of the smoke plume: the RGB observation operator.

This is the operator that turns "smoke occludes the fire" from a failure mode into
a measurement. A camera does not see the fire line directly, but it does see the
plume, and the plume is a *projection of the physical state*: its tilt carries
``atan(U / w_buoy)``, its opacity carries the source strength ``Q``, and its root
carries the fire-line position.

``render`` maps a 3-D smoke density field to what an oblique camera would record,
using standard emission-absorption volume rendering along perspective rays. It is
differentiable end to end, so image residuals backpropagate into wind, buoyancy
and source strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pyrofield.physics.interp import trilinear


# Extinction coefficient, calibrated so the plume core reaches an optical depth of
# roughly 3 -- opaque at the centre, semi-transparent at the edges, which is what a
# visible-band camera records of a real wildfire plume. The value matters: at the
# unphysical tau ~ 3000 the renderer produces a pure silhouette, which still carries
# the plume's geometry (and hence the wind) but destroys all information about the
# emission strength Q. See scripts/_probe_tau.py for the calibration.
DEFAULT_EXTINCTION = 1.0e-3


@dataclass
class Camera:
    """A pinhole camera looking at the fire from a fixed ground position."""

    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_deg: float = 40.0
    height: int = 64
    width: int = 64
    near: float = 50.0
    far: float = 4000.0
    n_samples: int = 96

    def rays(self, device: torch.device | str, dtype: torch.dtype = torch.float32):
        """Return ray origin ``(3,)`` and per-pixel directions ``(H, W, 3)``."""
        cam = torch.tensor(self.position, device=device, dtype=dtype)
        tgt = torch.tensor(self.look_at, device=device, dtype=dtype)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

        fwd = tgt - cam
        fwd = fwd / fwd.norm()
        right = torch.cross(fwd, world_up, dim=0)
        right = right / right.norm()
        up = torch.cross(right, fwd, dim=0)

        aspect = self.width / self.height
        tan_half = math.tan(math.radians(self.fov_deg) * 0.5)
        ys = torch.linspace(1.0, -1.0, self.height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, self.width, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        dirs = (
            fwd[None, None, :]
            + (gx * tan_half * aspect)[..., None] * right[None, None, :]
            + (gy * tan_half)[..., None] * up[None, None, :]
        )
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        return cam, dirs


def sample_density(
    rho: torch.Tensor,
    pts: torch.Tensor,
    dx: float,
    dz: float,
) -> torch.Tensor:
    """Trilinearly sample ``rho`` ``(nz, ny, nx)`` at world points ``pts`` ``(..., 3)``."""
    return trilinear(rho, pts[..., 0], pts[..., 1], pts[..., 2], dx, dz)


def render(
    rho: torch.Tensor,
    camera: Camera,
    dx: float,
    dz: float,
    extinction: float = DEFAULT_EXTINCTION,
    smoke_value: float = 0.92,
    background_value: float = 0.35,
) -> torch.Tensor:
    """Render the plume to a grayscale image ``(H, W)`` in [0, 1].

    Emission-absorption rendering: bright smoke against a darker background, which
    is what a visible-band tower or UAV camera records when it looks at a plume
    over terrain.
    """
    origin, dirs = camera.rays(rho.device, rho.dtype)
    ts = torch.linspace(
        camera.near, camera.far, camera.n_samples, device=rho.device, dtype=rho.dtype
    )
    delta = (camera.far - camera.near) / camera.n_samples

    pts = origin[None, None, None, :] + ts[None, None, :, None] * dirs[:, :, None, :]
    sigma = extinction * torch.clamp(sample_density(rho, pts, dx, dz), min=0.0)

    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[..., :-1]
    weights = trans * alpha
    acc = weights.sum(dim=-1)
    return acc * smoke_value + (1.0 - acc) * background_value
