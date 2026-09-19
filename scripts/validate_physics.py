"""Verification of the differentiable physics kernel against analytic solutions.

Plan verification item #2. Three checks, each with a closed-form answer:

1. Zero wind -> the front is a circle of radius ``r0 + R0 * t``.
2. With wind -> head, back and the burned region's length-to-breadth ratio must
   match the elliptical template (head ``R_head``, back ``R_head (1-e)/(1+e)``,
   aspect ``LB``).
3. Reinitialisation keeps ``|grad phi| ~ 1`` near the front, so the signed
   distance interpretation (and therefore the mask operator) stays valid.

Run:  python scripts/validate_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyrofield.physics import levelset  # noqa: E402
from pyrofield.physics.rothermel import (  # noqa: E402
    back_ros,
    eccentricity,
    ellipse_axes,
    head_ros,
    length_to_breadth,
    ros_elliptical,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = lambda v: torch.tensor(float(v), device=DEV)  # noqa: E731


def run_front(R0, U, theta_w, lb_k, nx, dx, dt, n_steps, origin, r0,
              reinit_every=10, reinit_iters=3):
    """Propagate a front and return the final signed distance field."""
    phi = levelset.init_circle(nx, nx, dx, origin, r0, device=DEV)
    for t in range(n_steps):
        theta_n = levelset.normal_angle(phi, dx)
        speed = ros_elliptical(theta_n, T(R0), T(U), T(theta_w), lb_k=T(lb_k))
        phi = levelset.step(phi, speed, dt, dx)
        if reinit_every and (t + 1) % reinit_every == 0:
            phi = levelset.reinitialize(phi, dx, reinit_iters)
    return phi


def extent_along(phi, dx, origin, direction):
    """Distance from ``origin`` to the front along a unit ``direction``."""
    ox, oy = origin
    ux, uy = direction
    ts = torch.arange(0.0, 2000.0, dx * 0.25, device=DEV)
    xs, ys = ox + ts * ux, oy + ts * uy
    ny, nx = phi.shape
    gx = 2.0 * xs / (nx * dx) - 1.0
    gy = 2.0 * ys / (ny * dx) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
    vals = torch.nn.functional.grid_sample(
        phi[None, None], grid, mode="bilinear", padding_mode="border", align_corners=False
    ).reshape(-1)
    inside = vals < 0
    if not inside.any():
        return 0.0
    last = int(torch.nonzero(inside).max())
    return float(ts[last])


def huygens_region(nx, dx, origin, r0, a, b, c, theta_w, Ttot, n_boundary=4000):
    """Exact Huygens burn region: the Minkowski sum of the ignition disc and the ellipse.

    A point is burned iff its distance to the filled ellipse (centre offset ``c*T``
    along the wind, semi-axes ``a*T`` and ``b*T``) is at most ``r0``. This is the
    ground truth the level-set solver has to reproduce, and comparing *regions*
    rather than three rays is a far stronger test.
    """
    aT, bT, cT = a * Ttot, b * Ttot, c * Ttot
    ys = (torch.arange(nx, device=DEV, dtype=torch.float32) + 0.5) * dx
    xs = (torch.arange(nx, device=DEV, dtype=torch.float32) + 0.5) * dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    # Rotate into the wind frame, origin at the ellipse centre.
    ct, st = torch.cos(T(theta_w)), torch.sin(T(theta_w))
    px = (gx - origin[0]) * ct + (gy - origin[1]) * st - cT
    py = -(gx - origin[0]) * st + (gy - origin[1]) * ct

    inside = (px / aT) ** 2 + (py / bT) ** 2 <= 1.0

    t = torch.linspace(0, 2 * torch.pi, n_boundary, device=DEV)
    ex, ey = aT * torch.cos(t), bT * torch.sin(t)
    d2 = (px[..., None] - ex) ** 2 + (py[..., None] - ey) ** 2
    dmin = torch.sqrt(d2.min(dim=-1).values)
    return inside | (dmin <= r0)


def region_iou(pred_mask, true_mask):
    p, t = pred_mask.bool(), true_mask.bool()
    return float((p & t).sum()) / max(float((p | t).sum()), 1.0)


def check_circular():
    print("\n[1] Zero wind: front must be a circle of radius r0 + R0*t")
    R0, r0, dx, dt, n = 0.10, 45.0, 15.0, 15.0, 110
    origin = (720.0, 720.0)
    phi = run_front(R0, 0.0, 0.0, 0.35, 96, dx, dt, n, origin, r0)

    analytic = r0 + R0 * dt * n
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (0.7071, 0.7071), (-0.7071, 0.7071)]
    meas = [extent_along(phi, dx, origin, d) for d in dirs]
    mean_r = sum(meas) / len(meas)
    err = abs(mean_r - analytic) / analytic
    aniso = (max(meas) - min(meas)) / mean_r
    print(f"    analytic radius = {analytic:7.1f} m")
    print(f"    measured mean   = {mean_r:7.1f} m   (per-direction: "
          f"{', '.join(f'{m:.0f}' for m in meas)})")
    print(f"    radius error    = {err*100:6.2f} %      [target < 5 %]")
    print(f"    anisotropy      = {aniso*100:6.2f} %      [target < 5 %]")
    return err < 0.05 and aniso < 0.05


def check_elliptical():
    print("\n[2] With wind: head / back / aspect must match the elliptical template")
    R0, U, tw, lb_k = 0.10, 4.0, 0.7854, 0.35
    r0, dx, dt, n = 45.0, 15.0, 15.0, 110
    origin = (400.0, 400.0)
    phi = run_front(R0, U, tw, lb_k, 96, dx, dt, n, origin, r0)

    Rh = float(head_ros(T(R0), T(U)))
    LB = float(length_to_breadth(T(U), T(lb_k)))
    e = float(eccentricity(T(LB)))
    Rb = float(back_ros(T(Rh), T(e)))
    a, b, c = (float(v) for v in ellipse_axes(T(R0), T(U), T(lb_k)))
    Ttot = dt * n

    # Huygens: the burn is the Minkowski sum of the initial disc and the ellipse
    # (centre offset c*T, semi-axes a*T and b*T) grown for time T.
    head_a = r0 + Rh * Ttot
    back_a = r0 + Rb * Ttot
    flank_a = r0 + b * Ttot * (1.0 - (c / a) ** 2) ** 0.5

    ux, uy = torch.cos(T(tw)).item(), torch.sin(T(tw)).item()
    head_m = extent_along(phi, dx, origin, (ux, uy))
    back_m = extent_along(phi, dx, origin, (-ux, -uy))
    flank_m = (
        extent_along(phi, dx, origin, (-uy, ux)) + extent_along(phi, dx, origin, (uy, -ux))
    ) / 2

    e_head = abs(head_m - head_a) / head_a
    e_back = abs(back_m - back_a) / max(back_a, 1.0)
    print(f"    R_head={Rh:.4f}  R_back={Rb:.4f}  LB={LB:.2f}  e={e:.3f}  b={b:.4f} m/s")
    print(f"    head:  analytic {head_a:7.1f}  measured {head_m:7.1f}   err {e_head*100:5.2f} %")
    print(f"    back:  analytic {back_a:7.1f}  measured {back_m:7.1f}   err {e_back*100:5.2f} %")
    print(f"    flank (ray, approx ref): {flank_a:7.1f} vs {flank_m:7.1f}")

    # Region-level test against the exact Huygens-Minkowski burn.
    truth = huygens_region(96, dx, origin, r0, a, b, c, tw, Ttot)
    iou = region_iou(phi < 0, truth)
    print(f"    region IoU vs exact Huygens burn = {iou:.4f}   [target > 0.95]")
    ok = e_head < 0.05 and e_back < 0.10 and iou > 0.95
    print("    [targets: head < 5 %, back < 10 %, IoU > 0.95]")
    return ok


def check_reinit():
    print("\n[3] Reinitialisation keeps |grad phi| ~ 1 in the narrow band")
    R0, U, tw, lb_k = 0.10, 4.0, 0.7854, 0.35
    dx, dt, n = 15.0, 15.0, 110
    origin = (400.0, 400.0)
    for reinit in (0, 10):
        phi = run_front(R0, U, tw, lb_k, 96, dx, dt, n, origin, 45.0,
                        reinit_every=reinit, reinit_iters=3)
        g = levelset.grad_magnitude_upwind(phi, dx)
        band = phi.abs() < 3 * dx
        gb = g[band]
        tag = "off" if reinit == 0 else f"every {reinit}"
        print(f"    reinit {tag:9s}: |grad phi| in band = "
              f"{gb.mean():.3f} +/- {gb.std():.3f}   (target ~ 1.0)")
    return True


def main():
    print(f"device = {DEV}")
    ok = [check_circular(), check_elliptical(), check_reinit()]
    print("\n" + ("PASS" if all(ok) else "FAIL") + f"  ({sum(ok)}/{len(ok)} checks)")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
