"""Differentiable elliptical fire spread rate (Rothermel-style wind/slope factors).

The rate of spread (ROS) of a wildfire front in direction ``theta_n`` is modelled
with the classical elliptical template (Anderson 1983): the head fire runs at
``R_head`` in the wind direction, and the front degenerates to an ellipse whose
eccentricity grows with wind speed.

    R_head = R0 * (1 + phi_w(U) + phi_s(slope))
    R(theta) = R_head * (1 - e) / (1 - e * cos(theta - theta_w))

Every function here is differentiable w.r.t. both the state (``theta_n``) and the
physical parameters (``R0``, ``U``, ``theta_w``), which is what makes the whole
inversion in :mod:`pyrofield.sim.synthetic` possible.

The confounding that Proposition P1 is about lives in ``R_head``: the observable
spread rate only ever sees the *product* ``R0 * (1 + phi_w(U))``, so ``R0`` (fuel
and moisture) and ``U`` (wind) cannot be told apart from front geometry alone.
"""

from __future__ import annotations

import torch

# Default coefficients. These are in the range reported for grass/shrub fuels;
# the exact values do not matter for the identifiability argument, only the
# functional form (R0 and U entering multiplicatively) does.
WIND_A = 0.35
WIND_B = 1.3
SLOPE_A = 5.275
SLOPE_B = 2.0
LB_K = 0.5


def wind_factor(U: torch.Tensor, a: float = WIND_A, b: float = WIND_B) -> torch.Tensor:
    """Rothermel wind coefficient ``phi_w``. ``U`` is midflame wind speed in m/s."""
    return a * torch.clamp(U, min=0.0) ** b


def slope_factor(slope_tan: torch.Tensor, a: float = SLOPE_A, b: float = SLOPE_B) -> torch.Tensor:
    """Rothermel slope coefficient ``phi_s``. ``slope_tan`` is tan of the slope angle."""
    return a * torch.clamp(slope_tan, min=0.0) ** b


# Anderson (1983) length-to-breadth ratio, in the form FARSITE and FlamMap actually use:
# the published relation with 0.397 subtracted so that LB = 1 at zero wind, and the whole
# thing truncated at 8 (the maximum in the empirical data Alexander 1985 collects).
# Wind speed enters in mi/h.
#   LB = 0.936 exp(0.2566 U) + 0.461 exp(-0.1548 U) - 0.397
# This matters more than a coefficient choice usually does. The relation carries **no free
# fuel parameter** -- it is stated as applying to any fuel type -- so in principle the burn
# shape alone determines the wind speed. But it reaches the cap at U = 3.79 m/s, and above
# that the shape is clamped and carries nothing. See scripts/farsite_lb.py.
MS_PER_MIH = 0.44704
LB_CAP = 8.0


def lb_anderson(
    U: torch.Tensor,
    cap: float | None = LB_CAP,
    scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """FARSITE's length-to-breadth ratio. ``U`` in m/s.

    ``scale`` multiplies the excess over 1, representing the fuel-to-fuel spread in the
    relation that FARSITE's single curve papers over (Anderson reports group-specific
    relations; Alexander 1985 shows the scatter).
    """
    u_mih = torch.clamp(U, min=0.0) / MS_PER_MIH
    lb = 0.936 * torch.exp(0.2566 * u_mih) + 0.461 * torch.exp(-0.1548 * u_mih) - 0.397
    lb = 1.0 + scale * (lb - 1.0)
    if cap is not None:
        lb = torch.clamp(lb, max=cap)
    return torch.clamp(lb, min=1.0)


def lb_saturation_wind(cap: float = LB_CAP, scale: float = 1.0) -> float:
    """Wind speed (m/s) at which the Anderson relation reaches the cap."""
    lo, hi = 0.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(lb_anderson(torch.tensor(mid), cap=None, scale=scale)) < cap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def length_to_breadth(U: torch.Tensor, k: torch.Tensor | float = LB_K) -> torch.Tensor:
    """Length-to-breadth ratio of the fire ellipse; 1 (circle) at zero wind.

    ``k`` is fuel-dependent -- Anderson's length-to-breadth relation is published
    per fuel model -- so in practice it is *unknown*, and the fire shape observed
    in a mask constrains only the product ``k * U``. That second confounding sits
    alongside the ``R0 * (1 + phi_w(U))`` one and is why mask-only inversion is
    under-determined rather than merely ill-conditioned.
    """
    return 1.0 + k * torch.clamp(U, min=0.0)


def eccentricity(LB: torch.Tensor) -> torch.Tensor:
    """Eccentricity of the fire ellipse from its length-to-breadth ratio."""
    return torch.sqrt(torch.clamp(LB**2 - 1.0, min=0.0)) / LB


def head_ros(
    R0: torch.Tensor,
    U: torch.Tensor,
    slope_tan: torch.Tensor | float = 0.0,
    wind_b: float = WIND_B,
) -> torch.Tensor:
    """Head-fire rate of spread.

    This single expression is the whole of the mask degeneracy: ``R0`` and ``U`` are
    only ever observed through this product, so front geometry alone identifies
    ``head_ros`` but never its factors. Note that the degeneracy does not depend on the
    *form* of the wind factor -- any ``R0 * g(U)`` has it -- which is why ``wind_b`` is
    exposed: ``scripts/misspecification.py`` fits with the wrong exponent to check that
    the conclusions are structural rather than an artefact of this particular g.
    """
    if not torch.is_tensor(slope_tan):
        slope_tan = torch.as_tensor(slope_tan, dtype=R0.dtype, device=R0.device)
    return R0 * (1.0 + wind_factor(U, b=wind_b) + slope_factor(slope_tan))


def back_ros(R_head: torch.Tensor, ecc: torch.Tensor) -> torch.Tensor:
    """Backing-fire rate of spread implied by the ellipse eccentricity."""
    return R_head * (1.0 - ecc) / (1.0 + ecc)


def combine_wind_slope(
    U: torch.Tensor,
    theta_w: torch.Tensor,
    phi_s: torch.Tensor | float = 0.0,
    slope_aspect: torch.Tensor | float = 0.0,
    wind_b: float = WIND_B,
):
    """Add the wind and slope forcing as vectors, the way FARSITE and Rothermel do.

    The wind factor and the slope factor are not separate influences on the fire: they add
    as vectors, and the fire responds only to their sum. That has a consequence this
    project cares about a great deal. On flat ground a burn scar points downwind, so a
    mask gives the wind direction away. On a slope it points along the *combined* vector,
    and a mask can no longer separate the two -- so in terrain, masks lose the one thing
    they were reliably good for.

    Returns ``(phi_eff, theta_eff, U_eff)``: the combined forcing magnitude, its direction,
    and the wind speed that would produce that magnitude on flat ground (which is what the
    length-to-breadth relation is written in terms of).
    """
    phi_w = wind_factor(U, b=wind_b)
    if not torch.is_tensor(phi_s):
        phi_s = torch.as_tensor(phi_s, dtype=U.dtype, device=U.device)
    if not torch.is_tensor(slope_aspect):
        slope_aspect = torch.as_tensor(slope_aspect, dtype=U.dtype, device=U.device)

    vx = phi_w * torch.cos(theta_w) + phi_s * torch.cos(slope_aspect)
    vy = phi_w * torch.sin(theta_w) + phi_s * torch.sin(slope_aspect)
    phi_eff = torch.sqrt(vx * vx + vy * vy + 1e-18)
    theta_eff = torch.atan2(vy, vx)
    U_eff = (torch.clamp(phi_eff, min=1e-9) / WIND_A) ** (1.0 / wind_b)
    return phi_eff, theta_eff, U_eff


def ellipse_axes(
    R0: torch.Tensor,
    U: torch.Tensor,
    lb_k: torch.Tensor | float = LB_K,
    slope_tan: torch.Tensor | float = 0.0,
    wind_b: float = WIND_B,
):
    """Semi-major ``a``, semi-minor ``b`` and focal offset ``c`` of the unit-time ellipse."""
    R_head = head_ros(R0, U, slope_tan, wind_b=wind_b)
    LB = length_to_breadth(U, lb_k)
    ecc = eccentricity(LB)
    R_back = back_ros(R_head, ecc)
    a = 0.5 * (R_head + R_back)
    c = 0.5 * (R_head - R_back)
    b = a / LB
    return a, b, c


def ros_with_slope(
    theta_n: torch.Tensor,
    R0: torch.Tensor,
    U: torch.Tensor,
    theta_w: torch.Tensor,
    lb_k: torch.Tensor | float,
    phi_s: torch.Tensor | float,
    slope_aspect: torch.Tensor | float,
    wind_b: float = WIND_B,
) -> torch.Tensor:
    """Normal-direction spread rate on sloping ground.

    Wind and slope are combined into one effective forcing vector
    (:func:`combine_wind_slope`), and the elliptical template is then built on that vector
    exactly as on flat ground. Setting ``phi_s = 0`` recovers :func:`ros_elliptical`.
    """
    phi_eff, theta_eff, U_eff = combine_wind_slope(U, theta_w, phi_s, slope_aspect, wind_b)
    R_head = R0 * (1.0 + phi_eff)
    LB = length_to_breadth(U_eff, lb_k)
    ecc = eccentricity(LB)
    R_back = back_ros(R_head, ecc)
    a = 0.5 * (R_head + R_back)
    c = 0.5 * (R_head - R_back)
    b = a / LB
    d = theta_n - theta_eff
    cos_d, sin_d = torch.cos(d), torch.sin(d)
    return c * cos_d + torch.sqrt((a * cos_d) ** 2 + (b * sin_d) ** 2 + 1e-18)


def ros_elliptical(
    theta_n: torch.Tensor,
    R0: torch.Tensor,
    U: torch.Tensor,
    theta_w: torch.Tensor,
    lb_k: torch.Tensor | float = LB_K,
    slope_tan: torch.Tensor | float = 0.0,
    wind_b: float = WIND_B,
    lb_law: str = "linear",
) -> torch.Tensor:
    """Normal-direction spread rate for level-set propagation.

    ``lb_law='linear'`` uses ``LB = 1 + lb_k*U``, a shape law with a free fuel-dependent
    coefficient. ``lb_law='anderson'`` uses FARSITE's actual relation, which has no free
    parameter but saturates at LB = 8; ``lb_k`` is then reinterpreted as the fuel-to-fuel
    scale factor on the relation, 1.0 being FARSITE's own curve.

    This is the **support function** of the elliptical Huygens wavelet, not its
    radius. The distinction matters and is easy to get wrong: propagating a front
    with the wavelet's *radial* distance as the normal speed starves the flanks
    and under-advances the head (verified in ``scripts/validate_physics.py``,
    where the radial form reaches only ~41% of the analytic head distance).
    Huygens' principle is reproduced only by the support function

        F(n) = c cos(d) + sqrt(a^2 cos^2(d) + b^2 sin^2(d)),   d = theta_n - theta_w

    which returns ``R_head`` at the head, ``R_back`` at the back and ``b`` on the
    flanks, as it must.

    Args:
        theta_n: outward normal angle of the front, any shape.
        R0: no-wind, no-slope rate of spread (m/s), scalar tensor.
        U: midflame wind speed (m/s), scalar tensor.
        theta_w: wind direction (radians, the direction the wind blows *towards*).
        lb_k: fuel-dependent length-to-breadth coefficient.
        slope_tan: terrain slope, scalar or same shape as ``theta_n``.

    Returns:
        Tensor of spread rates (m/s), broadcast to the shape of ``theta_n``.
    """
    if lb_law == "anderson":
        R_head = head_ros(R0, U, slope_tan, wind_b=wind_b)
        LB = lb_anderson(U, scale=lb_k)
        ecc = eccentricity(LB)
        R_back = back_ros(R_head, ecc)
        a = 0.5 * (R_head + R_back)
        c = 0.5 * (R_head - R_back)
        b = a / LB
    else:
        a, b, c = ellipse_axes(R0, U, lb_k, slope_tan, wind_b=wind_b)
    d = theta_n - theta_w
    cos_d, sin_d = torch.cos(d), torch.sin(d)
    return c * cos_d + torch.sqrt((a * cos_d) ** 2 + (b * sin_d) ** 2 + 1e-18)
