"""The full differentiable forward model: physical state -> heterogeneous observations.

This is PyroField's central object. A single physical parameter vector drives a
level-set fire front and a 3-D smoke field, and every sensor is then a *projection*
of that same state through its own differentiable operator:

    theta --> S(x, y, t) --> { g_mask(S), g_plume(S), g_conc(S) }

Because all three operators read from one state, fusing them is inversion rather
than feature concatenation, and because the map is differentiable we can ask the
identifiability question directly: which parameter combinations does a given
sensor subset actually pin down?

Parameters are carried in log space (except the wind direction), so the Fisher
analysis in :mod:`pyrofield.theory.identifiability` speaks in dimensionless
fractional changes and its eigenvectors are directly readable as confounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from pyrofield.models.operators import gaussian_plume, mask as mask_op
from pyrofield.models.operators.plume_render import Camera, render
from pyrofield.physics import levelset, smoke
from pyrofield.physics.rothermel import ros_elliptical, ros_with_slope

# Parameter vector layout. Positive quantities live in log space.
PARAM_NAMES = ["R0", "U", "theta_w", "lb_k", "w_buoy", "Q"]
LOG_PARAMS = [True, True, False, True, True, True]
PARAM_LATEX = [r"$R_0$", r"$U$", r"$\theta_w$", r"$k_{LB}$", r"$w_b$", r"$Q$"]


def pack(values: dict[str, float], device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Physical values -> the (6,) working vector used everywhere else."""
    out = []
    for name, is_log in zip(PARAM_NAMES, LOG_PARAMS):
        v = float(values[name])
        out.append(math.log(v) if is_log else v)
    return torch.tensor(out, device=device, dtype=dtype)


def unpack(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Working vector -> physical values."""
    out = {}
    for i, (name, is_log) in enumerate(zip(PARAM_NAMES, LOG_PARAMS)):
        out[name] = torch.exp(theta[i]) if is_log else theta[i]
    return out


@dataclass
class Scenario:
    """Grid, timing and sensor placement for one synthetic fire."""

    # Grid and timing validated in scripts/validate_physics.py: region IoU 0.986
    # against the exact Huygens-Minkowski burn, 0.00% grid anisotropy.
    nx: int = 96
    ny: int = 96
    nz: int = 24
    dx: float = 15.0
    dz: float = 30.0
    dt: float = 15.0
    n_steps: int = 110
    ignition_xy: tuple[float, float] = (400.0, 400.0)
    ignition_radius: float = 45.0
    reinit_every: int = 10
    reinit_iters: int = 3

    mask_times: tuple[int, ...] = (27, 54, 81, 109)
    plume_times: tuple[int, ...] = (54, 109)
    conc_times: tuple[int, ...] = (27, 54, 81, 109)
    receptors: tuple[tuple[float, float, float], ...] = (
        (950.0, 950.0, 2.0),
        (1100.0, 750.0, 2.0),
        (750.0, 1100.0, 2.0),
    )
    camera: Camera = field(
        default_factory=lambda: Camera(
            position=(1350.0, 150.0, 60.0),
            look_at=(620.0, 620.0, 260.0),
            fov_deg=48.0,
            height=56,
            width=56,
            near=80.0,
            far=2600.0,
            n_samples=96,
        )
    )

    # Calibrated so the plume core sits near optical depth 3 (see plume_render).
    plume_extinction: float = 1.0e-3

    # Spatially varying fuel, as a multiplier on R0 with shape ``(ny, nx)``. The claim it
    # exists to test is that heterogeneity does not break the mask degeneracy: rescaling
    # the whole field by a constant *is* the null direction, so a known field shape with
    # an unknown overall scale leaves the confound exactly where it was.
    fuel_field: torch.Tensor | None = None

    # Terrain. ``slope_phi`` is the Rothermel slope factor (dimensionless, comparable to
    # the wind factor: 2.1 at 4 m/s) and ``slope_aspect`` the upslope direction, which a
    # DEM gives exactly. Wind and slope add as vectors, so on a slope the burn scar points
    # along their sum rather than downwind -- see rothermel.combine_wind_slope.
    slope_phi: float = 0.0
    slope_aspect: float = 0.0

    # Wind-factor exponent of the spread model. Exposed so that data can be generated
    # with one value and fitted with another (scripts/misspecification.py): the mask
    # degeneracy is a property of the *form* R0 * g(U), not of this particular g, and
    # that claim has to be testable.
    wind_b: float = 1.3

    # 'linear' (LB = 1 + k*U, free fuel coefficient) or 'anderson' (FARSITE's own relation,
    # no free coefficient but capped at LB = 8). See scripts/farsite_lb.py.
    lb_law: str = "linear"

    # Optional exogenous wind change: (step_index, U, theta_w) in absolute units.
    # From that step on the fire is driven by this wind instead of the inferred one,
    # which is how forecasting actually works -- a weather model supplies the future
    # wind, while the fuel underneath stays whatever it was. It is also what makes the
    # fuel/wind degeneracy bite: the parameters still set the fuel, so a fit that put
    # the wrong split between fuel and wind gets the new spread rate wrong.
    wind_after: tuple[int, float, float] | None = None

    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def cell_centres(self) -> torch.Tensor:
        """``(ny*nx, 2)`` centres of every grid cell, in metres."""
        ys = (torch.arange(self.ny, device=self.device, dtype=self.dtype) + 0.5) * self.dx
        xs = (torch.arange(self.nx, device=self.device, dtype=self.dtype) + 0.5) * self.dx
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def forward(theta: torch.Tensor, scen: Scenario) -> dict[str, torch.Tensor]:
    """Run the coupled fire + smoke model and return every sensor's observation.

    Returns a dict with keys ``mask`` ``(T_m, ny, nx)``, ``plume`` ``(T_p, H, W)``
    and ``conc`` ``(T_c, n_receptors)``.
    """
    p = unpack(theta)
    R0, U, theta_w = p["R0"], p["U"], p["theta_w"]
    lb_k, w_buoy, Q = p["lb_k"], p["w_buoy"], p["Q"]

    phi = levelset.init_circle(
        scen.ny, scen.nx, scen.dx, scen.ignition_xy, scen.ignition_radius,
        device=scen.device, dtype=scen.dtype,
    )
    rho = torch.zeros(scen.nz, scen.ny, scen.nx, device=scen.device, dtype=scen.dtype)
    centres = scen.cell_centres()

    masks, plumes, concs = [], [], []
    mask_t, plume_t, conc_t = set(scen.mask_times), set(scen.plume_times), set(scen.conc_times)
    # Only the camera reads the 3-D smoke field; the point sensors use the analytic
    # Gaussian plume. Skipping the transport when no image is requested makes the
    # mask-only forward model several times cheaper.
    need_smoke = bool(plume_t)

    shift_at, U_after, tw_after = (scen.wind_after or (None, None, None))

    for t in range(scen.n_steps):
        if shift_at is not None and t >= shift_at:
            U_t = torch.as_tensor(U_after, device=U.device, dtype=U.dtype)
            tw_t = torch.as_tensor(tw_after, device=U.device, dtype=U.dtype)
        else:
            U_t, tw_t = U, theta_w

        theta_n = levelset.normal_angle(phi, scen.dx)
        R0_local = R0 if scen.fuel_field is None else R0 * scen.fuel_field
        if scen.slope_phi:
            speed = ros_with_slope(theta_n, R0_local, U_t, tw_t, lb_k,
                                   scen.slope_phi, scen.slope_aspect,
                                   wind_b=scen.wind_b)
        else:
            speed = ros_elliptical(theta_n, R0_local, U_t, tw_t, lb_k=lb_k,
                                   wind_b=scen.wind_b, lb_law=scen.lb_law)
        front = levelset.front_band(phi, scen.dx)

        if need_smoke:
            rho = smoke.step(rho, front, Q, U_t, tw_t, w_buoy, scen.dt, scen.dx, scen.dz)
        phi = levelset.step(phi, speed, scen.dt, scen.dx)
        if scen.reinit_every and (t + 1) % scen.reinit_every == 0:
            phi = levelset.reinitialize(phi, scen.dx, scen.reinit_iters)

        if t in mask_t:
            masks.append(mask_op.soft_mask(phi, scen.dx))
        if t in plume_t:
            plumes.append(
                render(rho, scen.camera, scen.dx, scen.dz,
                       extinction=scen.plume_extinction)
            )
        if t in conc_t:
            concs.append(
                gaussian_plume.concentration_field(
                    centres, front.reshape(-1), list(scen.receptors), Q, U_t, tw_t
                )
            )

    return {
        "mask": torch.stack(masks) if masks else torch.zeros(0, device=scen.device),
        "plume": torch.stack(plumes) if plumes else torch.zeros(0, device=scen.device),
        "conc": torch.stack(concs) if concs else torch.zeros(0, device=scen.device),
    }


# Observation subsets that the identifiability study compares. The names double as
# the row labels of the sensor-sufficiency diagram (Figure F4).
OBS_SETS: dict[str, tuple[str, ...]] = {
    "M  (mask only)": ("mask",),
    "M+P (mask + plume image)": ("mask", "plume"),
    "M+C (mask + air-quality)": ("mask", "conc"),
    "M+P+C (all three)": ("mask", "plume", "conc"),
}

# Per-modality observation noise, in the natural units of each operator.
NOISE_SIGMA = {"mask": 0.05, "plume": 0.02, "conc": 0.05}

# Air-quality monitors have a limit of detection: below it the reading is noise,
# not signal. Without this floor, a scenario whose plume misses every receptor
# would be credited with infinitely precise near-zero readings and the Fisher
# information would blow up (it produced NaNs before this was added).
CONC_LOD = 2.0e-5


def noise_sigma_for(obs: dict[str, torch.Tensor], key: str) -> float:
    """Absolute noise level for a modality.

    Concentrations get a proportional term plus an absolute detection floor, which
    is how a real PM/CO monitor behaves.
    """
    if key == "conc":
        scale = obs["conc"].abs().mean().item() if obs["conc"].numel() else 0.0
        return max(NOISE_SIGMA["conc"] * scale, CONC_LOD)
    return NOISE_SIGMA[key]


def add_noise(obs: dict[str, torch.Tensor], generator: torch.Generator | None = None):
    """Return a noisy copy of an observation dict."""
    out = {}
    for k, v in obs.items():
        if v.numel() == 0:
            out[k] = v
            continue
        s = noise_sigma_for(obs, k)
        out[k] = v + s * torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
    return out


def residual_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    sigmas: dict[str, float],
) -> torch.Tensor:
    """Noise-weighted least-squares misfit over the selected modalities."""
    total = torch.zeros((), device=pred[keys[0]].device, dtype=pred[keys[0]].dtype)
    for k in keys:
        total = total + ((pred[k] - target[k]) ** 2).mean() / (sigmas[k] ** 2)
    return total
