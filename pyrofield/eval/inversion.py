"""Levenberg-Marquardt inversion of the physical state from a chosen sensor subset.

LM is the right tool for a six-parameter nonlinear least-squares problem, and it
has a property worth more than speed here: the matrix it inverts each iteration,
``Jw^T Jw``, *is* the Gauss-Newton approximation to the Fisher information from
:mod:`pyrofield.theory.identifiability`. Theory and practice therefore run on the
same object -- when the analysis says a direction is unidentifiable, LM visibly
refuses to move along it, rather than wandering the way a first-order optimiser
would and leaving us unable to tell a confound from a bad learning rate.

The plume term is fitted coarse-to-fine. That is not a trick to flatter the
results: a rendered plume is a sharp structure in image space, so from a start
45% away the predicted and observed plumes barely overlap and the raw residual is
non-convex. Blurring first and sharpening later is the standard remedy for
image-based inverse problems, and without it the optimiser fails to reach an
optimum the Fisher analysis says is well determined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from pyrofield.sim.synthetic import (
    LOG_PARAMS,
    PARAM_NAMES,
    Scenario,
    forward,
    noise_sigma_for,
)
from pyrofield.theory.identifiability import jacobian_fd, stack_obs

# Physical bounds in working-space units (log for positive parameters).
BOUNDS = {
    "R0": (np.log(0.01), np.log(1.0)),
    "U": (np.log(0.2), np.log(20.0)),
    "theta_w": (-2.0 * np.pi, 2.0 * np.pi),
    "lb_k": (np.log(0.02), np.log(3.0)),
    "w_buoy": (np.log(0.2), np.log(20.0)),
    "Q": (np.log(0.02), np.log(50.0)),
}

# Coarse-to-fine ladder: (plume blur sigma in pixels, LM iterations at that level).
DEFAULT_SCHEDULE = ((4.0, 8), (2.0, 6), (1.0, 5), (0.0, 14))


def clip_to_bounds(theta: torch.Tensor) -> torch.Tensor:
    out = theta.clone()
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[name]
        out[i] = out[i].clamp(lo, hi)
    return out


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two axes of a ``(T, H, W)`` stack."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    v = x[:, None]
    v = F.conv2d(F.pad(v, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    v = F.conv2d(F.pad(v, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return v[:, 0]


def make_blur(sigma: float):
    """An observation transform that blurs only the plume image."""

    def f(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if sigma <= 0 or "plume" not in obs or obs["plume"].numel() == 0:
            return obs
        out = dict(obs)
        out["plume"] = gaussian_blur(obs["plume"], sigma)
        return out

    return f


def fixed_weights(
    target: dict[str, torch.Tensor], keys: tuple[str, ...], sigmas: dict[str, float]
) -> torch.Tensor:
    """Per-element ``1 / sigma``, fixed by the sensors rather than by the current fit."""
    parts = []
    for k in keys:
        n = target[k].numel()
        parts.append(
            torch.full((n,), 1.0 / sigmas[k], device=target[k].device, dtype=target[k].dtype)
        )
    return torch.cat(parts)


@dataclass
class InversionResult:
    theta_hat: torch.Tensor
    theta_true: torch.Tensor
    loss: float
    n_iter: int
    accepted: int
    history: list[float] = field(default_factory=list)

    def errors(self) -> dict[str, float]:
        """Fractional error for log parameters, absolute (degrees) for the wind direction."""
        out = {}
        for i, (name, is_log) in enumerate(zip(PARAM_NAMES, LOG_PARAMS)):
            d = float(self.theta_hat[i] - self.theta_true[i])
            if is_log:
                out[name] = abs(np.expm1(d))  # |exp(d) - 1| = relative error
            else:
                ang = (d + np.pi) % (2 * np.pi) - np.pi
                out[name] = abs(np.degrees(ang))
        return out


def levenberg_marquardt(
    theta_init: torch.Tensor,
    theta_true: torch.Tensor,
    target: dict[str, torch.Tensor],
    scen: Scenario,
    keys: tuple[str, ...],
    schedule: tuple[tuple[float, int], ...] = DEFAULT_SCHEDULE,
    eps: float = 0.02,
    lam0: float = 1e-2,
    verbose: bool = False,
    free_idx: list[int] | None = None,
) -> InversionResult:
    """Fit the parameter vector to ``target`` using only the modalities in ``keys``.

    ``free_idx`` restricts the fit to a subset of parameters, holding the rest at
    their initial values. That is what the profile-likelihood experiment needs:
    fix the parameter of interest, minimise over the nuisance parameters, and read
    the uncertainty off the resulting misfit curve.
    """
    device, dtype = theta_init.device, theta_init.dtype
    sigmas = {k: noise_sigma_for(target, k) for k in keys}
    w = fixed_weights(target, keys, sigmas)
    free = list(range(theta_init.numel())) if free_idx is None else list(free_idx)

    theta = clip_to_bounds(theta_init.clone())
    lam, accepted, total_iters = lam0, 0, 0
    history: list[float] = []

    for sigma_blur, n_iter in schedule:
        tf = make_blur(sigma_blur)
        y = stack_obs(target, keys, tf) * w

        def loss_at(t):
            return float((((stack_obs(forward(t, scen), keys, tf) * w) - y) ** 2).sum())

        loss = loss_at(theta)
        if not history:
            history.append(loss)

        for _ in range(n_iter):
            total_iters += 1
            J, base = jacobian_fd(theta, scen, keys, eps, mode="forward", transform=tf)
            Jw = (J * w[:, None]).double()[:, free]
            rw = (stack_obs(base, keys, tf) * w - y).double()

            A = Jw.T @ Jw
            g = Jw.T @ rw
            # Damp unobservable directions relative to the observable ones. An absolute
            # floor is not enough: a parameter a modality cannot see at all (smoke terms
            # for a mask) has an exactly zero diagonal, and the solve then returns a huge
            # step that only the bounds contain.
            dA = torch.diagonal(A)
            diagA = torch.diag(dA.clamp(min=float(dA.max()) * 1e-8 + 1e-30))

            improved = False
            for _ in range(8):  # damping back-off within one iteration
                try:
                    delta = torch.linalg.solve(A + lam * diagA, -g)
                except RuntimeError:
                    lam *= 10.0
                    continue
                step = torch.zeros_like(theta)
                step[free] = delta.to(dtype)
                cand = clip_to_bounds(theta + step)
                loss_c = loss_at(cand)
                if np.isfinite(loss_c) and loss_c < loss:
                    theta, loss = cand, loss_c
                    lam = max(lam / 3.0, 1e-9)
                    improved = True
                    accepted += 1
                    break
                lam *= 3.0
            history.append(loss)
            if verbose:
                print(f"      blur {sigma_blur:.1f}  loss {loss:.4e}  lam {lam:.2e} "
                      f"{'+' if improved else '.'}")
            if not improved and lam > 1e6:
                lam = lam0
                break
            if len(history) > 4 and history[-5] > 0:
                rel = (history[-5] - history[-1]) / abs(history[-5])
                if 0 <= rel < 1e-5:
                    break

    # Report the misfit at full resolution, whatever ladder was used to get there.
    final = float((((stack_obs(forward(theta, scen), keys) * w)
                    - stack_obs(target, keys) * w) ** 2).sum())
    return InversionResult(theta, theta_true, final, total_iters, accepted, history)


def random_start(
    theta_true: torch.Tensor, rng: np.random.Generator, scale: float = 0.45
) -> torch.Tensor:
    """A perturbed starting point: +/-45% on positive parameters, +/-26 deg on wind."""
    out = theta_true.clone()
    for i, _ in enumerate(LOG_PARAMS):
        out[i] = out[i] + float(rng.uniform(-scale, scale))
    return clip_to_bounds(out)
