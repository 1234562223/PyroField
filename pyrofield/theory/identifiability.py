"""Identifiability analysis of heterogeneous fire observations (Propositions P1-P3).

The question this module answers is not "how accurate is my model" but the prior
one: *given this set of sensors, is the physical state even determined?* For a
forward map ``g(theta)`` observed with independent Gaussian noise, the Fisher
information matrix

    F = sum_i (1 / sigma_i^2) (dg_i/dtheta)(dg_i/dtheta)^T

bounds any unbiased estimator through the Cramer-Rao inequality
``cov(theta_hat) >= F^-1``. A near-null eigenvector of ``F`` is a direction in
parameter space that the sensors cannot see at all -- a confound, not merely a
hard-to-estimate quantity.

Because parameters are carried in log space, eigenvector components read directly
as *fractional* trade-offs: an eigenvector ``(+1, -0.6, 0, ...)`` with tiny
eigenvalue says "raise R0 by 1%, drop U by 0.6%, and no sensor in this set will
notice".

Jacobians are taken by central finite differences. That is deliberate: the ENO
minmod limiter and the semi-Lagrangian backtrace both contain kinks where
autograd returns a valid but locally non-representative subgradient, while a
finite difference over a small physical perturbation measures the quantity the
Cramer-Rao bound is actually about. ``check_jacobian_against_autograd`` verifies
the two agree where the map is smooth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, forward, noise_sigma_for


def stack_obs(
    obs: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    transform=None,
) -> torch.Tensor:
    """Flatten the selected modalities into one observation vector.

    ``transform`` optionally rewrites the observation dict first; the inversion
    uses it for coarse-to-fine image blurring.
    """
    if transform is not None:
        obs = transform(obs)
    return torch.cat([obs[k].reshape(-1) for k in keys])


def weight_vector(
    obs: dict[str, torch.Tensor], keys: tuple[str, ...], device, dtype
) -> torch.Tensor:
    """Per-element ``1 / sigma`` for the stacked observation vector."""
    parts = []
    for k in keys:
        n = obs[k].numel()
        s = noise_sigma_for(obs, k)
        parts.append(torch.full((n,), 1.0 / s, device=device, dtype=dtype))
    return torch.cat(parts)


def jacobian_fd(
    theta: torch.Tensor,
    scen: Scenario,
    keys: tuple[str, ...],
    eps: float = 0.02,
    mode: str = "central",
    transform=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Finite-difference Jacobian ``(n_obs, n_params)`` and the base observation.

    ``mode='central'`` (2n forward runs, second-order accurate) is used for every
    reported Fisher analysis. ``mode='forward'`` (n runs) is used inside the LM
    loop, where only a descent direction is needed and halving the cost matters.
    """
    base = forward(theta, scen)
    g0 = stack_obs(base, keys, transform)
    cols = []
    for j in range(theta.numel()):
        tp = theta.clone()
        tp[j] += eps
        gp = stack_obs(forward(tp, scen), keys, transform)
        if mode == "central":
            tm = theta.clone()
            tm[j] -= eps
            gm = stack_obs(forward(tm, scen), keys, transform)
            cols.append((gp - gm) / (2.0 * eps))
        else:
            cols.append((gp - g0) / eps)
    return torch.stack(cols, dim=1), base


def jacobian_ad(
    theta: torch.Tensor,
    scen: Scenario,
    keys: tuple[str, ...],
    transform=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact Jacobian of the discretised model by forward-mode AD.

    One ``jvp`` per parameter, so six forward passes. This is the preferred estimator
    for every reported Fisher matrix: it has no step size, and the fire mask is sharp
    enough that finite differences of it are step-size dependent (see
    ``scripts/test_invariants.py``).
    """
    base = forward(theta, scen)

    def f(t):
        return stack_obs(forward(t, scen), keys, transform)

    cols = []
    for j in range(theta.numel()):
        v = torch.zeros_like(theta)
        v[j] = 1.0
        _, jv = torch.func.jvp(f, (theta,), (v,))
        cols.append(jv)
    return torch.stack(cols, dim=1), base


def jacobian(
    theta: torch.Tensor,
    scen: Scenario,
    keys: tuple[str, ...],
    eps: float = 0.02,
    method: str = "ad",
    transform=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Jacobian by forward-mode AD (default) or by finite differences."""
    if method == "ad":
        return jacobian_ad(theta, scen, keys, transform)
    return jacobian_fd(theta, scen, keys, eps, mode=method, transform=transform)


def fisher_information(
    theta: torch.Tensor,
    scen: Scenario,
    keys: tuple[str, ...],
    eps: float = 0.02,
) -> torch.Tensor:
    """Fisher information matrix ``(n_params, n_params)`` for one sensor subset."""
    J, base = jacobian_fd(theta, scen, keys, eps)
    w = weight_vector(base, keys, theta.device, theta.dtype)
    Jw = J * w[:, None]
    return (Jw.T @ Jw).double()


@dataclass
class IdentifiabilityReport:
    """Eigen-structure of one sensor subset's Fisher information."""

    name: str
    keys: tuple[str, ...]
    eigvals: np.ndarray  # ascending
    eigvecs: np.ndarray  # columns match eigvals
    crb_std: np.ndarray  # marginal Cramer-Rao standard deviations
    cond: float
    n_obs: int
    param_names: list[str] = field(default_factory=lambda: list(PARAM_NAMES))

    def weakest_direction(self) -> np.ndarray:
        return self.eigvecs[:, 0]

    def describe_weakest(self, top_k: int = 3) -> str:
        v = self.weakest_direction()
        order = np.argsort(-np.abs(v))[:top_k]
        terms = [f"{v[i]:+.2f}*{self.param_names[i]}" for i in order]
        return "  ".join(terms)

    def crb_of(self, name: str) -> float:
        return float(self.crb_std[self.param_names.index(name)])


def fisher_submatrix(
    theta: torch.Tensor,
    scen: Scenario,
    keys: tuple[str, ...],
    param_idx: list[int] | None,
    eps: float = 0.02,
    method: str = "ad",
) -> torch.Tensor:
    """Fisher information restricted to a subset of parameters.

    Restricting columns gives the *conditional* Fisher information: the information
    about those parameters when the others are known. That is the right object for
    asking "do masks confound fuel with wind speed?", because parameters a modality
    physically cannot see (smoke terms, for a mask) would otherwise dominate the
    null space and hide the confound we care about.
    """
    J, base = jacobian(theta, scen, keys, eps, method=method)
    w = weight_vector(base, keys, theta.device, theta.dtype)
    Jw = J * w[:, None]
    if param_idx is not None:
        Jw = Jw[:, param_idx]
    return (Jw.T @ Jw).double()


def analyze(
    theta: torch.Tensor,
    scen: Scenario,
    name: str,
    keys: tuple[str, ...],
    eps: float = 0.02,
    ridge_rel: float = 1e-10,
    params: list[str] | None = None,
    method: str = "central",
) -> IdentifiabilityReport:
    """Compute and diagonalise the Fisher information for one sensor subset.

    ``params`` optionally restricts the analysis to a subset of parameter names
    (see :func:`fisher_submatrix`).

    ``method`` defaults to central finite differences rather than AD. For the mask-only
    case the AD Fisher is demonstrably invalid -- it is non-monotone in the amount of
    data (see RESULTS.md section 5), which no Fisher matrix can be -- and finite
    differences additionally happen to be the estimator most favourable to the mask
    baseline, which keeps every comparison against it conservative.
    """
    names = list(PARAM_NAMES) if params is None else list(params)
    idx = None if params is None else [PARAM_NAMES.index(p) for p in names]
    F = fisher_submatrix(theta, scen, keys, idx, eps, method=method)
    Fn = F.cpu().numpy()
    n = Fn.shape[0]

    # Tiny ridge purely for numerical inversion; it is 10 orders below the scale of
    # F, so it cannot manufacture identifiability that is not there.
    ridge = ridge_rel * max(np.trace(Fn) / n, 1e-300)
    eigvals, eigvecs = np.linalg.eigh(Fn + ridge * np.eye(n))
    eigvals = np.clip(eigvals, ridge, None)

    Finv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    crb = np.sqrt(np.clip(np.diag(Finv), 0.0, None))
    cond = float(eigvals[-1] / eigvals[0])

    base = forward(theta, scen)
    n_obs = int(sum(base[k].numel() for k in keys))
    return IdentifiabilityReport(name, keys, eigvals, eigvecs, crb, cond, n_obs, names)


ALL_MODALITIES = ("mask", "plume", "conc")


def all_subsets(modalities: tuple[str, ...] = ALL_MODALITIES) -> list[tuple[str, ...]]:
    """Every non-empty modality subset, ordered by size then by the canonical order."""
    out = []
    for r in range(1, len(modalities) + 1):
        for i in range(1 << len(modalities)):
            sel = tuple(m for j, m in enumerate(modalities) if i >> j & 1)
            if len(sel) == r and sel not in out:
                out.append(sel)
    return out


def fisher_all_subsets(
    theta: torch.Tensor,
    scen: Scenario,
    eps: float = 0.02,
    modalities: tuple[str, ...] = ALL_MODALITIES,
    params: list[str] | None = None,
    method: str = "ad",
) -> dict[tuple[str, ...], np.ndarray]:
    """Fisher information for every modality subset, from a single Jacobian.

    The expensive part is the forward model, and the Jacobian rows for one modality
    do not depend on which other modalities are being fused. Computing the full
    Jacobian once and then selecting row blocks makes the whole 7-subset lattice
    cost the same as a single subset, which is what makes the cross-scenario sweep
    in ``scripts/sweep_identifiability.py`` affordable.
    """
    J, base = jacobian(theta, scen, modalities, eps, method=method)
    idx = None if params is None else [PARAM_NAMES.index(p) for p in params]

    offsets, start = {}, 0
    for m in modalities:
        n = base[m].numel()
        offsets[m] = (start, start + n)
        start += n

    out = {}
    for sel in all_subsets(modalities):
        rows = torch.cat([torch.arange(*offsets[m], device=J.device) for m in sel])
        w = weight_vector(base, sel, theta.device, theta.dtype)
        Jw = J[rows] * w[:, None]
        if idx is not None:
            Jw = Jw[:, idx]
        out[sel] = (Jw.T @ Jw).double().cpu().numpy()
    return out


def crb_from_fisher(F: np.ndarray, ridge_rel: float = 1e-10) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Cramer-Rao standard deviations, condition number and eigen-decomposition."""
    n = F.shape[0]
    ridge = ridge_rel * max(np.trace(F) / n, 1e-300)
    eigvals, eigvecs = np.linalg.eigh(F + ridge * np.eye(n))
    eigvals = np.clip(eigvals, ridge, None)
    Finv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    crb = np.sqrt(np.clip(np.diag(Finv), 0.0, None))
    return crb, float(eigvals[-1] / eigvals[0]), eigvals, eigvecs


def check_jacobian_against_autograd(
    theta: torch.Tensor, scen: Scenario, keys: tuple[str, ...], eps: float = 0.02
) -> dict[str, float]:
    """Sanity check: FD and autograd must agree on a smooth scalar summary.

    Compares ``d/dtheta sum(g(theta))``, which is smooth even where individual
    elements sit on a limiter kink.
    """
    t = theta.clone().requires_grad_(True)
    s = stack_obs(forward(t, scen), keys).sum()
    (grad_ad,) = torch.autograd.grad(s, t)

    grad_fd = torch.zeros_like(theta)
    for j in range(theta.numel()):
        tp, tm = theta.clone(), theta.clone()
        tp[j] += eps
        tm[j] -= eps
        grad_fd[j] = (
            stack_obs(forward(tp, scen), keys).sum() - stack_obs(forward(tm, scen), keys).sum()
        ) / (2 * eps)

    denom = torch.maximum(grad_ad.abs(), grad_fd.abs()).clamp(min=1e-8)
    rel = ((grad_ad - grad_fd).abs() / denom).cpu().numpy()
    return {PARAM_NAMES[i]: float(rel[i]) for i in range(len(PARAM_NAMES))}


def loss_landscape_2d(
    theta: torch.Tensor,
    scen: Scenario,
    target: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    i: int,
    j: int,
    span: float = 0.6,
    n: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan the noise-weighted misfit over two parameters, holding the rest fixed.

    Returns ``(xs, ys, Z)`` with offsets in working-space units (log-ratio for the
    positive parameters). A flat valley in ``Z`` is a confound made visible.
    """
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    sig = {k: noise_sigma_for(target, k) for k in keys}
    Z = np.zeros((n, n))
    for bi, dy in enumerate(ys):
        for ai, dxv in enumerate(xs):
            t = theta.clone()
            t[i] += float(dxv)
            t[j] += float(dy)
            pred = forward(t, scen)
            val = 0.0
            for k in keys:
                val += float(((pred[k] - target[k]) ** 2).sum() / (sig[k] ** 2))
            Z[bi, ai] = val
    return xs, ys, Z
