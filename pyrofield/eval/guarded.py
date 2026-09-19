"""Inversion that refuses to report what the observations do not determine.

The study's own results say what such a guard has to be, and what it cannot be.

Section 12.4 measured two kinds of failure. When the optimiser loses an answer the data
contained, the misfit at the answer it returns is 1.05 to 177 times the misfit at the
truth, so a goodness-of-fit test catches it without ground truth. When the *data* does not
contain the answer, the misfit at a parameter set wrong by 85 % is **1.000** times the
misfit at the truth, to three decimals. No residual test can catch that, because there is
no residual to see.

Section 11 rules out the other obvious guard. A Fisher matrix at the solution misses this
degeneracy entirely -- finite differences report a bound 33x too small and forward-mode AD
one 52x too large -- because the null space is a *curve*, and a Fisher matrix only ever
sees local curvature. Section 1 measured the gap: a 30 % wind error costs dchi2 = 0.20
along the exact curve and 791 along its tangent, a factor of 4000.

What is left is the thing that works: impose a perturbation on one parameter, re-minimise
over all the others, and read the misfit. That is one point of a profile likelihood, it is
derivative-free, and it follows the curve because the re-minimisation does. If the misfit
barely moves, the observations do not determine that parameter, whatever the fit reported.

`guard` returns a verdict per parameter rather than a number: **determined** with an
interval, or **not determined** with the perturbation that cost nothing. An inversion that
reports the second as a point estimate is the failure this whole study is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pyrofield.eval.inversion import levenberg_marquardt
from pyrofield.sim.synthetic import LOG_PARAMS, PARAM_NAMES, Scenario

# One profile point is a short fit, not a full one: the nuisance parameters start from the
# converged solution and only have to relax, so the coarse-to-fine blur ladder the main
# inversion needs is wasted here. Measured on four cases against the full ladder
# ((2.0, 3), (0.0, 5)), which costs 68-83 s per point:
#
#   schedule        mask+plume dchi2      mask-only dchi2     cost
#   ((2,3),(0,5))   5502.1 / 1387.8       0.0 / 0.4           68-83 s
#   ((0,4))         5506.6 / 1387.9       0.1 / 0.7           36-38 s
#   ((0,2))        18215.6 / 1438.2       1.3 / 34.9          19 s
#
# Four iterations agree with the full ladder to 0.1 % and halve the cost. Two do not:
# 34.9 against 0.4 would turn an undetermined parameter into a false "determined".
PROBE_SCHEDULE = ((0.0, 4),)

# Wilks: a 1-sigma interval for one parameter is where the profile rises by 1.
DCHI2_1SIGMA = 1.0
# Below this over the whole probe range, the parameter is not determined at all. Section 1
# measures dchi2 = 0.20 for a fully compensated 30 % wind error, so the floor sits above
# that and well below the 45 427 a determined parameter reaches in section 2.
DCHI2_FLAT = 4.0


@dataclass
class Verdict:
    name: str
    determined: bool
    dchi2: dict[float, float] = field(default_factory=dict)
    sigma_rel: float | None = None       # for log parameters
    sigma_abs: float | None = None       # for angles, in degrees

    def __str__(self):
        if not self.determined:
            worst = max(self.dchi2.values()) if self.dchi2 else float("nan")
            return (f"{self.name}: NOT DETERMINED "
                    f"(largest dchi2 over the probe: {worst:.2f})")
        if self.sigma_rel is not None:
            return f"{self.name}: determined, sigma ~ {self.sigma_rel * 100:.1f} %"
        return f"{self.name}: determined, sigma ~ {self.sigma_abs:.3f} deg"


def profile_point(theta_hat, target, scen, keys, idx, offset, base_loss):
    """Misfit cost of forcing parameter `idx` off by `offset`, others re-minimised."""
    start = theta_hat.clone()
    start[idx] = start[idx] + offset
    free = [j for j in range(theta_hat.numel()) if j != idx]
    res = levenberg_marquardt(start, theta_hat, target, scen, keys,
                              schedule=PROBE_SCHEDULE, free_idx=free)
    return max(res.loss - base_loss, 0.0)


def guard(theta_hat, target, scen, keys, base_loss=None,
          probes=(0.10, 0.30), params=None, verbose=False):
    """Which parameters these observations actually determine, and to what.

    `probes` are relative perturbations for the log parameters and radians for the angle,
    each applied in both directions. The cheaper direction is what counts: a parameter is
    undetermined if it can move *either* way for free.

    The probing is adaptive, because the two questions cost differently. The verdict needs
    only the large probe -- if a 30 % error is free, the parameter is not determined and
    no interval is worth computing. Only a parameter that survives that pays for the small
    probe that turns the profile into a sigma. On the arms this study cares about that
    halves the work, since the undetermined parameters are the common case.
    """
    if base_loss is None:
        res = levenberg_marquardt(theta_hat, theta_hat, target, scen, keys,
                                  schedule=((0.0, 1),))
        base_loss = res.loss
    want = set(params) if params is not None else set(PARAM_NAMES)
    p_big, p_small = max(probes), min(probes)

    out, n_fits = [], 0
    for i, (name, is_log) in enumerate(zip(PARAM_NAMES, LOG_PARAMS)):
        if name not in want:
            continue
        off = np.log1p(p_big) if is_log else p_big
        d = {p_big: float(min(
            profile_point(theta_hat, target, scen, keys, i, -off, base_loss),
            profile_point(theta_hat, target, scen, keys, i, +off, base_loss)))}
        n_fits += 2
        v = Verdict(name=name, determined=d[p_big] >= DCHI2_FLAT, dchi2=d)
        if v.determined:
            off = np.log1p(p_small) if is_log else p_small
            d[p_small] = float(min(
                profile_point(theta_hat, target, scen, keys, i, -off, base_loss),
                profile_point(theta_hat, target, scen, keys, i, +off, base_loss)))
            n_fits += 2
            # Where the cheaper side reaches dchi2 = 1, assuming the profile is locally
            # quadratic in the offset -- the same quadratic fit
            # scripts/profile_postprocess.py uses to turn a coarse scan into a sigma.
            c = d[p_small] / (off ** 2)
            off1 = float(np.sqrt(DCHI2_1SIGMA / c)) if c > 0 else float("inf")
            if is_log:
                v.sigma_rel = float(np.expm1(off1))
            else:
                v.sigma_abs = float(np.degrees(off1))
        out.append(v)
        if verbose:
            print(f"    {v}")
    return out, n_fits


def residual_check(loss, n_obs, tol=1.02):
    """The other guard, for contrast: is the fit as good as the noise allows?

    Section 12.4 shows this catches optimiser failures and is blind to the degeneracy,
    which is the reason `guard` exists. Included so the two can be compared on the same
    cases rather than argued about.
    """
    return bool(loss <= tol * n_obs)
