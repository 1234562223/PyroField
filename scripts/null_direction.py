"""The mask degeneracy is exact, and here is its direction in closed form.

Everything measured so far says masks cannot separate fuel from wind speed. This
script shows the statement is not empirical at all -- it follows from counting.

A burn mask is produced by the elliptical spread template, which depends on
(R0, U, k_LB) only through

    R_head = R0 (1 + a U^b)          LB = 1 + k_LB U

That map is 3 -> 2, so its Jacobian is a 2x3 matrix, its rank is at most 2, and its
null space has dimension at least one. **For every fire, at every wind speed, there is
an exact direction in parameter space along which no burn mask can ever change.**

Differentiating in log parameters (r, u, kappa) = (log R0, log U, log k_LB):

    dR_head = 0   =>   dr = -[a b U^b / (1 + a U^b)] du
    dLB     = 0   =>   dkappa = -du

so the null direction is

    v = ( -a b U^b / (1 + a U^b),  1,  -1 )

This is checked three ways: against the weakest eigenvector the Fisher analysis finds
numerically, by walking along v and confirming the predicted mask does not move, and
against a control direction that changes the same parameters by the same amount.

Run:  python scripts/null_direction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from pyrofield.physics.rothermel import WIND_A, WIND_B  # noqa: E402
from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, forward, pack  # noqa: E402
from pyrofield.theory.identifiability import crb_from_fisher, fisher_all_subsets  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
BLOCK = ["R0", "U", "lb_k"]


def analytic_null(U: float) -> np.ndarray:
    """Unit null *tangent* in (log R0, log U, log k_LB) at wind speed ``U``."""
    aub = WIND_A * U**WIND_B
    v = np.array([-WIND_B * aub / (1.0 + aub), 1.0, -1.0])
    return v / np.linalg.norm(v)


def null_curve(U_new: float, R_head: float, LB: float) -> tuple[float, float]:
    """The exact compensating (R0, k_LB) at a new wind speed.

    The null space is a *curve*, not a line: ``R0`` and ``k_LB`` depend nonlinearly on
    ``U``. ``analytic_null`` gives only its tangent at a point, which is what the Fisher
    eigenvector can be compared against, but walking a finite distance has to follow the
    curve itself. (Using the tangent for a 27 % step costs Delta chi^2 = 551, while the
    curve costs essentially nothing -- the difference is curvature, and it is exactly the
    curvature that makes a linearised Fisher analysis misleading here.)
    """
    R0 = R_head / (1.0 + WIND_A * U_new**WIND_B)
    k = (LB - 1.0) / U_new
    return R0, k


def cfl_scenario(U: float, R0: float, device: str, total_s: float = 1650.0,
                 dx: float = 15.0, target_cfl: float = 0.35) -> Scenario:
    """A scenario whose time step respects the CFL limit at this wind speed.

    At 9 m/s the default 15 s step puts the CFL number at 0.68, and the level-set
    solution degrades enough that discretisation error swamps the true null space.
    Holding the CFL number fixed instead of the time step keeps the comparison honest
    across the envelope.
    """
    aub = WIND_A * U**WIND_B
    r_head = R0 * (1.0 + aub)
    dt = min(15.0, target_cfl * dx / max(r_head, 1e-6))
    n_steps = max(20, int(round(total_s / dt)))
    times = tuple(range(n_steps // 12 - 1, n_steps, max(1, n_steps // 12)))
    return Scenario(device=device, dx=dx, dt=dt, n_steps=n_steps,
                    mask_times=times, plume_times=times, conc_times=times)


def embed(v3: np.ndarray) -> np.ndarray:
    full = np.zeros(len(PARAM_NAMES))
    for k, name in enumerate(BLOCK):
        full[PARAM_NAMES.index(name)] = v3[k]
    return full


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = tuple(range(8, 110, 9))
    scen = Scenario(device=device, n_steps=110, mask_times=times,
                    plume_times=times, conc_times=times)
    theta0 = pack(TRUTH, device=device)
    out = {}

    print("=" * 78)
    print("[1] closed-form null direction vs the weakest Fisher eigenvector")
    print("=" * 78)
    rows = []
    for U in (2.0, 3.0, 4.0, 6.0, 9.0):
        th = pack({**TRUTH, "U": U}, device=device)
        sc = cfl_scenario(U, TRUTH["R0"], device)
        cfl = TRUTH["R0"] * (1 + WIND_A * U**WIND_B) * sc.dt / sc.dx
        F = fisher_all_subsets(th, sc, eps=0.02, method="central",
                               params=BLOCK)[("mask",)]
        _, cond, ev, evec = crb_from_fisher(F)
        measured = evec[:, 0]
        v = analytic_null(U)
        cos = abs(float(np.dot(measured, v)))
        rows.append({"U": U, "cos": cos, "analytic": v.tolist(),
                     "measured": measured.tolist(), "eigvals": ev.tolist(),
                     "dt": sc.dt, "n_steps": sc.n_steps, "cfl": float(cfl)})
        print(f"  U={U:4.1f}  (dt={sc.dt:5.2f}s, {sc.n_steps} steps, CFL={cfl:.2f})")
        print(f"          analytic v = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")
        print(f"          measured   = ({measured[0]:+.4f}, {measured[1]:+.4f}, "
              f"{measured[2]:+.4f})   |cos| = {cos:.6f}")
        print(f"          eigenvalues: {', '.join(f'{e:.3e}' for e in ev)}"
              f"   ratio lam2/lam1 = {ev[1]/ev[0]:.1f}")
    worst = min(r["cos"] for r in rows)
    print(f"\n  worst |cos| over the envelope = {worst:.6f}   [target > 0.99]")
    out["alignment"] = rows

    print()
    print("=" * 78)
    print("[2] walking along the null CURVE: the mask must not move")
    print("=" * 78)
    aub0 = WIND_A * TRUTH["U"] ** WIND_B
    R_head0 = TRUTH["R0"] * (1.0 + aub0)
    LB0 = 1.0 + TRUTH["lb_k"] * TRUTH["U"]
    sigma = 0.05
    base = forward(theta0, scen)["mask"]
    n_obs = base.numel()
    print(f"  truth: R_head = {R_head0:.5f} m/s, LB = {LB0:.4f};  "
          f"noise floor chi^2 = N = {n_obs}")
    print(f"  {'U change':>10s} {'curve dchi2':>14s} {'tangent dchi2':>15s} "
          f"{'control dchi2':>15s}")

    walk = []
    v3 = analytic_null(TRUTH["U"])
    for frac in (0.05, 0.10, 0.20, 0.30):
        U_new = TRUTH["U"] * (1 + frac)
        # (a) exact null curve
        R0n, kn = null_curve(U_new, R_head0, LB0)
        t_curve = pack({**TRUTH, "U": U_new, "R0": R0n, "lb_k": kn}, device=device)
        # (b) its tangent, walked the same distance in log U
        s = np.log1p(frac) / v3[1]
        t_tan = theta0 + torch.tensor(embed(s * v3), device=device, dtype=theta0.dtype)
        # (c) control: change the wind only, compensating nothing
        t_ctl = pack({**TRUTH, "U": U_new}, device=device)

        def dchi2(t):
            return float((((forward(t, scen)["mask"] - base) / sigma) ** 2).sum())

        row = {"frac": frac, "curve": dchi2(t_curve), "tangent": dchi2(t_tan),
               "control": dchi2(t_ctl)}
        walk.append(row)
        print(f"  {frac*100:9.0f}% {row['curve']:14.2f} {row['tangent']:15.1f} "
              f"{row['control']:15.1f}")
    out["walk"] = walk

    biggest = walk[-1]["curve"]
    print(f"\n  a 30% wind error costs Delta chi^2 = {biggest:.2f} when compensated along")
    print(f"  the exact curve, against {walk[-1]['control']:.0f} when not compensated at all")
    print(f"  -- a factor of {walk[-1]['control']/max(biggest,1e-9):.0f}.")
    print( "  Following the tangent instead of the curve costs "
          f"{walk[-1]['tangent']:.0f}: that gap is pure curvature, and it is why a")
    print( "  linearised (Fisher) analysis cannot see this degeneracy.")
    ok_walk = biggest < 10.0
    print(f"  mask stays within Delta chi^2 < 10 along the null curve: {ok_walk}")

    print()
    print("=" * 78)
    print("[3] consequence for the Fisher analysis")
    print("=" * 78)
    F = fisher_all_subsets(theta0, scen, eps=0.02, method="central",
                           params=BLOCK)[("mask",)]
    _, _, ev, _ = crb_from_fisher(F)
    print(f"  smallest fire-block eigenvalue reported numerically: {ev[0]:.4e}")
    print(f"  exact value implied by the rank argument:            0")
    print( "  The nonzero number is discretisation error in the finite-difference")
    print( "  Jacobian. Any Cramer-Rao bound derived from it -- and therefore every")
    print( "  mask-only row of the sweep table -- is an artefact of that error, which")
    print( "  is why the profile likelihood and the closed-form argument agree with")
    print( "  each other and not with it.")

    ok = worst > 0.99 and ok_walk
    print("\n" + ("PASS" if ok else "FAIL"))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "null_direction.json").write_text(
        json.dumps({"truth": TRUTH, "block": BLOCK, **out,
                    "n_mask_obs": int(n_obs), "sigma_mask": sigma,
                    "min_cos": float(worst), "pass": bool(ok)}, indent=2),
        encoding="utf-8")
    print(f"wrote {RESULTS/'null_direction.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
