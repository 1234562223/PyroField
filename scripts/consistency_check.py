"""Cross-check the two Fisher code paths, and probe how the mask confound varies.

Two things are verified here:

1. ``analyze`` (one Jacobian per subset) and ``fisher_all_subsets`` (one Jacobian
   shared across subsets) must return the same mask-only Fisher. If they disagree,
   one of them is wrong and every downstream conclusion is suspect.

2. The mask-only Cramer-Rao bound is then swept over wind speed and fuel shape, to
   find out whether the fuel/wind confound is a property of the whole operational
   envelope or only of particular fires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, forward, pack  # noqa: E402
from pyrofield.theory.identifiability import (  # noqa: E402
    analyze,
    crb_from_fisher,
    fisher_all_subsets,
    fisher_submatrix,
)

REF = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scen = Scenario(device=device)
    theta = pack(REF, device=device)

    print("=" * 74)
    print("[A] the two Fisher paths must agree on the mask-only case")
    print("=" * 74)
    F_a = fisher_submatrix(theta, scen, ("mask",), None).cpu().numpy()
    F_b = fisher_all_subsets(theta, scen)[("mask",)]
    rel = np.abs(F_a - F_b).max() / max(np.abs(F_a).max(), 1e-30)
    print(f"    max relative difference = {rel:.3e}   [target < 1e-6]")
    crb_a, cond_a, ev_a, _ = crb_from_fisher(F_a)
    crb_b, cond_b, ev_b, _ = crb_from_fisher(F_b)
    print(f"    CRB via analyze-path       : {np.array2string(crb_a, precision=4)}")
    print(f"    CRB via shared-Jacobian    : {np.array2string(crb_b, precision=4)}")
    print(f"    eigenvalues (shared path)  : {', '.join(f'{v:.3e}' for v in ev_b)}")

    print()
    print("=" * 74)
    print("[B] how the mask-only bound moves across the operational envelope")
    print("=" * 74)
    print(f"    reference point: {REF}")
    print()
    hdr = f"    {'U':>5s} {'lb_k':>6s} {'R0':>6s} | " + \
          " ".join(f"{p:>10s}" for p in ("R0", "U", "lb_k", "theta_w")) + \
          f" | {'lam_min':>10s} {'burn%':>6s}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    for U in (1.5, 2.5, 4.0, 6.0, 9.0):
        for lb_k in (0.20, 0.35, 0.60):
            t = pack({**REF, "U": U, "lb_k": lb_k}, device=device)
            F = fisher_all_subsets(t, scen, params=["R0", "U", "theta_w", "lb_k"])[("mask",)]
            crb, cond, ev, _ = crb_from_fisher(F)
            names = ["R0", "U", "theta_w", "lb_k"]
            burn = float((forward(t, scen)["mask"][-1] > 0.5).float().mean()) * 100
            vals = {n: crb[i] for i, n in enumerate(names)}
            print(f"    {U:5.1f} {lb_k:6.2f} {REF['R0']:6.3f} | "
                  + " ".join(f"{vals[p]:10.4f}" for p in ("R0", "U", "lb_k", "theta_w"))
                  + f" | {ev[0]:10.2e} {burn:6.1f}")

    print()
    print("    Note: CRB is a *local* bound. A large value means the misfit is flat")
    print("    along some direction at that point; it is expected to depend on the")
    print("    operating point, and how strongly is exactly what this table measures.")


if __name__ == "__main__":
    main()
