"""Cross-scenario identifiability sweep: is the fuel/wind confound generic?

One well-chosen synthetic fire proves nothing -- a confound found at a single
point in parameter space could be an artefact of that point. This sweep samples
fires across the operational envelope (wind 1.5-9 m/s, a range of fuels, all wind
directions, varying buoyancy and emission strength) and recomputes the Fisher
information for all seven modality subsets in each one.

Reported per (subset, parameter): the distribution of Cramer-Rao sigma over
scenarios, and the fraction of scenarios in which the parameter clears its
operational tolerance. That fraction is the sensor-sufficiency statement an
agency can actually act on.

Run:  python scripts/sweep_identifiability.py --n 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, pack  # noqa: E402
from pyrofield.theory.identifiability import (  # noqa: E402
    all_subsets,
    crb_from_fisher,
    fisher_all_subsets,
)

# Operational envelope to sample from.
RANGES = {
    "R0": (0.05, 0.20),      # m/s, grass to light shrub
    "U": (1.5, 9.0),         # m/s midflame wind
    "theta_w": (0.0, 2 * np.pi),
    "lb_k": (0.20, 0.60),    # fuel-dependent shape coefficient
    "w_buoy": (1.5, 5.0),    # m/s plume buoyancy
    "Q": (0.5, 2.0),         # emission scale
}

TOLERANCE = {"R0": 0.10, "U": 0.10, "theta_w": np.radians(5.0),
             "lb_k": 0.15, "w_buoy": 0.15, "Q": 0.15}

SUBSET_LABEL = {
    ("mask",): "M",
    ("plume",): "P",
    ("conc",): "C",
    ("mask", "plume"): "M+P",
    ("mask", "conc"): "M+C",
    ("plume", "conc"): "P+C",
    ("mask", "plume", "conc"): "M+P+C",
}


def sample_truth(rng):
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scen = Scenario(device=device)
    rng = np.random.default_rng(args.seed)
    subsets = all_subsets()

    crb = {SUBSET_LABEL[s]: {p: [] for p in PARAM_NAMES} for s in subsets}
    conds = {SUBSET_LABEL[s]: [] for s in subsets}
    truths = []

    print(f"device={device}  scenarios={args.n}")
    t_start = time.time()
    for i in range(args.n):
        truth = sample_truth(rng)
        truths.append(truth)
        theta = pack(truth, device=device)
        t0 = time.time()
        # Central finite differences, not forward-mode AD. AD is exact for the smooth
        # operators but is corrupted where information flows through the ENO minmod
        # limiter, and it is the mask-only case that matters here. FD also happens to be
        # the estimator most favourable to the mask baseline, which keeps the comparison
        # conservative -- see RESULTS.md section 4.
        Fs = fisher_all_subsets(theta, scen, eps=0.02, method="central")
        for s in subsets:
            c, cond, _, _ = crb_from_fisher(Fs[s])
            lab = SUBSET_LABEL[s]
            conds[lab].append(cond)
            for j, p in enumerate(PARAM_NAMES):
                crb[lab][p].append(float(c[j]))
        print(f"  [{i+1:3d}/{args.n}] U={truth['U']:.1f} R0={truth['R0']:.3f} "
              f"k_LB={truth['lb_k']:.2f}  ({time.time()-t0:.1f}s)")

    print(f"\ntotal {time.time()-t_start:.0f}s\n")

    # ---- summary table: median CRB and the fraction of scenarios that clear tolerance ----
    labels = [SUBSET_LABEL[s] for s in subsets]
    header = f"{'subset':8s}" + "".join(f"{p:>14s}" for p in PARAM_NAMES)
    print("MEDIAN Cramer-Rao sigma  (fractional; theta_w in rad)")
    print(header)
    print("-" * len(header))
    for lab in labels:
        row = f"{lab:8s}"
        for p in PARAM_NAMES:
            m = float(np.median(crb[lab][p]))
            row += f"{m:14.4g}" if m < 1e4 else f"{m:14.2e}"
        print(row)

    print("\nFRACTION OF SCENARIOS MEETING OPERATIONAL TOLERANCE")
    print(header)
    print("-" * len(header))
    frac = {}
    for lab in labels:
        frac[lab] = {}
        row = f"{lab:8s}"
        for p in PARAM_NAMES:
            f = float(np.mean(np.array(crb[lab][p]) < TOLERANCE[p]))
            frac[lab][p] = f
            row += f"{f*100:13.0f}%"
        print(row)

    print("\nMEDIAN Fisher condition number")
    for lab in labels:
        print(f"  {lab:8s} {np.median(conds[lab]):.3e}")

    # ---- the headline numbers ----
    print("\n" + "=" * 70)
    print("HEADLINE")
    print("=" * 70)
    mU = np.array(crb["M"]["U"])
    mpU = np.array(crb["M+P"]["U"])
    print(f"  wind SPEED from masks alone      : median CRB {np.median(mU):.2f} "
          f"(IQR {np.percentile(mU,25):.2f}-{np.percentile(mU,75):.2f}), "
          f"{frac['M']['U']*100:.0f}% of scenarios usable")
    print(f"  wind DIRECTION from masks alone  : median CRB "
          f"{np.degrees(np.median(crb['M']['theta_w'])):.2f} deg, "
          f"{frac['M']['theta_w']*100:.0f}% of scenarios usable")
    print(f"  wind SPEED once the plume is added: median CRB {np.median(mpU):.4f}, "
          f"{frac['M+P']['U']*100:.0f}% of scenarios usable")
    print(f"  improvement factor on U          : "
          f"{np.median(mU)/max(np.median(mpU),1e-12):.0f}x")
    print(f"  emission Q, plume only vs +sensor : "
          f"{np.median(crb['M+P']['Q']):.4f} -> {np.median(crb['M+P+C']['Q']):.4f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "sweep_identifiability.json"
    out.write_text(json.dumps({
        "n": args.n, "seed": args.seed, "ranges": {k: list(v) for k, v in RANGES.items()},
        "tolerance": {k: float(v) for k, v in TOLERANCE.items()},
        "param_names": PARAM_NAMES,
        "subset_labels": labels,
        "truths": truths,
        "crb": crb, "cond": conds, "fraction_ok": frac,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
