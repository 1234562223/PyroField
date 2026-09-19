"""Lead time, recomputed without derivatives.

The first lead-time curves were built from Cramer-Rao bounds. Those turned out to
overstate what masks know, because the fuel/wind degeneracy is curved and a Fisher
matrix only sees local curvature (see RESULTS.md section 4). The crossing times in
that version are therefore optimistic for masks.

This redoes the measurement with the profile likelihood. At each observation window,
the wind speed is forced to a few wrong values, everything else is refitted, and the
resulting rise in misfit says how large a wind error the data can actually detect:

    Delta chi^2 = (relative wind error / sigma)^2    =>   sigma = error / sqrt(Delta chi^2)

sigma estimated this way needs no derivatives and no step size, and it accounts for
the curved valley that defeats the linearised bound.

Run:  python scripts/lead_time_profile.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

from pyrofield.eval.inversion import fixed_weights, levenberg_marquardt  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
    pack,
)
from pyrofield.theory.identifiability import stack_obs  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
T_STEPS = (9, 27, 45, 63, 81, 99)
CADENCE = 9
FRACS = (0.10, 0.25, 0.50)          # wind errors to impose
SETS = {"M": ("mask",), "M+P": ("mask", "plume")}
SCHEDULE = ((2.0, 3), (0.0, 9))
TOL_U = 0.10

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scenario_upto(T, device):
    times = tuple(t for t in range(CADENCE - 1, T, CADENCE)) or (T - 1,)
    return Scenario(device=device, n_steps=T, mask_times=times,
                    plume_times=times, conc_times=times)


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta0_full = pack(TRUTH, device=device)
    iu = PARAM_NAMES.index("U")
    free = [i for i in range(len(PARAM_NAMES)) if i != iu]

    out = {lab: {"t_min": [], "sigma": [], "detail": []} for lab in SETS}
    t_all = time.time()

    for T in T_STEPS:
        scen = scenario_upto(T, device)
        t_min = T * scen.dt / 60
        gen = torch.Generator(device=device).manual_seed(7)
        target = add_noise(forward(theta0_full, scen), gen)

        for lab, keys in SETS.items():
            sig = {k: noise_sigma_for(target, k) for k in keys}
            w = fixed_weights(target, keys, sig)
            y = stack_obs(target, keys) * w

            def chi2(t):
                return float((((stack_obs(forward(t, scen), keys) * w) - y) ** 2).sum())

            base = levenberg_marquardt(theta0_full.clone(), theta0_full, target, scen,
                                       keys, schedule=SCHEDULE, free_idx=free)
            c0 = chi2(base.theta_hat)

            sigmas, detail = [], []
            warm = base.theta_hat.clone()
            for f in FRACS:
                start = warm.clone()
                start[iu] = theta0_full[iu] + float(np.log1p(f))
                res = levenberg_marquardt(start, theta0_full, target, scen, keys,
                                          schedule=SCHEDULE, free_idx=free)
                d = chi2(res.theta_hat) - c0
                detail.append({"frac": f, "dchi2": d})
                if d > 0.05:                       # resolvable rise
                    sigmas.append(f / np.sqrt(d))
                warm = res.theta_hat.clone()

            # With several imposed errors the estimates should agree; take the median,
            # and if nothing produced a resolvable rise the wind is simply not
            # determined at this window -- record it as worse than the largest probe.
            s = float(np.median(sigmas)) if sigmas else float(max(FRACS) / np.sqrt(0.05))
            out[lab]["t_min"].append(t_min)
            out[lab]["sigma"].append(s)
            out[lab]["detail"].append(detail)
            print(f"  T={t_min:5.2f} min  {lab:4s}  sigma(U) = {s*100:8.2f}%   "
                  + "  ".join(f"d(+{d['frac']*100:.0f}%)={d['dchi2']:.2f}" for d in detail)
                  + f"   [{time.time()-t_all:.0f}s]")

    print(f"\ntotal {time.time()-t_all:.0f}s")

    print("\n" + "=" * 70)
    print("PROFILE-BASED TIME UNTIL WIND SPEED IS KNOWN TO 10 %")
    print("=" * 70)
    cross = {}
    for lab in SETS:
        t = np.array(out[lab]["t_min"])
        s = np.array(out[lab]["sigma"])
        below = np.where(s < TOL_U)[0]
        c = float(t[below.min()]) if len(below) else None
        cross[lab] = c
        print(f"  {lab:5s}: " + (f"{c:.1f} min" if c else "never within the scan"))
    if cross["M+P"] and not cross["M"]:
        print(f"\n  lead time gained by reading the plume: "
              f"> {T_STEPS[-1]*0.25 - cross['M+P']:.1f} min "
              f"(masks never get there inside {T_STEPS[-1]*0.25:.1f} min)")
    elif cross["M"] and cross["M+P"]:
        print(f"\n  lead time gained by reading the plume: "
              f"{cross['M'] - cross['M+P']:.1f} min")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for lab, c in (("M", "tab:red"), ("M+P", "tab:blue")):
        ax.semilogy(out[lab]["t_min"], np.array(out[lab]["sigma"]) * 100,
                    "o-", color=c, lw=2.0, ms=5, label=lab)
    ax.axhline(TOL_U * 100, color="0.35", ls="--", lw=1.0)
    ax.text(out["M"]["t_min"][0], TOL_U * 118, "10 % tolerance", fontsize=7.5, color="0.35")
    ax.set_xlabel("observation window (min)", fontsize=8.5)
    ax.set_ylabel("profile-likelihood $\\sigma$ on wind speed (%)", fontsize=8.5)
    ax.set_title("Lead time, measured without derivatives", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIGS / "fig_lead_time_profile.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "lead_time_profile.json").write_text(json.dumps(
        {"truth": TRUTH, "fracs": list(FRACS), "tol_U": TOL_U,
         "data": out, "crossings": cross}, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'lead_time_profile.json'}")


if __name__ == "__main__":
    main()
