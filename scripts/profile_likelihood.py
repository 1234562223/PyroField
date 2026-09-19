"""Profile likelihood: the derivative-free arbiter of how much the masks really know.

Two estimators of the mask-only Fisher information disagreed by three orders of
magnitude. Forward-mode AD -- the exact derivative of the discretised model --
reported CRB(U) around 38 (no usable information). Central finite differences
reported 0.007-0.026 (well determined), and drifted with the step size.

Neither can referee itself, so this experiment uses a quantity that involves no
derivatives at all. For each fixed value of the wind speed, the misfit is minimised
over every other parameter; the resulting profile chi-square curve is the likelihood
the data actually supports. The 1-sigma interval is where the curve rises by 1 above
its minimum, and that interval is the answer.

If the profile is flat over +/-50% in wind speed, the masks do not determine it and
AD was right. If it rises steeply, the finite differences were right.

Run:  python scripts/profile_likelihood.py
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

from pyrofield.eval.inversion import (  # noqa: E402
    fixed_weights,
    levenberg_marquardt,
)
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
    pack,
)
from pyrofield.theory.identifiability import (  # noqa: E402
    crb_from_fisher,
    fisher_all_subsets,
    stack_obs,
)

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
SETS = {"M": ("mask",), "M+P+C": ("mask", "plume", "conc")}
SCHEDULE = ((2.0, 4), (0.0, 10))

# Scan ranges per parameter, in working-space units (log-ratio, or radians for the
# wind direction). The claim "masks give direction but not speed" rests on the CRB,
# and the CRB proved unreliable for speed, so direction has to be checked the same
# derivative-free way rather than assumed.
SCANS = {
    "U": np.linspace(-0.55, 0.55, 15),              # -42% .. +73%
    # Wind direction is expected to be sharply determined (the burn ellipse points
    # downwind), so the scan is narrow: +/- 1.15 degrees, finest step 0.16 degrees.
    # A +/- 17 degree scan would turn the fire around entirely and tell us nothing
    # about the width of the optimum.
    "theta_w": np.linspace(-0.02, 0.02, 15),
    "R0": np.linspace(-0.55, 0.55, 15),
    "Q": np.linspace(-0.80, 0.80, 15),
}


def fmt_offset(param: str, d: float) -> str:
    if param == "theta_w":
        return f"{np.degrees(d):+6.1f} deg"
    return f"x{np.exp(d):5.2f}"


def to_physical_error(param: str, d: float) -> float:
    """Convert a working-space offset to the error unit the parameter is reported in."""
    return abs(np.degrees(d)) if param == "theta_w" else abs(np.expm1(d))

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="U", choices=sorted(SCANS))
    ap.add_argument("--sets", default=",".join(SETS),
                    help="comma-separated subset labels to profile")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any checkpoint and start over")
    args = ap.parse_args()
    param = args.param
    OFFSETS = SCANS[param]
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # These runs take the better part of an hour, so each point is checkpointed as it
    # finishes and a restart picks up where it left off. Without this, an interrupted
    # run loses everything.
    ckpt_path = RESULTS / f"profile_{param}_ckpt.json"
    ckpt = {}
    if ckpt_path.exists() and not args.fresh:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        done = {k: sum(1 for v in c if v is not None) for k, c in ckpt.items()}
        print(f"resuming from {ckpt_path.name}: {done}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = tuple(range(8, 110, 9))
    scen = Scenario(device=device, n_steps=110, mask_times=times,
                    plume_times=times, conc_times=times)
    theta0 = pack(TRUTH, device=device)

    gen = torch.Generator(device=device).manual_seed(7)
    target = add_noise(forward(theta0, scen), gen)
    iu = PARAM_NAMES.index(param)

    out = {}
    for label in wanted:
        keys = SETS[label]
        sig = {k: noise_sigma_for(target, k) for k in keys}
        w = fixed_weights(target, keys, sig)
        y = stack_obs(target, keys) * w

        def chi2(t):
            return float((((stack_obs(forward(t, scen), keys) * w) - y) ** 2).sum())

        free = [i for i in range(len(PARAM_NAMES)) if i != iu]
        print(f"\n{label}: profiling {param} over {len(OFFSETS)} fixed values")

        # Sweep outward from the truth in both directions, warm-starting each step.
        order = sorted(range(len(OFFSETS)), key=lambda i: abs(OFFSETS[i]))
        prof = np.full(len(OFFSETS), np.nan)
        saved = ckpt.get(label)
        if saved and len(saved) == len(OFFSETS):
            for i, v in enumerate(saved):
                if v is not None:
                    prof[i] = float(v)
        warm = {0: theta0.clone()}
        for i in order:
            if np.isfinite(prof[i]):
                print(f"   {param} {fmt_offset(param, OFFSETS[i])}  "
                      f"chi2 = {prof[i]:.6e}   (from checkpoint)")
                continue
            d = OFFSETS[i]
            near = min(warm, key=lambda k: abs(OFFSETS[k] - d) if k else abs(d))
            start = warm[near].clone() if near in warm else theta0.clone()
            start[iu] = theta0[iu] + float(d)
            t0 = time.time()
            res = levenberg_marquardt(start, theta0, target, scen, keys,
                                      schedule=SCHEDULE, free_idx=free)
            prof[i] = chi2(res.theta_hat)
            warm[i] = res.theta_hat.clone()
            print(f"   {param} {fmt_offset(param, d)}  chi2 = {prof[i]:.6e}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
            ckpt[label] = [None if not np.isfinite(v) else float(v) for v in prof]
            ckpt_path.write_text(json.dumps(ckpt, indent=1), encoding="utf-8")

        chi_min = float(np.nanmin(prof))
        dchi = prof - chi_min
        # 1-sigma is where the profile rises by 1 (Wilks, one parameter of interest).
        below = np.where(dchi <= 1.0)[0]
        if len(below):
            lo, hi = OFFSETS[below.min()], OFFSETS[below.max()]
            width = (to_physical_error(param, lo) + to_physical_error(param, hi)) / 2.0
            edge = below.min() == 0 or below.max() == len(OFFSETS) - 1
        else:
            width = 0.0
            edge = False
        unit = "deg" if param == "theta_w" else "% of truth"
        shown = width if param == "theta_w" else width * 100
        out[label] = {"param": param, "offsets": OFFSETS.tolist(),
                      "chi2": prof.tolist(), "dchi2": dchi.tolist(),
                      "sigma": float(width), "hit_edge": bool(edge)}
        tag = " (hits the scan edge -- even wider than this)" if edge else ""
        print(f"   1-sigma half-width on {param}: {shown:.2f} {unit}{tag}")

    # Fisher predictions for comparison.
    Fs_ad = fisher_all_subsets(theta0, scen, method="ad")
    Fs_fd = fisher_all_subsets(theta0, scen, eps=0.02, method="central")
    pred = {}
    for label in wanted:
        keys = SETS[label]
        a, _, _, _ = crb_from_fisher(Fs_ad[keys])
        f, _, _, _ = crb_from_fisher(Fs_fd[keys])
        pred[label] = {"ad": float(a[iu]), "fd": float(f[iu])}

    unit = "deg" if param == "theta_w" else "%"
    scale = 1.0 if param == "theta_w" else 100.0
    print("\n" + "=" * 78)
    print(f"VERDICT for {param}: which Jacobian estimator matches the likelihood?")
    print("=" * 78)
    print(f"  {'set':8s}{'profile 1-sigma':>18s}{'AD Fisher CRB':>16s}{'FD Fisher CRB':>16s}")
    for label in wanted:
        p = out[label]["sigma"]
        print(f"  {label:8s}{p*scale:15.3f} {unit}{pred[label]['ad']*scale:14.3f} {unit}"
              f"{pred[label]['fd']*scale:14.3f} {unit}")
    for label in wanted:
        p = max(out[label]["sigma"], 1e-12)
        ra = abs(np.log10(max(pred[label]["ad"], 1e-12) / p))
        rf = abs(np.log10(max(pred[label]["fd"], 1e-12) / p))
        win = "AD" if ra < rf else "FD"
        print(f"  {label}: profile is closer to {win} "
              f"(AD off by 10^{ra:.1f}, FD off by 10^{rf:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, (label, dd) in zip(axes, out.items()):
        offs = np.array(dd["offsets"])
        x = np.degrees(offs) if param == "theta_w" else (np.exp(offs) - 1) * 100
        ax.plot(x, dd["dchi2"], "o-", lw=1.8, ms=4, color="tab:purple")
        ax.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax.text(x[0], 1.4, "$\\Delta\\chi^2 = 1$  (1$\\sigma$)", fontsize=7.5, color="0.35")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(f"error imposed on {param} ({unit})", fontsize=8.5)
        ax.set_title(f"{label}   1$\\sigma$ = {dd['sigma']*scale:.3g} {unit}"
                     + (" (unbounded)" if dd["hit_edge"] else ""), fontsize=9.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("profile $\\Delta\\chi^2$", fontsize=8.5)
    fig.suptitle(f"Profile likelihood for {param}: how hard does the data push back?",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    p = FIGS / f"fig_profile_{param}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    dest = RESULTS / f"profile_{param}.json"
    dest.write_text(json.dumps(
        {"truth": TRUTH, "param": param, "profiles": out, "fisher_pred": pred},
        indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
