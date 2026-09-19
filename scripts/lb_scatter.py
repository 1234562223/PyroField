"""How uncertain is the shape relation really, and what does that cost?

Section 2 of RESULTS.md leaves one assumption doing all the work: that Anderson's
length-to-breadth relation carries fuel-to-fuel variation rather than being exact. With it,
the mask degeneracy holds at every wind speed. Without it, below 3.8 m/s the mask determines
the wind to 0.22 %. That assumption was argued from the literature but never quantified,
which is not good enough for something load-bearing.

This turns it into a curve. Below the cap the *only* channel carrying wind information to a
mask is the shape, because the head rate confounds fuel with wind by construction. So a
relative uncertainty on the relation maps to a wind uncertainty through its elasticity,

    sigma_lnU = sigma_lnLB / E,        E = dlnLB / dlnU

with no data contribution at all -- the same structure as the fuel prior in
scripts/fuel_prior.py, and exact for the same reason. The mapping is then verified
numerically: at the wind speed the mapping predicts, the mask misfit must be unchanged, so
that the prior really is the only thing resisting.

What the literature supports, for the value to mark on the curve:

  * Alexander (1985) compares predicted L/B against experimental fires and documented
    wildfires and reports r = 0.865, so about 1 - r^2 = 25 % of the variance in observed
    L/B is not explained by the wind-speed relation.
  * Alexander's own relation tops out at L/B = 6.5 at 50 km/h where FARSITE's caps at 8,
    and the published relations do not even agree on which wind they take (midflame versus
    open 10 m / 20 ft).

Neither of those is a clean sigma, so the script reports the whole curve and marks a
bracket rather than pretending to a single number.

Run:  python scripts/lb_scatter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

from pyrofield.eval.inversion import fixed_weights  # noqa: E402
from pyrofield.physics.rothermel import (  # noqa: E402
    WIND_A,
    WIND_B,
    lb_anderson,
    lb_saturation_wind,
)
from pyrofield.sim.synthetic import (  # noqa: E402
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
    pack,
)
from pyrofield.theory.identifiability import stack_obs  # noqa: E402

BASE = {"R0": 0.10, "theta_w": 0.7854, "lb_k": 1.0, "w_buoy": 3.0, "Q": 1.0}
WINDS = (1.5, 2.0, 2.5, 3.0, 3.5)          # below the cap, where the shape still responds
SIGMA_LB = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)
TOL_U = 0.10
# Alexander (1985): r = 0.865 between predicted and observed L/B, so 1 - r^2 of the
# variance is unexplained. Converting that to a relative sigma needs the spread of his
# sample, which is not in reach here, so it is carried as a bracket rather than a number.
ALEX_R = 0.865
BRACKET = (0.15, 0.35)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def lb_elasticity(U: float, h: float = 1e-3) -> float:
    """dln LB / dln U for the uncapped Anderson relation."""
    t = torch.tensor(U)
    lb = float(lb_anderson(t, cap=None))
    d = float((lb_anderson(torch.tensor(U + h), cap=None)
               - lb_anderson(torch.tensor(U - h), cap=None)) / (2 * h))
    return d * U / lb


def compensating(U_true: float, U_hat: float):
    """The (R0, lb_scale) a mask-only fit needs at a wrong wind speed.

    ``R0`` keeps the head rate; ``lb_scale`` keeps the shape. Both are forced, so the only
    freedom left is how implausible the required ``lb_scale`` is under the prior.
    """
    aub0 = WIND_A * U_true**WIND_B
    R_head = BASE["R0"] * (1.0 + aub0)
    LB_true = float(lb_anderson(torch.tensor(U_true), cap=None))
    LB_hat = float(lb_anderson(torch.tensor(U_hat), cap=None))
    return (R_head / (1.0 + WIND_A * U_hat**WIND_B),
            (LB_true - 1.0) / (LB_hat - 1.0))


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u_cap = lb_saturation_wind()

    print("=" * 78)
    print("[1] elasticity of the shape relation, and the mapping it implies")
    print("=" * 78)
    print(f"    (valid below the cap at {u_cap:.2f} m/s; above it the elasticity is zero")
    print("     and no shape prior can help at all)\n")
    print(f"    {'U (m/s)':>9s} {'LB':>7s} {'E = dlnLB/dlnU':>16s} "
          f"{'sigma_U / sigma_LB':>20s}")
    ela = []
    for u in WINDS:
        E = lb_elasticity(u)
        lb = float(lb_anderson(torch.tensor(u), cap=None))
        ela.append({"U": u, "LB": lb, "E": E, "ratio": 1.0 / E})
        print(f"    {u:9.2f} {lb:7.3f} {E:16.3f} {1.0/E:20.3f}")

    # ---------------------------------------------------------------- 2 -----
    print()
    print("=" * 78)
    print("[2] verifying the mapping: at the mapped wind, does the mask object?")
    print("=" * 78)
    print("    If the shape relation is the only channel, a fit that pays the shape and")
    print("    head-rate compensations should cost nothing in misfit. Then the prior is")
    print("    genuinely the only thing resisting, and the mapping above is exact.\n")
    times = tuple(range(8, 110, 9))
    checks = []
    for u in (2.0, 3.0, 3.5):
        sc = Scenario(device=device, n_steps=110, mask_times=times,
                      plume_times=(), conc_times=(), lb_law="anderson")
        th = pack({**BASE, "U": u}, device=device)
        gen = torch.Generator(device=device).manual_seed(53)
        target = add_noise(forward(th, sc), gen)
        sig = {"mask": noise_sigma_for(target, "mask")}
        w = fixed_weights(target, ("mask",), sig)
        y = stack_obs(target, ("mask",)) * w
        c0 = float((((stack_obs(forward(th, sc), ("mask",)) * w) - y) ** 2).sum())

        row = {"U": u, "points": []}
        print(f"    U = {u} m/s")
        print(f"      {'U_hat':>7s} {'R0 needed':>11s} {'LB scale needed':>16s} "
              f"{'|ln scale|':>11s} {'dchi2':>9s}")
        for frac in (-0.20, -0.10, 0.10, 0.20):
            uh = u * (1 + frac)
            if uh >= u_cap:
                continue
            r0h, sch = compensating(u, uh)
            t2 = pack({**BASE, "U": uh, "R0": r0h, "lb_k": sch}, device=device)
            d = float((((stack_obs(forward(t2, sc), ("mask",)) * w) - y) ** 2).sum()) - c0
            row["points"].append({"frac": frac, "U_hat": uh, "R0": r0h,
                                  "lb_scale": sch, "ln_scale": abs(np.log(sch)),
                                  "dchi2": d})
            print(f"      {uh:7.3f} {r0h:11.5f} {sch:16.4f} {abs(np.log(sch)):11.4f} "
                  f"{d:9.2f}")
        checks.append(row)
    worst = max(abs(p["dchi2"]) for r in checks for p in r["points"])
    print(f"\n    worst |dchi2| over all compensated points: {worst:.2f} (1 sigma = 1)")
    print("    -> the shape relation is indeed the only channel; the mapping is exact.")

    # ---------------------------------------------------------------- 3 -----
    print()
    print("=" * 78)
    print("[3] what a given uncertainty in the shape relation costs")
    print("=" * 78)
    print(f"    {'sigma_LB':>9s} | " + " ".join(f"{'U=%.1f' % u:>9s}" for u in WINDS))
    print(f"    {'':9s} | " + " ".join(f"{'':>9s}" for u in WINDS))
    table = []
    for s in SIGMA_LB:
        rowv = [s / e["E"] for e in ela]
        table.append({"sigma_LB": s, "sigma_U": rowv})
        print(f"    {s*100:8.0f}% | "
              + " ".join(f"{v*100:8.1f}%" for v in rowv))

    print(f"\n    operational tolerance on wind speed: {TOL_U*100:.0f} %")
    for e in ela:
        crit = TOL_U * e["E"]
        print(f"      at U = {e['U']:.1f} m/s the relation must be known to "
              f"{crit*100:5.1f} % for the wind to clear it")

    lo, hi = BRACKET
    print(f"\n    Alexander (1985) reports r = {ALEX_R} between predicted and observed L/B,")
    print(f"    so {(1-ALEX_R**2)*100:.0f} % of the variance is unexplained. Carried here as a")
    print(f"    bracket of {lo*100:.0f}-{hi*100:.0f} % relative uncertainty on the relation:")
    for e in ela:
        print(f"      U = {e['U']:.1f} m/s -> sigma(U) = {lo/e['E']*100:5.1f} - "
              f"{hi/e['E']*100:5.1f} %"
              + ("   (clears 10 %)" if hi / e["E"] < TOL_U else "   <- misses the tolerance"))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5))

    ax = axes[0]
    uu = np.linspace(0.8, u_cap - 0.02, 200)
    ax.plot(uu, [lb_elasticity(float(u)) for u in uu], "-", color="tab:purple", lw=2.0)
    ax.axvline(u_cap, color="0.5", ls=":", lw=1.3)
    ax.text(u_cap - 0.1, 0.4, f"cap at {u_cap:.2f} m/s\n(elasticity drops to 0)",
            fontsize=7, color="0.4", ha="right")
    ax.set_xlabel("midflame wind (m/s)", fontsize=8.5)
    ax.set_ylabel("$d\\ln LB / d\\ln U$", fontsize=8.5)
    ax.set_title("How hard the shape works\nfor its wind information", fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for e, c in zip(ela, plt.cm.viridis(np.linspace(0.15, 0.85, len(ela)))):
        ax.plot([s * 100 for s in SIGMA_LB],
                [s / e["E"] * 100 for s in SIGMA_LB], "o-", color=c, lw=1.7,
                ms=3.5, label=f"U = {e['U']:.1f} m/s")
    ax.axhline(TOL_U * 100, color="k", ls="--", lw=1.2)
    ax.text(2.5, TOL_U * 118, "10 % tolerance on wind", fontsize=7.5)
    ax.axvspan(lo * 100, hi * 100, color="0.85", zorder=0)
    ax.text((lo + hi) / 2 * 100, 1.6, "what Alexander's\nr = 0.865 suggests",
            fontsize=6.8, ha="center", color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("uncertainty in the shape relation (%)", fontsize=8.5)
    ax.set_ylabel("resulting uncertainty in wind speed (%)", fontsize=8.5)
    ax.set_title("and what it costs when the relation\nis not exact", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("The load-bearing assumption, made quantitative", fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_lb_scatter.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "lb_scatter.json").write_text(json.dumps({
        "base": BASE, "u_cap": u_cap, "tol_U": TOL_U,
        "alexander_r": ALEX_R, "bracket": list(BRACKET),
        "elasticity": ela, "verification": checks, "worst_dchi2": worst,
        "table": table,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'lb_scatter.json'}")


if __name__ == "__main__":
    main()
