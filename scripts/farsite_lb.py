"""The shape channel, under FARSITE's own length-to-breadth law.

Sections 1-2 of RESULTS.md establish the fuel/wind degeneracy using a shape law
``LB = 1 + k_LB * U`` with a free, fuel-dependent coefficient. That choice deserves
scrutiny, because FARSITE and FlamMap do not use it. They use Anderson (1983),

    LB = 0.936 exp(0.2566 U) + 0.461 exp(-0.1548 U) - 0.397,     U in mi/h

with no free fuel parameter at all -- the relation is stated as applying to any fuel type.
Taken at face value that would *break* the degeneracy: the burn shape would give the wind
speed, and the head rate would then give the fuel.

Two things stop it, and this script measures both rather than asserting them.

  1. **The relation saturates.** FARSITE truncates LB at 8, the maximum in the empirical
     data Alexander (1985) collects. That cap is reached at a midflame wind of 3.79 m/s --
     inside the ordinary operating range, not at some extreme. Above it the shape is
     clamped and carries nothing about the wind.

  2. **The single curve papers over fuel-to-fuel variation.** Anderson reports
     group-specific relations and Alexander documents the scatter. Treating the relation
     as exact is an assumption, and this measures what it costs when it is not.

The output is a 2x2: shape law known or uncertain, wind below or above the cap.

Run:  python scripts/farsite_lb.py
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
from pyrofield.physics.rothermel import (  # noqa: E402
    LB_CAP,
    MS_PER_MIH,
    lb_anderson,
    lb_saturation_wind,
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
    jacobian,
    stack_obs,
    weight_vector,
)

BASE = {"R0": 0.10, "theta_w": 0.7854, "lb_k": 1.0, "w_buoy": 3.0, "Q": 1.0}
WINDS = (1.5, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0)
PROFILE_OFFSETS = np.linspace(-0.45, 0.45, 11)
IU, ILB = PARAM_NAMES.index("U"), PARAM_NAMES.index("lb_k")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scen(device, n_steps=110):
    times = tuple(range(8, n_steps, 9))
    return Scenario(device=device, n_steps=n_steps, mask_times=times,
                    plume_times=times, conc_times=times, lb_law="anderson")


def asymmetry(prof):
    """Is the clamped-regime degeneracy one-sided?

    Above the cap the shape is pinned at LB = 8, so forcing the wind *up* changes nothing
    a mask can see. Forcing it *down* far enough leaves the cap, the shape responds again,
    and the mask notices. The two sides of the profile should therefore look completely
    different, with the break at the wind that re-enters the cap.

    This matters operationally rather than only structurally: section 5 shows that
    over-estimating the wind means under-estimating the fuel, which makes the forecast too
    slow and the warning late. That is precisely the side the cap leaves open.
    """
    frac = np.expm1(PROFILE_OFFSETS)
    print()
    print("=" * 78)
    print("[2b] the clamped-regime degeneracy is one-sided")
    print("=" * 78)
    rows = {}
    for key, p in prof.items():
        v = np.array(p["dchi2"])
        U = p["U"]
        lo, hi = v[frac < -0.02], v[frac > 0.02]
        el, eh = np.abs(frac[frac < -0.02]), frac[frac > 0.02]

        def one_sided(e, y):
            u = y > 1e-3
            s = float(np.median(e[u] / np.sqrt(y[u]))) if u.any() else float(e.max())
            return s, bool(y.max() < 4.0)

        s_lo, b_lo = one_sided(el, lo)
        s_hi, b_hi = one_sided(eh, hi)
        cross = (lb_saturation_wind() / U - 1) if U > lb_saturation_wind() else None
        rows[key] = {"sigma_under": s_lo, "under_is_bound": b_lo,
                     "sigma_over": s_hi, "over_is_bound": b_hi,
                     "max_dchi2_under": float(lo.max()), "max_dchi2_over": float(hi.max()),
                     "cap_reentry_frac": cross}
        print(f"\n  {key}")
        print("     imposed U error : " + " ".join(f"{f*100:>8.0f}%" for f in frac))
        print("     delta chi^2     : " + " ".join(f"{x:>9.1f}" for x in v))
        print(f"     under-estimating U: max dchi2 {lo.max():10.1f}   "
              f"sigma {'>' if b_lo else '='}{s_lo*100:7.1f} %")
        print(f"     over-estimating  U: max dchi2 {hi.max():10.1f}   "
              f"sigma {'>' if b_hi else '='}{s_hi*100:7.1f} %"
              + ("   <- the dangerous side, and it is open" if b_hi else ""))
        if cross is not None:
            print(f"     (the cap is re-entered at an imposed error of {cross*100:.0f} %, "
                  f"and that is where the profile turns on)")
    return rows


def mask_fisher(theta, sc, device, sub):
    J, base = jacobian(theta, sc, ("mask",), eps=0.02, method="central")
    w = weight_vector(base, ("mask",), device, theta.dtype)
    Jw = (J * w[:, None])[:, sub]
    return (Jw.T @ Jw).double().cpu().numpy()


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    # ---------------------------------------------------------------- 0 -----
    u_sat = lb_saturation_wind()
    print("=" * 78)
    print("[0] FARSITE's shape law and where it stops responding")
    print("=" * 78)
    print(f"    LB = 0.936 exp(0.2566 U_mih) + 0.461 exp(-0.1548 U_mih) - 0.397, "
          f"capped at {LB_CAP}")
    print(f"    the cap is reached at U = {u_sat:.2f} m/s "
          f"({u_sat/MS_PER_MIH:.2f} mi/h)\n")
    print(f"    {'U (m/s)':>9s} {'U (mi/h)':>9s} {'LB uncapped':>12s} {'LB as used':>11s} "
          f"{'dLB/dU':>9s}")
    curve = []
    for u in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.79, 4.0, 5.0, 6.0, 8.0):
        t = torch.tensor(u)
        raw = float(lb_anderson(t, cap=None))
        used = float(lb_anderson(t))
        d = float((lb_anderson(torch.tensor(u + 0.05)) - lb_anderson(torch.tensor(u - 0.05)))
                  / 0.1)
        curve.append({"U": u, "lb_raw": raw, "lb_used": used, "dlb_du": d})
        print(f"    {u:9.2f} {u/MS_PER_MIH:9.2f} {raw:12.3f} {used:11.3f} {d:9.3f}"
              + ("   <- clamped" if used >= LB_CAP - 1e-9 else ""))
    out["lb_curve"] = curve
    out["u_saturation"] = u_sat

    # ---------------------------------------------------------------- 1 -----
    print()
    print("=" * 78)
    print("[1] Fisher information about the wind from masks, under Anderson's law")
    print("=" * 78)
    print("    'law exact'   : the LB relation is taken as known (FARSITE's assumption)")
    print("    'law uncertain': its fuel-to-fuel scale is a free parameter")
    print()
    print(f"    {'U':>6s} {'LB':>7s} | {'law exact: CRB(R0)':>19s} {'CRB(U)':>10s} | "
          f"{'law uncertain: CRB(U)':>22s}")
    rows = []
    t0 = time.time()
    for u in WINDS:
        sc = scen(device)
        th = pack({**BASE, "U": u}, device=device)
        lb = float(lb_anderson(torch.tensor(u)))
        F_known = mask_fisher(th, sc, device, [0, IU])          # R0, U ; LB law fixed
        c_k, _, ev_k, _ = crb_from_fisher(F_known)
        F_free = mask_fisher(th, sc, device, [0, IU, ILB])      # + LB scale free
        c_f, _, ev_f, _ = crb_from_fisher(F_free)
        rows.append({"U": u, "LB": lb, "clamped": lb >= LB_CAP - 1e-9,
                     "crb_R0_known": float(c_k[0]), "crb_U_known": float(c_k[1]),
                     "eig_known": ev_k.tolist(),
                     "crb_U_free": float(c_f[1]), "eig_free": ev_f.tolist()})
        print(f"    {u:6.2f} {lb:7.3f} | {c_k[0]:19.5g} {c_k[1]:10.5g} | "
              f"{c_f[1]:22.5g}"
              + ("   <- LB clamped" if lb >= LB_CAP - 1e-9 else ""))
    print(f"    ({time.time()-t0:.0f}s)")
    out["fisher"] = rows

    # ---------------------------------------------------------------- 2 -----
    print()
    print("=" * 78)
    print("[2] profile likelihood: does the data actually resist a wrong wind?")
    print("=" * 78)
    prof = {}
    for u, tag in ((3.0, "below the cap"), (5.0, "above the cap")):
        sc = scen(device)
        th = pack({**BASE, "U": u}, device=device)
        gen = torch.Generator(device=device).manual_seed(41)
        target = add_noise(forward(th, sc), gen)
        sig = {"mask": noise_sigma_for(target, "mask")}
        w = fixed_weights(target, ("mask",), sig)
        y = stack_obs(target, ("mask",)) * w

        def chi2(t):
            return float((((stack_obs(forward(t, sc), ("mask",)) * w) - y) ** 2).sum())

        for law, free in (("law exact", [0, 2, 4, 5]),
                          ("law uncertain", [0, 2, 3, 4, 5])):
            vals, warm = [], th.clone()
            for d in PROFILE_OFFSETS:
                s = warm.clone()
                s[IU] = th[IU] + float(d)
                r = levenberg_marquardt(s, th, target, sc, ("mask",),
                                        schedule=((0.0, 10),), free_idx=free)
                vals.append(chi2(r.theta_hat))
                warm = r.theta_hat.clone()
            v = np.array(vals) - min(vals)
            # sigma from the parabola, or a lower bound if the profile never rises to 2 sigma
            err = np.abs(np.expm1(PROFILE_OFFSETS))
            use = (v > 1e-3) & (err > 0)
            if v.max() < 4.0 or not use.any():
                sigma, bound = float(err.max()), True
            else:
                sigma, bound = float(np.median(err[use] / np.sqrt(v[use]))), False
            prof[f"U={u} {law}"] = {"U": u, "law": law, "dchi2": v.tolist(),
                                    "sigma": sigma, "is_bound": bound,
                                    "max_dchi2": float(v.max())}
            print(f"    U = {u} m/s ({tag}), {law:14s}: "
                  f"max dchi2 over +/-45% = {v.max():10.1f}   "
                  f"sigma(U) {'>' if bound else '='} {sigma*100:7.2f} %")
    out["profiles"] = prof

    # --------------------------------------------------------------- 2b -----
    out["asymmetry"] = asymmetry(prof)

    # ---------------------------------------------------------------- 3 -----
    print()
    print("=" * 78)
    print("[3] the 2x2")
    print("=" * 78)
    a = prof["U=3.0 law exact"]["sigma"]
    b = prof["U=5.0 law exact"]["sigma"]
    c = prof["U=3.0 law uncertain"]["sigma"]
    d = prof["U=5.0 law uncertain"]["sigma"]
    print(f"    {'':22s} {'U = 3 m/s (LB responds)':>26s} {'U = 5 m/s (LB clamped)':>26s}")
    print(f"    {'shape law exact':22s} "
          f"{('>' if prof['U=3.0 law exact']['is_bound'] else '') + f'{a*100:.2f} %':>26s} "
          f"{('>' if prof['U=5.0 law exact']['is_bound'] else '') + f'{b*100:.2f} %':>26s}")
    print(f"    {'shape law uncertain':22s} "
          f"{('>' if prof['U=3.0 law uncertain']['is_bound'] else '') + f'{c*100:.2f} %':>26s} "
          f"{('>' if prof['U=5.0 law uncertain']['is_bound'] else '') + f'{d*100:.2f} %':>26s}")
    print()
    print("    Three of the four cells are degenerate. The one that is not requires both")
    print("    a wind below 3.8 m/s and Anderson's relation taken as exact for the fuel")
    print("    in front of you -- an assumption FARSITE makes for convenience and that")
    print("    Anderson's own group-specific relations do not support.")

    plot(out, u_sat, rows, prof)

    (RESULTS / "farsite_lb.json").write_text(json.dumps(
        {"base": BASE, "lb_cap": LB_CAP, **out}, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'farsite_lb.json'}")


def plot(out, u_sat, rows, prof):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    us = np.linspace(0.2, 8.0, 300)
    raw = [float(lb_anderson(torch.tensor(float(u)), cap=None)) for u in us]
    used = [float(lb_anderson(torch.tensor(float(u)))) for u in us]
    ax.plot(us, raw, "--", color="0.6", lw=1.5, label="Anderson (1983)")
    ax.plot(us, used, "-", color="tab:red", lw=2.2, label="as FARSITE uses it")
    ax.axhline(LB_CAP, color="0.35", ls=":", lw=1.2)
    ax.axvline(u_sat, color="tab:blue", ls=":", lw=1.4)
    ax.text(u_sat + 0.12, 2.0, f"{u_sat:.2f} m/s", fontsize=7.5, color="tab:blue")
    ax.set_xlabel("midflame wind (m/s)", fontsize=8.5)
    ax.set_ylabel("length-to-breadth ratio", fontsize=8.5)
    ax.set_title("The shape channel saturates\ninside the operating range", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    uu = [r["U"] for r in rows]
    ax.semilogy(uu, [r["crb_U_known"] * 100 for r in rows], "o-", color="tab:blue",
                lw=1.9, label="shape law exact")
    ax.semilogy(uu, [r["crb_U_free"] * 100 for r in rows], "s-", color="tab:red",
                lw=1.9, label="shape law uncertain")
    # Past the cap the shape derivative is exactly zero, so the mask-only Fisher is
    # singular in (R0, U) and the finite-difference bound there is discretisation error,
    # not information -- the same pathology section 10 documents. Say so on the figure
    # rather than letting the curve imply the wind is well determined at 7 m/s.
    ax.axvspan(u_sat, max(uu) * 1.02, color="0.88", zorder=0)
    ax.text(u_sat + 0.1, 3e-2, "bound not trustworthy here:\nshape derivative is zero,\n"
            "see section 10 and the\nprofiles at right",
            fontsize=6.4, color="0.35", va="bottom")
    ax.axvline(u_sat, color="0.5", ls=":", lw=1.3)
    ax.axhline(10, color="0.35", ls="--", lw=1.0)
    ax.text(uu[0], 12, "10 % tolerance", fontsize=7, color="0.35")
    ax.set_xlabel("true wind speed (m/s)", fontsize=8.5)
    ax.set_ylabel("linearised bound on wind speed (%)", fontsize=8.5)
    ax.set_title("Wind information in the mask,\nfrom the shape alone", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    x = (np.exp(PROFILE_OFFSETS) - 1) * 100
    style = {"U=3.0 law exact": ("tab:blue", "-"), "U=3.0 law uncertain": ("tab:red", "-"),
             "U=5.0 law exact": ("tab:blue", "--"), "U=5.0 law uncertain": ("tab:red", "--")}
    for k, (col, ls) in style.items():
        ax.plot(x, np.maximum(prof[k]["dchi2"], 1e-2), ls, color=col, lw=1.8,
                label=k.replace("law ", ""))
    ax.axhline(1.0, color="0.35", ls=":", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("wind error imposed (%)", fontsize=8.5)
    ax.set_ylabel(r"profile $\Delta\chi^2$", fontsize=8.5)
    ax.set_title("Only one configuration resists", fontsize=9.5)
    ax.legend(fontsize=6.6, frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Under FARSITE's own shape law the degeneracy survives in three of "
                 "four regimes", fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_farsite_lb.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    if "--reanalyse" in sys.argv:
        d = json.loads((RESULTS / "farsite_lb.json").read_text(encoding="utf-8"))
        d["asymmetry"] = asymmetry(d["profiles"])
        plot(d, d["u_saturation"], d["fisher"], d["profiles"])
        (RESULTS / "farsite_lb.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"wrote {RESULTS/'farsite_lb.json'}")
    else:
        main()
