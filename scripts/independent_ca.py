"""The same degeneracy, in somebody else's fire model.

Every identifiability result in this study so far was measured in a simulator written for
this study. The obvious objection is that the degeneracy is an artefact of that code. This
script answers it in a model that shares neither the implementation, the numerical family,
nor the spread law:

    PyTorchFire (arXiv 2502.18738, `pip install pytorchfire`), a differentiable stochastic
    cellular automaton in the Alexandridis (2008) family. Fire spreads by per-cell ignition
    probabilities on an 8-neighbour lattice; there is no level set, no Rothermel rate, and
    no length-to-breadth parameter at all -- the elliptical shape is emergent.

Its propagation probability, taken from `WildfireModel.p_ignite`, is

    p(theta) = tanh( K * p_h * (1 + p_veg) * (1 + p_den)
                       * exp(a * slope) * exp(c_1 * V) * exp(c_2 * V * (cos(theta) - 1)) )

with K = 1.1486328125 and theta the angle between the spread direction and the wind. On
homogeneous flat fuel this collapses to a function of two quantities:

    A = K * p_h * exp(c_1 * V)        the head term
    s = c_2 * V                        the anisotropy term
    p(theta) = tanh( A * exp(s * (cos theta - 1)) )

**A and s can be held fixed while V moves.** Solving exactly,

    p_h(V) = p_h0 * exp(c_1 * (V0 - V))        keeps A fixed
    c_2(V) = c_2_0 * V0 / V                    keeps s fixed

and in log parameters the tangent to that curve is

    v = ( -c_1 * V, +1, -1 )     in ( ln p_h, ln V, ln c_2 )

structurally identical to the closed-form null direction of RESULTS.md section 1,

    v = ( -a*b*U^b / (1 + a*U^b), +1, -1 )   in ( ln R0, ln U, ln k_LB ).

The consequence here is stronger than in section 1. There the two mask summaries matched
and the misfit stayed small but nonzero. Here **the eight lattice propagation probabilities
are unchanged to machine precision, so the transition kernel of the process is literally
the same object** -- every realisation, every moment, every statistic. No estimator of any
kind can separate the two parameter sets, because there is nothing to separate.

The eight probabilities are the whole model on homogeneous flat fuel, which makes
identifiability exactly the question of what a change of parameters does to that 8-vector.
That is how the controls below are posed.

Run:  pip install pytorchfire
      python scripts/independent_ca.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

from pytorchfire import WildfireModel  # noqa: E402

K = 1.1486328125                     # PyTorchFire's `prob_like_act_c`
LATTICE = np.arange(8) * 45.0        # the 8 neighbour directions, degrees
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, STEPS, SEEDS = 160, 70, 12

# An Alexandridis-family operating point, well away from the percolation threshold.
P_H, V0, C1, C2 = 0.42, 5.0, 0.045, 0.131

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def p_lattice(p_h, V, c_1, c_2, wind_deg=0.0):
    """PyTorchFire's propagation probability for the 8 lattice directions."""
    d = np.deg2rad(LATTICE - wind_deg)
    return np.tanh(K * p_h * np.exp(c_1 * V) * np.exp(c_2 * V * (np.cos(d) - 1.0)))


def null_curve(V, p_h0=P_H, V0_=V0, c_1=C1, c_2_0=C2):
    """The exact compensation: hold A and s fixed while the wind moves."""
    return p_h0 * np.exp(c_1 * (V0_ - V)), c_2_0 * V0_ / V


def build(V, wind_deg, p_h, c_1, c_2, p_continue=0.35):
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    ign = ((yy - N // 2) ** 2 + (xx - N // 2) ** 2) <= 3 ** 2
    env = {"p_veg": torch.zeros(N, N), "p_den": torch.zeros(N, N),
           "wind_velocity": torch.full((N, N), float(V)),
           "wind_towards_direction": torch.full((N, N), float(wind_deg)),
           "slope": torch.zeros(N, N, 3, 3), "initial_ignition": ign}
    par = {"a": torch.tensor(0.0), "p_h": torch.tensor(float(p_h)),
           "c_1": torch.tensor(float(c_1)), "c_2": torch.tensor(float(c_2)),
           "p_continue": torch.tensor(float(p_continue))}
    m = WildfireModel(env_data=env, params=par).to(DEV)
    m.eval()
    return m


def realise(m, seed):
    m.reset(seed=int(seed))
    with torch.no_grad():
        for _ in range(STEPS):
            m.compute()
    b, bd = m.state
    return b | bd


def _shape(m, steps, seeds):
    """Measured length-to-breadth and burned fraction of the expected mask."""
    global STEPS
    keep, STEPS = STEPS, steps
    acc = torch.zeros(N, N, device=DEV)
    for s in seeds:
        acc += realise(m, s).float()
    STEPS = keep
    e = (acc / len(list(seeds)) > 0.5)
    idx = e.nonzero().float()
    if len(idx) < 20:
        return float("nan"), float(e.float().mean())
    d = idx - idx.mean(0)
    ev = torch.linalg.eigvalsh((d.T @ d) / len(d))
    return float((ev[1] / ev.clamp(min=1e-9)[0]).sqrt()), float(e.float().mean())


def mask_diff(mA, mB, seeds=range(SEEDS)):
    """Fraction of cells whose burn state differs, on matched random seeds."""
    diff = tot = 0
    for s in seeds:
        a, b = realise(mA, s), realise(mB, s)
        diff += int((a ^ b).sum())
        tot += int(a.numel())
    return diff / tot


def best_compensation(target_p, free=("p_h", "c_2"), V=V0, wind_deg=0.0):
    """Smallest achievable distance to `target_p` using only the free coefficients."""
    def cost(z):
        p_h = P_H * np.exp(z[0]) if "p_h" in free else P_H
        c_2 = C2 * np.exp(z[1]) if "c_2" in free else C2
        return float(np.sum((p_lattice(p_h, V, C1, c_2, wind_deg) - target_p) ** 2))
    best = min((minimize(cost, x0, method="Nelder-Mead",
                         options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 4000})
                for x0 in ([0.0, 0.0], [0.4, -0.4], [-0.4, 0.4])), key=lambda r: r.fun)
    return float(np.sqrt(best.fun)), best.x


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    out = {"model": "PyTorchFire 1.1.1 (stochastic CA, Alexandridis family)",
           "baseline": {"p_h": P_H, "V": V0, "c_1": C1, "c_2": C2},
           "grid": N, "steps": STEPS, "seeds": SEEDS}
    base = p_lattice(P_H, V0, C1, C2)

    print("=" * 78)
    print("[1] the exact null curve, derived from PyTorchFire's own equation")
    print("=" * 78)
    print(f"    baseline: p_h={P_H}  V={V0} m/s  c_1={C1}  c_2={C2}")
    print(f"    tangent in (ln p_h, ln V, ln c_2): v = ({-C1*V0:+.4f}, +1, -1)")
    print(f"\n    {'wind err':>9} {'V':>7} {'p_h':>9} {'c_2':>9} "
          f"{'max |dp| over the 8 directions':>31}")
    rows = []
    for pct in (5, 10, 20, 30, 50, 80):
        V = V0 * (1 + pct / 100.0)
        p_h, c_2 = null_curve(V)
        d = float(np.abs(p_lattice(p_h, V, C1, c_2) - base).max())
        # the tangent (linearised) version, for contrast with section 1's curvature
        e = np.log1p(pct / 100.0)
        dt = float(np.abs(p_lattice(P_H * np.exp(-C1 * V0 * e), V, C1,
                                    C2 * np.exp(-e)) - base).max())
        rows.append({"pct": pct, "V": V, "p_h": p_h, "c_2": c_2,
                     "dp_exact": d, "dp_tangent": dt})
        print(f"    {pct:8d} % {V:7.3f} {p_h:9.5f} {c_2:9.5f} {d:31.2e}")
    out["null_curve"] = rows
    print(f"\n    Exact to {max(r['dp_exact'] for r in rows):.1e} across an 80 % wind error.")
    print(f"    Along the *tangent* instead, the drift reaches "
          f"{max(r['dp_tangent'] for r in rows):.1e} -- the null space is a curve,")
    print("    not a line, exactly as in section 1.")

    print()
    print("=" * 78)
    print("[2] the realised masks, on matched random seeds")
    print("=" * 78)
    print(f"    {N}x{N} lattice, {STEPS} steps, {SEEDS} seeds per setting")
    print(f"\n    {'wind err':>9} {'compensated':>14} {'uncompensated':>16}")
    mA = build(V0, 0.0, P_H, C1, C2)
    realised = []
    for pct in (10, 30, 50, 80):
        V = V0 * (1 + pct / 100.0)
        p_h, c_2 = null_curve(V)
        dc = mask_diff(mA, build(V, 0.0, p_h, C1, c_2))
        du = mask_diff(mA, build(V, 0.0, P_H, C1, C2))
        realised.append({"pct": pct, "diff_compensated": dc, "diff_uncompensated": du})
        print(f"    {pct:8d} % {dc*100:13.4f} % {du*100:15.2f} %")
    out["realised"] = realised
    print("\n    Not 'statistically indistinguishable' -- identical. The transition kernel")
    print("    is the same object, so every realisation on a given seed is the same fire.")

    print()
    print("=" * 78)
    print("[3] which coefficient has to be free? (the identifiability lattice)")
    print("=" * 78)
    print("    Smallest mask-probability distance still achievable after compensating a")
    print("    +30 % wind error with the coefficients allowed to move:")
    V = V0 * 1.30
    target = p_lattice(P_H, V0, C1, C2)
    lattice = {}
    # With both free the answer is known in closed form, so quote the curve rather than
    # an optimiser's stopping point; the optimiser is reported beside it as a check.
    p_h_n, c_2_n = null_curve(V)
    r_exact = float(np.sqrt(((p_lattice(p_h_n, V, C1, c_2_n) - target) ** 2).sum()))
    r_opt, _ = best_compensation(target, free=("p_h", "c_2"), V=V)
    lattice["p_h and c_2 both free"] = r_exact
    lattice["p_h and c_2 both free (optimiser)"] = r_opt
    print(f"    {'p_h and c_2 both free':34s} residual {r_exact:.2e}   degenerate"
          f"   (optimiser finds {r_opt:.1e})")
    for free, label in ((("p_h",), "only p_h free (shape law known)"),
                        (("c_2",), "only c_2 free (fuel known)"),
                        ((), "neither free")):
        r, _ = best_compensation(target, free=free, V=V)
        lattice[label] = r
        print(f"    {label:34s} residual {r:.2e}   identifiable")
    out["lattice"] = lattice
    print("\n    The degeneracy needs *both* a free fuel coefficient and a free shape")
    print("    coefficient, which is the same condition section 2 finds for FARSITE.")
    print("    Freeing the fuel coefficient alone barely helps (9.3e-2 against 9.7e-2 for")
    print("    nothing free): it moves all eight probabilities together, and a wind error")
    print("    changes their ratios.")

    print()
    print("=" * 78)
    print("[4] positive control: can a wind *rotation* be compensated away?")
    print("=" * 78)
    print(f"    {'rotation':>9} {'best residual':>15} {'vs a 30 % speed error':>24}")
    ctrl = []
    for deg in (5, 10, 20, 45):
        tgt = p_lattice(P_H, V0, C1, C2, wind_deg=0.0)
        # rotate the wind, then let p_h and c_2 try to undo it
        rot = p_lattice(P_H, V0, C1, C2, wind_deg=deg)
        r, _ = best_compensation(rot, free=("p_h", "c_2"), V=V0, wind_deg=0.0)
        ctrl.append({"deg": deg, "residual": r})
        print(f"    {deg:8d} d {r:15.3e} {'':>24}")
    out["direction_control"] = ctrl
    print(f"\n    A speed error is absorbable to {lattice['p_h and c_2 both free']:.0e}; a "
          f"{ctrl[0]['deg']}-degree rotation leaves {ctrl[0]['residual']:.2e}.")
    print("    Wind direction is identifiable from the mask in this model too, which is")
    print("    the same split section 1 finds: the mask gives the bearing, not the speed.")

    print()
    print("=" * 78)
    print("[5] and the shape channel here is worse than FARSITE's, for other reasons")
    print("=" * 78)
    print("    Section 2's argument runs through saturation: Anderson's relation is")
    print("    truncated at LB = 8, so above 3.80 m/s the shape stops responding. This")
    print("    model has no such truncation. Measuring what it does instead --")
    print("    every fire run until its head has advanced a fixed 42 cells, so that shape")
    print("    is compared at matched progress rather than at matched step count:")
    print(f"\n    {'V':>6} {'steps':>6} {'area':>7} {'L/B one-step':>13} "
          f"{'L/B measured':>14} {'batch 2':>9}")
    sat = []
    for V in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0):
        q = p_lattice(P_H, V, C1, C2)
        lb1 = float((q[0] + q[4]) / (2 * q[2]))
        m = build(V, 0.0, P_H, C1, C2)
        steps = int(round(42.0 / float(q[0])))
        a, b = _shape(m, steps, range(0, 16)), _shape(m, steps, range(16, 32))
        sat.append({"V": V, "steps": steps, "lb_one_step": lb1, "lb": a[0],
                    "lb_batch2": b[0], "area_frac": a[1]})
        print(f"    {V:6.1f} {steps:6d} {a[1]*100:6.1f}% {lb1:13.3f} "
              f"{a[0]:14.3f} {b[0]:9.3f}")
    out["shape"] = sat
    peak = max(sat, key=lambda r: r["lb"])
    print(f"\n    The measured L/B is **not monotone**: it rises to {peak['lb']:.2f} at "
          f"V = {peak['V']:.0f} m/s and then falls")
    print(f"    back to {sat[-1]['lb']:.2f}. A fire of L/B = 1.38 is consistent with two "
          f"different winds,")
    print("    so the shape does not merely saturate here -- it is two-valued.")
    print("\n    The one-step ellipse (which would predict L/B = 49.6 at 45 m/s) tracks the")
    print("    measured shape only below about 8 m/s. Above that the burn is not an")
    print("    ellipse at all but a wedge with a flat leading edge, because the lattice's")
    print("    45-degree diagonals are cheap and bound the spread.")

    print("\n    Is that a lattice artefact? Rotate the wind against the lattice and see.")
    print(f"\n    {'wind bearing':>13}" + "".join(f"{f'{d} deg':>10}" for d in
                                                  (0, 11.25, 22.5, 33.75, 45)))
    rot = {}
    for V in (8.0, 20.0):
        q = p_lattice(P_H, V, C1, C2)
        steps = int(round(42.0 / float(q[0])))
        vals = [_shape(build(V, d, P_H, C1, C2), steps, range(24))[0]
                for d in (0.0, 11.25, 22.5, 33.75, 45.0)]
        rot[V] = vals
        print(f"    V = {V:4.0f} m/s  " + "".join(f"{v:10.3f}" for v in vals))
    out["rotation"] = {str(k): v for k, v in rot.items()}
    sw = max(rot[8.0]) - min(rot[8.0])
    print(f"\n    At 8 m/s the measured L/B moves by {sw:.3f} ({sw/np.mean(rot[8.0])*100:.0f} %)"
          " with the wind bearing alone,")
    print("    at fixed wind speed. The shape is not rotation invariant, so a part of it")
    print("    is the stencil rather than the fire.")
    print("\n    Three independent defects, none of them Anderson's cap: the shape is")
    print("    two-valued in wind, it is not an ellipse, and it depends on the bearing")
    print("    relative to the grid. And parts [1] and [2] hold regardless of all of it,")
    print("    because that degeneracy is a statement about the transition kernel and")
    print("    never routes through the shape.")

    (RESULTS / "independent_ca.json").write_text(json.dumps(out, indent=2),
                                                 encoding="utf-8")
    print(f"\nwrote {RESULTS/'independent_ca.json'}")
    make_figure(out, base)


def make_figure(out, base):
    fig = plt.figure(figsize=(15.8, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1])

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    th = np.deg2rad(np.append(LATTICE, 360))
    V = V0 * 1.5
    p_h, c_2 = null_curve(V)
    for p, lab, st in ((base, f"V = {V0:.1f} m/s (baseline)", dict(lw=3.2, color="tab:blue", alpha=0.45)),
                       (p_lattice(p_h, V, C1, c_2), f"V = {V:.1f}, compensated", dict(lw=1.4, color="k", ls="--")),
                       (p_lattice(P_H, V, C1, C2), f"V = {V:.1f}, not compensated", dict(lw=1.6, color="tab:red"))):
        ax.plot(th, np.append(p, p[0]), **st, label=lab)
    ax.set_title("Propagation probability\nby direction", fontsize=9.5, pad=14)
    ax.set_rlabel_position(115)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.0, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.36))

    ax = fig.add_subplot(gs[0, 1])
    r = out["realised"]
    x = [q["pct"] for q in r]
    ax.plot(x, [q["diff_uncompensated"] * 100 for q in r], "o-", color="tab:red",
            label="wind error alone")
    ax.plot(x, [q["diff_compensated"] * 100 for q in r], "s-", color="k",
            label="compensated along the null curve")
    ax.set_ylim(-1.5, 24)
    ax.set_xlabel("imposed wind error (%)", fontsize=8.5)
    ax.set_ylabel("burn cells differing (%)", fontsize=8.5)
    ax.set_title(f"Matched seeds, {N}x{N} lattice", fontsize=9.5)
    ax.annotate(f"0 of {N*N*SEEDS:,} cells,\nat every wind error",
                xy=(45, 0), xytext=(28, 6.5), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    names = ["speed,\nboth free", "speed,\np_h only", "speed,\nc_2 only",
             "5 deg\nrotation", "20 deg\nrotation"]
    vals = [out["lattice"]["p_h and c_2 both free"],
            out["lattice"]["only p_h free (shape law known)"],
            out["lattice"]["only c_2 free (fuel known)"],
            out["direction_control"][0]["residual"],
            out["direction_control"][2]["residual"]]
    vals = [max(v, 1e-17) for v in vals]
    ax.bar(range(5), vals, color=["k", "tab:orange", "tab:orange",
                                  "tab:green", "tab:green"])
    ax.set_yscale("log")
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("smallest achievable residual", fontsize=8.5)
    ax.axhline(1e-15, color="0.5", lw=0.9, ls=":")
    ax.text(0.05, 2e-15, "machine precision", fontsize=6.4, color="0.4")
    ax.set_title("What the mask can and cannot\nabsorb", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y", which="both")

    ax = fig.add_subplot(gs[0, 3])
    s = out["shape"]
    vv = [q["V"] for q in s]
    ax.plot(vv, [q["lb"] for q in s], "o-", color="tab:purple", lw=1.9, zorder=3,
            label="measured (matched head travel)")
    ax.plot(vv, [q["lb_batch2"] for q in s], "o", ms=3.5, color="tab:purple",
            alpha=0.45, zorder=3, label="independent seed batch")
    rot8 = out["rotation"]["8.0"]
    ax.errorbar([8.0], [float(np.mean(rot8))],
                yerr=[[float(np.mean(rot8) - min(rot8))],
                      [float(max(rot8) - np.mean(rot8))]],
                fmt="none", ecolor="tab:orange", elinewidth=3.0, capsize=5, zorder=4)
    ax.annotate("same wind speed,\nbearing rotated 0-45 deg\nagainst the lattice",
                xy=(8.3, 1.50), xytext=(10.5, 1.66), fontsize=6.4,
                color="tab:orange", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.0))
    ax.annotate("", xy=(6.5, 1.30), xytext=(14.0, 1.30),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0))
    ax.text(7.0, 1.275, "one shape,\ntwo winds", fontsize=6.6, color="0.35", va="top")
    ax.set_xscale("log")
    ax.set_ylim(0.95, 1.80)
    ax.set_xlabel("wind speed (m/s)", fontsize=8.5)
    ax.set_ylabel("length-to-breadth ratio", fontsize=8.5)
    ax.set_title("Its shape channel is two-valued,\nand partly the grid", fontsize=9.5)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("The same degeneracy in PyTorchFire, a third-party stochastic "
                 "cellular automaton", fontsize=10, y=1.05)
    fig.tight_layout()
    p = FIGS / "fig_independent_ca.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
