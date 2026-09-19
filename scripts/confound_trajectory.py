"""What exactly do the masks conserve? Tracing the confound and testing it against theory.

The profile likelihood showed that fire masks do not determine wind speed: the misfit
stays flat while the wind is forced across +/-35%. Something must be absorbing that
change. The theory says exactly what:

    R_head = R0 * (1 + phi_w(U))        (masks see only this product)
    LB     = 1 + k_LB * U               (masks see only this shape)

So if the wind is forced to a wrong value U', a mask-only fit should compensate with

    R0'   = R_head_true / (1 + phi_w(U'))
    k_LB' = (LB_true - 1) / U'

and nothing else. This experiment forces U across a range, refits everything else to
the masks, and compares the fitted trajectory with those two closed-form predictions.
Agreement turns "the masks are degenerate" from an observation into an identification
of precisely which quantity the data does and does not constrain.

Run:  python scripts/confound_trajectory.py
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
from pyrofield.physics.rothermel import head_ros, length_to_breadth  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
    pack,
    unpack,
)
from pyrofield.theory.identifiability import stack_obs  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
OFFSETS = np.linspace(-0.40, 0.40, 9)
SETS = {"M": ("mask",), "M+P+C": ("mask", "plume", "conc")}
SCHEDULE = ((2.0, 4), (0.0, 12))

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot(out, Rh_true, LB_true):
    """Draw the trajectory figure from collected results (also used by --replot)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.3))
    rows_m = out["M"]["rows"]
    U = np.array([r["U"] for r in rows_m])
    uu = np.linspace(U.min(), U.max(), 200)

    ax = axes[0]
    ax.plot(U, [r["R0"] for r in rows_m], "o", ms=6, color="tab:red",
            label="mask-only fit")
    ax.plot(uu, Rh_true / (1 + 0.35 * uu**1.3), "-", color="k", lw=1.6,
            label=r"theory: $R_0=R_{head}/(1+\phi_w(U))$")
    if "M+P+C" in out:
        rm = out["M+P+C"]["rows"]
        ax.plot([r["U"] for r in rm], [r["R0"] for r in rm], "^", ms=5,
                color="tab:blue", label="all sensors fit")
    ax.plot(TRUTH["U"], TRUTH["R0"], "*", ms=15, color="tab:green", label="truth")
    ax.set_xlabel("wind speed forced to (m/s)", fontsize=8.5)
    ax.set_ylabel("fitted $R_0$ (m/s)", fontsize=8.5)
    ax.set_title("The masks trade fuel against wind\nalong the predicted curve",
                 fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(U, [r["lb_k"] for r in rows_m], "o", ms=6, color="tab:red")
    ax.plot(uu, (LB_true - 1) / uu, "-", color="k", lw=1.6,
            label=r"theory: $k_{LB}=(LB-1)/U$")
    ax.plot(TRUTH["U"], TRUTH["lb_k"], "*", ms=15, color="tab:green")
    ax.set_xlabel("wind speed forced to (m/s)", fontsize=8.5)
    ax.set_ylabel("fitted $k_{LB}$", fontsize=8.5)
    ax.set_title("and fuel shape against wind,\nalso as predicted", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[2]
    for label, c, mk in (("M", "tab:red", "o"), ("M+P+C", "tab:blue", "^")):
        if label not in out:
            continue
        rr = out[label]["rows"]
        cm = min(r["chi2"] for r in rr)
        ax.plot([r["U"] for r in rr], [max(r["chi2"] - cm, 1e-2) for r in rr],
                mk + "-", ms=5, color=c, label=label, lw=1.6)
    ax.axhline(1.0, color="0.35", ls="--", lw=1.0)
    ax.text(U.min(), 1.4, r"$\Delta\chi^2=1$", fontsize=7.5, color="0.35")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e6)
    ax.set_xlabel("wind speed forced to (m/s)", fontsize=8.5)
    ax.set_ylabel(r"profile $\Delta\chi^2$", fontsize=8.5)
    ax.set_title("What it costs to be wrong about\nthe wind", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    p = FIGS / "fig_confound_trajectory.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def main():
    if "--replot" in sys.argv:
        d = json.loads((RESULTS / "confound_trajectory.json").read_text(encoding="utf-8"))
        any_set = next(iter(d["sets"].values()))
        plot(d["sets"], any_set["Rhead_true"], any_set["LB_true"])
        return

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = tuple(range(8, 110, 9))
    scen = Scenario(device=device, n_steps=110, mask_times=times,
                    plume_times=times, conc_times=times)
    theta0 = pack(TRUTH, device=device)
    gen = torch.Generator(device=device).manual_seed(7)
    target = add_noise(forward(theta0, scen), gen)

    iu = PARAM_NAMES.index("U")
    free = [i for i in range(len(PARAM_NAMES)) if i != iu]

    T = lambda v: torch.tensor(float(v), device=device)  # noqa: E731
    Rh_true = float(head_ros(T(TRUTH["R0"]), T(TRUTH["U"])))
    LB_true = float(length_to_breadth(T(TRUTH["U"]), T(TRUTH["lb_k"])))
    print(f"truth: R_head = {Rh_true:.5f} m/s,  LB = {LB_true:.4f}")

    out = {}
    for label, keys in SETS.items():
        sig = {k: noise_sigma_for(target, k) for k in keys}
        w = fixed_weights(target, keys, sig)
        y = stack_obs(target, keys) * w

        def chi2(t):
            return float((((stack_obs(forward(t, scen), keys) * w) - y) ** 2).sum())

        print(f"\n{label}")
        print(f"  {'U/U_true':>9s} {'R0_fit':>9s} {'R0_pred':>9s} {'kLB_fit':>9s} "
              f"{'kLB_pred':>9s} {'R_head':>9s} {'LB':>7s} {'chi2-min':>10s}")
        rows = []
        order = sorted(range(len(OFFSETS)), key=lambda i: abs(OFFSETS[i]))
        warm = {}
        for i in order:
            d = float(OFFSETS[i])
            near = min(warm, key=lambda k: abs(OFFSETS[k] - d)) if warm else None
            start = (warm[near].clone() if near is not None else theta0.clone())
            start[iu] = theta0[iu] + d
            res = levenberg_marquardt(start, theta0, target, scen, keys,
                                      schedule=SCHEDULE, free_idx=free)
            warm[i] = res.theta_hat.clone()
            p = unpack(res.theta_hat)
            Uf = float(p["U"])
            R0f, kf = float(p["R0"]), float(p["lb_k"])
            # Closed-form compensation the theory predicts.
            R0_pred = Rh_true / float(1.0 + 0.35 * Uf**1.3)
            k_pred = (LB_true - 1.0) / Uf
            rows.append({
                "offset": d, "U": Uf, "R0": R0f, "lb_k": kf,
                "R0_pred": R0_pred, "lb_k_pred": k_pred,
                "R_head": float(head_ros(T(R0f), T(Uf))),
                "LB": float(length_to_breadth(T(Uf), T(kf))),
                "theta_w": float(p["theta_w"]), "chi2": chi2(res.theta_hat),
            })
        rows.sort(key=lambda r: r["offset"])
        cmin = min(r["chi2"] for r in rows)
        for r in rows:
            print(f"  {r['U']/TRUTH['U']:9.3f} {r['R0']:9.5f} {r['R0_pred']:9.5f} "
                  f"{r['lb_k']:9.4f} {r['lb_k_pred']:9.4f} {r['R_head']:9.5f} "
                  f"{r['LB']:7.3f} {r['chi2']-cmin:10.2f}")

        # How well does the fit track the predicted compensation?
        r0 = np.array([r["R0"] for r in rows])
        r0p = np.array([r["R0_pred"] for r in rows])
        kk = np.array([r["lb_k"] for r in rows])
        kp = np.array([r["lb_k_pred"] for r in rows])
        rh = np.array([r["R_head"] for r in rows])
        lb = np.array([r["LB"] for r in rows])
        err_r0 = float(np.median(np.abs(r0 / r0p - 1)))
        err_k = float(np.median(np.abs(kk / kp - 1)))
        cv_rh = float(np.std(rh) / np.mean(rh))
        cv_lb = float(np.std(lb) / np.mean(lb))
        print(f"  median |R0_fit / R0_pred - 1| = {err_r0*100:.1f}%")
        print(f"  median |kLB_fit / kLB_pred - 1| = {err_k*100:.1f}%")
        print(f"  R_head conserved along the trajectory: CV = {cv_rh*100:.2f}%"
              f"   (truth {Rh_true:.5f})")
        print(f"  LB     conserved along the trajectory: CV = {cv_lb*100:.2f}%"
              f"   (truth {LB_true:.4f})")
        out[label] = {"rows": rows, "err_R0": err_r0, "err_lbk": err_k,
                      "cv_Rhead": cv_rh, "cv_LB": cv_lb,
                      "Rhead_true": Rh_true, "LB_true": LB_true}

    plot(out, Rh_true, LB_true)

    (RESULTS / "confound_trajectory.json").write_text(
        json.dumps({"truth": TRUTH, "sets": out}, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'confound_trajectory.json'}")


if __name__ == "__main__":
    main()
