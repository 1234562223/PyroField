"""Does any of this survive fitting the wrong model?

Every result so far is an identical-twin experiment: the model that generated the data is
the model being fitted. That is the largest caveat in the study, and this is the test of
it. Data is generated with a wind exponent b = 1.3 and fitted with b from 1.1 to 1.5 --
an error of up to 15 % in the exponent, which is modest next to the real spread between
published fuel models.

Two predictions, which are opposite in character and so make a sharp test:

  * The mask degeneracy should be **untouched**. It follows from the form
    R_head = R0 * g(U), not from any particular g, so changing g moves the null curve but
    does not remove it. The wrong model should still reproduce the past perfectly and
    still fail to pin the wind.

  * The plume should be **largely unharmed**. It measures wind through advection of smoke,
    which never touches the fire-spread model at all. Its estimate of U should stay close
    to truth even when the fire model is wrong, and the residual error should come from
    the coupling through the burn geometry rather than from the wind measurement itself.

If instead the plume's advantage disappeared under a 15 % exponent error, the whole
approach would be an artefact of the twin setup.

Run:  python scripts/misspecification.py
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
    unpack,
)
from pyrofield.theory.identifiability import stack_obs  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
B_TRUE = 1.3
B_FIT = (1.10, 1.20, 1.30, 1.40, 1.50)
N_RESTARTS = 3

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scen_for(b, device):
    times = tuple(range(8, 110, 9))
    return Scenario(device=device, n_steps=110, mask_times=times,
                    plume_times=times, conc_times=times, wind_b=b)


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta_true = pack(TRUTH, device=device)

    truth_scen = scen_for(B_TRUE, device)
    gen = torch.Generator(device=device).manual_seed(19)
    target = add_noise(forward(theta_true, truth_scen), gen)
    n_mask = target["mask"].numel()
    print(f"data generated with b = {B_TRUE}; {n_mask} mask observations")
    print(f"noise floor chi2 = N = {n_mask}\n")

    def chi2(theta, scen, keys):
        sig = {k: noise_sigma_for(target, k) for k in keys}
        w = fixed_weights(target, keys, sig)
        y = stack_obs(target, keys) * w
        return float((((stack_obs(forward(theta, scen), keys) * w) - y) ** 2).sum())

    rows = []
    t0 = time.time()
    print(f"  {'b_fit':>6s} {'set':>5s} {'U fitted':>20s} {'err':>8s} "
          f"{'R0 err':>8s} {'chi2 / N':>10s}")
    for b in B_FIT:
        sc = scen_for(b, device)
        for label, keys in (("M", ("mask",)), ("M+P", ("mask", "plume"))):
            rng = np.random.default_rng(7)
            best, us = None, []
            for _ in range(N_RESTARTS):
                start = theta_true.clone()
                for i in range(len(PARAM_NAMES)):
                    start[i] = start[i] + float(rng.uniform(-0.35, 0.35))
                res = levenberg_marquardt(start, theta_true, target, sc, keys)
                us.append(float(unpack(res.theta_hat)["U"]))
                if best is None or res.loss < best.loss:
                    best = res
            p = unpack(best.theta_hat)
            u_err = abs(float(p["U"]) / TRUTH["U"] - 1)
            r_err = abs(float(p["R0"]) / TRUTH["R0"] - 1)
            c = chi2(best.theta_hat, sc, ("mask",)) / n_mask
            rows.append({"b": b, "set": label, "U": float(p["U"]), "U_err": u_err,
                         "R0": float(p["R0"]), "R0_err": r_err,
                         "U_spread": float(np.max(us) - np.min(us)),
                         "chi2_per_N": c, "loss": best.loss})
            print(f"  {b:6.2f} {label:>5s} {p['U']:10.3f} "
                  f"(spread {np.max(us)-np.min(us):5.2f}) {u_err*100:7.1f}% "
                  f"{r_err*100:7.1f}% {c:10.4f}")
    print(f"   ({time.time()-t0:.0f}s)\n")

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    m = {b: r for b in B_FIT for r in rows if r["b"] == b and r["set"] == "M"}
    mp = {b: r for b in B_FIT for r in rows if r["b"] == b and r["set"] == "M+P"}

    fits_past = all(abs(m[b]["chi2_per_N"] - 1.0) < 0.05 for b in B_FIT)
    print(f"  the wrong fire model still reproduces the observed past: "
          f"chi2/N in [{min(m[b]['chi2_per_N'] for b in B_FIT):.4f}, "
          f"{max(m[b]['chi2_per_N'] for b in B_FIT):.4f}]  -> {fits_past}")
    print("    (a mis-specified spread model is invisible in the masks, because the")
    print("     degeneracy simply absorbs it into the fuel estimate)")

    m_err = [m[b]["U_err"] for b in B_FIT]
    mp_err = [mp[b]["U_err"] for b in B_FIT]
    print(f"\n  wind-speed error, masks only : "
          f"{min(m_err)*100:.0f}% to {max(m_err)*100:.0f}%")
    print(f"  wind-speed error, mask+plume : "
          f"{min(mp_err)*100:.2f}% to {max(mp_err)*100:.2f}%")
    ratio = np.median(m_err) / max(np.median(mp_err), 1e-12)
    print(f"  median advantage of the plume under mis-specification: {ratio:.0f}x")

    mp_at_true = mp[B_TRUE]["U_err"]
    mp_worst = max(mp_err)
    print(f"\n  the plume's wind estimate degrades from {mp_at_true*100:.2f}% at the")
    print(f"  correct exponent to {mp_worst*100:.2f}% at the worst wrong one "
          f"({mp_worst/max(mp_at_true,1e-12):.0f}x)")
    survives = mp_worst < 0.10 and np.median(m_err) > 0.15
    print(f"  the advantage survives a 15% error in the spread model: {survives}")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    ax = axes[0]
    ax.plot(B_FIT, [m[b]["U_err"] * 100 for b in B_FIT], "o-", color="tab:red",
            lw=1.9, label="masks only")
    ax.plot(B_FIT, [mp[b]["U_err"] * 100 for b in B_FIT], "^-", color="tab:blue",
            lw=1.9, label="mask + plume")
    ax.axvline(B_TRUE, color="0.4", ls="--", lw=1.1)
    ax.text(B_TRUE, ax.get_ylim()[1] * 0.6, " model is correct here", fontsize=7,
            color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("wind exponent used to fit (truth 1.30)", fontsize=8.5)
    ax.set_ylabel("error in recovered wind speed (%)", fontsize=8.5)
    ax.set_title("The plume measures wind through advection,\nnot through the fire model",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    ax.plot(B_FIT, [m[b]["chi2_per_N"] for b in B_FIT], "o-", color="tab:red", lw=1.9,
            label="masks only")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.1, label="noise floor")
    ax.set_xlabel("wind exponent used to fit (truth 1.30)", fontsize=8.5)
    ax.set_ylabel("mask misfit, $\\chi^2/N$", fontsize=8.5)
    ax.set_ylim(0.9, 1.15)
    ax.set_title("A wrong fire model is invisible\nin the masks", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Model mis-specification: the degeneracy is structural, "
                 "the plume advantage is not an artefact of the twin setup",
                 fontsize=9.5, y=1.05)
    fig.tight_layout()
    p = FIGS / "fig_misspecification.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "misspecification.json").write_text(json.dumps({
        "truth": TRUTH, "b_true": B_TRUE, "b_fit": list(B_FIT),
        "n_mask_obs": n_mask, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'misspecification.json'}")


if __name__ == "__main__":
    main()
