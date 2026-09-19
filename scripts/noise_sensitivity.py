"""How good does the camera have to be, and how bad do the masks have to be?

The conclusion "read the plume" is only useful if it survives realistic sensor quality.
This sweeps both noise levels and asks where the crossover is.

It is nearly free. The Fisher information is

    F = sum_m (1 / sigma_m^2) * J_m^T J_m

so one Jacobian, computed once, evaluates every noise combination instantly -- no
further forward-model runs.

Bounds here use central finite differences, the estimator that *flatters* the mask-only
baseline (the profile likelihood shows masks do more than 30x worse than the FD bound
suggests). Using the baseline-favourable estimator keeps the comparison conservative:
the real advantage of the plume is larger than what is plotted.

Run:  python scripts/noise_sensitivity.py
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
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, forward, pack  # noqa: E402
from pyrofield.theory.identifiability import (  # noqa: E402
    crb_from_fisher,
    jacobian,
)

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
MODALITIES = ("mask", "plume", "conc")
SIG_MASK = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
SIG_PLUME = np.array([0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
TOL_U = 0.10

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = tuple(range(8, 110, 9))
    scen = Scenario(device=device, n_steps=110, mask_times=times,
                    plume_times=times, conc_times=times)
    theta = pack(TRUTH, device=device)

    print("computing the Jacobian once ...")
    J, base = jacobian(theta, scen, MODALITIES, eps=0.02, method="central")

    # Split rows by modality and precompute each block's unweighted information.
    blocks, start = {}, 0
    for m in MODALITIES:
        n = base[m].numel()
        Jm = J[start:start + n].double()
        blocks[m] = (Jm.T @ Jm).cpu().numpy()
        start += n
        print(f"  {m:6s}: {n} observations")

    conc_sigma = float(base["conc"].abs().mean().item()) * 0.05 + 2.0e-5
    iu = PARAM_NAMES.index("U")

    def crb_U(sig_mask=None, sig_plume=None, sig_conc=None):
        F = np.zeros_like(blocks["mask"])
        if sig_mask:
            F += blocks["mask"] / sig_mask**2
        if sig_plume:
            F += blocks["plume"] / sig_plume**2
        if sig_conc:
            F += blocks["conc"] / sig_conc**2
        return float(crb_from_fisher(F)[0][iu])

    M = np.zeros((len(SIG_MASK), len(SIG_PLUME)))
    Mm = np.zeros(len(SIG_MASK))
    for i, sm in enumerate(SIG_MASK):
        Mm[i] = crb_U(sig_mask=sm)
        for j, sp in enumerate(SIG_PLUME):
            M[i, j] = crb_U(sig_mask=sm, sig_plume=sp)

    print("\nCRB(U) with masks alone, by mask noise:")
    for sm, v in zip(SIG_MASK, Mm):
        print(f"  sigma_mask={sm:5.3f}  ->  {v*100:9.3f} %"
              + ("   (meets 10% tolerance)" if v < TOL_U else ""))

    print("\nCRB(U) with mask + plume  [rows: mask noise, cols: plume noise]")
    print("        " + "".join(f"{sp:>10.3f}" for sp in SIG_PLUME))
    for i, sm in enumerate(SIG_MASK):
        print(f"  {sm:5.3f} " + "".join(f"{M[i,j]*100:10.4f}" for j in range(len(SIG_PLUME))))

    # How bad can the camera get before it stops helping?
    print("\nHow noisy can the camera be and still beat masks alone?")
    for i, sm in enumerate(SIG_MASK):
        worse = np.where(M[i] >= Mm[i])[0]
        limit = SIG_PLUME[worse.min()] if len(worse) else None
        gain_at_default = Mm[i] / M[i, list(SIG_PLUME).index(0.02)]
        print(f"  sigma_mask={sm:5.3f}: plume still helps up to "
              + (f"sigma_plume={limit:.3f}" if limit else f"sigma_plume>{SIG_PLUME[-1]:.2f}")
              + f"   (gain at sigma_plume=0.02: {gain_at_default:.0f}x)")

    # Where does each option meet the operational tolerance?
    ok_mask = SIG_MASK[Mm < TOL_U]
    print(f"\n  masks alone meet the 10% tolerance for sigma_mask <= "
          + (f"{ok_mask.max():.3f}" if len(ok_mask) else "never"))
    worst_ok = [(SIG_MASK[i], SIG_PLUME[j]) for i in range(len(SIG_MASK))
                for j in range(len(SIG_PLUME)) if M[i, j] < TOL_U]
    if worst_ok:
        print(f"  mask+plume meets it even at sigma_mask={max(a for a,_ in worst_ok):.2f}, "
              f"sigma_plume={max(b for _,b in worst_ok):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))

    ax = axes[0]
    im = ax.imshow(M * 100, cmap="RdYlGn_r", norm=LogNorm(vmin=1e-3, vmax=1e2),
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(SIG_PLUME)))
    ax.set_xticklabels([f"{s:g}" for s in SIG_PLUME], fontsize=7.5)
    ax.set_yticks(range(len(SIG_MASK)))
    ax.set_yticklabels([f"{s:g}" for s in SIG_MASK], fontsize=7.5)
    ax.set_xlabel("camera noise $\\sigma_{plume}$", fontsize=8.5)
    ax.set_ylabel("mask noise $\\sigma_{mask}$", fontsize=8.5)
    ax.set_title("CRB on wind speed, mask + plume (%)", fontsize=9.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)

    ax = axes[1]
    for i, sm in enumerate(SIG_MASK):
        ax.loglog(SIG_PLUME, M[i] * 100, "o-", ms=3.5, lw=1.5,
                  label=f"$\\sigma_m$={sm:g}")
        ax.axhline(Mm[i] * 100, ls=":", lw=0.9, color=f"C{i}")
    ax.axhline(TOL_U * 100, color="k", ls="--", lw=1.1)
    ax.text(SIG_PLUME[0], TOL_U * 115, "10 % tolerance", fontsize=7.5)
    ax.set_xlabel("camera noise $\\sigma_{plume}$", fontsize=8.5)
    ax.set_ylabel("CRB on wind speed (%)", fontsize=8.5)
    ax.set_title("Solid: mask + plume.  Dotted: same mask noise, no camera.",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Does the plume advantage survive realistic sensor quality?",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    out = FIGS / "fig_noise_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    (RESULTS / "noise_sensitivity.json").write_text(json.dumps({
        "truth": TRUTH, "sigma_mask": SIG_MASK.tolist(),
        "sigma_plume": SIG_PLUME.tolist(), "conc_sigma": conc_sigma,
        "crb_U_mask_only": Mm.tolist(), "crb_U_mask_plume": M.tolist(),
        "tol_U": TOL_U,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'noise_sensitivity.json'}")


if __name__ == "__main__":
    main()
