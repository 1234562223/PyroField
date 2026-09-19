"""Figures for the Week-0 identifiability study.

Produces, from ``results/week0.json``:

  fig_sensor_sufficiency.png  -- which sensor subset determines which state
                                 component (plan figure F4)
  fig_fisher_spectra.png      -- Fisher eigenvalue spectra; flat-bottomed spectra
                                 are confounds
  fig_recovery.png            -- MLE recovery error against the Cramer-Rao bound
  fig_scenario.png            -- the synthetic fire: masks, plume view, geometry

Run:  python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

PRETTY = {"R0": r"$R_0$", "U": r"$U$", "theta_w": r"$\theta_w$",
          "lb_k": r"$k_{LB}$", "w_buoy": r"$w_b$", "Q": r"$Q$"}

# Operational tolerances: what "good enough to act on" means per quantity.
# Fractional for the positive parameters, radians for the wind direction.
TOLERANCE = {"R0": 0.10, "U": 0.10, "theta_w": np.radians(5.0),
             "lb_k": 0.15, "w_buoy": 0.15, "Q": 0.15}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})


def load():
    for name in ("week0.json", "week0_quick.json"):
        p = RESULTS / name
        if p.exists():
            print(f"reading {p}")
            return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit("no results/week0*.json -- run scripts/week0_gonogo.py first")


def fig_sensor_sufficiency(d):
    """Sensor subsets x state components, at an early and a late observation window.

    Values are linearised (Cramér–Rao) lower bounds computed with central finite
    differences, which is the estimator most *favourable* to the mask-only baseline
    (the profile likelihood shows masks do far worse than this). Using the optimistic
    estimator for the baseline keeps the comparison conservative.
    """
    import torch

    from pyrofield.sim.synthetic import Scenario, pack
    from pyrofield.theory.identifiability import crb_from_fisher, fisher_all_subsets

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = d["param_names"]
    theta = pack(d["truth"], device=device)
    subsets = [("mask",), ("mask", "plume"), ("mask", "conc"),
               ("mask", "plume", "conc")]
    labels = ["M", "M+P", "M+C", "M+P+C"]
    tol = np.array([TOLERANCE[p] for p in params])

    panels = []
    for title, T in (("early: 9 min of observation", 36),
                     ("late: 27.5 min of observation", 110)):
        times = tuple(t for t in range(8, T, 9)) or (T - 1,)
        scen = Scenario(device=device, n_steps=T, mask_times=times,
                        plume_times=times, conc_times=times)
        Fs = fisher_all_subsets(theta, scen, eps=0.02, method="central")
        M = np.array([crb_from_fisher(Fs[s])[0] for s in subsets])
        panels.append((title, M))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 2.9))
    for ax, (title, M) in zip(axes, panels):
        ratio = M / tol[None, :]
        im = ax.imshow(ratio, cmap="RdYlGn_r",
                       norm=LogNorm(vmin=1e-2, vmax=1e3), aspect="auto")
        for i in range(len(labels)):
            for j in range(len(params)):
                v, r = M[i, j], ratio[i, j]
                txt = f"{v:.3f}" if v < 100 else f"{v:.0e}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.2,
                        color="white" if (r > 30 or r < 0.05) else "black")
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([PRETTY[p] for p in params], fontsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("physical state component", fontsize=8.5)
        ax.grid(False)
        ax.set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)

    fig.suptitle(
        "Sensor sufficiency: linearised (Cramér–Rao) lower bound on each state component\n"
        "colour = $\\sigma$ / operational tolerance; green = determined. Masks alone are "
        "red early and only turn green once the fire is large —\n"
        "and even that green is illusory: the profile likelihood puts the true mask-only "
        "$\\sigma(U)$ above 73 % at 27.5 min (RESULTS.md §5).",
        fontsize=9, y=1.19)
    fig.tight_layout()
    out = FIGS / "fig_sensor_sufficiency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_fisher_spectra(d):
    names = list(d["fisher"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

    for n in names:
        ev = np.array(d["fisher"][n]["eigvals"])
        axes[0].semilogy(range(1, len(ev) + 1), ev, "o-", label=n.split("(")[0].strip(), ms=4)
    axes[0].set_xlabel("eigenvalue index (ascending)")
    axes[0].set_ylabel("Fisher eigenvalue")
    axes[0].set_title("Full 6-parameter Fisher spectrum", fontsize=9.5)
    axes[0].legend(fontsize=7.5, frameon=False)

    if "fisher_fire_block" in d:
        fb = d["fisher_fire_block"]
        for n in names:
            ev = np.array(fb[n]["eigvals"])
            axes[1].semilogy(range(1, len(ev) + 1), ev, "s-",
                             label=n.split("(")[0].strip(), ms=4)
        axes[1].set_xlabel("eigenvalue index (ascending)")
        axes[1].set_title("Fire-block Fisher spectrum\n"
                          "($R_0$, $U$, $\\theta_w$, $k_{LB}$ only)", fontsize=9.5)
        axes[1].legend(fontsize=7.5, frameon=False)

    fig.tight_layout()
    out = FIGS / "fig_fisher_spectra.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_recovery(d):
    names = list(d["fisher"].keys())
    params = d["param_names"]
    inv = d["inversion"]
    if not inv or "best" not in next(iter(inv.values())):
        print("  (skipping fig_recovery: results predate the best-of-restarts format)")
        return

    fig, axes = plt.subplots(1, len(params), figsize=(13.5, 2.9), sharey=False)
    x = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    for j, p in enumerate(params):
        ax = axes[j]
        vals = [inv[n]["best"][p] for n in names]
        crb = [d["fisher"][n]["crb_std"][j] for n in names]
        # Wind direction errors are reported in degrees; put the bound in degrees too.
        if p == "theta_w":
            crb = [np.degrees(c) for c in crb]
        ax.bar(x, vals, color=colors, width=0.66)
        ax.plot(x, crb, "k_", ms=16, mew=1.6, label="Cramér–Rao $\\sigma$")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["M", "M+P", "M+C", "M+P+C"], fontsize=7.5, rotation=0)
        ax.set_title(PRETTY[p], fontsize=11)
        ax.set_ylabel("error (deg)" if p == "theta_w" else "relative error", fontsize=7.5)
        if j == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    fig.suptitle("MLE recovery error by sensor subset, against the Cramér–Rao bound",
                 fontsize=10, y=1.06)
    fig.tight_layout()
    out = FIGS / "fig_recovery.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_scenario(d):
    """Render the synthetic fire itself: masks over time, the camera view, the layout."""
    import torch

    from pyrofield.sim.synthetic import Scenario, forward, pack

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scen = Scenario(device=device)
    obs = forward(pack(d["truth"], device=device), scen)

    masks = obs["mask"].detach().cpu().numpy()
    plume = obs["plume"].detach().cpu().numpy()
    extent = [0, scen.nx * scen.dx, 0, scen.ny * scen.dx]

    fig = plt.figure(figsize=(12.0, 3.0))
    gs = fig.add_gridspec(1, len(masks) + 2, wspace=0.28)

    for i, m in enumerate(masks):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(m, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1)
        t_min = scen.mask_times[i] * scen.dt / 60
        ax.set_title(f"burn mask  t = {t_min:.0f} min", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(gs[0, len(masks)])
    ax.imshow(plume[-1], origin="upper", cmap="bone", vmin=0.3, vmax=0.95)
    ax.set_title("camera view of plume", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(gs[0, len(masks) + 1])
    ax.imshow(masks[-1], origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=2)
    for r in scen.receptors:
        ax.plot(r[0], r[1], "v", color="tab:blue", ms=7)
    ax.plot(*scen.camera.position[:2], "s", color="tab:red", ms=7)
    ax.plot(*scen.ignition_xy, "*", color="tab:orange", ms=11)
    th = d["truth"]["theta_w"]
    ax.arrow(120, 120, 220 * np.cos(th), 220 * np.sin(th), width=14,
             color="tab:green", alpha=0.75)
    ax.set_title("layout: camera ■ sensors ▾\nignition ★  wind →", fontsize=8)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    out = FIGS / "fig_scenario.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    FIGS.mkdir(exist_ok=True)
    d = load()
    fig_sensor_sufficiency(d)
    fig_fisher_spectra(d)
    fig_recovery(d)
    try:
        fig_scenario(d)
    except Exception as exc:  # scenario rendering needs torch + a GPU run
        print(f"  (skipping fig_scenario: {exc})")


if __name__ == "__main__":
    main()
