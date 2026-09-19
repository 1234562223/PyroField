"""The one figure the repository's front page carries.

Three panels, one per pillar of the argument: the degeneracy is exact, it holds on real
fires, and it has a cost. Everything is read from `results/*.json` and
`results/fusion_preds.npz`, so it cannot drift from the numbers in RESULTS.md.

Run:  python scripts/make_hero.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

K = 1.1486328125
LATTICE = np.arange(8) * 45.0

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 190, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def p_lattice(p_h, V, c_1, c_2):
    """PyTorchFire's propagation probability for the eight lattice directions."""
    d = np.deg2rad(LATTICE)
    return np.tanh(K * p_h * np.exp(c_1 * V) * np.exp(c_2 * V * (np.cos(d) - 1.0)))


def main():
    ca = json.loads((RESULTS / "independent_ca.json").read_text(encoding="utf-8"))
    rf = json.loads((RESULTS / "real_fire_inversion.json").read_text(encoding="utf-8"))
    preds = np.load(RESULTS / "fusion_preds.npz")

    fig = plt.figure(figsize=(13.2, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.32)

    # --- 1. exact, and not this code's -------------------------------------------
    b = ca["baseline"]
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    th = np.deg2rad(np.append(LATTICE, 360))
    V = b["V"] * 1.8
    p_h = b["p_h"] * np.exp(b["c_1"] * (b["V"] - V))
    c_2 = b["c_2"] * b["V"] / V
    base = p_lattice(b["p_h"], b["V"], b["c_1"], b["c_2"])
    comp = p_lattice(p_h, V, b["c_1"], c_2)
    unco = p_lattice(b["p_h"], V, b["c_1"], b["c_2"])
    ax.plot(th, np.append(base, base[0]), lw=5.5, color="tab:blue", alpha=0.35,
            label=f"{b['V']:.0f} m/s")
    ax.plot(th, np.append(comp, comp[0]), lw=1.5, color="k", ls="--",
            label=f"{V:.0f} m/s, fuel compensated")
    ax.plot(th, np.append(unco, unco[0]), lw=1.6, color="tab:red",
            label=f"{V:.0f} m/s, not compensated")
    ax.set_rlabel_position(118)
    ax.tick_params(labelsize=6)
    ax.set_title("An 80 % wind error, hidden\nexactly", fontsize=10, pad=16)
    ax.legend(fontsize=6.2, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.34))
    ax.text(np.deg2rad(210), ax.get_rmax() * 1.62,
            "0 of 307 200 burn cells differ\nin a third-party simulator",
            fontsize=7, ha="center", color="0.25")

    # --- 2. on real fires ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    rows = rf["rows"]
    ug = np.array([r["U_gridmet"] for r in rows])
    order = np.argsort(ug)
    cols = {1.5: "tab:blue", 3.0: "tab:orange", 6.0: "tab:red"}
    for j, i in enumerate(order):
        fits = rows[i]["fits"]
        ax.plot([j, j], [min(f["U_fit"] for f in fits), max(f["U_fit"] for f in fits)],
                color="0.78", lw=1.0, zorder=1)
        for f in fits:
            ax.plot(j, f["U_fit"], "o", ms=3.6, zorder=2,
                    color=cols.get(f["u_start"], "k"),
                    label=f"started at {f['u_start']} m/s" if j == 0 else None)
    ax.plot(range(len(order)), ug[order], "k_", ms=10, mew=2, zorder=3,
            label="the real wind")
    ax.set_yscale("log")
    ax.set_ylim(0.12, 40)
    ax.set_xlabel(f"{rf['n']} real fires, ordered by wind", fontsize=8.5)
    ax.set_ylabel("wind speed (m/s)", fontsize=8.5)
    ax.set_title("Fitted from three starting winds,\nthe answers stay where they began",
                 fontsize=10)
    ax.legend(fontsize=5.9, frameon=False, ncol=2, loc="upper center")
    ax.grid(alpha=0.22, which="both")

    # --- 3. the cost ----------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    IU = 1
    for tag, c, lab in (("CNN mask only", "tab:red", "network, masks only"),
                        ("CNN mask + plume", "tab:blue", "network, masks + plume")):
        for split, mk in (("test_id", "o"), ("test_ood", "^")):
            p, t = preds[f"{tag}|{split}__pred"], preds[f"{tag}|{split}__true"]
            ax.scatter(np.exp(t[:, IU]), np.exp(p[:, IU]), s=5, alpha=0.28,
                       marker=mk, color=c, zorder=2,
                       label=lab if split == "test_id" else None)
    k = "state mask + plume"
    for split, mk in (("test_id", "o"), ("test_ood", "^")):
        p, t = preds[f"{k}|{split}__pred"], preds[f"{k}|{split}__true"]
        ax.scatter(np.exp(t[:, IU]), np.exp(p[:, IU]), s=30, marker=mk,
                   facecolor="none", edgecolor="k", linewidth=1.1, zorder=4,
                   label="inversion, masks + plume" if split == "test_id" else None)
    lims = [1.5, 8.5]
    ax.axvspan(5.0, 8.5, color="0.92", zorder=0)
    ax.plot(lims, lims, "k--", lw=1.0, zorder=3)
    ax.text(5.2, 1.95, "outside\ntraining", fontsize=6.6, color="0.35")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("true wind (m/s)", fontsize=8.5)
    ax.set_ylabel("estimated wind (m/s)", fontsize=8.5)
    ax.set_title("A network asked for something\nthe data does not contain", fontsize=10)
    ax.legend(fontsize=5.9, frameon=False, loc="upper left")
    ax.grid(alpha=0.22)

    p = FIGS / "fig_hero.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
