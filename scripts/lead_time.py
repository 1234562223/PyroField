"""Lead time: when does each sensor subset first pin down the fire's physics?

The cross-envelope sweep showed that mask-only observation determines fuel and
wind *once the burn is large and fast* -- which is to say, once it is too late to
act. This experiment measures that directly in the currency public safety cares
about: minutes.

For a growing observation window T, the Fisher information from each sensor subset
is recomputed, and we record the first T at which wind speed clears its operational
tolerance. The gap between subsets is the lead time that reading the plume buys.

Run:  python scripts/lead_time.py
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

from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, forward, pack  # noqa: E402
from pyrofield.theory.identifiability import (  # noqa: E402
    crb_from_fisher,
    fisher_all_subsets,
)

BASE = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
WINDS = (2.0, 4.0, 6.0)
FIRE_BLOCK = ["R0", "U", "theta_w", "lb_k"]
TOL = {"R0": 0.10, "U": 0.10, "theta_w": np.radians(5.0), "lb_k": 0.15}

CADENCE = 9            # observation epoch every 9 steps = 2.25 min
T_STEPS = tuple(range(9, 111, 9))
LABEL = {("mask",): "M", ("mask", "plume"): "M+P",
         ("mask", "conc"): "M+C", ("mask", "plume", "conc"): "M+P+C"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scenario_upto(T: int, device: str) -> Scenario:
    """A scenario truncated at step ``T`` with observations on a fixed cadence."""
    times = tuple(t for t in range(CADENCE - 1, T, CADENCE))
    if not times:
        times = (T - 1,)
    return Scenario(
        device=device, n_steps=T,
        mask_times=times, plume_times=times, conc_times=times,
    )


def first_crossing(ts_min, values, tol):
    """First time the curve drops below ``tol``, linearly interpolated; None if never."""
    v = np.asarray(values)
    below = v < tol
    if not below.any():
        return None
    k = int(np.argmax(below))
    if k == 0:
        return float(ts_min[0])
    x0, x1 = ts_min[k - 1], ts_min[k]
    y0, y1 = np.log(v[k - 1]), np.log(v[k])
    lt = np.log(tol)
    return float(x0 + (x1 - x0) * (y0 - lt) / (y0 - y1 + 1e-30))


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subsets = list(LABEL.keys())

    out = {}
    t_all = time.time()
    for U in WINDS:
        truth = {**BASE, "U": U}
        # Two analyses. The joint one over all six parameters is the honest answer to
        # "what does this sensor set actually determine", because nothing is assumed
        # known. The fire-block one conditions on the smoke parameters (Q, w_buoy)
        # being known, which flatters the air-quality sensor enormously: given Q, one
        # concentration reading hands you U through the Q/U term. Both are reported;
        # the joint one drives the figure and the lead-time table.
        curves = {LABEL[s]: {p: [] for p in PARAM_NAMES} for s in subsets}
        curves_fire = {LABEL[s]: {p: [] for p in FIRE_BLOCK} for s in subsets}
        burn, ts_min = [], []
        print(f"\nU = {U} m/s")
        for T in T_STEPS:
            scen = scenario_upto(T, device)
            theta = pack(truth, device=device)
            t0 = time.time()
            Fs = fisher_all_subsets(theta, scen)
            Ff = fisher_all_subsets(theta, scen, params=FIRE_BLOCK)
            for s in subsets:
                crb, _, _, _ = crb_from_fisher(Fs[s])
                for i, p in enumerate(PARAM_NAMES):
                    curves[LABEL[s]][p].append(float(crb[i]))
                crbf, _, _, _ = crb_from_fisher(Ff[s])
                for i, p in enumerate(FIRE_BLOCK):
                    curves_fire[LABEL[s]][p].append(float(crbf[i]))
            b = float((forward(theta, scen)["mask"][-1] > 0.5).float().mean()) * 100
            burn.append(b)
            ts_min.append(T * scen.dt / 60)
            print(f"  T={T*scen.dt/60:5.2f} min  burn {b:5.2f}%   "
                  + "  ".join(f"{LABEL[s]} CRB(U)={curves[LABEL[s]]['U'][-1]:9.4g}"
                              for s in subsets)
                  + f"   ({time.time()-t0:.1f}s)")
        out[str(U)] = {"t_min": ts_min, "burn_pct": burn,
                       "curves": curves, "curves_fire_block": curves_fire}

    print(f"\ntotal {time.time()-t_all:.0f}s")

    # ---- time-to-tolerance table ----
    print("\n" + "=" * 78)
    print("TIME UNTIL WIND SPEED IS DETERMINED TO WITHIN 10 %")
    print("=" * 78)
    print(f"  {'wind':>6s}  " + "".join(f"{LABEL[s]:>12s}" for s in subsets)
          + f"{'lead gained':>14s}")
    lead = {}
    for U in WINDS:
        d = out[str(U)]
        row = f"  {U:4.1f}  "
        cross = {}
        for s in subsets:
            c = first_crossing(d["t_min"], d["curves"][LABEL[s]]["U"], TOL["U"])
            cross[LABEL[s]] = c
            row += f"{('%.1f min' % c) if c else '   never':>12s}"
        cm, cmp_ = cross["M"], cross["M+P"]
        if cm is None and cmp_ is not None:
            g = f">{d['t_min'][-1] - cmp_:.1f} min"
            gain = float(d["t_min"][-1] - cmp_)
        elif cm is not None and cmp_ is not None:
            g = f"{cm - cmp_:.1f} min"
            gain = float(cm - cmp_)
        else:
            g, gain = "n/a", None
        lead[str(U)] = {"crossings": cross, "gain_min": gain}
        print(row + f"{g:>14s}")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(WINDS), figsize=(12.4, 3.5), sharey=True)
    style = {
        "M":     dict(color="tab:red",    ls="-",  marker="o", lw=2.0, ms=3.5, zorder=5),
        "M+C":   dict(color="tab:orange", ls="-",  marker="s", lw=1.6, ms=3.0, zorder=4),
        "M+P":   dict(color="tab:blue",   ls="--", marker="^", lw=2.2, ms=4.0, zorder=3),
        "M+P+C": dict(color="tab:green",  ls=":",  marker="v", lw=1.6, ms=3.0, zorder=2),
    }
    for ax, U in zip(axes, WINDS):
        d = out[str(U)]
        for s in subsets:
            lab = LABEL[s]
            ax.semilogy(d["t_min"], d["curves"][lab]["U"], label=lab, **style[lab])
        ax.axhline(TOL["U"], color="0.35", ls="--", lw=1.0)
        ax.text(d["t_min"][0], TOL["U"] * 1.3, "10 % tolerance", fontsize=7, color="0.35")
        c_m, c_mp = lead[str(U)]["crossings"]["M"], lead[str(U)]["crossings"]["M+P"]
        y = TOL["U"] * 0.22
        if c_m and c_mp:
            ax.annotate("", xy=(c_mp, y), xytext=(c_m, y),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
            ax.text((c_m + c_mp) / 2, y * 1.25, f"{c_m - c_mp:.0f} min gained",
                    ha="center", fontsize=8)
        elif c_mp:
            ax.annotate("", xy=(c_mp, y), xytext=(d["t_min"][-1], y),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
            ax.text((c_mp + d["t_min"][-1]) / 2, y * 1.25,
                    "masks: never", ha="center", fontsize=8)
        ax.set_title(f"wind {U:.0f} m/s", fontsize=9.5)
        ax.set_xlabel("observation window (min)", fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cramér–Rao $\\sigma$ on wind speed\n(fractional, all 6 parameters free)",
                       fontsize=8.5)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    fig.suptitle("When does each sensor set pin down the wind? "
                 "Masks get there only once the fire is already large.",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    p = FIGS / "fig_lead_time.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "lead_time.json").write_text(json.dumps({
        "base": BASE, "winds": list(WINDS), "cadence_steps": CADENCE,
        "tolerance": {k: float(v) for k, v in TOL.items()},
        "fire_block": FIRE_BLOCK, "data": out, "lead": lead,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'lead_time.json'}")


if __name__ == "__main__":
    main()
