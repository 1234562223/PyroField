"""Which sensors determine which parameters -- measured with the instrument that works.

The project plan's second contribution was a sensor sufficiency diagram: modality subsets
down one axis, state components across the other, each cell marked determined or not. This
repository already has one, in `figures/fig_sensor_sufficiency.png`, and section 11 is the
reason it should not be trusted: it is built from Cramer-Rao bounds, and on the degenerate
direction those are wrong by 33x one way with finite differences and 52x the other with
forward-mode AD. A diagram whose entries are wrong by 1500x between two implementations of
the same quantity is not a diagram.

So this rebuilds it with the profile probe from `pyrofield/eval/guarded.py`, which is
derivative-free and follows the null *curve* rather than its tangent. The question asked of
each cell is the operational one and needs no inversion to answer: standing at the true
parameters, how much does the misfit rise if this parameter is forced off by 30 % and every
other parameter is allowed to re-absorb it?

Cells are decided by that number alone. Where a parameter is determined, a second smaller
probe turns the profile into a sigma, so the diagram carries a precision and not just a
tick.

Run:  python scripts/sufficiency_profile.py
"""

from __future__ import annotations

import argparse
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

from pyrofield.eval.guarded import DCHI2_FLAT, guard  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    pack,
)

SUBSETS = {
    "masks": ("mask",),
    "masks + air quality": ("mask", "conc"),
    "masks + plume": ("mask", "plume"),
    "all three": ("mask", "plume", "conc"),
}
MASK_TIMES = (17, 35, 53, 71, 89, 109)
PLUME_TIMES = (53, 109)
CONC_TIMES = (35, 71, 109)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scenarios(device):
    """Two scenes: flat ground, and the slope section 5 says takes the bearing too."""
    base = dict(device=device, n_steps=110, mask_times=MASK_TIMES,
                plume_times=PLUME_TIMES, conc_times=CONC_TIMES)
    return {
        "flat, 3 m/s": (Scenario(**base),
                        {"R0": 0.10, "U": 3.0, "theta_w": 0.7854, "lb_k": 0.35,
                         "w_buoy": 3.0, "Q": 1.0}),
        "20 deg slope, 3 m/s": (Scenario(**base, slope_phi=0.36, slope_aspect=2.09),
                                {"R0": 0.10, "U": 3.0, "theta_w": 0.7854, "lb_k": 0.35,
                                 "w_buoy": 3.0, "Q": 1.0}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=float, default=0.30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    out = {"probe": args.probe, "dchi2_flat": DCHI2_FLAT, "scenes": {}}
    t0 = time.time()
    for scene, (sc, truth) in scenarios(device).items():
        th = pack(truth, device=device)
        gen = torch.Generator(device=device).manual_seed(11)
        target = add_noise(forward(th, sc), gen)
        print("=" * 78)
        print(f"[{scene}]  truth: " + ", ".join(f"{k}={v}" for k, v in truth.items()))
        print("=" * 78)
        cells = {}
        for label, keys in SUBSETS.items():
            if any(target[k].numel() == 0 for k in keys):
                continue
            print(f"\n  {label}")
            verdicts, nf = guard(th, target, sc, keys,
                                 probes=(0.10, args.probe), verbose=True)
            cells[label] = {v.name: {"determined": v.determined,
                                     "dchi2": v.dchi2,
                                     "sigma_rel": v.sigma_rel,
                                     "sigma_abs": v.sigma_abs} for v in verdicts}
            print(f"    ({nf} profile fits, {time.time()-t0:.0f}s elapsed)")
        out["scenes"][scene] = cells

    (RESULTS / "sufficiency_profile.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")
    print(f"\nwrote {RESULTS/'sufficiency_profile.json'}")
    make_figure(out)
    summarise(out)


def summarise(out):
    print()
    print("=" * 78)
    print("what changed against the Cramer-Rao version (section 11)")
    print("=" * 78)
    for scene, cells in out["scenes"].items():
        print(f"\n  {scene}")
        print(f"    {'sensors':>22} " + "".join(f"{p:>10}" for p in PARAM_NAMES))
        for label, row in cells.items():
            marks = []
            for p in PARAM_NAMES:
                v = row.get(p)
                if v is None:
                    marks.append("     -")
                elif not v["determined"]:
                    marks.append("      no")
                elif v["sigma_rel"] is not None:
                    marks.append(f"{v['sigma_rel']*100:8.2f}%")
                else:
                    marks.append(f"{v['sigma_abs']:7.3f}d")
            print(f"    {label:>22} " + "".join(f"{m:>10}" for m in marks))


def make_figure(out):
    scenes = list(out["scenes"])
    fig, axes = plt.subplots(1, len(scenes), figsize=(6.6 * len(scenes), 3.4),
                             squeeze=False)
    for ax, scene in zip(axes[0], scenes):
        cells = out["scenes"][scene]
        labels = list(cells)
        grid = np.zeros((len(labels), len(PARAM_NAMES)))
        text = [["" for _ in PARAM_NAMES] for _ in labels]
        for i, lab in enumerate(labels):
            for j, p in enumerate(PARAM_NAMES):
                v = cells[lab].get(p)
                if v is None:
                    grid[i, j] = np.nan
                    continue
                big = max(v["dchi2"].values())
                grid[i, j] = np.log10(max(big, 1e-2))
                if not v["determined"]:
                    text[i][j] = "no"
                elif v["sigma_rel"] is not None:
                    text[i][j] = (f"{v['sigma_rel']*100:.2f}%"
                                  if v["sigma_rel"] < 1 else ">100%")
                else:
                    text[i][j] = f"{v['sigma_abs']:.2f}°"
        im = ax.imshow(grid, cmap="RdYlGn", vmin=-2, vmax=5, aspect="auto")
        for i in range(len(labels)):
            for j in range(len(PARAM_NAMES)):
                if text[i][j]:
                    ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6.6)
        ax.set_xticks(range(len(PARAM_NAMES)))
        ax.set_xticklabels(PARAM_NAMES, fontsize=7.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(scene, fontsize=9.5)
        plt.colorbar(im, ax=ax, fraction=0.03,
                     label=r"$\log_{10}\,\Delta\chi^2$ for a 30 % error")
    fig.suptitle("Sensor sufficiency, measured by profile probe rather than by a "
                 "Cramér–Rao bound", fontsize=10, y=1.06)
    fig.tight_layout()
    p = FIGS / "fig_sufficiency_profile.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
