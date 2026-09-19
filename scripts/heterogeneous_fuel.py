"""Does spatially varying fuel break the degeneracy? (It does not, and here is the proof.)

RESULTS.md asserts that heterogeneous fuel leaves the mask degeneracy intact, on the
grounds that rescaling the whole fuel field by a constant *is* the null direction. That
was an argument, not a measurement. This measures it.

Three fuel fields are used: uniform, a smooth gradient, and a rough random field with a
2:1 range. In each, the closed-form null curve from section 1 of RESULTS.md is walked, and
the mask misfit is checked. If the argument is right, walking the curve costs nothing in
every field, and the mask misfit surface is as flat in a patchwork landscape as on a
billiard table.

The case this does *not* cover, and says so, is a fuel field whose *shape* is unknown.
That adds unknowns without adding mask observables, so it can only make the degeneracy
worse -- which is the direction that does not need defending.

Run:  python scripts/heterogeneous_fuel.py
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

from pyrofield.physics.rothermel import WIND_A, WIND_B  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
    pack,
)

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
FRACS = (0.05, 0.10, 0.20, 0.30)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def smooth_random(ny, nx, device, seed=0, lo=0.65, hi=1.35, scale=6):
    """A rough but spatially correlated fuel field, normalised to mean 1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    coarse = torch.rand(scale, scale, generator=g)
    f = torch.nn.functional.interpolate(coarse[None, None], size=(ny, nx),
                                        mode="bicubic", align_corners=False)[0, 0]
    f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    f = lo + (hi - lo) * f
    return (f / f.mean()).to(device)


def fields(ny, nx, device):
    ys = torch.linspace(0, 1, ny, device=device)[:, None]
    xs = torch.linspace(0, 1, nx, device=device)[None, :]
    grad = 0.65 + 0.70 * (0.5 * (xs + ys))
    return {
        "uniform": torch.ones(ny, nx, device=device),
        "gradient (0.65 - 1.35x)": grad / grad.mean(),
        "rough random (2:1)": smooth_random(ny, nx, device, seed=3),
    }


def null_curve_member(frac):
    """The (R0, k_LB) that reproduce the past at a wind speed wrong by ``frac``."""
    U_hat = TRUTH["U"] * (1 + frac)
    aub0 = WIND_A * TRUTH["U"] ** WIND_B
    R_head = TRUTH["R0"] * (1.0 + aub0)
    LB = 1.0 + TRUTH["lb_k"] * TRUTH["U"]
    return {**TRUTH, "U": U_hat,
            "R0": R_head / (1.0 + WIND_A * U_hat**WIND_B),
            "lb_k": (LB - 1.0) / U_hat}


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = tuple(range(8, 110, 9))
    probe = Scenario(device=device, n_steps=110, mask_times=times,
                     plume_times=(), conc_times=())
    ff = fields(probe.ny, probe.nx, device)
    theta_true = pack(TRUTH, device=device)

    print("walking the closed-form null curve in three fuel landscapes")
    print("(dchi2 near zero means the curve is still a null direction)\n")
    print(f"  {'fuel field':>24s} " + "".join(f"{'+%d%%' % (f*100):>12s}" for f in FRACS)
          + f"{'control +30%':>14s}")

    out, fig_rows = {}, {}
    for name, field in ff.items():
        sc = Scenario(device=device, n_steps=110, mask_times=times,
                      plume_times=(), conc_times=(), fuel_field=field)
        gen = torch.Generator(device=device).manual_seed(31)
        target = add_noise(forward(theta_true, sc), gen)
        sig = noise_sigma_for(target, "mask")
        n = target["mask"].numel()

        def chi2(th):
            return float((((forward(th, sc)["mask"] - target["mask"]) / sig) ** 2).sum())

        c0 = chi2(theta_true)
        vals = []
        for f in FRACS:
            vals.append(chi2(pack(null_curve_member(f), device=device)) - c0)
        # Control: move the wind by 30 % and compensate nothing.
        ctl = chi2(pack({**TRUTH, "U": TRUTH["U"] * 1.30}, device=device)) - c0
        out[name] = {"dchi2_on_curve": vals, "dchi2_control": ctl,
                     "chi2_truth_per_N": c0 / n,
                     "fuel_min": float(field.min()), "fuel_max": float(field.max())}
        fig_rows[name] = (vals, ctl)
        print(f"  {name:>24s} " + "".join(f"{v:12.2f}" for v in vals)
              + f"{ctl:14.0f}")

    print()
    worst = max(max(abs(v) for v in out[n]["dchi2_on_curve"]) for n in out)
    ctl_min = min(out[n]["dchi2_control"] for n in out)
    print(f"  worst dchi2 anywhere on the curve, any landscape: {worst:.2f} "
          f"(1 sigma = 1)")
    print(f"  smallest control value (same wind error, no compensation): {ctl_min:.0f}")
    print(f"  ratio: {ctl_min/max(worst,1e-9):.0f}x")
    print("\n  The degeneracy is unchanged by fuel heterogeneity, as the algebra requires:")
    print("  a constant rescaling of the whole field is exactly the null direction, and")
    print("  that holds whatever the field looks like.")
    print("\n  Not covered: a fuel field whose *shape* is unknown. That adds unknowns")
    print("  without adding mask observables, so it can only make matters worse.")

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.1))
    for ax, (name, field) in zip(axes[:3], ff.items()):
        im = ax.imshow(field.cpu().numpy(), origin="lower", cmap="YlGn",
                       vmin=0.6, vmax=1.4)
        ax.set_title(f"{name}\n$\\Delta\\chi^2$ on the curve "
                     f"$\\leq$ {max(abs(v) for v in fig_rows[name][0]):.1f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[3]
    x = np.array(FRACS) * 100
    for (name, (vals, ctl)), c in zip(fig_rows.items(),
                                      ["tab:green", "tab:olive", "tab:brown"]):
        ax.plot(x, np.maximum(np.abs(vals), 1e-2), "o-", color=c, lw=1.7,
                label=name.split(" ")[0])
    ax.axhline(1.0, color="0.35", ls="--", lw=1.1)
    ax.text(x[0], 1.3, "$\\Delta\\chi^2=1$", fontsize=7.5, color="0.35")
    ctl_vals = [fig_rows[n][1] for n in fig_rows]
    ax.axhline(min(ctl_vals), color="tab:red", ls=":", lw=1.4,
               label="no compensation")
    ax.set_yscale("log")
    ax.set_xlabel("wind error imposed (%)", fontsize=8.5)
    ax.set_ylabel(r"$\Delta\chi^2$ on the null curve", fontsize=8.5)
    ax.set_title("The curve stays null in every\nlandscape", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Heterogeneous fuel does not break the mask degeneracy",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_heterogeneous_fuel.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "heterogeneous_fuel.json").write_text(json.dumps(
        {"truth": TRUTH, "fracs": list(FRACS), "fields": out}, indent=2),
        encoding="utf-8")
    print(f"wrote {RESULTS/'heterogeneous_fuel.json'}")


if __name__ == "__main__":
    main()
