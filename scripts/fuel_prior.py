"""How good does your fuel map have to be for masks to be enough?

The obvious objection to everything measured so far is that fuel is not really unknown:
LANDFIRE and its equivalents give a fuel model for every pixel. If fuel were known
exactly, the degeneracy would not bite -- masks would give the wind.

That objection is quantitative, so this is the quantitative answer. Because the mask
information *along* the null curve is exactly zero (scripts/null_direction.py), the
posterior along it is exactly the prior projected onto it, with no data contribution at
all. A relative prior uncertainty sigma on the fuel spread rate therefore maps to a wind
uncertainty

    sigma_lnU = sigma_lnR0 / E,      E = b·a·U^b / (1 + a·U^b)

with no approximation. E is the elasticity of the wind factor, and at 4 m/s it is 0.88 --
so the fuel prior transfers to wind essentially one-for-one.

The script then pushes that uncertainty through the wind-shift forecast of
scripts/wind_shift_forecast.py and reports the answer in minutes, which is the form a
fire agency can act on: how wrong your arrival-time forecast is, as a function of how
good your fuel map is, and what fuel-map accuracy would be needed to match a camera.

Run:  python scripts/fuel_prior.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyrofield.physics.rothermel import WIND_A, WIND_B  # noqa: E402
from pyrofield.sim.synthetic import Scenario, forward, pack  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
T_OBS, T_END = 80, 160
U_NEW, TW_NEW = 7.0, float(np.pi / 2)
FORECAST_CADENCE = 4
ROAD_Y, ROAD_X = 1080.0, (350.0, 1150.0)

# Relative 1-sigma uncertainty on the fuel spread rate. The upper end is where an
# operational fuel model sits: fuel-model-derived R0 is routinely wrong by tens of
# percent, and the fuel *moisture* that multiplies it more still.
PRIOR_SIGMAS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00)

# Measured elsewhere in this study (scripts/profile_postprocess.py).
PLUME_SIGMA_U = 0.00095

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def elasticity(U: float) -> float:
    aub = WIND_A * U**WIND_B
    return WIND_B * aub / (1.0 + aub)


def U_for_R0(R0_target: float, R_head: float) -> float:
    """Invert R0 = R_head / (1 + a U^b) for U."""
    def f(U):
        return R_head / (1.0 + WIND_A * U**WIND_B) - R0_target
    return brentq(f, 0.05, 40.0)


def road_crossing_minute(masks, steps, dx):
    j0 = int(ROAD_Y / dx)
    i0, i1 = int(ROAD_X[0] / dx), int(ROAD_X[1] / dx)
    for m, s in zip(masks, steps):
        if float(m[j0, i0:i1].max()) > 0.5:
            return (s + 1) * 15.0 / 60.0
    return None


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = tuple(range(T_OBS, T_END, FORECAST_CADENCE))
    fsc = Scenario(device=device, n_steps=T_END, mask_times=steps,
                   plume_times=(), conc_times=(),
                   wind_after=(T_OBS, U_NEW, TW_NEW))

    aub0 = WIND_A * TRUTH["U"] ** WIND_B
    R_head = TRUTH["R0"] * (1.0 + aub0)
    LB = 1.0 + TRUTH["lb_k"] * TRUTH["U"]
    E = elasticity(TRUTH["U"])
    print(f"truth: R_head = {R_head:.5f} m/s, LB = {LB:.4f}, "
          f"wind-factor elasticity E = {E:.4f}")
    print(f"so a fuel prior of sigma maps to a wind uncertainty of sigma/{E:.3f} "
          f"= {1/E:.2f} x sigma\n")

    t_true = road_crossing_minute(forward(pack(TRUTH, device=device), fsc)["mask"],
                                  steps, fsc.dx)
    print(f"truth: fire reaches the road at {t_true:.1f} min\n")

    def forecast_time(U_hat):
        m = {**TRUTH, "U": float(U_hat),
             "R0": R_head / (1.0 + WIND_A * U_hat**WIND_B),
             "lb_k": (LB - 1.0) / U_hat}
        fm = forward(pack(m, device=device), fsc)["mask"]
        return road_crossing_minute(fm, steps, fsc.dx)

    print(f"  {'fuel prior':>11s} {'-> wind sigma':>14s} {'U range (m/s)':>18s} "
          f"{'road-time window':>20s} {'width':>8s}")
    rows = []
    for s in PRIOR_SIGMAS:
        # The prior on ln R0 maps exactly onto the null curve; +/-1 sigma in R0 gives the
        # 1-sigma interval in U, because the masks contribute nothing along this curve.
        lo = U_for_R0(TRUTH["R0"] * np.exp(+s), R_head)   # more fuel -> less wind
        hi = U_for_R0(TRUTH["R0"] * np.exp(-s), R_head)
        t_lo, t_hi = forecast_time(lo), forecast_time(hi)
        sig_U = s / E
        width = (abs(t_hi - t_true) + abs(t_true - t_lo)) if (t_lo and t_hi) else None
        rows.append({"prior_sigma": s, "sigma_U": sig_U, "U_lo": lo, "U_hi": hi,
                     "t_lo": t_lo, "t_hi": t_hi, "window_min": width})
        print(f"  {s*100:10.0f}% {sig_U*100:13.0f}% "
              f"{lo:8.2f} - {hi:<7.2f} "
              + (f"{t_lo:8.1f} - {t_hi:<8.1f}" if (t_lo and t_hi) else f"{'n/a':>18s}")
              + (f" {width:7.1f} min" if width else ""))

    # What fuel accuracy would be needed to match the camera?
    needed = PLUME_SIGMA_U * E
    print(f"\n  the plume measures wind speed to sigma = {PLUME_SIGMA_U*100:.3f} %")
    print(f"  matching that from a fuel map alone would require the fuel spread rate")
    print(f"  known to {needed*100:.3f} % -- about "
          f"{0.35/needed:.0f}x better than an operational fuel model (~35 %).")

    ok_rows = [r for r in rows if r["window_min"] is not None]
    if ok_rows:
        r35 = min(ok_rows, key=lambda r: abs(r["prior_sigma"] - 0.35))
        print(f"\n  at a realistic 35 % fuel prior the arrival-time forecast spans "
              f"{r35['window_min']:.0f} minutes;")
        print(f"  the plume collapses that to under a minute.")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))

    ax = axes[0]
    ax.loglog([r["prior_sigma"] * 100 for r in rows],
              [r["sigma_U"] * 100 for r in rows], "o-", color="tab:red", lw=1.9,
              label="masks + a fuel prior")
    ax.axhline(PLUME_SIGMA_U * 100, color="tab:blue", ls="--", lw=1.6,
               label="masks + plume (measured)")
    ax.axvspan(20, 100, color="0.85", zorder=0)
    ax.text(30, PLUME_SIGMA_U * 100 * 3, "where operational\nfuel models sit",
            fontsize=7, color="0.35")
    ax.set_xlabel("1$\\sigma$ uncertainty on the fuel spread rate (%)", fontsize=8.5)
    ax.set_ylabel("resulting 1$\\sigma$ on wind speed (%)", fontsize=8.5)
    ax.set_title("A fuel prior transfers to wind one-for-one", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    xs = [r["prior_sigma"] * 100 for r in ok_rows]
    ax.plot(xs, [r["window_min"] for r in ok_rows], "o-", color="tab:red", lw=1.9,
            label="masks + a fuel prior")
    ax.axhline(1.0, color="tab:blue", ls="--", lw=1.6, label="masks + plume (< 1 min)")
    ax.axvspan(20, 100, color="0.85", zorder=0)
    ax.set_xlabel("1$\\sigma$ uncertainty on the fuel spread rate (%)", fontsize=8.5)
    ax.set_ylabel("width of the predicted\narrival-time window (min)", fontsize=8.5)
    ax.set_title("and straight through to the forecast", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("\"But we know the fuel\" — how well would you have to?",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_fuel_prior.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "fuel_prior.json").write_text(json.dumps({
        "truth": TRUTH, "R_head": R_head, "LB": LB, "elasticity": E,
        "truth_road_min": t_true, "plume_sigma_U": PLUME_SIGMA_U,
        "fuel_accuracy_to_match_plume": needed, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'fuel_prior.json'}")


if __name__ == "__main__":
    main()
