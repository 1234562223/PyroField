"""The wind-shift result, over an ensemble rather than one fire.

`scripts/wind_shift_forecast.py` shows a 10-minute spread in predicted arrival time on
one fire. One fire is an anecdote. This repeats the measurement over a sample of the
operational envelope -- fuel, wind before and after the shift, veer angle, and fuel shape
all drawn at random -- and reports the distribution.

It is cheap because the family of indistinguishable fits is available in closed form
(section 1 of RESULTS.md): no optimiser is needed, only forward runs along the null curve.

To keep scenarios comparable, the road is placed per-fire at the distance the true fire
reaches halfway through the forecast window, so the truth always arrives mid-window and
the spread around it is what varies.

Run:  python scripts/forecast_ensemble.py --n 24
"""

from __future__ import annotations

import argparse
import json
import math
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

from pyrofield.models.operators.mask import iou  # noqa: E402
from pyrofield.physics.rothermel import WIND_A, WIND_B  # noqa: E402
from pyrofield.sim.synthetic import Scenario, forward, pack  # noqa: E402

T_OBS, T_END = 80, 160
CADENCE = 4
# A larger domain than the default 96 x 15 m. The first version of this sweep rejected
# 53 % of draws because fast fires left the grid, and that rejection is not random --
# it removes exactly the fires that spread furthest. Widening the domain keeps them.
NX = NY = 160
IGNITION = (700.0, 700.0)

# Operational envelope to draw from.
RANGES = {
    "R0": (0.06, 0.16),
    "U0": (2.0, 6.0),
    "theta0": (0.0, 2 * math.pi),
    "lb_k": (0.25, 0.50),
    "U1_ratio": (1.2, 2.2),      # the shift strengthens the wind
    "veer_deg": (20.0, 70.0),    # and turns it
}
# Fractional wind-speed offsets that a mask-only fit cannot tell apart, relative to truth.
VALLEY_REL = np.array([0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def valley_member(truth, U_hat):
    """The (R0, k_LB) reproducing the observed past at a wrong wind speed."""
    aub0 = WIND_A * truth["U"] ** WIND_B
    R_head = truth["R0"] * (1.0 + aub0)
    LB = 1.0 + truth["lb_k"] * truth["U"]
    return {**truth, "U": float(U_hat),
            "R0": R_head / (1.0 + WIND_A * U_hat**WIND_B),
            "lb_k": (LB - 1.0) / U_hat}


def front_extent(mask, dx, origin, direction):
    """Distance from origin to the far edge of the burn along a unit direction."""
    ox, oy = origin
    ux, uy = direction
    ts = torch.arange(0.0, 1600.0, dx * 0.25, device=mask.device)
    xs, ys = ox + ts * ux, oy + ts * uy
    ny, nx = mask.shape
    gx = 2.0 * xs / (nx * dx) - 1.0
    gy = 2.0 * ys / (ny * dx) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
    v = torch.nn.functional.grid_sample(mask[None, None], grid, mode="bilinear",
                                        padding_mode="zeros",
                                        align_corners=False).reshape(-1)
    inside = v > 0.5
    if not bool(inside.any()):
        return 0.0
    return float(ts[int(torch.nonzero(inside).max())])


def crossing_minute(masks, steps, dx, origin, direction, dist, dt):
    """First sampled time at which the burn crosses a line perpendicular to ``direction``.

    The road has to be perpendicular to the post-shift wind, not axis-aligned. Placing a
    constant-y line at a distance measured along a slanted wind put the road far too close
    in most scenarios, and the fire crossed it at the first forecast sample -- which
    showed up as a spurious zero-width window.
    """
    ny, nx = masks.shape[1], masks.shape[2]
    ys = (torch.arange(ny, device=masks.device, dtype=masks.dtype) + 0.5) * dx
    xs = (torch.arange(nx, device=masks.device, dtype=masks.dtype) + 0.5) * dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    proj = (gx - origin[0]) * direction[0] + (gy - origin[1]) * direction[1]
    beyond = proj >= dist
    if not bool(beyond.any()):
        return None
    for m, s in zip(masks, steps):
        if bool(((m > 0.5) & beyond).any()):
            return (s + 1) * dt / 60.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    steps = tuple(range(T_OBS, T_END, CADENCE))

    print(f"device={device}  scenarios={args.n}")
    print(f"  {'#':>3s} {'U0':>5s} {'U1':>5s} {'veer':>6s} {'R0':>6s} "
          f"{'truth(min)':>11s} {'window(min)':>12s} {'worst err':>10s} {'IoU range':>16s}")

    out, t0 = [], time.time()
    kept = 0
    for i in range(args.n):
        s = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
        truth = {"R0": s["R0"], "U": s["U0"], "theta_w": s["theta0"],
                 "lb_k": s["lb_k"], "w_buoy": 3.0, "Q": 1.0}
        U1 = s["U0"] * s["U1_ratio"]
        tw1 = s["theta0"] + math.radians(s["veer_deg"])

        fsc = Scenario(device=device, nx=NX, ny=NY, n_steps=T_END, mask_times=steps,
                       plume_times=(), conc_times=(),
                       ignition_xy=IGNITION, wind_after=(T_OBS, U1, float(tw1)))
        th_true = pack(truth, device=device)
        tm = forward(th_true, fsc)["mask"]

        # Place the road where the true fire is halfway through the forecast window,
        # measured along the post-shift wind, so every scenario is scored comparably.
        mid = len(steps) // 2
        wdir = (math.cos(tw1), math.sin(tw1))
        d_mid = front_extent(tm[mid], fsc.dx, IGNITION, wdir)
        if d_mid < 60.0:
            continue                      # fire barely moved; nothing to score
        # The road must stay on the grid in both coordinates.
        rx = IGNITION[0] + d_mid * wdir[0]
        ry = IGNITION[1] + d_mid * wdir[1]
        lim = fsc.nx * fsc.dx
        if not (80.0 < rx < lim - 80.0 and 80.0 < ry < lim - 80.0):
            continue
        t_true = crossing_minute(tm, steps, fsc.dx, IGNITION, wdir, d_mid, fsc.dt)
        if t_true is None:
            continue

        times, ious = [], []
        for rel in VALLEY_REL:
            m = valley_member(truth, truth["U"] * float(rel))
            fm = forward(pack(m, device=device), fsc)["mask"]
            tt = crossing_minute(fm, steps, fsc.dx, IGNITION, wdir, d_mid, fsc.dt)
            if tt is not None:
                times.append(tt)
            ious.append(float(iou(fm[-1], tm[-1])))
        if len(times) < 4:
            continue

        kept += 1
        width = max(times) - min(times)
        worst = max(abs(t - t_true) for t in times)
        out.append({"U0": s["U0"], "U1": U1, "veer_deg": s["veer_deg"],
                    "R0": s["R0"], "lb_k": s["lb_k"], "truth_min": t_true,
                    "window_min": width, "worst_err_min": worst,
                    "times": times, "iou": ious})
        print(f"  {kept:3d} {s['U0']:5.2f} {U1:5.2f} {s['veer_deg']:6.1f} "
              f"{s['R0']:6.3f} {t_true:11.1f} {width:12.1f} {worst:10.1f} "
              f"{min(ious):7.3f}-{max(ious):.3f}")

    print(f"\n  kept {kept} of {args.n} scenarios ({time.time()-t0:.0f}s)")
    if not out:
        raise SystemExit("no usable scenarios")

    w = np.array([r["window_min"] for r in out])
    e = np.array([r["worst_err_min"] for r in out])
    lo_iou = np.array([min(r["iou"]) for r in out])

    print("\n" + "=" * 78)
    print("ENSEMBLE RESULT")
    print("=" * 78)
    print(f"  width of the arrival-time window that masks cannot narrow:")
    print(f"     median {np.median(w):.1f} min,  IQR {np.percentile(w,25):.1f}"
          f"-{np.percentile(w,75):.1f},  max {w.max():.1f}")
    print(f"  worst arrival-time error across the indistinguishable family:")
    print(f"     median {np.median(e):.1f} min,  max {e.max():.1f}")
    print(f"  worst forecast IoU at +20 min within the family:")
    print(f"     median {np.median(lo_iou):.3f},  min {lo_iou.min():.3f}")
    frac5 = float((w >= 5).mean())
    print(f"  fraction of fires where the window is at least 5 minutes wide: "
          f"{frac5*100:.0f}%")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2))
    ax = axes[0]
    ax.hist(w, bins=10, color="tab:red", alpha=0.8)
    ax.axvline(np.median(w), color="k", ls="--", lw=1.2,
               label=f"median {np.median(w):.1f} min")
    ax.set_xlabel("width of the arrival-time window (min)", fontsize=8.5)
    ax.set_ylabel("fires", fontsize=8.5)
    ax.set_title("Masks cannot narrow this", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter([r["U1"] / r["U0"] for r in out], w, c=[r["veer_deg"] for r in out],
               cmap="viridis", s=34)
    ax.set_xlabel("wind strengthening factor", fontsize=8.5)
    ax.set_ylabel("window width (min)", fontsize=8.5)
    ax.set_title("Bigger shift, worse spread\n(colour = veer angle)", fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.scatter([r["U0"] for r in out], lo_iou, color="tab:red", s=34)
    ax.set_xlabel("wind before the shift (m/s)", fontsize=8.5)
    ax.set_ylabel("worst forecast IoU in the family", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_title("Worst member of the\nindistinguishable family", fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle(f"Forecast spread induced by the mask degeneracy, over {kept} fires",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_forecast_ensemble.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "forecast_ensemble.json").write_text(json.dumps({
        "n_requested": args.n, "n_kept": kept, "ranges": {k: list(v) for k, v in RANGES.items()},
        "valley_rel": VALLEY_REL.tolist(), "scenarios": out,
        "summary": {"window_median": float(np.median(w)), "window_max": float(w.max()),
                    "worst_err_median": float(np.median(e)),
                    "worst_err_max": float(e.max()),
                    "min_iou_median": float(np.median(lo_iou))},
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'forecast_ensemble.json'}")


if __name__ == "__main__":
    main()
