"""Where do you have to put the camera?

Every plume result so far comes from one camera position. That is a deployment
question disguised as a nuisance parameter: a tower sited along the wind axis sees the
plume end-on and foreshortened, while one sited across the wind sees its tilt directly.
If the advantage only exists at a lucky azimuth, it is not an advantage.

This sweeps the camera around the fire and measures the bound on wind speed from mask +
plume at each position, plus a sweep of standoff distance and mast height.

Bounds are Fisher-based, which is legitimate here in a way it is not for masks alone:
information from the plume flows through smooth operators (volume rendering), where AD
and finite differences agree. The script checks that agreement rather than assuming it.

Run:  python scripts/camera_geometry.py
"""

from __future__ import annotations

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

from pyrofield.models.operators.plume_render import Camera  # noqa: E402
from pyrofield.physics.rothermel import head_ros  # noqa: E402
from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, pack  # noqa: E402
from pyrofield.theory.identifiability import crb_from_fisher, fisher_all_subsets  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
AZIMUTHS = np.arange(0, 360, 22.5)
DISTANCES = (400.0, 700.0, 1000.0, 1500.0, 2200.0)
HEIGHTS = (10.0, 30.0, 60.0, 120.0, 250.0)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fire_centre(scen: Scenario) -> tuple[float, float]:
    """Roughly where the burn sits at mid-run, so the camera has something to aim at."""
    T = torch.tensor
    rh = float(head_ros(T(TRUTH["R0"]), T(TRUTH["U"])))
    travel = 0.40 * rh * scen.n_steps * scen.dt
    return (TRUTH["ignition_xy"][0] + travel * math.cos(TRUTH["theta_w"]),
            TRUTH["ignition_xy"][1] + travel * math.sin(TRUTH["theta_w"])) \
        if "ignition_xy" in TRUTH else \
        (scen.ignition_xy[0] + travel * math.cos(TRUTH["theta_w"]),
         scen.ignition_xy[1] + travel * math.sin(TRUTH["theta_w"]))


def make_camera(centre, azimuth_deg, distance, height, aim_height=260.0):
    a = math.radians(azimuth_deg)
    pos = (centre[0] + distance * math.cos(a),
           centre[1] + distance * math.sin(a), height)
    return Camera(position=pos, look_at=(centre[0], centre[1], aim_height),
                  fov_deg=48.0, height=56, width=56,
                  near=max(30.0, 0.15 * distance), far=2.6 * distance, n_samples=96)


# Reference for "how much better than masks". Fisher bounds on the mask-only case are
# not usable (RESULTS.md section 5: AD overshoots by 52x and is not a valid bound, FD is
# valid but loose by 33x), so the comparison is made against the profile-likelihood
# result instead -- which is a *lower* bound on the mask uncertainty, making every gain
# quoted here conservative.
MASK_SIGMA_U = 0.7333  # profile likelihood, scan-limited: masks do no better than this


def crb_U(scen, theta, method="ad", eps=0.02):
    Fs = fisher_all_subsets(theta, scen, eps=eps, method=method)
    iu = PARAM_NAMES.index("U")
    return {"M+P": float(crb_from_fisher(Fs[("mask", "plume")])[0][iu]),
            "M_fisher": float(crb_from_fisher(Fs[("mask",)])[0][iu]),
            "M": MASK_SIGMA_U}


def report(d):
    """Print the tables and draw the figure from collected results."""
    az_rows, dist_rows, h_rows = d["azimuth"], d["distance"], d["height"]
    for rows in (az_rows, dist_rows, h_rows):
        for r in rows:
            r["M"] = MASK_SIGMA_U          # reference fixed to the profile bound
    wind_deg = d["wind_deg"]

    print(f"\n[1] camera azimuth, at 900 m standoff and 60 m mast")
    print(f"    ('vs masks' is against the profile-likelihood mask bound of "
          f"{MASK_SIGMA_U*100:.0f} %, so it is a lower bound on the gain)")
    print(f"    {'azimuth':>8s} {'rel. to wind':>13s} {'CRB(U) M+P':>13s} {'vs masks':>11s}")
    for r in az_rows:
        print(f"    {r['azimuth']:8.1f} {r['rel_to_wind']:13.1f} {r['M+P']*100:12.5f}% "
              f"{MASK_SIGMA_U/max(r['M+P'],1e-30):10.0f}x")
    best = min(az_rows, key=lambda r: r["M+P"])
    worst = max(az_rows, key=lambda r: r["M+P"])
    spread = worst["M+P"] / max(best["M+P"], 1e-30)
    print(f"\n    best  azimuth {best['azimuth']:.0f} deg "
          f"({best['rel_to_wind']:+.0f} from wind): {best['M+P']*100:.5f} %")
    print(f"    worst azimuth {worst['azimuth']:.0f} deg "
          f"({worst['rel_to_wind']:+.0f} from wind): {worst['M+P']*100:.5f} %")
    print(f"    spread across azimuth: {spread:.2f}x")
    print(f"    worst-case gain over masks across all azimuths: "
          f"{MASK_SIGMA_U/max(worst['M+P'],1e-30):.0f}x")

    for title, rows, key, unit in (
        ("[2] standoff distance (across the wind, 60 m mast)", dist_rows, "distance", "m"),
        ("[3] mast height (across the wind, 900 m standoff)", h_rows, "height", "m"),
    ):
        print(f"\n{title}")
        for r in rows:
            print(f"    {r[key]:7.0f} {unit}   CRB(U) = {r['M+P']*100:9.5f} %   "
                  f"{MASK_SIGMA_U/max(r['M+P'],1e-30):7.0f}x better than masks")

    fig = plt.figure(figsize=(12.2, 3.4))
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    th_r = np.radians([r["azimuth"] for r in az_rows] + [az_rows[0]["azimuth"]])
    vals = np.array([r["M+P"] for r in az_rows] + [az_rows[0]["M+P"]]) * 100
    ax.plot(th_r, vals, "o-", color="tab:blue", lw=1.8, ms=4)
    ax.set_theta_zero_location("E")
    ax.plot([math.radians(wind_deg)] * 2, [0, vals.max()], "-",
            color="tab:green", lw=2.5, alpha=0.7)
    ax.set_ylim(0, vals.max() * 1.1)
    ax.set_title("CRB on wind speed (%) vs camera azimuth\n"
                 f"green = wind direction; spread only {spread:.2f}x", fontsize=8.5)
    ax.tick_params(labelsize=7)

    for idx, (rows, key, xlabel, title) in enumerate((
        (dist_rows, "distance", "standoff distance (m)", "standoff"),
        (h_rows, "height", "mast height (m)", "mast height"),
    ), start=2):
        ax = fig.add_subplot(1, 3, idx)
        ax.plot([r[key] for r in rows], [r["M+P"] * 100 for r in rows],
                "o-", color="tab:blue", lw=1.8)
        ax.axhline(MASK_SIGMA_U * 100, color="tab:red", ls="--", lw=1.2,
                   label="masks alone (profile bound)")
        ax.set_xlabel(xlabel, fontsize=8.5)
        if idx == 2:
            ax.set_ylabel("CRB on wind speed (%)", fontsize=8.5)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=6.5, frameon=False)
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("Does the plume advantage depend on where the camera is?",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    p = FIGS / "fig_camera_geometry.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")
    return spread


def main():
    if "--replot" in sys.argv:
        d = json.loads((RESULTS / "camera_geometry.json").read_text(encoding="utf-8"))
        spread = report(d)
        d["azimuth_spread"] = float(spread)
        d["mask_reference"] = MASK_SIGMA_U
        (RESULTS / "camera_geometry.json").write_text(json.dumps(d, indent=2),
                                                      encoding="utf-8")
        return

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = tuple(range(8, 110, 9))
    base = Scenario(device=device, n_steps=110, mask_times=times,
                    plume_times=times, conc_times=times)
    theta = pack(TRUTH, device=device)
    centre = fire_centre(base)
    wind_deg = math.degrees(TRUTH["theta_w"])
    print(f"fire centre ~ ({centre[0]:.0f}, {centre[1]:.0f}) m,  "
          f"wind blows towards {wind_deg:.0f} deg")

    def scen_with(cam):
        return Scenario(device=device, n_steps=110, mask_times=times,
                        plume_times=times, conc_times=times, camera=cam)

    # --- agreement check before trusting AD on this quantity ---
    print("\n[0] AD vs finite differences for mask+plume at the default geometry")
    ref = scen_with(make_camera(centre, wind_deg + 90, 900.0, 60.0))
    a = crb_U(ref, theta, method="ad")["M+P"]
    f = crb_U(ref, theta, method="central", eps=0.02)["M+P"]
    print(f"    AD {a*100:.5f} %   FD {f*100:.5f} %   ratio {a/max(f,1e-30):.3f}"
          f"   [target within 1.3x]")
    ok_agree = 0.77 < a / max(f, 1e-30) < 1.3
    print(f"    agreement acceptable: {ok_agree}")

    # --- azimuth sweep ---
    print("\n[1] camera azimuth, at 900 m standoff and 60 m mast")
    print(f"    ('vs masks' is against the profile-likelihood mask bound of "
          f"{MASK_SIGMA_U*100:.0f} %, so it is a lower bound on the gain)")
    print(f"    {'azimuth':>8s} {'rel. to wind':>13s} {'CRB(U) M+P':>13s} {'vs masks':>11s}")
    az_rows = []
    t0 = time.time()
    for az in AZIMUTHS:
        sc = scen_with(make_camera(centre, az, 900.0, 60.0))
        c = crb_U(sc, theta)
        rel = (az - wind_deg) % 360
        rel = rel - 360 if rel > 180 else rel
        az_rows.append({"azimuth": float(az), "rel_to_wind": float(rel), **c})
        print(f"    {az:8.1f} {rel:13.1f} {c['M+P']*100:12.5f}% "
              f"{c['M']/max(c['M+P'],1e-30):9.0f}x")
    print(f"    ({time.time()-t0:.0f}s)")

    best = min(az_rows, key=lambda r: r["M+P"])
    worst = max(az_rows, key=lambda r: r["M+P"])
    spread = worst["M+P"] / max(best["M+P"], 1e-30)
    print(f"\n    best  azimuth {best['azimuth']:.0f} deg "
          f"({best['rel_to_wind']:+.0f} deg from wind): {best['M+P']*100:.5f} %")
    print(f"    worst azimuth {worst['azimuth']:.0f} deg "
          f"({worst['rel_to_wind']:+.0f} deg from wind): {worst['M+P']*100:.5f} %")
    print(f"    spread across azimuth: {spread:.1f}x")
    worst_gain = min(r["M"] / max(r["M+P"], 1e-30) for r in az_rows)
    print(f"    worst-case gain over masks, over all azimuths: {worst_gain:.0f}x")

    # --- distance and height ---
    print("\n[2] standoff distance (azimuth fixed 90 deg across the wind, 60 m mast)")
    dist_rows = []
    for d in DISTANCES:
        sc = scen_with(make_camera(centre, wind_deg + 90, d, 60.0))
        c = crb_U(sc, theta)
        dist_rows.append({"distance": float(d), **c})
        print(f"    {d:7.0f} m   CRB(U) = {c['M+P']*100:9.5f} %   "
              f"{c['M']/max(c['M+P'],1e-30):7.0f}x better than masks")

    print("\n[3] mast height (azimuth 90 deg, 900 m standoff)")
    h_rows = []
    for h in HEIGHTS:
        sc = scen_with(make_camera(centre, wind_deg + 90, 900.0, h))
        c = crb_U(sc, theta)
        h_rows.append({"height": float(h), **c})
        print(f"    {h:7.0f} m   CRB(U) = {c['M+P']*100:9.5f} %   "
              f"{c['M']/max(c['M+P'],1e-30):7.0f}x better than masks")

    # --- figure ---
    fig = plt.figure(figsize=(12.2, 3.4))
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    th_r = np.radians([r["azimuth"] for r in az_rows] + [az_rows[0]["azimuth"]])
    vals = np.array([r["M+P"] for r in az_rows] + [az_rows[0]["M+P"]]) * 100
    ax.plot(th_r, vals, "o-", color="tab:blue", lw=1.8, ms=4)
    ax.set_theta_zero_location("E")
    ax.plot([TRUTH["theta_w"], TRUTH["theta_w"]], [0, vals.max()], "-",
            color="tab:green", lw=2.5, alpha=0.7)
    ax.set_title("CRB on wind speed (%) vs camera azimuth\n"
                 "green = wind direction", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot([r["distance"] for r in dist_rows],
            [r["M+P"] * 100 for r in dist_rows], "o-", color="tab:blue", lw=1.8)
    ax.axhline(az_rows[0]["M"] * 100, color="tab:red", ls="--", lw=1.2,
               label="masks alone")
    ax.set_xlabel("standoff distance (m)", fontsize=8.5)
    ax.set_ylabel("CRB on wind speed (%)", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_title("standoff", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot([r["height"] for r in h_rows],
            [r["M+P"] * 100 for r in h_rows], "o-", color="tab:blue", lw=1.8)
    ax.axhline(az_rows[0]["M"] * 100, color="tab:red", ls="--", lw=1.2,
               label="masks alone")
    ax.set_xlabel("mast height (m)", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_title("mast height", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Does the plume advantage depend on where the camera is?",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    p = FIGS / "fig_camera_geometry.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "camera_geometry.json").write_text(json.dumps({
        "truth": TRUTH, "centre": list(centre), "wind_deg": wind_deg,
        "ad_fd_check": {"ad": a, "fd": f, "ok": bool(ok_agree)},
        "azimuth": az_rows, "distance": dist_rows, "height": h_rows,
        "azimuth_spread": float(spread), "worst_case_gain": float(worst_gain),
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'camera_geometry.json'}")


if __name__ == "__main__":
    main()
