"""The degeneracy is invisible in the past and catastrophic in the future.

Everything measured so far says a burn mask cannot separate fuel from wind speed. That
is a statement about *estimation*. This is the statement about *consequences*, and it is
the reason any of it matters.

Wildfire forecasts fail, and people die, when the wind shifts: a flank becomes a head.
Operationally the future wind is the part you *do* know -- a weather model supplies it --
while the fuel underneath is what you had to infer from watching the fire. So forecasting
through a wind shift requires exactly the split that masks cannot make:

    R_head(new wind) = R0 路 (1 + phi_w(U_new))

A mask-only fit knows the product R0路(1 + phi_w(U_old)) but not R0. Every point on the
null curve reproduces the observed past identically and predicts a *different* future.

The experiment:
  1. watch a fire for 20 min under one wind,
  2. the wind shifts (stronger, veered 45 deg) -- and the shift is known,
  3. forecast the next 20 min,
  4. ask when the fire reaches a road.

The mask-only family is enumerated along the closed-form null curve, and each member is
verified to fit the observed past before its forecast is scored. That makes the spread an
irreducible property of the data, not an artefact of an optimiser stopping somewhere.

Run:  python scripts/wind_shift_forecast.py
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
from pyrofield.models.operators.mask import iou  # noqa: E402
from pyrofield.physics.rothermel import WIND_A, WIND_B  # noqa: E402
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

T_OBS = 80            # 20 min of observation at dt = 15 s
T_END = 160           # forecast to 40 min
U_NEW, TW_NEW = 7.0, np.pi / 2       # stronger wind, veered 45 deg to due north
HORIZONS = (20, 40, 80)              # +5, +10, +20 min after the shift
# Sample the forecast every minute so the road-crossing time is resolved rather than
# quantised onto the scoring horizons.
FORECAST_CADENCE = 4                 # steps; 4 x 15 s = 1 min

# A road to defend, downwind of the *new* wind direction.
ROAD_Y = 1080.0
ROAD_X = (350.0, 1150.0)

# Valley members: wind speeds that a mask-only fit cannot tell apart.
VALLEY_U = np.array([2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.6, 6.4])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def obs_scenario(device, n_steps=T_OBS):
    times = tuple(range(8, n_steps, 9))
    return Scenario(device=device, n_steps=n_steps, mask_times=times,
                    plume_times=times, conc_times=times)


def forecast_scenario(device, sample_steps):
    return Scenario(device=device, n_steps=T_END, mask_times=tuple(sample_steps),
                    plume_times=(), conc_times=(),
                    wind_after=(T_OBS, U_NEW, float(TW_NEW)))


def valley_member(U_hat: float) -> dict:
    """The (R0, k_LB) that reproduce the observed past at a wrong wind speed."""
    aub0 = WIND_A * TRUTH["U"] ** WIND_B
    R_head = TRUTH["R0"] * (1.0 + aub0)
    LB = 1.0 + TRUTH["lb_k"] * TRUTH["U"]
    return {**TRUTH, "U": float(U_hat),
            "R0": R_head / (1.0 + WIND_A * U_hat**WIND_B),
            "lb_k": (LB - 1.0) / U_hat}


def road_crossing_minute(masks, steps, dx):
    """First sampled time at which the burn reaches the road, in minutes from ignition."""
    j0 = int(ROAD_Y / dx)
    i0, i1 = int(ROAD_X[0] / dx), int(ROAD_X[1] / dx)
    for m, s in zip(masks, steps):
        if float(m[j0, i0:i1].max()) > 0.5:
            return (s + 1) * 15.0 / 60.0
    return None


def missed_area_ha(pred, truth, dx):
    """Area that burns but was not forecast to, in hectares -- the error that hurts."""
    miss = ((truth > 0.5) & (pred <= 0.5)).float().sum().item()
    return miss * dx * dx / 1e4


def over_area_ha(pred, truth, dx):
    """Area forecast to burn that does not -- the error that causes evacuation fatigue."""
    over = ((pred > 0.5) & (truth <= 0.5)).float().sum().item()
    return over * dx * dx / 1e4


def front_error_m(pred, truth, dx):
    """Largest distance from a truly-burned cell to the nearest forecast-burned cell."""
    t = (truth > 0.5)
    p = (pred > 0.5)
    if not bool(t.any()):
        return 0.0
    if not bool(p.any()):
        return float("inf")
    ys, xs = torch.nonzero(t, as_tuple=True)
    yp, xp = torch.nonzero(p, as_tuple=True)
    # Chunked to keep the pairwise distance matrix small.
    worst = 0.0
    for s in range(0, ys.numel(), 4096):
        dy = ys[s:s + 4096, None].float() - yp[None, :].float()
        dx_ = xs[s:s + 4096, None].float() - xp[None, :].float()
        d = torch.sqrt(dy * dy + dx_ * dx_).min(dim=1).values.max().item()
        worst = max(worst, d)
    return worst * dx


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta_true = pack(TRUTH, device=device)

    steps = tuple(range(T_OBS, T_END, FORECAST_CADENCE))
    score_at = [min(range(len(steps)), key=lambda i: abs(steps[i] - (T_OBS + h - 1)))
                for h in HORIZONS]
    fsc = forecast_scenario(device, steps)
    osc = obs_scenario(device)

    # --- ground truth future ---
    truth_masks = forward(theta_true, fsc)["mask"]
    t_true = road_crossing_minute(truth_masks, steps, fsc.dx)
    print(f"truth: fire reaches the road at "
          + (f"{t_true:.1f} min" if t_true else "not within 40 min"))
    print(f"forecast sampled every {FORECAST_CADENCE*15/60:.0f} min; "
          f"scored at +{', +'.join(str(h*15//60) for h in HORIZONS)} min")

    # --- observations of the past ---
    gen = torch.Generator(device=device).manual_seed(11)
    target = add_noise(forward(theta_true, osc), gen)

    def past_chi2(theta, keys):
        sig = {k: noise_sigma_for(target, k) for k in keys}
        w = fixed_weights(target, keys, sig)
        y = stack_obs(target, keys) * w
        return float((((stack_obs(forward(theta, osc), keys) * w) - y) ** 2).sum())

    chi2_truth = past_chi2(theta_true, ("mask",))
    print(f"chi2 of the truth on the observed past (masks only): {chi2_truth:.4e}")

    # --- the family of mask-only fits ---
    print(f"\n{'U_hat':>7s} {'R0_hat':>9s} {'k_LB_hat':>9s} {'past dchi2':>11s} "
          f"{'road(min)':>10s} {'IoU+20':>8s} {'missed ha':>10s} {'over ha':>9s} "
          f"{'front m':>9s}")
    rows = []
    t0 = time.time()
    for U_hat in VALLEY_U:
        m = valley_member(float(U_hat))
        th = pack(m, device=device)
        d = past_chi2(th, ("mask",)) - chi2_truth
        fm = forward(th, fsc)["mask"]
        ious = [float(iou(fm[i], truth_masks[i])) for i in score_at]
        missed = [missed_area_ha(fm[i], truth_masks[i], fsc.dx) for i in score_at]
        over = [over_area_ha(fm[i], truth_masks[i], fsc.dx) for i in score_at]
        ferr = [front_error_m(fm[i], truth_masks[i], fsc.dx) for i in score_at]
        tr = road_crossing_minute(fm, steps, fsc.dx)
        rows.append({"U_hat": float(U_hat), "R0": m["R0"], "lb_k": m["lb_k"],
                     "past_dchi2": d, "road_min": tr, "iou": ious,
                     "missed_ha": missed, "over_ha": over, "front_err_m": ferr})
        print(f"{U_hat:7.2f} {m['R0']:9.5f} {m['lb_k']:9.4f} {d:11.2f} "
              + (f"{tr:10.1f}" if tr else f"{'never':>10s}")
              + f" {ious[-1]:8.3f} {missed[-1]:10.1f} {over[-1]:9.1f} {ferr[-1]:9.0f}")
    print(f"   ({time.time()-t0:.0f}s)")

    print("\n  The two directions of error are not equivalent:")
    print("    under-estimating the wind -> over-estimating the fuel -> forecast too fast")
    print("       -> early warning, wasted evacuation (the 'over ha' column)")
    print("    over-estimating the wind  -> under-estimating the fuel -> forecast too slow")
    print("       -> LATE warning, land burns unforecast (the 'missed ha' column)")
    print("    A mask-only fit has no way to know which side of the valley it is on.")

    # --- what actual fits return, from several starting points ---
    # The family above is irreducible; this shows it is not hypothetical. Which member an
    # optimiser hands you is decided by where it happened to start, and the resulting
    # forecast moves with it.
    print("\nrepeated mask-only fits from different starting points")
    rng = np.random.default_rng(3)
    restarts = []
    for r in range(6):
        start = theta_true.clone()
        for i in range(len(PARAM_NAMES)):
            start[i] = start[i] + float(rng.uniform(-0.40, 0.40))
        res = levenberg_marquardt(start, theta_true, target, osc, ("mask",))
        p = unpack(res.theta_hat)
        fm = forward(res.theta_hat, fsc)["mask"]
        tr = road_crossing_minute(fm, steps, fsc.dx)
        restarts.append({
            "U": float(p["U"]), "R0": float(p["R0"]), "loss": res.loss,
            "road_min": tr,
            "iou20": float(iou(fm[score_at[-1]], truth_masks[score_at[-1]])),
            "missed_ha": missed_area_ha(fm[score_at[-1]], truth_masks[score_at[-1]], fsc.dx),
        })
        print(f"  start {r}: fitted U = {p['U']:5.3f} m/s, R0 = {p['R0']:.5f}"
              f"   past loss {res.loss:.4e}   road at "
              + (f"{tr:.1f} min ({tr-t_true:+.1f})" if tr else "never")
              + f"   IoU+20 {restarts[-1]['iou20']:.3f}")
    ru = [x["U"] for x in restarts]
    rr = [x["road_min"] for x in restarts if x["road_min"]]
    print(f"  fitted wind speed across restarts: {min(ru):.2f} to {max(ru):.2f} m/s "
          f"(truth {TRUTH['U']})")
    if rr:
        print(f"  predicted road-crossing time:      {min(rr):.1f} to {max(rr):.1f} min "
              f"(truth {t_true:.1f})")

    print("\nfitting the observed past")
    fits = {}
    rng = np.random.default_rng(3)
    for label, keys in (("M", ("mask",)), ("M+P", ("mask", "plume"))):
        start = theta_true.clone()
        for i in range(len(PARAM_NAMES)):
            start[i] = start[i] + float(rng.uniform(-0.35, 0.35))
        res = levenberg_marquardt(start, theta_true, target, osc, keys)
        p = unpack(res.theta_hat)
        fm = forward(res.theta_hat, fsc)["mask"]
        ious = [float(iou(fm[i], truth_masks[i])) for i in score_at]
        missed = [missed_area_ha(fm[i], truth_masks[i], fsc.dx) for i in score_at]
        tr = road_crossing_minute(fm, steps, fsc.dx)
        fits[label] = {
            "U": float(p["U"]), "R0": float(p["R0"]), "lb_k": float(p["lb_k"]),
            "past_dchi2": past_chi2(res.theta_hat, keys) - past_chi2(theta_true, keys),
            "road_min": tr, "iou": ious, "missed_ha": missed,
            "front_err_m": [front_error_m(fm[i], truth_masks[i], fsc.dx) for i in score_at],
        }
        print(f"  {label:4s}: U={p['U']:.3f} (true {TRUTH['U']}), "
              f"R0={p['R0']:.5f} (true {TRUTH['R0']})   "
              f"road at " + (f"{tr:.1f} min" if tr else "never")
              + f"   IoU+20 {ious[-1]:.3f}   missed {missed[-1]:.1f} ha")

    # --- headline ---
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    past_spread = max(r["past_dchi2"] for r in rows)
    crossings = [r["road_min"] for r in rows if r["road_min"] is not None]
    never = sum(1 for r in rows if r["road_min"] is None)
    iou_last = [r["iou"][-1] for r in rows]
    print(f"  all {len(rows)} mask-only fits agree on the past to within "
          f"delta chi^2 = {past_spread:.1f} (1 sigma = 1)")
    if crossings:
        print(f"  they predict the fire reaches the road between {min(crossings):.1f} and "
              f"{max(crossings):.1f} min" + (f", and {never} say never" if never else ""))
        print(f"  truth: {t_true:.1f} min")
        worst = max(abs(c - t_true) for c in crossings)
        print(f"  worst error in the crossing time: {worst:.1f} min")
    print(f"  forecast IoU at +20 min ranges {min(iou_last):.3f} to {max(iou_last):.3f}")
    missed_last = [r["missed_ha"][-1] for r in rows]
    print(f"  land that burns but was not forecast to: "
          f"{min(missed_last):.0f} to {max(missed_last):.0f} ha")
    print(f"  worst front-position error at +20 min: "
          f"{max(r['front_err_m'][-1] for r in rows):.0f} m")
    print(f"\n  the single mask-only fit an optimiser returns: road at "
          f"{fits['M']['road_min']:.1f} min "
          f"({fits['M']['road_min']-t_true:+.1f} min), missed {fits['M']['missed_ha'][-1]:.0f} ha")
    print(f"  the mask+plume fit: road at {fits['M+P']['road_min']:.1f} min "
          f"({fits['M+P']['road_min']-t_true:+.1f} min), "
          f"IoU {fits['M+P']['iou'][-1]:.3f}, missed {fits['M+P']['missed_ha'][-1]:.0f} ha")

    # --- figure ---
    fig = plt.figure(figsize=(12.6, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.3)
    extent = [0, fsc.nx * fsc.dx, 0, fsc.ny * fsc.dx]

    obs_last = forward(theta_true, osc)["mask"][-1].detach().cpu().numpy()
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(obs_last, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1)
    ax.axhline(ROAD_Y, color="cyan", lw=1.6)
    ax.set_title("what was observed\n(20 min, before the shift)", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    lo = min(rows, key=lambda r: r["U_hat"])
    hi = max(rows, key=lambda r: r["U_hat"])
    for k, (r, ttl) in enumerate(((lo, f"fit with U={lo['U_hat']:.1f} m/s"),
                                  (hi, f"fit with U={hi['U_hat']:.1f} m/s")), start=1):
        th = pack(valley_member(r["U_hat"]), device=device)
        fm = forward(th, fsc)["mask"][-1].detach().cpu().numpy()
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(truth_masks[-1].detach().cpu().numpy(), origin="lower", extent=extent,
                  cmap="Greys", vmin=0, vmax=2.2)
        ax.contour(np.linspace(extent[0], extent[1], fsc.nx),
                   np.linspace(extent[2], extent[3], fsc.ny),
                   fm, levels=[0.5], colors="tab:red", linewidths=1.8)
        ax.axhline(ROAD_Y, color="cyan", lw=1.6)
        ax.set_title(f"{ttl}\nforecast +20 min, IoU {r['iou'][-1]:.2f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(gs[0, 3])
    u = [r["U_hat"] for r in rows]
    cm = [r["road_min"] if r["road_min"] else np.nan for r in rows]
    ax.plot(u, cm, "o-", color="tab:red", lw=1.8, label="mask-only fits")
    ax.axhline(t_true, color="k", ls="--", lw=1.4, label="truth")
    if fits["M+P"]["road_min"]:
        ax.plot([fits["M+P"]["U"]], [fits["M+P"]["road_min"]], "*", ms=16,
                color="tab:blue", label="mask+plume fit")
    ax.set_xlabel("wind speed the fit settled on (m/s)", fontsize=8.5)
    ax.set_ylabel("predicted time the fire\nreaches the road (min)", fontsize=8.5)
    ax.set_title("All of these fit the past equally well", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("A wind shift turns an invisible degeneracy into a forecast failure",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_wind_shift_forecast.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "wind_shift_forecast.json").write_text(json.dumps({
        "truth": TRUTH, "T_obs": T_OBS, "T_end": T_END,
        "wind_after": {"U": U_NEW, "theta_w": float(TW_NEW)},
        "road": {"y": ROAD_Y, "x": list(ROAD_X)},
        "truth_road_min": t_true, "horizons_steps": list(HORIZONS),
        "valley": rows, "fits": fits, "restarts": restarts,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'wind_shift_forecast.json'}")


if __name__ == "__main__":
    main()

