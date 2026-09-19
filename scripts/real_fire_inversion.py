"""Invert a real fire, and check the answer against the wind that actually blew.

Section 20 lists this as the study's largest untested claim: every inversion so far has
been an identical-twin experiment, and "no real fire in this study was inverted for its
parameters". WildfireSpreadTS makes it possible -- a real fire's observed day-by-day
progression, with the GRIDMET wind that drove it recorded independently -- so the excuse
is gone.

The fit is the project's own forward model, run at the satellite's 375 m resolution against
the observed burn masks, with fuel, wind speed, wind bearing and the shape coefficient
free. Then three things are checked, and the third is the one the study is about.

  1. **Does it fit?** A model that cannot reproduce a real burn at all says nothing about
     identifiability.
  2. **Is the recovered wind right?** Fitted bearing and speed against GRIDMET.
  3. **Does the guard know?** `pyrofield/eval/guarded.py` is run on every fit. Sections 1
     and 13 predict that wind speed is not determined by a burn mask, so on real fires the
     guard should refuse it -- and the error against GRIDMET should be large whether or not
     the fit looks good.

Everything here is harder than the synthetic case and expected to be messier: the model is
elliptical and real perimeters are not (section 3: median IoU 0.705), the fuel is uniform
in the model and is not in the world, and GRIDMET is a 4 km daily summary rather than the
wind at the flame. Those are reasons the *absolute* errors will be poor. They are not
reasons the guard's verdict should be wrong.

Run:  python scripts/fetch_wsts.py --fires 0
      python scripts/real_fire_inversion.py --fires 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
import torch
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WSTS, RESULTS, FIGS = ROOT / "data" / "wsts", ROOT / "results", ROOT / "figures"

from pyrofield.eval.guarded import guard  # noqa: E402
from pyrofield.eval.inversion import fixed_weights, levenberg_marquardt  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    forward,
    noise_sigma_for,
    pack,
)
from pyrofield.theory.identifiability import stack_obs  # noqa: E402

B_WIND, B_DIR, B_AF = 6, 7, 22
PX = 375.0
MAX_JUMP_M, SWATH_FRAC = 4000.0, 0.80
NX = 96                      # model grid; the crop is resampled onto it
MIN_BURN, MIN_DAYS = 40, 4   # a fire worth fitting
FREE = ["R0", "U", "theta_w", "lb_k"]   # the smoke parameters are unobservable here
U_STARTS = (1.5, 3.0, 6.0)   # the same fire fitted from three plausible winds
MASK_SCHEDULE = ((0.0, 14),)  # no plume here, so no blur ladder is needed


def gated_series(fire_dir):
    """Cumulative gated burn masks per day, plus that day's GRIDMET wind."""
    out, cum = [], None
    for p in sorted(fire_dir.glob("*.tif")):
        try:
            with rasterio.open(p) as src:
                a = src.read()
        except Exception:
            continue
        af = np.isfinite(a[B_AF])
        if cum is None:
            if not af.any():
                continue
            lab, n = ndimage.label(af, structure=np.ones((3, 3)))
            sizes = ndimage.sum(af, lab, range(1, n + 1))
            cum = lab == (int(np.argmax(sizes)) + 1)
        raw_new = af & ~cum
        if raw_new.any():
            rr, cc = np.nonzero(raw_new)
            if (max((rr.max() - rr.min() + 1) / af.shape[0],
                    (cc.max() - cc.min() + 1) / af.shape[1]) > SWATH_FRAC
                    and raw_new.sum() > 500):
                continue
            reach = ndimage.distance_transform_edt(~cum) * PX
            cum = cum | (raw_new & (reach <= MAX_JUMP_M))
        out.append({"date": p.stem, "cum": cum.copy(),
                    "wind": float(np.nanmedian(a[B_WIND][cum])) if cum.any()
                    else float(np.nanmedian(a[B_WIND])),
                    "dir": float(np.nanmedian(a[B_DIR][cum])) if cum.any()
                    else float(np.nanmedian(a[B_DIR]))})
    return out


def to_model_grid(masks, nx=NX):
    """Crop to the fire and resample onto the model's square grid.

    Returns the masks, the ignition cell in model coordinates, and the metres a model
    cell represents, so the fitted rate can be read back in m/day.
    """
    last = masks[-1]
    rr, cc = np.nonzero(last)
    r0, r1 = rr.min(), rr.max() + 1
    c0, c1 = cc.min(), cc.max() + 1
    # a margin so the front is not against the boundary
    h, w = r1 - r0, c1 - c0
    m = int(0.35 * max(h, w)) + 2
    r0, r1 = max(r0 - m, 0), min(r1 + m, last.shape[0])
    c0, c1 = max(c0 - m, 0), min(c1 + m, last.shape[1])
    side = max(r1 - r0, c1 - c0)
    out = []
    for mk in masks:
        sub = np.zeros((side, side), bool)
        blk = mk[r0:r1, c0:c1]
        sub[:blk.shape[0], :blk.shape[1]] = blk
        z = ndimage.zoom(sub.astype(np.float32), nx / side, order=1) > 0.5
        z = z[:nx, :nx]
        pad = np.zeros((nx, nx), bool)
        pad[:z.shape[0], :z.shape[1]] = z
        out.append(pad)
    first = out[0]
    if first.any():
        ir, ic = ndimage.center_of_mass(first)
    else:
        ir = ic = nx / 2
    return np.array(out), (float(ir), float(ic)), side * PX / nx


def observed_rate(masks, cell_m, days):
    """Equivalent-radius growth rate of the observed burn, in m/s.

    The scale matters more than it looks. A first version fixed the model's time step at
    days/110 and started R0 at 0.05 m/s, which is roughly forty times what these fires
    actually do; the level set then ran at CFL 5-19, went NaN, and every fit froze exactly
    at its starting values -- producing an apparent 5 % wind-speed accuracy that was
    nothing but the starting guess sitting near the typical GRIDMET wind. Section 19's own
    lesson is to hold the CFL number fixed rather than the time step, and this is where the
    rate to hold it against comes from.
    """
    a0 = float(masks[0].sum()) * cell_m * cell_m
    a1 = float(masks[-1].sum()) * cell_m * cell_m
    dr = np.sqrt(a1 / np.pi) - np.sqrt(a0 / np.pi)
    return max(dr, cell_m) / (days * 86400.0)


def fit_fire(masks, ign, cell_m, days, device, u_start=3.0, cfl=0.20, max_steps=240):
    """Fit the forward model to an observed progression. Returns the result and scenario."""
    r_obs = observed_rate(masks, cell_m, days)
    total = days * 86400.0
    # Head rate at the starting parameters, with headroom for the fit to speed up.
    head = 3.0 * r_obs * (1.0 + 0.35 * u_start ** 1.3)
    dt = min(cfl * cell_m / head, total / 8.0)
    n_steps = int(min(max(round(total / dt), 24), max_steps))
    dt = total / n_steps
    times = np.linspace(0, n_steps, len(masks) + 1)[1:].astype(int) - 1
    sc = Scenario(device=device, nx=NX, ny=NX, dx=cell_m, dt=dt, n_steps=n_steps,
                  mask_times=tuple(int(t) for t in times),
                  plume_times=(), conc_times=(),
                  ignition_xy=(ign[1] * cell_m, ign[0] * cell_m),
                  ignition_radius=max(1.5 * cell_m, np.sqrt(
                      float(masks[0].sum()) * cell_m * cell_m / np.pi)))
    target = {"mask": torch.from_numpy(masks.astype(np.float32)).to(device),
              "plume": torch.zeros(0, device=device),
              "conc": torch.zeros(0, device=device)}
    keys = ("mask",)
    start = pack({"R0": float(r_obs), "U": float(u_start), "theta_w": 0.0,
                  "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}, device=device)
    free_idx = [PARAM_NAMES.index(k) for k in FREE]
    # The default schedule's coarse-to-fine ladder blurs the plume image, and there is no
    # plume here, so its four stages are 33 plain iterations doing the work of 14.
    res = levenberg_marquardt(start, start, target, sc, keys, free_idx=free_idx,
                              schedule=MASK_SCHEDULE)
    return res, sc, target, keys, {"r_obs": float(r_obs), "dt": float(dt),
                                   "n_steps": int(n_steps),
                                   "cfl": float(head * dt / cell_m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fires", type=int, default=40)
    ap.add_argument("--guard", action="store_true", default=True)
    ap.add_argument("--report-only", action="store_true",
                    help="re-analyse results/real_fire_inversion.json without refitting")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    if args.report_only:
        saved = json.loads((RESULTS / "real_fire_inversion.json").read_text(
            encoding="utf-8"))
        report(saved["rows"])
        return

    dirs = sorted({p.parent for p in WSTS.glob("*/*/*.tif")
                   if not p.name.endswith(".part")})
    print(f"{len(dirs)} fire events on disk")

    rows, t0, scanned = [], time.time(), 0
    seen_masks, dupes = set(), 0
    for fd in dirs:
        if len(rows) >= args.fires:
            break
        scanned += 1
        if scanned % 25 == 0:
            print(f"    scanned {scanned} fires, {len(rows)} fitted "
                  f"({time.time()-t0:.0f}s)", flush=True)
        ser = gated_series(fd)
        if len(ser) < MIN_DAYS or ser[-1]["cum"].sum() < MIN_BURN:
            continue
        keep = [s for s in ser if s["cum"].sum() >= 8]
        if len(keep) < MIN_DAYS:
            continue
        keep = keep[:8]
        masks_full = np.array([s["cum"] for s in keep])
        d0 = np.datetime64(keep[0]["date"])
        days = float((np.datetime64(keep[-1]["date"]) - d0) / np.timedelta64(1, "D"))
        if days < 2:
            continue
        masks, ign, cell_m = to_model_grid(masks_full)
        # the fire has to actually grow, or there is no progression to fit
        if (masks[-1].sum() < 40 or masks[-1].mean() > 0.6
                or masks[-1].sum() < 1.5 * masks[0].sum()):
            continue
        # GlobFire, which WildfireSpreadTS is built from, records some physical fires
        # twice under neighbouring ids with different date ranges and tile bounds. Two
        # such records reduce to the same resampled mask stack and then to bit-identical
        # fits, which would quietly inflate the sample. Drop the repeat.
        h = hash(masks.tobytes())
        if h in seen_masks:
            dupes += 1
            continue
        seen_masks.add(h)

        w_obs = float(np.median([s["wind"] for s in keep]))
        d_obs = float(np.median([s["dir"] for s in keep]))
        brg_obs = (d_obs + 180.0) % 360.0     # GRIDMET names where the wind comes from

        # The identifiability test: fit the same fire from wind speeds spanning the
        # plausible range. If the burn determines the wind, the fits converge on one
        # answer; if it does not, each fit stays near wherever it started and they all
        # explain the data equally well.
        fits = []
        for u0 in U_STARTS:
            try:
                res, sc, target, keys, meta = fit_fire(masks, ign, cell_m, days,
                                                       device, u_start=u0)
            except Exception as e:
                print(f"    {fd.name}: fit from {u0} m/s failed, {repr(e)[:50]}")
                continue
            if not np.isfinite(res.loss):
                continue
            pred = forward(res.theta_hat, sc)["mask"]
            obs = target["mask"]
            iou = float(((pred > 0.5) & (obs > 0.5)).sum()) / max(
                float(((pred > 0.5) | (obs > 0.5)).sum()), 1.0)
            th_fit = float(res.theta_hat[PARAM_NAMES.index("theta_w")])
            fits.append({
                "u_start": u0,
                "U_fit": float(torch.exp(res.theta_hat[PARAM_NAMES.index("U")])),
                "R0_fit": float(torch.exp(res.theta_hat[PARAM_NAMES.index("R0")])),
                "bearing_fit": float(np.degrees(np.arctan2(
                    np.sin(th_fit), np.cos(th_fit))) % 360),
                "iou": iou, "loss": float(res.loss),
                "chi2_per_obs": float(res.loss) / max(obs.numel(), 1),
                **meta})
        if len(fits) < len(U_STARTS):
            continue

        best = min(fits, key=lambda f: f["loss"])
        losses = np.array([f["loss"] for f in fits])
        us = np.array([f["U_fit"] for f in fits])
        rec = {"fire": f"{fd.parent.name}/{fd.name}", "days": days,
               "n_masks": len(keep), "cell_m": cell_m,
               "U_gridmet": w_obs, "bearing_gridmet": brg_obs,
               "fits": fits,
               "iou": best["iou"],
               "U_fit": best["U_fit"],
               "U_rel_err": float(abs(best["U_fit"] - w_obs) / max(w_obs, 1e-6)),
               "bearing_fit": best["bearing_fit"],
               "bearing_err": float(abs((best["bearing_fit"] - brg_obs + 180) % 360 - 180)),
               # how much the answer depends on where the fit started
               "U_spread": float(us.max() / max(us.min(), 1e-9)),
               "U_start_spread": float(max(U_STARTS) / min(U_STARTS)),
               "loss_spread": float(losses.max() / max(losses.min(), 1e-9))}

        if args.guard:
            res, sc, target, keys, _ = fit_fire(masks, ign, cell_m, days, device,
                                                u_start=best["u_start"])
            w = fixed_weights(target, keys,
                              {k: noise_sigma_for(target, k) for k in keys})
            base = float((((stack_obs(forward(res.theta_hat, sc), keys) * w)
                           - stack_obs(target, keys) * w) ** 2).sum())
            v, _ = guard(res.theta_hat, target, sc, keys, base_loss=base,
                         params=("U", "theta_w"))
            rec["guard"] = {x.name: {"determined": x.determined,
                                     "dchi2": x.dchi2,
                                     "sigma_rel": x.sigma_rel,
                                     "sigma_abs": x.sigma_abs} for x in v}
        rows.append(rec)
        g = rec.get("guard", {})
        print(f"  {len(rows):3d} {rec['fire']:22s} IoU {rec['iou']:.3f}  "
              f"U {rec['U_fit']:5.2f} vs {w_obs:5.2f}  "
              f"spread {rec['U_spread']:4.1f}x (starts {rec['U_start_spread']:.0f}x)  "
              f"misfit spread {rec['loss_spread']:.3f}x  "
              f"brg err {rec['bearing_err']:5.1f}d  "
              f"{'det' if g.get('U', {}).get('determined') else 'NOT det'}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("no fire met the criteria")
    print(f"\n  scanned {scanned} fire records, fitted {len(rows)}, "
          f"dropped {dupes} as duplicate footprints")
    report(rows, dupes)


def report(rows, dupes=None):
    # Exact duplicates that predate the mask-hash check show up as identical fits.
    sig = {}
    for r in rows:
        k = (round(r["iou"], 6), round(r["U_fit"], 6), round(r["bearing_fit"], 4))
        sig.setdefault(k, []).append(r["fire"])
    repeats = {k: v for k, v in sig.items() if len(v) > 1}
    if repeats:
        n_drop = sum(len(v) - 1 for v in repeats.values())
        keep = {v[0] for v in sig.values()}
        rows = [r for r in rows if r["fire"] in keep]
        print(f"\n  NOTE: {n_drop} of the fits are bit-identical to another fire's and "
              f"have been dropped\n        as duplicate GlobFire records: "
              + "; ".join(" = ".join(v) for v in list(repeats.values())[:3]))
        print(f"        {len(rows)} independent fires remain.")
    iou = np.array([r["iou"] for r in rows])
    ue = np.array([r["U_rel_err"] for r in rows])
    be = np.array([r["bearing_err"] for r in rows])
    det = np.array([r["guard"]["U"]["determined"] for r in rows if "guard" in r])
    detd = np.array([r["guard"]["theta_w"]["determined"] for r in rows if "guard" in r])

    print()
    print("=" * 78)
    print(f"[1] does the model fit a real burn at all?  ({len(rows)} fires)")
    print("=" * 78)
    print(f"    IoU of the fitted final front with the observed one: "
          f"median {np.median(iou):.3f}, p25 {np.percentile(iou,25):.3f}, "
          f"p75 {np.percentile(iou,75):.3f}")
    print(f"    for scale, section 3 found the best area-matched ellipse reaches a median")
    print(f"    IoU of 0.705 on real final perimeters, so this is the ceiling to beat.")

    print()
    print("=" * 78)
    print("[2] does the answer depend on where the fit started?")
    print("=" * 78)
    us = np.array([r["U_spread"] for r in rows])
    ls = np.array([r["loss_spread"] for r in rows])
    print(f"    Each fire was fitted three times, from {U_STARTS[0]}, {U_STARTS[1]} and "
          f"{U_STARTS[2]} m/s -- a {max(U_STARTS)/min(U_STARTS):.0f}x span.")
    print(f"    ratio of largest to smallest fitted wind, per fire: median "
          f"{np.median(us):.2f}x, p90 {np.percentile(us,90):.2f}x")
    print(f"    ratio of largest to smallest final misfit:          median "
          f"{np.median(ls):.4f}x, p90 {np.percentile(ls,90):.4f}x")
    print(f"    fires where the three fits stay more than 1.5x apart: "
          f"{float((us>1.5).mean())*100:.0f} %")
    print("\n    If the burn determined the wind, the three fits would converge on one")
    print("    answer and only one of them would fit well. A spread in the answers with")
    print("    no spread in the misfit is the degeneracy, on real fires.")

    print()
    print("=" * 78)
    print("[3] is the recovered wind right?")
    print("=" * 78)
    print(f"    wind speed, relative error against GRIDMET: median "
          f"{np.median(ue)*100:.0f} %, p25 {np.percentile(ue,25)*100:.0f} %, "
          f"p75 {np.percentile(ue,75)*100:.0f} %")
    print(f"    wind bearing, absolute error: median {np.median(be):.0f} deg, "
          f"p25 {np.percentile(be,25):.0f}, p75 {np.percentile(be,75):.0f}")
    print(f"    fires whose bearing lands within 45 deg: "
          f"{float((be<=45).mean())*100:.0f} %; a random bearing would give 25 %")

    print()
    print("=" * 78)
    print("[4] does the guard know it cannot have the wind speed?")
    print("=" * 78)
    if det.size:
        print(f"    wind speed called determined: {det.mean()*100:.0f} % of fires")
        print(f"    wind bearing called determined: {detd.mean()*100:.0f} %")
        print("\n    Sections 1 and 13 say a burn mask cannot give the speed and can give")
        print("    the bearing on flat ground. The guard is applied to real fires with no")
        print("    knowledge of either, and its verdicts are what they are.")

    print()
    print("=" * 78)
    print("[5] separating 'not identifiable' from 'the model does not fit'")
    print("=" * 78)
    print("    On real fires those two explanations are confounded, and the honest way to")
    print("    look at it is to ask whether the fires the model *does* fit behave any")
    print("    differently. If the wind pins down where the fit is good, the failure is")
    print("    misspecification; if it does not, misspecification is not the explanation.")
    usp = np.array([r["U_spread"] for r in rows])
    print(f"\n    {'fit quality':>18} {'fires':>7} {'median U spread':>17} "
          f"{'median U error':>16} {'bearing err':>13}")
    strat = {}
    qs = np.percentile(iou, [0, 50, 100])
    for lo, hi, lab in ((qs[0] - 1e-9, qs[1], f"IoU < {qs[1]:.2f}"),
                        (qs[1], qs[2] + 1e-9, f"IoU >= {qs[1]:.2f}")):
        m = (iou >= lo) & (iou <= hi if hi > qs[1] else iou < hi)
        if m.sum() < 3:
            continue
        strat[lab] = {"n": int(m.sum()), "U_spread": float(np.median(usp[m])),
                      "U_rel_err": float(np.median(ue[m])),
                      "bearing_err": float(np.median(be[m]))}
        print(f"    {lab:>18} {m.sum():7d} {np.median(usp[m]):16.1f}x "
              f"{np.median(ue[m])*100:15.0f} % {np.median(be[m]):12.0f}d")
    globals()["_STRAT"] = strat

    usp = np.array([r["U_spread"] for r in rows])
    lsp = np.array([r["loss_spread"] for r in rows])
    out = {"n": len(rows), "u_starts": list(U_STARTS),
           "iou": {"median": float(np.median(iou)), "p25": float(np.percentile(iou, 25)),
                   "p75": float(np.percentile(iou, 75))},
           "U_spread": {"median": float(np.median(usp)),
                        "p90": float(np.percentile(usp, 90)),
                        "frac_above_1p5": float((usp > 1.5).mean())},
           "loss_spread": {"median": float(np.median(lsp)),
                           "p90": float(np.percentile(lsp, 90))},
           "U_rel_err": {"median": float(np.median(ue)),
                         "p25": float(np.percentile(ue, 25)),
                         "p75": float(np.percentile(ue, 75))},
           "bearing_err": {"median": float(np.median(be)),
                           "p25": float(np.percentile(be, 25)),
                           "p75": float(np.percentile(be, 75)),
                           "within_45": float((be <= 45).mean())},
           "guard": {"U_determined": float(det.mean()) if det.size else None,
                     "theta_w_determined": float(detd.mean()) if detd.size else None},
           "by_fit_quality": globals().get("_STRAT", {}),
           "rows": rows}
    (RESULTS / "real_fire_inversion.json").write_text(json.dumps(out, indent=1),
                                                      encoding="utf-8")
    print(f"\nwrote {RESULTS/'real_fire_inversion.json'}")
    make_figure(out, rows)


def make_figure(out, rows):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    iou = np.array([r["iou"] for r in rows])
    uf = np.array([r["U_fit"] for r in rows])
    ug = np.array([r["U_gridmet"] for r in rows])
    be = np.array([r["bearing_err"] for r in rows])
    det = np.array([r["guard"]["U"]["determined"] for r in rows])

    ax = axes[0]
    ax.hist(iou, bins=16, color="tab:purple", alpha=0.85)
    ax.axvline(0.705, color="k", ls="--", lw=1.2)
    ax.text(0.71, ax.get_ylim()[1] * 0.85, "best ellipse\n(section 3)", fontsize=6.6)
    ax.set_xlabel("IoU of fitted front with observed burn", fontsize=8.5)
    ax.set_ylabel("fires", fontsize=8.5)
    ax.set_title(f"Fitting the model to {out['n']} real fires", fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    order = np.argsort(ug)
    cols = {1.5: "tab:blue", 3.0: "tab:orange", 6.0: "tab:red"}
    for j, i in enumerate(order):
        fits = rows[i]["fits"]
        ax.plot([j, j], [min(f["U_fit"] for f in fits),
                         max(f["U_fit"] for f in fits)],
                color="0.7", lw=1.0, zorder=1)
        for f in fits:
            ax.plot(j, f["U_fit"], "o", ms=4, zorder=2,
                    color=cols.get(f["u_start"], "k"),
                    label=f"fit started at {f['u_start']} m/s" if j == 0 else None)
    ax.plot(range(len(order)), ug[order], "k_", ms=11, mew=2, zorder=3,
            label="GRIDMET wind (the answer)")
    ax.set_yscale("log")
    ax.set_xlabel("fires, ordered by the real wind", fontsize=8.5)
    ax.set_ylabel("wind speed (m/s)", fontsize=8.5)
    ax.set_title("Three starts, one fire each: the fits\ndo not converge on the answer",
                 fontsize=9.5)
    ax.set_ylim(0.12, 40)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    ax.hist(be, bins=np.arange(0, 190, 15), color="tab:green", alpha=0.85)
    ax.axvline(45, color="k", ls="--", lw=1.2)
    ax.text(47, ax.get_ylim()[1] * 0.85, "within 45°", fontsize=6.6)
    ax.set_xlabel("bearing error against GRIDMET (deg)", fontsize=8.5)
    ax.set_ylabel("fires", fontsize=8.5)
    ax.set_title(f"Bearing: {out['bearing_err']['within_45']*100:.0f} % within 45°\n"
                 "(25 % if guessing)", fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle("Inverting real fires, and checking against the wind that actually blew",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_real_fire_inversion.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
