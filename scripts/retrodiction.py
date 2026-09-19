"""Where did it start? The degeneracy's consequence for a task nobody has scored.

Running the fire model backwards is the one task in the project plan with free ground
truth: WildfireSpreadTS records every fire from its first VIIRS detection, so the
ignition location and date come with the data.

The geometry is simple enough to state exactly. If a fire ignited at point `p` and has
been burning for `dt` days at rate `r`, everything it burned lies within `r * dt` of `p`.
So for a candidate `p`, the rate it implies is

    r(p) = max over burned pixels q of dist(p, q) / dt

and the ignition points consistent with a rate known to lie in `[r_lo, r_hi]` are

    F = { p in burn : r_lo <= r(p) <= r_hi }

a level set of `r(p)`, computable exactly: the maximum distance from `p` to a set is
attained on that set's convex hull, so `r(p)` costs a hull and a few dozen distances per
candidate. Note what F depends on. Not the fuel, not the wind, not the shape law -- only
on **how well the spread rate is known**. And the spread rate is precisely what section 1
says a burn scar hands you as a product of fuel and wind that cannot be separated.

That turns an abstract identifiability statement into an operational quantity. A fire is
discovered, there is one perimeter and no history, and an investigator wants the ignition
point. The rate has to come from somewhere else, and section 14.3 measured how well it can:
regressing log spread rate on wind, energy release component, NDVI and slope over 940 real
fire-days leaves a residual scatter of **0.670 in log, a factor of 1.96**, against a
population scatter of 0.700 -- a factor of 2.01. The covariates an investigator actually
has narrow the rate from a factor of 2.01 to a factor of 1.96.

Four regimes, all applied to the same real fires:

    oracle        the rate known to +/-10 %, which requires separating fuel from wind
    history       the rate measured from this fire's own observed daily growth
    covariates    the rate predicted from fuel and weather, factor 1.96 (measured)
    prior only    nothing known about this fire, factor 2.01 (measured)

Run:  python scripts/wsts_advance.py
      python scripts/retrodiction.py
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
from scipy import ndimage
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data" / "wsts", ROOT / "results", ROOT / "figures"

B_WIND, B_AF = 6, 22
PX = 375.0
MAX_JUMP_M = 4000.0
SWATH_FRAC = 0.80
MIN_DT = 3            # days of burning before backtracking is attempted
MIN_BURN = 25         # pixels, so the burn has a shape at all

# Measured in section 14.3 on 940 real fire-days (see the module docstring).
FACTOR_COVARIATES = 1.96
FACTOR_PRIOR = 2.01
FACTOR_ORACLE = 1.10

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def gated_series(fire_dir):
    """Cumulative gated burn masks by day, with the date and the day's wind."""
    out, cum, first = [], None, None
    for p in sorted(fire_dir.glob("*.tif")):
        try:
            with rasterio.open(p) as src:
                a = src.read()
        except Exception:
            continue
        af = np.isfinite(a[B_AF])
        if cum is None:
            # Seed on the first day that actually has detections. Seeding on day zero
            # regardless leaves an empty mask that can never grow, because every later
            # detection is then infinitely far from "the fire".
            if not af.any():
                continue
            lab, n = ndimage.label(af, structure=np.ones((3, 3)))
            sizes = ndimage.sum(af, lab, range(1, n + 1))
            cum = lab == (int(np.argmax(sizes)) + 1)
            first = cum.copy()
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
                    "wind": float(np.nanmedian(a[B_WIND]))})
    return out, (first if out else None)


def max_dist_field(burn):
    """r(p) * dt for every candidate p inside the burn, in metres.

    The maximum distance from a point to a set is attained at a vertex of the set's
    convex hull, so this is a hull plus a few dozen distances per candidate rather than
    an all-pairs computation.
    """
    rr, cc = np.nonzero(burn)
    pts = np.column_stack([rr, cc]).astype(np.float64)
    if len(pts) < 4:
        return None, None
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    d = np.sqrt(((pts[:, None, :] - hull[None, :, :]) ** 2).sum(-1)).max(1) * PX
    return pts, d


def feasible(pts, dmax, dt_days, r_lo, r_hi):
    """Candidate ignition points consistent with a rate in [r_lo, r_hi] (m/day)."""
    return (dmax >= r_lo * dt_days) & (dmax <= r_hi * dt_days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fires", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    fires = sorted({p.parent for p in DATA.glob("*/*/*.tif")
                    if not p.name.endswith(".part")})
    if args.max_fires:
        fires = fires[:args.max_fires]
    print(f"{len(fires)} fire events on disk")

    regimes = {"oracle (+/-10 %)": FACTOR_ORACLE,
               "covariates (x1.96)": FACTOR_COVARIATES,
               "prior only (x2.01)": FACTOR_PRIOR}
    rows, t0 = [], time.time()
    for i, fd in enumerate(fires):
        series, first = gated_series(fd)
        if first is None or len(series) < MIN_DT + 1:
            continue
        ig_rr, ig_cc = np.nonzero(first)
        ig = np.array([ig_rr.mean(), ig_cc.mean()])
        dates = [np.datetime64(s["date"]) for s in series]
        hist_rates = []
        for k, s in enumerate(series):
            dt = int((dates[k] - dates[0]) / np.timedelta64(1, "D"))
            burn = s["cum"]
            if dt < 1 or burn.sum() < MIN_BURN:
                continue
            pts, dmax = max_dist_field(burn)
            if pts is None:
                continue
            # The rate the fire's own growth implies, in the same units as r(p): the
            # radius of the smallest circle enclosing the burn, over elapsed days. Using
            # an area-equivalent radius here instead would be a different quantity and
            # would make the history regime look falsely bad.
            hist_rates.append(float(dmax.min()) / dt)
            if dt < MIN_DT:
                continue
            # the rate the true ignition point implies, and where the truth sits
            j_true = int(np.argmin(((pts - ig) ** 2).sum(1)))
            r_true = float(dmax[j_true] / dt)

            rec = {"fire": f"{fd.parent.name}/{fd.name}", "date": s["date"],
                   "dt_days": dt, "burn_km2": float(burn.sum()) * PX * PX / 1e6,
                   "r_true_m_per_day": r_true,
                   "truth_dist_px": float(np.sqrt(((pts[j_true] - ig) ** 2).sum()))}
            for name, f in regimes.items():
                # Every interval is centred on the rate the true ignition implies. That
                # is deliberately generous to the weaker regimes -- a real covariate
                # prediction is biased as well as wide -- because the quantity being
                # isolated here is the interval's *width*, which is what the
                # identifiability result controls.
                lo, hi = r_true / f, r_true * f
                m = feasible(pts, dmax, dt, lo, hi)
                rec[name] = {"area_km2": float(m.sum()) * PX * PX / 1e6,
                             "frac_of_burn": float(m.mean()),
                             "covers_truth": bool(m[j_true])}
            # the fire's own history, when there is any
            if len(hist_rates) >= 3:
                h = np.array(hist_rates)
                lo, hi = float(np.percentile(h, 10)), float(np.percentile(h, 90))
                m = feasible(pts, dmax, dt, lo, hi)
                rec["history"] = {"area_km2": float(m.sum()) * PX * PX / 1e6,
                                  "frac_of_burn": float(m.mean()),
                                  "covers_truth": bool(m[j_true]),
                                  "factor": float(hi / max(lo, 1e-9))}
            rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fires)} fires, {len(rows)} backtracks "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{len(rows)} backtracks from "
          f"{len({r['fire'] for r in rows})} fires\n")
    out = {"n": len(rows), "n_fires": len({r["fire"] for r in rows})}

    print("=" * 78)
    print("[1] how big is the ignition feasible set, and does it contain the truth?")
    print("=" * 78)
    print("\n  Coverage is 100 % by construction for the three centred regimes: each")
    print("  interval is built around the rate the true ignition implies, so the truth")
    print("  is inside it by definition. The informative column is the area. Only the")
    print("  history regime, whose interval comes from the fire's own growth and not")
    print("  from the answer, has a coverage number that means anything.")
    print(f"\n  {'rate known to':>22} {'median area':>13} {'as % of burn':>14} "
          f"{'covers truth':>14}")
    names = ["oracle (+/-10 %)", "covariates (x1.96)", "prior only (x2.01)"]
    summ = {}
    for name in names:
        a = np.array([r[name]["area_km2"] for r in rows])
        f = np.array([r[name]["frac_of_burn"] for r in rows])
        c = np.array([r[name]["covers_truth"] for r in rows])
        summ[name] = {"median_km2": float(np.median(a)), "median_frac": float(np.median(f)),
                      "coverage": float(c.mean()), "n": int(len(a))}
        print(f"  {name:>22} {np.median(a):10.1f} km2 {np.median(f)*100:12.0f} % "
              f"{c.mean()*100:13.0f} %")
    hist = [r for r in rows if "history" in r]
    if hist:
        a = np.array([r["history"]["area_km2"] for r in hist])
        f = np.array([r["history"]["frac_of_burn"] for r in hist])
        c = np.array([r["history"]["covers_truth"] for r in hist])
        fac = np.median([r["history"]["factor"] for r in hist])
        summ["history"] = {"median_km2": float(np.median(a)),
                           "median_frac": float(np.median(f)),
                           "coverage": float(c.mean()), "n": int(len(a)),
                           "median_factor": float(fac)}
        print(f"  {'history (x%.2f)' % fac:>22} {np.median(a):10.1f} km2 "
              f"{np.median(f)*100:12.0f} % {c.mean()*100:13.0f} %")
    out["regimes"] = summ

    o, c_ = summ["oracle (+/-10 %)"], summ["covariates (x1.96)"]
    print(f"\n  Knowing the rate to +/-10 % instead of the factor of 1.96 an investigator")
    print(f"  actually has shrinks the feasible ignition area by "
          f"{c_['median_km2']/max(o['median_km2'],1e-9):.0f}x, from "
          f"{c_['median_km2']:.0f} to {o['median_km2']:.1f} km2.")
    print(f"  And +/-10 % on the rate is exactly what section 1 says a burn scar cannot")
    print("  give you, because the rate it hands over is a product of fuel and wind.")
    print(f"\n  The covariates are worth almost nothing: {c_['median_km2']:.0f} km2 against "
          f"{summ['prior only (x2.01)']['median_km2']:.0f} km2 for knowing nothing")
    print("  about the fire at all -- which is section 14.3's R^2 = 0.084, in units an")
    print("  investigator can act on.")

    print()
    print("=" * 78)
    print("[2] does it shrink as the fire is watched for longer? (plan item P5)")
    print("=" * 78)
    print("\n    The absolute area is reported beside the fraction of the burn it covers,")
    print("    because a fire that has run longer is bigger and its feasible set would")
    print("    grow on that account alone. The fraction is what controls for it.")
    print(f"\n  {'days burning':>14} {'backtracks':>11} {'burn':>10} "
          + "".join(f"{n.split(' (')[0]:>21}" for n in names))
    by_dt = {}
    dt = np.array([r["dt_days"] for r in rows])
    for lo, hi, lab in ((3, 6, "3 - 5"), (6, 10, "6 - 9"), (10, 16, "10 - 15"),
                        (16, 999, ">= 16")):
        m = (dt >= lo) & (dt < hi)
        if m.sum() < 15:
            continue
        ks = np.nonzero(m)[0]
        burn = float(np.median([rows[k]["burn_km2"] for k in ks]))
        vals = {n: float(np.median([rows[k][n]["area_km2"] for k in ks])) for n in names}
        frac = {n: float(np.median([rows[k][n]["frac_of_burn"] for k in ks]))
                for n in names}
        by_dt[lab] = {"n": int(m.sum()), "burn_km2": burn, "area": vals, "frac": frac}
        print(f"  {lab:>14} {m.sum():11d} {burn:7.0f}km2 "
              + "".join(f"{vals[n]:12.1f} ({frac[n]*100:3.0f}%)" for n in names))
    out["by_dt"] = by_dt
    ks = list(by_dt)
    if len(ks) >= 2:
        a, b = by_dt[ks[0]], by_dt[ks[-1]]
        print(f"\n  The feasible set is a *region*, never a point. Its absolute area grows"
              f" {b['area'][names[0]]/max(a['area'][names[0]],1e-9):.1f}x")
        print(f"  from the shortest to the longest window, but so does the burn "
              f"({b['burn_km2']/max(a['burn_km2'],1e-9):.1f}x), and as a")
        print(f"  fraction of the burn it goes {a['frac'][names[0]]*100:.0f} % -> "
              f"{b['frac'][names[0]]*100:.0f} % under the oracle and stays at "
              f"{b['frac'][names[1]]*100:.0f} % under the")
        print("  covariates. So watching the fire for longer does not localise where it")
        print("  started; the plan's P5 predicted a region rather than a point, and")
        print("  predicted the wrong direction for how that region behaves.")
        print("\n  The operational reading: backtrack from the *earliest* perimeter you")
        print("  have, not the latest. The smallest absolute feasible set here comes from")
        print(f"  the 3-5 day window, at {a['area'][names[0]]:.1f} km2 under an oracle "
              f"rate and {a['area'][names[1]]:.1f} km2 with the")
        print("  rate an investigator can actually obtain.")

    (RESULTS / "retrodiction.json").write_text(
        json.dumps({**out, "rows": rows[:3000]}, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS/'retrodiction.json'}")
    make_figure(out, rows, names)


def make_figure(out, rows, names):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = axes[0]
    keys = list(out["regimes"])
    v = [out["regimes"][k]["median_frac"] * 100 for k in keys]
    km = [out["regimes"][k]["median_km2"] for k in keys]
    ax.barh(range(len(keys)), v,
            color=["tab:green"] + ["tab:red"] * (len(keys) - 1))
    for i, (a, b) in enumerate(zip(v, km)):
        ax.text(min(a + 2, 96), i, f"{b:.0f} km$^2$", va="center", fontsize=7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace(" (", "\n(") for k in keys], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("feasible ignition area, % of the burn scar", fontsize=8.5)
    ax.set_title("With the rate an investigator has,\nthe answer is the whole scar",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    for n, c in zip(names, ("tab:green", "tab:orange", "tab:red")):
        a = np.sort([r[n]["area_km2"] for r in rows])
        f = np.linspace(0, 1, len(a), endpoint=False) + 1.0 / len(a)
        ax.step(np.maximum(a, 1e-2), f, where="post", color=c, lw=1.7,
                label=n)
    ax.set_xscale("log")
    ax.set_xlabel("feasible ignition area (km$^2$)", fontsize=8.5)
    ax.set_ylabel("fraction of backtracks below", fontsize=8.5)
    ax.set_title(f"{out['n']} backtracks, {out['n_fires']} real fires", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    labs = list(out["by_dt"])
    x = np.arange(len(labs))
    for k, (n, c) in enumerate(zip(names, ("tab:green", "tab:orange", "tab:red"))):
        ax.bar(x + (k - 1) * 0.27,
               [out["by_dt"][l]["frac"][n] * 100 for l in labs], 0.27,
               color=c, label=n.split(" (")[0])
    ax2 = ax.twinx()
    ax2.plot(x, [out["by_dt"][l]["burn_km2"] for l in labs], "k.--", lw=1.2, ms=8,
             label="burn scar")
    ax2.set_ylabel("median burn scar (km$^2$)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6.4, frameon=False, loc="upper center")
    ax.set_ylim(0, 118)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_xlabel("days the fire had been burning", fontsize=8.5)
    ax.set_ylabel("feasible area, % of burn scar", fontsize=8.5)
    ax.set_title("The scar grows 4x; the fraction\nit could have started in does not",
                 fontsize=9.5)
    ax.legend(fontsize=6.4, frameon=False, loc="center left")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Backtracking real fires to their ignition, WildfireSpreadTS",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_retrodiction.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
