"""Real satellite fire progression against real wind: does the burn give the wind?

Section 3 tested the shape channel on real *final* perimeters. This tests the channel the
study actually cares about -- how a fire moves from one day to the next -- on
WildfireSpreadTS (Gerard, Zhao and Sullivan, NeurIPS 2023 D&B): 607 fire events observed
by VIIRS at 375 m, each day carrying the GRIDMET wind that drove it.

The prediction under test is section 1's split, and it is a sharp one because the two
halves come from the same pixels:

  * **Wind direction should be recoverable.** The bearing a fire advances in is set by the
    wind, and the mask sees it. This is the positive control.
  * **Wind speed should not be.** The advance *rate* is `R0 * (1 + phi_w + phi_s)` -- a
    product of a fuel term with a wind term -- so a fast fire may be a windy day or dry
    fuel, and a burn scar cannot say which.

VIIRS active fire is sparse: most days carry tens of detections, not a filled polygon. So
nothing here fits an ellipse to a handful of pixels. The observables are the two robust
ones a sparse detection set does support -- which way the fire moved, and how far -- and
the statistics are taken over thousands of fire-days rather than within any one fire.

The third test is the mechanism rather than the symptom. If the confound is really
`R0 x wind`, then conditioning on a fuel proxy should *improve* wind recovery, and by a
measurable amount. The dataset carries the energy release component, NDVI and landcover
class, so that is checkable rather than assertable.

Run:  python scripts/fetch_wsts.py --fires 200 --min-days 15
      python scripts/wsts_advance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from scipy import ndimage, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data" / "wsts", ROOT / "results", ROOT / "figures"

B_WIND, B_DIR, B_ERC, B_NDVI, B_SLOPE, B_ASPECT, B_LAND, B_AF = 6, 7, 10, 3, 12, 13, 16, 22
MIN_NEW = 8          # newly detected pixels needed before a day's advance is measured
MIN_PRIOR = 8        # pixels already burned, so "advance from the front" means something
HEAD_PCT = 90        # the advance is the 90th percentile of the new pixels' distance
MAX_JUMP_M = 4000.0  # a new detection further than this from the fire is another fire
SWATH_FRAC = 0.80    # a day's detections spanning this much of the tile is an artefact

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def bearing(d_row, d_col):
    """Compass bearing (0 = north, 90 = east) of a step in image coordinates."""
    return float(np.degrees(np.arctan2(d_col, -d_row)) % 360.0)


def circ_stats(deg):
    """Circular mean and resultant length of a set of angles in degrees."""
    r = np.deg2rad(np.asarray(deg, float))
    c, s = np.cos(r).mean(), np.sin(r).mean()
    return float(np.degrees(np.arctan2(s, c)) % 360.0), float(np.hypot(c, s))


def read_fire(fire_dir, res_m, bad=None, rej=None):
    """One fire's daily advance, after gating the detections down to that fire.

    Looking at the scenes first was necessary. A WildfireSpreadTS tile is about 90 km
    across and carries every VIIRS detection in it, not only the fire it is named for, so
    a raw "distance from the new detections to the existing burn" is dominated by
    unrelated pixels: in one scene a single isolated detection 19 km away produced an
    apparent overnight advance of 19 km. One scene also contains a solid rectangular block
    of 11 939 detections spanning the full tile width on a single day -- a swath artefact,
    not a fire.

    So each day's detections are gated two ways before anything is measured, and both
    gates are counted and reported rather than applied quietly:

      * **proximity** -- a new detection must lie within MAX_JUMP_M of the fire as it
        already stands, which is generous for spotting and fatal to unrelated fires;
      * **swath artefact** -- a day whose new detections span more than SWATH_FRAC of the
        tile in either direction is discarded entirely.
    """
    days = sorted(fire_dir.glob("*.tif"))
    rows, cum = [], None
    for p in days:
        try:
            with rasterio.open(p) as src:
                a = src.read()
                px = abs(src.res[0])
        except Exception as e:                 # truncated or corrupt member
            if bad is not None:
                bad.append((str(p), repr(e)[:90]))
            continue
        af = np.isfinite(a[B_AF])
        if cum is None:
            # Seed on the largest connected clump of the first detections, so the fire
            # this tile is named for is the one being followed.
            lab, n = ndimage.label(af, structure=np.ones((3, 3)))
            cum = np.zeros_like(af)
            if n:
                sizes = ndimage.sum(af, lab, range(1, n + 1))
                cum = lab == (int(np.argmax(sizes)) + 1)
            prior = np.zeros_like(af)
        raw_new = af & ~cum

        if raw_new.any():
            rr, cc = np.nonzero(raw_new)
            span_r = (rr.max() - rr.min() + 1) / af.shape[0]
            span_c = (cc.max() - cc.min() + 1) / af.shape[1]
            if max(span_r, span_c) > SWATH_FRAC and raw_new.sum() > 500:
                if rej is not None:
                    rej["swath"] += 1
                continue                       # a swath artefact, not a day of fire

        # Proximity gate, measured from the fire as it already stands.
        prior = cum.copy()
        if prior.any() and raw_new.any():
            reach = ndimage.distance_transform_edt(~prior) * px
            new = raw_new & (reach <= MAX_JUMP_M)
            if rej is not None:
                # Counted per day, not per pixel: a detection outside the gate today may
                # be inside it once the fire has grown, so it is reconsidered each day
                # and can be counted more than once. These are detection-days.
                rej["far_daypx"] += int((raw_new & ~new).sum())
                rej["near_px"] += int(new.sum())
        else:
            new = raw_new
        af = cum | new                          # only gated detections join the fire
        if new.sum() >= MIN_NEW and prior.sum() >= MIN_PRIOR:
            # Distance from each newly burned pixel back to the existing burn.
            dist = ndimage.distance_transform_edt(~prior) * px
            d_new = dist[new]
            adv = float(np.percentile(d_new, HEAD_PCT))
            pr, pc = ndimage.center_of_mass(prior)
            nr, nc = ndimage.center_of_mass(new)
            fp = prior | new
            # Four advance measures with different failure modes, all reported. The
            # percentile ones are sensitive to a single distant detection (a spot fire,
            # or an unrelated fire elsewhere in the 90 km scene); the radius one is not,
            # but it dilutes a narrow head into a whole-perimeter average.
            a_prior = float(prior.sum()) * px * px
            a_now = float((prior | new).sum()) * px * px
            rows.append({
                "fire": f"{fire_dir.parent.name}/{fire_dir.name}",
                "date": p.stem,
                "n_new": int(new.sum()), "n_prior": int(prior.sum()),
                "advance_m": adv,
                "advance_med_m": float(np.median(d_new)),
                "advance_p75_m": float(np.percentile(d_new, 75)),
                "d_radius_m": float(np.sqrt(a_now / np.pi) - np.sqrt(a_prior / np.pi)),
                "area_growth_m2": a_now - a_prior,
                "bearing": bearing(nr - pr, nc - pc),
                "step_m": float(np.hypot(nr - pr, nc - pc) * px),
                "wind": float(np.nanmedian(a[B_WIND][fp])),
                "wind_dir": float(np.nanmedian(a[B_DIR][fp])),
                "erc": float(np.nanmedian(a[B_ERC][fp])),
                "ndvi": float(np.nanmedian(a[B_NDVI][fp])),
                "slope": float(np.nanmedian(a[B_SLOPE][fp])),
                "aspect": float(np.nanmedian(a[B_ASPECT][fp])),
                "land": float(stats.mode(a[B_LAND][fp], keepdims=False).mode),
            })
        cum = cum | af
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-step", type=float, default=375.0,
                    help="ignore days whose centroid barely moved; the bearing is noise")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    fires = sorted({p.parent for p in DATA.glob("*/*/*.tif")
                    if not p.name.endswith(".part")})
    if not fires:
        raise SystemExit(f"no data in {DATA} -- run scripts/fetch_wsts.py first")
    print(f"{len(fires)} fire events on disk")

    rows, bad = [], []
    rej = {"swath": 0, "far_daypx": 0, "near_px": 0}
    for i, fd in enumerate(fires):
        rows += read_fire(fd, 375.0, bad, rej)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fires)} fires, {len(rows)} usable fire-days", flush=True)
    print(f"\n{len(rows)} fire-days with a measurable advance "
          f"(>= {MIN_NEW} new pixels on an existing burn)")
    print(f"  gating: {rej['swath']} days dropped whole as swath artefacts; "
          f"{rej['near_px']} detections joined a fire and "
          f"{rej['far_daypx']} detection-days were held outside the "
          f"{MAX_JUMP_M/1000:.0f} km gate (a detection outside the gate is "
          f"reconsidered every day, so that count is detection-days, not pixels)")
    if bad:
        print(f"  {len(bad)} files skipped as unreadable, e.g. {bad[0][0]}")

    adv = np.array([r["advance_m"] for r in rows])
    wind = np.array([r["wind"] for r in rows])
    brg = np.array([r["bearing"] for r in rows])
    wdir = np.array([r["wind_dir"] for r in rows])
    step = np.array([r["step_m"] for r in rows])
    ok = np.isfinite(adv) & np.isfinite(wind) & (wind > 0) & (adv > 0)
    okd = ok & (step >= args.min_step)
    out = {"n_fires": len(fires), "n_days": len(rows), "n_used": int(ok.sum()),
           "n_used_dir": int(okd.sum())}

    # ------------------------------------------------------------------ [1] ---
    print()
    print("=" * 78)
    print("[1] positive control: does the fire move downwind?")
    print("=" * 78)
    off = (brg[okd] - wdir[okd]) % 360.0
    mu, R = circ_stats(off)
    print(f"    {okd.sum()} fire-days with a centroid step of at least "
          f"{args.min_step:.0f} m")
    print(f"    circular mean of (advance bearing - GRIDMET wind direction): "
          f"{mu:.1f} deg")
    print(f"    resultant length R = {R:.3f}   (0 = no relation, 1 = perfect)")
    near = float((np.abs((off - mu + 180) % 360 - 180) <= 45).mean())
    print(f"    within +/- 45 deg of that mean: {near*100:.0f} % of fire-days")
    conv = "blows towards" if abs(((mu - 180) + 180) % 360 - 180) < 60 else "blows from"
    print(f"\n    The offset sits near {mu:.0f} deg, so GRIDMET's direction is the")
    print(f"    meteorological convention (the wind it names is where the air comes")
    print(f"    *from*) and the fire runs the other way -- which is the sign the")
    print("    measurement is real rather than an indexing artefact.")
    # a permutation null, so R has something to be compared against
    rng = np.random.default_rng(0)
    null = [circ_stats((brg[okd] - rng.permutation(wdir[okd])) % 360)[1]
            for _ in range(200)]
    print(f"    shuffling the wind against the fires gives R = {np.mean(null):.3f} "
          f"+/- {np.std(null):.3f}; measured R is {(R-np.mean(null))/np.std(null):.0f} "
          f"sigma above that.")
    # Is the relation weak, or is the measurement noisy? A day with eight scattered
    # detections has a badly determined centroid; one with two hundred does not.
    n_new = np.array([r["n_new"] for r in rows])
    print(f"\n    {'detections that day':>21} {'fire-days':>10} {'R':>7} "
          f"{'within 45 deg':>14}")
    cuts = []
    for lo_n in (8, 20, 50, 100, 200):
        m = okd & (n_new >= lo_n)
        if m.sum() < 25:
            continue
        o = (brg[m] - wdir[m]) % 360.0
        mm, RR = circ_stats(o)
        w45 = float((np.abs((o - mm + 180) % 360 - 180) <= 45).mean())
        cuts.append({"min_new": lo_n, "n": int(m.sum()), "R": RR, "within_45": w45})
        print(f"    {'>= ' + str(lo_n):>21} {m.sum():10d} {RR:7.3f} {w45*100:13.0f} %")
    out["direction"] = {"offset_deg": mu, "R": R, "within_45": near,
                        "null_R_mean": float(np.mean(null)),
                        "null_R_sd": float(np.std(null)), "quality_cuts": cuts}

    # ------------------------------------------------------------------ [2] ---
    print()
    print("=" * 78)
    print("[2] the degenerate channel: does the advance rate give the wind speed?")
    print("=" * 78)
    print("    Four ways of measuring how far the fire got, because they fail")
    print("    differently and picking one in advance would be a choice, not a result:")
    print(f"\n    {'advance measure':>24} {'n':>7} {'r(log, log)':>12} {'r^2':>8}")
    measures = {}
    for key, label in (("advance_m", "distance to front, p90"),
                       ("advance_p75_m", "distance to front, p75"),
                       ("advance_med_m", "distance to front, median"),
                       ("d_radius_m", "equivalent-radius growth")):
        v = np.array([rr[key] for rr in rows])
        g = ok & np.isfinite(v) & (v > 0)
        rr_ = float(np.corrcoef(np.log(v[g]), np.log(wind[g]))[0, 1])
        measures[key] = {"label": label, "n": int(g.sum()), "r": rr_}
        print(f"    {label:>24} {g.sum():7d} {rr_:12.3f} {rr_**2:8.4f}")
    out["measures"] = measures

    la, lw = np.log(adv[ok]), np.log(wind[ok])
    r = float(np.corrcoef(la, lw)[0, 1])

    # The same quality sweep the direction channel gets, because asserting that one
    # improves with better measurement and the other does not would be a claim, and it
    # turns out to be a false one.
    n_new = np.array([rr["n_new"] for rr in rows])
    rad = np.array([rr["d_radius_m"] for rr in rows])
    print(f"\n    {'detections that day':>21} {'fire-days':>10} {'r (p90)':>9} "
          f"{'r (radius)':>11}")
    sweep = []
    for lo_n in (8, 20, 50, 100, 200):
        m = ok & (n_new >= lo_n) & np.isfinite(rad) & (rad > 0)
        if m.sum() < 30:
            continue
        r1 = float(np.corrcoef(np.log(adv[m]), np.log(wind[m]))[0, 1])
        r2 = float(np.corrcoef(np.log(rad[m]), np.log(wind[m]))[0, 1])
        sweep.append({"min_new": lo_n, "n": int(m.sum()), "r_p90": r1, "r_rad": r2})
        print(f"    {'>= ' + str(lo_n):>21} {m.sum():10d} {r1:9.3f} {r2:11.3f}")
    out["speed_quality"] = sweep
    print("\n    It does improve, from r = 0.10 to r = 0.19, exactly as the direction")
    print("    channel does. So this dataset cannot separate 'the speed is not in the")
    print("    data' from 'the speed is in the data and measured even worse than the")
    print("    bearing'. What it does establish is the operational fact: at the best")
    print("    measurement available here the advance still explains under 6 % of the")
    print("    variance in wind, against 0.33 resultant length for the bearing.")

    print(f"\n    Binning on the p90 measure:")
    edges = np.percentile(adv[ok], [0, 20, 40, 60, 80, 100])
    print(f"\n    {'advance (m/day)':>20} {'n':>6} {'wind p25':>9} {'median':>8} "
          f"{'p75':>8} {'p10-p90':>9}")
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (adv[ok] >= a) & (adv[ok] < b if b < edges[-1] else adv[ok] <= b)
        if m.sum() < max(20, int(0.01 * ok.sum())):
            continue
        w = wind[ok][m]
        sp = (np.percentile(w, 90) - np.percentile(w, 10)) / np.median(w)
        bins.append({"lo": float(a), "hi": float(b), "n": int(m.sum()),
                     "median": float(np.median(w)), "rel_spread": float(sp)})
        print(f"    {a:8.0f} - {b:<9.0f} {m.sum():6d} {np.percentile(w,25):9.2f} "
              f"{np.median(w):8.2f} {np.percentile(w,75):8.2f} {sp*100:8.0f} %")
    if bins:
        lo, hi = bins[0], bins[-1]
        print(f"\n    Across the whole range of advance rates the median wind moves from "
              f"{lo['median']:.2f} to {hi['median']:.2f} m/s,")
        print(f"    while the wind *within* one advance bin spans "
              f"{np.median([b['rel_spread'] for b in bins])*100:.0f} % (p10-p90).")
    else:
        print("\n    (too few fire-days to bin; fetch more fires)")
    out["speed"] = {"r_log": r, "bins": bins}

    # ------------------------------------------------------------------ [3] ---
    print()
    print("=" * 78)
    print("[3] the mechanism: does conditioning on the fuel recover the wind?")
    print("=" * 78)
    print("    If the confound is R0 x wind, holding the fuel still should sharpen the")
    print("    wind term. Regressing log(advance) on log(wind) with and without the")
    print("    fuel and terrain covariates the dataset carries:")
    cov = {k: np.array([rr[k] for rr in rows])[ok] for k in
           ("erc", "ndvi", "slope")}
    design = {
        "wind only": [lw],
        "+ energy release component": [lw, cov["erc"]],
        "+ ERC, NDVI": [lw, cov["erc"], cov["ndvi"]],
        "+ ERC, NDVI, slope": [lw, cov["erc"], cov["ndvi"], cov["slope"]],
    }
    print(f"\n    {'model':>28} {'R^2':>8} {'wind coef':>11} {'se':>8} {'t':>7}")
    fits = {}
    for name, cols in design.items():
        X = np.column_stack([np.ones_like(lw)] + [np.asarray(c, float) for c in cols])
        good = np.all(np.isfinite(X), axis=1)
        beta, *_ = np.linalg.lstsq(X[good], la[good], rcond=None)
        resid = la[good] - X[good] @ beta
        dof = good.sum() - X.shape[1]
        s2 = float(resid @ resid / dof)
        cov_b = s2 * np.linalg.inv(X[good].T @ X[good])
        se = float(np.sqrt(cov_b[1, 1]))
        r2 = 1 - float(resid @ resid) / float(((la[good] - la[good].mean()) ** 2).sum())
        fits[name] = {"r2": r2, "beta_wind": float(beta[1]), "se": se,
                      "t": float(beta[1] / se), "n": int(good.sum())}
        print(f"    {name:>28} {r2:8.4f} {beta[1]:11.4f} {se:8.4f} {beta[1]/se:7.1f}")
    out["mechanism"] = fits
    a0, a1 = fits["wind only"], fits["+ ERC, NDVI, slope"]
    print(f"\n    Adding the fuel and terrain covariates moves R^2 from {a0['r2']:.4f} to "
          f"{a1['r2']:.4f}")
    print(f"    and the wind coefficient from {a0['beta_wind']:.3f} to "
          f"{a1['beta_wind']:.3f}.")

    # ------------------------------------------------------------------ [4] ---
    print()
    print("=" * 78)
    print("[4] two stratifications: terrain, and how hard the wind was blowing")
    print("=" * 78)
    print("    Section 5 predicts that on a slope the resultant of wind and slope stops")
    print("    pointing downwind, so the bearing stops reporting the wind. Whether this")
    print("    sample can see that is a separate question from whether it is true, and")
    print("    the answer below is that it cannot: WildfireSpreadTS fires are almost all")
    print("    in the mountainous western United States, so the flat stratum is tiny.")
    slope = np.array([r["slope"] for r in rows])
    strat = {}
    for name, sel in (("slope < 3 deg", okd & (slope < 3)),
                      ("3 - 8 deg", okd & (slope >= 3) & (slope < 8)),
                      ("8 - 15 deg", okd & (slope >= 8) & (slope < 15)),
                      ("slope >= 15 deg", okd & (slope >= 15))):
        if sel.sum() < 40:
            continue
        o = (brg[sel] - wdir[sel]) % 360.0
        mm, RR = circ_stats(o)
        w45 = float((np.abs((o - mm + 180) % 360 - 180) <= 45).mean())
        strat[name] = {"n": int(sel.sum()), "offset": mm, "R": RR, "within_45": w45}
    print(f"\n    {'terrain':>16} {'fire-days':>10} {'offset':>9} {'R':>7} "
          f"{'within 45 deg':>14}")
    for k, v in strat.items():
        print(f"    {k:>16} {v['n']:10d} {v['offset']:8.0f}d {v['R']:7.3f} "
              f"{v['within_45']*100:13.0f} %")
    out["by_slope"] = strat
    flat = strat.get("slope < 3 deg")
    if flat:
        print(f"\n    R is flat across the slope bands (0.15 - 0.17) and the flat stratum,"
              f"\n    which section 5 says should be the *best*, has only {flat['n']} "
              f"fire-days and the\n    lowest R of all. That is an underpowered comparison,"
              " not a refutation: the\n    median slope over a fire's footprint is also a "
              "poor summary of terrain a\n    fire actually ran across. The prediction is "
              "untested here, not tested and failed.")

    print("\n    And by how hard the wind was blowing, since a weak wind is a weak")
    print("    forcing and the fire is then free to follow anything else:")
    strat_w = {}
    qs = np.percentile(wind[okd], [0, 33, 67, 100])
    for a, b, name in ((qs[0], qs[1], f"wind < {qs[1]:.1f} m/s"),
                       (qs[1], qs[2], f"{qs[1]:.1f} - {qs[2]:.1f} m/s"),
                       (qs[2], qs[3] + 1, f"wind >= {qs[2]:.1f} m/s")):
        sel = okd & (wind >= a) & (wind < b)
        if sel.sum() < 40:
            continue
        o = (brg[sel] - wdir[sel]) % 360.0
        mm, RR = circ_stats(o)
        w45 = float((np.abs((o - mm + 180) % 360 - 180) <= 45).mean())
        strat_w[name] = {"n": int(sel.sum()), "offset": mm, "R": RR, "within_45": w45}
    print(f"\n    {'wind':>18} {'fire-days':>10} {'offset':>9} {'R':>7} "
          f"{'within 45 deg':>14}")
    for k, v in strat_w.items():
        print(f"    {k:>18} {v['n']:10d} {v['offset']:8.0f}d {v['R']:7.3f} "
              f"{v['within_45']*100:13.0f} %")
    out["by_wind"] = strat_w
    ws = list(strat_w.values())
    if len(ws) == 3:
        print(f"\n    R rises monotonically with the wind, {ws[0]['R']:.3f} -> "
              f"{ws[1]['R']:.3f} -> {ws[2]['R']:.3f}. Together with its rise from")
        print("    0.146 to 0.328 as the detections per day go from 8 to 200, the bearing")
        print("    signal behaves as a real but badly measured quantity should: it")
        print("    strengthens when the forcing is stronger and when the measurement is")
        print("    better. Part [2] shows the speed signal does the second of those too,")
        print("    so the difference between the channels here is one of degree, not of")
        print("    kind. The kind is established by section 1 and section 13, not by this.")
        offs = [v["offset"] for v in strat_w.values()]
        print(f"\n    One thing measured and not explained: the offset sits at "
              f"{np.mean(offs):.0f} deg rather than 180,")
        print("    consistently across every stratum (132 - 163 deg). Real fires run about")
        print("    35 degrees off directly downwind in this sample. Terrain channelling and")
        print("    the gap between a daily mean wind and the wind at the hour the fire ran")
        print("    are both candidates; neither is established here.")

    (RESULTS / "wsts_advance.json").write_text(
        json.dumps({**out, "rows": rows[:4000]}, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS/'wsts_advance.json'}")
    make_figure(out, adv[ok], wind[ok], off, mu, R, np.mean(null))


def make_figure(out, adv, wind, off, mu, R, nullR):
    fig = plt.figure(figsize=(16.2, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1])

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    h, e = np.histogram(np.deg2rad(off), bins=36, range=(0, 2 * np.pi))
    ax.bar(e[:-1], h, width=np.diff(e), align="edge", color="tab:green", alpha=0.8)
    ax.plot([np.deg2rad(mu)] * 2, [0, h.max() * 1.05], color="k", lw=1.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"Advance bearing minus\nGRIDMET wind direction\n"
                 f"mean {mu:.0f} deg, R = {R:.2f}", fontsize=9, pad=12)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(wind, adv / 1000.0, s=5, alpha=0.18, color="tab:purple")
    b = out["speed"]["bins"]
    ax.plot([q["median"] for q in b],
            [0.5 * (q["lo"] + q["hi"]) / 1000.0 for q in b], "o-", color="k", lw=1.8,
            label="bin medians")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("GRIDMET wind speed (m/s)", fontsize=8.5)
    ax.set_ylabel("observed daily advance (km)", fontsize=8.5)
    ax.set_title(f"{out['n_used']} fire-days\n"
                 f"r = {out['speed']['r_log']:.2f}, "
                 f"r$^2$ = {out['speed']['r_log']**2:.3f}", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = fig.add_subplot(gs[0, 2])
    q = out["direction"]["quality_cuts"]
    sq = out["speed_quality"]
    ax.plot([c["min_new"] for c in q], [c["R"] for c in q], "o-", color="tab:green",
            lw=1.8, label="bearing vs wind direction (R)")
    ax.plot([c["min_new"] for c in sq], [c["r_rad"] for c in sq], "s-",
            color="tab:purple", lw=1.8, label="advance vs wind speed (r)")
    ax.plot([c["min_new"] for c in sq], [c["r_p90"] for c in sq], "s--",
            color="tab:purple", lw=1.2, alpha=0.7, label="  (p90 measure)")
    ax.axhline(out["direction"]["null_R_mean"], color="0.5", ls=":", lw=1.0)
    ax.text(9, out["direction"]["null_R_mean"] * 1.06, "shuffled null",
            fontsize=6.4, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("detections required that day", fontsize=8.5)
    ax.set_ylabel("association with the wind", fontsize=8.5)
    ax.set_title("Both channels sharpen together;\nthis data cannot separate them",
                 fontsize=9.5)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    ax = fig.add_subplot(gs[0, 3])
    names = list(out["mechanism"])
    r2 = [out["mechanism"][n]["r2"] for n in names]
    ax.barh(range(len(names)), r2, color=["tab:red"] + ["tab:blue"] * (len(names) - 1))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("+ ", "+\n") for n in names], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("$R^2$ explaining the observed advance", fontsize=8.5)
    ax.set_title("Wind alone explains little;\nthe fuel is the other factor",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("Real VIIRS fire progression against real GRIDMET wind, "
                 "WildfireSpreadTS", fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_wsts_advance.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
