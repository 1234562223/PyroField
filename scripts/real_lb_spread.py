"""How precisely can a length-to-breadth ratio be read off a real perimeter?

Section 2 of RESULTS.md needs one number and does not have it: the uncertainty on the
shape relation. It was inferred from Alexander's r = 0.865 and carried as a 15-35 %
bracket, flagged as the weakest link in the study.

Real perimeters supply a lower bound on it directly, and without needing wind data.
`scripts/real_perimeters.py` shows that a real burn is not an ellipse -- median IoU 0.70
against the best area-matched ellipse, not one fire in 895 above 0.90. So "the"
length-to-breadth ratio of a real fire is not a well-posed quantity: it depends on how you
choose to measure it. Four defensible estimators are computed for every fire and their
spread is reported.

That spread is a *lower* bound on the uncertainty, not the whole of it: it captures only
the ambiguity in reading L/B off a given perimeter, not the fuel-to-fuel variation in how
L/B relates to wind, nor suppression, nor the perimeter's own mapping error.

Run:  python scripts/real_lb_spread.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon, shape  # noqa: E402
from shapely.ops import transform as shp_transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

RASTER = 420
MIN_VERTICES = 60
# Smoothing radii, as a fraction of the equivalent radius, for the smoothed-outline
# estimator. A mask from a coarse sensor is effectively a smoothed perimeter.
SMOOTH_FRAC = 0.06


def to_local_metres(geom):
    c = geom.centroid
    tr = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m +no_defs",
        always_xy=True)
    return shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)


def rasterise(poly: Polygon, n=RASTER):
    minx, miny, maxx, maxy = poly.bounds
    span = max(maxx - minx, maxy - miny)
    if span <= 0:
        return None, None
    px = span / (n - 4)
    w, h = int((maxx - minx) / px) + 4, int((maxy - miny) / px) + 4
    img = np.zeros((h, w), np.uint8)

    def ring_px(coords):
        a = np.asarray(coords, float)
        return np.stack([(a[:, 0] - minx) / px + 2, (a[:, 1] - miny) / px + 2],
                        axis=1).astype(np.int32)

    cv2.fillPoly(img, [ring_px(poly.exterior.coords)], 1)
    for hole in poly.interiors:
        cv2.fillPoly(img, [ring_px(hole.coords)], 0)
    return img.astype(bool), px


def moment_ratio(mask):
    """Axis ratio of the ellipse with the same second moments."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    dx, dy = xs - xs.mean(), ys - ys.mean()
    cov = np.array([[(dx * dx).mean(), (dx * dy).mean()],
                    [(dx * dy).mean(), (dy * dy).mean()]])
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return math.sqrt(ev[0] / ev[1]) if ev[1] > 0 else None


def estimators(mask):
    """Four defensible ways to put a length-to-breadth ratio on one burned region."""
    out = {}
    out["moment"] = moment_ratio(mask)

    # Convex hull: ignores concavities carved by fuel breaks and suppression.
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        hm = np.zeros_like(mask, np.uint8)
        cv2.fillPoly(hm, [hull], 1)
        out["hull"] = moment_ratio(hm.astype(bool))

        # Minimum-area rotated rectangle: what a bounding-box reading would give.
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) > 0:
            out["min_rect"] = max(w, h) / min(w, h)

        # Direct fit of an ellipse to the outline, rather than to the filled area.
        if len(c) >= 5:
            (_, _), (ma, mi), _ = cv2.fitEllipse(c)
            if min(ma, mi) > 0:
                out["fit_outline"] = max(ma, mi) / min(ma, mi)

    # A coarse sensor sees a smoothed outline; smoothing should not change the answer.
    r = max(1, int(round(SMOOTH_FRAC * math.sqrt(mask.sum() / math.pi))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sm = cv2.morphologyEx(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k),
                          cv2.MORPH_OPEN, k)
    if sm.sum() > 400:
        out["smoothed"] = moment_ratio(sm.astype(bool))

    return {k: v for k, v in out.items() if v is not None and np.isfinite(v)}


def main():
    src = DATA / "wfigs_perimeters.geojson"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/fetch_perimeters.py first")
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    feats = json.loads(src.read_text(encoding="utf-8"))["features"]
    print(f"reading {len(feats)} perimeters\n")

    keys = ["moment", "hull", "min_rect", "fit_outline", "smoothed"]
    rows = []
    for i, f in enumerate(feats):
        try:
            g = to_local_metres(shape(f["geometry"]))
        except Exception:
            continue
        parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
        parts = [p for p in parts if p.area > 0]
        if not parts:
            continue
        poly = max(parts, key=lambda p: p.area)
        if len(poly.exterior.coords) < MIN_VERTICES:
            continue
        mask, px = rasterise(poly)
        if mask is None or mask.sum() < 400:
            continue
        est = estimators(mask)
        if len(est) < 4:
            continue
        vals = np.array([est[k] for k in keys if k in est])
        rows.append({"name": f["properties"].get("poly_IncidentName"),
                     "acres": f["properties"].get("poly_GISAcres"),
                     **{k: float(est[k]) for k in keys if k in est},
                     "spread_rel": float((vals.max() - vals.min()) / vals.mean()),
                     "sd_rel": float(vals.std(ddof=1) / vals.mean())})
        if (i + 1) % 150 == 0:
            print(f"  processed {i+1}/{len(feats)}  (kept {len(rows)})", flush=True)

    print(f"\n  kept {len(rows)}\n")
    sd = np.array([r["sd_rel"] for r in rows])
    sp = np.array([r["spread_rel"] for r in rows])

    print("=" * 78)
    print("[1] the same fire, measured four or five defensible ways")
    print("=" * 78)
    print(f"    {'estimator':>14s} {'median LB':>11s} {'p25':>8s} {'p75':>8s}")
    med = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r])
        if len(v) < 50:
            continue
        med[k] = float(np.median(v))
        print(f"    {k:>14s} {np.median(v):11.2f} {np.percentile(v,25):8.2f} "
              f"{np.percentile(v,75):8.2f}")
    if med:
        lo_k = min(med, key=med.get)
        hi_k = max(med, key=med.get)
        print(f"\n    the estimators disagree in the median by "
              f"{(med[hi_k]/med[lo_k]-1)*100:.0f} % "
              f"({lo_k} = {med[lo_k]:.2f} vs {hi_k} = {med[hi_k]:.2f})")

    print()
    print("=" * 78)
    print("[2] per-fire disagreement: how well-posed is 'the' L/B of a real fire?")
    print("=" * 78)
    for q in (25, 50, 75, 90):
        print(f"    relative standard deviation across estimators, p{q:02d}: "
              f"{np.percentile(sd, q)*100:5.1f} %")
    print(f"    mean {sd.mean()*100:.1f} %")
    print(f"    full range (max-min)/mean, median: {np.median(sp)*100:.1f} %")
    print(f"    fraction of fires where the estimators differ by more than 20 %: "
          f"{(sp > 0.20).mean()*100:.1f} %")

    print()
    print("=" * 78)
    print("[3] what this does to the wind")
    print("=" * 78)
    print("    Section 2 needs an uncertainty on the shape relation. This measures only")
    print("    one part of it -- the ambiguity in reading L/B off a perimeter at all --")
    print("    and that alone is already:")
    s_med = float(np.median(sd))
    for U, E in ((1.5, 0.843), (2.0, 1.159), (2.5, 1.468), (3.0, 1.769), (3.5, 2.062)):
        print(f"      U = {U:.1f} m/s (elasticity {E:.3f}) -> sigma(U) >= "
              f"{s_med/E*100:5.1f} %"
              + ("   <- misses the 10 % tolerance" if s_med / E > 0.10 else ""))
    print("\n    and the fuel-to-fuel variation in the relation itself, the suppression")
    print("    that shaped these perimeters, and the mapping error in them are all on top.")

    # ---------------------------------------------------------------- 4 -----
    # Where the observed L/B distribution actually lands on the elasticity curve.
    # This is the sharper finding: the elasticity vanishes as LB -> 1, so the shape is
    # least informative about wind exactly where real fires are found.
    import torch

    from pyrofield.physics.rothermel import lb_anderson

    def U_for_LB(target, lo=0.05, hi=12.0):
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(lb_anderson(torch.tensor(mid), cap=None)) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def elasticity(U, h=1e-3):
        lb = float(lb_anderson(torch.tensor(U), cap=None))
        d = float((lb_anderson(torch.tensor(U + h), cap=None)
                   - lb_anderson(torch.tensor(U - h), cap=None)) / (2 * h))
        return d * U / lb

    print()
    print("=" * 78)
    print("[4] real fires sit where the shape channel is weakest")
    print("=" * 78)
    mm = np.array([r["moment"] for r in rows])
    print(f"    {'observed LB':>13s} {'implied U':>11s} {'elasticity':>12s} "
          f"{'sigma(U) from a 6.7% LB reading':>32s}")
    lb_map = []
    for q in (10, 25, 50, 75, 90):
        L = float(np.percentile(mm, q))
        U = U_for_LB(L)
        E = elasticity(U)
        s = s_med / E
        lb_map.append({"pct": q, "LB": L, "U": U, "E": E, "sigma_U": s})
        print(f"    p{q:02d}  {L:7.2f} {U:11.2f} {E:12.3f} {s*100:29.0f} %"
              + ("   MISSES" if s > 0.10 else ""))
    med_row = next(r for r in lb_map if r["pct"] == 50)
    print(f"\n    The median real fire has LB = {med_row['LB']:.2f}, which Anderson's")
    print(f"    relation places at {med_row['U']:.2f} m/s midflame with an elasticity of")
    print(f"    only {med_row['E']:.3f}. The elasticity vanishes as LB approaches 1, so the")
    print("    shape carries least about the wind exactly where real fires are found.")
    frac_miss = float(np.mean([s_med / elasticity(U_for_LB(float(v))) > 0.10
                               for v in mm]))
    print(f"    Fraction of real fires where the measured L/B ambiguity alone already")
    print(f"    puts the wind outside a 10 % tolerance: {frac_miss*100:.0f} %")

    out = {"n": len(rows), "estimators": keys, "median_by_estimator": med,
           "sd_rel": {f"p{q}": float(np.percentile(sd, q)) for q in (25, 50, 75, 90)},
           "sd_rel_mean": float(sd.mean()),
           "spread_rel_median": float(np.median(sp)),
           "frac_spread_above_20pct": float((sp > 0.20).mean()),
           "lb_to_wind": lb_map, "frac_miss_tolerance": frac_miss,
           "rows": rows}
    (RESULTS / "real_lb_spread.json").write_text(json.dumps(out, indent=2),
                                                 encoding="utf-8")
    print(f"\nwrote {RESULTS/'real_lb_spread.json'}")

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    for k, c in zip(keys, plt.cm.viridis(np.linspace(0.1, 0.85, len(keys)))):
        v = np.array([r[k] for r in rows if k in r])
        if len(v) < 50:
            continue
        ax.hist(v, bins=np.linspace(1, 6, 40), histtype="step", lw=1.7,
                color=c, label=k)
    ax.set_xlabel("length-to-breadth ratio", fontsize=8.5)
    ax.set_ylabel("fires", fontsize=8.5)
    ax.set_title("Five readings of the same\n895 perimeters", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(sd * 100, bins=32, color="tab:orange", alpha=0.85)
    ax.axvline(np.median(sd) * 100, color="k", ls="--", lw=1.3,
               label=f"median {np.median(sd)*100:.0f} %")
    ax.set_xlabel("relative sd across estimators (%)", fontsize=8.5)
    ax.set_title("'The' L/B of a real fire is not\na well-posed number", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[2]
    Us = np.linspace(1.2, 3.7, 60)
    Es = np.interp(Us, [1.5, 2.0, 2.5, 3.0, 3.5],
                   [0.843, 1.159, 1.468, 1.769, 2.062])
    for q, style in ((25, ":"), (50, "-"), (75, "--")):
        ax.plot(Us, np.percentile(sd, q) / Es * 100, style, color="tab:red", lw=1.8,
                label=f"p{q} of measurement spread")
    ax.axhline(10, color="k", ls="--", lw=1.1)
    ax.text(1.25, 10.8, "10 % tolerance on wind", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("midflame wind (m/s)", fontsize=8.5)
    ax.set_ylabel("implied $\\sigma$ on wind speed (%)", fontsize=8.5)
    ax.set_title("Lower bound on the wind\nuncertainty this alone causes", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Reading a length-to-breadth ratio off a real fire perimeter",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_real_lb_spread.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
