"""Are real fire perimeters ellipses? The premise this whole study rests on.

Every synthetic result here follows from one modelling premise: a burn scar is an ellipse,
and its head rate, length-to-breadth ratio and orientation are the complete observable a
mask offers. If that premise holds, the degeneracy analysis applies to real fires. If it
does not, the analysis needs re-reading -- in one direction or the other, and it matters
which.

  * If real perimeters are *more* than ellipses -- extra, reproducible shape structure --
    a mask carries more than three numbers and the degeneracy is softer than claimed.
  * If real perimeters are *less* than ellipses -- shapes dominated by terrain, fuel breaks
    and suppression rather than by wind -- then the length-to-breadth signal is buried in
    variation the wind did not cause, and the degeneracy is worse than the idealised
    analysis says.

900 final wildfire perimeters from NIFC's WFIGS service are fitted with the ellipse that
has the same second moments and the same area, and three things are measured: how well
that ellipse reproduces the perimeter, what length-to-breadth ratios real fires actually
show, and how often a fire is a single connected blob at all.

Run:  python scripts/fetch_perimeters.py --n 900
      python scripts/real_perimeters.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon, shape  # noqa: E402
from shapely.ops import transform as shp_transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

import torch  # noqa: E402

from pyrofield.physics.rothermel import LB_CAP, lb_anderson  # noqa: E402

RASTER = 420          # pixels across the long side of the bounding box
MIN_VERTICES = 60     # below this the outline is too coarse to judge shape


def to_local_metres(geom):
    """Project to an azimuthal-equidistant CRS centred on the fire, so metres are metres."""
    c = geom.centroid
    tr = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m +no_defs",
        always_xy=True)
    return shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)


def rasterise(poly: Polygon, n=RASTER):
    """Burn a polygon (with holes) into a boolean image; returns mask and pixel size."""
    minx, miny, maxx, maxy = poly.bounds
    span = max(maxx - minx, maxy - miny)
    if span <= 0:
        return None, None
    px = span / (n - 4)
    w = int((maxx - minx) / px) + 4
    h = int((maxy - miny) / px) + 4
    img = np.zeros((h, w), np.uint8)

    def ring_px(coords):
        a = np.asarray(coords, float)
        return np.stack([(a[:, 0] - minx) / px + 2, (a[:, 1] - miny) / px + 2],
                        axis=1).astype(np.int32)

    cv2.fillPoly(img, [ring_px(poly.exterior.coords)], 1)
    for hole in poly.interiors:
        cv2.fillPoly(img, [ring_px(hole.coords)], 0)
    return img.astype(bool), px


def moment_ellipse(mask):
    """Orientation and axis ratio of the ellipse with the same second moments."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x0, y0 = xs.mean(), ys.mean()
    dx, dy = xs - x0, ys - y0
    cxx, cyy, cxy = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(-ev)
    ev, evec = ev[order], evec[:, order]
    if ev[1] <= 0:
        return None
    ratio = math.sqrt(ev[0] / ev[1])
    ang = math.atan2(evec[1, 0], evec[0, 0])
    return {"cx": x0, "cy": y0, "ratio": ratio, "angle": ang,
            "area_px": int(mask.sum())}


def ellipse_mask(shape_hw, e):
    """The area-matched ellipse with that orientation and axis ratio."""
    h, w = shape_hw
    # area = pi*a*b, b = a/ratio  ->  a = sqrt(area*ratio/pi)
    a = math.sqrt(e["area_px"] * e["ratio"] / math.pi)
    b = a / e["ratio"]
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, (int(round(e["cx"])), int(round(e["cy"]))),
                (max(int(round(a)), 1), max(int(round(b)), 1)),
                math.degrees(e["angle"]), 0, 360, 1, -1)
    return img.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    src = DATA / "wfigs_perimeters.geojson"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/fetch_perimeters.py first")
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    print(f"reading {src.name} ({src.stat().st_size/1e6:.0f} MB)")
    feats = json.loads(src.read_text(encoding="utf-8"))["features"]
    print(f"  {len(feats)} perimeters\n")

    rows, skipped = [], 0
    for i, f in enumerate(feats):
        try:
            g = shape(f["geometry"])
        except Exception:
            skipped += 1
            continue
        if g.is_empty:
            skipped += 1
            continue
        g = to_local_metres(g)
        parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
        parts = [p for p in parts if p.area > 0]
        if not parts:
            skipped += 1
            continue
        total_area = sum(p.area for p in parts)
        main_poly = max(parts, key=lambda p: p.area)
        frac_main = main_poly.area / total_area
        n_vert = len(main_poly.exterior.coords)
        if n_vert < MIN_VERTICES:
            skipped += 1
            continue

        mask, px = rasterise(main_poly)
        if mask is None or mask.sum() < 400:
            skipped += 1
            continue
        e = moment_ellipse(mask)
        if e is None:
            skipped += 1
            continue
        em = ellipse_mask(mask.shape, e)

        p = f["properties"]
        rows.append({
            "name": p.get("poly_IncidentName"), "state": p.get("attr_POOState"),
            "acres": p.get("poly_GISAcres"),
            "area_km2": total_area / 1e6,
            "n_parts": len(parts), "frac_main": frac_main, "n_vertices": n_vert,
            "LB": e["ratio"], "angle_deg": math.degrees(e["angle"]),
            "ellipse_iou": iou(mask, em),
            "px_m": px,
        })
        if (i + 1) % 150 == 0:
            print(f"  processed {i+1}/{len(feats)}  (kept {len(rows)})", flush=True)

    print(f"\n  kept {len(rows)}, skipped {skipped}\n")
    lb = np.array([r["LB"] for r in rows])
    io = np.array([r["ellipse_iou"] for r in rows])
    fm = np.array([r["frac_main"] for r in rows])
    npart = np.array([r["n_parts"] for r in rows])

    print("=" * 78)
    print("[1] how well does an ellipse describe a real fire perimeter?")
    print("=" * 78)
    for q in (5, 25, 50, 75, 95):
        print(f"    IoU with the area-matched moment ellipse, p{q:02d}: "
              f"{np.percentile(io, q):.3f}")
    print(f"    mean {io.mean():.3f}")
    print(f"    fraction above 0.90: {(io > 0.90).mean()*100:.1f} %")
    print(f"    fraction above 0.80: {(io > 0.80).mean()*100:.1f} %")
    print(f"    fraction below 0.70: {(io < 0.70).mean()*100:.1f} %")

    print()
    print("=" * 78)
    print("[2] are they even single blobs?")
    print("=" * 78)
    print(f"    median number of disjoint parts: {np.median(npart):.0f}")
    print(f"    fraction with more than one part: {(npart > 1).mean()*100:.1f} %")
    print(f"    largest part holds, median: {np.median(fm)*100:.1f} % of the burned area")
    print(f"    fraction where the largest part holds under 90 %: "
          f"{(fm < 0.90).mean()*100:.1f} %")

    print()
    print("=" * 78)
    print("[3] where do real length-to-breadth ratios sit?")
    print("=" * 78)
    for q in (5, 25, 50, 75, 95, 99):
        print(f"    LB p{q:02d}: {np.percentile(lb, q):.2f}")
    print(f"    max {lb.max():.2f}")
    print(f"    fraction above FARSITE's cap of {LB_CAP}: {(lb > LB_CAP).mean()*100:.2f} %")
    wind_for = {}
    for target in (2.0, 3.0, 4.0, 5.0):
        v = float(lb_anderson(torch.tensor(target), cap=None))
        wind_for[target] = v
    print("\n    for reference, Anderson's relation predicts")
    for k, v in wind_for.items():
        print(f"      LB = {v:6.2f} at {k:.1f} m/s midflame wind")
    print(f"    so the median real fire, LB = {np.median(lb):.2f}, corresponds to a")
    lo = min(wind_for, key=lambda k: abs(wind_for[k] - np.median(lb)))
    print(f"    midflame wind of roughly {lo:.0f} m/s on that relation.")

    out = {
        "n": len(rows), "skipped": skipped,
        "iou": {"mean": float(io.mean()),
                **{f"p{q}": float(np.percentile(io, q)) for q in (5, 25, 50, 75, 95)},
                "frac_above_0.9": float((io > 0.9).mean()),
                "frac_above_0.8": float((io > 0.8).mean()),
                "frac_below_0.7": float((io < 0.7).mean())},
        "parts": {"median": float(np.median(npart)),
                  "frac_multi": float((npart > 1).mean()),
                  "frac_main_median": float(np.median(fm)),
                  "frac_main_below_0.9": float((fm < 0.9).mean())},
        "LB": {**{f"p{q}": float(np.percentile(lb, q)) for q in (5, 25, 50, 75, 95, 99)},
               "max": float(lb.max()),
               "frac_above_cap": float((lb > LB_CAP).mean())},
        "rows": rows,
    }
    (RESULTS / "real_perimeters.json").write_text(json.dumps(out, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {RESULTS/'real_perimeters.json'}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    ax.hist(io, bins=32, color="tab:red", alpha=0.85)
    ax.axvline(np.median(io), color="k", ls="--", lw=1.3,
               label=f"median {np.median(io):.2f}")
    ax.set_xlabel("IoU of the best area-matched ellipse", fontsize=8.5)
    ax.set_ylabel("fires", fontsize=8.5)
    ax.set_title(f"How elliptical are real fires?\n({len(rows)} NIFC perimeters)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(lb, bins=np.linspace(1, 8, 36), color="tab:blue", alpha=0.85)
    ax.axvline(LB_CAP, color="0.35", ls=":", lw=1.4)
    ax.axvline(np.median(lb), color="k", ls="--", lw=1.3,
               label=f"median {np.median(lb):.2f}")
    ax.set_xlabel("length-to-breadth ratio", fontsize=8.5)
    ax.set_title(f"Real L/B\n({(lb>LB_CAP).mean()*100:.1f} % above FARSITE's cap of 8)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.scatter(lb, io, s=10, alpha=0.4, color="tab:purple")
    ax.set_xlabel("length-to-breadth ratio", fontsize=8.5)
    ax.set_ylabel("ellipse IoU", fontsize=8.5)
    ax.set_xlim(1, min(8, lb.max()))
    ax.set_ylim(0, 1)
    ax.set_title("Elongated fires are not\nbetter-described ellipses", fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle("Real wildfire perimeters against the elliptical template",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_real_perimeters.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
