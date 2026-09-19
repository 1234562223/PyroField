"""Download real wildfire perimeters from NIFC's WFIGS service.

Every result in this study so far is synthetic. This is the first real data: the
operational fire perimeters that US federal and state agencies actually record, served
from the Wildland Fire Interagency Geospatial Services (WFIGS) feature service.

They are the right object to test against. The whole degeneracy argument assumes a burn
scar is an ellipse whose head rate, length-to-breadth ratio and orientation are the
complete observable. Real perimeters are the direct measurement of whether that is true.

Perimeters are fetched as GeoJSON, filtered to final perimeters of fires large enough to
be well resolved, and cached to disk so the analysis can be re-run offline.

Run:  python scripts/fetch_perimeters.py --n 800
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

SERVICE = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
           "WFIGS_Interagency_Perimeters/FeatureServer/0/query")
FIELDS = ["poly_IncidentName", "poly_FeatureCategory", "poly_GISAcres",
          "poly_PolygonDateTime", "poly_IRWINID", "attr_FinalAcres",
          "attr_FireDiscoveryDateTime", "attr_POOState"]
# Large enough that the perimeter is resolved by many vertices, small enough to stay a
# single wind-driven event rather than a month-long complex.
MIN_ACRES, MAX_ACRES = 300.0, 200_000.0


def probe(session):
    """What categories exist, and how many records of each."""
    r = session.get(SERVICE, params={
        "where": "1=1", "outFields": "poly_FeatureCategory",
        "returnDistinctValues": "true", "returnGeometry": "false", "f": "json",
    }, timeout=90)
    r.raise_for_status()
    cats = sorted({f["attributes"]["poly_FeatureCategory"]
                   for f in r.json().get("features", [])
                   if f["attributes"].get("poly_FeatureCategory")})
    print("  feature categories present:")
    for c in cats:
        n = count(session, f"poly_FeatureCategory = '{c}'")
        print(f"    {c:44s} {n:>8,}")
    return cats


def count(session, where):
    r = session.get(SERVICE, params={"where": where, "returnCountOnly": "true",
                                     "f": "json"}, timeout=90)
    r.raise_for_status()
    return r.json().get("count", 0)


def fetch(session, where, n_max, page=250):
    """Page through the service and return GeoJSON features."""
    feats, offset = [], 0
    while len(feats) < n_max:
        r = session.get(SERVICE, params={
            "where": where, "outFields": ",".join(FIELDS), "f": "geojson",
            "outSR": 4326, "resultRecordCount": page, "resultOffset": offset,
            "orderByFields": "poly_GISAcres DESC",
        }, timeout=180)
        r.raise_for_status()
        js = r.json()
        got = js.get("features", [])
        if not got:
            break
        feats.extend(got)
        offset += len(got)
        print(f"    fetched {len(feats):,} ...", flush=True)
        if len(got) < page:
            break
        time.sleep(0.4)
    return feats[:n_max]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "pyrofield-research/1.0"

    print("probing the service")
    cats = probe(s)
    if args.probe_only:
        return

    # Prefer final perimeters; fall back to whatever category dominates.
    final = [c for c in cats if "Final" in c]
    cat_clause = (" OR ".join(f"poly_FeatureCategory = '{c}'" for c in final)
                  if final else "1=1")
    where = (f"({cat_clause}) AND poly_GISAcres >= {MIN_ACRES} "
             f"AND poly_GISAcres <= {MAX_ACRES}")
    total = count(s, where)
    print(f"\n  matching records: {total:,}")
    print(f"  where: {where}\n")

    feats = fetch(s, where, args.n)
    out = DATA / "wfigs_perimeters.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                   encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"\n  saved {len(feats):,} perimeters to {out}  ({mb:.1f} MB)")

    acres = sorted(f["properties"].get("poly_GISAcres") or 0 for f in feats)
    if acres:
        print(f"  acres: min {acres[0]:,.0f}  median {acres[len(acres)//2]:,.0f}  "
              f"max {acres[-1]:,.0f}")


if __name__ == "__main__":
    main()
