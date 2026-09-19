"""Attach ERA5 reanalysis wind to each real fire perimeter.

With wind, the real perimeters answer the question this study has been inferring around:
does a real fire's shape predict the real wind that drove it? That is Alexander's (1985)
r = 0.865 analysis, redone on 895 modern fires, and it replaces the 15-35 % bracket that
RESULTS.md section 2 has been carrying as its weakest link.

Wind comes from the ERA5 reanalysis through Open-Meteo's archive API: daily maximum 10 m
wind speed and dominant direction over each fire's active period, at the perimeter
centroid.

Two honest caveats, carried forward into the analysis rather than hidden:
  * ERA5 is ~25 km gridded reanalysis. It is not the wind at the fire; it is the synoptic
    wind over the area, and terrain channelling is invisible to it.
  * The fire responds to *midflame* wind, which is the 10 m wind times a wind adjustment
    factor of roughly 0.1-0.5 depending on canopy. That factor is not known per fire, so
    the analysis fits it rather than assuming it.

Run:  python scripts/fetch_wind.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["wind_speed_10m_max", "wind_direction_10m_dominant", "wind_gusts_10m_max"]
MAX_DAYS = 21          # cap the active window; beyond this it is not one wind event
MIN_DAYS = 1
EARLIEST = datetime(1980, 1, 1, tzinfo=timezone.utc)


def parse_ms(v):
    """WFIGS dates arrive as epoch milliseconds."""
    if v in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def window(props):
    """Active period: discovery to containment, capped, with sane fallbacks."""
    start = parse_ms(props.get("attr_FireDiscoveryDateTime"))
    end = (parse_ms(props.get("attr_ContainmentDateTime"))
           or parse_ms(props.get("attr_ControlDateTime"))
           or parse_ms(props.get("poly_PolygonDateTime")))
    if start is None:
        return None
    if end is None or end <= start:
        end = start + timedelta(days=7)
    days = (end - start).days
    if days < MIN_DAYS:
        end = start + timedelta(days=MIN_DAYS)
    if days > MAX_DAYS:
        end = start + timedelta(days=MAX_DAYS)
    if start < EARLIEST or end > datetime.now(timezone.utc) - timedelta(days=6):
        return None
    return start, end


def main():
    src = DATA / "wfigs_perimeters.geojson"
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/fetch_perimeters.py first")
    feats = json.loads(src.read_text(encoding="utf-8"))["features"]
    print(f"{len(feats)} perimeters")

    jobs = []
    for f in feats:
        p = f["properties"]
        w = window(p)
        if w is None:
            continue
        try:
            c = shape(f["geometry"]).centroid
        except Exception:
            continue
        if not (np.isfinite(c.x) and np.isfinite(c.y)):
            continue
        jobs.append({"irwin": p.get("poly_IRWINID"), "name": p.get("poly_IncidentName"),
                     "lat": round(float(c.y), 4), "lon": round(float(c.x), 4),
                     "start": w[0].strftime("%Y-%m-%d"), "end": w[1].strftime("%Y-%m-%d"),
                     "acres": p.get("poly_GISAcres")})
    print(f"  {len(jobs)} have a usable date window and centroid\n")

    s = requests.Session()
    s.headers["User-Agent"] = "pyrofield-research/1.0"
    out, fails, t0 = [], 0, time.time()
    for i, j in enumerate(jobs):
        try:
            r = s.get(API, params={
                "latitude": j["lat"], "longitude": j["lon"],
                "start_date": j["start"], "end_date": j["end"],
                "daily": ",".join(DAILY), "wind_speed_unit": "ms", "timezone": "UTC",
            }, timeout=60)
            r.raise_for_status()
            d = r.json().get("daily", {})
            sp = np.array([v for v in d.get("wind_speed_10m_max", []) if v is not None],
                          float)
            di = np.array([v for v in d.get("wind_direction_10m_dominant", [])
                           if v is not None], float)
            gu = np.array([v for v in d.get("wind_gusts_10m_max", []) if v is not None],
                          float)
            if len(sp) == 0:
                fails += 1
                continue
            # Circular mean of the dominant daily directions, and how coherent they were.
            rad = np.deg2rad(di) if len(di) else np.array([0.0])
            cx, cy = np.cos(rad).mean(), np.sin(rad).mean()
            out.append({**j,
                        "n_days": int(len(sp)),
                        "wind10_max": float(sp.max()),
                        "wind10_mean_of_daily_max": float(sp.mean()),
                        "wind10_p75": float(np.percentile(sp, 75)),
                        "gust_max": float(gu.max()) if len(gu) else None,
                        "dir_mean_deg": float(np.rad2deg(np.arctan2(cy, cx)) % 360),
                        "dir_coherence": float(np.hypot(cx, cy))})
        except Exception:
            fails += 1
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(jobs)}  kept {len(out)}  failed {fails}  "
                  f"({el:.0f}s, eta {el/(i+1)*(len(jobs)-i-1):.0f}s)", flush=True)
        time.sleep(0.12)          # stay well inside the free-tier rate limit

    dest = DATA / "fire_wind.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n  kept {len(out)}, failed {fails}")
    print(f"  wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
    if out:
        w = np.array([o["wind10_mean_of_daily_max"] for o in out])
        print(f"  10 m wind (mean of daily maxima): median {np.median(w):.2f} m/s, "
              f"p10 {np.percentile(w,10):.2f}, p90 {np.percentile(w,90):.2f}")
        coh = np.array([o["dir_coherence"] for o in out])
        print(f"  direction coherence over the window: median {np.median(coh):.2f} "
              f"(1 = steady, 0 = boxing the compass)")


if __name__ == "__main__":
    main()
