"""Fetch a subset of WildfireSpreadTS without downloading the 45 GB archive.

WildfireSpreadTS (Gerard, Zhao and Sullivan, NeurIPS 2023 Datasets and Benchmarks,
https://doi.org/10.5281/zenodo.8006177) ships as a single 45 GB zip. A zip keeps its table
of contents at the end of the file and stores each member independently, so the archive can
be read selectively: one range request for the central directory, then one per member
wanted. Pulling 200 fires costs about 12 GB and twenty minutes instead of 45 GB and three
hours, and anyone reproducing this does not need the full archive either.

Each member is `YEAR/fire_<globfire id>/YYYY-MM-DD.tif`: a 23-band GeoTIFF at 375 m in the
local UTM zone. The bands, in order, are

    0 VIIRS M11          6 wind speed          12 slope        18 forecast wind speed
    1 VIIRS I2           7 wind direction      13 aspect       19 forecast wind direction
    2 VIIRS I1           8 min temperature     14 elevation    20 forecast temperature
    3 NDVI               9 max temperature     15 PDSI         21 forecast humidity
    4 EVI2              10 energy release      16 landcover    22 active fire
    5 total precip      11 specific humidity   17 forecast precip

Band 22 (active fire) is NaN where VIIRS made no detection, which is most pixels on most
days; bands 6 and 7 are the GRIDMET daily wind that the fire actually experienced. Those
three are what this study needs: an observed fire progression with the driving wind
measured independently of it.

Run:  python scripts/fetch_wsts.py --fires 200 --min-days 15
"""

from __future__ import annotations

import argparse
import json
import io
import struct
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEST = ROOT / "data" / "wsts"

URL = "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip?download=1"


def ranged(s, a, b, retries=5):
    """A byte span, retried. The last span of the archive is short by design, so a
    response shorter than asked for is only an error when it is also empty."""
    want = b - a + 1
    for k in range(retries):
        try:
            r = s.get(URL, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            r.raise_for_status()
            if r.content:
                return r.content
        except Exception:
            pass
        time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"range {a}-{b} ({want} bytes) failed")


def central_directory(s):
    """Every member's name, local-header offset and compressed size.

    The directory is handed to `zipfile` rather than parsed by hand: this archive is
    ZIP64, so every member past the 4 GB mark has 0xFFFFFFFF in the 32-bit local-header
    offset slot and its real offset in a Zip64 extra field. Parsing that by hand and
    getting it wrong produces range requests that land in the middle of another member
    and fail to inflate, which is exactly what happened first time round.
    """
    h = s.head(URL, allow_redirects=True, timeout=60)
    size = int(h.headers.get("Content-Length", 0))
    # Zenodo answers an overloaded backend with a 92-byte "504 Gateway Time-out" page,
    # and taking its Content-Length as the archive size sends the next request to a
    # negative offset with a confusing error. Fail here instead, where it is obvious.
    if h.status_code != 200 or size < 1_000_000_000:
        raise SystemExit(
            f"the archive is not being served right now: HTTP {h.status_code}, "
            f"Content-Length {size}. Zenodo returns 504 when its backend is busy; "
            f"wait and rerun -- already-fetched files are skipped.")
    tail = ranged(s, size - 65_600, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    n, cds, cdo = struct.unpack("<HII", tail[i + 10:i + 20])
    if cdo == 0xFFFFFFFF or n == 0xFFFF:
        j = tail.rfind(b"PK\x06\x06")
        n, cds, cdo = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = ranged(s, cdo, cdo + cds - 1)

    # A minimal zip made of just this directory, so zipfile will decode it (including
    # the Zip64 extras) with the central directory sitting at offset 0.
    eocd = struct.pack("<IHHHHIIH", 0x06054b50, 0, 0,
                       min(n, 0xFFFF), min(n, 0xFFFF), len(cd), 0, 0)
    zf = zipfile.ZipFile(io.BytesIO(cd + eocd))
    out = [{"name": z.filename, "lho": z.header_offset, "csize": z.compress_size,
            "usize": z.file_size, "method": z.compress_type}
           for z in zf.infolist() if z.filename.endswith(".tif")]
    bad = [m for m in out if m["lho"] >= size or m["lho"] < 0]
    if bad:
        raise RuntimeError(f"{len(bad)} members have an offset outside the archive")
    return size, out


LOCAL_HEADER_SLACK = 1024   # 30 fixed bytes + filename + extra field, generously


def member_bytes(s, m):
    """One range request per member, not two.

    The obvious implementation reads the 30-byte local header, learns how long the name
    and extra field are, and then asks for the data. That doubles the request count and
    makes half of them tiny, and Zenodo starts refusing the tiny ones: a first version
    stalled after 7600 files with every 30-byte header request failing while the byte
    rate was fine, which is rate limiting on requests rather than bandwidth. Asking for
    the header and the data together in one span and slicing locally removes both
    problems.
    """
    span = ranged(s, m["lho"], m["lho"] + LOCAL_HEADER_SLACK + m["csize"] - 1)
    if span[:4] != b"PK\x03\x04":
        raise RuntimeError(f"no local header at {m['lho']} for {m['name']}")
    nlen, elen = struct.unpack("<HH", span[26:30])
    start = 30 + nlen + elen
    raw = span[start:start + m["csize"]]
    if len(raw) != m["csize"]:
        raise RuntimeError(f"short read for {m['name']}: "
                           f"{len(raw)} of {m['csize']} bytes")
    if m["method"] == 0:
        return raw
    if m["method"] == 8:
        return zlib.decompressobj(-15).decompress(raw)
    raise RuntimeError(f"unsupported compression {m['method']} for {m['name']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fires", type=int, default=200,
                    help="how many fire events to fetch (0 = all 607)")
    ap.add_argument("--min-days", type=int, default=15,
                    help="skip fires with fewer observation days than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "pyrofield-research/1.0"

    print("reading the archive's table of contents (two range requests)")
    size, members = central_directory(s)
    print(f"  {size/1e9:.1f} GB archive, {len(members)} GeoTIFFs")

    by_fire = defaultdict(list)
    for m in members:
        year, fire, _ = m["name"].split("/")
        by_fire[(year, fire)].append(m)
    eligible = {k: v for k, v in by_fire.items() if len(v) >= args.min_days}
    print(f"  {len(by_fire)} fire events, {len(eligible)} with >= {args.min_days} days")

    # Deterministic choice, spread across years rather than taking the first ones.
    keys = sorted(eligible, key=lambda k: (k[0], k[1]))
    if args.fires and args.fires < len(keys):
        import numpy as np
        rng = np.random.default_rng(args.seed)
        by_year = defaultdict(list)
        for k in keys:
            by_year[k[0]].append(k)
        quota = {y: max(1, round(args.fires * len(v) / len(keys)))
                 for y, v in by_year.items()}
        keys = []
        for y, v in sorted(by_year.items()):
            idx = rng.choice(len(v), size=min(quota[y], len(v)), replace=False)
            keys += [v[int(i)] for i in sorted(idx)]
        keys = keys[:args.fires]
    todo = [m for k in keys for m in sorted(eligible[k], key=lambda m: m["name"])]
    want = sum(m["csize"] for m in todo)
    print(f"  fetching {len(keys)} fires, {len(todo)} days, {want/1e9:.1f} GB\n")

    got = skipped = failed = 0
    t0 = time.time()
    for i, m in enumerate(todo):
        dest = DEST / m["name"]
        if dest.exists() and dest.stat().st_size == m["usize"]:
            skipped += 1
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so a reader never sees a half-written GeoTIFF.
            tmp = dest.with_suffix(".tif.part")
            tmp.write_bytes(member_bytes(s, m))
            tmp.replace(dest)
            got += 1
        except Exception as e:
            failed += 1
            print(f"    failed {m['name']}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            done = got + skipped
            mb = sum(x["csize"] for x in todo[:i + 1]) / 1e6
            print(f"  {i+1}/{len(todo)}  kept {done}  failed {failed}  "
                  f"{mb/1e3:.1f} GB  ({el:.0f}s, {mb/max(el,1):.1f} MB/s, "
                  f"eta {el/(i+1)*(len(todo)-i-1)/60:.0f} min)", flush=True)

    manifest = {"source": URL, "fires": [f"{y}/{f}" for y, f in keys],
                "n_days": len(todo), "fetched": got, "skipped": skipped,
                "failed": failed, "min_days": args.min_days}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  fetched {got}, already present {skipped}, failed {failed}")
    print(f"  wrote {DEST/'manifest.json'}")


if __name__ == "__main__":
    main()
