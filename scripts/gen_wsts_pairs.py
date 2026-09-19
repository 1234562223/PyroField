"""Build next-day fire prediction pairs from WildfireSpreadTS.

This is WildfireSpreadTS's own benchmark task -- predict tomorrow's active fire from
today's observations -- set up so that the *channels* can be ablated. The question is not
whether a U-net can be trained on this; the dataset's authors already showed that. It is
what the wind channels are worth, because sections 1 and 13 say a fire's own footprint
cannot supply the wind, and that predicts something measurable about a next-day model:
taking the weather away should cost it, and the cost should concentrate where the wind
matters.

Each pair is one fire on two consecutive observation days, cropped to a window centred on
the fire as it stands on day t:

    input   22 observation channels on day t, plus day t's binary active-fire mask
    target  day t+1's binary active-fire mask

Splits are by year, which is what the dataset's authors recommend, since the distributions
differ a lot between years: train on 2018 and 2020, validate on 2019, test on 2021. No fire
appears in two splits, because a fire belongs to one year.

Run:  python scripts/fetch_wsts.py --fires 200
      python scripts/gen_wsts_pairs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS = ROOT / "data" / "wsts", ROOT / "results"
OUT = ROOT / "data" / "wsts_pairs"

B_AF = 22
CROP = 128
SPLITS = {"train": ("2018", "2020"), "val": ("2019",), "test": ("2021",)}
MAX_JUMP_M = 4000.0        # same proximity gate as scripts/wsts_advance.py
SWATH_FRAC = 0.80          # same swath-artefact rejection
MIN_TODAY = 4              # pixels burning today, so there is a fire to propagate
MIN_TOMORROW = 1           # at least one detection to predict


def gated_days(fire_dir):
    """Day-by-day (channels, gated fire mask), with the same gates as the advance study."""
    out, cum = [], None
    for p in sorted(fire_dir.glob("*.tif")):
        try:
            with rasterio.open(p) as src:
                a = src.read().astype(np.float32)
                px = abs(src.res[0])
        except Exception:
            continue
        af = np.isfinite(a[B_AF])
        if cum is None:
            lab, n = ndimage.label(af, structure=np.ones((3, 3)))
            cum = np.zeros_like(af)
            if n:
                sizes = ndimage.sum(af, lab, range(1, n + 1))
                cum = lab == (int(np.argmax(sizes)) + 1)
        raw_new = af & ~cum
        if raw_new.any():
            rr, cc = np.nonzero(raw_new)
            if (max((rr.max() - rr.min() + 1) / af.shape[0],
                    (cc.max() - cc.min() + 1) / af.shape[1]) > SWATH_FRAC
                    and raw_new.sum() > 500):
                continue
        if cum.any() and raw_new.any():
            reach = ndimage.distance_transform_edt(~cum) * px
            new = raw_new & (reach <= MAX_JUMP_M)
        else:
            new = raw_new
        today = cum | new
        out.append((p.stem, a, today.copy()))
        cum = today
    return out


def crop_around(arr, mask, size=CROP):
    """A `size` window centred on the fire, clamped to the tile."""
    h, w = mask.shape
    if mask.any():
        rr, cc = np.nonzero(mask)
        r0, c0 = int(rr.mean()), int(cc.mean())
    else:
        r0, c0 = h // 2, w // 2
    r0 = int(np.clip(r0 - size // 2, 0, max(h - size, 0)))
    c0 = int(np.clip(c0 - size // 2, 0, max(w - size, 0)))
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    out = arr[..., sl[0], sl[1]]
    if out.shape[-2:] != (size, size):          # tile smaller than the window
        pad = [(0, 0)] * (out.ndim - 2) + [
            (0, size - out.shape[-2]), (0, size - out.shape[-1])]
        out = np.pad(out, pad, constant_values=np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--target", choices=("new", "full"), default="new",
                    help="'new' = tomorrow's newly burning pixels (the spread problem); "
                         "'full' = tomorrow's whole active fire (the published framing)")
    ap.add_argument("--out", default=None, help="directory name under data/")
    args = ap.parse_args()
    out_dir = ROOT / "data" / (args.out or f"wsts_pairs_{args.target}"
                               if args.target != "new" else "wsts_pairs")
    globals()["OUT"] = out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"target = {args.target}  ->  {OUT}")

    fires = sorted({p.parent for p in DATA.glob("*/*/*.tif")
                    if not p.name.endswith(".part")})
    by_split = {k: [] for k in SPLITS}
    for fd in fires:
        for k, years in SPLITS.items():
            if fd.parent.name in years:
                by_split[k].append(fd)
    print({k: len(v) for k, v in by_split.items()})

    meta = {}
    for split, dirs in by_split.items():
        X, Y, W, F = [], [], [], []
        t0 = time.time()
        for i, fd in enumerate(dirs):
            days = gated_days(fd)
            for (d0, a0, m0), (d1, a1, m1) in zip(days[:-1], days[1:]):
                # 'new' is the spread problem: what burns tomorrow that is not already
                # burning. 'full' is the published framing, dominated by the fire staying
                # where it is, which a persistence rule gets nearly for free.
                tomorrow = (m1 & ~m0) if args.target == "new" else m1
                if m0.sum() < MIN_TODAY or (m1 & ~m0).sum() < MIN_TOMORROW:
                    continue
                stack = np.concatenate([a0[:B_AF], m0[None].astype(np.float32)], 0)
                x = crop_around(stack, m0, args.crop)
                y = crop_around(tomorrow.astype(np.float32), m0, args.crop)
                if not np.isfinite(y).any() or y.sum() < 1:
                    continue
                X.append(x.astype(np.float16))
                Y.append((y > 0).astype(np.uint8))
                W.append(np.float32(np.nanmedian(a0[6])))
                F.append(f"{fd.parent.name}/{fd.name}/{d0}")
            if (i + 1) % 20 == 0:
                print(f"  {split} {i+1}/{len(dirs)} fires, {len(X)} pairs "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if not X:
            print(f"  {split}: no pairs")
            continue
        X = np.stack(X)
        Y = np.stack(Y)
        p = OUT / f"{split}.npz"
        np.savez_compressed(p, x=X, y=Y, wind=np.array(W, np.float32),
                            key=np.array(F))
        pos = float(Y.mean())
        meta[split] = {"n": int(len(X)), "n_fires": len(dirs),
                       "positive_rate": pos, "mb": p.stat().st_size / 1e6}
        print(f"  {split:5s} {len(X):5d} pairs from {len(dirs):3d} fires, "
              f"{pos*100:.2f} % positive pixels -> {p.name} "
              f"({p.stat().st_size/1e6:.0f} MB)\n")

    name = "wsts_pairs.json" if args.target == "new" else f"wsts_pairs_{args.target}.json"
    (RESULTS / name).write_text(json.dumps({"target": args.target, **meta}, indent=2),
                                encoding="utf-8")
    print(f"wrote {RESULTS/name}")


if __name__ == "__main__":
    main()
