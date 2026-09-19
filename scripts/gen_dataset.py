"""Generate the simulated dataset for the fusion-level comparison.

The comparison in `scripts/fusion_level.py` needs a learned baseline, and a learned
baseline needs data. Fires are drawn from an operational envelope, split so that one test
set sits inside the training distribution and one sits outside it in wind speed.

That split is the point. A network fused at the feature level can do well in distribution
by learning the training prior over wind, whether or not the wind is in the data at all.
Only a test outside that prior separates having learned the physics from having learned
the sampling.

Stored as uint8 so the whole set fits in a few hundred MB.

Run:  python scripts/gen_dataset.py --n-train 2500 --n-test 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, add_noise, forward, pack  # noqa: E402

MASK_TIMES = (17, 35, 53, 71, 89, 109)
PLUME_TIMES = (53, 109)

# Training envelope. Wind is deliberately bounded so that an out-of-distribution test is
# possible; everything else is drawn from the same ranges in all splits.
TRAIN_U = (2.0, 5.0)
OOD_U = (5.5, 8.0)
RANGES = {
    "R0": (0.06, 0.16),
    "theta_w": (0.0, 2 * np.pi),
    "lb_k": (0.22, 0.52),
    "w_buoy": (2.0, 4.5),
    "Q": (0.6, 1.6),
}


def scen(device):
    return Scenario(device=device, n_steps=110, mask_times=MASK_TIMES,
                    plume_times=PLUME_TIMES, conc_times=())


def sample(rng, u_range):
    v = {k: float(rng.uniform(*r)) for k, r in RANGES.items()}
    v["U"] = float(rng.uniform(*u_range))
    return v


def build(n, u_range, seed, device, tag):
    sc = scen(device)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 99991)
    masks = np.zeros((n, len(MASK_TIMES), sc.ny, sc.nx), np.uint8)
    plumes = np.zeros((n, len(PLUME_TIMES), sc.camera.height, sc.camera.width), np.uint8)
    params = np.zeros((n, len(PARAM_NAMES)), np.float32)

    t0 = time.time()
    for i in range(n):
        v = sample(rng, u_range)
        th = pack(v, device=device)
        obs = add_noise(forward(th, sc), gen)
        masks[i] = (obs["mask"].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        # The rendered image lives in [0.35, 0.92]; stretch before quantising so the
        # uint8 step is small compared with the sensor noise rather than dominating it.
        pl = ((obs["plume"] - 0.30) / 0.65).clamp(0, 1)
        plumes[i] = (pl * 255).to(torch.uint8).cpu().numpy()
        params[i] = th.cpu().numpy()
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"    {tag} {i+1}/{n}  ({el:.0f}s, eta {el/(i+1)*(n-i-1):.0f}s)",
                  flush=True)
    return masks, plumes, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=500)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"  train/val/id-test wind: {TRAIN_U[0]}-{TRAIN_U[1]} m/s")
    print(f"  ood-test wind:          {OOD_U[0]}-{OOD_U[1]} m/s")
    print(f"  masks {len(MASK_TIMES)} x 96x96, plume {len(PLUME_TIMES)} x 56x56\n")

    splits = {
        "train": (args.n_train, TRAIN_U, 1),
        "val": (max(args.n_test // 2, 100), TRAIN_U, 2),
        "test_id": (args.n_test, TRAIN_U, 3),
        "test_ood": (args.n_test, OOD_U, 4),
    }
    for name, (n, ur, seed) in splits.items():
        m, p, y = build(n, ur, seed, device, name)
        out = DATA / f"fusion_{name}.npz"
        np.savez_compressed(out, mask=m, plume=p, params=y,
                            param_names=np.array(PARAM_NAMES))
        print(f"  {name:9s} n={n:5d}  ->  {out.name}  "
              f"({out.stat().st_size/1e6:.0f} MB)\n")

    print("done")


if __name__ == "__main__":
    main()
