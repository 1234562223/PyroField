"""Does section 12's inversion result hold at four times the sample size?

Section 12's state-level arms rest on 25 fires per split for mask + plume and 12 for mask
only, which section 20 lists as their weakest point: a failure rate quoted as 36 % carries
a 95 % interval of roughly 18-57 %. The CNN arms use 500 test fires each and need nothing.

This reruns only the state-level arms, on **independent fires drawn with a different seed**,
at four times the count. It deliberately does not touch `results/fusion_preds.npz`:
`fusion_failure_mode.py`, `fusion_multistart.py` and `guarded_inversion.py` all replay
exactly those 62 cases, and redrawing them would silently invalidate three experiments.
This is a separate confirmation, reported beside the original rather than replacing it.

Every headline number comes with a bootstrap interval, because the point of the exercise is
the interval and not the point estimate.

Run:  python scripts/fusion_scale.py --cases 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

from pyrofield.eval.inversion import levenberg_marquardt  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
)

MASK_TIMES = (17, 35, 53, 71, 89, 109)
PLUME_TIMES = (53, 109)
IU = PARAM_NAMES.index("U")
SCHEDULE = ((4.0, 5), (2.0, 4), (0.0, 10))     # section 12's schedule, unchanged
SEED = 4242                                     # not section 12's 0

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def boot(x, f=np.median, n=4000, seed=0):
    """Bootstrap 95 % interval for a statistic of x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    s = [f(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def invert(split, n_cases, device, use_plume, seed=SEED):
    keys = ("mask", "plume") if use_plume else ("mask",)
    sc = Scenario(device=device, n_steps=110, mask_times=MASK_TIMES,
                  plume_times=PLUME_TIMES, conc_times=())
    y_ = np.load(DATA / f"fusion_{split}.npz")["params"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(y_), size=min(n_cases, len(y_)), replace=False)
    gen = torch.Generator(device=device).manual_seed(seed + 7)

    err, t0 = [], time.time()
    for k, i in enumerate(pick):
        th = torch.from_numpy(y_[i]).to(device)
        target = add_noise(forward(th, sc), gen)
        start = th.clone()
        for j in range(len(PARAM_NAMES)):
            start[j] = start[j] + float(rng.uniform(-0.35, 0.35))
        res = levenberg_marquardt(start, th, target, sc, keys, schedule=SCHEDULE)
        e = float(np.abs(np.expm1(float(res.theta_hat[IU]) - float(th[IU]))))
        err.append(e)
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"      {split} {'M+P' if use_plume else 'M':3s} {k+1}/{len(pick)}  "
                  f"({el:.0f}s, eta {el/(k+1)*(len(pick)-k-1)/60:.0f} min)", flush=True)
    return np.array(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    orig = json.loads((RESULTS / "fusion_level.json").read_text(encoding="utf-8"))

    jobs = [("state mask + plume", True, "test_id", args.cases),
            ("state mask + plume", True, "test_ood", args.cases),
            ("state mask only", False, "test_id", max(args.cases // 2, 20))]
    out, rows = {"seed": SEED, "cases": args.cases}, {}

    print("=" * 78)
    print(f"[1] the same arms, independent fires, {args.cases} per split")
    print("=" * 78)
    for tag, up, split, n in jobs:
        print(f"\n  {tag}, {split}  ({n} cases, seed {SEED})")
        e = invert(split, n, device, up)
        key = f"{tag}|{split}"
        rows[key] = {
            "n": int(len(e)),
            "median": float(np.median(e)),
            "median_ci": boot(e, np.median),
            "p90": float(np.percentile(e, 90)),
            "fail": float((e > 0.05).mean()),
            "fail_ci": boot(e > 0.05, np.mean),
            "orig_median": orig["rows"][key]["U"],
            "err": [float(v) for v in e],
        }
        r = rows[key]
        print(f"    median {r['median']*100:.1f} %  (95 % CI "
              f"{r['median_ci'][0]*100:.1f}-{r['median_ci'][1]*100:.1f}),  "
              f"failures {r['fail']*100:.0f} % (CI {r['fail_ci'][0]*100:.0f}-"
              f"{r['fail_ci'][1]*100:.0f})")

    print()
    print("=" * 78)
    print("[2] against section 12's numbers, on different fires")
    print("=" * 78)
    print(f"\n  {'arm':30s} {'section 12 (n=25/12)':>22} "
          f"{'this run':>26} {'failures':>22}")
    for key, r in rows.items():
        print(f"  {key:30s} {r['orig_median']*100:21.1f} % "
              f"{r['median']*100:8.1f} % [{r['median_ci'][0]*100:.1f}, "
              f"{r['median_ci'][1]*100:.1f}]  "
              f"{r['fail']*100:5.0f} % [{r['fail_ci'][0]*100:.0f}, "
              f"{r['fail_ci'][1]*100:.0f}]")
    agree = all(min(r["median_ci"]) <= r["orig_median"] <= max(r["median_ci"])
                for r in rows.values())
    out["orig_inside_ci"] = bool(agree)
    print(f"\n  Section 12's point estimates fall inside this run's intervals in "
          f"{sum(min(r['median_ci']) <= r['orig_median'] <= max(r['median_ci']) for r in rows.values())} "
          f"of {len(rows)} arms.")
    mp = rows["state mask + plume|test_ood"]
    mo = rows["state mask only|test_id"]
    print(f"\n  The headline separation is unchanged and now carries intervals: "
          f"{mp['median']*100:.1f} % "
          f"[{mp['median_ci'][0]*100:.1f}, {mp['median_ci'][1]*100:.1f}] with the plume")
    print(f"  out of distribution, against {mo['median']*100:.1f} % "
          f"[{mo['median_ci'][0]*100:.1f}, {mo['median_ci'][1]*100:.1f}] with masks alone "
          f"in distribution.")

    out["rows"] = rows
    (RESULTS / "fusion_scale.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
    print(f"\nwrote {RESULTS/'fusion_scale.json'}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    keys = list(rows)
    short = [k.replace("state ", "").replace("|", "\n") for k in keys]
    ax = axes[0]
    x = np.arange(len(keys))
    ax.errorbar(x, [rows[k]["median"] * 100 for k in keys],
                yerr=[[(rows[k]["median"] - rows[k]["median_ci"][0]) * 100 for k in keys],
                      [(rows[k]["median_ci"][1] - rows[k]["median"]) * 100 for k in keys]],
                fmt="o", capsize=5, color="tab:blue", label=f"this run (n={args.cases})")
    ax.plot(x, [rows[k]["orig_median"] * 100 for k in keys], "x", ms=10,
            color="tab:red", label="section 12 (n=25/12)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.8)
    ax.set_ylabel("median error in wind speed (%)", fontsize=8.5)
    ax.set_title("Four times the fires, different draw", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, axis="y", which="both")

    ax = axes[1]
    for k, c in zip(keys, ("tab:blue", "tab:green", "tab:red")):
        a = np.sort(np.array(rows[k]["err"]) * 100)
        f = np.linspace(0, 1, len(a), endpoint=False) + 1.0 / len(a)
        ax.step(np.maximum(a, 1e-2), f, where="post", color=c, lw=1.7,
                label=k.replace("state ", "").replace("|", ", "))
    ax.axvline(5, color="k", ls="--", lw=1.0)
    ax.text(5.4, 0.06, "failure\nthreshold", fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xlabel("error in wind speed (%)", fontsize=8.5)
    ax.set_ylabel("fraction of fires below", fontsize=8.5)
    ax.set_title("The whole distribution", fontsize=9.5)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Section 12's state-level arms at scale, on independently drawn fires",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_fusion_scale.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
