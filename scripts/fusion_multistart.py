"""Does multi-start fix the inversion's failures? It depends which failures.

`scripts/fusion_failure_mode.py` split the state-level inversion's failures in two:

  * mask + plume, out of distribution -- 8 of 9 are **optimiser** failures. The truth fits
    the data better than the answer found, by factors up to 177. The information is there
    and Levenberg-Marquardt, started 35 % away on a non-convex image residual, missed it.
  * mask only -- all twelve are **information** failures, fitting the data exactly as well
    as the truth does.

The first kind should be repairable by restarting from several points and keeping the fit
with the smallest misfit, which needs no ground truth. The second kind has a floor that no
optimiser can cross, and the floor is computable in advance: the starting perturbation is
drawn isotropically in all six parameters, so exactly 1/6 of its squared length lies along
section 1's null direction `v` and is invisible to the masks, while the other 5/6 lies in
directions that are at least partly identifiable. Restarting can clean up the second part
and not the first, which puts the mask-only arm's best achievable median wind-speed error
at about **8.6 %** however many starts are used.

Running both arms is the point, and the prediction is quantitative rather than qualitative:
restarting should push the plume arm far below 8.6 % and leave the mask-only arm at or
above it.

Cost is controlled by screening each start with a short coarse-to-fine ladder and then
refining only the best. The first start is drawn from the same random stream in the same
order as `fusion_level.state_level`, so single-start results are reproduced exactly as a
subset and the comparison against `results/fusion_preds.npz` is paired: same fires, same
observations, same first guess.

Run:  python scripts/fusion_level.py --epochs 200 --lm-cases 25
      python scripts/fusion_multistart.py --starts 3
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
    LOG_PARAMS,
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
)

MASK_TIMES = (17, 35, 53, 71, 89, 109)
PLUME_TIMES = (53, 109)
IU = PARAM_NAMES.index("U")
SCREEN = ((4.0, 3), (1.0, 3))          # cheap ladder, just to rank the starts
REFINE = ((4.0, 5), (2.0, 4), (0.0, 10))   # the schedule fusion_level.py uses
# Median |wind error| left if the only surviving error is the null-direction component of
# an isotropic +/-0.35 start: 1/6 of the squared length, times the direction's U component.
NULL_FLOOR = 0.086

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def rel_err(pred, true, i=IU):
    return np.abs(np.expm1(pred[:, i] - true[:, i]))


def multistart(split, n_cases, device, use_plume, n_starts, seed=0):
    keys = ("mask", "plume") if use_plume else ("mask",)
    sc = Scenario(device=device, n_steps=110, mask_times=MASK_TIMES,
                  plume_times=PLUME_TIMES, conc_times=())
    y_ = np.load(DATA / f"fusion_{split}.npz")["params"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(y_), size=min(n_cases, len(y_)), replace=False)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    rng_extra = np.random.default_rng(seed + 12345)

    pred, true, chosen, t0 = [], [], [], time.time()
    for k, i in enumerate(pick):
        th = torch.from_numpy(y_[i]).to(device)
        target = add_noise(forward(th, sc), gen)

        # Start 1 is drawn exactly as state_level draws it, in the same order.
        starts = [th.clone()]
        for j in range(len(PARAM_NAMES)):
            starts[0][j] = starts[0][j] + float(rng.uniform(-0.35, 0.35))
        for _ in range(n_starts - 1):
            s = th.clone()
            for j in range(len(PARAM_NAMES)):
                s[j] = s[j] + float(rng_extra.uniform(-0.35, 0.35))
            starts.append(s)

        screened = [levenberg_marquardt(s, th, target, sc, keys, schedule=SCREEN)
                    for s in starts]
        best = int(np.argmin([r.loss for r in screened]))
        res = levenberg_marquardt(starts[best], th, target, sc, keys, schedule=REFINE)
        pred.append(res.theta_hat.cpu().numpy())
        true.append(y_[i])
        chosen.append(best)
        if (k + 1) % 5 == 0:
            el = time.time() - t0
            print(f"      {split} {'M+P' if use_plume else 'M':3s} {k+1}/{len(pick)}  "
                  f"({el:.0f}s, eta {el/(k+1)*(len(pick)-k-1):.0f}s)", flush=True)
    return np.array(pred), np.array(true), np.array(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--lm-cases", type=int, default=25)
    ap.add_argument("--report-only", action="store_true",
                    help="re-print and re-plot from results/fusion_multistart.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    single = np.load(RESULTS / "fusion_preds.npz")
    out, rows = {"n_starts": args.starts}, {}

    if args.report_only:
        out = json.loads((RESULTS / "fusion_multistart.json").read_text(encoding="utf-8"))
        rows = out["rows"]
        args.starts = out.get("n_starts", args.starts)
        report(out, rows, args)
        return

    print("=" * 78)
    print(f"[1] {args.starts} starts, screened cheaply and the best one refined")
    print("=" * 78)
    jobs = [("state mask + plume", True, "test_id", args.lm_cases),
            ("state mask + plume", True, "test_ood", args.lm_cases),
            ("state mask only", False, "test_id", max(args.lm_cases // 2, 8))]
    for tag, up, split, n in jobs:
        print(f"\n  {tag}, {split}")
        pred, true, chosen = multistart(split, n, device, up, args.starts)
        key = f"{tag}|{split}"
        e_multi = rel_err(pred, true)
        e_single = rel_err(single[f"{key}__pred"], single[f"{key}__true"])[:len(e_multi)]
        rows[key] = {
            "n": int(len(e_multi)),
            "median_single": float(np.median(e_single)),
            "median_multi": float(np.median(e_multi)),
            "fail_single": float((e_single > 0.05).mean()),
            "fail_multi": float((e_multi > 0.05).mean()),
            "p90_single": float(np.percentile(e_single, 90)),
            "p90_multi": float(np.percentile(e_multi, 90)),
            "start1_kept": float((chosen == 0).mean()),
            "err_multi": [float(v) for v in e_multi],
            "err_single": [float(v) for v in e_single],
        }
        r = rows[key]
        print(f"    median   {r['median_single']*100:6.1f} % -> {r['median_multi']*100:6.1f} %")
        print(f"    p90      {r['p90_single']*100:6.1f} % -> {r['p90_multi']*100:6.1f} %")
        print(f"    failures {r['fail_single']*100:6.0f} % -> {r['fail_multi']*100:6.0f} %"
              f"   (start 1 kept in {r['start1_kept']*100:.0f} % of cases)")

    out["rows"] = rows
    (RESULTS / "fusion_multistart.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    print(f"\nwrote {RESULTS/'fusion_multistart.json'}")
    report(out, rows, args)


def report(out, rows, args):
    print()
    print("=" * 78)
    print("[2] the two failure modes respond differently")
    print("=" * 78)
    print(f"  {'arm':28s} {'median':>18s} {'failures > 5 %':>22s}")
    for key, r in rows.items():
        print(f"  {key:28s} {r['median_single']*100:7.1f} -> {r['median_multi']*100:5.1f} % "
              f"{r['fail_single']*100:12.0f} -> {r['fail_multi']*100:5.0f} %")

    mp = rows["state mask + plume|test_ood"]
    mpi = rows["state mask + plume|test_id"]
    mo = rows["state mask only|test_id"]
    print(f"\n  With the plume, restarting removes "
          f"{(mpi['fail_single']-mpi['fail_multi'])/max(mpi['fail_single'],1e-9)*100:.0f} %"
          f" of the in-distribution failures and "
          f"{(mp['fail_single']-mp['fail_multi'])/max(mp['fail_single'],1e-9)*100:.0f} % of"
          " the")
    print("  out-of-distribution ones, because those were the optimiser losing an answer")
    print(f"  the data contained. The median reaches {mpi['median_multi']*100:.1f} %.")
    print(f"\n  With masks alone it moves the median from {mo['median_single']*100:.1f} % "
          f"to {mo['median_multi']*100:.1f} % and the p90 from "
          f"{mo['p90_single']*100:.0f} % to {mo['p90_multi']*100:.0f} %, and leaves "
          f"{mo['fail_multi']*100:.0f} % of cases failing.")
    print(f"  {NULL_FLOOR*100:.1f} % of that is the floor: the component of the starting")
    print("  error along the null direction, which no optimiser can see. The improvement")
    print("  it does make is the other five sixths of the perturbation being cleaned up,")
    print("  and it does not finish the job -- with masks only the misfit surface is")
    print("  nearly flat, so ranking three starts by misfit is a weak signal.")
    print(f"\n  The comparison that matters: {mpi['median_multi']*100:.1f} % with the "
          f"plume against {mo['median_multi']*100:.1f} % without, a factor of "
          f"{mo['median_multi']/max(mpi['median_multi'],1e-9):.0f},")
    print(f"  with the mask-only arm unable to go below {NULL_FLOOR*100:.1f} % at any "
          "number of starts.")

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    labels = {"state mask + plume|test_id": "mask + plume\nin distribution",
              "state mask + plume|test_ood": "mask + plume\nout of distribution",
              "state mask only|test_id": "mask only\nin distribution"}

    ax = axes[0]
    ks = list(rows)
    x = np.arange(len(ks))
    ax.bar(x - 0.2, [rows[k]["fail_single"] * 100 for k in ks], 0.4,
           color="tab:red", label=f"1 start")
    ax.bar(x + 0.2, [rows[k]["fail_multi"] * 100 for k in ks], 0.4,
           color="tab:blue", label=f"{args.starts} starts")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in ks], fontsize=6.8)
    ax.set_ylabel("cases failing (> 5 % on wind speed)", fontsize=8.5)
    ax.set_title("Restarting fixes solver failures\nand nothing else", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for k, c in (("state mask + plume|test_ood", "tab:blue"),
                 ("state mask only|test_id", "tab:red")):
        a = np.sort(np.array(rows[k]["err_single"])) * 100
        b = np.sort(np.array(rows[k]["err_multi"])) * 100
        f = np.linspace(0, 1, len(a), endpoint=False) + 1 / len(a)
        ax.step(np.maximum(a, 1e-2), f, where="post", color=c, ls="--", lw=1.3,
                label=f"{labels[k].replace(chr(10), ', ')}, 1 start")
        ax.step(np.maximum(b, 1e-2), f, where="post", color=c, lw=2.0,
                label=f"{labels[k].replace(chr(10), ', ')}, {args.starts} starts")
    ax.set_xscale("log")
    ax.set_xlabel("error in wind speed (%)", fontsize=8.5)
    ax.set_ylabel("fraction of cases below", fontsize=8.5)
    ax.set_title("The whole distribution,\nnot just the median", fontsize=9.5)
    ax.legend(fontsize=5.8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    k = "state mask + plume|test_ood"
    a = np.array(rows[k]["err_single"]) * 100
    b = np.array(rows[k]["err_multi"]) * 100
    ax.scatter(np.maximum(a, 1e-2), np.maximum(b, 1e-2), s=26, color="tab:blue",
               edgecolor="k", linewidth=0.5, zorder=3)
    lim = [1e-2, 120]
    ax.plot(lim, lim, "k--", lw=1.0)
    ax.fill_between(lim, [1e-2, 1e-2], lim, color="tab:green", alpha=0.10)
    ax.text(0.03, 30, "restarting helped", fontsize=7, color="tab:green")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("error with 1 start (%)", fontsize=8.5)
    ax.set_ylabel(f"error with {args.starts} starts (%)", fontsize=8.5)
    ax.set_title("Paired, same fire and same\nfirst guess", fontsize=9.5)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Multi-start separates a solver failure from a missing-information "
                 "failure", fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_fusion_multistart.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
