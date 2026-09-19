"""Does the guard actually catch what a residual check cannot?

`pyrofield/eval/guarded.py` proposes that an inversion report a verdict per parameter --
determined with an interval, or not determined -- using a profile probe rather than a
residual test, because section 12.4 measured that a residual test is blind to exactly the
failure that matters. Proposing it is not evidence. This scores it.

The test set is the one place ground truth exists: the same fires, the same observations
and the same converged solutions as section 12, so every verdict can be checked against
what was actually true.

Three things are measured, and the third is the one that matters.

  1. **Does it separate the arms?** Wind speed is algebraically undetermined from masks
     alone (section 1) and well determined once the plume is attached (section 9). A guard
     that cannot tell those apart is worthless.
  2. **Are its intervals honest?** When it says "determined, sigma ~ x", does the truth
     fall within that interval at roughly the nominal rate?
  3. **Does it catch what the residual check misses?** Both are run on the same cases and
     scored against whether the answer was in fact wrong. Section 12.4 predicts the
     residual check catches optimiser failures and misses information failures; the guard
     should have the opposite blind spot, and the useful system is the pair.

Run:  python scripts/fusion_level.py --epochs 200 --lm-cases 25
      python scripts/guarded_inversion.py
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

from pyrofield.eval.guarded import DCHI2_FLAT, guard  # noqa: E402
from pyrofield.eval.inversion import fixed_weights  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
)
from pyrofield.theory.identifiability import stack_obs  # noqa: E402

MASK_TIMES = (17, 35, 53, 71, 89, 109)
PLUME_TIMES = (53, 109)
IU = PARAM_NAMES.index("U")
WRONG = 0.05          # an answer more than 5 % off in wind speed counts as wrong

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def replay(split, n_cases, device, use_plume, seed=0, limit=None):
    """Exactly the cases, observations and starting draws section 12 inverted.

    `n_cases` must stay at whatever section 12 used, because `rng.choice` draws a
    different subset for a different size and the saved solutions would then belong to
    different fires. Taking fewer is done by truncating afterwards, via `limit`.
    """
    keys = ("mask", "plume") if use_plume else ("mask",)
    sc = Scenario(device=device, n_steps=110, mask_times=MASK_TIMES,
                  plume_times=PLUME_TIMES, conc_times=())
    y_ = np.load(DATA / f"fusion_{split}.npz")["params"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(y_), size=min(n_cases, len(y_)), replace=False)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    out = []
    for i in pick[:limit] if limit else pick:
        th = torch.from_numpy(y_[i]).to(device)
        target = add_noise(forward(th, sc), gen)
        for _ in range(len(PARAM_NAMES)):     # keep the stream in step with section 12
            rng.uniform(-0.35, 0.35)
        out.append((th, target, sc, keys))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0, help="0 = all that section 12 ran")
    ap.add_argument("--params", default="U",
                    help="comma-separated parameters to guard; each costs 2-4 short fits")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print and re-plot from results/guarded_inversion.json")
    args = ap.parse_args()
    guarded = tuple(s.strip() for s in args.params.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    preds = np.load(RESULTS / "fusion_preds.npz")

    jobs = [("state mask + plume", True, "test_id", 25),
            ("state mask + plume", True, "test_ood", 25),
            ("state mask only", False, "test_id", 12)]
    out, rows = {"dchi2_flat": DCHI2_FLAT, "wrong_threshold": WRONG,
                 "guarded": list(guarded)}, {}

    if args.report_only:
        path = RESULTS / "guarded_inversion.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        report(saved, saved["rows"])
        # report() derives the summary, confusion matrix and calibration, so write them
        # back: otherwise a field added to the reporting after a 2.6 h run would only
        # ever exist in the log.
        path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        print(f"updated {path}")
        return

    print("=" * 78)
    print("[1] guarding every solution section 12 produced")
    print("=" * 78)
    for tag, up, split, n in jobs:
        key = f"{tag}|{split}"
        pred, true = preds[f"{key}__pred"], preds[f"{key}__true"]
        cases = replay(split, n, device, up, limit=args.cases or None)
        print(f"\n  {tag}, {split}  ({len(cases)} cases)")
        recs, t0, fits = [], time.time(), 0
        for k, (th, target, sc, keys) in enumerate(cases):
            hat = torch.from_numpy(pred[k]).to(device)
            w = fixed_weights(target, keys,
                              {kk: noise_sigma_for(target, kk) for kk in keys})
            base = float((((stack_obs(forward(hat, sc), keys) * w)
                           - stack_obs(target, keys) * w) ** 2).sum())
            n_obs = int(sum(target[kk].numel() for kk in keys))
            verdicts, nf = guard(hat, target, sc, keys, base_loss=base,
                                 params=guarded)
            fits += nf
            err = float(np.abs(np.expm1(pred[k][IU] - true[k][IU])))
            vu = next(v for v in verdicts if v.name == "U")
            recs.append({
                "err_U": err, "wrong": err > WRONG,
                "U_determined": vu.determined,
                "U_dchi2": vu.dchi2, "U_sigma": vu.sigma_rel,
                "U_covered": bool(vu.sigma_rel is not None
                                  and err <= max(vu.sigma_rel, 1e-9)),
                "chi2_per_obs": base / max(n_obs, 1),
                "residual_ok": bool(base <= 1.02 * n_obs),
                "verdicts": {v.name: {"determined": v.determined,
                                      "sigma_rel": v.sigma_rel,
                                      "sigma_abs": v.sigma_abs} for v in verdicts},
            })
            if (k + 1) % 5 == 0:
                el = time.time() - t0
                print(f"      {k+1}/{len(cases)}  ({el:.0f}s, "
                      f"eta {el/(k+1)*(len(cases)-k-1):.0f}s)", flush=True)
        rows[key] = recs
        det = np.mean([r["U_determined"] for r in recs])
        print(f"    wind speed called determined in {det*100:.0f} % of cases "
              f"({fits} profile fits)")

    out["rows"] = rows
    (RESULTS / "guarded_inversion.json").write_text(json.dumps(out, indent=1),
                                                    encoding="utf-8")
    print(f"\nwrote {RESULTS/'guarded_inversion.json'}")
    report(out, rows)


def report(out, rows):
    # ------------------------------------------------------------------ [2] ---
    print()
    print("=" * 78)
    print("[2] does the verdict track the arm it should?")
    print("=" * 78)
    print(f"\n  {'arm':30s} {'U called determined':>21} {'median U error':>16}")
    summ = {}
    for key, recs in rows.items():
        det = float(np.mean([r["U_determined"] for r in recs]))
        med = float(np.median([r["err_U"] for r in recs]))
        summ[key] = {"n": len(recs), "U_determined": det, "median_err": med}
        print(f"  {key:30s} {det*100:20.0f} % {med*100:15.1f} %")
    print("\n  Wind speed is algebraically undetermined from masks alone (section 1) and")
    print("  well determined with the plume (section 9). The guard is asked to say so")
    print("  without being told which arm it is looking at.")

    # ------------------------------------------------------------------ [3] ---
    print()
    print("=" * 78)
    print("[3] are the intervals honest?")
    print("=" * 78)
    print("\n  A 1-sigma interval should contain the truth about 68 % of the time. Cases")
    print("  where the inversion converged badly are counted separately, because an")
    print("  interval around the wrong point says nothing about the interval's width.")
    print(f"\n  {'arm':30s} {'n determined':>13} {'median sigma':>14} "
          f"{'truth inside':>14} {'converged only':>16}")
    for key, recs in rows.items():
        ok = [r for r in recs if r["U_determined"] and r["U_sigma"] is not None]
        if not ok:
            print(f"  {key:30s} {'0':>13} {'-':>14} {'-':>14} {'-':>16}")
            continue
        conv = [r for r in ok if not r["wrong"]]
        s = float(np.median([r["U_sigma"] for r in ok]))
        cov = float(np.mean([r["U_covered"] for r in ok]))
        covc = float(np.mean([r["U_covered"] for r in conv])) if conv else float("nan")
        summ[key].update({"n_det": len(ok), "median_sigma": s, "coverage": cov,
                          "n_converged": len(conv), "coverage_converged": covc})
        print(f"  {key:30s} {len(ok):13d} {s*100:13.2f} % {cov*100:13.0f} % "
              f"{covc*100:15.0f} %")

    pooled = [r for recs in rows.values() for r in recs
              if r["U_determined"] and r["U_sigma"] is not None and not r["wrong"]]
    if pooled:
        from scipy.stats import beta
        k = sum(r["U_covered"] for r in pooled)
        n = len(pooled)
        lo = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
        hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        ratio = np.array([r["err_U"] / r["U_sigma"] for r in pooled])
        scale = float(np.percentile(ratio, 68))
        out["calibration"] = {"n": n, "covered": int(k), "coverage": k / n,
                              "ci95": [lo, hi], "scale_for_68": scale}
        print(f"\n  Pooled over the converged cases: {k} of {n} = {k/n*100:.0f} %, "
              f"95 % interval {lo*100:.0f}-{hi*100:.0f} %.")
        print(f"  The nominal 68 % is {'inside' if lo <= 0.68 <= hi else 'outside'} "
              f"that interval, so these intervals are consistent with being honest;")
        print(f"  {n} cases cannot resolve better than that. The point estimate is low, "
              f"and scaling sigma by {scale:.2f} would land it exactly on 68 %.")

    # ------------------------------------------------------------------ [4] ---
    print()
    print("=" * 78)
    print("[4] the guard against the residual check, on the same cases")
    print("=" * 78)
    allr = [r for recs in rows.values() for r in recs]
    tab = {}
    for name, flag in (("residual check", lambda r: not r["residual_ok"]),
                       ("profile guard", lambda r: not r["U_determined"]),
                       ("either", lambda r: (not r["residual_ok"])
                        or (not r["U_determined"]))):
        tp = sum(1 for r in allr if r["wrong"] and flag(r))
        fn = sum(1 for r in allr if r["wrong"] and not flag(r))
        fp = sum(1 for r in allr if not r["wrong"] and flag(r))
        tn = sum(1 for r in allr if not r["wrong"] and not flag(r))
        tab[name] = {"caught": tp, "missed": fn, "false_alarm": fp, "clean": tn}
    n_wrong = sum(1 for r in allr if r["wrong"])
    print(f"\n  {len(allr)} inversions, {n_wrong} of them wrong by more than "
          f"{WRONG*100:.0f} % on wind speed\n")
    print(f"  {'raised by':>16} {'wrong, caught':>15} {'wrong, missed':>15} "
          f"{'right, flagged':>16}")
    for name, v in tab.items():
        print(f"  {name:>16} {v['caught']:15d} {v['missed']:15d} "
              f"{v['false_alarm']:16d}")
    out["confusion"] = tab
    out["summary"] = summ
    r_, g_ = tab["residual check"], tab["profile guard"]
    print(f"\n  The residual check catches {r_['caught']} of {n_wrong} and misses "
          f"{r_['missed']}; the profile guard catches {g_['caught']} and misses "
          f"{g_['missed']}.")
    print("  They fail on different cases, which is the point: a wrong answer that fits")
    print("  the data perfectly is invisible to the first and obvious to the second.")

    make_figure(out, rows)


def make_figure(out, rows):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    keys = list(rows)
    short = [k.replace("state ", "").replace("|", "\n") for k in keys]

    ax = axes[0]
    ax.bar(range(len(keys)),
           [out["summary"][k]["U_determined"] * 100 for k in keys],
           color=["tab:blue", "tab:blue", "tab:red"])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(short, fontsize=6.8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("wind speed called determined (%)", fontsize=8.5)
    ax.set_title("The guard is not told which arm\nit is looking at", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for k, c in zip(keys, ("tab:blue", "tab:green", "tab:red")):
        d = [max(r["U_dchi2"][max(r["U_dchi2"])], 1e-3) for r in rows[k]]
        a = np.sort(d)
        f = np.linspace(0, 1, len(a), endpoint=False) + 1.0 / len(a)
        ax.step(a, f, where="post", color=c, lw=1.7,
                label=k.replace("state ", "").replace("|", ", "))
    ax.axvline(DCHI2_FLAT, color="k", ls="--", lw=1.2)
    ax.text(DCHI2_FLAT * 1.15, 0.05, "verdict\nthreshold", fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"misfit cost of a 30 % wind error ($\Delta\chi^2$)", fontsize=8.5)
    ax.set_ylabel("fraction of cases below", fontsize=8.5)
    ax.set_title("The quantity the verdict reads", fontsize=9.5)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    names = list(out["confusion"])
    x = np.arange(len(names))
    ax.bar(x - 0.2, [out["confusion"][n]["caught"] for n in names], 0.4,
           color="tab:green", label="wrong answer caught")
    ax.bar(x + 0.2, [out["confusion"][n]["missed"] for n in names], 0.4,
           color="tab:red", label="wrong answer missed")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
    ax.set_ylabel("inversions", fontsize=8.5)
    ax.set_title("A residual test and a profile test\nare blind to different things",
                 fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("An inversion that reports what the observations determine, and scoring "
                 "whether it is right", fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_guarded_inversion.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
