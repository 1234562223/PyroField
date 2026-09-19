"""Are the state-level inversion's failures optimiser failures or information failures?

`scripts/fusion_level.py` reports a median wind-speed error of 0.7-0.8 % for the
mask + plume inversion, but the distribution is bimodal: when it converges the answer is
essentially exact (p25 = 0.2-0.4 %), and on 16 % of in-distribution and 36 % of
out-of-distribution cases it lands far away. A median alone would hide that, so this
script asks which of two very different things is happening.

  * **Optimiser failure.** The truth fits the data better than the answer found. The
    information is present; Levenberg-Marquardt, started 35 % away, stopped in a local
    minimum. Fixable with multi-start, and not a statement about the sensors.
  * **Information failure.** The answer found fits the data as well as the truth does, or
    better. Then the observations genuinely do not distinguish them, and no optimiser
    would have helped.

The test is direct: recompute the weighted misfit at the recovered parameters and at the
true ones, on exactly the observations the inversion saw. The random draws are replayed in
the same order as :func:`fusion_level.state_level`, so the targets are identical.

Run:  python scripts/fusion_level.py --epochs 200 --lm-cases 25
      python scripts/fusion_failure_mode.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS = ROOT / "data", ROOT / "results"

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
FAIL = 0.05          # a case counts as failed above 5 % error in wind speed


def misfit(theta, target, scen, keys, w):
    return float((((stack_obs(forward(theta, scen), keys) * w)
                   - stack_obs(target, keys) * w) ** 2).sum())


def replay(split, n_cases, device, use_plume, seed=0):
    """Reproduce exactly the targets `fusion_level.state_level` inverted."""
    keys = ("mask", "plume") if use_plume else ("mask",)
    sc = Scenario(device=device, n_steps=110, mask_times=MASK_TIMES,
                  plume_times=PLUME_TIMES, conc_times=())
    y_ = np.load(DATA / f"fusion_{split}.npz")["params"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(y_), size=min(n_cases, len(y_)), replace=False)
    gen = torch.Generator(device=device).manual_seed(seed + 7)

    out = []
    for i in pick:
        th = torch.from_numpy(y_[i]).to(device)
        target = add_noise(forward(th, sc), gen)
        # The same six draws state_level makes for the starting point, so that the
        # generator stays in step even though the start is not needed here.
        for _ in range(len(PARAM_NAMES)):
            rng.uniform(-0.35, 0.35)
        out.append((th, target, sc, keys))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(RESULTS / "fusion_preds.npz")
    report = {}

    print("=" * 78)
    print("Do the failed inversions fit the data better or worse than the truth?")
    print("=" * 78)

    for use_plume, tag, n in ((True, "state mask + plume", 25),
                              (False, "state mask only", 12)):
        for split in ("test_id", "test_ood"):
            key = f"{tag}|{split}"
            if f"{key}__pred" not in z.files:
                continue
            pred, true = z[f"{key}__pred"], z[f"{key}__true"]
            err = np.abs(np.expm1(pred[:, IU] - true[:, IU]))
            cases = replay(split, n, device, use_plume)

            rows = []
            for k, (th, target, sc, keys) in enumerate(cases):
                w = fixed_weights(target, keys,
                                  {kk: noise_sigma_for(target, kk) for kk in keys})
                hat = torch.from_numpy(pred[k]).to(device)
                l_hat, l_true = misfit(hat, target, sc, keys, w), misfit(
                    th, target, sc, keys, w)
                rows.append({"err_U": float(err[k]), "loss_hat": l_hat,
                             "loss_true": l_true, "ratio": l_hat / max(l_true, 1e-30)})

            bad = [r for r in rows if r["err_U"] > FAIL]
            ok = [r for r in rows if r["err_U"] <= FAIL]
            print(f"\n  {tag}, {split}   ({len(ok)} converged, {len(bad)} failed "
                  f"of {len(rows)})")
            if ok:
                print(f"    converged: median misfit / misfit at truth = "
                      f"{np.median([r['ratio'] for r in ok]):.3f}")
            for r in sorted(bad, key=lambda r: -r["err_U"]):
                verdict = ("optimiser: the truth fits better"
                           if r["ratio"] > 1.02 else
                           "information: the wrong answer fits as well")
                print(f"    err {r['err_U']*100:5.1f}%   misfit/truth "
                      f"{r['ratio']:8.3f}   {verdict}")
            if bad:
                n_opt = sum(1 for r in bad if r["ratio"] > 1.02)
                print(f"    -> {n_opt} of {len(bad)} failures are optimiser failures "
                      f"({n_opt/len(bad)*100:.0f} %)")
            report[key] = {"n": len(rows), "n_failed": len(bad),
                           "n_optimiser": sum(1 for r in bad if r["ratio"] > 1.02),
                           "rows": rows}

    (RESULTS / "fusion_failure_mode.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
    print(f"\nwrote {RESULTS/'fusion_failure_mode.json'}")


if __name__ == "__main__":
    main()
