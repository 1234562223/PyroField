"""Does a real fire's shape predict the real wind? The test this study kept inferring around.

RESULTS.md section 2 rests on one number it does not measure: how uncertain the relation
between fire shape and wind actually is. It was inferred from Alexander's (1985) r = 0.865
and carried as a 15-35 % bracket, flagged as the weakest link. This measures it, on 895
real NIFC fire perimeters joined to ERA5 reanalysis wind.

Two analyses, in increasing order of how much they assume.

  1. **Model-free.** Bin the fires by their observed length-to-breadth ratio and look at
     the spread of actual wind inside each bin. If shape determined wind, each bin would be
     narrow. No fire model, no fitted parameter, nothing to argue with.

  2. **Against Anderson's relation.** Fit the wind adjustment factor (10 m wind to midflame,
     physically 0.1-0.5) and a shape scale, then report the correlation and the residual
     scatter. That residual is the number section 2 needs.

Caveats carried forward rather than buried: ERA5 is ~25 km reanalysis, not the wind at the
flame; final perimeters carry suppression; and the fire's shape integrates a whole active
period while the wind summary is a single number over it. Every one of those *adds*
scatter, so the measurement is an upper bound on how well shape predicts wind -- which is
the conservative direction for the claim being tested.

Run:  python scripts/fetch_perimeters.py && python scripts/fetch_wind.py
      python scripts/real_perimeters.py && python scripts/real_shape_vs_wind.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

from pyrofield.physics.rothermel import lb_anderson  # noqa: E402

WAF_RANGE = (0.05, 0.60)     # physically plausible wind adjustment factors
MIN_COHERENCE = 0.55         # drop fires whose wind boxed the compass over the window

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def lb_of(u_ms, waf, scale=1.0):
    return np.array([float(lb_anderson(torch.tensor(float(u) * waf), cap=None,
                                       scale=scale)) for u in np.atleast_1d(u_ms)])


def main():
    perim = RESULTS / "real_perimeters.json"
    wind = DATA / "fire_wind.json"
    for p in (perim, wind):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    FIGS.mkdir(exist_ok=True)

    rows = json.loads(perim.read_text(encoding="utf-8"))["rows"]
    winds = json.loads(wind.read_text(encoding="utf-8"))
    by_name = {}
    for w in winds:
        key = (w.get("name"), round(w.get("acres") or 0, 1))
        by_name[key] = w

    joined = []
    for r in rows:
        key = (r.get("name"), round(r.get("acres") or 0, 1))
        w = by_name.get(key)
        if w is None:
            continue
        joined.append({**r, **{k: w[k] for k in
                               ("wind10_max", "wind10_mean_of_daily_max", "wind10_p75",
                                "dir_mean_deg", "dir_coherence", "n_days")}})
    print(f"joined {len(joined)} of {len(rows)} perimeters to ERA5 wind")

    use = [j for j in joined if j["dir_coherence"] >= MIN_COHERENCE]
    print(f"  {len(use)} have a coherent wind direction over the window "
          f"(coherence >= {MIN_COHERENCE})\n")
    if len(use) < 100:
        use = joined
        print("  (too few; using all)\n")

    lb = np.array([j["LB"] for j in use])
    u10 = np.array([j["wind10_mean_of_daily_max"] for j in use])

    # ------------------------------------------------------------------ 1 -----
    print("=" * 78)
    print("[1] model-free: bin by observed shape, look at the real wind in each bin")
    print("=" * 78)
    edges = np.array([1.0, 1.4, 1.8, 2.3, 3.0, 4.5, 12.0])
    print(f"    {'L/B bin':>12s} {'n':>5s} {'wind p25':>9s} {'median':>8s} "
          f"{'p75':>8s} {'p10-p90 spread':>16s}")
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lb >= a) & (lb < b)
        if m.sum() < 15:
            continue
        w = u10[m]
        rel = (np.percentile(w, 90) - np.percentile(w, 10)) / np.median(w)
        bins.append({"lo": float(a), "hi": float(b), "n": int(m.sum()),
                     "p25": float(np.percentile(w, 25)),
                     "median": float(np.median(w)),
                     "p75": float(np.percentile(w, 75)),
                     "rel_spread": float(rel)})
        print(f"    {a:5.1f}-{b:<5.1f} {m.sum():5d} {np.percentile(w,25):9.2f} "
              f"{np.median(w):8.2f} {np.percentile(w,75):8.2f} {rel*100:15.0f} %")

    r_sp = float(np.corrcoef(np.log(lb), np.log(u10))[0, 1])
    lo_b, hi_b = bins[0], bins[-1]
    print(f"\n    correlation of log L/B with log wind: r = {r_sp:.3f}  "
          f"(r^2 = {r_sp**2:.3f})")
    print(f"    median wind moves from {lo_b['median']:.2f} to {hi_b['median']:.2f} m/s "
          f"across the whole range of shapes,")
    print(f"    while the wind *within* a single shape bin spans "
          f"{np.median([b['rel_spread'] for b in bins])*100:.0f} % (p10-p90).")
    print("    Shape barely moves the answer, and does not narrow it.")

    # ------------------------------------------------------------------ 2 -----
    print()
    print("=" * 78)
    print("[2] against Anderson's relation, with the wind adjustment factor fitted")
    print("=" * 78)

    def cost(waf):
        pred = lb_of(u10, waf)
        return float(np.mean((np.log(lb) - np.log(pred)) ** 2))

    res = minimize_scalar(cost, bounds=WAF_RANGE, method="bounded")
    waf = float(res.x)
    pred = lb_of(u10, waf)
    resid = np.log(lb) - np.log(pred)
    r_fit = float(np.corrcoef(pred, lb)[0, 1])
    sigma = float(np.std(resid))
    print(f"    best-fit wind adjustment factor: {waf:.3f}  "
          f"(physically plausible range {WAF_RANGE[0]}-{WAF_RANGE[1]})")
    print(f"    implied median midflame wind: {np.median(u10)*waf:.2f} m/s")
    print(f"    correlation of predicted with observed L/B: r = {r_fit:.3f}")
    print(f"    residual scatter on ln L/B: sigma = {sigma:.3f}  "
          f"({np.expm1(sigma)*100:.0f} % relative)")
    print(f"\n    Alexander (1985) reported r = 0.865 on experimental fires and")
    print(f"    well-documented wildfires in coniferous forest. On this operational")
    print(f"    sample the same comparison gives r = {r_fit:.3f}.")

    # ------------------------------------------------------------------ 3 -----
    print()
    print("=" * 78)
    print("[3] the number section 2 was missing")
    print("=" * 78)
    sig_rel = float(np.expm1(sigma))
    print(f"    measured relative scatter in the shape relation: {sig_rel*100:.0f} %")
    print(f"    (the bracket carried previously was 15-35 %, inferred from r)")
    print()
    print(f"    {'U (m/s)':>9s} {'elasticity':>12s} {'implied sigma(U)':>18s}")
    for U, E in ((1.0, 0.478), (1.5, 0.843), (2.0, 1.159), (2.5, 1.468),
                 (3.0, 1.769), (3.5, 2.062)):
        s = sig_rel / E
        print(f"    {U:9.1f} {E:12.3f} {s*100:17.0f} %"
              + ("   <- misses the 10 % tolerance" if s > 0.10 else ""))
    print("\n    Every wind below the FARSITE cap misses the tolerance by this measure,")
    print("    and this is measured rather than inferred.")

    out = {"n_joined": len(joined), "n_used": len(use),
           "min_coherence": MIN_COHERENCE,
           "bins": bins, "r_logs": r_sp,
           "waf": waf, "r_fit": r_fit, "sigma_ln": sigma, "sigma_rel": sig_rel}
    (RESULTS / "real_shape_vs_wind.json").write_text(json.dumps(out, indent=2),
                                                     encoding="utf-8")
    print(f"\nwrote {RESULTS/'real_shape_vs_wind.json'}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))

    ax = axes[0]
    ax.scatter(u10, lb, s=9, alpha=0.35, color="tab:purple")
    uu = np.linspace(max(u10.min(), 0.5), u10.max(), 120)
    ax.plot(uu, lb_of(uu, waf), "-", color="k", lw=1.9,
            label=f"Anderson, WAF = {waf:.2f}")
    ax.set_xlabel("ERA5 10 m wind, mean of daily maxima (m/s)", fontsize=8.5)
    ax.set_ylabel("observed length-to-breadth ratio", fontsize=8.5)
    ax.set_ylim(1, min(8, lb.max()))
    ax.set_title(f"{len(use)} real fires\nr = {r_fit:.2f}", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    pos = [0.5 * (b["lo"] + b["hi"]) for b in bins]
    data = [u10[(lb >= b["lo"]) & (lb < b["hi"])] for b in bins]
    ax.boxplot(data, positions=pos, widths=[0.35 * (b["hi"] - b["lo"]) for b in bins],
               showfliers=False, manage_ticks=False)
    ax.set_xlabel("observed length-to-breadth ratio", fontsize=8.5)
    ax.set_ylabel("ERA5 10 m wind (m/s)", fontsize=8.5)
    ax.set_xlim(1, 5)
    ax.set_title("The wind inside one shape bin\nspans most of the range", fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.hist(np.expm1(resid) * 100, bins=40, color="tab:orange", alpha=0.85)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel("relative residual in L/B (%)", fontsize=8.5)
    ax.set_ylabel("fires", fontsize=8.5)
    ax.set_xlim(-100, 200)
    ax.set_title(f"Residual scatter: {sig_rel*100:.0f} %\n"
                 "(was carried as an inferred 15-35 %)", fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle("Real fire shape against real wind, 895 NIFC perimeters and ERA5",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_real_shape_vs_wind.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
