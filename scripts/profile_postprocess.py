"""Turn a profile-likelihood scan into a 1-sigma number, correctly at both extremes.

Reading sigma off a scan by asking "which grid points have Delta chi^2 <= 1" breaks at
both ends. If the profile is flat, every point qualifies and sigma is reported as the
scan half-width when the truth is "wider than the scan". If the profile is steep, no
point except the minimum qualifies and sigma comes out as exactly zero -- which is what
the first M+P+C run reported, purely because the finest grid offset was 8 % and the
misfit had already risen by 5e4 there.

Both are handled here by using the shape of the curve rather than its intersection with
a grid: near the optimum a profile behaves as

    Delta chi^2 = (relative error / sigma)^2

so each scan point gives an estimate sigma = error / sqrt(Delta chi^2), and points in a
usable range are combined. A profile that never rises above 1 yields a lower bound
instead of an estimate.

Run:  python scripts/profile_postprocess.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sigma_from_profile(offsets, dchi2, param="U"):
    """Estimate the 1-sigma half-width, or a lower bound if the profile is flat."""
    offs = np.asarray(offsets, float)
    d = np.asarray(dchi2, float)
    err = np.abs(np.degrees(offs)) if param == "theta_w" else np.abs(np.expm1(offs))

    # A profile that never reaches Delta chi^2 = 4 (two sigma) anywhere in the scan has
    # not actually been resolved: the small rises it does show are within optimiser
    # noise, and fitting a parabola to them would dress that noise up as a measurement.
    # In that case the only defensible output is a lower bound at the scan edge.
    usable = (d > 1e-3) & (err > 0)
    if not usable.any() or d.max() < 4.0:
        return {"sigma": float(err.max()), "is_lower_bound": True, "n_used": 0,
                "max_dchi2": float(d.max())}

    est = err[usable] / np.sqrt(d[usable])
    # Points closest to Delta chi^2 = 1 are the least affected by higher-order terms.
    w = 1.0 / (1.0 + np.abs(np.log10(d[usable])))
    sigma = float(np.sum(w * est) / np.sum(w))
    return {"sigma": sigma, "is_lower_bound": False, "n_used": int(usable.sum()),
            "spread": float(np.std(est) / max(np.mean(est), 1e-30))}


def main():
    param_arg = sys.argv[1] if len(sys.argv) > 1 else "U"
    candidates = [RESULTS / f"profile_{param_arg}.json", RESULTS / "profile_likelihood.json"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise SystemExit("no profile scan found -- run scripts/profile_likelihood.py first")
    print(f"reading {src.name}")
    d = json.loads(src.read_text(encoding="utf-8"))
    param = d.get("param", "U")
    unit = "deg" if param == "theta_w" else "%"
    scale = 1.0 if param == "theta_w" else 100.0

    print("=" * 84)
    print(f"Profile-likelihood uncertainty on {param}, and what each Fisher estimator claimed")
    print("=" * 84)
    print(f"  {'set':10s}{'profile 1-sigma':>20s}{'AD Fisher':>14s}{'FD Fisher':>14s}"
          f"{'AD error':>11s}{'FD error':>11s}")

    summary = {}
    for label, prof in d["profiles"].items():
        s = sigma_from_profile(prof["offsets"], prof["dchi2"], param)
        pred = dict(d["fisher_pred"][label])
        # Cramér–Rao values come back in the parameter's own units (radians for the wind
        # direction), while sigma_from_profile reports degrees. Put them on one scale
        # before comparing, or the wind-direction row silently compares radians to degrees.
        if param == "theta_w":
            pred = {k: float(np.degrees(v)) for k, v in pred.items()}
        tag = ">" if s["is_lower_bound"] else " "
        ad_x = pred["ad"] / max(s["sigma"], 1e-30)
        fd_x = pred["fd"] / max(s["sigma"], 1e-30)
        print(f"  {label:10s}{tag}{s['sigma']*scale:15.4g} {unit}"
              f"{pred['ad']*scale:12.4g} {unit}{pred['fd']*scale:12.4g} {unit}"
              f"{ad_x:10.1f}x{fd_x:10.1f}x")
        summary[label] = {**s, "ad": pred["ad"], "fd": pred["fd"],
                          "ad_ratio": float(ad_x), "fd_ratio": float(fd_x)}

    print()
    print("  The 'error' columns are each Fisher estimate divided by the likelihood answer.")
    print("  A Cramér–Rao value is a *lower* bound, so a ratio below 1 is mathematically")
    print("  legitimate -- but a ratio far below 1 means the bound is loose to the point of")
    print("  being misleading if it is read, as it usually is, as the achievable accuracy.")
    print("  A ratio above 1 is worse than loose: it is not a valid bound at all, and")
    print("  indicates the computed Fisher information is too small.")
    for label, s in summary.items():
        if s["ad_ratio"] > 1.5:
            print(f"    - {label}: the AD bound EXCEEDS the measured uncertainty "
                  f"({s['ad_ratio']:.0f}x), so it is not a valid bound. The minmod limiter")
            print("      zeroes derivative paths, making the computed information too small.")
        if 0 < s["fd_ratio"] < 0.2:
            print(f"    - {label}: the FD bound is valid but loose by "
                  f"{1/s['fd_ratio']:.0f}x -- the degeneracy is curved, and a Fisher")
            print("      matrix only ever sees local curvature.")

    labels = list(summary)
    if len(labels) >= 2:
        a, b = labels[0], labels[-1]
        ratio = summary[a]["sigma"] / max(summary[b]["sigma"], 1e-30)
        arrow = "at least " if summary[a]["is_lower_bound"] else ""
        print(f"\n  {a} vs {b} on {param}: {arrow}{ratio:.0f}x better with the full sensor set")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(summary), figsize=(4.9 * len(summary), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (label, prof) in zip(axes, d["profiles"].items()):
        offs = np.array(prof["offsets"])
        x = np.degrees(offs) if param == "theta_w" else (np.exp(offs) - 1) * 100
        dd = np.array(prof["dchi2"])
        ax.plot(x, np.maximum(dd, 1e-3), "o-", lw=1.8, ms=4, color="tab:purple")
        ax.axhline(1.0, color="0.35", ls="--", lw=1.0)
        s = summary[label]
        if not s["is_lower_bound"]:
            xx = np.linspace(x.min(), x.max(), 200)
            frac = np.abs(xx) / 100 if param != "theta_w" else np.abs(xx)
            ax.plot(xx, np.maximum((frac / s["sigma"]) ** 2, 1e-3), "-",
                    color="0.6", lw=1.2, label=r"$(\delta/\sigma)^2$ fit")
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, max(1e2, dd.max() * 3))
        ax.set_xlabel(f"error imposed on {param} ({unit})", fontsize=8.5)
        lbl = (f"$\\sigma > {s['sigma']*scale:.0f}$ {unit} (scan-limited)"
               if s["is_lower_bound"] else f"$\\sigma = {s['sigma']*scale:.3g}$ {unit}")
        ax.set_title(f"{label}   {lbl}", fontsize=9.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"profile $\Delta\chi^2$", fontsize=8.5)
    fig.suptitle(f"How hard does the data push back on a wrong {param}?", fontsize=10, y=1.03)
    fig.tight_layout()
    out = FIGS / f"fig_profile_{param}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    dest = RESULTS / f"profile_summary_{param}.json"
    dest.write_text(json.dumps({"param": param, "summary": summary}, indent=2),
                    encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
