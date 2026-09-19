"""Which way the hill faces: the variable section 4 did not sweep.

`scripts/terrain.py` measures one slope magnitude at one aspect and sweeps magnitude only.
But aspect relative to the wind is the variable that decides how much direction
information a mask loses, and it has structure worth measuring rather than assuming:

  * **aligned** (0 deg): the two vectors add along one line. The burn scar still points
    downwind, so direction survives -- but the head rate now confounds *three* quantities,
    R0, wind and slope, instead of two.
  * **across** (90 deg): maximum direction error.
  * **opposed** (180 deg): direction is right again by symmetry, unless the slope factor
    exceeds the wind factor, in which case the scar points the *opposite* way and a mask
    reports a wind blowing backwards.

The geometry is closed-form; the identifiability is measured with the mask-only Fisher.

Run:  python scripts/terrain_aspect.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

from pyrofield.physics.rothermel import WIND_A, WIND_B, combine_wind_slope  # noqa: E402
from pyrofield.sim.synthetic import PARAM_NAMES, Scenario, forward, pack  # noqa: E402
from pyrofield.theory.identifiability import (  # noqa: E402
    crb_from_fisher,
    stack_obs,
    weight_vector,
)

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": math.pi / 4, "lb_k": 0.35,
         "w_buoy": 3.0, "Q": 1.0}
SLOPES = (0.5, 1.0, 2.0, 3.0)
DELTAS = (0, 30, 60, 90, 120, 150, 180)
FISHER_SLOPE = 1.0
EXT = ["R0", "U", "theta_w", "lb_k", "slope_phi"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scar_direction(phi_s, delta_deg):
    T = lambda v: torch.tensor(float(v))  # noqa: E731
    aspect = TRUTH["theta_w"] + math.radians(delta_deg)
    _, te, _ = combine_wind_slope(T(TRUTH["U"]), T(TRUTH["theta_w"]), T(phi_s), T(aspect))
    err = (float(te) - TRUTH["theta_w"] + math.pi) % (2 * math.pi) - math.pi
    return float(te), abs(math.degrees(err))


def ext_forward(theta7, base, device):
    s = Scenario(**{**base.__dict__, "slope_phi": float(torch.exp(theta7[6]))})
    return forward(theta7[:6], s)


def ext_fisher(theta7, base, device, keys, eps=0.02):
    b0 = ext_forward(theta7, base, device)
    cols = []
    for j in range(7):
        tp, tm = theta7.clone(), theta7.clone()
        tp[j] += eps
        tm[j] -= eps
        cols.append((stack_obs(ext_forward(tp, base, device), keys)
                     - stack_obs(ext_forward(tm, base, device), keys)) / (2 * eps))
    J = torch.stack(cols, dim=1)
    w = weight_vector(b0, keys, theta7.device, theta7.dtype)
    Jw = (J * w[:, None])[:, [0, 1, 2, 3, 6]]
    return (Jw.T @ Jw).double().cpu().numpy()


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phi_w = WIND_A * TRUTH["U"] ** WIND_B
    out = {"phi_w": phi_w}

    print("=" * 78)
    print("[1] where the burn scar points, as the hill turns")
    print("=" * 78)
    print(f"    wind at {math.degrees(TRUTH['theta_w']):.0f} deg, wind factor "
          f"phi_w = {phi_w:.3f}\n")
    print(f"    {'aspect - wind':>14s} | " + " ".join(f"{'phi_s=%.1f' % s:>12s}"
                                                     for s in SLOPES))
    geo = []
    for d in DELTAS:
        row = {"delta_deg": d, "err": {}}
        line = f"    {d:11d} deg | "
        for s in SLOPES:
            _, err = scar_direction(s, d)
            row["err"][str(s)] = err
            flip = " *" if (s > phi_w and d > 120) else ""
            line += f"{err:10.1f}{flip:>2s} "
        geo.append(row)
        print(line)
    print(f"\n    * = slope factor exceeds the wind factor ({phi_w:.2f}) and the scar has")
    print("      flipped: a mask there reports a wind blowing the other way.")
    out["geometry"] = geo

    worst = max(r["err"][str(s)] for r in geo for s in SLOPES)
    print(f"    worst direction error anywhere in this sweep: {worst:.1f} deg")

    # ---------------------------------------------------------------- 2 -----
    print()
    print("=" * 78)
    print(f"[2] identifiability vs aspect (mask only, slope factor {FISHER_SLOPE} free)")
    print("=" * 78)
    print(f"    {'aspect - wind':>14s} {'scar err':>10s} {'CRB(theta_w)':>13s} "
          f"{'CRB(U)':>10s} {'CRB(R0)':>10s} {'gap':>8s}")
    times = tuple(range(8, 110, 9))
    fis = []
    for d in DELTAS:
        aspect = TRUTH["theta_w"] + math.radians(d)
        base = Scenario(device=device, n_steps=110, mask_times=times,
                        plume_times=times, conc_times=times,
                        slope_phi=FISHER_SLOPE, slope_aspect=float(aspect))
        th7 = torch.cat([pack(TRUTH, device=device),
                         torch.tensor([math.log(FISHER_SLOPE)], device=device)])
        F = ext_fisher(th7, base, device, ("mask",))
        crb, _, ev, _ = crb_from_fisher(F)
        gaps = ev[1:] / np.maximum(ev[:-1], 1e-300)
        _, err = scar_direction(FISHER_SLOPE, d)
        fis.append({"delta_deg": d, "scar_err_deg": err,
                    "crb": {EXT[i]: float(crb[i]) for i in range(len(EXT))},
                    "eigvals": ev.tolist(), "max_gap": float(gaps.max()),
                    "n_weak": int(np.argmax(gaps) + 1)})
        print(f"    {d:11d} deg {err:9.1f}  {crb[2]:13.4g} {crb[1]:10.4g} "
              f"{crb[0]:10.4g} {gaps.max():8.0f}")
    out["fisher"] = fis

    aligned = next(r for r in fis if r["delta_deg"] == 0)
    across = next(r for r in fis if r["delta_deg"] == 90)
    print(f"\n    aligned (0 deg):  scar points downwind, direction error "
          f"{aligned['scar_err_deg']:.1f} deg -- but CRB(U) = {aligned['crb']['U']:.3g}")
    print(f"      the head rate now confounds three quantities instead of two, so the")
    print("      speed is no better off for the direction being right.")
    print(f"    across  (90 deg): direction error {across['scar_err_deg']:.1f} deg, "
          f"CRB(theta_w) = {across['crb']['theta_w']:.3g} rad "
          f"({math.degrees(across['crb']['theta_w']):.2f} deg)")
    ratio = across["crb"]["theta_w"] / max(aligned["crb"]["theta_w"], 1e-30)
    print(f"      direction is {ratio:.0f}x less determined across the wind than along it.")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    for s, c in zip(SLOPES, plt.cm.copper(np.linspace(0.2, 0.85, len(SLOPES)))):
        ax.plot(DELTAS, [r["err"][str(s)] for r in geo], "o-", color=c, lw=1.8,
                ms=4, label=f"$\\phi_s$ = {s}")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(DELTAS)
    ax.set_xlabel("slope aspect minus wind direction (deg)", fontsize=8.5)
    ax.set_ylabel("error in wind direction read\noff the burn scar (deg)", fontsize=8.5)
    ax.set_title("Aspect decides how much\ndirection is lost", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.semilogy(DELTAS, [math.degrees(r["crb"]["theta_w"]) for r in fis], "o-",
                color="tab:red", lw=1.9, label="wind direction (deg)")
    ax.semilogy(DELTAS, [r["crb"]["U"] * 100 for r in fis], "s-",
                color="tab:orange", lw=1.9, label="wind speed (%)")
    ax.set_xticks(DELTAS)
    ax.set_xlabel("slope aspect minus wind direction (deg)", fontsize=8.5)
    ax.set_ylabel("linearised bound (mask only)", fontsize=8.5)
    ax.set_title(f"Identifiability at $\\phi_s$ = {FISHER_SLOPE}", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    r = 1.0
    for ang, lab, col, w in ((TRUTH["theta_w"], "wind", "tab:green", 2.8),
                             (TRUTH["theta_w"] + math.pi / 2, "up-slope (90 deg)",
                              "tab:brown", 2.0),
                             (scar_direction(FISHER_SLOPE, 90)[0], "scar", "tab:red", 2.8)):
        ax.annotate("", xy=(r * math.cos(ang), r * math.sin(ang)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=w))
        ax.text(1.22 * r * math.cos(ang), 1.22 * r * math.sin(ang), lab,
                ha="center", va="center", fontsize=8, color=col)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.25, 1.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("the worst case, 90 deg apart", fontsize=9.5)

    fig.suptitle("Terrain aspect: the variable that decides how much a mask loses",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_terrain_aspect.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")

    (RESULTS / "terrain_aspect.json").write_text(json.dumps(
        {"truth": TRUTH, "slopes": list(SLOPES), "deltas": list(DELTAS),
         "fisher_slope": FISHER_SLOPE, **out}, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'terrain_aspect.json'}")


if __name__ == "__main__":
    main()
