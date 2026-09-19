"""Terrain: where masks lose the one thing they were reliably good for.

On flat ground a burn scar points downwind, and the measurements in RESULTS.md section 5
show masks give wind *direction* to 0.02 degrees even though they cannot give wind speed
at all. That is the one place a mask was trustworthy.

It does not survive a slope. Rothermel and FARSITE combine the wind factor and the slope
factor as **vectors**, and the fire responds only to their sum:

    (phi_eff, theta_eff) = phi_w(U) * u(theta_w)  +  phi_s * u(aspect)

A burn scar therefore points along theta_eff, not theta_w. Reading the wind off a mask on
sloping ground reads the sum and calls it the wind.

Three things are measured here:

  1. the null space of the mask-only Fisher grows from one dimension to two once the slope
     factor is uncertain (the DEM gives the aspect exactly, but the slope *coefficient* is
     fuel-dependent and is not known);
  2. what a flat-ground fit -- which is what mask-space methods implicitly are -- reports
     for the wind when the data came off a slope;
  3. whether the plume cares. It should not: smoke is advected by wind and does not feel
     the ground at all.

Run:  python scripts/terrain.py
"""

from __future__ import annotations

import json
import math
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
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

from pyrofield.eval.inversion import levenberg_marquardt  # noqa: E402
from pyrofield.physics.rothermel import WIND_A, WIND_B, combine_wind_slope  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    noise_sigma_for,
    pack,
    unpack,
)
from pyrofield.theory.identifiability import (  # noqa: E402
    crb_from_fisher,
    stack_obs,
    weight_vector,
)

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": math.pi / 4, "lb_k": 0.35,
         "w_buoy": 3.0, "Q": 1.0}
SLOPE_PHI = 1.0                    # Rothermel slope factor; phi_w at 4 m/s is 2.12
SLOPE_ASPECT = 3 * math.pi / 4     # up-slope 90 degrees off the wind
EXT_NAMES = ["R0", "U", "theta_w", "lb_k", "slope_phi"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def scen(device, slope_phi, aspect=SLOPE_ASPECT, n_steps=110):
    times = tuple(range(8, n_steps, 9))
    return Scenario(device=device, n_steps=n_steps, mask_times=times,
                    plume_times=times, conc_times=times,
                    slope_phi=float(slope_phi), slope_aspect=float(aspect))


def effective_direction(U, theta_w, phi_s, aspect):
    T = lambda v: torch.tensor(float(v))  # noqa: E731
    _, te, _ = combine_wind_slope(T(U), T(theta_w), T(phi_s), T(aspect))
    return float(te)


def ext_forward(theta7, base_scen, device):
    """Forward model over the extended vector (6 parameters + log slope factor)."""
    s = Scenario(**{**base_scen.__dict__, "slope_phi": float(torch.exp(theta7[6]))})
    return forward(theta7[:6], s)


def ext_jacobian(theta7, base_scen, device, keys, eps=0.02):
    base = ext_forward(theta7, base_scen, device)
    cols = []
    for j in range(7):
        tp, tm = theta7.clone(), theta7.clone()
        tp[j] += eps
        tm[j] -= eps
        gp = stack_obs(ext_forward(tp, base_scen, device), keys)
        gm = stack_obs(ext_forward(tm, base_scen, device), keys)
        cols.append((gp - gm) / (2 * eps))
    return torch.stack(cols, dim=1), base


def ext_fisher(theta7, base_scen, device, keys, sub):
    J, base = ext_jacobian(theta7, base_scen, device, keys)
    w = weight_vector(base, keys, theta7.device, theta7.dtype)
    Jw = (J * w[:, None])[:, sub]
    return (Jw.T @ Jw).double().cpu().numpy()


def main():
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    te = effective_direction(TRUTH["U"], TRUTH["theta_w"], SLOPE_PHI, SLOPE_ASPECT)
    phi_w = WIND_A * TRUTH["U"] ** WIND_B
    print("=" * 78)
    print("[0] the geometry")
    print("=" * 78)
    print(f"    wind   : {math.degrees(TRUTH['theta_w']):6.1f} deg, factor phi_w = {phi_w:.3f}")
    print(f"    slope  : {math.degrees(SLOPE_ASPECT):6.1f} deg, factor phi_s = {SLOPE_PHI:.3f}")
    print(f"    the burn scar points along {math.degrees(te):6.1f} deg")
    print(f"    -> reading the wind off the mask is wrong by "
          f"{abs(math.degrees(te - TRUTH['theta_w'])):.1f} deg")
    out["geometry"] = {"theta_w_deg": math.degrees(TRUTH["theta_w"]),
                       "aspect_deg": math.degrees(SLOPE_ASPECT),
                       "phi_w": phi_w, "phi_s": SLOPE_PHI,
                       "theta_eff_deg": math.degrees(te)}

    # ---------------------------------------------------------------- 1 -----
    print()
    print("=" * 78)
    print("[1] the null space grows from one dimension to two")
    print("=" * 78)
    base = scen(device, SLOPE_PHI)
    theta7 = torch.cat([pack(TRUTH, device=device),
                        torch.tensor([math.log(SLOPE_PHI)], device=device)])
    sub = [0, 1, 2, 3, 6]          # R0, U, theta_w, k_LB, slope_phi
    for label, keys in (("masks only", ("mask",)),
                        ("mask + plume", ("mask", "plume"))):
        F = ext_fisher(theta7, base, device, keys, sub)
        crb, cond, ev, evec = crb_from_fisher(F)
        # An absolute "near zero" threshold is useless here: finite-difference error puts
        # a floor under the small eigenvalues (RESULTS.md section 6). What identifies the
        # null space is the *gap* -- the largest jump in the spectrum.
        gaps = ev[1:] / np.maximum(ev[:-1], 1e-300)
        tiny = int(np.argmax(gaps) + 1)
        print(f"\n  {label}")
        print(f"    eigenvalues: {', '.join(f'{v:.3e}' for v in ev)}")
        print(f"    largest spectral gap after index {tiny} "
              f"(factor {gaps.max():.0f}) -> {tiny} weak direction(s)")
        for k in range(min(2, len(ev))):
            v = evec[:, k]
            order = np.argsort(-np.abs(v))[:3]
            terms = "  ".join(f"{v[i]:+.2f}*{EXT_NAMES[i]}" for i in order)
            print(f"      dir {k+1} (lambda = {ev[k]:.3e}): {terms}")
        print(f"    CRB: " + "  ".join(f"{EXT_NAMES[i]}={crb[i]:.4g}"
                                       for i in range(len(EXT_NAMES))))
        out[f"fisher_{label.replace(' ', '_')}"] = {
            "eigvals": ev.tolist(), "crb": crb.tolist(), "n_null": tiny,
            "params": EXT_NAMES,
        }

    # ---------------------------------------------------------------- 2 -----
    print()
    print("=" * 78)
    print("[2] what a flat-ground fit reports when the data came off a slope")
    print("=" * 78)
    print("    (this is what every mask-space method implicitly does)")
    gen = torch.Generator(device=device).manual_seed(23)
    target = add_noise(forward(pack(TRUTH, device=device), base), gen)
    flat = scen(device, 0.0)
    theta_true = pack(TRUTH, device=device)

    fits = {}
    rng = np.random.default_rng(5)
    for label, keys in (("M", ("mask",)), ("M+P", ("mask", "plume"))):
        best = None
        for _ in range(3):
            start = theta_true.clone()
            for i in range(len(PARAM_NAMES)):
                start[i] = start[i] + float(rng.uniform(-0.30, 0.30))
            res = levenberg_marquardt(start, theta_true, target, flat, keys)
            if best is None or res.loss < best.loss:
                best = res
        p = unpack(best.theta_hat)
        d = (float(p["theta_w"]) - TRUTH["theta_w"] + math.pi) % (2 * math.pi) - math.pi
        sig = {k: noise_sigma_for(target, k) for k in keys}
        n = sum(target[k].numel() for k in keys)
        chi2 = sum(float((((forward(best.theta_hat, flat)[k] - target[k]) / sig[k]) ** 2).sum())
                   for k in keys)
        fits[label] = {"theta_w_deg": math.degrees(float(p["theta_w"])),
                       "err_deg": abs(math.degrees(d)),
                       "U": float(p["U"]), "chi2_per_N": chi2 / n}
        print(f"    {label:4s}: wind direction reported "
              f"{math.degrees(float(p['theta_w'])):6.1f} deg "
              f"(truth {math.degrees(TRUTH['theta_w']):.1f}) -> "
              f"error {abs(math.degrees(d)):5.2f} deg     chi2/N = {chi2/n:.4f}")
    out["flat_fit"] = fits
    print(f"\n    the mask-only answer sits within "
          f"{abs(fits['M']['theta_w_deg'] - math.degrees(te)):.1f} deg of the combined")
    print(f"    wind+slope direction ({math.degrees(te):.1f} deg): the mask reported the")
    print("    vector sum and called it the wind, exactly as the algebra says it must.")

    # --------------------------------------------------------------- 2b -----
    # Fitting a flat model is the wrong procedure; the right one is to let the slope
    # factor float. That is only worth doing if the data can support it, which is the
    # question. Small dedicated LM over the extended 7-vector.
    print()
    print("=" * 78)
    print("[2b] the right procedure: let the slope factor float too")
    print("=" * 78)

    def ext_loss(t7, keys, sig):
        pred = ext_forward(t7, base, device)
        return sum(float((((pred[k] - target[k]) / sig[k]) ** 2).sum()) for k in keys)

    theta7_true = torch.cat([theta_true,
                             torch.tensor([math.log(SLOPE_PHI)], device=device)])
    free_fit = {}
    for label, keys in (("M", ("mask",)), ("M+P", ("mask", "plume"))):
        sig = {k: noise_sigma_for(target, k) for k in keys}
        n = sum(target[k].numel() for k in keys)
        rng2 = np.random.default_rng(11)
        best_t, best_l, seeds = None, None, []
        for _ in range(4):
            t7 = theta7_true.clone()
            for i in (0, 1, 2, 3, 6):
                t7[i] = t7[i] + float(rng2.uniform(-0.30, 0.30))
            lam, loss = 1e-2, ext_loss(t7, keys, sig)
            for _ in range(28):
                J, b0 = ext_jacobian(t7, base, device, keys)
                w = weight_vector(b0, keys, device, t7.dtype)
                Jw = (J * w[:, None]).double()[:, [0, 1, 2, 3, 6]]
                rw = ((stack_obs(b0, keys) - stack_obs(target, keys)) * w).double()
                A, g = Jw.T @ Jw, Jw.T @ rw
                dA = torch.diagonal(A)
                D = torch.diag(dA.clamp(min=float(dA.max()) * 1e-8 + 1e-30))
                improved = False
                for _ in range(8):
                    try:
                        step = torch.linalg.solve(A + lam * D, -g)
                    except RuntimeError:
                        lam *= 10
                        continue
                    cand = t7.clone()
                    for si, pi in enumerate([0, 1, 2, 3, 6]):
                        cand[pi] = cand[pi] + step[si].to(t7.dtype)
                    lc = ext_loss(cand, keys, sig)
                    if np.isfinite(lc) and lc < loss:
                        t7, loss, lam, improved = cand, lc, max(lam / 3, 1e-9), True
                        break
                    lam *= 3
                if not improved and lam > 1e6:
                    break
            seeds.append(float(t7[2]))
            if best_l is None or loss < best_l:
                best_t, best_l = t7.clone(), loss
        p = unpack(best_t[:6])
        d = (float(p["theta_w"]) - TRUTH["theta_w"] + math.pi) % (2 * math.pi) - math.pi
        sp = float(torch.exp(best_t[6]))
        spread = math.degrees(max(seeds) - min(seeds))
        free_fit[label] = {"theta_w_deg": math.degrees(float(p["theta_w"])),
                           "err_deg": abs(math.degrees(d)), "slope_phi": sp,
                           "slope_err": abs(sp / SLOPE_PHI - 1),
                           "restart_spread_deg": spread,
                           "chi2_per_N": best_l / n}
        print(f"    {label:4s}: wind {math.degrees(float(p['theta_w'])):6.1f} deg "
              f"(err {abs(math.degrees(d)):5.2f}, restart spread {spread:5.2f} deg)   "
              f"phi_s {sp:5.3f} (truth {SLOPE_PHI}, err {abs(sp/SLOPE_PHI-1)*100:4.1f}%)"
              f"   chi2/N {best_l/n:.4f}")
    out["slope_free_fit"] = free_fit
    print("\n    Letting the slope float removes the bias, but for masks it only converts")
    print("    a confident wrong answer into an unconstrained one: the restart spread is")
    print(f"    {free_fit['M']['restart_spread_deg']:.1f} deg against "
          f"{free_fit['M+P']['restart_spread_deg']:.2f} deg with the plume.")

    # ---------------------------------------------------------------- 3 -----
    print()
    print("=" * 78)
    print("[3] how the error grows with the slope")
    print("=" * 78)
    print(f"    {'phi_s':>7s} {'slope deg':>10s} {'scar dir':>10s} {'naive wind err':>15s}")
    sweep = []
    for phi_s in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        t = effective_direction(TRUTH["U"], TRUTH["theta_w"], phi_s, SLOPE_ASPECT)
        err = abs(math.degrees(t - TRUTH["theta_w"]))
        # Invert the Rothermel slope factor 5.275 * tan(s)^2 for a readable slope angle.
        tan_s = (phi_s / 5.275) ** 0.5 if phi_s > 0 else 0.0
        sweep.append({"phi_s": phi_s, "slope_deg": math.degrees(math.atan(tan_s)),
                      "scar_deg": math.degrees(t), "err_deg": err})
        print(f"    {phi_s:7.2f} {math.degrees(math.atan(tan_s)):10.1f} "
              f"{math.degrees(t):10.1f} {err:14.1f} deg")
    out["sweep"] = sweep

    plot(out)
    (RESULTS / "terrain.json").write_text(json.dumps(
        {"truth": TRUTH, "slope_phi": SLOPE_PHI,
         "slope_aspect": SLOPE_ASPECT, **out}, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS/'terrain.json'}")


def plot(out):
    te = math.radians(out["geometry"]["theta_eff_deg"])
    fits = out["flat_fit"]
    free_fit = out["slope_free_fit"]
    sweep = out["sweep"]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    ev_m = np.array(out["fisher_masks_only"]["eigvals"])
    ev_p = np.array(out["fisher_mask_+_plume"]["eigvals"])
    ax.semilogy(range(1, len(ev_m) + 1), np.maximum(ev_m, 1e-12), "o-",
                color="tab:red", lw=1.9, label="masks only")
    ax.semilogy(range(1, len(ev_p) + 1), np.maximum(ev_p, 1e-12), "^-",
                color="tab:blue", lw=1.9, label="mask + plume")
    ax.set_xlabel("eigenvalue index (ascending)", fontsize=8.5)
    ax.set_ylabel("Fisher eigenvalue", fontsize=8.5)
    ax.set_title("On a slope the mask null space\nis two-dimensional", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    ax.plot([s["phi_s"] for s in sweep], [s["err_deg"] for s in sweep], "o-",
            color="tab:red", lw=1.9, label="mask, flat-ground model")
    # The honest comparison is the correct procedure on both sides: slope left free.
    mk = free_fit["M"]
    ax.errorbar([SLOPE_PHI], [mk["err_deg"]],
                yerr=[[min(mk["err_deg"], mk["restart_spread_deg"] / 2)],
                      [mk["restart_spread_deg"] / 2]],
                fmt="s", color="tab:orange", capsize=4, lw=1.6, ms=6,
                label=f"mask, slope free ({mk['err_deg']:.1f} deg,\n"
                      f"  +/-{mk['restart_spread_deg']/2:.0f} deg over restarts)")
    ax.axhline(free_fit["M+P"]["err_deg"], color="tab:blue", ls="--", lw=1.8,
               label=f"mask + plume, slope free ({free_fit['M+P']['err_deg']:.2f} deg)")
    ax.axvline(SLOPE_PHI, color="0.5", ls=":", lw=1.1)
    ax.set_xlabel("slope factor $\\phi_s$", fontsize=8.5)
    ax.set_ylabel("error in wind direction (deg)", fontsize=8.5)
    ax.set_title("and the scar stops pointing\ndownwind", fontsize=9.5)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[2]
    r = 1.0
    for ang, lab, col, w in ((TRUTH["theta_w"], "wind", "tab:green", 2.8),
                             (SLOPE_ASPECT, "up-slope", "tab:brown", 2.2),
                             (te, "burn scar", "tab:red", 2.8)):
        ax.annotate("", xy=(r * math.cos(ang), r * math.sin(ang)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=w))
        ax.text(1.22 * r * math.cos(ang), 1.22 * r * math.sin(ang), lab,
                ha="center", va="center", fontsize=8.5, color=col)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.25, 1.55)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"wind {math.degrees(TRUTH['theta_w']):.0f} deg, "
                 f"scar {math.degrees(te):.0f} deg\n"
                 f"the mask reports the scar", fontsize=9)

    fig.suptitle("In terrain, a burn scar points along wind + slope — "
                 "so a mask cannot even give the wind direction",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_terrain.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        plot(json.loads((RESULTS / "terrain.json").read_text(encoding="utf-8")))
    else:
        main()
