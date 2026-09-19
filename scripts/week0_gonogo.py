"""Week-0 Go/No-Go: does the physics-state formulation actually buy identifiability?

This is the experiment the whole project is staked on. On a synthetic fire whose
ground truth is known exactly, it asks what each sensor subset determines.

The propositions below are stated as the measurements found them, not as they were first
guessed. Three guesses did not survive contact with the data and were corrected:

  * Masks turned out to determine wind *direction* very well (the burn ellipse points
    downwind), so the claim is specifically about wind *speed*.
  * Mask-only identifiability is not a fixed property; it depends on how far the fire
    has run, and the degeneracy is worst exactly in the early window where intervention
    is still possible.
  * The air-quality sensor does not supply emission strength unconditionally. It does so
    only when the plume image saturates. That first measurement was taken on a renderer
    running at optical depth ~3200; once the extinction was calibrated to tau ~ 3 the
    camera determines Q by itself, and the sensor's marginal value on Q disappears.

  P1  mask only    -> R0, U and k_LB collapse onto one confounded direction, severe
                      in the early fire; wind direction is nonetheless well determined
  P2  + plume      -> smoke is advected by wind independently of fuel, so the plume
                      supplies wind information that does not wait for the burn to grow
  P3  + air-quality-> supplies emission strength Q only when the camera cannot, i.e.
                      when the image saturates -- a conditional, not a general, claim
  P4  lead time    -> the practical consequence, in minutes

A caveat on this script's standing: every Fisher number it reports is a linearised
bound, and RESULTS.md section 5 shows those bounds are unreliable for exactly the
mask-only case P1 is about. The propositions are established far more rigorously by
scripts/null_direction.py (exact), scripts/confound_trajectory.py (exact) and the
profile-likelihood scripts (derivative-free). This remains useful as a fast screen and
as the source of the inversion results, not as the primary evidence.

Run:   python scripts/week0_gonogo.py            (full)
       python scripts/week0_gonogo.py --quick    (fast smoke run)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrofield.eval.inversion import levenberg_marquardt, random_start  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    OBS_SETS,
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    pack,
)
from pyrofield.theory.identifiability import (  # noqa: E402
    analyze,
    check_jacobian_against_autograd,
)

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}
RESULTS = ROOT / "results"


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def step_jacobian_check(theta, scen):
    banner("[0] Jacobian sanity: finite differences vs autograd")
    rel = check_jacobian_against_autograd(theta, scen, ("mask", "plume", "conc"))
    for k, v in rel.items():
        print(f"    {k:8s} relative disagreement = {v*100:6.2f} %")
    worst = max(rel.values())
    print(f"    worst = {worst*100:.2f} %   [target < 15 % on a kinked, limited scheme]")
    return rel


# The four parameters that govern the fire front itself. Restricting the Fisher
# analysis to this block answers "do masks confound fuel with wind speed?" without
# the smoke-only parameters (Q, w_buoy), which no mask can see, swamping the null
# space and hiding the confound of interest.
FIRE_BLOCK = ["R0", "U", "theta_w", "lb_k"]


def step_early_vs_full(theta, scen, device):
    """The same analysis at an early and a late observation window."""
    banner("[1a] Identifiability vs how far the fire has run (fire block)")
    from pyrofield.sim.synthetic import Scenario as _S
    from pyrofield.theory.identifiability import crb_from_fisher, fisher_all_subsets

    windows = {}
    for label, T in (("early  (9.0 min)", 36), ("full  (27.5 min)", scen.n_steps)):
        times = tuple(t for t in range(8, T, 9)) or (T - 1,)
        s = _S(device=device, n_steps=T, mask_times=times,
               plume_times=times, conc_times=times)
        Fs = fisher_all_subsets(theta, s, eps=0.02, method="central",
                                params=FIRE_BLOCK)
        burn = float((forward(theta, s)["mask"][-1] > 0.5).float().mean()) * 100
        row = {}
        for keys in (("mask",), ("mask", "plume"), ("mask", "conc"),
                     ("mask", "plume", "conc")):
            crb, cond, ev, _ = crb_from_fisher(Fs[keys])
            row["+".join(k[0].upper() for k in keys)] = {
                "crb": {p: float(crb[i]) for i, p in enumerate(FIRE_BLOCK)},
                "cond": cond, "lam_min": float(ev[0]),
            }
        windows[label] = {"burn_pct": burn, "subsets": row}
        print(f"\n  {label}   burn = {burn:.2f}% of domain")
        print(f"    {'subset':8s}" + "".join(f"{p:>12s}" for p in FIRE_BLOCK))
        for k, v in row.items():
            print(f"    {k:8s}" + "".join(f"{v['crb'][p]:12.4g}" for p in FIRE_BLOCK))
    e = windows["early  (9.0 min)"]["subsets"]
    f = windows["full  (27.5 min)"]["subsets"]
    print(f"\n    mask-only CRB(U):  early {e['M']['crb']['U']:.3g}  ->  "
          f"full {f['M']['crb']['U']:.3g}   "
          f"({e['M']['crb']['U']/max(f['M']['crb']['U'],1e-30):.0f}x worse early)")
    print(f"    mask+plume CRB(U): early {e['M+P']['crb']['U']:.3g}  ->  "
          f"full {f['M+P']['crb']['U']:.3g}   (essentially flat in time)")
    return windows


def step_saturation(theta, scen, device):
    """Is the air-quality sensor's value conditional on the camera's dynamic range?

    The first version of this study credited a ground concentration monitor with
    supplying the emission strength Q that the plume image could not. That was measured
    on a renderer whose plume core sat at optical depth ~3200 -- a pure silhouette, which
    carries the plume's geometry but nothing about how much smoke is in it. After the
    extinction was calibrated to tau ~ 3, the camera determines Q on its own.

    So the corrected claim is conditional, and this measures both sides of it.
    """
    from pyrofield.sim.synthetic import Scenario as _S
    from pyrofield.theory.identifiability import crb_from_fisher, fisher_all_subsets

    banner("[1b] The air-quality sensor's value vs the camera's dynamic range")
    iq = PARAM_NAMES.index("Q")
    out = {}
    for label, ext in (("calibrated  (tau ~ 3)", 1.0e-3), ("saturated  (tau ~ 3200)", 1.0)):
        s = _S(device=device, n_steps=scen.n_steps, mask_times=scen.mask_times,
               plume_times=scen.plume_times, conc_times=scen.conc_times,
               plume_extinction=ext)
        Fs = fisher_all_subsets(theta, s, eps=0.02, method="central")
        mp = float(crb_from_fisher(Fs[("mask", "plume")])[0][iq])
        mpc = float(crb_from_fisher(Fs[("mask", "plume", "conc")])[0][iq])
        img = forward(theta, s)["plume"]
        frac_sat = float((img > 0.90).float().mean())
        out[label] = {"extinction": ext, "crb_Q_MP": mp, "crb_Q_MPC": mpc,
                      "gain": mp / max(mpc, 1e-30), "frac_saturated": frac_sat}
        print(f"  {label}: {frac_sat*100:.0f}% of pixels near saturation")
        print(f"     CRB(Q) from mask+plume        = {mp:.5f}")
        print(f"     CRB(Q) adding the point sensor = {mpc:.5f}"
              f"   ({mp/max(mpc,1e-30):.1f}x better)")
    return out


def step_fisher(theta, scen):
    banner("[1] Fisher information per sensor subset (Cramer-Rao bounds)")
    reports, reports_fire = {}, {}
    for name, keys in OBS_SETS.items():
        t0 = time.time()
        rep = analyze(theta, scen, name, keys)
        rep_f = analyze(theta, scen, name, keys, params=FIRE_BLOCK)
        reports[name], reports_fire[name] = rep, rep_f
        print(f"\n  {name}   ({rep.n_obs} observations, {time.time()-t0:.1f}s)")
        print(f"    eigenvalues (asc): {', '.join(f'{v:.3e}' for v in rep.eigvals)}")
        print(f"    condition number : {rep.cond:.3e}")
        print(f"    weakest direction (all 6 params) : {rep.describe_weakest()}")
        print(f"    weakest direction (fire block)   : {rep_f.describe_weakest()}"
              f"   [lambda_min = {rep_f.eigvals[0]:.3e}, cond = {rep_f.cond:.2e}]")
        print("    Cramer-Rao sigma (fractional for R0/U/k_LB/w_b/Q, rad for theta_w):")
        for i, p in enumerate(PARAM_NAMES):
            s = rep.crb_std[i]
            shown = f"{s:9.3f}" if s < 1e3 else f"{s:9.2e}"
            flag = "  UNIDENTIFIABLE" if s > 1.0 else ("  weak" if s > 0.2 else "")
            print(f"      {p:8s} {shown}{flag}")
    return reports, reports_fire


def build_schedule(scale: float) -> tuple[tuple[float, int], ...]:
    """Scale the coarse-to-fine ladder; ``scale=1`` is the full budget."""
    base = ((4.0, 8), (2.0, 6), (1.0, 5), (0.0, 14))
    return tuple((s, max(2, int(round(n * scale)))) for s, n in base)


def step_inversion(theta_true, scen, n_restarts, schedule, seed=0):
    banner(f"[2] Actual inversion: LM from {n_restarts} random starts per subset")
    print(f"    coarse-to-fine ladder (plume blur sigma, iters): {schedule}")
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=scen.device).manual_seed(seed)

    truth_obs = forward(theta_true, scen)
    target = add_noise(truth_obs, gen)

    out = {}
    for name, keys in OBS_SETS.items():
        print(f"\n  {name}")
        per_param = {p: [] for p in PARAM_NAMES}
        losses, best = [], None
        for r in range(n_restarts):
            init = random_start(theta_true, rng)
            t0 = time.time()
            res = levenberg_marquardt(
                init, theta_true, target, scen, keys, schedule=schedule
            )
            errs = res.errors()
            for p in PARAM_NAMES:
                per_param[p].append(errs[p])
            losses.append(res.loss)
            if best is None or res.loss < best[0]:
                best = (res.loss, errs)
            print(
                f"    start {r}: loss {res.history[0]:.2e} -> {res.loss:.2e} "
                f"({res.accepted} accepted / {res.n_iter} it, {time.time()-t0:.0f}s)   "
                + "  ".join(
                    f"{p}={errs[p]*100:.0f}%" if p != "theta_w" else f"{p}={errs[p]:.1f}deg"
                    for p in ("R0", "U", "theta_w", "Q")
                )
            )
        # The estimator is the lowest-misfit solution -- that is the MLE, and it is
        # what the Cramer-Rao bound describes. Spread across restarts is reported
        # separately so optimiser robustness is not confused with identifiability.
        best_errs = best[1]
        print(f"    MLE (best of {n_restarts}, loss {best[0]:.3e}):")
        for p in PARAM_NAMES:
            v = best_errs[p]
            unit = "deg" if p == "theta_w" else "%"
            val = v if p == "theta_w" else v * 100
            spread = float(np.std(per_param[p]))
            sp = spread if p == "theta_w" else spread * 100
            print(f"      {p:8s} {val:9.2f} {unit}   (restart sd {sp:.2f})")
        out[name] = {
            "per_restart": {p: [float(v) for v in per_param[p]] for p in PARAM_NAMES},
            "best": {p: float(best_errs[p]) for p in PARAM_NAMES},
            "losses": [float(x) for x in losses],
        }
    return out


def verdict(reports, reports_fire, inversion, windows, saturation):
    banner("[3] VERDICT")
    names = list(OBS_SETS.keys())
    m, mp, mc, mpc = names

    def err(setname, p):
        return float(inversion[setname]["best"][p])

    checks = []

    # --- P1 -------------------------------------------------------------------
    # Masks must collapse R0, U and k_LB onto one direction, and that degeneracy
    # must be severe in the early fire. Wind *direction* is a separate matter: the
    # burn ellipse points downwind, so masks get it, and we say so rather than
    # overclaiming.
    rf = reports_fire[m]
    w = np.abs(rf.weakest_direction())
    i_r0 = rf.param_names.index("R0")
    i_u = rf.param_names.index("U")
    i_lb = rf.param_names.index("lb_k")
    load = w[i_r0] + w[i_u] + w[i_lb]
    early = windows["early  (9.0 min)"]["subsets"]["M"]["crb"]["U"]
    full = windows["full  (27.5 min)"]["subsets"]["M"]["crb"]["U"]

    p1_dir = load > 0.8
    p1_early = early > 0.10
    p1_time = (early / max(full, 1e-30)) > 10.0
    p1_inv = err(m, "U") > 0.15
    p1_theta = err(m, "theta_w") < 2.0
    print("  P1 masks confound fuel with wind SPEED (but do give wind DIRECTION)")
    print(f"     weak direction is the R0/U/k_LB trade-off (load {load:.2f}): "
          f"{rf.describe_weakest()}   -> {'OK' if p1_dir else 'NO'}")
    print(f"     early-window CRB(U) = {early:.3g} (above the 10% tolerance)"
          f"   -> {'OK' if p1_early else 'NO'}")
    print(f"     degeneracy is a function of burn development: "
          f"{early/max(full,1e-30):.0f}x worse at 9 min than at 27.5 min"
          f"   -> {'OK' if p1_time else 'NO'}")
    print(f"     MLE error on U = {err(m,'U')*100:.0f}% while theta_w is only "
          f"{err(m,'theta_w'):.2f} deg   -> {'OK' if (p1_inv and p1_theta) else 'NO'}")
    checks.append(p1_dir and p1_early and p1_time and p1_inv and p1_theta)

    # --- P2 -------------------------------------------------------------------
    rf2 = reports_fire[mp]
    gain_crb = rf.crb_of("U") / max(rf2.crb_of("U"), 1e-12)
    gain_inv = err(m, "U") / max(err(mp, "U"), 1e-9)
    e_mp = windows["early  (9.0 min)"]["subsets"]["M+P"]["crb"]["U"]
    gain_early = early / max(e_mp, 1e-30)
    p2 = gain_crb > 5.0 and err(mp, "U") < 0.15 and e_mp < 0.10
    print("\n  P2 the plume image breaks that confound (smoke is advected by wind,")
    print("     independently of fuel)")
    print(f"     CRB(U) at full window: {rf.crb_of('U'):.4f} -> {rf2.crb_of('U'):.5f}"
          f"   ({gain_crb:.0f}x)")
    print(f"     CRB(U) at 9 min:       {early:.4g} -> {e_mp:.5f}"
          f"   ({gain_early:.0f}x)  <- the window that matters")
    print(f"     MLE error on U: {err(m,'U')*100:.0f}% -> {err(mp,'U')*100:.1f}%  "
          f"({gain_inv:.1f}x)   -> {'OK' if p2 else 'NO'}")
    checks.append(p2)

    # --- P3 (revised after the extinction calibration) -------------------------
    cal = saturation["calibrated  (tau ~ 3)"]
    sat = saturation["saturated  (tau ~ 3200)"]
    p3_sat = sat["gain"] > 2.0
    p3_cal = cal["gain"] < 1.5
    p3 = p3_sat and p3_cal
    print("\n  P3 the air-quality point supplies emission strength Q only when the")
    print("     camera cannot -- its value is conditional on the image not saturating")
    print(f"     saturated image  ({sat['frac_saturated']*100:.0f}% of pixels at the ceiling):"
          f" adding the sensor improves CRB(Q) {sat['gain']:.1f}x"
          f"   -> {'OK' if p3_sat else 'NO'}")
    print(f"     calibrated image ({cal['frac_saturated']*100:.0f}% at the ceiling):"
          f" adding the sensor improves CRB(Q) {cal['gain']:.1f}x"
          f"   -> {'OK' if p3_cal else 'NO'}")
    print(f"     (MLE error on Q with mask+plume alone: {err(mp,'Q')*100:.2f}%)")
    checks.append(p3)

    # --- P4 -------------------------------------------------------------------
    lt = RESULTS / "lead_time.json"
    if lt.exists():
        d = json.loads(lt.read_text(encoding="utf-8"))
        gains = [v["gain_min"] for v in d["lead"].values() if v["gain_min"] is not None]
        p4 = bool(gains) and min(gains) >= 10.0
        print("\n  P4 the practical consequence, in minutes")
        for U, v in d["lead"].items():
            c = v["crossings"]
            print(f"     wind {float(U):.0f} m/s: U known to 10% at "
                  f"{('%.1f min' % c['M']) if c['M'] else 'never (>27 min)':>16s} with masks, "
                  f"{c['M+P']:.1f} min with the plume  "
                  f"-> {v['gain_min']:.1f} min gained")
        print(f"     -> {'OK' if p4 else 'NO'}")
        checks.append(p4)
    else:
        print("\n  P4 skipped -- run scripts/lead_time.py first")

    # Context: what the mask-only baseline cannot see at all.
    print(f"\n  (context) mask-only CRB on smoke terms: "
          f"Q={reports[m].crb_of('Q'):.1e}  w_buoy={reports[m].crb_of('w_buoy'):.1e}"
          "  -- masks carry no information about either")

    ok = all(checks)
    print("\n  " + ("*** GO ***  the state-space formulation is doing real work"
                    if ok else
                    "*** NO-GO ***  revisit the formulation before scaling up"))
    return ok, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--budget", type=float, default=1.0,
                    help="scale on the coarse-to-fine iteration ladder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scen = Scenario(device=device)
    if args.quick:
        scen.n_steps = 40
        scen.mask_times = (13, 27, 39)
        scen.plume_times = (39,)
        scen.conc_times = (13, 27, 39)
        args.restarts, args.budget = 1, 0.25

    theta = pack(TRUTH, device=device)
    print(f"device={device}  truth={TRUTH}")
    print(f"scenario: {scen.nx}x{scen.ny} @ {scen.dx}m, {scen.n_steps} steps x {scen.dt}s "
          f"= {scen.n_steps*scen.dt/60:.0f} min")

    t_start = time.time()
    jac = step_jacobian_check(theta, scen)
    windows = step_early_vs_full(theta, scen, device)
    saturation = step_saturation(theta, scen, device)
    reports, reports_fire = step_fisher(theta, scen)
    schedule = build_schedule(args.budget)
    inv = step_inversion(theta, scen, args.restarts, schedule)
    ok, checks = verdict(reports, reports_fire, inv, windows, saturation)
    print(f"\n  total wall time: {(time.time()-t_start)/60:.1f} min")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "truth": TRUTH,
        "quick": args.quick,
        "schedule": [list(s) for s in schedule],
        "restarts": args.restarts,
        "jacobian_check": jac,
        "fisher": {
            name: {
                "eigvals": rep.eigvals.tolist(),
                "eigvecs": rep.eigvecs.tolist(),
                "crb_std": rep.crb_std.tolist(),
                "cond": rep.cond,
                "n_obs": rep.n_obs,
                "weakest": rep.describe_weakest(),
            }
            for name, rep in reports.items()
        },
        "fisher_fire_block": {
            name: {
                "params": rep.param_names,
                "eigvals": rep.eigvals.tolist(),
                "eigvecs": rep.eigvecs.tolist(),
                "crb_std": rep.crb_std.tolist(),
                "cond": rep.cond,
                "weakest": rep.describe_weakest(),
            }
            for name, rep in reports_fire.items()
        },
        "inversion": inv,
        "windows": windows,
        "saturation": saturation,
        "verdict": {"go": bool(ok),
                    **{f"P{i+1}": bool(c) for i, c in enumerate(checks)}},
        "param_names": PARAM_NAMES,
    }
    out = RESULTS / ("week0_quick.json" if args.quick else "week0.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

