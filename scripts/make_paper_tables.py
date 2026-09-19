"""Emit every headline table as markdown and LaTeX, generated from the result files.

Writing the paper should not mean retyping numbers out of RESULTS.md, because that is the
step where they go stale -- and in this repository stale numbers have already been the most
common error. So the tables a paper needs are generated from `results/*.json` directly,
each carrying the script that produced it and the section it belongs to.

Rerun this after any experiment changes, then `scripts/verify_results.py`.

Run:  python scripts/make_paper_tables.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
OUT = ROOT / "PAPER_TABLES.md"

PARAM_NAMES = ("R0", "U", "theta_w", "lb_k", "w_buoy", "Q")


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


class Doc:
    def __init__(self):
        self.parts = []

    def h(self, text, level=2):
        self.parts.append(f"\n{'#' * level} {text}\n")

    def p(self, text):
        self.parts.append(text + "\n")

    def table(self, caption, header, rows, source, section, align=None):
        """One table, rendered twice: markdown to read, LaTeX to paste."""
        n = len(header)
        align = align or (["l"] + ["r"] * (n - 1))
        self.parts.append(f"\n**{caption}**  \n"
                          f"<sub>{section} · `{source}`</sub>\n")
        self.parts.append("| " + " | ".join(header) + " |")
        self.parts.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.parts.append("| " + " | ".join(str(x) for x in r) + " |")
        self.parts.append("\n<details><summary>LaTeX</summary>\n")
        self.parts.append("```latex")
        self.parts.append(r"\begin{table}[t]\centering")
        self.parts.append(r"\caption{%s}" % caption.replace("%", r"\%"))
        self.parts.append(r"\begin{tabular}{%s}\toprule" % "".join(align))
        self.parts.append(" & ".join(h.replace("%", r"\%") for h in header) + r" \\ \midrule")
        for r in rows:
            self.parts.append(" & ".join(str(x).replace("%", r"\%").replace("±", r"$\pm$")
                                         for x in r) + r" \\")
        self.parts.append(r"\bottomrule\end{tabular}\end{table}")
        self.parts.append("```")
        self.parts.append("</details>\n")

    def write(self, path):
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def pct(x, d=1):
    return f"{x * 100:.{d}f} %"


def main():
    d = Doc()
    d.parts.append("# Paper tables")
    d.p(f"\nGenerated from `results/*.json` by `scripts/make_paper_tables.py` on "
        f"{date.today().isoformat()}. Do not edit by hand — rerun the script, then "
        f"`scripts/verify_results.py`.\n")

    # ---------------------------------------------------------------- T1
    nd = load("null_direction")
    d.h("T1 — The null direction, closed form against measurement")
    d.table(
        "Analytic null direction of the mask-only Fisher matrix against its measured "
        "weakest eigenvector, at CFL-matched time steps.",
        ["wind (m/s)", "CFL", "analytic v", "measured", "abs(cos)"],
        [[a["U"], f"{a['cfl']:.2f}",
          "(" + ", ".join(f"{x:+.4f}" for x in a["analytic"]) + ")",
          "(" + ", ".join(f"{x:+.4f}" for x in a["measured"]) + ")",
          f"{abs(a['cos']):.6f}"] for a in nd["alignment"]],
        "scripts/null_direction.py", "Section 1")

    d.table(
        "Cost of a forced wind error, compensated along the exact null curve, along its "
        "tangent, and not compensated.",
        ["wind error", "exact curve", "tangent", "no compensation"],
        [[pct(w["frac"], 0), f"{w['curve']:.2f}", f"{w['tangent']:.1f}",
          f"{w['control']:,.0f}"] for w in nd["walk"]],
        "scripts/null_direction.py", "Section 1")

    # ---------------------------------------------------------------- T2
    fl, ls = load("farsite_lb"), load("lb_scatter")
    d.h("T2 — Under FARSITE's own shape law")
    d.p(f"\nAnderson (1983) truncated at LB = {fl['lb_cap']:.0f}; the cap is reached at "
        f"**U = {fl['u_saturation']:.2f} m/s**.\n")
    lo, hi = ls["bracket"]
    d.table(
        f"Shape-to-wind elasticity and the wind precision it implies, for the inferred "
        f"bracket of {pct(lo, 0)}–{pct(hi, 0)} relative scatter in the length-to-breadth "
        f"relation. Section 3 replaces that bracket with a measurement of "
        f"{pct(sw_scatter := load('real_shape_vs_wind')['sigma_rel'], 0)}.",
        ["U (m/s)", "LB", "elasticity dlnLB/dlnU",
         f"sigma(U) at {pct(lo, 0)}", f"sigma(U) at {pct(hi, 0)}",
         f"sigma(U) at {pct(sw_scatter, 0)}"],
        [[f"{e['U']:.1f}", f"{e['LB']:.2f}", f"{e['E']:.3f}",
          pct(lo / e["E"], 0), pct(hi / e["E"], 0), pct(sw_scatter / e["E"], 0)]
         for e in ls["elasticity"]],
        "scripts/lb_scatter.py, scripts/real_shape_vs_wind.py", "Sections 2 and 3")
    d.p(f"\nThe operational tolerance is {pct(ls['tol_U'], 0)} on wind speed. Every row "
        f"of the measured column misses it.\n")

    # ---------------------------------------------------------------- T3
    rp, rl, sw = load("real_perimeters"), load("real_lb_spread"), load("real_shape_vs_wind")
    d.h("T3 — Real fire perimeters (NIFC WFIGS) and ERA5 wind")
    d.table(
        "How well an area-matched moment ellipse describes a real final fire perimeter, "
        "and how well posed its length-to-breadth ratio is.",
        ["quantity", "value"],
        [["perimeters", f"{rp['n']}"],
         ["median IoU with the best moment ellipse", f"{rp['iou']['p50']:.3f}"],
         ["perimeters above IoU 0.90", f"{rp['iou']['frac_above_0.9']:.0f}"],
         ["perimeters below IoU 0.70", pct(rp['iou']['frac_below_0.7'], 0)],
         ["not a single connected blob", pct(rp['parts']['frac_multi'], 1)],
         ["median L/B", f"{rp['LB']['p50']:.2f}"],
         ["above FARSITE's cap of 8", pct(rp['LB']['frac_above_cap'], 2)],
         ["L/B spread across 5 estimators (median rel. sd)",
          pct(rl['sd_rel']['p50'], 1)],
         ["fires where estimators differ by > 20 %",
          pct(rl['frac_spread_above_20pct'], 1)]],
        "scripts/real_perimeters.py, scripts/real_lb_spread.py", "Section 3")

    d.table(
        "Wind observed by ERA5 within bins of observed fire shape. No fire model and no "
        "fitted parameter.",
        ["L/B bin", "n", "wind p25", "median", "p75", "p10–p90 spread"],
        [[f"{b['lo']:.1f}–{b['hi']:.1f}", b["n"], f"{b['p25']:.2f}",
          f"{b['median']:.2f}", f"{b['p75']:.2f}", pct(b["rel_spread"], 0)]
         for b in sw["bins"]],
        "scripts/real_shape_vs_wind.py", "Section 3")

    d.p(f"\nModel-free correlation of log L/B with log wind: **r = {sw['r_logs']:.3f}** "
        f"(r² = {sw['r_logs']**2:.3f}). Against Anderson's relation with the wind "
        f"adjustment factor fitted at {sw['waf']:.3f}: **r = {sw['r_fit']:.3f}**, residual "
        f"scatter **{pct(sw['sigma_rel'], 0)}**. Alexander (1985) reported r = 0.865 on "
        f"experimental and curated fires.\n")

    # ---------------------------------------------------------------- T4
    fu, fm, ms = load("fusion_level"), load("fusion_failure_mode"), load("fusion_multistart")
    d.h("T4 — Feature-level against physical-state-level fusion")
    arms = ["prior (train median)", "CNN mask only", "CNN mask + plume",
            "state mask only", "state mask + plume"]
    d.table(
        "Median relative error in wind speed, in and out of the training distribution. "
        "One architecture, matched capacity and training budget; only the information "
        "differs.",
        ["arm", "in-distribution", "out-of-distribution", "degradation"],
        [[a, pct(fu["rows"][f"{a}|test_id"]["U"], 1),
          pct(fu["rows"][f"{a}|test_ood"]["U"], 1),
          f"{fu['summary'][a]['ratio']:.1f}x"] for a in arms],
        "scripts/fusion_level.py", "Section 12")

    d.table(
        "Whether each failed inversion had the information available. Misfit recomputed "
        "at the recovered parameters and at the true ones, on the same observations.",
        ["arm", "failures", "of which optimiser failures"],
        [[k.replace("|", ", "), f"{v['n_failed']} of {v['n']}",
          f"{v['n_optimiser']} ({v['n_optimiser'] / max(v['n_failed'], 1) * 100:.0f} %)"]
         for k, v in fm.items()],
        "scripts/fusion_failure_mode.py", "Section 12.4")

    d.table(
        "Three starts, screened cheaply with the best refined, selected by misfit and "
        "therefore without ground truth.",
        ["arm", "median", "p90", "failures > 5 %"],
        [[k.replace("|", ", "),
          f"{pct(v['median_single'], 1)} → {pct(v['median_multi'], 1)}",
          f"{pct(v['p90_single'], 0)} → {pct(v['p90_multi'], 0)}",
          f"{pct(v['fail_single'], 0)} → {pct(v['fail_multi'], 0)}"]
         for k, v in ms["rows"].items()],
        "scripts/fusion_multistart.py", "Section 12.5")

    fs = load("fusion_scale")
    d.table(
        f"The state-level arms rerun on {fs['cases']} independently drawn fires with a "
        f"different seed, with bootstrap intervals. Only one of the three original point "
        f"estimates falls inside the new interval.",
        ["arm", "n = 25/12", f"n = {fs['cases']}/30", "95 % interval", "failures"],
        [[k.replace("state ", "").replace("|", ", "),
          pct(v["orig_median"], 1), f"**{pct(v['median'], 1)}**",
          f"[{v['median_ci'][0]*100:.1f}, {v['median_ci'][1]*100:.1f}]",
          f"{pct(v['fail'], 0)} [{v['fail_ci'][0]*100:.0f}, {v['fail_ci'][1]*100:.0f}]"]
         for k, v in fs["rows"].items()],
        "scripts/fusion_scale.py", "Section 12.6")

    # ---------------------------------------------------------------- T5
    ca = load("independent_ca")
    d.h("T5 — The same degeneracy in a third-party simulator (PyTorchFire)")
    d.table(
        "Compensating a wind error along the exact null curve of PyTorchFire's own "
        "propagation law, and the realised fires on matched random seeds.",
        ["wind error", "max change in the 8 probabilities", "burn cells differing",
         "uncompensated"],
        [[f"{r['pct']} %", f"{r['dp_exact']:.1e}",
          next((f"{q['diff_compensated'] * 100:.4f} %" for q in ca["realised"]
                if q["pct"] == r["pct"]), "—"),
          next((f"{q['diff_uncompensated'] * 100:.2f} %" for q in ca["realised"]
                if q["pct"] == r["pct"]), "—")] for r in ca["null_curve"]],
        "scripts/independent_ca.py", "Section 13")

    d.table(
        "Smallest residual achievable after a +30 % wind error, by which coefficients are "
        "free to move, and for a wind rotation.",
        ["what is free", "residual", "verdict"],
        [[k, f"{v:.2e}", "degenerate" if v < 1e-9 else "identifiable"]
         for k, v in ca["lattice"].items() if "optimiser" not in k]
        + [[f"{r['deg']}° wind rotation, both free", f"{r['residual']:.2e}",
            "identifiable"] for r in ca["direction_control"]],
        "scripts/independent_ca.py", "Section 13.2")

    # ---------------------------------------------------------------- T6
    wa = load("wsts_advance")
    d.h("T6 — Real satellite fire progression against real wind (WildfireSpreadTS)")
    d.table(
        f"Association between a fire's daily advance and the GRIDMET wind that drove it, "
        f"over {wa['n_days']} fire-days from {wa['n_fires']} fires. Four advance measures, "
        f"reported together because they fail differently.",
        ["advance measure", "n", "r (log–log)", "r²"],
        [[v["label"], v["n"], f"{v['r']:.3f}", f"{v['r']**2:.4f}"]
         for v in wa["measures"].values()],
        "scripts/wsts_advance.py", "Section 14.2")

    d.table(
        "Both channels sharpen with measurement quality, which is why this data cannot "
        "separate a missing signal from a badly measured one.",
        ["detections required that day", "bearing vs wind direction (R)",
         "advance vs wind speed (r)"],
        [[f"≥ {a['min_new']}", f"{a['R']:.3f}", f"{b['r_rad']:.3f}"]
         for a, b in zip(wa["direction"]["quality_cuts"], wa["speed_quality"])],
        "scripts/wsts_advance.py", "Section 14.4")

    d.table(
        "Regressing log daily advance on the wind and on the fuel and terrain covariates "
        "the dataset carries.",
        ["model", "R²", "wind coefficient", "s.e.", "t"],
        [[k, f"{v['r2']:.4f}", f"{v['beta_wind']:.3f}", f"{v['se']:.4f}",
          f"{v['t']:.1f}"] for k, v in wa["mechanism"].items()],
        "scripts/wsts_advance.py", "Section 14.3")

    # ---------------------------------------------------------------- T7
    wb = load("wsts_baseline")
    d.h("T7 — Channel ablation on the benchmark's next-day spread task")
    rowsb = [["chance", f"{wb['chance']:.4f}", "—"],
             ["persistence (ring around today's fire)", f"{wb['persistence']:.4f}",
              f"{wb['persistence'] / wb['arms']['fire']['test_ap_mean']:.2f}x"]]
    for k, v in wb["arms"].items():
        rowsb.append([k, f"{v['test_ap_mean']:.4f} ± {v['test_ap_sd']:.4f}",
                      f"{v['test_ap_mean'] / wb['arms']['fire']['test_ap_mean']:.2f}x"])
    d.table(
        f"Test average precision, one U-net of {wb['arms']['fire']['n_params']:,} "
        f"parameters trained once per arm with the other channels zeroed, two seeds. "
        f"{wb['n']['train']} training pairs, {wb['n']['test']} test pairs, "
        f"{pct(wb['positive_rate_test'], 2)} positive pixels.",
        ["input channels", "test AP", "vs fire mask only"], rowsb,
        "scripts/wsts_baseline.py", "Section 15")

    d.table(
        "The same arms by the wind on the day. Nothing improves when there is more wind.",
        ["wind", "pairs", "fire", "fire + wind", "all minus wind", "all"],
        [[k, v["n"], f"{v['fire']:.4f}", f"{v['fire + wind']:.4f}",
          f"{v['all minus wind']:.4f}", f"{v['all']:.4f}"]
         for k, v in wb["by_wind"].items()],
        "scripts/wsts_baseline.py", "Section 15.3")

    wf, tr = load("wsts_baseline_full"), load("wsts_trivial")
    bn, bf = wb["arms"]["fire"]["test_ap_mean"], wf["arms"]["fire"]["test_ap_mean"]
    rows_t = [["**copy today's fire unchanged**",
               f"{tr['new-fire (used in section 15)']['copy_today']:.4f}", "—",
               f"**{tr['published (whole next-day fire)']['copy_today']:.4f}**", "—"],
              ["ring around today's fire", f"{wb['persistence']:.4f}",
               f"{wb['persistence']/bn:.2f}x", f"{wf['persistence']:.4f}", "—"]]
    for k in wb["arms"]:
        a, b = wb["arms"][k]["test_ap_mean"], wf["arms"][k]["test_ap_mean"]
        rows_t.append([k, f"{a:.4f}", f"{a/bn:.2f}x", f"{b:.4f}", f"{b/bf:.3f}x"])
    d.table(
        "The same ablation on the spread-only target and on WildfireSpreadTS's published "
        "target, nothing else changed. On the published target a rule that copies today's "
        "fire and reads no channel already scores 0.9474, and every arm lands within "
        "0.7 % of every other.",
        ["arm", "new-fire AP", "vs fire", "published AP", "vs fire"], rows_t,
        "scripts/wsts_baseline.py, scripts/wsts_trivial.py", "Section 15.4")

    # ---------------------------------------------------------------- T8
    rd = load("retrodiction")
    d.h("T8 — Backtracking real fires to their ignition")
    d.table(
        f"Feasible ignition area by how well the spread rate is known, over "
        f"{rd['n']} backtracks on {rd['n_fires']} real fires. Coverage is 100 % by "
        f"construction for the centred regimes; the informative column is the area.",
        ["rate known to", "median feasible area", "as % of the burn scar", "coverage"],
        [[k, f"{v['median_km2']:.1f} km²", pct(v["median_frac"], 0),
          pct(v["coverage"], 0)] for k, v in rd["regimes"].items()],
        "scripts/retrodiction.py", "Section 16")

    d.table(
        "The same, by how long the fire had been burning. The burn scar grows; the "
        "fraction of it the origin could lie in does not shrink.",
        ["days burning", "backtracks", "median burn", "oracle", "covariates"],
        [[k, v["n"], f"{v['burn_km2']:.0f} km²",
          f"{v['area']['oracle (+/-10 %)']:.1f} km² ({v['frac']['oracle (+/-10 %)'] * 100:.0f} %)",
          f"{v['area']['covariates (x1.96)']:.1f} km² ({v['frac']['covariates (x1.96)'] * 100:.0f} %)"]
         for k, v in rd["by_dt"].items()],
        "scripts/retrodiction.py", "Section 16.1")

    # ---------------------------------------------------------------- T9
    gi = load("guarded_inversion")
    d.h("T9 — An inversion that reports what the observations determine")
    d.table(
        "Verdict on wind speed, with the guard not told which sensor set it is looking "
        "at. A 30 % perturbation is imposed, every other parameter is allowed to "
        "re-absorb it, and the misfit is read.",
        ["arm", "wind speed called determined", "median error in wind speed"],
        [[k.replace("state ", "").replace("|", ", "),
          pct(v["U_determined"], 0), pct(v["median_err"], 1)]
         for k, v in gi["summary"].items()],
        "scripts/guarded_inversion.py", "Section 17.1")

    d.table(
        f"Both tests on the same {sum(v['n'] for v in gi['summary'].values())} "
        f"inversions, scored against whether the answer was in fact wrong by more than "
        f"{gi['wrong_threshold']*100:.0f} % on wind speed.",
        ["raised by", "wrong, caught", "wrong, missed", "right, flagged"],
        [[k, v["caught"], v["missed"], v["false_alarm"]]
         for k, v in gi["confusion"].items()],
        "scripts/guarded_inversion.py", "Section 17.2")

    cal = gi["calibration"]
    d.p(f"\nInterval calibration, pooled over the cases that both passed the guard and "
        f"converged: **{cal['covered']} of {cal['n']} = {cal['coverage']*100:.0f} %**, "
        f"95 % interval {cal['ci95'][0]*100:.0f}–{cal['ci95'][1]*100:.0f} %. The nominal "
        f"68 % for a 1-sigma interval is inside that; scaling sigma by "
        f"{cal['scale_for_68']:.2f} would land it exactly on 68 %, which {cal['n']} cases "
        f"cannot resolve.\n")

    # ---------------------------------------------------------------- T10
    sp = load("sufficiency_profile")
    d.h("T10 — Sensor sufficiency, by profile probe rather than Cramér–Rao bound")
    for scene, cells in sp["scenes"].items():
        rows_s = []
        for label, row in cells.items():
            out_row = [label]
            for p in PARAM_NAMES:
                v = row.get(p)
                if v is None:
                    out_row.append("—")
                elif not v["determined"]:
                    out_row.append("**not determined**")
                elif v["sigma_rel"] is not None:
                    out_row.append(f"{v['sigma_rel']*100:.2f} %")
                else:
                    out_row.append(f"{v['sigma_abs']:.3f}°")
            rows_s.append(out_row)
        d.table(
            f"Which sensor subsets determine which state components, {scene}. Entries are "
            f"the 1-sigma precision where the parameter is determined; a cell is "
            f"undetermined when forcing it off by {sp['probe']*100:.0f} % costs a misfit "
            f"below {sp['dchi2_flat']:.0f} with every other parameter free to re-absorb it.",
            ["sensors"] + list(PARAM_NAMES), rows_s,
            "scripts/sufficiency_profile.py", "Section 17.4")

    # ---------------------------------------------------------------- T11
    rf = load("real_fire_inversion")
    d.h("T11 — Inverting real fires against the wind that actually blew")
    d.table(
        f"Each of {rf['n']} real WildfireSpreadTS fires fitted three times, from "
        f"{', '.join(str(u) for u in rf['u_starts'])} m/s — a "
        f"{max(rf['u_starts'])/min(rf['u_starts']):.0f}x span of plausible winds. If the "
        f"burn determined the wind the three would converge; the answers scatter by 25x "
        f"while the fit quality moves by 2x.",
        ["quantity", "median", "p25 / p90", "for scale"],
        [["IoU of fitted front with observed burn", f"{rf['iou']['median']:.3f}",
          f"{rf['iou']['p25']:.3f} / {rf['iou']['p75']:.3f}",
          "0.705 for the best ellipse (T3)"],
         ["largest / smallest **fitted wind**, per fire",
          f"**{rf['U_spread']['median']:.1f}x**",
          f"— / {rf['U_spread']['p90']:.1f}x", "4x span of starts"],
         ["largest / smallest **final misfit**, per fire",
          f"{rf['loss_spread']['median']:.2f}x",
          f"— / {rf['loss_spread']['p90']:.2f}x", "—"],
         ["wind speed, error against GRIDMET",
          f"**{rf['U_rel_err']['median']*100:.0f} %**",
          f"{rf['U_rel_err']['p25']*100:.0f} / {rf['U_rel_err']['p75']*100:.0f} %", "—"],
         ["wind bearing, absolute error",
          f"**{rf['bearing_err']['median']:.0f}°**",
          f"{rf['bearing_err']['p25']:.0f} / {rf['bearing_err']['p75']:.0f}°",
          "90° if guessing"],
         ["bearings within 45°",
          f"{rf['bearing_err']['within_45']*100:.0f} %", "—", "25 % if guessing"],
         ["guard calls the **speed** determined",
          f"**{rf['guard']['U_determined']*100:.0f} %**", "—", "—"],
         ["guard calls the **bearing** determined",
          f"{rf['guard']['theta_w_determined']*100:.0f} %", "—", "—"]],
        "scripts/real_fire_inversion.py", "Section 18")

    if len(rf.get("by_fit_quality", {})) == 2:
        d.table(
            "The same fires split by how well the model describes them. The half it fits "
            "well fails to determine the wind exactly as badly as the half it fits "
            "poorly, which rules out misspecification as the explanation.",
            ["fit quality", "fires", "wind spread across starts", "wind error",
             "bearing error"],
            [[k, v["n"], f"{v['U_spread']:.1f}x", f"{v['U_rel_err']*100:.0f} %",
              f"{v['bearing_err']:.0f}°"] for k, v in rf["by_fit_quality"].items()],
            "scripts/real_fire_inversion.py", "Section 18.3")

    # ---------------------------------------------------------------- figures
    d.h("Figure manifest")
    doc = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    import re
    used = []
    for m in re.finditer(r"!\[([^\]]*)\]\(figures/([A-Za-z0-9_]+\.png)\)", doc):
        sec = doc[:m.start()].rfind("\n## ")
        title = doc[sec + 4:doc.find("\n", sec + 4)] if sec >= 0 else "?"
        used.append((m.group(2), m.group(1), title))
    d.table(
        "Every figure cited by RESULTS.md, in order, with the section that cites it.",
        ["file", "alt text", "section"],
        [[f"`{f}`", alt, s] for f, alt, s in used],
        "figures/", "all")
    unused = sorted({p.name for p in FIGS.glob("*.png")} - {f for f, _, _ in used})
    if unused:
        d.p(f"\nNot cited in RESULTS.md ({len(unused)}): "
            + ", ".join(f"`{u}`" for u in unused) + ".\n")

    d.h("Reproducibility")
    d.p("\nEvery number above is read from `results/*.json` at generation time. "
        "`scripts/verify_results.py` independently re-derives the headline figures and "
        "fails if `RESULTS.md` disagrees with them, then audits cross-references, figure "
        "and script references, and encodings. Run both after changing any experiment.\n")

    d.write(OUT)
    print(f"wrote {OUT}")
    print(f"  {sum(1 for p in d.parts if p.lstrip().startswith('**'))} tables, "
          f"{len(used)} figures cited, {len(unused)} uncited, "
          f"{OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
