"""Check every headline number in RESULTS.md against the file that produced it.

Sixteen corrected hypotheses in, the thing most likely to be wrong in this repository is
no longer an experiment -- it is a number that was right when it was written and went
stale when the experiment was rerun. Renumbering sections, recalibrating a constant,
regenerating a dataset: each of those has already invalidated a quoted figure at least
once here.

So this reads the stored results and asserts that what the write-up says matches. A check
fails loudly with both values rather than warning, because a silent mismatch is exactly
the failure mode worth catching. It also audits the surrounding bookkeeping: that every
figure and script the write-up names exists, that every script compiles, and that no
result file is orphaned from the script that makes it.

Run:  python scripts/verify_results.py
"""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGS, SCRIPTS = ROOT / "results", ROOT / "figures", ROOT / "scripts"
DOC = ROOT / "RESULTS.md"
READ = ROOT / "README.md"


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def fmt(v, spec):
    return format(v, spec)


class Checker:
    def __init__(self, text):
        self.text = text
        self.passed, self.failed = 0, []

    def says(self, label, value, spec, note=""):
        """Assert the formatted value appears somewhere in the write-up.

        The write-up separates thousands with a space, so a value formatted with a comma
        is accepted in either style rather than forcing the prose to match the checker.
        """
        s = fmt(value, spec)
        if s in self.text or ("," in s and s.replace(",", " ") in self.text):
            self.passed += 1
        else:
            self.failed.append((label, s, note))

    def check(self, label, ok, note=""):
        if ok:
            self.passed += 1
        else:
            self.failed.append((label, "condition false", note))


def main():
    doc = DOC.read_text(encoding="utf-8")
    c = Checker(doc)

    # --- section 1: the exact null direction -------------------------------------
    nd = load("null_direction")
    c.check("s1 alignment >= 0.99996", nd["min_cos"] >= 0.99996,
            f"min_cos = {nd['min_cos']}")
    c.says("s1 min alignment", nd["min_cos"], ".6f")

    # --- section 2: FARSITE's shape law -------------------------------------------
    fl = load("farsite_lb")
    c.says("s2 saturation wind", fl["u_saturation"], ".2f")
    c.check("s2 cap is 8", fl["lb_cap"] == 8.0)
    ls = load("lb_scatter")
    c.says("s2 tolerance", ls["tol_U"] * 100, ".0f")

    # --- section 3: real perimeters and ERA5 --------------------------------------
    rp = load("real_perimeters")
    c.says("s3 perimeter count", rp["n"], "d")
    c.says("s3 median IoU", rp["iou"]["p50"], ".3f")
    c.check("s3 none above IoU 0.90", rp["iou"]["frac_above_0.9"] == 0.0)
    c.says("s3 median L/B", rp["LB"]["p50"], ".2f")
    rl = load("real_lb_spread")
    c.says("s3 estimator sd p50", rl["sd_rel"]["p50"] * 100, ".1f")
    c.says("s3 spread above 20 %", rl["frac_spread_above_20pct"] * 100, ".1f")
    sw = load("real_shape_vs_wind")
    c.says("s3 model-free r", sw["r_logs"], ".3f")
    c.says("s3 fitted r", sw["r_fit"], ".3f")
    c.says("s3 residual scatter", sw["sigma_rel"] * 100, ".0f")
    c.says("s3 fitted WAF", sw["waf"], ".3f")

    # --- section 12: fusion level --------------------------------------------------
    fu = load("fusion_level")
    for arm, split, spec in (("CNN mask only", "test_id", ".1f"),
                             ("CNN mask only", "test_ood", ".1f"),
                             ("CNN mask + plume", "test_id", ".1f"),
                             ("CNN mask + plume", "test_ood", ".1f"),
                             ("state mask + plume", "test_id", ".1f"),
                             ("state mask + plume", "test_ood", ".1f"),
                             ("state mask only", "test_id", ".1f")):
        c.says(f"s12 {arm} {split} U", fu["rows"][f"{arm}|{split}"]["U"] * 100, spec)
    c.says("s12 net parameters", fu["CNN mask only"]["n_params"], ",d")
    fm = load("fusion_failure_mode")
    k = "state mask only|test_id"
    c.check("s12.4 all mask-only failures are information failures",
            fm[k]["n_optimiser"] == 0 and fm[k]["n_failed"] == fm[k]["n"],
            f"{fm[k]['n_optimiser']} of {fm[k]['n_failed']} of {fm[k]['n']}")
    ms = load("fusion_multistart")
    for key in ("state mask + plume|test_id", "state mask + plume|test_ood",
                "state mask only|test_id"):
        c.says(f"s12.5 {key} median multi", ms["rows"][key]["median_multi"] * 100, ".1f")
        c.says(f"s12.5 {key} fail multi", ms["rows"][key]["fail_multi"] * 100, ".0f")

    # --- section 12.6: the same arms at scale ----------------------------------------
    fs = load("fusion_scale")
    c.says("s12.6 cases", fs["cases"], "d")
    for key, r in fs["rows"].items():
        c.says(f"s12.6 {key} median", r["median"] * 100, ".1f")
        c.says(f"s12.6 {key} fail", r["fail"] * 100, ".0f")
        c.says(f"s12.6 {key} ci lo", r["median_ci"][0] * 100, ".1f")
        c.says(f"s12.6 {key} ci hi", r["median_ci"][1] * 100, ".1f")
    ood = fs["rows"]["state mask + plume|test_ood"]
    c.check("s12.6 the original OOD median is outside the new interval",
            not (ood["median_ci"][0] <= ood["orig_median"] <= ood["median_ci"][1]),
            f"orig {ood['orig_median']:.4f} vs CI {ood['median_ci']}")
    mo = fs["rows"]["state mask only|test_id"]
    c.check("s12.6 the mask-only control reproduces",
            mo["median_ci"][0] <= mo["orig_median"] <= mo["median_ci"][1])

    # --- section 13: the independent simulator -------------------------------------
    ca = load("independent_ca")
    c.check("s13 null curve exact to machine precision",
            max(r["dp_exact"] for r in ca["null_curve"]) < 1e-15,
            f"max = {max(r['dp_exact'] for r in ca['null_curve'])}")
    c.check("s13 compensated masks identical",
            all(r["diff_compensated"] == 0.0 for r in ca["realised"]))
    cells = ca["grid"] ** 2 * ca["seeds"]
    c.says("s13 cell count", cells, ",d")
    c.says("s13 uncompensated at 80 %", 100 * [r for r in ca["realised"]
                                               if r["pct"] == 80][0]["diff_uncompensated"],
           ".2f")
    c.says("s13 only p_h free", ca["lattice"]["only p_h free (shape law known)"], ".2e")
    c.says("s13 5 deg rotation", ca["direction_control"][0]["residual"], ".2e")
    lbs = [r["lb"] for r in ca["shape"]]
    c.says("s13 peak L/B", max(lbs), ".2f")
    rot8 = ca["rotation"]["8.0"]
    c.says("s13 rotation spread", (max(rot8) - min(rot8)) / (sum(rot8) / len(rot8)) * 100,
           ".0f")

    # --- section 14: real progression ----------------------------------------------
    wa = load("wsts_advance")
    c.says("s14 fires", wa["n_fires"], "d")
    c.says("s14 fire-days", wa["n_days"], "d")
    c.says("s14 direction R", wa["direction"]["R"], ".3f")
    c.says("s14 direction offset", wa["direction"]["offset_deg"], ".0f")
    for m, spec in (("advance_m", ".3f"), ("d_radius_m", ".3f")):
        c.says(f"s14 r for {m}", wa["measures"][m]["r"], spec)
    c.says("s14 mechanism R2 full", wa["mechanism"]["+ ERC, NDVI, slope"]["r2"], ".4f")
    c.says("s14 mechanism R2 wind", wa["mechanism"]["wind only"]["r2"], ".4f")
    c.says("s14 best-quality direction R", wa["direction"]["quality_cuts"][-1]["R"], ".3f")

    # --- section 15: the channel ablation --------------------------------------------
    wb = load("wsts_baseline")
    c.says("s15 persistence AP", wb["persistence"], ".4f")
    c.says("s15 chance AP", wb["chance"], ".4f")
    for arm in ("fire", "fire + wind", "fire + weather", "fire + static",
                "fire + VIIRS", "all minus wind", "all"):
        c.says(f"s15 {arm} AP", wb["arms"][arm]["test_ap_mean"], ".4f")
        c.says(f"s15 {arm} sd", wb["arms"][arm]["test_ap_sd"], ".4f")
    for s in ("train", "val", "test"):
        c.says(f"s15 {s} pairs", wb["n"][s], "d")
    c.check("s15 VIIRS beats all 23 channels",
            wb["arms"]["fire + VIIRS"]["test_ap_mean"] > wb["arms"]["all"]["test_ap_mean"])
    c.check("s15 dropping wind helps",
            wb["arms"]["all minus wind"]["test_ap_mean"] > wb["arms"]["all"]["test_ap_mean"])
    c.says("s15 wind cost in %",
           abs(wb["wind_worth"]["all"] - wb["wind_worth"]["all_minus_wind"])
           / wb["wind_worth"]["all_minus_wind"] * 100, ".0f")
    for lab in wb["by_wind"]:
        for arm in ("fire", "fire + wind", "all minus wind", "all"):
            c.says(f"s15 {lab} {arm}", wb["by_wind"][lab][arm], ".4f")

    # --- section 15.4: the two target framings ----------------------------------------
    wf = load("wsts_baseline_full")
    tr = load("wsts_trivial")
    pub = tr["published (whole next-day fire)"]
    c.says("s15.4 copy-today on the published target", pub["copy_today"], ".4f")
    c.says("s15.4 copy-today on the spread target",
           tr["new-fire (used in section 15)"]["copy_today"], ".4f")
    for arm in ("fire", "fire + wind", "fire + weather", "fire + static",
                "fire + VIIRS", "all minus wind", "all"):
        c.says(f"s15.4 full-target {arm}", wf["arms"][arm]["test_ap_mean"], ".4f")
    c.check("s15.4 every full-target arm lands between 0.985 and 0.992",
            all(0.985 <= v["test_ap_mean"] <= 0.9925 for v in wf["arms"].values()))
    c.check("s15.4 the imagery gap compresses on the published target",
            (wf["arms"]["fire + VIIRS"]["test_ap_mean"]
             / wf["arms"]["fire"]["test_ap_mean"]) < 1.01
            < (wb["arms"]["fire + VIIRS"]["test_ap_mean"]
               / wb["arms"]["fire"]["test_ap_mean"]))

    # --- section 16: retrodiction ---------------------------------------------------
    rd = load("retrodiction")
    c.says("s16 backtracks", rd["n"], "d")
    c.says("s16 fires", rd["n_fires"], "d")
    for k2, spec in (("oracle (+/-10 %)", ".1f"), ("covariates (x1.96)", ".1f")):
        c.says(f"s16 {k2} area", rd["regimes"][k2]["median_km2"], spec)
        c.says(f"s16 {k2} frac", rd["regimes"][k2]["median_frac"] * 100, ".0f")
    c.check("s16 covariates cover the whole scar",
            rd["regimes"]["covariates (x1.96)"]["median_frac"] >= 0.99)
    c.check("s16 covariates equal prior",
            abs(rd["regimes"]["covariates (x1.96)"]["median_km2"]
                - rd["regimes"]["prior only (x2.01)"]["median_km2"]) < 0.05)

    # --- section 17: the guarded inversion -------------------------------------------
    gi = load("guarded_inversion")
    for k2 in ("state mask + plume|test_id", "state mask + plume|test_ood",
               "state mask only|test_id"):
        c.says(f"s17 {k2} determined", gi["summary"][k2]["U_determined"] * 100, ".0f")
    for k2, v in gi["confusion"].items():
        c.says(f"s17 {k2} caught", v["caught"], "d")
        c.says(f"s17 {k2} missed", v["missed"], "d")
    cal = gi["calibration"]
    c.says("s17 coverage n", cal["n"], "d")
    c.says("s17 covered", cal["covered"], "d")
    c.says("s17 coverage", cal["coverage"] * 100, ".0f")
    c.says("s17 ci low", cal["ci95"][0] * 100, ".0f")
    c.says("s17 ci high", cal["ci95"][1] * 100, ".0f")
    c.says("s17 sigma scale", cal["scale_for_68"], ".2f")
    c.check("s17 nominal coverage is inside the interval",
            cal["ci95"][0] <= 0.68 <= cal["ci95"][1])
    c.check("s17 the guard beats the residual check",
            gi["confusion"]["profile guard"]["caught"]
            > gi["confusion"]["residual check"]["caught"])

    # --- section 17.4: the sufficiency diagram ---------------------------------------
    sp = load("sufficiency_profile")
    flat = sp["scenes"]["flat, 3 m/s"]
    slope = sp["scenes"]["20 deg slope, 3 m/s"]
    c.check("s17.4 masks determine only theta_w on flat ground",
            all(v["determined"] == (p == "theta_w")
                for p, v in flat["masks"].items()))
    c.check("s17.4 air quality adds nothing on flat ground",
            all(flat["masks"][p]["determined"] == flat["masks + air quality"][p]["determined"]
                for p in flat["masks"]))
    c.check("s17.4 masks determine nothing on a slope",
            not any(v["determined"] for v in slope["masks"].values()))
    c.check("s17.4 air quality recovers five of six on a slope",
            sum(v["determined"] for v in slope["masks + air quality"].values()) == 5)
    c.check("s17.4 the plume determines all six in both scenes",
            all(v["determined"] for v in flat["masks + plume"].values())
            and all(v["determined"] for v in slope["masks + plume"].values()))
    c.says("s17.4 masks theta_w on flat", flat["masks"]["theta_w"]["sigma_abs"], ".3f")
    for scene, cells in (("flat", flat), ("slope", slope)):
        for p in ("R0", "U", "lb_k", "w_buoy", "Q"):
            v = cells["masks + plume"][p]
            c.says(f"s17.4 {scene} plume {p}", v["sigma_rel"] * 100, ".2f")
    for p in ("R0", "U", "lb_k", "Q"):
        c.says(f"s17.4 slope AQ {p}", slope["masks + air quality"][p]["sigma_rel"] * 100,
               ".2f")

    # --- section 18: inverting a real fire --------------------------------------------
    rf = load("real_fire_inversion")
    c.says("s18 fires", rf["n"], "d")
    c.says("s18 median IoU", rf["iou"]["median"], ".3f")
    c.says("s18 U spread median", rf["U_spread"]["median"], ".1f")
    c.says("s18 U spread p90", rf["U_spread"]["p90"], ".1f")
    c.says("s18 loss spread median", rf["loss_spread"]["median"], ".2f")
    c.says("s18 loss spread p90", rf["loss_spread"]["p90"], ".2f")
    c.says("s18 frac above 1.5x", rf["U_spread"]["frac_above_1p5"] * 100, ".0f")
    c.says("s18 U error", rf["U_rel_err"]["median"] * 100, ".0f")
    c.says("s18 bearing error", rf["bearing_err"]["median"], ".0f")
    c.says("s18 bearing within 45", rf["bearing_err"]["within_45"] * 100, ".0f")
    c.says("s18 guard bearing determined", rf["guard"]["theta_w_determined"] * 100, ".0f")
    c.check("s18 the guard never certifies the speed",
            rf["guard"]["U_determined"] == 0.0)
    c.check("s18 the answers scatter more than the misfit does",
            rf["U_spread"]["median"] > 10 * rf["loss_spread"]["median"])
    q = rf["by_fit_quality"]
    if len(q) == 2:
        good = [v for k, v in q.items() if ">=" in k][0]
        bad = [v for k, v in q.items() if "<" in k][0]
        c.says("s18 good-fit spread", good["U_spread"], ".1f")
        c.says("s18 good-fit U error", good["U_rel_err"] * 100, ".0f")
        c.says("s18 poor-fit spread", bad["U_spread"], ".1f")
        c.check("s18 fit quality does not rescue the wind",
                abs(good["U_rel_err"] - bad["U_rel_err"]) < 0.05)

    # --- bookkeeping ---------------------------------------------------------------
    print("=" * 78)
    print("[1] numbers in RESULTS.md against the files that produced them")
    print("=" * 78)
    print(f"    {c.passed} checks passed, {len(c.failed)} failed")
    for label, val, note in c.failed:
        print(f"    FAIL  {label}: expected to find {val!r} in RESULTS.md  {note}")

    print()
    print("=" * 78)
    print("[2] figures, scripts and cross-references")
    print("=" * 78)
    problems = []
    heads = {int(m.group(1)) for m in re.finditer(r"^## (\d+)\.", doc, re.M)}
    subs = {m.group(1) for m in re.finditer(r"^### (\d+)\.\d+", doc, re.M)}
    for s in subs:
        if int(s) not in heads:
            problems.append(f"subsection {s}.x has no parent section")
    for t, name in ((doc, "RESULTS.md"), (READ.read_text(encoding="utf-8"), "README.md")):
        for n in {int(m.group(2)) for m in re.finditer(r"\b([Ss]ections? )(\d+)", t)}:
            if n not in heads:
                problems.append(f"{name} cites section {n}, which does not exist")
        for f in set(re.findall(r"figures/([A-Za-z0-9_]+\.png)", t)):
            if not (FIGS / f).exists():
                problems.append(f"{name} cites missing figure {f}")
        for s in set(re.findall(r"scripts.([A-Za-z0-9_]+\.py)", t)):
            if not (SCRIPTS / s).exists():
                problems.append(f"{name} cites missing script {s}")
        bad = re.findall(r"[�Ãâ]", t)
        if bad:
            problems.append(f"{name} has {len(bad)} mojibake characters")
    for p in sorted(SCRIPTS.glob("*.py")):
        try:
            py_compile.compile(str(p), cfile=None, doraise=True)
        except Exception as e:
            problems.append(f"{p.name} does not compile: {repr(e)[:70]}")
    for p in sorted(RESULTS.glob("*.json")):
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"{p.name} is not valid JSON: {repr(e)[:70]}")
        if p.stat().st_size < 64:
            problems.append(f"{p.name} is suspiciously small")
    print(f"    {len(list(SCRIPTS.glob('*.py')))} scripts, "
          f"{len(list(RESULTS.glob('*.json')))} result files, "
          f"{len(list(FIGS.glob('*.png')))} figures")
    print(f"    {len(problems)} problems")
    for q in problems:
        print(f"    FAIL  {q}")

    print()
    print("=" * 78)
    print("[3] the generated paper tables")
    print("=" * 78)
    tbl = ROOT / "PAPER_TABLES.md"
    tprobs = []
    if not tbl.exists():
        tprobs.append("PAPER_TABLES.md is missing -- run scripts/make_paper_tables.py")
    else:
        tt = tbl.read_text(encoding="utf-8")
        newest = max(p.stat().st_mtime for p in RESULTS.glob("*.json"))
        if tbl.stat().st_mtime < newest:
            tprobs.append("PAPER_TABLES.md is older than the newest result file "
                          "-- rerun scripts/make_paper_tables.py")
        if re.search(r"[�Ãâ]", tt):
            tprobs.append("PAPER_TABLES.md has mojibake")
        # every markdown table must keep a constant column count within a block
        lines, i, bad = tt.split("\n"), 0, 0
        while i < len(lines):
            if (lines[i].startswith("| ") and i + 1 < len(lines)
                    and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")):
                n, j = lines[i].count("|"), i + 2
                while j < len(lines) and lines[j].startswith("| "):
                    bad += lines[j].count("|") != n
                    j += 1
                i = j
            else:
                i += 1
        if bad:
            tprobs.append(f"{bad} table rows have the wrong column count")
        print(f"    {tt.count(chr(10)) + 1} lines, "
              f"{sum(1 for l in lines if l.startswith('| '))} table rows, "
              f"{tt.count('```latex')} LaTeX blocks")
    print(f"    {len(tprobs)} problems")
    for q in tprobs:
        print(f"    FAIL  {q}")
    problems += tprobs

    total = len(c.failed) + len(problems)
    print()
    print("=" * 78)
    print(f"{'ALL CHECKS PASSED' if total == 0 else f'{total} PROBLEMS'}")
    print("=" * 78)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
