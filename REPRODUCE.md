# Reproducing every experiment

Dependencies: `torch`, `numpy`, `scipy`, `matplotlib`, `rasterio`, `scikit-learn`,
`requests`, `shapely`, `pyproj`. A 24 GB consumer GPU is ample — peak usage is about
0.5 GB and the full forward model takes 0.3–0.6 s. Runtimes below are for one RTX 3090.

Correctness first:

```bash
python scripts/validate_physics.py      # physics vs closed-form solutions
python scripts/test_invariants.py       # regression tests on the invariants
python scripts/verify_results.py        # every number in RESULTS.md vs its source file
```

`verify_results.py` re-derives all **169** headline figures from `results/*.json` and fails
if the write-up disagrees, then audits cross-references, figures, scripts, encodings and
the generated tables. Run it after changing anything.

```bash
python scripts/make_paper_tables.py     # regenerate PAPER_TABLES.md from results/*.json
python scripts/make_hero.py             # regenerate the front-page figure
```

The experiments, roughly in order of how much they settle:

```bash
# what the degeneracy is
python scripts/null_direction.py        # exact null space, in closed form         (~2 min)
python scripts/farsite_lb.py            # ...under FARSITE's own shape law         (~40 min)
python scripts/confound_trajectory.py   # what the masks conserve                 (~50 min)

# what does and does not change it
python scripts/lb_scatter.py            # the load-bearing assumption, quantified   (~2 min)
python scripts/terrain.py               # on a slope it gets worse                (~60 min)
python scripts/terrain_aspect.py        # which way the hill faces                 (~2 min)
python scripts/heterogeneous_fuel.py    # patchy fuel does not help                (~3 min)
python scripts/misspecification.py      # fit the wrong fire model on purpose     (~75 min)

# what it costs
python scripts/wind_shift_forecast.py   # forecasting through a wind shift        (~12 min)
python scripts/forecast_ensemble.py --n 80   # the same, over 69 fires             (~7 min)
python scripts/fuel_prior.py            # "but we know the fuel" - how well?       (~2 min)
python scripts/lead_time_profile.py     # lead time, derivative-free              (~40 min)

# how much to trust the numbers
python scripts/profile_likelihood.py --param U        # no derivatives, no step size
python scripts/profile_likelihood.py --param theta_w  # the positive control
python scripts/profile_postprocess.py U               # turn a scan into a 1-sigma number
python scripts/noise_sensitivity.py     # survive bad sensors                      (~2 min)
python scripts/camera_geometry.py       # survive bad camera siting               (~12 min)

# is it this code's fault? (needs `pip install pytorchfire`)
python scripts/independent_ca.py       # the same degeneracy in a third-party CA  (~12 min)

# real fires (needs network; ~160 MB of perimeters + one API pass)
python scripts/fetch_perimeters.py      # 900 NIFC WFIGS final perimeters          (~6 min)
python scripts/real_perimeters.py       # are they ellipses? what is their L/B?    (~2 min)
python scripts/fetch_wind.py            # ERA5 wind over each fire's active period (~4 min)
python scripts/real_lb_spread.py        # is "the" L/B even well posed?            (~2 min)
python scripts/real_shape_vs_wind.py    # does shape predict wind? the decisive one (~1 min)

# real fire progression vs real wind (WildfireSpreadTS, 17 GB via range requests)
python scripts/fetch_wsts.py --fires 200    # selective pull from a 45 GB zip     (~130 min)
python scripts/wsts_advance.py          # 1044 real fire-days against GRIDMET wind (~12 min)
python scripts/retrodiction.py          # backtrack 121 real fires to ignition      (~4 min)
python scripts/real_fire_inversion.py --fires 24    # invert real fires           (~180 min)
python scripts/real_fire_inversion.py --report-only # re-analyse, no refitting
python scripts/gen_wsts_pairs.py        # next-day spread pairs, split by year      (~4 min)
python scripts/wsts_baseline.py --epochs 60 # what the wind channels are worth    (~190 min)
python scripts/gen_wsts_pairs.py --target full      # the benchmark’s published framing
python scripts/wsts_baseline.py --epochs 30 --pairs wsts_pairs_full --tag _full
python scripts/wsts_trivial.py          # what a rule with no model scores on each  (~2 min)
python scripts/wsts_baseline.py --figure-only       # redraw from the json, no rerun

# where you fuse: feature level vs physical-state level
python scripts/gen_dataset.py           # 3750 simulated fires, split in wind      (~25 min)
python scripts/fusion_level.py          # the A8 ablation, matched budget          (~95 min)
python scripts/fusion_level.py --figure-only        # rebuild the figure, no rerun
python scripts/fusion_failure_mode.py   # optimiser failure or information failure? (~3 min)
python scripts/fusion_multistart.py --starts 3      # can a better solver fix it?  (~120 min)
python scripts/fusion_multistart.py --report-only   # re-print and re-plot, no rerun
python scripts/fusion_scale.py --cases 60           # the same arms, 60 fresh fires (~2 h)

# the method the diagnosis implies
python scripts/guarded_inversion.py     # score the guard where truth is known   (~160 min)
python scripts/guarded_inversion.py --report-only   # re-print and re-plot, no rerun
python scripts/sufficiency_profile.py   # sensor sufficiency, by profile not CRB  (~60 min)

# Fisher-based, kept for contrast (see RESULTS.md section 11)
python scripts/week0_gonogo.py          # the original go/no-go                   (~50 min)
python scripts/lead_time.py             # Fisher-based lead time (superseded)      (~7 min)
python scripts/sweep_identifiability.py --n 30
python scripts/consistency_check.py     # the two Fisher paths agree; envelope sweep
python scripts/make_figures.py
```

The long runs checkpoint after every scan point and resume automatically, so an
interrupted `profile_likelihood.py` picks up where it stopped.

`lead_time.py` and the mask-only rows of `sweep_identifiability.py` are built from
Cramér–Rao bounds and are kept because RESULTS.md section 11 uses them to show how
misleading those bounds are here. Do not quote them as results.

---

## Nine things worth knowing before extending this

**Plot the scenes before computing anything from them.** The first pass over
WildfireSpreadTS returned r = −0.266 between a fire's daily advance and the wind: windier
days, slower fires. The sign was wrong, and the reason was visible the moment the scenes
were rendered — a 90 km satellite tile carries every VIIRS detection in it, not only the
fire it is named for, so one isolated detection 19 km away became a 19 km overnight run,
and one scene holds a rectangular swath artefact of 11 939 detections. Gating fixed it; no
amount of staring at the correlation would have. `RESULTS.md` section 14.1.

**A mean-field summary of a stochastic model is not that model.** Reading PyTorchFire's
one-step probability ellipse said its length-to-breadth ratio grows without bound. Running
it said the opposite: L/B peaks at 1.54 near 8 m/s and falls back, the one-step ellipse is
wrong by 44x at high wind, and a rotation test showed 19 % of the measured shape is the
lattice rather than the fire. Two other claims written before checking also failed. The
measurement cost minutes. RESULTS.md section 13.4.

**Look at the tail before quoting the median.** The state-level inversion's headline is a
median wind-speed error of 0.7 % in distribution and 0.8 % out of it. Both are true and both
hide a failure rate of 16 % and 36 %. Asking what the failures *were* — recompute the misfit
at the recovered parameters and at the true ones — turned an embarrassing caveat into the
sharpest result in the study: out of distribution they are solver failures that a
goodness-of-fit check catches without ground truth, while the mask-only failures fit the
data *exactly as well as the truth* and nothing can catch them. `RESULTS.md` section 12.4.

**Measure the assumption you are leaning on, if it can be measured.** This study spent a
long time reasoning carefully about a number it had inferred (how much the shape-to-wind
relation scatters). Downloading 895 real perimeters and one free weather API took under an
hour and replaced the inference with a measurement that was *larger* than the inferred
bracket — and surfaced something the reasoning had missed entirely: real fires cluster at
L/B ≈ 1.8, below the saturation regime the argument had been built around, weak for a
different reason. `RESULTS.md` section 3.

**Check the spread law against the published model, not against memory.** The head-rate
half of what this study assumes is Rothermel (1972) verbatim; the shape half was invented,
and FARSITE's actual relation (Anderson 1983, capped at LB = 8) behaves differently enough
to have changed the headline. `RESULTS.md` section 2.

**The spread rate is a support function.** Propagating a level set with the elliptical
wavelet's *radial* distance as the normal speed is wrong and quietly so: it starves the
flanks and the head reaches only ~41 % of the analytic distance. Huygens' principle is
reproduced by the support function `c·cos d + sqrt(a²cos²d + b²sin²d)`. Checked in
`validate_physics.py` against the exact Minkowski-sum burn (region IoU 0.986).

**Do not trust a Fisher matrix here without checking it against a profile likelihood.**
Forward-mode AD and finite differences disagreed by five orders of magnitude on the
mask-only bound, and both were wrong, for different reasons — the degeneracy is *curved*,
which finite differences cannot see, and the ENO minmod limiter zeroes derivative paths,
which corrupts AD. 

**Hold the CFL number fixed, not the time step.** At 9 m/s the default 15 s step puts
the CFL number at 0.68 and the level-set solution degrades enough to destroy the exact
null space the analysis depends on (alignment with the closed-form direction falls from
0.999999 to 0.17). `null_direction.cfl_scenario` picks the step instead.

**The plume renderer's extinction is calibrated, not arbitrary.** At the original value
the plume core reached optical depth ≈ 3200 — a pure silhouette, which preserves plume
geometry (and hence the wind) but destroys all information about emission strength.
`DEFAULT_EXTINCTION` targets τ ≈ 3.
