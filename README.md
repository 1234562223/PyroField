# PyroField — physics-state-level fusion for wildfire observation

A differentiable forward model in which every wildfire sensor is a **projection of one
physical state field**, plus the identifiability machinery to ask what each sensor
subset actually determines.

```
theta = (R0, U, theta_w, k_LB, w_buoy, Q)
   |
   v  level-set fire front + 3-D smoke transport
S(x, y, t)
   |
   +--> g_mask   burn mask            sigmoid(-phi / tau)
   +--> g_plume  camera image         differentiable volume rendering of the plume
   +--> g_conc   air-quality point    analytic Gaussian plume
```

Fusing these is then **inversion**, not feature concatenation, and the question
"is the state even determined?" becomes answerable.

## Status

Complete. The central result is algebraic, not statistical:

> A burn mask constrains the fire only through its outline: the head spread rate and the
> shape. Rothermel's head rate is `R0 (1 + phi_w + phi_s)`, so fuel and wind enter as a
> product and cannot be separated there — which leaves the shape as the only channel, and
> that channel is weak and saturating. Under FARSITE's own length-to-breadth law it stops
> responding to wind above 3.80 m/s, and below that it separates fuel from wind only if
> Anderson's single empirical curve is exact for the fuel present.

**The degeneracy is not this code's.** In [PyTorchFire](https://arxiv.org/pdf/2502.18738) —
a third-party differentiable stochastic cellular automaton sharing no implementation, no
numerical family and no spread law — the same two trades hold *exactly*: compensate an 80 %
wind error along the null curve and the eight lattice propagation probabilities are
unchanged to machine precision, so the transition kernel is the same object and the realised
fires are identical in **0 of 307 200 cells**. Not indistinguishable — identical. No
estimator of any kind can separate those parameter sets.

The closed-form null direction matches the measured one to six significant figures. Across
69 fires, parameter sets that reproduce an observed fire *identically* predict its arrival
at a road over a 6-minute window (up to 16); reading the smoke plume collapses that to the
right answer. On a slope the mask loses wind *direction* too. The degeneracy survives
patchy fuel and a wrong spread model; the plume advantage survives both as well, because it
measures wind through smoke advection and never routes through the fire model at all.

The load-bearing assumption behind all of that — how much the shape-to-wind relation
scatters between fuels — is measured rather than assumed, on **895 real NIFC fire
perimeters joined to ERA5 wind**. With no fire model at all, shape explains **2.3 %** of the
variance in wind; against Anderson's own relation it scatters by **46 %**, where separating
fuel from wind would need 8–21 %.

On **1044 fire-days from 201 real fires** in
[WildfireSpreadTS](https://doi.org/10.5281/zenodo.8006177) — VIIRS progression with the
GRIDMET wind that drove it — the daily advance explains about **1 %** of the variance in
wind speed, and under 6 % at the best measurement quality available; fuel and terrain
explain nine times more of it than the wind does. `fetch_wsts.py` reads that 45 GB archive
selectively with range requests, so reproducing this needs 17 GB, not 45.

A **channel ablation** on that benchmark's own next-day spread task — one U-net trained
seven times with channels zeroed, at matched budget, on all 607 fires — puts a number on
what the weather is worth to a learned spread model: the wind channels score **0.99x** the
fire mask alone, and removing them from the full 23-channel input **improves** it by 6 %.
What nearly triples the score is the raw VIIRS imagery, which shows the fire's thermal
signature directly. Two things kept that honest: two extra arms to separate imagery from
weather, without which the write-up would have claimed the opposite; and rerunning on 2.8x
the data once the full archive downloaded, which turned "the wind channels hurt" (0.86x on
714 pairs) into "they buy nothing" and reversed the terrain arm from 1.04x to 1.31x.

That ablation deliberately targets tomorrow's *newly* burning pixels rather than the
benchmark's published target of tomorrow's whole active fire, and the reason is measured
rather than asserted: on the published target **a rule that copies today's fire and reads
no channel at all scores 0.9474**, every one of the eight arms then lands between 0.985 and
0.992, and the channel set worth **2.67x** on the spread problem is worth **1.006x** there.

The same identifiability result decides a task nobody has scored: **backtracking a fire to
where it started**. The set of ignition points consistent with a burn depends on one thing
only — how well the spread rate is known — and the spread rate is the product a burn scar
cannot factor. Over **2135 backtracks on 121 real fires**, an oracle rate (±10 %) narrows
the origin to a third of the burn scar; the rate an investigator can actually obtain from
fuel and weather maps (a factor of 1.96, measured) narrows it to **100 % of the scar**. The
answer is "somewhere in the burned area", which is where you started.

All of that is diagnosis, and it implies one thing worth building: an inversion that
**reports a verdict per parameter rather than a number** — determined with an interval, or
not determined. It cannot use a residual test, because the failure that matters leaves no
residual; it cannot use a Fisher matrix, because the null space is a curve. What works is a
profile probe: force one parameter off by 30 %, let every other parameter re-absorb it, and
read the misfit. Scored on 62 inversions where the truth is known, it calls wind speed
determined in **88 %** of mask+plume cases and **17 %** of mask-only ones, and catches
**19 of 25** wrong answers against the residual check's 10.

The same probe rebuilds the **sensor sufficiency diagram** the plan asked for, without the
Cramér–Rao bounds that section 11 shows are wrong by 1500x between two implementations. It
answers "which sensor should I buy" with a terrain-dependent verdict: masks alone determine
one parameter of six on flat ground and **none at all** on a 20-degree slope, and a point
air-quality reading — worth nothing whatever on flat ground — recovers **five of the six**
on the slope.

And the study's own largest untested claim — that it had never inverted a **real** fire —
is now tested. Fitting the forward model to **23 real fires** from three starting winds
spanning 4x: the answers scatter by **25x** while the fit quality moves by 2x, the wind
speed lands **94 %** from GRIDMET, the bearing lands within 45° on **65 %** of fires
against 25 % for guessing, and the guard certifies the speed on **none** of them. The
obvious alternative explanation is testable and fails: the half of the fires the model
describes *well* fails to determine the wind exactly as badly as the half it describes
poorly. Getting there needed a bug fixed first — a CFL violation froze every fit at its
starting values and produced a spurious 5 % accuracy that was only the starting guess.

Fusing at the level of the physical state rather than at the level of learned features is
then tested directly, at matched backbone and budget. In distribution a mask-only CNN looks
*better* than the honest inversion on wind speed (10.3 % against 26.8 %) on a parameter
that is provably absent from its input; outside the training range it answers 4.80 m/s for
fires blowing at 6.79. Every mask-only inversion that lands far from the truth fits the
observations exactly as well as the truth does — the failure no residual check can catch,
and the reason identifiability has to be settled before accuracy.

Rerunning those inversion arms on **60 independently drawn fires** then moved one of this
repository's own headline numbers: the state-level out-of-distribution median went from
0.8 % to **5.0 %**, with an interval that excludes the original, and a claim that "the gap
widens eightfold when the data moves" had to be retracted — the gap narrows. The cause was
a bimodal distribution whose median is unstable near a 50 % failure rate, which the
write-up had already identified one subsection before quoting the median anyway.

See `RESULTS.md`, including twenty-two hypotheses the data corrected — three from bugs of mine,
and one from checking my own physics against the published model instead of my memory of
it.

## Layout

```
pyrofield/
  physics/     rothermel.py   elliptical spread rate (support function, not radius)
               levelset.py    ENO2 upwind level-set propagation + reinitialisation
               smoke.py       semi-Lagrangian 3-D smoke transport
               interp.py      trilinear sampling in primitive ops (forward-AD-able)
  models/operators/
               mask.py            burn-mask operator
               plume_render.py    differentiable volume rendering
               gaussian_plume.py  analytic point-concentration operator
  sim/         synthetic.py   the coupled forward model and scenario definition
  theory/      identifiability.py  Fisher information, CRB, sensor-subset lattice
  eval/        inversion.py   Levenberg-Marquardt with coarse-to-fine image fitting
               guarded.py     inversion that reports a verdict per parameter, not a number
scripts/       experiments and figures (see below)
data/          downloaded NIFC perimeters + ERA5 wind, generated .npz splits
results/       json + logs
figures/       png
```

## Running things

Dependencies: `torch`, `numpy`, `matplotlib`. A 24 GB consumer GPU is ample — peak
usage is about 0.5 GB and the full forward model takes 0.3–0.6 s.

Correctness first, then the results:

```bash
python scripts/validate_physics.py      # physics vs closed-form solutions
python scripts/test_invariants.py       # regression tests on the invariants
python scripts/verify_results.py        # every number in RESULTS.md vs its source file
python scripts/null_direction.py        # the exact degeneracy, in closed form   <-- start here
```

`verify_results.py` is the one to run after changing anything. It re-derives all **169**
headline figures from `results/*.json` and fails if the write-up disagrees, then audits
cross-references, figures, scripts and encodings. A stale number that was correct when it
was written is the most likely error in a repository this size, and it has happened here.

For writing up: **`PAPER_TABLES.md`** holds all 27 headline tables, each rendered as
markdown to read and as a `\begin{table}` block to paste, plus the figure manifest — and is
generated from the result files rather than typed, so it cannot drift from them.

```bash
python scripts/make_paper_tables.py     # regenerate PAPER_TABLES.md from results/*.json
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
