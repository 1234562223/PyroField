# Paper tables

Generated from `results/*.json` by `scripts/make_paper_tables.py` on 2026-09-19. Do not edit by hand — rerun the script, then `scripts/verify_results.py`.



## T1 — The null direction, closed form against measurement


**Analytic null direction of the mask-only Fisher matrix against its measured weakest eigenvector, at CFL-matched time steps.**  
<sub>Section 1 · `scripts/null_direction.py`</sub>

| wind (m/s) | CFL | analytic v | measured | abs(cos) |
|---|---|---|---|---|
| 2.0 | 0.19 | (-0.3915, +0.6507, -0.6507) | (-0.3919, +0.6494, -0.6517) | 0.999999 |
| 3.0 | 0.25 | (-0.4789, +0.6207, -0.6207) | (-0.4794, +0.6184, -0.6227) | 0.999995 |
| 4.0 | 0.31 | (-0.5299, +0.5997, -0.5997) | (-0.5301, +0.5991, -0.6000) | 1.000000 |
| 6.0 | 0.35 | (-0.5839, +0.5741, -0.5741) | (-0.5851, +0.5767, -0.5701) | 0.999988 |
| 9.0 | 0.35 | (-0.6197, +0.5550, -0.5550) | (-0.6216, +0.5595, -0.5482) | 0.999965 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Analytic null direction of the mask-only Fisher matrix against its measured weakest eigenvector, at CFL-matched time steps.}
\begin{tabular}{lrrrr}\toprule
wind (m/s) & CFL & analytic v & measured & abs(cos) \\ \midrule
2.0 & 0.19 & (-0.3915, +0.6507, -0.6507) & (-0.3919, +0.6494, -0.6517) & 0.999999 \\
3.0 & 0.25 & (-0.4789, +0.6207, -0.6207) & (-0.4794, +0.6184, -0.6227) & 0.999995 \\
4.0 & 0.31 & (-0.5299, +0.5997, -0.5997) & (-0.5301, +0.5991, -0.6000) & 1.000000 \\
6.0 & 0.35 & (-0.5839, +0.5741, -0.5741) & (-0.5851, +0.5767, -0.5701) & 0.999988 \\
9.0 & 0.35 & (-0.6197, +0.5550, -0.5550) & (-0.6216, +0.5595, -0.5482) & 0.999965 \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Cost of a forced wind error, compensated along the exact null curve, along its tangent, and not compensated.**  
<sub>Section 1 · `scripts/null_direction.py`</sub>

| wind error | exact curve | tangent | no compensation |
|---|---|---|---|
| 5 % | 0.52 | 1.8 | 8,214 |
| 10 % | 0.13 | 14.8 | 31,728 |
| 20 % | 0.52 | 179.1 | 112,735 |
| 30 % | 0.20 | 790.8 | 220,364 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Cost of a forced wind error, compensated along the exact null curve, along its tangent, and not compensated.}
\begin{tabular}{lrrr}\toprule
wind error & exact curve & tangent & no compensation \\ \midrule
5 \% & 0.52 & 1.8 & 8,214 \\
10 \% & 0.13 & 14.8 & 31,728 \\
20 \% & 0.52 & 179.1 & 112,735 \\
30 \% & 0.20 & 790.8 & 220,364 \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T2 — Under FARSITE's own shape law


Anderson (1983) truncated at LB = 8; the cap is reached at **U = 3.80 m/s**.



**Shape-to-wind elasticity and the wind precision it implies, for the inferred bracket of 15 %–35 % relative scatter in the length-to-breadth relation. Section 3 replaces that bracket with a measurement of 46 %.**  
<sub>Sections 2 and 3 · `scripts/lb_scatter.py, scripts/real_shape_vs_wind.py`</sub>

| U (m/s) | LB | elasticity dlnLB/dlnU | sigma(U) at 15 % | sigma(U) at 35 % | sigma(U) at 46 % |
|---|---|---|---|---|---|
| 1.5 | 2.09 | 0.843 | 18 % | 41 % | 55 % |
| 2.0 | 2.78 | 1.159 | 13 % | 30 % | 40 % |
| 2.5 | 3.73 | 1.468 | 10 % | 24 % | 31 % |
| 3.0 | 5.00 | 1.769 | 8 % | 20 % | 26 % |
| 3.5 | 6.72 | 2.062 | 7 % | 17 % | 22 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Shape-to-wind elasticity and the wind precision it implies, for the inferred bracket of 15 \%–35 \% relative scatter in the length-to-breadth relation. Section 3 replaces that bracket with a measurement of 46 \%.}
\begin{tabular}{lrrrrr}\toprule
U (m/s) & LB & elasticity dlnLB/dlnU & sigma(U) at 15 \% & sigma(U) at 35 \% & sigma(U) at 46 \% \\ \midrule
1.5 & 2.09 & 0.843 & 18 \% & 41 \% & 55 \% \\
2.0 & 2.78 & 1.159 & 13 \% & 30 \% & 40 \% \\
2.5 & 3.73 & 1.468 & 10 \% & 24 \% & 31 \% \\
3.0 & 5.00 & 1.769 & 8 \% & 20 \% & 26 \% \\
3.5 & 6.72 & 2.062 & 7 \% & 17 \% & 22 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


The operational tolerance is 10 % on wind speed. Every row of the measured column misses it.



## T3 — Real fire perimeters (NIFC WFIGS) and ERA5 wind


**How well an area-matched moment ellipse describes a real final fire perimeter, and how well posed its length-to-breadth ratio is.**  
<sub>Section 3 · `scripts/real_perimeters.py, scripts/real_lb_spread.py`</sub>

| quantity | value |
|---|---|
| perimeters | 895 |
| median IoU with the best moment ellipse | 0.705 |
| perimeters above IoU 0.90 | 0 |
| perimeters below IoU 0.70 | 48 % |
| not a single connected blob | 51.5 % |
| median L/B | 1.80 |
| above FARSITE's cap of 8 | 0.11 % |
| L/B spread across 5 estimators (median rel. sd) | 6.7 % |
| fires where estimators differ by > 20 % | 32.4 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{How well an area-matched moment ellipse describes a real final fire perimeter, and how well posed its length-to-breadth ratio is.}
\begin{tabular}{lr}\toprule
quantity & value \\ \midrule
perimeters & 895 \\
median IoU with the best moment ellipse & 0.705 \\
perimeters above IoU 0.90 & 0 \\
perimeters below IoU 0.70 & 48 \% \\
not a single connected blob & 51.5 \% \\
median L/B & 1.80 \\
above FARSITE's cap of 8 & 0.11 \% \\
L/B spread across 5 estimators (median rel. sd) & 6.7 \% \\
fires where estimators differ by > 20 \% & 32.4 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Wind observed by ERA5 within bins of observed fire shape. No fire model and no fitted parameter.**  
<sub>Section 3 · `scripts/real_shape_vs_wind.py`</sub>

| L/B bin | n | wind p25 | median | p75 | p10–p90 spread |
|---|---|---|---|---|---|
| 1.0–1.4 | 104 | 3.77 | 4.41 | 5.51 | 76 % |
| 1.4–1.8 | 120 | 3.95 | 4.64 | 5.59 | 72 % |
| 1.8–2.3 | 95 | 3.69 | 4.75 | 5.56 | 69 % |
| 2.3–3.0 | 69 | 3.62 | 4.46 | 5.44 | 82 % |
| 3.0–4.5 | 44 | 3.99 | 5.44 | 6.36 | 75 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Wind observed by ERA5 within bins of observed fire shape. No fire model and no fitted parameter.}
\begin{tabular}{lrrrrr}\toprule
L/B bin & n & wind p25 & median & p75 & p10–p90 spread \\ \midrule
1.0–1.4 & 104 & 3.77 & 4.41 & 5.51 & 76 \% \\
1.4–1.8 & 120 & 3.95 & 4.64 & 5.59 & 72 \% \\
1.8–2.3 & 95 & 3.69 & 4.75 & 5.56 & 69 \% \\
2.3–3.0 & 69 & 3.62 & 4.46 & 5.44 & 82 \% \\
3.0–4.5 & 44 & 3.99 & 5.44 & 6.36 & 75 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


Model-free correlation of log L/B with log wind: **r = 0.151** (r² = 0.023). Against Anderson's relation with the wind adjustment factor fitted at 0.244: **r = 0.387**, residual scatter **46 %**. Alexander (1985) reported r = 0.865 on experimental and curated fires.



## T4 — Feature-level against physical-state-level fusion


**Median relative error in wind speed, in and out of the training distribution. One architecture, matched capacity and training budget; only the information differs.**  
<sub>Section 12 · `scripts/fusion_level.py`</sub>

| arm | in-distribution | out-of-distribution | degradation |
|---|---|---|---|
| prior (train median) | 21.8 % | 48.4 % | 2.2x |
| CNN mask only | 10.3 % | 30.6 % | 3.0x |
| CNN mask + plume | 1.8 % | 16.5 % | 9.1x |
| state mask only | 26.8 % | 19.2 % | 0.7x |
| state mask + plume | 0.7 % | 0.8 % | 1.1x |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Median relative error in wind speed, in and out of the training distribution. One architecture, matched capacity and training budget; only the information differs.}
\begin{tabular}{lrrr}\toprule
arm & in-distribution & out-of-distribution & degradation \\ \midrule
prior (train median) & 21.8 \% & 48.4 \% & 2.2x \\
CNN mask only & 10.3 \% & 30.6 \% & 3.0x \\
CNN mask + plume & 1.8 \% & 16.5 \% & 9.1x \\
state mask only & 26.8 \% & 19.2 \% & 0.7x \\
state mask + plume & 0.7 \% & 0.8 \% & 1.1x \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Whether each failed inversion had the information available. Misfit recomputed at the recovered parameters and at the true ones, on the same observations.**  
<sub>Section 12.4 · `scripts/fusion_failure_mode.py`</sub>

| arm | failures | of which optimiser failures |
|---|---|---|
| state mask + plume, test_id | 4 of 25 | 2 (50 %) |
| state mask + plume, test_ood | 9 of 25 | 8 (89 %) |
| state mask only, test_id | 12 of 12 | 0 (0 %) |
| state mask only, test_ood | 11 of 12 | 4 (36 %) |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Whether each failed inversion had the information available. Misfit recomputed at the recovered parameters and at the true ones, on the same observations.}
\begin{tabular}{lrr}\toprule
arm & failures & of which optimiser failures \\ \midrule
state mask + plume, test_id & 4 of 25 & 2 (50 \%) \\
state mask + plume, test_ood & 9 of 25 & 8 (89 \%) \\
state mask only, test_id & 12 of 12 & 0 (0 \%) \\
state mask only, test_ood & 11 of 12 & 4 (36 \%) \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Three starts, screened cheaply with the best refined, selected by misfit and therefore without ground truth.**  
<sub>Section 12.5 · `scripts/fusion_multistart.py`</sub>

| arm | median | p90 | failures > 5 % |
|---|---|---|---|
| state mask + plume, test_id | 0.7 % → 0.4 % | 28 % → 1 % | 16 % → 4 % |
| state mask + plume, test_ood | 0.8 % → 0.8 % | 31 % → 15 % | 36 % → 20 % |
| state mask only, test_id | 26.8 % → 19.1 % | 82 % → 41 % | 100 % → 83 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Three starts, screened cheaply with the best refined, selected by misfit and therefore without ground truth.}
\begin{tabular}{lrrr}\toprule
arm & median & p90 & failures > 5 \% \\ \midrule
state mask + plume, test_id & 0.7 \% → 0.4 \% & 28 \% → 1 \% & 16 \% → 4 \% \\
state mask + plume, test_ood & 0.8 \% → 0.8 \% & 31 \% → 15 \% & 36 \% → 20 \% \\
state mask only, test_id & 26.8 \% → 19.1 \% & 82 \% → 41 \% & 100 \% → 83 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


**The state-level arms rerun on 60 independently drawn fires with a different seed, with bootstrap intervals. Only one of the three original point estimates falls inside the new interval.**  
<sub>Section 12.6 · `scripts/fusion_scale.py`</sub>

| arm | n = 25/12 | n = 60/30 | 95 % interval | failures |
|---|---|---|---|---|
| mask + plume, test_id | 0.7 % | **0.3 %** | [0.3, 0.6] | 12 % [5, 20] |
| mask + plume, test_ood | 0.8 % | **5.0 %** | [1.6, 13.2] | 50 % [37, 63] |
| mask only, test_id | 26.8 % | **26.9 %** | [15.1, 50.7] | 83 % [70, 97] |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{The state-level arms rerun on 60 independently drawn fires with a different seed, with bootstrap intervals. Only one of the three original point estimates falls inside the new interval.}
\begin{tabular}{lrrrr}\toprule
arm & n = 25/12 & n = 60/30 & 95 \% interval & failures \\ \midrule
mask + plume, test_id & 0.7 \% & **0.3 \%** & [0.3, 0.6] & 12 \% [5, 20] \\
mask + plume, test_ood & 0.8 \% & **5.0 \%** & [1.6, 13.2] & 50 \% [37, 63] \\
mask only, test_id & 26.8 \% & **26.9 \%** & [15.1, 50.7] & 83 \% [70, 97] \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T5 — The same degeneracy in a third-party simulator (PyTorchFire)


**Compensating a wind error along the exact null curve of PyTorchFire's own propagation law, and the realised fires on matched random seeds.**  
<sub>Section 13 · `scripts/independent_ca.py`</sub>

| wind error | max change in the 8 probabilities | burn cells differing | uncompensated |
|---|---|---|---|
| 5 % | 0.0e+00 | — | — |
| 10 % | 1.1e-16 | 0.0000 % | 4.25 % |
| 20 % | 0.0e+00 | — | — |
| 30 % | 0.0e+00 | 0.0000 % | 10.80 % |
| 50 % | 1.1e-16 | 0.0000 % | 15.99 % |
| 80 % | 1.1e-16 | 0.0000 % | 20.34 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Compensating a wind error along the exact null curve of PyTorchFire's own propagation law, and the realised fires on matched random seeds.}
\begin{tabular}{lrrr}\toprule
wind error & max change in the 8 probabilities & burn cells differing & uncompensated \\ \midrule
5 \% & 0.0e+00 & — & — \\
10 \% & 1.1e-16 & 0.0000 \% & 4.25 \% \\
20 \% & 0.0e+00 & — & — \\
30 \% & 0.0e+00 & 0.0000 \% & 10.80 \% \\
50 \% & 1.1e-16 & 0.0000 \% & 15.99 \% \\
80 \% & 1.1e-16 & 0.0000 \% & 20.34 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Smallest residual achievable after a +30 % wind error, by which coefficients are free to move, and for a wind rotation.**  
<sub>Section 13.2 · `scripts/independent_ca.py`</sub>

| what is free | residual | verdict |
|---|---|---|
| p_h and c_2 both free | 0.00e+00 | degenerate |
| only p_h free (shape law known) | 9.34e-02 | identifiable |
| only c_2 free (fuel known) | 4.24e-02 | identifiable |
| neither free | 9.66e-02 | identifiable |
| 5° wind rotation, both free | 3.39e-02 | identifiable |
| 10° wind rotation, both free | 6.75e-02 | identifiable |
| 20° wind rotation, both free | 1.33e-01 | identifiable |
| 45° wind rotation, both free | 2.72e-01 | identifiable |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Smallest residual achievable after a +30 \% wind error, by which coefficients are free to move, and for a wind rotation.}
\begin{tabular}{lrr}\toprule
what is free & residual & verdict \\ \midrule
p_h and c_2 both free & 0.00e+00 & degenerate \\
only p_h free (shape law known) & 9.34e-02 & identifiable \\
only c_2 free (fuel known) & 4.24e-02 & identifiable \\
neither free & 9.66e-02 & identifiable \\
5° wind rotation, both free & 3.39e-02 & identifiable \\
10° wind rotation, both free & 6.75e-02 & identifiable \\
20° wind rotation, both free & 1.33e-01 & identifiable \\
45° wind rotation, both free & 2.72e-01 & identifiable \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T6 — Real satellite fire progression against real wind (WildfireSpreadTS)


**Association between a fire's daily advance and the GRIDMET wind that drove it, over 1044 fire-days from 201 fires. Four advance measures, reported together because they fail differently.**  
<sub>Section 14.2 · `scripts/wsts_advance.py`</sub>

| advance measure | n | r (log–log) | r² |
|---|---|---|---|
| distance to front, p90 | 1044 | 0.096 | 0.0093 |
| distance to front, p75 | 1044 | 0.083 | 0.0068 |
| distance to front, median | 1044 | 0.076 | 0.0058 |
| equivalent-radius growth | 1044 | 0.118 | 0.0139 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Association between a fire's daily advance and the GRIDMET wind that drove it, over 1044 fire-days from 201 fires. Four advance measures, reported together because they fail differently.}
\begin{tabular}{lrrr}\toprule
advance measure & n & r (log–log) & r² \\ \midrule
distance to front, p90 & 1044 & 0.096 & 0.0093 \\
distance to front, p75 & 1044 & 0.083 & 0.0068 \\
distance to front, median & 1044 & 0.076 & 0.0058 \\
equivalent-radius growth & 1044 & 0.118 & 0.0139 \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Both channels sharpen with measurement quality, which is why this data cannot separate a missing signal from a badly measured one.**  
<sub>Section 14.4 · `scripts/wsts_advance.py`</sub>

| detections required that day | bearing vs wind direction (R) | advance vs wind speed (r) |
|---|---|---|
| ≥ 8 | 0.146 | 0.118 |
| ≥ 20 | 0.170 | 0.148 |
| ≥ 50 | 0.156 | 0.191 |
| ≥ 100 | 0.189 | 0.192 |
| ≥ 200 | 0.328 | 0.234 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Both channels sharpen with measurement quality, which is why this data cannot separate a missing signal from a badly measured one.}
\begin{tabular}{lrr}\toprule
detections required that day & bearing vs wind direction (R) & advance vs wind speed (r) \\ \midrule
≥ 8 & 0.146 & 0.118 \\
≥ 20 & 0.170 & 0.148 \\
≥ 50 & 0.156 & 0.191 \\
≥ 100 & 0.189 & 0.192 \\
≥ 200 & 0.328 & 0.234 \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Regressing log daily advance on the wind and on the fuel and terrain covariates the dataset carries.**  
<sub>Section 14.3 · `scripts/wsts_advance.py`</sub>

| model | R² | wind coefficient | s.e. | t |
|---|---|---|---|---|
| wind only | 0.0093 | 0.192 | 0.0615 | 3.1 |
| + energy release component | 0.0297 | 0.231 | 0.0614 | 3.8 |
| + ERC, NDVI | 0.0624 | 0.222 | 0.0628 | 3.5 |
| + ERC, NDVI, slope | 0.0839 | 0.198 | 0.0623 | 3.2 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Regressing log daily advance on the wind and on the fuel and terrain covariates the dataset carries.}
\begin{tabular}{lrrrr}\toprule
model & R² & wind coefficient & s.e. & t \\ \midrule
wind only & 0.0093 & 0.192 & 0.0615 & 3.1 \\
+ energy release component & 0.0297 & 0.231 & 0.0614 & 3.8 \\
+ ERC, NDVI & 0.0624 & 0.222 & 0.0628 & 3.5 \\
+ ERC, NDVI, slope & 0.0839 & 0.198 & 0.0623 & 3.2 \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T7 — Channel ablation on the benchmark's next-day spread task


**Test average precision, one U-net of 486,321 parameters trained once per arm with the other channels zeroed, two seeds. 1995 training pairs, 1430 test pairs, 0.36 % positive pixels.**  
<sub>Section 15 · `scripts/wsts_baseline.py`</sub>

| input channels | test AP | vs fire mask only |
|---|---|---|
| chance | 0.0036 | — |
| persistence (ring around today's fire) | 0.0755 | 0.66x |
| fire | 0.1139 ± 0.0007 | 1.00x |
| fire + wind | 0.1131 ± 0.0021 | 0.99x |
| fire + weather | 0.1180 ± 0.0087 | 1.04x |
| fire + static | 0.1488 ± 0.0035 | 1.31x |
| fire + VIIRS | 0.3046 ± 0.0040 | 2.67x |
| all minus wind | 0.2844 ± 0.0032 | 2.50x |
| all | 0.2668 ± 0.0090 | 2.34x |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Test average precision, one U-net of 486,321 parameters trained once per arm with the other channels zeroed, two seeds. 1995 training pairs, 1430 test pairs, 0.36 \% positive pixels.}
\begin{tabular}{lrr}\toprule
input channels & test AP & vs fire mask only \\ \midrule
chance & 0.0036 & — \\
persistence (ring around today's fire) & 0.0755 & 0.66x \\
fire & 0.1139 $\pm$ 0.0007 & 1.00x \\
fire + wind & 0.1131 $\pm$ 0.0021 & 0.99x \\
fire + weather & 0.1180 $\pm$ 0.0087 & 1.04x \\
fire + static & 0.1488 $\pm$ 0.0035 & 1.31x \\
fire + VIIRS & 0.3046 $\pm$ 0.0040 & 2.67x \\
all minus wind & 0.2844 $\pm$ 0.0032 & 2.50x \\
all & 0.2668 $\pm$ 0.0090 & 2.34x \\
\bottomrule\end{tabular}\end{table}
```
</details>


**The same arms by the wind on the day. Nothing improves when there is more wind.**  
<sub>Section 15.3 · `scripts/wsts_baseline.py`</sub>

| wind | pairs | fire | fire + wind | all minus wind | all |
|---|---|---|---|---|---|
| < 2.9 m/s | 472 | 0.1202 | 0.1212 | 0.2830 | 0.2960 |
| 2.9 - 3.8 | 474 | 0.1100 | 0.1103 | 0.2931 | 0.2836 |
| >= 3.8 m/s | 484 | 0.1101 | 0.1055 | 0.2700 | 0.2321 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{The same arms by the wind on the day. Nothing improves when there is more wind.}
\begin{tabular}{lrrrrr}\toprule
wind & pairs & fire & fire + wind & all minus wind & all \\ \midrule
< 2.9 m/s & 472 & 0.1202 & 0.1212 & 0.2830 & 0.2960 \\
2.9 - 3.8 & 474 & 0.1100 & 0.1103 & 0.2931 & 0.2836 \\
>= 3.8 m/s & 484 & 0.1101 & 0.1055 & 0.2700 & 0.2321 \\
\bottomrule\end{tabular}\end{table}
```
</details>


**The same ablation on the spread-only target and on WildfireSpreadTS's published target, nothing else changed. On the published target a rule that copies today's fire and reads no channel already scores 0.9474, and every arm lands within 0.7 % of every other.**  
<sub>Section 15.4 · `scripts/wsts_baseline.py, scripts/wsts_trivial.py`</sub>

| arm | new-fire AP | vs fire | published AP | vs fire |
|---|---|---|---|---|
| **copy today's fire unchanged** | 0.0036 | — | **0.9474** | — |
| ring around today's fire | 0.0755 | 0.66x | 0.0646 | — |
| fire | 0.1139 | 1.00x | 0.9861 | 1.000x |
| fire + wind | 0.1131 | 0.99x | 0.9852 | 0.999x |
| fire + weather | 0.1180 | 1.04x | 0.9865 | 1.000x |
| fire + static | 0.1488 | 1.31x | 0.9866 | 1.001x |
| fire + VIIRS | 0.3046 | 2.67x | 0.9922 | 1.006x |
| all minus wind | 0.2844 | 2.50x | 0.9919 | 1.006x |
| all | 0.2668 | 2.34x | 0.9918 | 1.006x |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{The same ablation on the spread-only target and on WildfireSpreadTS's published target, nothing else changed. On the published target a rule that copies today's fire and reads no channel already scores 0.9474, and every arm lands within 0.7 \% of every other.}
\begin{tabular}{lrrrr}\toprule
arm & new-fire AP & vs fire & published AP & vs fire \\ \midrule
**copy today's fire unchanged** & 0.0036 & — & **0.9474** & — \\
ring around today's fire & 0.0755 & 0.66x & 0.0646 & — \\
fire & 0.1139 & 1.00x & 0.9861 & 1.000x \\
fire + wind & 0.1131 & 0.99x & 0.9852 & 0.999x \\
fire + weather & 0.1180 & 1.04x & 0.9865 & 1.000x \\
fire + static & 0.1488 & 1.31x & 0.9866 & 1.001x \\
fire + VIIRS & 0.3046 & 2.67x & 0.9922 & 1.006x \\
all minus wind & 0.2844 & 2.50x & 0.9919 & 1.006x \\
all & 0.2668 & 2.34x & 0.9918 & 1.006x \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T8 — Backtracking real fires to their ignition


**Feasible ignition area by how well the spread rate is known, over 2135 backtracks on 121 real fires. Coverage is 100 % by construction for the centred regimes; the informative column is the area.**  
<sub>Section 16 · `scripts/retrodiction.py`</sub>

| rate known to | median feasible area | as % of the burn scar | coverage |
|---|---|---|---|
| oracle (+/-10 %) | 14.3 km² | 34 % | 100 % |
| covariates (x1.96) | 47.1 km² | 100 % | 100 % |
| prior only (x2.01) | 47.1 km² | 100 % | 100 % |
| history | 38.1 km² | 94 % | 85 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Feasible ignition area by how well the spread rate is known, over 2135 backtracks on 121 real fires. Coverage is 100 \% by construction for the centred regimes; the informative column is the area.}
\begin{tabular}{lrrr}\toprule
rate known to & median feasible area & as \% of the burn scar & coverage \\ \midrule
oracle (+/-10 \%) & 14.3 km² & 34 \% & 100 \% \\
covariates (x1.96) & 47.1 km² & 100 \% & 100 \% \\
prior only (x2.01) & 47.1 km² & 100 \% & 100 \% \\
history & 38.1 km² & 94 \% & 85 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


**The same, by how long the fire had been burning. The burn scar grows; the fraction of it the origin could lie in does not shrink.**  
<sub>Section 16.1 · `scripts/retrodiction.py`</sub>

| days burning | backtracks | median burn | oracle | covariates |
|---|---|---|---|---|
| 3 - 5 | 333 | 25 km² | 7.2 km² (32 %) | 25.0 km² (100 %) |
| 6 - 9 | 447 | 30 km² | 8.9 km² (33 %) | 29.7 km² (100 %) |
| 10 - 15 | 508 | 40 km² | 12.2 km² (35 %) | 39.8 km² (100 %) |
| >= 16 | 847 | 106 km² | 40.4 km² (36 %) | 106.3 km² (100 %) |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{The same, by how long the fire had been burning. The burn scar grows; the fraction of it the origin could lie in does not shrink.}
\begin{tabular}{lrrrr}\toprule
days burning & backtracks & median burn & oracle & covariates \\ \midrule
3 - 5 & 333 & 25 km² & 7.2 km² (32 \%) & 25.0 km² (100 \%) \\
6 - 9 & 447 & 30 km² & 8.9 km² (33 \%) & 29.7 km² (100 \%) \\
10 - 15 & 508 & 40 km² & 12.2 km² (35 \%) & 39.8 km² (100 \%) \\
>= 16 & 847 & 106 km² & 40.4 km² (36 \%) & 106.3 km² (100 \%) \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T9 — An inversion that reports what the observations determine


**Verdict on wind speed, with the guard not told which sensor set it is looking at. A 30 % perturbation is imposed, every other parameter is allowed to re-absorb it, and the misfit is read.**  
<sub>Section 17.1 · `scripts/guarded_inversion.py`</sub>

| arm | wind speed called determined | median error in wind speed |
|---|---|---|
| mask + plume, test_id | 88 % | 0.7 % |
| mask + plume, test_ood | 72 % | 0.8 % |
| mask only, test_id | 17 % | 26.8 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Verdict on wind speed, with the guard not told which sensor set it is looking at. A 30 \% perturbation is imposed, every other parameter is allowed to re-absorb it, and the misfit is read.}
\begin{tabular}{lrr}\toprule
arm & wind speed called determined & median error in wind speed \\ \midrule
mask + plume, test_id & 88 \% & 0.7 \% \\
mask + plume, test_ood & 72 \% & 0.8 \% \\
mask only, test_id & 17 \% & 26.8 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Both tests on the same 62 inversions, scored against whether the answer was in fact wrong by more than 5 % on wind speed.**  
<sub>Section 17.2 · `scripts/guarded_inversion.py`</sub>

| raised by | wrong, caught | wrong, missed | right, flagged |
|---|---|---|---|
| residual check | 10 | 15 | 2 |
| profile guard | 19 | 6 | 1 |
| either | 21 | 4 | 2 |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Both tests on the same 62 inversions, scored against whether the answer was in fact wrong by more than 5 \% on wind speed.}
\begin{tabular}{lrrr}\toprule
raised by & wrong, caught & wrong, missed & right, flagged \\ \midrule
residual check & 10 & 15 & 2 \\
profile guard & 19 & 6 & 1 \\
either & 21 & 4 & 2 \\
\bottomrule\end{tabular}\end{table}
```
</details>


Interval calibration, pooled over the cases that both passed the guard and converged: **22 of 36 = 61 %**, 95 % interval 43–77 %. The nominal 68 % for a 1-sigma interval is inside that; scaling sigma by 1.06 would land it exactly on 68 %, which 36 cases cannot resolve.



## T10 — Sensor sufficiency, by profile probe rather than Cramér–Rao bound


**Which sensor subsets determine which state components, flat, 3 m/s. Entries are the 1-sigma precision where the parameter is determined; a cell is undetermined when forcing it off by 30 % costs a misfit below 4 with every other parameter free to re-absorb it.**  
<sub>Section 17.4 · `scripts/sufficiency_profile.py`</sub>

| sensors | R0 | U | theta_w | lb_k | w_buoy | Q |
|---|---|---|---|---|---|---|
| masks | **not determined** | **not determined** | 0.046° | **not determined** | **not determined** | **not determined** |
| masks + air quality | **not determined** | **not determined** | 0.042° | **not determined** | **not determined** | **not determined** |
| masks + plume | 0.33 % | 0.40 % | 0.044° | 0.50 % | 0.40 % | 0.46 % |
| all three | 0.33 % | 0.40 % | 0.038° | 0.50 % | 0.40 % | 0.45 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Which sensor subsets determine which state components, flat, 3 m/s. Entries are the 1-sigma precision where the parameter is determined; a cell is undetermined when forcing it off by 30 \% costs a misfit below 4 with every other parameter free to re-absorb it.}
\begin{tabular}{lrrrrrr}\toprule
sensors & R0 & U & theta_w & lb_k & w_buoy & Q \\ \midrule
masks & **not determined** & **not determined** & 0.046° & **not determined** & **not determined** & **not determined** \\
masks + air quality & **not determined** & **not determined** & 0.042° & **not determined** & **not determined** & **not determined** \\
masks + plume & 0.33 \% & 0.40 \% & 0.044° & 0.50 \% & 0.40 \% & 0.46 \% \\
all three & 0.33 \% & 0.40 \% & 0.038° & 0.50 \% & 0.40 \% & 0.45 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


**Which sensor subsets determine which state components, 20 deg slope, 3 m/s. Entries are the 1-sigma precision where the parameter is determined; a cell is undetermined when forcing it off by 30 % costs a misfit below 4 with every other parameter free to re-absorb it.**  
<sub>Section 17.4 · `scripts/sufficiency_profile.py`</sub>

| sensors | R0 | U | theta_w | lb_k | w_buoy | Q |
|---|---|---|---|---|---|---|
| masks | **not determined** | **not determined** | **not determined** | **not determined** | **not determined** | **not determined** |
| masks + air quality | 0.64 % | 0.85 % | 0.233° | 0.83 % | **not determined** | 1.26 % |
| masks + plume | 0.28 % | 0.36 % | 0.103° | 0.46 % | 0.41 % | 0.50 % |
| all three | 0.26 % | 0.33 % | 0.094° | 0.42 % | 0.37 % | 0.41 % |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Which sensor subsets determine which state components, 20 deg slope, 3 m/s. Entries are the 1-sigma precision where the parameter is determined; a cell is undetermined when forcing it off by 30 \% costs a misfit below 4 with every other parameter free to re-absorb it.}
\begin{tabular}{lrrrrrr}\toprule
sensors & R0 & U & theta_w & lb_k & w_buoy & Q \\ \midrule
masks & **not determined** & **not determined** & **not determined** & **not determined** & **not determined** & **not determined** \\
masks + air quality & 0.64 \% & 0.85 \% & 0.233° & 0.83 \% & **not determined** & 1.26 \% \\
masks + plume & 0.28 \% & 0.36 \% & 0.103° & 0.46 \% & 0.41 \% & 0.50 \% \\
all three & 0.26 \% & 0.33 \% & 0.094° & 0.42 \% & 0.37 \% & 0.41 \% \\
\bottomrule\end{tabular}\end{table}
```
</details>


## T11 — Inverting real fires against the wind that actually blew


**Each of 23 real WildfireSpreadTS fires fitted three times, from 1.5, 3.0, 6.0 m/s — a 4x span of plausible winds. If the burn determined the wind the three would converge; the answers scatter by 25x while the fit quality moves by 2x.**  
<sub>Section 18 · `scripts/real_fire_inversion.py`</sub>

| quantity | median | p25 / p90 | for scale |
|---|---|---|---|
| IoU of fitted front with observed burn | 0.376 | 0.227 / 0.462 | 0.705 for the best ellipse (T3) |
| largest / smallest **fitted wind**, per fire | **25.4x** | — / 29.7x | 4x span of starts |
| largest / smallest **final misfit**, per fire | 1.92x | — / 2.79x | — |
| wind speed, error against GRIDMET | **94 %** | 88 / 95 % | — |
| wind bearing, absolute error | **29°** | 15 / 56° | 90° if guessing |
| bearings within 45° | 65 % | — | 25 % if guessing |
| guard calls the **speed** determined | **0 %** | — | — |
| guard calls the **bearing** determined | 26 % | — | — |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Each of 23 real WildfireSpreadTS fires fitted three times, from 1.5, 3.0, 6.0 m/s — a 4x span of plausible winds. If the burn determined the wind the three would converge; the answers scatter by 25x while the fit quality moves by 2x.}
\begin{tabular}{lrrr}\toprule
quantity & median & p25 / p90 & for scale \\ \midrule
IoU of fitted front with observed burn & 0.376 & 0.227 / 0.462 & 0.705 for the best ellipse (T3) \\
largest / smallest **fitted wind**, per fire & **25.4x** & — / 29.7x & 4x span of starts \\
largest / smallest **final misfit**, per fire & 1.92x & — / 2.79x & — \\
wind speed, error against GRIDMET & **94 \%** & 88 / 95 \% & — \\
wind bearing, absolute error & **29°** & 15 / 56° & 90° if guessing \\
bearings within 45° & 65 \% & — & 25 \% if guessing \\
guard calls the **speed** determined & **0 \%** & — & — \\
guard calls the **bearing** determined & 26 \% & — & — \\
\bottomrule\end{tabular}\end{table}
```
</details>


**The same fires split by how well the model describes them. The half it fits well fails to determine the wind exactly as badly as the half it fits poorly, which rules out misspecification as the explanation.**  
<sub>Section 18.3 · `scripts/real_fire_inversion.py`</sub>

| fit quality | fires | wind spread across starts | wind error | bearing error |
|---|---|---|---|---|
| IoU < 0.38 | 11 | 26.6x | 93 % | 29° |
| IoU >= 0.38 | 12 | 23.8x | 94 % | 35° |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{The same fires split by how well the model describes them. The half it fits well fails to determine the wind exactly as badly as the half it fits poorly, which rules out misspecification as the explanation.}
\begin{tabular}{lrrrr}\toprule
fit quality & fires & wind spread across starts & wind error & bearing error \\ \midrule
IoU < 0.38 & 11 & 26.6x & 93 \% & 29° \\
IoU >= 0.38 & 12 & 23.8x & 94 \% & 35° \\
\bottomrule\end{tabular}\end{table}
```
</details>


## Figure manifest


**Every figure cited by RESULTS.md, in order, with the section that cites it.**  
<sub>all · `figures/`</sub>

| file | alt text | section |
|---|---|---|
| `fig_farsite_lb.png` | FARSITE shape law | 2. Does it survive FARSITE's own shape law? |
| `fig_lb_scatter.png` | shape relation uncertainty | 2. Does it survive FARSITE's own shape law? |
| `fig_real_perimeters.png` | real perimeters | 3. Real fires: does the shape predict the wind? |
| `fig_real_lb_spread.png` | real L/B spread | 3. Real fires: does the shape predict the wind? |
| `fig_real_shape_vs_wind.png` | real shape vs wind | 3. Real fires: does the shape predict the wind? |
| `fig_confound_trajectory.png` | confound trajectory | 4. What the fitted parameters actually do |
| `fig_terrain.png` | terrain | 5. Terrain makes it worse, and takes the wind direction too |
| `fig_terrain_aspect.png` | terrain aspect | 5. Terrain makes it worse, and takes the wind direction too |
| `fig_heterogeneous_fuel.png` | heterogeneous fuel | 6. Heterogeneous fuel does not make it better |
| `fig_wind_shift_forecast.png` | wind shift forecast | 7. The consequence: a wind shift turns the degeneracy into a forecast failure |
| `fig_forecast_ensemble.png` | forecast ensemble | 7. The consequence: a wind shift turns the degeneracy into a forecast failure |
| `fig_fuel_prior.png` | fuel prior | 7. The consequence: a wind shift turns the degeneracy into a forecast failure |
| `fig_lead_time_profile.png` | lead time | 8. The consequence: lead time |
| `fig_sensor_sufficiency.png` | sensor sufficiency | 9. The three modalities do different jobs |
| `fig_camera_geometry.png` | camera geometry | 9. The three modalities do different jobs |
| `fig_noise_sensitivity.png` | noise sensitivity | 9. The three modalities do different jobs |
| `fig_recovery.png` | recovery | 9. The three modalities do different jobs |
| `fig_misspecification.png` | misspecification | 10. Does it survive fitting the wrong model? |
| `fig_profile_U.png` | profile likelihood | 11. Methodological finding: the Fisher matrix is the wrong instrument here |
| `fig_profile_theta_w.png` | profile for wind direction | 11. Methodological finding: the Fisher matrix is the wrong instrument here |
| `fig_fusion_level.png` | fusion level | 12. Where you fuse: feature level versus physical-state level |
| `fig_fusion_multistart.png` | multi-start | 12. Where you fuse: feature level versus physical-state level |
| `fig_fusion_scale.png` | fusion at scale | 12. Where you fuse: feature level versus physical-state level |
| `fig_independent_ca.png` | independent CA | 13. The same degeneracy in somebody else's model |
| `fig_wsts_advance.png` | WSTS advance | 14. Real fire progression against real wind |
| `fig_wsts_baseline.png` | WSTS baseline | 15. What are the wind channels worth? A channel ablation on the benchmark's own task |
| `fig_wsts_baseline_full.png` | WSTS baseline, published target | 15. What are the wind channels worth? A channel ablation on the benchmark's own task |
| `fig_retrodiction.png` | retrodiction | 16. Where did it start? The degeneracy's cost for backtracking |
| `fig_guarded_inversion.png` | guarded inversion | 17. An inversion that refuses to report what it cannot determine |
| `fig_sufficiency_profile.png` | sufficiency by profile | 17. An inversion that refuses to report what it cannot determine |
| `fig_real_fire_inversion.png` | real fire inversion | 18. Inverting a real fire |
| `fig_scenario.png` | scenario | 19. The physics kernel is correct |

<details><summary>LaTeX</summary>

```latex
\begin{table}[t]\centering
\caption{Every figure cited by RESULTS.md, in order, with the section that cites it.}
\begin{tabular}{lrr}\toprule
file & alt text & section \\ \midrule
`fig_farsite_lb.png` & FARSITE shape law & 2. Does it survive FARSITE's own shape law? \\
`fig_lb_scatter.png` & shape relation uncertainty & 2. Does it survive FARSITE's own shape law? \\
`fig_real_perimeters.png` & real perimeters & 3. Real fires: does the shape predict the wind? \\
`fig_real_lb_spread.png` & real L/B spread & 3. Real fires: does the shape predict the wind? \\
`fig_real_shape_vs_wind.png` & real shape vs wind & 3. Real fires: does the shape predict the wind? \\
`fig_confound_trajectory.png` & confound trajectory & 4. What the fitted parameters actually do \\
`fig_terrain.png` & terrain & 5. Terrain makes it worse, and takes the wind direction too \\
`fig_terrain_aspect.png` & terrain aspect & 5. Terrain makes it worse, and takes the wind direction too \\
`fig_heterogeneous_fuel.png` & heterogeneous fuel & 6. Heterogeneous fuel does not make it better \\
`fig_wind_shift_forecast.png` & wind shift forecast & 7. The consequence: a wind shift turns the degeneracy into a forecast failure \\
`fig_forecast_ensemble.png` & forecast ensemble & 7. The consequence: a wind shift turns the degeneracy into a forecast failure \\
`fig_fuel_prior.png` & fuel prior & 7. The consequence: a wind shift turns the degeneracy into a forecast failure \\
`fig_lead_time_profile.png` & lead time & 8. The consequence: lead time \\
`fig_sensor_sufficiency.png` & sensor sufficiency & 9. The three modalities do different jobs \\
`fig_camera_geometry.png` & camera geometry & 9. The three modalities do different jobs \\
`fig_noise_sensitivity.png` & noise sensitivity & 9. The three modalities do different jobs \\
`fig_recovery.png` & recovery & 9. The three modalities do different jobs \\
`fig_misspecification.png` & misspecification & 10. Does it survive fitting the wrong model? \\
`fig_profile_U.png` & profile likelihood & 11. Methodological finding: the Fisher matrix is the wrong instrument here \\
`fig_profile_theta_w.png` & profile for wind direction & 11. Methodological finding: the Fisher matrix is the wrong instrument here \\
`fig_fusion_level.png` & fusion level & 12. Where you fuse: feature level versus physical-state level \\
`fig_fusion_multistart.png` & multi-start & 12. Where you fuse: feature level versus physical-state level \\
`fig_fusion_scale.png` & fusion at scale & 12. Where you fuse: feature level versus physical-state level \\
`fig_independent_ca.png` & independent CA & 13. The same degeneracy in somebody else's model \\
`fig_wsts_advance.png` & WSTS advance & 14. Real fire progression against real wind \\
`fig_wsts_baseline.png` & WSTS baseline & 15. What are the wind channels worth? A channel ablation on the benchmark's own task \\
`fig_wsts_baseline_full.png` & WSTS baseline, published target & 15. What are the wind channels worth? A channel ablation on the benchmark's own task \\
`fig_retrodiction.png` & retrodiction & 16. Where did it start? The degeneracy's cost for backtracking \\
`fig_guarded_inversion.png` & guarded inversion & 17. An inversion that refuses to report what it cannot determine \\
`fig_sufficiency_profile.png` & sufficiency by profile & 17. An inversion that refuses to report what it cannot determine \\
`fig_real_fire_inversion.png` & real fire inversion & 18. Inverting a real fire \\
`fig_scenario.png` & scenario & 19. The physics kernel is correct \\
\bottomrule\end{tabular}\end{table}
```
</details>


Not cited in RESULTS.md (2): `fig_fisher_spectra.png`, `fig_lead_time.png`.



## Reproducibility


Every number above is read from `results/*.json` at generation time. `scripts/verify_results.py` independently re-derives the headline figures and fails if `RESULTS.md` disagrees with them, then audits cross-references, figure and script references, and encodings. Run both after changing any experiment.


