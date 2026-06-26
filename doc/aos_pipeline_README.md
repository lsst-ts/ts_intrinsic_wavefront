# AOS Measured Intrinsics and Wavefront Analysis Pipelines

Analysis of the Rubin Active Optics System from Full Array Mode (FAM) data:
per-donut wavefront tables, Double-Zernike (DZ) fits, the *measured intrinsic*
wavefront (telescope + camera split), DOF look-up tables, and operational
studies (bounce tests, correlations). A Snakemake pipeline runs the full chain
per `param_set`; standalone notebooks cover the analyses that have not (yet)
been ported into it, plus AOS closed-loop control studies.

## Pipeline overview

The pipeline (`Snakefile`) processes each `param_set` (a FAM Butler
collection + processing variant, e.g. `fam_danish_1_0_wep17_3_0_bin2x`)
end-to-end. Launch with `./run_snake.sh` on the RSP (see [Running](#running)).

```
Phase 1 (per param_set)                Phase 2 (per param_set × mi_name)
─────────────────────────              ──────────────────────────────────
mktable ──► fit          (per chunk)   build_intrinsic   (per rotator bin)
   │         │                              │
   ▼         ▼                              ├──► study_radialbins (WFS-radius
combine_{donuts,fits,visits}               │         vs rotator, pre-split)
   │                                        ▼
   ├──► plots                          intrinsic_split (OCS + CCS)
   ├──► aberration_pairs                    │
   └──► (feeds Phase 2) ──────────►         ▼
                                       intrinsic_sidecar (per-donut zk)
                                            │
                                            ▼
                                       refit_mi (DZ refit, measured intrinsic)
                                            │
Phase 3 — analyses (per param_set × mi_name)│
────────────────────────────────────────────│
                              ┌─────────────┼──────────────┬──────────┐
                              ▼             ▼              ▼          ▼
                          build_lut   dz_correlations   thermal_   bounce
                                                      correlations
```

| Phase | Step | Granularity | Short description |
|-------|------|-------------|-------------------|
| 1 | `mktable` | per chunk | Butler → per-donut Zernike table + per-visit table |
| 1 | `fit` | per chunk | Double-Zernike fit of (data − batoid intrinsic) per visit |
| 1 | `combine_*` | per param_set | Concatenate chunks → one donuts/fits/visits table each |
| 1 | `plots` | per param_set | Trio validation plots (data / model / residual) on the combined tables |
| 1 | `aberration_pairs` | per param_set | Per-donut primary→secondary aberration-pair correlations |
| 2 | `build_intrinsic` | per (ps, mi, rotator bin) | Measured-intrinsic focal-plane grid (Path-A U-mode constrained) |
| 2 | `intrinsic_split` | per (ps, mi) | Decompose the grids into telescope-fixed (OCS) + camera-fixed (CCS) parts |
| 2 | `study_radialbins` | per (ps, mi) | OCS measured intrinsic in 4 WFS radial shells, overlaid by rotator bin (pre-split) |
| 2 | `intrinsic_sidecar` | per (ps, mi) | Per-donut measured-intrinsic Zernikes, row-aligned to `donuts.parquet` |
| 2 | `wfs_mimic` | per (ps, mi) | FAM-mimicked 4-corner WFS deviation covariance (84×84 + 21×21), image-sampled |
| 2 | `refit_mi` | per (ps, mi) | DZ refit subtracting the *measured* intrinsic instead of batoid |
| 3 | `build_lut` | per (ps, mi) | Averaged-DOF look-up table from the per-visit DZ fits |
| 3 | `dz_correlations` | per (ps, mi) | DZ↔DZ Pearson correlations on the MI-refit residuals |
| 3 | `thermal_correlations` | per (ps, mi) | DZ↔EFD-temperature correlations on the MI-refit residuals |
| 3 | `bounce` | per (ps, mi) | FAM bounce-test paired Δ (DZ / v-mode / DOF) with significance maps |

Planned additions: port `study_compare_donuts.ipynb` to a pipeline script and add
it to the Snakemake DAG. The WFS-mimic *covariance* core is now the `wfs_mimic`
step; the remaining SVD / DOF-recovery / FWHM exploration still lives in
`study_wfs_mimic.ipynb`.

## Pipeline steps in detail

### Phase 1 — donut tables and DZ fits (per `param_set`)

**`mktable`** — `code/run_mktable.py` (library: `intrinsics_lib.py`). Per date
chunk: queries ConsDB for FAM visits, extracts per-donut Zernikes via the
Butler, attaches OCS/CCS field angles and the tabulated batoid intrinsic
(`zk_intrinsic_{OCS,CCS}`), and writes
`output/<ps>/chunks/<dmin>_<dmax>/{donuts,visits}.parquet`. The expensive
Butler step — deliberately *not* re-triggered by code edits (see Snakefile
comments). Requires RSP (Butler + ConsDB).

**`fit`** — `code/run_dz_fit.py` (library: `dz_fitting.py`). Per chunk: robust
(Huber) Double-Zernike fit of the per-donut residual
`zk_data − zk_intrinsic_<coord>` for each visit, producing per-visit DZ
coefficients, errors, and quality flags in `chunks/<d>_<d>/fits.parquet`.
`coord_sys` (OCS default) set per param_set in `snake_config.yaml`.

**`combine_donuts` / `combine_fits` / `combine_visits`** —
`code/combine_parquets.py`. Concatenate the chunk tables into one
param_set-level table each: `output/<ps>/{donuts,fits,visits}.parquet`.
**All downstream steps use the combined tables.** Adding data = adding/editing
a chunk in `snake_config.yaml`; Snakemake re-runs combine + everything
downstream automatically.

**`plots`** — `code/run_dz_plots.py` (library: `dz_plotting.py`). Validation
trio plots (data / DZ model / residual across the focal plane) on the combined
tables → `output/<ps>/plots/trio_comparison_all.pdf`. Memory-heavy (loads the
full donut table); the Snakefile's `mem_mb` throttle serializes it.

**`aberration_pairs`** — `code/run_aberration_pairs.py` (port of
`study_aberrationpairs.ipynb`). Per-donut primary→secondary aberration-pair
analysis (e.g. defocus→spherical, astig→2nd-astig): quartile-of-primary OLS
slope/r plus density pages → `output/<ps>/plots/aberration_pairs.pdf` +
`aberration_pairs_summary.parquet`. Knobs in `analysis_config.yaml`.

### Phase 2 — measured intrinsic (per `param_set` × `mi_name`)

Each `mi_name` entry in `mi_config.yaml` (e.g. `pathA_50_34_i`) defines one
measured-intrinsic build: path, `n_dof`/`n_keep`, band/program/elevation
selection, rotator bins, and build/split parameters. Outputs live under
`output/<ps>/<mi>/`. RSP-only (needs `lsst.ts.ofc`/`wep`,
`$TS_CONFIG_MTTCS_DIR`, batoid height maps).

**`build_intrinsic`** — `code/run_build_intrinsic.py` (libraries:
`measured_intrinsic.py`, `intrinsic_build_plots.py`). Per rotator bin: builds
the empirical focal-plane intrinsic Zernike grid from the FAM donuts via the
Path-A U-mode-constrained method (iterated DZ removal of the reachable
wavefront), with CCD-height Z4 handling →
`output/<ps>/<mi>/build/rot_<lo>_<hi>/intrinsic_grid.parquet` + validation
plots. (Script version of `build_measured_intrinsic.ipynb`.)

**`intrinsic_split`** — `code/run_intrinsic_split.py` (library:
`intrinsic_split.py`). Decomposes the per-rotator-bin grids into a
telescope-fixed component **O** (OCS frame) and a camera-fixed component **C**
(CCS frame, rotating with the rotator) for every Noll Zernike in use (4–19,
22–26). Spin-aware: astig/coma/trefoil doublets are combined as
Z_cos + i·Z_sin and decomposed with the spin model; hole-aware least-squares
keeps O hole-free → `intrinsic_split.parquet`, `intrinsic_split_decomp.npz`,
`intrinsic_split.pdf`. (Script version of
`intrinsic_camera_telescope_split.ipynb`.)

**`study_radialbins`** — `code/run_study_radialbins.py`. Reads the per-rotator-bin
grids *before* the split and, for each pupil Zernike j, makes one page of four
full-width panels = the four WFS radial shells, each overlaying the rotator-angle
samples (median OCS measured intrinsic vs focal-plane azimuth, one colour+marker
per rotator bin) → `study_radialbins.pdf`. Shows how consistent the intrinsic is
across rotator at the radius the wavefront sensors see. The shell inner edge is
the extra-focal `SW0` inner-corner field radius from the `PIXELS→FIELD_ANGLE`
camera transform (1.5178°; derivation documented in the script docstring), the
outer edge the AOS-online 1.725° limit; 4 equal-width bins. Knobs in
`analysis_config.yaml` (`study_radialbins`).

**`intrinsic_sidecar`** — `code/run_make_intrinsic_sidecar.py`. Evaluates the
O + C decomposition at every donut's field position (full spin reconstruction,
C evaluated at the donut's CCS coordinates so the camera term rotates
correctly; plus CCD-height Z4) → `zk_intrinsic.parquet`, row-aligned to the
combined `donuts.parquet`.

**`wfs_mimic`** — `code/run_wfs_mimic.py`. Mimics the four corner wavefront
sensors from FAM donuts: per image, donuts in four annular wedges at the WFS
radius (centred `delta + [0,90,180,270]°`) give four pseudo-WFS Zernike vectors,
and the per-donut deviation (measured − `zk_intrinsic` sidecar, so the OCS/CCS
intrinsic is subtracted at each image's rotator) is median-pooled per wedge.
Covariance over **images** (not donuts) → `wfs_mimic_cov84.parquet` (84×84
cross-corner `Z{j}_c{0..3}`), `wfs_mimic_cov21.parquet` (21×21 = mean of the four
diagonal corner blocks), `wfs_mimic_cov_bins.parquet` (per rotator-subset, tidy),
and `wfs_mimic.pdf`. Image (rotator-angle) selection via the `analysis_config.yaml`
`wfs_mimic.rotator_keep` ranges drops out-of-family rotator points, independent of
`split.rotator_select`. The 84×84 needs ≥84 images for full rank (logged).

**`refit_mi`** — `code/run_dz_fit.py --intrinsic-sidecar`. Re-runs the DZ fit
subtracting the *measured* intrinsic instead of the batoid column →
`output/<ps>/<mi>/fits.parquet`. Why re-fit rather than patch: DZ fitting is
linear, so swapping the intrinsic is cleanest as a recompute. Note for
difference analyses (bounce): any intrinsic fixed in the fitting frame cancels
in a Δ — it is the rotating camera term **C** that changes the rotator-bounce
result, which is why the O + C split matters.

### Phase 3 — analyses on the MI-refit fits (per `param_set` × `mi_name`)

`dz_correlations`, `thermal_correlations`, and `bounce` run on the *residual*
(measured-intrinsic-subtracted) DZ in `output/<ps>/<mi>/fits.parquet`;
`build_lut` currently projects the Phase-1 `fits.parquet`. Knobs in
`analysis_config.yaml`.

**`build_lut`** — `code/run_build_lut.py` (library: `ofc_svd.py`). Averaged-DOF
look-up table: projects the per-visit DZ fits onto the OFC sensitivity-matrix
SVD (settable `n_dof`/`n_keep`), recovers DOF per visit, and collapses over
**all** elevation and rotator angle (median by default) →
`output/<ps>/<mi>/lut/lut.parquet` (per-DOF) + `lut_dz.parquet`
(per-(k, j) raw/fit/residual DZ) + `lut.pdf`. (Supersedes the removed
`study_50dofLUT.ipynb`. Uses the Phase-1 `fits.parquet`.)

**`dz_correlations`** — `code/run_dz_correlations.py` (port of
`study_doublezernike.ipynb` §7–§10). DZ_kj ↔ DZ_k'j' Pearson heatmap, top-|r|
pair scatters, astigmatism-symmetry pairs, per-correlation **conjugate-orbit
scatter grids** (rows/cols = independent focal-k / pupil-j doublet-flips of each
endpoint, up to 4×4), and **significance** (Fisher-z σ, with `n`/`se_r`) in the
pairs parquet; optional exhaustive (k1, j1)×(k2, j2) scan →
`output/<ps>/<mi>/plots/dz_correlations.pdf` + `_pairs.parquet`.

**`dz_correlations_optcorr`** — same analysis on the DZ that *remains after the
n_dof/n_keep OFC correction* (`W_resid = (I − U_eff U_effᵀ)·W`, SVD from the
mi_config entry) → `output/<ps>/<mi>/plots/dz_correlations_optcorr.pdf` +
`_pairs.parquet`. Sits beside the raw analysis for before/after comparison.
RSP-only (builds the OFC SVD via `lsst.ts.ofc`).

**`thermal_correlations`** — `code/run_thermal_correlations.py` (port of
`intrinsics_thermal_correlations.ipynb`). DZ_kj × EFD temperature-variable
Pearson heatmap plus per-term scatter pages →
`output/<ps>/<mi>/plots/thermal_correlations.pdf` + `_summary.parquet`.

**`bounce`** — `code/run_bounce.py` + `bounce_lib.py` (port of
`study_bounce.ipynb`). FAM bounce-test paired Δ (BLOCK-T720 elevation 40↔70°,
BLOCK-T724 rotator 0↔60°): time-ordered within-night comp−ref pairs for DZ
coefficients, OFC v-modes, and physical DOF, with significance/pass heatmaps,
vs-ordinal pages, and night cross-scatter; optional EFD MTAOS Trim overlay
(`add_dof_trim`) → `output/<ps>/<mi>/plots/bounce_*.pdf` +
`bounce_kj_stats.parquet`.

## Configuration

| File | Contents |
|------|----------|
| `param_sets.yaml` | Butler repo / FAM collection definitions per param_set (referenced by name) |
| `snake_config.yaml` | Which param_sets to build, their date chunks, and `coord_sys` |
| `mi_config.yaml` | Measured-intrinsic entries per param_set: path, `n_dof`/`n_keep` (scalar or explicit index list), band/program/elevation selection, rotator bins, build + split parameters. `defaults:` block applies to every entry |
| `analysis_config.yaml` | Analysis-only knobs (`lut`, `aberration_pairs`, `dz_correlations`, `thermal_correlations`, `bounce`), deep-merged: `defaults` ← per-param_set ← per-(param_set, mi_name) overrides. Kept separate from `mi_config.yaml` so editing analysis knobs never re-triggers the slow intrinsic builds |

Rules consume the **resolved per-entry config as a Snakemake `params` value**,
not the config *file* as an input. The `params` rerun-trigger then fires only
when *that* entry's resolved settings change — adding or editing one param_set
(or one analysis section) never invalidates another's cached outputs via the
shared file's mtime. (Editing a shared `defaults:` block still propagates to
every entry, as it should.)

## Running

On the RSP (Butler, ConsDB, `lsst.ts.ofc`/`wep`, `$TS_CONFIG_MTTCS_DIR`
required):

```bash
./run_snake.sh                  # detached run (survives dropped SSH), logs to logs/
./run_snake.sh -n               # dry-run: show what is stale / would run
./run_snake.sh --until combine_donuts
snakemake --dag | dot -Tpng > dag.png
```

`run_snake.sh` uses `-j 4 --resources mem_mb=14000 --keep-going`, tuned for
the RSP terminal allocation (~4 usable cores, 16 GiB); per-rule `mem_mb`
declarations throttle the memory-heavy steps. `git pull` (or `sync.sh`) first
to pick up code changes.

## Output layout

```
output/<param_set>/
  chunks/<dmin>_<dmax>/ {donuts,fits,visits}.parquet     # per chunk
  {donuts,fits,visits}.parquet                           # combined (downstream input)
  plots/                                                 # trio validation, aberration_pairs
  <mi_name>/
    build/rot_<lo>_<hi>/intrinsic_grid.parquet           # per rotator bin
    build/rot_<lo>_<hi>/intrinsic_cov_edge.parquet       # FoV-edge 21x21 (per-donut residual)
    intrinsic_split_{maps,decomp,rms}.parquet  intrinsic_split.pdf
    study_radialbins.pdf                                 # MI at WFS radius vs rotator
    zk_intrinsic.parquet                                 # per-donut sidecar
    wfs_mimic/ wfs_mimic_cov{84,21}.parquet  wfs_mimic_cov_bins.parquet  wfs_mimic.pdf
    fits.parquet                                         # MI-refit DZ fits
    lut/ {lut,lut_dz}.parquet  lut.pdf
    bounce_kj_stats.parquet
    plots/ {dz_correlations,dz_correlations_optcorr,thermal_correlations,bounce_*}.pdf
```

## Notebooks

### AOS control

Closed-loop AOS performance, separate from the FAM wavefront pipeline.
(Candidates to move to their own `aos_control/` topic directory.)

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `nightly_tablemaker.ipynb` | Extract per-exposure AOS data (EFD + ConsDB + Butler) into a single parquet table with vmodes, per-corner Zernikes, and summary arrays. Based on `nightly_report_ts_version`. | 2026-03-07 | 2026-03-13 |
| `aos_nightly_plots.ipynb` | Plots of AOS FWHM and Zernike deviations from multiple nights. Loads nightly_tablemaker output, computes mean 4-corner Zernike deviations, and produces time-series and histogram plots of vmodes and DOF states. | 2026-03-14 | 2026-03-14 |
| `aos_openloop.ipynb` | Reconstruct closed-loop PID behavior of the AOS system. Builds sensitivity matrix SVD, projects 4-corner Zernikes onto vmodes, runs per-vmode PID simulation, and validates against actual DOF corrections. | 2026-03-11 | 2026-03-13 |

### Active analyses (not in the pipeline)

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `smatrix_vmode_info.ipynb` | Comprehensive analysis of the AOS sensitivity matrix: SVD (`StateEstimator` + custom-SVD validation), v-mode composition, wavefront signatures, control equations, noise/gain, plus a normalization-scheme deep-dive (unit-invariance) and double-Zernike field patterns & physical impact. Consolidates the former `normalization_study` and `smatrix_doublez`. Shared primitives in `code/ofc_svd.py`. | 2026-03-08 | 2026-06-11 |
| `intrinsics_mktable.ipynb` | Create a table of Zernike wavefront measurements from FAM cwfs images with model intrinsic values. Queries ConsDB for FAM visits, extracts Zernikes via Butler, interpolates Batoid intrinsic model, saves to parquet. (Interactive counterpart of the pipeline `mktable` step.) | 2026-02-23 | 2026-03-13 |
| `intrinsics_plots.ipynb` | Analyze the FAM Zernike table from `intrinsics_mktable`. Plots data vs model comparisons (trio plots) for each Zernike term across the focal plane. | 2026-02-23 | 2026-02-23 |
| `intrinsics_checkZ4.ipynb` | Check that the Z4 intrinsic map correctly accounts for CCD-to-CCD height variation: subtract the per-visit linear (tilt/tip/piston) Z4, bin in focal plane, compare against independent intrinsic computations and the CCD height map. | 2026-04-20 | 2026-04-21 |
| `intrinsic_Zj.ipynb` | Extend the Z4 check to every Zernike carried by the danish fit (Noll 4–19, 22–26): rotator ≈ 0° donut selection, per-Zj focal-plane median maps of data − model. | 2026-04-22 | 2026-04-22 |
| `study_compare_donuts.ipynb` | Compare per-donut wavefront Zernikes between two processing runs (param_set A vs B — code version, binning, or algorithm). Per-CCD positional donut matching, then coverage maps, per-visit large-\|Δ\|, density (hist or hexbin) over the full focal plane and an edge annulus, difference histograms, focal-plane Δ maps (OCS+CCS), and an optional per-visit double-Zernike-fit comparison. Consolidates the former `study_danish_v0p6_vs_v1`, `study_binning`, and `donutalgo_comparison`. Shared code in `code/compare_donuts.py`. **TODO: port to a pipeline script.** | 2026-06-11 | 2026-06-11 |
| `study_wfs_mimic.ipynb` | Study whether the 4 corner WFS can reconstruct the optical state from FAM observations. Mimics WFS measurements by averaging FAM donuts in annular wedges at the WFS field radius, subtracts measured intrinsic, builds WFS-specific SVD of the sensitivity matrix, recovers DOFs, and compares against full FAM DOF analysis. The mimic + **covariance** core is now the `wfs_mimic` pipeline step (`run_wfs_mimic.py`); the SVD / DOF-recovery / FWHM exploration here is **TODO: rewire onto the pipeline outputs.** | 2026-06-04 | 2026-06-04 |
| `wfs_mimic_covariance.ipynb` | Thin reader for the `wfs_mimic` step's covariance products: loads `wfs_mimic_cov{84,21}.parquet` + `_cov_bins.parquet`, plots the 84×84 / 21×21 covariance & correlation heatmaps, the per-Zernike between-corner correlation, and the per rotator-subset rank/n_images summary. | 2026-06-24 | 2026-06-24 |

### Superseded by pipeline steps (kept as reference)

| Notebook | Pipeline step | Created | Last Modified |
|----------|---------------|---------|---------------|
| `intrinsics_fit.ipynb` | `fit` (`run_dz_fit.py`) — robust DZ focal-plane fitting, z1toz3 + z1toz6 | 2026-02-23 | 2026-03-13 |
| `build_measured_intrinsic.ipynb` | `build_intrinsic` (`run_build_intrinsic.py`) — also documents the Path-B (reachability-thresholded) alternative, FWHM-equivalent diagnostics, and DOF recovery | 2026-04-28 | 2026-05-14 |
| `intrinsic_camera_telescope_split.ipynb` | `intrinsic_split` (`run_intrinsic_split.py`) — OCS/CCS spin decomposition; core in `code/intrinsic_split.py` | 2026-06-10 | 2026-06-10 |
| `study_aberrationpairs.ipynb` | `aberration_pairs` (`run_aberration_pairs.py`) | 2026-05-11 | 2026-05-11 |
| `study_doublezernike.ipynb` | `dz_correlations` (`run_dz_correlations.py`, §7–§10); the LUT-exploration sections evolved into `build_lut` | 2026-04-12 | 2026-04-12 |
| `intrinsics_thermal_correlations.ipynb` | `thermal_correlations` (`run_thermal_correlations.py`) | 2026-04-06 | 2026-04-06 |
| `study_bounce.ipynb` | `bounce` (`run_bounce.py` + `bounce_lib.py`) | 2026-05-06 | 2026-06-09 |

## Data dependencies

- **Pipeline Phase 1** (`mktable`): EFD/ConsDB + Butler — RSP only. `fit`/`combine`/`plots`/`aberration_pairs` run anywhere the parquet outputs exist.
- **Pipeline Phase 2 + analyses**: `lsst.ts.ofc` + `lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, batoid height maps — RSP only. `bounce` with `add_dof_trim` additionally queries EFD/ConsDB live.
- **nightly_tablemaker**: EFD, ConsDB, and Butler access (run on RSP)
- **aos_nightly_plots**: parquet output from nightly_tablemaker
- **aos_openloop**: parquet output from nightly_tablemaker + `ts_config_mttcs` OFC config
- **smatrix_vmode_info**: `lsst.ts.ofc` + `lsst.ts.wep` and `$TS_CONFIG_MTTCS_DIR` OFC config (run on RSP)
- **intrinsics_mktable**: Butler FAM collections + ConsDB (run on RSP)
- **intrinsics_plots / intrinsics_checkZ4 / intrinsic_Zj**: parquet output from intrinsics_mktable / pipeline Phase 1
- **study_compare_donuts**: two runs' `output/<param_set>/{donuts,visits}.parquet` (and `fits.parquet` for the optional DZ-fit comparison); numpy/scipy/pyarrow only (no LSST stack)
- **study_wfs_mimic**: measured-intrinsic build outputs (dz_fits + grid parquets), donut parquets, `lsst.ts.ofc`, and CCD height map (run on RSP)

## Other docs

- `double_zernike_convention_validation.md` — validation of the DZ index/normalization conventions used throughout.
