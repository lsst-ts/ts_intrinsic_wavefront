# Double Zernike Convention Validation — AOS FAM Fit vs GalSim / OFC

**Author:** Aaron Roodman
**Date:** 2026-04-19
**Purpose:** Verify that the Double Zernike (DZ) conventions used by the
Full Array Mode (FAM) fit in this repository are consistent with the
`galsim.zernike.DoubleZernike` class and with how `ts_ofc` uses it.

## Codebase summaries

### AOS FAM (this repo)

- Focal-plane Zernike basis: Noll-orthonormal polynomials evaluated on the
  unit disk (`code/dz_fitting.py:88-105`):
  `Z1 = 1`, `Z2 = 2x`, `Z3 = 2y`, `Z4 = √3 (2r² − 1)`,
  `Z5 = 2√6 xy`, `Z6 = √6 (x² − y²)`.
- Field-angle normalization: `x = thx_deg / fp_radius`,
  `y = thy_deg / fp_radius`, with `fp_radius = 1.75°`
  (`dz_fitting.py:72, 81-82`). Field angles enter in **degrees**.
- Coefficient storage columns: `{prefix}_z{j}_c{k}`
  (`dz_fitting.py:197-199`) with `j` = pupil Noll index, `k` = focal Noll
  index. `prefix` is `z1toz3` or `z1toz6`.
- Pupil Zernike amplitudes: already evaluated per donut in the Butler
  `aggregateAOSVisitTableRaw` table; pupil range is typically Z4–Z23 (19
  terms) or Z4–Z28 (25 terms), inferred from the `nollIndices` metadata or
  from the column width (`dz_fitting.py:24-56`). Values are in **µm**.
- Coordinate systems: OCS default, CCS selected via `--coord-sys CCS`;
  OCS ↔ CCS rotation is applied upstream in `intrinsics_lib.py` from the
  EFD rotator angle.

### GalSim `DoubleZernike`

- Signature
  `DoubleZernike(coef, uv_outer=1.0, uv_inner=0.0, xy_outer=1.0, xy_inner=0.0)`.
- Indexing: `coef[k, j]` — **first axis is field (uv)**, **second axis is
  pupil (xy)**. Both axes use 1-based Noll indexing; the 0-th row and column
  are ignored.
- Orthonormality over the annulus:
  `∫ DZ_{k,j} DZ_{k',j'} = A₁ A₂ δ_{kk'} δ_{jj'}`.
- Composition: `DZ(u, v, x, y) = Σ_{k,j} coef[k,j] Z_k(u, v) Z_j(x, y)`.
- Standard Noll sign convention (Z2 cosine along x, Z3 sine along y).

### OFC usage (`ts_ofc`)

- Constructs a `DoubleZernike` per DOF
  (`python/lsst/ts/ofc/sensitivity_matrix.py`), then rotates by the
  rotator angle:
  ```python
  galsim.zernike.DoubleZernike(
      self.ofc_data.sensitivity_matrix[..., dof_idx],
      uv_inner=self.ofc_data.config["field"]["radius_inner"],
      uv_outer=self.ofc_data.config["field"]["radius_outer"],
      xy_inner=self.ofc_data.config["pupil"]["radius_inner"],
      xy_outer=self.ofc_data.config["pupil"]["radius_outer"],
  ).rotate(theta_uv=rotation_angle)
  ```
- YAML config (`policy/configurations/lsst.yaml`):
  - `field.radius_outer = 1.75 deg`, `field.radius_inner = 0.0 deg`
  - `pupil.radius_outer = 4.18 m`, `pupil.radius_inner = 2.558 m`
- Default Noll pupil range: `znmin = 4`, `znmax = 28`.
- `evaluate(field_angles)` expects field angles and rotation in **degrees**.
- After evaluation, `einsum("ijk -> jki")` brings the sensitivity result to
  shape `(n_zernikes, n_field_points, n_dofs)` and is sliced on `znmin:znmax+1`.

## Validation table

| Convention | AOS FAM (this repo) | galsim.DoubleZernike | OFC (ts_ofc) | Match? |
|---|---|---|---|---|
| **Zernike indexing** | Noll (Z1 = piston, Z4 = defocus, Z5 / Z6 = astig, …) | Noll, 1-based; `coef[0,:]` and `coef[:,0]` unused | Noll; pupil range `znmin..znmax` ⇒ Z4..Z28 by default | ✅ |
| **Normalization (single Z)** | Noll-orthonormal on unit disk | Noll-orthonormal: `∫ Zⱼ Zⱼ' = π δⱼⱼ'` | Inherits galsim | ✅ |
| **2-D coef array layout** | Flat columns `{prefix}_z{j}_c{k}`, `j` = pupil Noll, `k` = focal Noll (`dz_fitting.py:197-199`) | `coef[k, j]`: **first axis = field (uv)**, **second axis = pupil (xy)** | `sensitivity_matrix[..., dof_idx]` passed straight to galsim | ⚠️ *inverted*: we index `[j, k]` in the column name, galsim expects `[k, j]` — build a `DZ.coef` array by transposing |
| **Field (uv) outer radius** | `fp_radius = 1.75°` (`dz_fitting.py:72, 82`) | `uv_outer` (caller supplies) | `field.radius_outer = 1.75 deg` in `lsst.yaml` | ✅ |
| **Field (uv) inner radius** | Not used (full disk) | `uv_inner = 0` default | `field.radius_inner = 0.0 deg` | ✅ |
| **Field angle input units** | **Degrees**; normalized `x = thx_deg / fp_radius` | Same units as `uv_outer` | `evaluate(field_angles)` in **degrees** | ✅ |
| **Pupil (xy) coordinates** | Pupil zk are already evaluated per donut by the AOS pipeline upstream — we never re-expand on the pupil in the fit | `xy_outer`, `xy_inner` in meters | `pupil.radius_outer = 4.18 m`, `pupil.radius_inner = 2.558 m` (LSST M1) | ✅ upstream; we consume pre-computed per-donut Zernike amplitudes |
| **Axis conventions (Z2 / Z3)** | `Z2 = 2·thx/fp_r` (tilt along x), `Z3 = 2·thy/fp_r` (tilt along y) | Standard Noll: Z2 cosine along x, Z3 sine along y | Inherits galsim | ✅ |
| **Coefficient units** | µm (from `aggregateAOSVisitTableRaw`; docstring `dz_fitting.py:140-142`) | Units of expansion equal units of `coef` | µm wavefront per DOF-unit | ✅ |
| **Coordinate frame** | OCS default; CCS via `--coord-sys CCS`; OCS ↔ CCS rotation applied upstream from EFD rotator angle | Agnostic — caller provides field angles | OCS; `.rotate(theta_uv=rotation_angle)` applied for camera rotator | ✅ |
| **Pupil Noll range** | Typically Z4–Z23 (19) or Z4–Z28 (25); derived from metadata or data width (`dz_fitting.py:24-56`) | No range — caller chooses | `znmin = 4`, `znmax = 28` default | ✅ |

## One thing to watch — axis order when interoperating with galsim

Our coefficient storage is `z{j_pupil}_c{k_focal}`, but galsim's `coef`
array is indexed `coef[k_focal, j_pupil]` (first axis = field-dependent
uv, second axis = pupil-dependent xy). When passing our fit into a
`galsim.zernike.DoubleZernike`, transpose:

```python
import numpy as np
import galsim.zernike as gsz

max_k = 6        # focal Noll max
max_j = 28       # pupil Noll max
coef = np.zeros((max_k + 1, max_j + 1))
for j in pupil_noll_list:           # e.g. 4..28
    for k in range(1, max_k + 1):   # 1..6
        col = f'{prefix}_z{j}_c{k}'
        if col in fit_row:
            coef[k, j] = fit_row[col]

dz = gsz.DoubleZernike(
    coef,
    uv_outer=1.75, uv_inner=0.0,   # field, degrees
    xy_outer=4.18, xy_inner=2.558, # pupil, meters
)
```

## Conclusion

The Double Zernike convention in the AOS FAM fit is consistent with
`galsim.zernike.DoubleZernike` and with how `ts_ofc` uses it. All
normalization, indexing, unit, and frame conventions match; the
`fp_radius = 1.75°` in `dz_fitting.py` is the same value as the OFC
configuration's `field.radius_outer`. (The `FP_RADIUS` constant used by
FAM elsewhere in the pipeline serves a different purpose and is not part
of this DZ-fit basis.) The only housekeeping item is the axis order when
converting our per-column storage to a 2-D `coef` array for galsim — our
columns are `[j_pupil][k_focal]`, galsim expects `[k_focal][j_pupil]`.
