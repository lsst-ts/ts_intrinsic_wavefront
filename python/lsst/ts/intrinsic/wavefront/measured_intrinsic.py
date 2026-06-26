"""Build a *measured* intrinsic wavefront for the Rubin AOS FAM pipeline.

The "measured intrinsic" is an empirical estimate of the intrinsic
focal-plane Zernike map, derived from FAM donut data after fitting and
subtracting a chosen subset of Double-Zernike modes (k focal × j pupil).

Workflow (matches the build_measured_intrinsic.ipynb notebook):

  1. Load donut + visit info parquet pair, apply day_obs / alt /
     rotator_angle / seq_num filters.
  2. For each visit, fit a *specified* subset of (k, j) DZ modes to
     `data - tabulated_intrinsic` per pupil Zernike.
  3. Subtract only the fitted DZ contribution from the data — this leaves
     the intrinsic content plus residuals as the *measured wavefront*.
  4. Take the median measured wavefront vs (thx, thy) per pupil j over a
     binned focal-plane grid.  This is the iter-1 measured intrinsic.
  5. Iterate: replace the tabulated intrinsic with the iter-1 estimate
     (linearly interpolated at each donut's field position) and repeat
     fit + subtraction + median to get iter-2.

The DZ removal spec is sparse in (k, j): different focal-k orders may
have different pupil-j ranges.  The default used in the notebook is

    k=1: j = 4..19, 22..26          (omit Z20, Z21 pentafoil)
    k=2,3: j = 4..17
    k=4..6: j = 4..10

Functions
---------
expand_removal_spec
    Normalize a sparse spec into list[(k, j)] tuples and helper maps.
fit_dz_modes_per_visit
    Per-visit RLM fit of the chosen modes (in-memory).
fit_dz_modes_streaming
    Streaming variant that reads donut parquet one row group per visit.
evaluate_dz_subtraction
    Evaluate the fitted DZ contribution at each donut's field position.
bin_median_focal
    Median focal-plane map per pupil j on a binned grid.
interpolate_grid_at_donuts
    Sample a per-pupil-j focal grid at arbitrary (thx, thy) positions.
build_measured_intrinsic
    High-level driver: runs the iteration `n_iter` times and returns the
    intermediate maps + DZ fit tables for plotting / output.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from astropy.table import QTable
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic_2d

# Progress bar — pick the notebook variant when running inside Jupyter,
# fall back to plain tqdm otherwise.  We never want a hard tqdm dependency
# to break the import.
# Use plain-text tqdm rather than tqdm.auto.  The .auto variant chooses
# tqdm.notebook (ipywidgets) inside a Jupyter kernel, which renders as
# "Error displaying widget: model not found" if the widget extension
# isn't fully initialized (common on the RSP).  Plain tqdm always prints
# a stable text bar.
try:
    from tqdm import tqdm as _tqdm
except Exception:                                 # pragma: no cover
    def _tqdm(it, *args, **kwargs):                # noqa: ARG001
        return it

from lsst.ts.intrinsic.wavefront.dz_fitting import focal_plane_zernike_basis, derive_noll_indices


# ----------------------------------------------------------------------
# OFC sensitivity-matrix utilities (used by the U-mode-constrained and
# reachability-thresholded paths in build_measured_intrinsic.ipynb).
# ----------------------------------------------------------------------

def removal_spec_from_reachability(frac_2d, k_range, j_range, threshold):
    """Build a ``{k: [j, j, ...]}`` removal spec from a reachability grid.

    Parameters
    ----------
    frac_2d : ndarray, shape (n_k, n_j)
        Per-(k, j) reachability fraction in [0, 1], same orientation as
        the heatmaps produced by the §6 SVD / reachability cells.
    k_range, j_range : iterable of int
        The k and j values labeling the rows / cols of `frac_2d`.
    threshold : float
        Lower bound on `frac_2d[i, j]` for a cell to enter the spec.

    Returns
    -------
    dict {k: [j, ...]} containing only the (k, j) cells that pass.
    """
    k_list = list(k_range)
    j_list = list(j_range)
    spec = {}
    for ki, k in enumerate(k_list):
        js = [int(j) for ji, j in enumerate(j_list)
              if float(frac_2d[ki, ji]) >= float(threshold)]
        if js:
            spec[int(k)] = js
    return spec


# ----------------------------------------------------------------------
# Removal-spec utilities
# ----------------------------------------------------------------------

def expand_removal_spec(spec):
    """Expand a (k -> iterable of j) mapping into a flat list of (k, j) tuples.

    `spec` can be either:
      * a dict like {1: [4, 5, ..., 26], 2: range(4, 18), ...}
      * a flat iterable of (k, j) tuples already in that form

    Returns
    -------
    pairs : list of (k, j) sorted by (k, j)
    by_pupil : dict j -> sorted list of k
    by_focal : dict k -> sorted list of j
    """
    if isinstance(spec, dict):
        pairs = sorted({(int(k), int(j)) for k, jj in spec.items() for j in jj})
    else:
        pairs = sorted({(int(k), int(j)) for (k, j) in spec})
    by_pupil = defaultdict(list)
    by_focal = defaultdict(list)
    for k, j in pairs:
        by_pupil[j].append(k)
        by_focal[k].append(j)
    by_pupil = {j: sorted(ks) for j, ks in by_pupil.items()}
    by_focal = {k: sorted(js) for k, js in by_focal.items()}
    return pairs, by_pupil, by_focal


def default_removal_spec():
    """Default removal spec used by the notebook.

    k=1 covers all pupil Z indices in 4..26 except Z20 and Z21
    (pentafoil_x / pentafoil_y).  k=2, 3 cover 4..17.
    k=4, 5, 6 cover 4..10.
    """
    return {
        1: [j for j in range(4, 27) if j not in (20, 21)],
        2: list(range(4, 18)),
        3: list(range(4, 18)),
        4: list(range(4, 11)),
        5: list(range(4, 11)),
        6: list(range(4, 11)),
    }


# ----------------------------------------------------------------------
# Per-visit DZ fit (sparse k, j spec)
# ----------------------------------------------------------------------

def _fit_one_image_subset(thx_deg, thy_deg, zk_data, zk_intrinsic, iZidx,
                          by_pupil, fp_radius, dobs, snum, img_idx):
    """Fit the requested (k, j) subset for one visit.

    For each pupil j in `by_pupil`, build the focal-plane Z basis using
    only the requested k indices and fit `data - intrinsic` to it via
    Huber RLM (lstsq fallback).  Returns a row dict with coefficients
    keyed `dz_z{j}_c{k}` (and `_err`) plus a `dz_z{j}_scale`, and the
    per-donut DZ contribution evaluated at (thx, thy) for every j.
    """
    n_donuts = len(thx_deg)
    img_params = {'day_obs': int(dobs), 'seq_num': int(snum),
                  'image_idx': int(img_idx), 'n_donuts': int(n_donuts)}

    # We need a max-Noll basis covering the largest k anywhere; we will
    # then slice the columns we want per pupil j.
    max_k = max(max(ks) for ks in by_pupil.values())
    A_full, _ = focal_plane_zernike_basis(thx_deg, thy_deg, max_k, fp_radius)

    # Output: per-donut DZ contribution for each pupil j we touched.
    # Indexed in iZidx layout so callers can subtract directly from zk.
    n_zern = len(iZidx)
    dz_contrib = np.zeros((n_donuts, n_zern))

    for j, ks in by_pupil.items():
        if j not in iZidx:
            continue
        col_j = iZidx[j]
        resid = zk_data[:, col_j] - zk_intrinsic[:, col_j]
        if not np.any(np.isfinite(resid)):
            for k in ks:
                img_params[f'dz_z{j}_c{k}'] = np.nan
                img_params[f'dz_z{j}_c{k}_err'] = np.nan
            img_params[f'dz_z{j}_scale'] = np.nan
            continue

        cols = [k - 1 for k in ks]   # k=1 -> column 0, k=2 -> column 1, ...
        A_j = A_full[:, cols]

        try:
            rlm = sm.RLM(resid, A_j, M=sm.robust.norms.HuberT()).fit()
            coeffs = rlm.params
            bse = rlm.bse
            scale = float(rlm.scale)
        except Exception:
            coeffs, _, _, _ = np.linalg.lstsq(A_j, resid, rcond=None)
            bse = np.full(len(ks), np.nan)
            scale = float(np.std(resid - A_j @ coeffs))

        for ki, k in enumerate(ks):
            img_params[f'dz_z{j}_c{k}'] = float(coeffs[ki])
            img_params[f'dz_z{j}_c{k}_err'] = float(bse[ki])
        img_params[f'dz_z{j}_scale'] = scale

        # Per-donut DZ contribution we will subtract from zk[:, col_j]
        dz_contrib[:, col_j] = A_j @ coeffs

    return img_params, dz_contrib


def fit_dz_modes_per_visit(donut_df, visit_table, coord_sys, iZs,
                           removal_spec, fp_radius=1.75):
    """In-memory fit + per-donut DZ contribution.

    Parameters
    ----------
    donut_df : pandas.DataFrame
        Per-donut data with columns:
          - day_obs, seq_num
          - thx_{coord_sys}, thy_{coord_sys}  (radians)
          - zk_{coord_sys}, zk_intrinsic_{coord_sys}  (each is a list/array column)
    visit_table : QTable or DataFrame
        Visit info (day_obs, seq_num) — used to keep visit ordering stable.
    coord_sys : str
    iZs : list of int
        Pupil Noll indices present in the zk arrays (in order).
    removal_spec : dict or list
        Passed through expand_removal_spec.
    fp_radius : float

    Returns
    -------
    fit_rows : list of dict   (one per visit)
    dz_contrib : ndarray (n_donuts, n_zernikes) — DZ-only contribution
        in the same row order as donut_df, in iZidx layout.
    """
    pairs, by_pupil, by_focal = expand_removal_spec(removal_spec)
    iZidx = {iZ: i for i, iZ in enumerate(iZs)}

    thx = np.rad2deg(np.asarray(donut_df[f'thx_{coord_sys}'], dtype=float))
    thy = np.rad2deg(np.asarray(donut_df[f'thy_{coord_sys}'], dtype=float))
    zk_data = np.stack(donut_df[f'zk_{coord_sys}'].values)
    zk_intrinsic = np.stack(donut_df[f'zk_intrinsic_{coord_sys}'].values)
    dobs_arr = np.asarray(donut_df['day_obs'])
    snum_arr = np.asarray(donut_df['seq_num'])

    fit_rows = []
    dz_contrib = np.zeros_like(zk_data)
    images = sorted(set(zip(dobs_arr.tolist(), snum_arr.tolist())))

    for img_idx, (dobs, snum) in enumerate(images):
        mask = (dobs_arr == dobs) & (snum_arr == snum)
        params, contrib = _fit_one_image_subset(
            thx[mask], thy[mask], zk_data[mask], zk_intrinsic[mask],
            iZidx, by_pupil, fp_radius, dobs, snum, img_idx)
        fit_rows.append(params)
        dz_contrib[mask] = contrib

    print(f"DZ subset fit: {len(images)} images, "
          f"{len(pairs)} (k,j) modes total")
    return fit_rows, dz_contrib


def fit_dz_modes_streaming(donut_parquet, visit_table, coord_sys, iZs,
                           removal_spec, fp_radius=1.75,
                           intrinsic_override=None):
    """Streaming variant: read donut parquet one row group per visit.

    `intrinsic_override` may be a callable
        (thx_deg, thy_deg, j_index_in_iZs) -> ndarray
    used to substitute a per-donut intrinsic estimate (e.g. an iterated
    measured-intrinsic grid).  When None, the column
    `zk_intrinsic_{coord_sys}` is used.

    Returns
    -------
    fit_rows : list of dict
    dz_contrib_blocks : list of (mask_index_array, contrib_ndarray)
        One block per visit so callers can stitch back into a global
        order if they kept the donut DataFrame in memory; if not, this
        module will not assemble a global array (caller's choice).
    """
    pairs, by_pupil, by_focal = expand_removal_spec(removal_spec)
    iZidx = {iZ: i for i, iZ in enumerate(iZs)}

    pf = pq.ParquetFile(str(donut_parquet))

    # row-group lookup
    rg_index = {}
    for i in range(pf.num_row_groups):
        meta = pf.metadata.row_group(i)
        d = s = None
        for ci in range(meta.num_columns):
            cmeta = meta.column(ci)
            name = cmeta.path_in_schema
            if name == 'day_obs' and cmeta.statistics is not None:
                d = cmeta.statistics.min
            elif name == 'seq_num' and cmeta.statistics is not None:
                s = cmeta.statistics.min
        if d is not None and s is not None:
            rg_index[(int(d), int(s))] = i

    fit_rows = []
    dz_contrib_blocks = []
    thx_col = f'thx_{coord_sys}'
    thy_col = f'thy_{coord_sys}'
    zk_col = f'zk_{coord_sys}'
    zk_intr_col = f'zk_intrinsic_{coord_sys}'

    for img_idx, v in enumerate(visit_table):
        dobs = int(v['day_obs'])
        snum = int(v['seq_num'])
        rg_i = rg_index.get((dobs, snum))
        if rg_i is None:
            continue
        df = pf.read_row_group(rg_i).to_pandas()
        if len(df) == 0:
            continue

        thx = np.rad2deg(df[thx_col].to_numpy(dtype=float))
        thy = np.rad2deg(df[thy_col].to_numpy(dtype=float))
        zk_data = np.stack(df[zk_col].values)
        if intrinsic_override is not None:
            zk_intr = intrinsic_override(thx, thy, iZs)
        else:
            zk_intr = np.stack(df[zk_intr_col].values)

        params, contrib = _fit_one_image_subset(
            thx, thy, zk_data, zk_intr, iZidx, by_pupil,
            fp_radius, dobs, snum, img_idx)
        fit_rows.append(params)
        dz_contrib_blocks.append((dobs, snum, contrib))

    print(f"DZ subset fit (streamed): {len(fit_rows)} visits, "
          f"{len(pairs)} (k,j) modes total")
    return fit_rows, dz_contrib_blocks


# ----------------------------------------------------------------------
# Focal-plane median maps
# ----------------------------------------------------------------------

def make_focal_grid(n_bins, fp_radius=1.8):
    """Return (xbins, ybins, xcent, ycent) for the focal-plane bin grid."""
    edges = np.linspace(-fp_radius, fp_radius, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, edges, centers, centers


def bin_median_focal(thx_deg, thy_deg, values_2d, iZidx, n_bins=73,
                    fp_radius=1.8, statistic='median'):
    """Bin per-donut values onto a (n_bins x n_bins) focal-plane grid.

    Parameters
    ----------
    thx_deg, thy_deg : ndarray (n_donuts,)
    values_2d : ndarray (n_donuts, n_zernikes)
        e.g. the data minus DZ contribution.
    iZidx : dict pupil_noll -> column index in values_2d
    n_bins : int
    fp_radius : float

    Returns
    -------
    grid : dict pupil_noll -> ndarray (n_bins, n_bins) with binned stat
        Indexing: grid[j][ix, iy], where x = thy_deg, y = thx_deg
        (matches the existing trio plot orientation).
    xbins, ybins, xcent, ycent : grid edges/centers (deg)
    """
    xbins, ybins, xcent, ycent = make_focal_grid(n_bins, fp_radius)
    grid = {}
    # Guard against an empty donut set (e.g. a rotator/elevation window
    # that catches no visits): binned_statistic_2d would raise on a
    # zero-size array.  Return all-NaN grids instead.
    if len(thx_deg) == 0:
        empty = np.full((n_bins, n_bins), np.nan)
        for j in iZidx:
            grid[j] = empty.copy()
        return grid, xbins, ybins, xcent, ycent
    for j, col in iZidx.items():
        z = values_2d[:, col]
        # Same orientation as plot_zernike_trio: x = thy, y = thx
        stat_val, _, _, _ = binned_statistic_2d(
            thy_deg, thx_deg, z, statistic=statistic,
            bins=[xbins, ybins])
        grid[j] = stat_val
    return grid, xbins, ybins, xcent, ycent


def bin_count_rms_focal(thx_deg, thy_deg, values_2d, iZidx, n_bins=73,
                        fp_radius=1.8):
    """Per-cell donut count and per-Zernike RMS on the focal grid.

    Same binning / orientation as :func:`bin_median_focal` (x = thy, y = thx).
    Lets the caller form an error on the per-cell (median) value:
    SE_median ≈ 1.2533 · rms / sqrt(count).

    Returns
    -------
    count : ndarray (n_bins, n_bins)
        Donut count per cell (Zernike-independent).
    rms : dict pupil_noll -> ndarray (n_bins, n_bins)
        Std (RMS about the mean) of the donut values per cell, per Zernike.
    xbins, ybins, xcent, ycent : grid edges/centres (deg).
    """
    xbins, ybins, xcent, ycent = make_focal_grid(n_bins, fp_radius)
    if len(thx_deg) == 0:
        empty = np.full((n_bins, n_bins), np.nan)
        return (empty.copy(), {j: empty.copy() for j in iZidx},
                xbins, ybins, xcent, ycent)
    count, _, _, _ = binned_statistic_2d(
        thy_deg, thx_deg, np.ones(len(thx_deg)), statistic='count', bins=[xbins, ybins])
    rms = {}
    for j, col in iZidx.items():
        s, _, _, _ = binned_statistic_2d(
            thy_deg, thx_deg, values_2d[:, col], statistic='std',
            bins=[xbins, ybins])
        rms[j] = s
    return count, rms, xbins, ybins, xcent, ycent


def interpolate_grid_at_donuts(grid, xcent, ycent, thx_deg, thy_deg, iZs,
                               fallback=None):
    """Interpolate a per-pupil-j focal grid at arbitrary donut positions.

    Returns ndarray (n_donuts, len(iZs)) suitable to use as
    intrinsic_override.

    `fallback` may be an array (n_donuts, n_zern) used wherever the
    interpolated grid is NaN (e.g. empty bins outside the data envelope).
    """
    n_donuts = len(thx_deg)
    out = np.full((n_donuts, len(iZs)), np.nan)
    # x coord = thy_deg, y coord = thx_deg (matches binning)
    pts = np.column_stack([thy_deg, thx_deg])
    for col, iZ in enumerate(iZs):
        if iZ not in grid:
            if fallback is not None:
                out[:, col] = fallback[:, col]
            continue
        z = grid[iZ]
        interp = RegularGridInterpolator(
            (xcent, ycent), z, method='linear',
            bounds_error=False, fill_value=np.nan)
        vals = interp(pts)
        if fallback is not None:
            nan_mask = np.isnan(vals)
            vals[nan_mask] = fallback[nan_mask, col]
        out[:, col] = vals
    return out


# ----------------------------------------------------------------------
# High-level driver: 2-iteration measured-intrinsic build
# ----------------------------------------------------------------------

def _flag_bad_visits(fit_rows, by_pupil, threshold, min_donuts):
    """Set fit_rows[i]['bad_fit'] = True for visits failing quality cuts.

    Bad if either:
      * n_donuts < min_donuts, OR
      * |dz_z{j}_c{k}| > threshold (μm) for any (k, j) in the spec
        (NaN coefficients also count as bad).

    Returns the list of (day_obs, seq_num) pairs flagged bad.
    """
    bad = []
    for r in fit_rows:
        bad_flag = False
        if int(r.get('n_donuts', 0)) < min_donuts:
            bad_flag = True
        else:
            for j, ks in by_pupil.items():
                for k in ks:
                    val = r.get(f'dz_z{j}_c{k}', np.nan)
                    if not np.isfinite(val) or abs(float(val)) > threshold:
                        bad_flag = True
                        break
                if bad_flag:
                    break
        r['bad_fit'] = bool(bad_flag)
        if bad_flag:
            bad.append((int(r['day_obs']), int(r['seq_num'])))
    return bad


def _grid_l2_change(g_new, g_old):
    """RMS change between two {j: 2D grid} measured maps over jointly-finite
    cells (μm).  NaN if no overlap or no previous grid."""
    if g_old is None:
        return np.nan
    num = 0.0; den = 0
    for j, a in g_new.items():
        b = g_old.get(j)
        if b is None:
            continue
        m = np.isfinite(a) & np.isfinite(b)
        num += float(np.sum((a[m] - b[m]) ** 2)); den += int(m.sum())
    return float(np.sqrt(num / den)) if den else np.nan


def _report_convergence(iter_results, converge_tol):
    """Log the RMS change between successive measured grids; warn if the final
    step exceeds converge_tol (MI build may need more iterations)."""
    deltas = [_grid_l2_change(iter_results[i]['measured_grid'],
                              iter_results[i - 1]['measured_grid'])
              for i in range(1, len(iter_results))]
    if not deltas:
        return
    print('  MI convergence (RMS Δ between iters, μm): '
          + ', '.join(f'{d:.2e}' for d in deltas))
    if np.isfinite(deltas[-1]) and deltas[-1] > converge_tol:
        print(f'  WARNING: last-iter Δ {deltas[-1]:.2e} > tol {converge_tol:.1e} μm'
              f' — measured intrinsic may not be converged (raise n_iter)')


def build_measured_intrinsic(donut_df, visit_table, coord_sys, iZs,
                             removal_spec, n_iter=2,
                             n_bins=73,
                             fp_radius_basis=1.75,
                             fp_radius_grid=1.8,
                             min_donuts=500,
                             bad_fit_threshold=2.0,
                             data_offset=None,
                             intrinsic_offset=None,
                             converge_tol=1e-3):
    """Iterate fit -> subtract -> median to build the measured intrinsic.

    Per-visit quality cuts (matching run_pipeline's fit step):
      * `min_donuts` — require at least this many donuts per visit
      * `bad_fit_threshold` — flag a visit if any |coeff| > this (μm)
    Donuts from bad-fit visits are excluded from the median grid.

    Per-pupil offsets (subtracted from zk arrays before any binning or
    fitting):
      * `data_offset[j]`      — per-donut shift removed from `zk_data[:, j]`
      * `intrinsic_offset[j]` — per-donut shift removed from
                                 `zk_intrinsic_tab[:, j]`
    Used e.g. for the Z4 CCD-height correction:
      data_offset       = {4: Z4hgt}             (height at donut)
      intrinsic_offset  = {4: Z4hgt_transpose}   (height at per-CCD
                                                  x<->y transpose, to
                                                  match the pipeline's
                                                  intrinsic-Zernike
                                                  transpose bug).

    Returns
    -------
    out : dict with keys
        'iZs', 'iZidx'
        'original_median'   : grid dict (pupil j -> 2D array) of median raw zk
                              (good visits only)
        'tabulated_median'  : grid dict of median tabulated intrinsic
                              (good visits only)
        'iter_results'      : list of length n_iter; each element is a dict
                              {'fit_rows', 'dz_contrib', 'measured_grid',
                               'bad_visits'}
        'xbins','ybins','xcent','ycent'
    """
    iZidx = {iZ: i for i, iZ in enumerate(iZs)}
    thx = np.rad2deg(np.asarray(donut_df[f'thx_{coord_sys}'], dtype=float))
    thy = np.rad2deg(np.asarray(donut_df[f'thy_{coord_sys}'], dtype=float))
    zk_data = np.stack(donut_df[f'zk_{coord_sys}'].values).astype(float).copy()
    zk_intrinsic_tab = np.stack(
        donut_df[f'zk_intrinsic_{coord_sys}'].values).astype(float).copy()
    dobs_arr = np.asarray(donut_df['day_obs'])
    snum_arr = np.asarray(donut_df['seq_num'])

    # Apply per-pupil offsets (e.g. Z4 CCD-height correction)
    if data_offset:
        for j, off in data_offset.items():
            if j in iZidx:
                zk_data[:, iZidx[j]] -= np.asarray(off, dtype=float)
                print(f"  data_offset applied to pupil j={j}: "
                      f"shift mean={float(np.nanmean(off)):.4f} μm, "
                      f"std={float(np.nanstd(off)):.4f} μm")
    if intrinsic_offset:
        for j, off in intrinsic_offset.items():
            if j in iZidx:
                zk_intrinsic_tab[:, iZidx[j]] -= np.asarray(off, dtype=float)
                print(f"  intrinsic_offset applied to pupil j={j}: "
                      f"shift mean={float(np.nanmean(off)):.4f} μm, "
                      f"std={float(np.nanstd(off)):.4f} μm")

    pairs, by_pupil, _ = expand_removal_spec(removal_spec)
    images = sorted(set(zip(dobs_arr.tolist(), snum_arr.tolist())))

    iter_results = []
    intrinsic_per_donut = zk_intrinsic_tab.copy()
    # Reference maps (built using only the visits surviving iter-1 cuts —
    # set after the first pass below).
    original_median = None
    tabulated_median = None
    xbins = ybins = xcent = ycent = None

    for it in range(n_iter):
        # 1. Fit DZ subset using current intrinsic (every visit, every donut)
        fit_rows = []
        dz_contrib = np.zeros_like(zk_data)
        bar = _tqdm(enumerate(images), total=len(images),
                    desc=f'iter {it + 1}/{n_iter} fits',
                    leave=True)
        for img_idx, (dobs, snum) in bar:
            mask = (dobs_arr == dobs) & (snum_arr == snum)
            params, contrib = _fit_one_image_subset(
                thx[mask], thy[mask], zk_data[mask],
                intrinsic_per_donut[mask],
                iZidx, by_pupil, fp_radius_basis, dobs, snum, img_idx)
            fit_rows.append(params)
            dz_contrib[mask] = contrib

        # 2. Flag bad visits using *this* iteration's fits
        bad_visits = _flag_bad_visits(fit_rows, by_pupil,
                                      bad_fit_threshold, min_donuts)
        bad_set = set(bad_visits)

        # 3. Per-donut good-mask: drop donuts from any bad-flagged visit
        good_donut_mask = np.array([
            (int(d), int(s)) not in bad_set
            for d, s in zip(dobs_arr, snum_arr)
        ])

        n_visits_good = len(images) - len(bad_visits)
        n_donuts_good = int(good_donut_mask.sum())

        # 4. Subtract DZ contribution from data (but NOT the intrinsic)
        wfd_subtracted = zk_data - dz_contrib

        # 5. Median per pupil j on the focal grid — GOOD visits only
        measured_grid, xbins, ybins, xcent, ycent = bin_median_focal(
            thx[good_donut_mask], thy[good_donut_mask],
            wfd_subtracted[good_donut_mask],
            iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)

        # On the first iteration, also build the reference (Original /
        # Tabulated) maps using the same good-visit subset.
        if it == 0:
            original_median, *_ = bin_median_focal(
                thx[good_donut_mask], thy[good_donut_mask],
                zk_data[good_donut_mask],
                iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)
            tabulated_median, *_ = bin_median_focal(
                thx[good_donut_mask], thy[good_donut_mask],
                zk_intrinsic_tab[good_donut_mask],
                iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)

        iter_results.append({
            'fit_rows': fit_rows,
            'dz_contrib': dz_contrib,
            'wfd_subtracted': wfd_subtracted,
            'measured_grid': measured_grid,
            'bad_visits': bad_visits,
            'good_donut_mask': good_donut_mask,
        })

        # 6. Prepare next iteration's intrinsic from this measured map
        intrinsic_per_donut = interpolate_grid_at_donuts(
            measured_grid, xcent, ycent, thx, thy, iZs,
            fallback=zk_intrinsic_tab)

        n_modes = sum(len(ks) for ks in by_pupil.values())
        print(f"  iter {it + 1}/{n_iter}: fit {n_modes} DZ modes for "
              f"{len(images)} visits ({n_visits_good} good, "
              f"{len(bad_visits)} flagged bad); "
              f"median over {n_donuts_good} donuts "
              f"on {n_bins}x{n_bins} grid")

    _report_convergence(iter_results, converge_tol)
    return {
        'iZs': iZs,
        'iZidx': iZidx,
        'original_median': original_median,
        'tabulated_median': tabulated_median,
        'iter_results': iter_results,
        'xbins': xbins, 'ybins': ybins,
        'xcent': xcent, 'ycent': ycent,
    }


# ----------------------------------------------------------------------
# U-mode constrained variant
# ----------------------------------------------------------------------

def _dz_contrib_from_params(params, by_pupil, thx_deg, thy_deg,
                            iZidx, fp_radius):
    """Rebuild the per-donut DZ contribution from a coefficient dict.

    Mirrors the contrib computation inside `_fit_one_image_subset`,
    but reads coefficients from an externally-supplied params dict (so
    a caller can substitute U-mode-projected coefficients).
    """
    max_k = max(max(ks) for ks in by_pupil.values())
    A_full, _ = focal_plane_zernike_basis(
        thx_deg, thy_deg, max_k, fp_radius)
    n_donuts = len(thx_deg)
    n_zern = len(iZidx)
    dz_contrib = np.zeros((n_donuts, n_zern))
    for j, ks in by_pupil.items():
        if j not in iZidx:
            continue
        col_j = iZidx[j]
        cols = [k - 1 for k in ks]
        A_j = A_full[:, cols]
        coeffs = np.array([
            float(params.get(f'dz_z{j}_c{k}', 0.0))
            for k in ks])
        if not np.any(np.isfinite(coeffs)):
            continue
        coeffs = np.where(np.isfinite(coeffs), coeffs, 0.0)
        dz_contrib[:, col_j] = A_j @ coeffs
    return dz_contrib


def _apply_uconstraint(params, kj_grid, U_eff):
    """Project a per-(k, j) coefficient dict onto the first ``n_keep``
    U-modes.  Returns (params_proj, w_raw, w_proj) where the raw and
    projected coefficient vectors are in the row order of `kj_grid`.

    `kj_grid` is a list of ``(k, j)`` tuples whose order must match the
    rows of `U_eff` (i.e. how `S` was flattened before the SVD).
    """
    w = np.array([float(params.get(f'dz_z{j}_c{k}', np.nan))
                  for k, j in kj_grid])
    finite = np.isfinite(w)
    w_use = np.where(finite, w, 0.0)
    w_proj = U_eff @ (U_eff.T @ w_use)
    out = dict(params)
    for idx, (k, j) in enumerate(kj_grid):
        out[f'dz_z{j}_c{k}'] = float(w_proj[idx])
    return out, w, w_proj


def build_measured_intrinsic_uconstrained(donut_df, visit_table,
                                          coord_sys, iZs,
                                          kj_grid, U_eff,
                                          n_iter=2, n_bins=73,
                                          fp_radius_basis=1.75,
                                          fp_radius_grid=1.8,
                                          min_donuts=500,
                                          bad_fit_threshold=2.0,
                                          data_offset=None,
                                          intrinsic_offset=None,
                                          converge_tol=1e-3):
    """U-mode-constrained variant of `build_measured_intrinsic`.

    For each visit, fit *all* (k, j) DZ coefficients in `kj_grid`,
    then project the coefficient vector onto the first `n_keep` U-modes
    via ``w_proj = U_eff @ (U_eff.T @ w)`` before reconstructing the
    per-donut DZ contribution.  Same `data_offset` / `intrinsic_offset`
    conventions as the original function (used for the Z4 height
    correction).

    `kj_grid` must list ``(k, j)`` tuples in the same row order that
    was used when computing the SVD of `S`.  `U_eff` is therefore
    ``shape (len(kj_grid), n_keep)``.
    """
    from collections import defaultdict
    spec = defaultdict(list)
    for k, j in kj_grid:
        spec[int(k)].append(int(j))
    pairs, by_pupil, by_focal = expand_removal_spec(dict(spec))
    iZidx = {iZ: i for i, iZ in enumerate(iZs)}

    thx = np.rad2deg(np.asarray(donut_df[f'thx_{coord_sys}'], dtype=float))
    thy = np.rad2deg(np.asarray(donut_df[f'thy_{coord_sys}'], dtype=float))
    zk_data = np.stack(
        donut_df[f'zk_{coord_sys}'].values).astype(float).copy()
    zk_intrinsic_tab = np.stack(
        donut_df[f'zk_intrinsic_{coord_sys}'].values).astype(float).copy()
    dobs_arr = np.asarray(donut_df['day_obs'])
    snum_arr = np.asarray(donut_df['seq_num'])

    if data_offset:
        for j, off in data_offset.items():
            if j in iZidx:
                zk_data[:, iZidx[j]] -= np.asarray(off, dtype=float)
                print(f"  data_offset applied to pupil j={j}")
    if intrinsic_offset:
        for j, off in intrinsic_offset.items():
            if j in iZidx:
                zk_intrinsic_tab[:, iZidx[j]] -= np.asarray(off, dtype=float)
                print(f"  intrinsic_offset applied to pupil j={j}")

    images = sorted(set(zip(dobs_arr.tolist(), snum_arr.tolist())))

    iter_results = []
    intrinsic_per_donut = zk_intrinsic_tab.copy()
    original_median = None
    tabulated_median = None
    xbins = ybins = xcent = ycent = None

    n_modes = sum(len(ks) for ks in by_pupil.values())
    print(f"U-constrained DZ fit:  "
          f"{n_modes} (k,j) modes total; U_eff = {U_eff.shape}  "
          f"(n_keep = {U_eff.shape[1]})")

    for it in range(n_iter):
        fit_rows_proj = []
        fit_rows_raw  = []
        dz_contrib = np.zeros_like(zk_data)
        bar = _tqdm(enumerate(images), total=len(images),
                    desc=f'iter {it + 1}/{n_iter} U-constrained',
                    leave=True)
        for img_idx, (dobs, snum) in bar:
            mask = (dobs_arr == dobs) & (snum_arr == snum)
            params_raw, _ = _fit_one_image_subset(
                thx[mask], thy[mask],
                zk_data[mask], intrinsic_per_donut[mask],
                iZidx, by_pupil, fp_radius_basis,
                dobs, snum, img_idx)
            fit_rows_raw.append(params_raw)
            params_proj, _w, _w_p = _apply_uconstraint(
                params_raw, kj_grid, U_eff)
            fit_rows_proj.append(params_proj)
            dz_contrib[mask] = _dz_contrib_from_params(
                params_proj, by_pupil, thx[mask], thy[mask],
                iZidx, fp_radius_basis)

        bad_visits = _flag_bad_visits(
            fit_rows_proj, by_pupil, bad_fit_threshold, min_donuts)
        bad_set = set(bad_visits)
        good_donut_mask = np.array([
            (int(d), int(s)) not in bad_set
            for d, s in zip(dobs_arr, snum_arr)
        ])
        n_visits_good = len(images) - len(bad_visits)
        n_donuts_good = int(good_donut_mask.sum())

        wfd_subtracted = zk_data - dz_contrib
        measured_grid, xbins, ybins, xcent, ycent = bin_median_focal(
            thx[good_donut_mask], thy[good_donut_mask],
            wfd_subtracted[good_donut_mask],
            iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)

        if it == 0:
            original_median, *_ = bin_median_focal(
                thx[good_donut_mask], thy[good_donut_mask],
                zk_data[good_donut_mask],
                iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)
            tabulated_median, *_ = bin_median_focal(
                thx[good_donut_mask], thy[good_donut_mask],
                zk_intrinsic_tab[good_donut_mask],
                iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)
        if it == n_iter - 1:
            # per-cell donut count + RMS of the FINAL measured intrinsic, for the
            # error on the (median) per-cell value (SE ≈ 1.2533·rms/√count)
            measured_count, measured_rms, *_ = bin_count_rms_focal(
                thx[good_donut_mask], thy[good_donut_mask],
                wfd_subtracted[good_donut_mask],
                iZidx, n_bins=n_bins, fp_radius=fp_radius_grid)

        iter_results.append({
            'fit_rows':         fit_rows_proj,
            'fit_rows_raw':     fit_rows_raw,
            'dz_contrib':       dz_contrib,
            'wfd_subtracted':   wfd_subtracted,
            'measured_grid':    measured_grid,
            'bad_visits':       bad_visits,
            'good_donut_mask':  good_donut_mask,
        })

        intrinsic_per_donut = interpolate_grid_at_donuts(
            measured_grid, xcent, ycent, thx, thy, iZs,
            fallback=zk_intrinsic_tab)

        print(f"  iter {it + 1}/{n_iter}: U-constrained over "
              f"{len(images)} visits ({n_visits_good} good, "
              f"{len(bad_visits)} flagged bad); "
              f"median over {n_donuts_good} donuts "
              f"on {n_bins}x{n_bins} grid")

    _report_convergence(iter_results, converge_tol)
    return {
        'iZs': iZs,
        'iZidx': iZidx,
        'original_median': original_median,
        'tabulated_median': tabulated_median,
        'measured_count': measured_count,      # final-iter per-cell donut count
        'measured_rms': measured_rms,          # final-iter per-cell RMS (per j)
        'iter_results': iter_results,
        'xbins': xbins, 'ybins': ybins,
        'xcent': xcent, 'ycent': ycent,
        'kj_grid': list(kj_grid),
        'U_eff_shape': tuple(U_eff.shape),
    }


# ----------------------------------------------------------------------
# Output table assembly
# ----------------------------------------------------------------------

def assemble_intrinsic_table(grid, iZs, xcent, ycent,
                             coord_sys_grid, alt_coord_xform=None,
                             extra_cols=None, rms_grid=None, tabulated_grid=None):
    """Flatten a per-pupil-j focal grid into a long-format QTable.

    Each row is one (thx, thy) bin centre carrying:
      - `thx`, `thy`              in `coord_sys_grid` (degrees)
      - `thx_alt`, `thy_alt`      in the alternate coord system if
                                   `alt_coord_xform` is provided
      - `zk_<j>` for each pupil j
      - `nollIndices`             list of j (broadcast same value)

    The grid orientation is the same one used by `bin_median_focal`:
    x = thy_deg, y = thx_deg.

    Parameters
    ----------
    grid : dict pupil_j -> 2D array
    iZs : list of int
    xcent, ycent : ndarray (n_bins,) — bin centres in degrees
    coord_sys_grid : 'OCS' or 'CCS' — what the grid coords represent
    alt_coord_xform : callable (thx, thy) -> (thx_alt, thy_alt), optional

    Returns
    -------
    QTable with one row per non-empty bin (NaN bins excluded).
    """
    XX, YY = np.meshgrid(xcent, ycent, indexing='ij')
    # XX[i,k] = thy_centers[i], YY[i,k] = thx_centers[k]  (matches binning)
    thy_flat = XX.ravel()
    thx_flat = YY.ravel()

    cols = {
        'thx_deg': thx_flat,
        'thy_deg': thy_flat,
        'coord_sys': np.array([coord_sys_grid] * len(thx_flat)),
    }
    if alt_coord_xform is not None:
        thx_alt, thy_alt = alt_coord_xform(thx_flat, thy_flat)
        cols['thx_deg_alt'] = thx_alt
        cols['thy_deg_alt'] = thy_alt

    n_pix = len(thx_flat)
    n_zern = len(iZs)
    zk_arr = np.full((n_pix, n_zern), np.nan)
    for col_idx, j in enumerate(iZs):
        if j in grid:
            zk_arr[:, col_idx] = grid[j].ravel()
    cols['zk'] = list(zk_arr)
    cols['nollIndices'] = [list(iZs)] * n_pix

    # parallel per-Zernike list columns (same layout as zk): the donut RMS and
    # the tabulated (batoid) intrinsic, both medianed onto the same grid.
    for cname, cgrid in (('zk_rms', rms_grid), ('zk_tabulated', tabulated_grid)):
        if cgrid is not None:
            arr = np.full((n_pix, n_zern), np.nan)
            for col_idx, j in enumerate(iZs):
                if j in cgrid:
                    arr[:, col_idx] = cgrid[j].ravel()
            cols[cname] = list(arr)

    # Optional extra per-bin columns (e.g. z4_optical_OCS).  Each
    # must be a 2D grid in the same orientation as `grid[j]`.
    if extra_cols:
        for cname, cgrid in extra_cols.items():
            cols[cname] = np.asarray(cgrid, dtype=float).ravel()

    tbl = QTable(cols)
    # Drop bins where every zk is NaN (typically corners outside the FoV).
    keep = ~np.all(np.isnan(zk_arr), axis=1)
    return tbl[keep]


def save_dz_fits(fit_rows, output_path):
    """Save a list of DZ fit dicts to a parquet file."""
    out = QTable(fit_rows)
    out.write(str(output_path), format='parquet', overwrite=True)
    print(f"Saved DZ fit table: {output_path} "
          f"({len(out)} visits, {len(out.columns)} columns)")
    return out


# ----------------------------------------------------------------------
# Filtering helpers
# ----------------------------------------------------------------------

def apply_visit_filters(visit_table,
                        day_obs_min=None, day_obs_max=None,
                        alt_min_deg=None, alt_max_deg=None,
                        rotator_min_deg=None, rotator_max_deg=None,
                        seq_num_min=None, seq_num_max=None):
    """Return a subset of `visit_table` rows that match all filters.

    `alt` is treated as RADIANS in visit_table (Butler convention) — the
    range comparison is performed in degrees against rad2deg(alt).
    `rotator_angle` is already in degrees.
    """
    keep = np.ones(len(visit_table), dtype=bool)
    if 'day_obs' in visit_table.colnames:
        d = np.asarray(visit_table['day_obs'])
        if day_obs_min is not None:
            keep &= d >= day_obs_min
        if day_obs_max is not None:
            keep &= d <= day_obs_max
    if 'seq_num' in visit_table.colnames:
        s = np.asarray(visit_table['seq_num'])
        if seq_num_min is not None:
            keep &= s >= seq_num_min
        if seq_num_max is not None:
            keep &= s <= seq_num_max
    if 'alt' in visit_table.colnames and (
            alt_min_deg is not None or alt_max_deg is not None):
        alt = np.asarray(visit_table['alt'], dtype=float)
        if np.nanmax(np.abs(alt)) < 2.0 * np.pi + 1e-3:
            alt = np.rad2deg(alt)
        if alt_min_deg is not None:
            keep &= alt >= alt_min_deg
        if alt_max_deg is not None:
            keep &= alt <= alt_max_deg
    rot_col = ('rotator_angle' if 'rotator_angle' in visit_table.colnames
               else 'rotAngle' if 'rotAngle' in visit_table.colnames
               else None)
    if rot_col and (rotator_min_deg is not None
                    or rotator_max_deg is not None):
        r = np.asarray(visit_table[rot_col], dtype=float)
        # Half-open bins [min, max): the upper edge is exclusive so a visit at
        # exactly a shared bin edge (FAM sits at discrete rotator angles) lands
        # in only ONE contiguous bin.  The outermost bin's upper edge should be
        # set above the data (e.g. 90 > max |rot|) so nothing is lost.
        if rotator_min_deg is not None:
            keep &= r >= rotator_min_deg
        if rotator_max_deg is not None:
            keep &= r < rotator_max_deg
    return visit_table[keep]
