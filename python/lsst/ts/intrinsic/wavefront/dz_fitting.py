"""Double Zernike focal-plane fitting for Rubin AOS wavefront data.

Fits focal-plane Noll Zernike polynomials (Z1-Z3 or Z1-Z6) to per-image
donut wavefront residuals using robust regression (Huber M-estimator).

Can be used as a library or via the companion CLI script run_dz_fit.py.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from astropy.table import QTable, join
from pathlib import Path



# ============================================================
# Noll index utilities
# ============================================================

def derive_noll_indices(nZk, noll_indices_arr=None):
    """Derive Noll Zernike indices and index mapping.

    Parameters
    ----------
    nZk : int
        Number of Zernike terms in the data array.
    noll_indices_arr : array-like, optional
        Explicit Noll indices (e.g. from visit_info nollIndices column).
        If None, inferred from nZk assuming contiguous from Z4.

    Returns
    -------
    iZs : list of int
        Noll indices (e.g. [4, 5, 6, ..., 22, 23, 24, 25, 26]).
    iZidx : dict
        Mapping from Noll index to column position in zk arrays.
    """
    if noll_indices_arr is not None:
        iZs = [int(n) for n in noll_indices_arr]
        if len(iZs) != nZk:
            print(f"WARNING: nollIndices length ({len(iZs)}) != zk array width ({nZk})")
            print(f"  nollIndices: {iZs}")
            print(f"  Falling back to contiguous range")
            iZs = list(range(4, 4 + nZk))
    else:
        if nZk == 19:
            iZs = list(range(4, 23))
        else:
            iZs = list(range(4, 4 + nZk))

    iZidx = {iZ: i for i, iZ in enumerate(iZs)}
    return iZs, iZidx


# ============================================================
# Focal-plane Zernike basis
# ============================================================

def focal_plane_zernike_basis(thx_deg, thy_deg, max_noll, fp_radius=1.75):
    """Build focal-plane Noll Zernike basis matrix.

    Coordinates are normalized to the focal-plane radius so that the
    Zernike polynomials are evaluated on a unit disk.  All basis functions
    are dimensionless, so fit coefficients have the same units as the data (μm).

    Parameters
    ----------
    thx_deg, thy_deg : ndarray
        Field angles in degrees.
    max_noll : int
        Maximum Noll index (1-6 supported).
    fp_radius : float
        Focal plane radius in degrees for normalization (default 1.75).

    Returns
    -------
    A : ndarray, shape (n_points, max_noll)
        Design matrix with one column per focal Zernike term.
    labels : list of str
        Labels for each column (e.g. 'Z1_piston', 'Z2_tilt', ...).
    """
    x = thx_deg / fp_radius
    y = thy_deg / fp_radius
    r2 = x**2 + y**2

    cols = []
    labels = []

    if max_noll >= 1:
        cols.append(np.ones_like(x))
        labels.append('Z1_piston')
    if max_noll >= 2:
        cols.append(2.0 * x)
        labels.append('Z2_tilt')
    if max_noll >= 3:
        cols.append(2.0 * y)
        labels.append('Z3_tip')
    if max_noll >= 4:
        cols.append(np.sqrt(3) * (2.0 * r2 - 1.0))
        labels.append('Z4_defocus')
    if max_noll >= 5:
        cols.append(2.0 * np.sqrt(6) * x * y)
        labels.append('Z5_astig45')
    if max_noll >= 6:
        cols.append(np.sqrt(6) * (x**2 - y**2))
        labels.append('Z6_astig0')

    return np.column_stack(cols), labels


# ============================================================
# Core fitting function
# ============================================================

def fit_focal_zernikes(day_obs_arr, seq_num_arr, thx_deg, thy_deg,
                       zk_data, zk_intrinsic, iZs,
                       max_focal_noll=3, include_intrinsic=True,
                       fp_radius=1.75, prefix='z1toz3'):
    """Fit focal-plane Noll Zernikes to per-image wavefront residuals.

    For each image (unique day_obs, seq_num) and each pupil Zernike iZ, fits:
        residual = k1*Zfocal_1 + k2*Zfocal_2 + ... + kN*Zfocal_N
    where residual = zk_data - zk_intrinsic (if include_intrinsic) or zk_data.

    Uses robust regression (Huber M-estimator) with fallback to least squares.

    Parameters
    ----------
    day_obs_arr : ndarray of int
        Day observation IDs per donut.
    seq_num_arr : ndarray of int
        Sequence numbers per donut.
    thx_deg, thy_deg : ndarray
        Field angles in degrees per donut.
    zk_data : ndarray, shape (n_donuts, n_zernikes)
        Measured Zernike values in μm.
    zk_intrinsic : ndarray, shape (n_donuts, n_zernikes)
        Intrinsic model Zernike values in μm.
    iZs : list of int
        Noll indices corresponding to columns of zk_data/zk_intrinsic.
    max_focal_noll : int
        Maximum focal Noll index for fit (default 3).
    include_intrinsic : bool
        If True, subtract intrinsic before fitting (default True).
    fp_radius : float
        Focal plane radius in degrees (default 1.75).
    prefix : str
        Column name prefix for output (e.g. 'z1toz3').

    Returns
    -------
    fit_rows : list of dict
        One dict per image with fit parameters.
    zk_fit_vals : ndarray, shape (n_donuts, n_zernikes)
        Per-donut fitted values.
    zk_rlm_weights : ndarray, shape (n_donuts, n_zernikes)
        Per-donut RLM weights.
    """
    images = sorted(set(zip(day_obs_arr.tolist(), seq_num_arr.tolist())))
    n_donuts = len(day_obs_arr)
    n_zernikes = len(iZs)

    zk_fit_vals = np.zeros((n_donuts, n_zernikes))
    zk_rlm_weights = np.ones((n_donuts, n_zernikes))
    fit_rows = []

    for img_idx, (dobs, snum) in enumerate(images):
        mask = (day_obs_arr == dobs) & (seq_num_arr == snum)
        img_params, fit_vals_i, weights_i = _fit_one_image(
            thx_deg[mask], thy_deg[mask],
            zk_data[mask], zk_intrinsic[mask],
            iZs, max_focal_noll, include_intrinsic, fp_radius, prefix,
            dobs, snum, img_idx)
        zk_fit_vals[mask] = fit_vals_i
        zk_rlm_weights[mask] = weights_i
        fit_rows.append(img_params)

    print(f"Fit '{prefix}' (focal Noll 1-{max_focal_noll}): "
          f"{len(images)} images, {n_donuts} donuts, "
          f"include_intrinsic={include_intrinsic}")

    return fit_rows, zk_fit_vals, zk_rlm_weights


def _fit_one_image(thx_deg, thy_deg, zk_data, zk_intrinsic, iZs,
                   max_focal_noll, include_intrinsic, fp_radius, prefix,
                   dobs, snum, img_idx):
    """Fit one image's donuts. Returns (img_params, fit_vals, rlm_weights)."""
    A, _ = focal_plane_zernike_basis(thx_deg, thy_deg, max_focal_noll, fp_radius)
    n_pts = len(thx_deg)
    n_zernikes = len(iZs)
    n_coeffs = max_focal_noll

    fit_vals = np.zeros((n_pts, n_zernikes))
    rlm_weights = np.ones((n_pts, n_zernikes))
    img_params = {'day_obs': int(dobs), 'seq_num': int(snum),
                  'image_idx': int(img_idx), 'n_donuts': int(n_pts)}

    for j_idx, iZ in enumerate(iZs):
        if include_intrinsic:
            resid = zk_data[:, j_idx] - zk_intrinsic[:, j_idx]
        else:
            resid = zk_data[:, j_idx].copy()

        try:
            rlm_model = sm.RLM(resid, A, M=sm.robust.norms.HuberT())
            rlm_results = rlm_model.fit()
            coeffs = rlm_results.params
            bse = rlm_results.bse
            scale = float(rlm_results.scale)
            weights = rlm_results.weights
        except Exception as e:
            print(f"  RLM fit failed (falling back to lstsq) for "
                  f"day_obs={int(dobs)} seq_num={int(snum)} z{iZ}: "
                  f"{type(e).__name__}: {e}")
            coeffs, _, _, _ = np.linalg.lstsq(A, resid, rcond=None)
            bse = np.full(n_coeffs, np.nan)
            scale = float(np.std(resid - A @ coeffs))
            weights = np.ones(n_pts)

        for ci in range(n_coeffs):
            img_params[f'{prefix}_z{iZ}_c{ci+1}'] = float(coeffs[ci])
            img_params[f'{prefix}_z{iZ}_c{ci+1}_err'] = float(bse[ci])
        img_params[f'{prefix}_z{iZ}_scale'] = scale

        fit_vals[:, j_idx] = A @ coeffs
        rlm_weights[:, j_idx] = weights

    return img_params, fit_vals, rlm_weights


def _intrinsic_key(dobs, snum, det, cx, cy, cxe, cye):
    # Round centroids to the nearest pixel: donuts are >> 1 px apart, so this is
    # a unique key, while avoiding lookup misses from sub-LSB float drift across
    # a parquet read/write round-trip (the old round(...,3) keyed on 0.001 px).
    # Both intra- and extra-focal centroids are included so the key is unique
    # even if two donuts in a visit+detector share a rounded intra centroid.
    return (int(dobs), int(snum), str(det),
            round(float(cx)), round(float(cy)),
            round(float(cxe)), round(float(cye)))


def load_intrinsic_lookup(sidecar_path, iZs):
    """Build a per-donut measured-intrinsic lookup from a zk_intrinsic sidecar
    (run_make_intrinsic_sidecar.py), reordered to the fit's ``iZs``.

    Key: (day_obs, seq_num, detector, centroid_x/y_intra, centroid_x/y_extra).
    Value: the zk_intrinsic_MI vector reordered to iZs column order.
    """
    t = pq.read_table(str(sidecar_path))
    meta = t.schema.metadata or {}
    side_noll = (np.frombuffer(meta[b'nollIndices'], dtype=int).tolist()
                 if b'nollIndices' in meta else list(iZs))
    col = [side_noll.index(int(j)) for j in iZs]      # reorder to fit iZs
    df = t.to_pandas()
    mi = np.stack(df['zk_intrinsic_MI'].values)[:, col]
    dobs = df['day_obs'].to_numpy(); snum = df['seq_num'].to_numpy()
    det = df['detector'].astype(str).to_numpy()
    cx = df['centroid_x_intra'].to_numpy(float); cy = df['centroid_y_intra'].to_numpy(float)
    cxe = df['centroid_x_extra'].to_numpy(float); cye = df['centroid_y_extra'].to_numpy(float)
    return {_intrinsic_key(dobs[r], snum[r], det[r],
                           cx[r], cy[r], cxe[r], cye[r]): mi[r]
            for r in range(len(df))}


def fit_focal_zernikes_streaming(input_file, visit_info, coord_sys, iZs,
                                 max_focal_noll=3, include_intrinsic=True,
                                 fp_radius=1.75, prefix='z1toz3',
                                 intrinsic_lookup=None):
    """Streaming variant: read donuts one row group (= one visit) at a time.

    Reads the donuts parquet file written by stream_zernikes_to_parquet,
    where each visit is stored as a single row group. Per-visit reads
    use row-group stats to avoid scanning the whole file.

    Returns fit_rows (list of dicts) — zk_fit_vals and zk_rlm_weights
    are not accumulated (they weren't used downstream anyway).
    """
    pf = pq.ParquetFile(str(input_file))
    fit_rows = []
    total_donuts = 0

    thx_col = f'thx_{coord_sys}'
    thy_col = f'thy_{coord_sys}'
    zk_col = f'zk_{coord_sys}'
    zk_intr_col = f'zk_intrinsic_{coord_sys}'

    # Build a (day_obs, seq_num) -> row_group_idx lookup from row-group stats
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

    for img_idx, v in enumerate(visit_info):
        dobs = int(v['day_obs'])
        snum = int(v['seq_num'])
        rg_i = rg_index.get((dobs, snum))
        if rg_i is None:
            continue

        df = pf.read_row_group(rg_i).to_pandas()
        if len(df) == 0:
            continue

        thx_deg = np.rad2deg(df[thx_col].to_numpy(dtype=float))
        thy_deg = np.rad2deg(df[thy_col].to_numpy(dtype=float))
        zk_data = np.stack(df[zk_col].values)
        if intrinsic_lookup is not None:
            # Use the measured-intrinsic sidecar instead of the tabulated
            # zk_intrinsic; drop donuts without an MI value (NaN).
            det = df['detector'].astype(str).to_numpy()
            cx = df['centroid_x_intra'].to_numpy(float)
            cy = df['centroid_y_intra'].to_numpy(float)
            cxe = df['centroid_x_extra'].to_numpy(float)
            cye = df['centroid_y_extra'].to_numpy(float)
            zk_intrinsic = np.full_like(zk_data, np.nan)
            for r in range(len(df)):
                mi_val = intrinsic_lookup.get(
                    _intrinsic_key(dobs, snum, det[r],
                                   cx[r], cy[r], cxe[r], cye[r]))
                if mi_val is not None:
                    zk_intrinsic[r] = mi_val
            fin = np.isfinite(zk_intrinsic).all(axis=1)
            if not fin.any():
                continue
            thx_deg, thy_deg = thx_deg[fin], thy_deg[fin]
            zk_data, zk_intrinsic = zk_data[fin], zk_intrinsic[fin]
        else:
            zk_intrinsic = np.stack(df[zk_intr_col].values)

        img_params, _, _ = _fit_one_image(
            thx_deg, thy_deg, zk_data, zk_intrinsic, iZs,
            max_focal_noll, include_intrinsic, fp_radius, prefix,
            dobs, snum, img_idx)
        fit_rows.append(img_params)
        total_donuts += len(df)

    print(f"Fit '{prefix}' (focal Noll 1-{max_focal_noll}): "
          f"{len(fit_rows)} images, {total_donuts} donuts (streamed), "
          f"include_intrinsic={include_intrinsic}")

    return fit_rows


# ============================================================
# Scalar focal-plane fit (used for donut_blur and similar per-donut
# scalar fields whose focal-plane variation we want to characterize
# with the same Z1..Zn basis used for the Zernike fits).
# ============================================================

def _fit_one_image_scalar(thx_deg, thy_deg, values, max_focal_noll,
                          fp_radius, prefix, dobs, snum, img_idx):
    """Fit `values` (1-D scalar per donut) to focal-plane Noll Z1..Zn.

    Returns an img_params dict keyed by `{prefix}_c{ci+1}` (and `_err`),
    plus `{prefix}_scale`, `n_donuts`, `image_idx`, `day_obs`, `seq_num`.
    Robust regression (Huber M-estimator) with lstsq fallback, mirroring
    `_fit_one_image` but for a scalar field instead of a Zernike vector.
    """
    valid = (np.isfinite(values) & np.isfinite(thx_deg)
             & np.isfinite(thy_deg))
    n_pts = int(valid.sum())
    img_params = {'day_obs': int(dobs), 'seq_num': int(snum),
                  'image_idx': int(img_idx), 'n_donuts': n_pts}

    if n_pts < max_focal_noll + 1:
        for ci in range(max_focal_noll):
            img_params[f'{prefix}_c{ci + 1}'] = np.nan
            img_params[f'{prefix}_c{ci + 1}_err'] = np.nan
        img_params[f'{prefix}_scale'] = np.nan
        return img_params

    A, _ = focal_plane_zernike_basis(
        thx_deg[valid], thy_deg[valid], max_focal_noll, fp_radius)
    vals = values[valid]
    try:
        rlm = sm.RLM(vals, A, M=sm.robust.norms.HuberT()).fit()
        coeffs = rlm.params
        bse = rlm.bse
        scale = float(rlm.scale)
    except Exception:
        coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
        bse = np.full(max_focal_noll, np.nan)
        scale = float(np.std(vals - A @ coeffs))

    for ci in range(max_focal_noll):
        img_params[f'{prefix}_c{ci + 1}'] = float(coeffs[ci])
        img_params[f'{prefix}_c{ci + 1}_err'] = float(bse[ci])
    img_params[f'{prefix}_scale'] = scale
    return img_params


def fit_focal_scalar(day_obs_arr, seq_num_arr, thx_deg, thy_deg,
                     values, max_focal_noll=6, fp_radius=1.75,
                     prefix='blur'):
    """Fit a scalar per-donut field over the focal plane (in-memory)."""
    images = sorted(set(zip(day_obs_arr.tolist(), seq_num_arr.tolist())))
    fit_rows = []
    for img_idx, (dobs, snum) in enumerate(images):
        mask = (day_obs_arr == dobs) & (seq_num_arr == snum)
        img_params = _fit_one_image_scalar(
            thx_deg[mask], thy_deg[mask], values[mask],
            max_focal_noll, fp_radius, prefix, dobs, snum, img_idx)
        fit_rows.append(img_params)
    print(f"Fit '{prefix}' (focal Noll 1-{max_focal_noll}, scalar): "
          f"{len(images)} images")
    return fit_rows


def fit_focal_scalar_streaming(input_file, visit_info, coord_sys,
                               value_col='donut_blur',
                               max_focal_noll=6, fp_radius=1.75,
                               prefix='blur'):
    """Streaming scalar fit: read donut parquet one row group at a time.

    Returns list of img_params dicts.  Skips visits whose row group lacks
    `value_col` or has fewer than `max_focal_noll + 1` valid points.
    """
    pf = pq.ParquetFile(str(input_file))

    # Bail early if the column is absent from the schema entirely.
    if value_col not in pf.schema_arrow.names:
        print(f"  '{value_col}' not found in {input_file}; skipping "
              f"'{prefix}' fit")
        return []

    fit_rows = []
    total = 0
    thx_col = f'thx_{coord_sys}'
    thy_col = f'thy_{coord_sys}'

    # Build a (day_obs, seq_num) -> row_group_idx lookup from row-group stats
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

    for img_idx, v in enumerate(visit_info):
        dobs = int(v['day_obs'])
        snum = int(v['seq_num'])
        rg_i = rg_index.get((dobs, snum))
        if rg_i is None:
            continue
        df = pf.read_row_group(rg_i).to_pandas()
        if len(df) == 0 or value_col not in df.columns:
            continue
        thx_deg = np.rad2deg(df[thx_col].to_numpy(dtype=float))
        thy_deg = np.rad2deg(df[thy_col].to_numpy(dtype=float))
        vals = df[value_col].to_numpy(dtype=float)
        img_params = _fit_one_image_scalar(
            thx_deg, thy_deg, vals, max_focal_noll, fp_radius, prefix,
            dobs, snum, img_idx)
        fit_rows.append(img_params)
        total += len(df)

    print(f"Fit '{prefix}' from '{value_col}' "
          f"(focal Noll 1-{max_focal_noll}): "
          f"{len(fit_rows)} images, {total} donuts (streamed)")
    return fit_rows


# ============================================================
# Bad-fit flagging
# ============================================================

def flag_bad_fits(fit_table, prefix, threshold=2.0, min_donuts=200):
    """Flag visits with bad fits based on coefficient magnitude and donut count.

    Parameters
    ----------
    fit_table : QTable
        Fit parameter table (one row per image).
    prefix : str
        Fit prefix (e.g. 'z1toz3').
    threshold : float
        Maximum allowed |coefficient| in μm (default 2.0).
    min_donuts : int
        Minimum donuts required for a valid fit (default 200).

    Returns
    -------
    bad_mask : ndarray of bool
        True for bad-fit rows.
    """
    coeff_cols = [c for c in fit_table.colnames if c.startswith(f'{prefix}_z')
                  and '_c' in c and not c.endswith('_err') and not c.endswith('_scale')]
    coeff_arr = np.column_stack([np.array(fit_table[c]) for c in coeff_cols])
    bad_coeff = np.any(np.abs(coeff_arr) > threshold, axis=1)
    bad_ndonuts = np.array(fit_table['n_donuts']) < min_donuts
    bad_mask = bad_coeff | bad_ndonuts

    n_bad = np.sum(bad_mask)
    n_bad_coeff = np.sum(bad_coeff & ~bad_ndonuts)
    n_bad_ndonuts = np.sum(bad_ndonuts)
    print(f"{prefix}: {n_bad}/{len(fit_table)} visits flagged as bad_fit")
    print(f"  {n_bad_coeff} with |coeff| > {threshold} μm, "
          f"{n_bad_ndonuts} with n_donuts < {min_donuts}")

    if n_bad > 0:
        for i in range(len(fit_table)):
            if not bad_mask[i]:
                continue
            row = fit_table[i]
            reasons = []
            if row['n_donuts'] < min_donuts:
                reasons.append(f"n_donuts={row['n_donuts']}")
            for c in coeff_cols:
                if abs(row[c]) > threshold:
                    reasons.append(f"{c}={row[c]:.3f}")
            print(f"  day_obs={row['day_obs']} seq_num={row['seq_num']}  "
                  + ', '.join(reasons))

    return bad_mask


# ============================================================
# High-level pipeline
# ============================================================

def run_double_zernike_fits(input_file, coord_sys='OCS',
                            output_file=None, bad_fit_threshold=2.0,
                            min_donuts=200, visits_file=None,
                            intrinsic_sidecar=None, min_detectors=None):
    """Run the full Double Zernike fitting pipeline.

    Loads input HDF5 (donuts + visits tables), derives Noll indices,
    validates data, runs z1toz3 and z1toz6 fits, flags bad fits, merges
    with visit_info, and saves output.

    Parameters
    ----------
    input_file : str or Path
        Path to HDF5 file containing 'donuts' and 'visits' tables
        (from intrinsics_mktable).
    coord_sys : str
        Coordinate system: 'OCS' or 'CCS'.
    output_file : str or Path, optional
        Output parquet path. If None, derived as {stem}_fits.parquet.
    bad_fit_threshold : float
        Flag fits with |coefficient| > this (μm). Default 2.0.
    min_donuts : int
        Flag fits with fewer donuts than this. Default 200.
    min_detectors : int or None
        Opt-in override of the per-visit quality selection.  When None (default)
        behavior is unchanged — the precomputed ``visit_quality_pass`` column is
        used as-is (or ``quality_visit_mask`` with default thresholds).  When set,
        the mask is recomputed from the metric columns with
        ``min_detectors_per_visit=min_detectors``, relaxing ONLY the
        ``n_detectors_with_min_donuts`` cut (the n_donuts / blur cuts keep their
        defaults) and bypassing the precomputed flag (which is fixed at 170).
        Used by the bounce analysis to recover marginal low-CCD visits; leave
        None for the MIW calibration and all other outputs.

    Returns
    -------
    fit_merged : QTable
        Combined fit table with both z1toz3 and z1toz6 results.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Detect format by suffix: .parquet (new, streaming) or .hdf5 (legacy)
    is_parquet = input_file.suffix == '.parquet'

    if is_parquet:
        # Resolve the visits sidecar.  An explicit visits_file wins; otherwise
        # try the legacy <stem>_visits.parquet, then fall back to visits.parquet
        # in the same directory (the Snakemake output/<ps>/chunks/<...>/ layout).
        if visits_file is not None:
            visits_file = Path(visits_file)
        else:
            cand = input_file.parent / f'{input_file.stem}_visits.parquet'
            visits_file = cand if cand.exists() else input_file.parent / 'visits.parquet'
        if output_file is None:
            output_file = input_file.parent / f'{input_file.stem}_fits.parquet'
        print(f"Loading: {input_file} (parquet, one row group per visit)")
        print(f"  visits: {visits_file}")
        visit_info = QTable.read(str(visits_file))
        print(f"  {len(visit_info)} visits")
    else:
        # Legacy HDF5: donuts and visits in same file
        if output_file is None:
            output_file = input_file.parent / f'{input_file.stem}_fits.parquet'
        print(f"Loading: {input_file} (legacy HDF5)")
        visit_info = QTable.read(str(input_file), path='visits')
        print(f"  {len(visit_info)} visits")

    # Apply per-visit quality cuts (n_donuts, n_detectors, median_blur_arcsec)
    # if those columns are present (mktable >= 2026-05-06).
    if min_detectors is not None:
        # Opt-in override (bounce): recompute the mask from the metric columns,
        # relaxing ONLY the CCD-count cut to `min_detectors` and keeping the
        # standard n_donuts / blur cuts.  Deliberately bypasses the precomputed
        # visit_quality_pass (fixed at min_detectors=170) so this DOES NOT change
        # the default path used by MIW calibration and every other output.
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
        keep = quality_visit_mask(visit_info, min_detectors_per_visit=min_detectors,
                                  verbose=True)
        print(f"  Per-visit quality cuts (min_detectors={min_detectors} override): "
              f"{int(keep.sum())}/{len(visit_info)} visits pass")
        visit_info = visit_info[keep]
    elif 'visit_quality_pass' in visit_info.colnames:
        keep = np.asarray(visit_info['visit_quality_pass'], dtype=bool)
        n_pass = int(keep.sum())
        print(f"  Applying per-visit quality cuts: "
              f"{n_pass}/{len(visit_info)} visits pass")
        visit_info = visit_info[keep]
    elif {'n_donuts', 'n_detectors_with_min_donuts',
          'median_blur_arcsec'}.issubset(set(visit_info.colnames)):
        # Visit metrics present but no precomputed flag — apply on the fly
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
        keep = quality_visit_mask(visit_info, verbose=True)
        visit_info = visit_info[keep]

    # Derive Noll indices
    noll_arr = None
    if 'nollIndices' in visit_info.colnames:
        noll_arr = np.array(visit_info['nollIndices'][0])

    if is_parquet:
        # Probe one row group to determine nZk
        pf = pq.ParquetFile(str(input_file))
        df0 = pf.read_row_group(0).to_pandas()
        zk_sample = np.stack(df0[f'zk_{coord_sys}'].values)
        nZk = zk_sample.shape[1]
        iZs, iZidx = derive_noll_indices(nZk, noll_arr)
        print(f"  Noll indices ({len(iZs)} terms): {iZs}")
        del df0, zk_sample, pf

        ilookup = None
        if intrinsic_sidecar is not None:
            ilookup = load_intrinsic_lookup(intrinsic_sidecar, iZs)
            print(f"  Using measured-intrinsic sidecar: {intrinsic_sidecar} "
                  f"({len(ilookup)} donuts)")

        # Fit via per-visit row-group reads
        rows_z3 = fit_focal_zernikes_streaming(
            input_file, visit_info, coord_sys, iZs,
            max_focal_noll=3, prefix='z1toz3', intrinsic_lookup=ilookup)
        rows_z6 = fit_focal_zernikes_streaming(
            input_file, visit_info, coord_sys, iZs,
            max_focal_noll=6, prefix='z1toz6', intrinsic_lookup=ilookup)
    else:
        # Legacy path: load the whole donuts table via astropy
        aosTable = QTable.read(str(input_file), path='donuts')
        print(f"  {len(aosTable)} donuts, {len(aosTable.columns)} columns")

        zk_data = np.stack(aosTable[f'zk_{coord_sys}'])
        zk_intrinsic = np.stack(aosTable[f'zk_intrinsic_{coord_sys}'])
        nZk = zk_data.shape[1]
        iZs, iZidx = derive_noll_indices(nZk, noll_arr)
        print(f"  Noll indices ({len(iZs)} terms): {iZs}")

        # Validate zk = residual + intrinsic (legacy-only sanity check)
        resid_col = f'zk_residual_{coord_sys}'
        if resid_col in aosTable.colnames:
            zk_resid = np.stack(aosTable[resid_col])
            diff = zk_data - (zk_resid + zk_intrinsic)
            max_abs_diff = np.max(np.abs(diff))
            print(f"  Validation: max |zk - (residual + intrinsic)| = "
                  f"{max_abs_diff:.2e} μm",
                  "PASSED" if max_abs_diff <= 0.01 else "WARNING")

        day_obs_arr = np.array(aosTable['day_obs'])
        seq_num_arr = np.array(aosTable['seq_num'])
        thx_deg = np.rad2deg(np.array(aosTable[f'thx_{coord_sys}']))
        thy_deg = np.rad2deg(np.array(aosTable[f'thy_{coord_sys}']))

        rows_z3, _, _ = fit_focal_zernikes(
            day_obs_arr, seq_num_arr, thx_deg, thy_deg,
            zk_data, zk_intrinsic, iZs,
            max_focal_noll=3, prefix='z1toz3')
        rows_z6, _, _ = fit_focal_zernikes(
            day_obs_arr, seq_num_arr, thx_deg, thy_deg,
            zk_data, zk_intrinsic, iZs,
            max_focal_noll=6, prefix='z1toz6')

    fit_table_z3 = QTable(rows_z3)
    fit_table_z6 = QTable(rows_z6)

    # Flag bad fits
    bad_z3 = flag_bad_fits(fit_table_z3, 'z1toz3', bad_fit_threshold, min_donuts)
    fit_table_z3['z1toz3_bad_fit'] = bad_z3
    bad_z6 = flag_bad_fits(fit_table_z6, 'z1toz6', bad_fit_threshold, min_donuts)
    fit_table_z6['z1toz6_bad_fit'] = bad_z6

    # Combine into single table
    fit_combined = fit_table_z3.copy()
    for col in fit_table_z6.colnames:
        if col.startswith('z1toz6_'):
            fit_combined[col] = fit_table_z6[col]
    fit_combined['bad_fit'] = bad_z3 | bad_z6
    n_bad = np.sum(fit_combined['bad_fit'])
    print(f"\nCombined: {n_bad}/{len(fit_combined)} visits flagged as bad_fit")

    # ----- Optional donut_blur fit (k=1..6) -----
    # Per-donut scalar 'donut_blur' is treated like a Zernike: fit the
    # focal-plane variation with the same Z1..Z6 basis.  Output columns
    # blur_c1..blur_c6 (and _err / _scale) are merged into fit_combined.
    if is_parquet:
        pf_check = pq.ParquetFile(str(input_file))
        has_blur = 'donut_blur' in pf_check.schema_arrow.names
        del pf_check
    else:
        has_blur = 'donut_blur' in aosTable.colnames

    if has_blur:
        print("\nFitting donut_blur over focal plane (k=1..6)...")
        if is_parquet:
            blur_rows = fit_focal_scalar_streaming(
                input_file, visit_info, coord_sys,
                value_col='donut_blur',
                max_focal_noll=6, prefix='blur')
        else:
            blur_arr = np.array(aosTable['donut_blur'], dtype=float)
            blur_rows = fit_focal_scalar(
                day_obs_arr, seq_num_arr, thx_deg, thy_deg, blur_arr,
                max_focal_noll=6, prefix='blur')
        if blur_rows:
            fit_table_blur = QTable(blur_rows)
            keep = ['day_obs', 'seq_num'] + [
                c for c in fit_table_blur.colnames if c.startswith('blur_')]
            fit_combined = join(
                fit_combined, fit_table_blur[keep],
                keys=['day_obs', 'seq_num'], join_type='left')
            print(f"  blur fit merged: {len(fit_table_blur)} visits, "
                  f"{len([c for c in keep if c.startswith('blur_')])} "
                  f"new columns")
    else:
        print("\nNo 'donut_blur' column found — skipping blur fit")

    # Merge with visit_info
    fit_merged = join(fit_combined, visit_info,
                      keys=['day_obs', 'seq_num'], join_type='left')
    # left join must not add/drop rows; a visit missing from visit_info would
    # silently carry NaN metadata — surface it instead
    assert len(fit_merged) == len(fit_combined), (
        f"visit_info join changed row count "
        f"({len(fit_combined)} -> {len(fit_merged)}); a visit is missing or "
        f"duplicated in visit_info")
    print(f"Merged with visit_info: {len(fit_merged)} rows, "
          f"{len(fit_merged.columns)} columns")

    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fit_merged.write(str(output_file), format='parquet', overwrite=True)
    print(f"\nSaved: {output_file}")
    print(f"  {len(fit_merged)} rows x {len(fit_merged.columns)} columns")

    return fit_merged
