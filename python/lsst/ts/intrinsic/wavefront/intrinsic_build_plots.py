"""Diagnostic plots and analysis helpers for building the measured intrinsic.

Extracted verbatim from build_measured_intrinsic.ipynb so the build can run as a
script (run_build_intrinsic.py).  Pure plotting + per-visit analysis helpers;
the numeric core lives in measured_intrinsic.py and ofc_svd.py.  RSP-only
(needs the LSST stack via the imports below).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import binned_statistic, binned_statistic_2d

from lsst.ts.intrinsic.wavefront.common.zernike_names import (
    NOLL_NAMES, NOLL_FORMULAS, FOCAL_NAMES, PUPIL_NAMES)
from lsst.ts.intrinsic.wavefront.measured_intrinsic import bin_median_focal, interpolate_grid_at_donuts
from lsst.ts.intrinsic.wavefront.ofc_svd import (
    recover_dof_per_visit, vmode_fwhm_scale, LABELS_50DOF, DOF_UNITS_50)

try:
    from lsst.ts.intrinsic.wavefront.intrinsics_lib import (classify_visit, visit_marker_style,
                                build_visit_marker_lookup, markers_legend_figure)
except Exception:  # marker styling is optional for the plots
    classify_visit = visit_marker_style = build_visit_marker_lookup = None
    markers_legend_figure = None
try:
    from lsst.ts.intrinsic.wavefront.ccd_height import HEIGHT_TO_Z4_UM_PER_MM
except Exception:
    HEIGHT_TO_Z4_UM_PER_MM = 15.0

# WEP utility for Zernike -> PSF-FWHM conversion (LSST env only).
try:
    from lsst.ts.wep.utils import convertZernikesToPsfWidth
    _wep_ok = True
except Exception:
    convertZernikesToPsfWidth = None
    _wep_ok = False

# DOF axis grouping for the V-plot (label, start, stop), notebook layout.
DOF_GROUPS = [
    ('M2 Hex',    0,  5),
    ('Cam Hex',   5, 10),
    ('M1M3',     10, 30),
    ('M2',       30, 50),
]
_WFS_EDGE_CACHE = {}



def _per_j_panel_layout(iZs_arr):
    """Page layout: panel 0 = (1,4) alone, panel 1 = (k=2..6 of j=4),
    panels 2..21 = each remaining j in iZs_arr with k = 1..6."""
    js_rest = [int(j) for j in iZs_arr if int(j) != 4]
    return js_rest


def _set_dof_yaxis(ax):
    for _, start, _stop in DOF_GROUPS[1:]:
        ax.axhline(start - 0.5, color='black', lw=0.5, alpha=0.5)
    # Place group labels well to the left of the tick numbers so they
    # never overlap; use axes-fraction coords on x.
    for name, start, stop in DOF_GROUPS:
        y_data = 0.5 * (start + stop - 1)
        ax.annotate(name,
                    xy=(0, y_data), xycoords=('axes fraction', 'data'),
                    xytext=(-44, 0), textcoords='offset points',
                    ha='right', va='center', rotation=90,
                    fontsize=9)
    # Leave room on the left for the rotated labels.
    ax.figure.subplots_adjust(left=0.16)


def _set_kj_yaxis(ax, n_k, n_j, k_min, iZs_arr, fontsize=6):
    n_rows = n_k * n_j
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f'({k_min+ki},{iZs_arr[ji]})'
                        for ki in range(n_k) for ji in range(n_j)],
                       fontsize=fontsize)
    for ki in range(1, n_k):
        ax.axhline(ki * n_j - 0.5, color='black', lw=0.4, alpha=0.4)


def _one_based_xticks(ax, n_modes, step=None):
    """Label the SVD-mode x-axis 1..n_modes (instead of 0..n-1)."""
    if step is None:
        step = max(1, n_modes // 12)
    pos = np.arange(0, n_modes, step)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(int(p + 1)) for p in pos])


def _elev_group(alt_deg):
    if not np.isfinite(alt_deg):
        return 'unknown'
    if alt_deg < 35: return '<35'
    if alt_deg < 50: return '35-50'
    if alt_deg < 65: return '50-65'
    if alt_deg < 80: return '65-80'
    return '>=80'


def _rot_group(rot_deg):
    if not np.isfinite(rot_deg):
        return 'unknown'
    r = abs(rot_deg)
    if r <= 3:   return '|rot|<=3'
    if r <= 45:  return '3<|rot|<=45'
    return '|rot|>45'


def _build_visit_lookup(visits_table):
    """{(day_obs, seq_num): {alt_deg, rot_deg}} keyed by visit."""
    if visits_table is None:
        return {}
    cols = visits_table.colnames
    dobs = np.asarray(visits_table['day_obs']).tolist()
    snum = np.asarray(visits_table['seq_num']).tolist()
    if 'alt' in cols:
        alt = np.asarray(visits_table['alt'], dtype=float)
        if np.nanmax(np.abs(alt)) < 2.0 * np.pi + 1e-3:
            alt = np.rad2deg(alt)
    else:
        alt = np.full(len(visits_table), np.nan)
    rot_col = ('rotator_angle' if 'rotator_angle' in cols
               else 'rotAngle' if 'rotAngle' in cols
               else None)
    rot = (np.asarray(visits_table[rot_col], dtype=float)
           if rot_col else np.full(len(visits_table), np.nan))
    return {(int(d), int(s)): {'alt_deg': float(a), 'rot_deg': float(r)}
            for d, s, a, r in zip(dobs, snum, alt, rot)}


def removal_spec_kj_list(by_focal):
    """Flatten a `{k: [j, ...]}` removal spec into a sorted list of
    ``(k, j)`` tuples (k outer, j inner)."""
    return sorted(((int(k), int(j))
                   for k, js in by_focal.items() for j in js),
                  key=lambda kj: (kj[0], kj[1]))


def stack_per_visit_coeffs(fit_rows, kj_list):
    """Stack a list of per-visit param dicts into a (n_visits, n_kj) array.

    `kj_list` is the row order (a list of ``(k, j)`` tuples).  Missing
    entries are NaN.  Bad-flagged visits are kept (they will appear as
    NaN columns if the fit didn't populate the keys).
    """
    n_v = len(fit_rows)
    n_m = len(kj_list)
    W = np.full((n_v, n_m), np.nan)
    for v, row in enumerate(fit_rows):
        for m, (k, j) in enumerate(kj_list):
            W[v, m] = float(row.get(f'dz_z{j}_c{k}', np.nan))
    return W


def _stack_per_visit_err_A(fit_rows, kj_list):
    n_v = len(fit_rows); n_m = len(kj_list)
    E = np.full((n_v, n_m), np.nan)
    for v, row in enumerate(fit_rows):
        for m, (k, j) in enumerate(kj_list):
            E[v, m] = float(row.get(f'dz_z{j}_c{k}_err', np.nan))
    return E


def bin_single_focal(thx_deg, thy_deg, values, n_bins, fp_radius,
                     statistic='median'):
    """Median (or other) of a single per-donut value on the focal grid.

    Same orientation as `bin_median_focal`: bins on (thy, thx).  Returns
    an (n_bins, n_bins) array (all-NaN if no input donuts).
    """
    from scipy.stats import binned_statistic_2d
    edges = np.linspace(-fp_radius, fp_radius, n_bins + 1)
    if len(thx_deg) == 0:
        return np.full((n_bins, n_bins), np.nan)
    stat, _, _, _ = binned_statistic_2d(
        thy_deg, thx_deg, values, statistic=statistic, bins=[edges, edges])
    return stat


def resolve_wfs_inner_edge(param_value=None, fallback_deg=1.59):
    """Inner field-angle radius (deg) of the corner-WFS coverage.

    If `param_value` is given, use it.  Otherwise compute the minimum
    field radius over the SW0 corner sensors from cameraGeom (cached);
    fall back to `fallback_deg` if the LSST stack is unavailable.
    """
    if param_value is not None:
        return float(param_value)
    if 'inner' in _WFS_EDGE_CACHE:
        return _WFS_EDGE_CACHE['inner']
    try:
        from lsst.obs.lsst import LsstCam
        from lsst.ts.intrinsic.wavefront.ccd_height import wfs_field_radius_range
        r_min, r_max = wfs_field_radius_range(LsstCam.getCamera())
        _WFS_EDGE_CACHE['inner'] = r_min
        print(f'WFS field coverage from cameraGeom: '
              f'inner edge {r_min:.4f}°, outer edge {r_max:.4f}°')
        return r_min
    except Exception as e:
        print(f'(WFS inner edge: cameraGeom unavailable '
              f'[{type(e).__name__}]; using fallback {fallback_deg}°)')
        _WFS_EDGE_CACHE['inner'] = fallback_deg
        return fallback_deg


def _pad_zk_to_z4_block(values_2d, iZs):
    """Pad a (..., n_zk_kept) zk array to (..., j_max - 3) starting at Z4.

    Missing Noll indices (e.g. j=20, j=21) are filled with zeros so that
    convertZernikesToPsfWidth, which expects a contiguous Z4..Zj_max
    block, sees the right indices.
    """
    j_max = int(max(iZs))
    n_pad = j_max - 4 + 1
    arr = np.asarray(values_2d, dtype=float)
    if arr.ndim == 1:
        out = np.zeros(n_pad, dtype=float)
        for ji, j in enumerate(iZs):
            out[int(j) - 4] = arr[ji]
        return out
    out = np.zeros(arr.shape[:-1] + (n_pad,), dtype=float)
    for ji, j in enumerate(iZs):
        out[..., int(j) - 4] = arr[..., ji]
    return out


def donut_fwhm_from_zk(zk_2d, iZs):
    """Per-donut total FWHM-equivalent (arcsec) from a (n_donuts, n_zk) array."""
    if not _wep_ok:
        return np.full(len(zk_2d), np.nan)
    padded = _pad_zk_to_z4_block(zk_2d, iZs)
    psf_widths = np.asarray(convertZernikesToPsfWidth(padded))
    return np.sqrt(np.sum(psf_widths ** 2, axis=-1))


def sigma_to_fwhm(sigma_per_j, iZs):
    """Scalar FWHM-equivalent (arcsec) from a per-j sigma array (μm)."""
    if not _wep_ok:
        return np.nan
    full = _pad_zk_to_z4_block(np.atleast_1d(sigma_per_j), iZs)
    psf = np.asarray(convertZernikesToPsfWidth(full[None, :]))[0]
    return float(np.sqrt(np.sum(psf ** 2)))


def common_color_scale(grids, plo=5.0, phi=95.0):
    """Return (vmin, vmax) covering the plo-phi percentile of *all* grids
    (skip NaNs; ignore empty grids)."""
    pooled = np.concatenate(
        [g.ravel()[~np.isnan(g.ravel())] for g in grids if g is not None
         and np.any(np.isfinite(g))])
    if pooled.size == 0:
        return -1.0, 1.0
    return tuple(np.nanpercentile(pooled, [plo, phi]))


def _build_iter_grids_list(result_obj):
    """Return ``[('Iter-1 measured', grid_j_dict), ...]`` in order."""
    return [(f'Iter-{i + 1} measured', it['measured_grid'])
            for i, it in enumerate(result_obj['iter_results'])]


def _build_ccs_grid(donut_df, result, iZidx, n_bins, fp_radius):
    """Bin the iter-final wfd_subtracted on a (thx_CCS, thy_CCS) grid.

    Excludes donuts from bad-flagged visits (uses the
    `good_donut_mask` saved by build_measured_intrinsic).
    """
    if 'thx_CCS' not in donut_df.columns or 'thy_CCS' not in donut_df.columns:
        raise KeyError(
            "donut parquet has no thx_CCS / thy_CCS columns — cannot make "
            "the CCS-binned map.  Re-run mktable to add them.")
    final_iter = result['iter_results'][-1]
    wfd_sub = final_iter['wfd_subtracted']
    gd_mask = final_iter.get(
        'good_donut_mask',
        np.ones(len(donut_df), dtype=bool))
    thx = np.rad2deg(np.asarray(donut_df['thx_CCS'], dtype=float))
    thy = np.rad2deg(np.asarray(donut_df['thy_CCS'], dtype=float))
    return bin_median_focal(
        thx[gd_mask], thy[gd_mask],
        wfd_sub[gd_mask], iZidx,
        n_bins=n_bins, fp_radius=fp_radius)


def compute_per_visit_sigmas(donut_df, residuals, fit_rows,
                             j_list, iZidx):
    """For each good visit, return per-j sigma_std and sigma_MAD arrays.

    Returns a dict with keys 'visit_ordinal', 'day_obs', 'seq_num',
    'sigma' (n_visits, n_j), 'sigma_mad' (n_visits, n_j), and the same
    pooled across j as 'sigma_pool', 'sigma_mad_pool'.
    """
    dobs_arr = np.asarray(donut_df['day_obs'])
    snum_arr = np.asarray(donut_df['seq_num'])
    good = [r for r in fit_rows if not r.get('bad_fit', False)]
    n = len(good)
    n_j = len(j_list)
    sig = np.full((n, n_j), np.nan)
    smd = np.full((n, n_j), np.nan)
    pool = np.full(n, np.nan)
    pool_m = np.full(n, np.nan)
    dobs_out = np.zeros(n, dtype=int)
    snum_out = np.zeros(n, dtype=int)
    for v, r in enumerate(good):
        d = int(r['day_obs']); s = int(r['seq_num'])
        dobs_out[v] = d
        snum_out[v] = s
        mask = (dobs_arr == d) & (snum_arr == s)
        if not np.any(mask):
            continue
        for ji, j in enumerate(j_list):
            res = residuals[mask][:, iZidx[j]]
            res = res[np.isfinite(res)]
            if len(res) < 5:
                continue
            sig[v, ji] = float(np.std(res))
            mad = float(np.median(np.abs(res - np.median(res))))
            smd[v, ji] = 1.4826 * mad
        # pooled = sqrt(SUM σ_j^2) over j with finite σ (quadrature sum)
        sj = sig[v]
        sj = sj[np.isfinite(sj)]
        if sj.size:
            pool[v] = float(np.sqrt(np.sum(sj ** 2)))
        mj = smd[v]
        mj = mj[np.isfinite(mj)]
        if mj.size:
            pool_m[v] = float(np.sqrt(np.sum(mj ** 2)))
    return dict(visit_ordinal=np.arange(n),
                day_obs=dobs_out, seq_num=snum_out,
                sigma=sig, sigma_mad=smd,
                sigma_pool=pool, sigma_mad_pool=pool_m)


def example_visit_index(sigmas):
    """Pick the visit with median pooled σ.  Falls back to first visit."""
    pool = sigmas['sigma_pool']
    good = np.where(np.isfinite(pool))[0]
    if good.size == 0:
        return 0
    return int(good[np.argsort(pool[good])[len(good) // 2]])


def compute_validation_residual(donut_df, result, coord_sys, iZs):
    """Per-donut residual after subtracting iter-final intrinsic AND
    iter-final DZ fit."""
    final_iter = result['iter_results'][-1]
    wfd_sub = final_iter['wfd_subtracted']
    measured_grid = final_iter['measured_grid']
    xcent, ycent = result['xcent'], result['ycent']
    thx = np.rad2deg(np.asarray(donut_df[f'thx_{coord_sys}'], dtype=float))
    thy = np.rad2deg(np.asarray(donut_df[f'thy_{coord_sys}'], dtype=float))
    intrinsic_at_donut = interpolate_grid_at_donuts(
        measured_grid, xcent, ycent, thx, thy, iZs)
    return wfd_sub - intrinsic_at_donut


def augment_fit_rows_with_modes(fit_rows, A_modes, V_modes, dof,
                                dof_labels, median_fwhm_by_visit,
                                V_fwhm=None, W_resid=None, kj_list=None,
                                W_raw=None, W_corr=None):
    """Add per-visit u-mode / v-mode / DOF / median-FWHM columns to each
    fit-row dict (modified in place).

    Row ``v`` of ``A_modes`` / ``V_modes`` / ``dof`` must correspond to
    ``fit_rows[v]`` (same visit order — guaranteed because all three are
    built by iterating ``fit_rows``).

    Columns added (flattened so the parquet is directly queryable):
      umode_1 .. umode_{n_keep}     u-mode amplitudes  a_i = u_iᵀ w   (μm)
      vmode_1 .. vmode_{n_keep}     v-mode amplitudes  c_i = a_i / σ_i
      dof_{name}                    physical DOF per dof_labels
      dz_z{j}_c{k} (+_err)          RAW DZ fit to the donut Zernikes
                                    (w_raw; overwritten from W_raw)
      dz_corr_z{j}_c{k}             correctable portion = n_keep v-mode
                                    projection (w_fit), if W_corr given
      dz_resid_z{j}_c{k}            residual w_raw - w_fit (uncorrectable),
                                    if W_resid given
      median_fwhm_arcsec            median per-donut FWHM-equivalent
    """
    n_modes = A_modes.shape[1]
    for v, row in enumerate(fit_rows):
        for m in range(n_modes):
            row[f'umode_{m + 1}'] = float(A_modes[v, m])
            row[f'vmode_{m + 1}'] = float(V_modes[v, m])
            if V_fwhm is not None:
                row[f'vmode_fwhm_{m + 1}'] = float(V_fwhm[v, m])
        for di, name in enumerate(dof_labels):
            row[f'dof_{name}'] = float(dof[v, di])
        if kj_list is not None:
            for m, (k, j) in enumerate(kj_list):
                if W_corr is not None:
                    # correctable portion = n_keep v-mode projection (w_fit)
                    row[f'dz_corr_z{j}_c{k}'] = float(W_corr[v, m])
                if W_resid is not None:
                    # residual = raw fit − correctable
                    row[f'dz_resid_z{j}_c{k}'] = float(W_resid[v, m])
                if W_raw is not None:
                    # OVERWRITE dz_z{j}_c{k} so it holds the RAW fit to the
                    # donut measured Zernikes (the _err stays = raw-fit error).
                    row[f'dz_z{j}_c{k}'] = float(W_raw[v, m])
        key = (int(row['day_obs']), int(row['seq_num']))
        row['median_fwhm_arcsec'] = float(
            median_fwhm_by_visit.get(key, np.nan))
    return fit_rows


def plot_w_heatmap(W, kj_list, n_k_for_lines, n_j_for_lines,
                   title, vmax=None, k_min=None):
    """Heatmap of (visit ordinal, kj-row) for a coefficient stack.

    `n_k_for_lines`, `n_j_for_lines`, and `k_min` are used to draw
    k-boundary lines when the kj_list happens to be the contiguous
    n_k × n_j grid; otherwise the dividers are skipped.
    """
    n_v, n_m = W.shape
    if vmax is None:
        finite = W[np.isfinite(W)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1e-3
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-3
    fig, ax = plt.subplots(figsize=(11.5, 7.0), layout='constrained')
    im = ax.imshow(W.T, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=+vmax,
                   extent=[-0.5, n_v - 0.5, n_m - 0.5, -0.5])
    ax.set_xlabel('Visit ordinal')
    ax.set_ylabel('(k, j) row')
    ax.set_title(title)
    # y tick labels per cell (small)
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels([f'({k},{j})' for k, j in kj_list], fontsize=6)
    plt.colorbar(im, ax=ax, label='μm')
    return fig


def plot_per_kj_vs_visit_page(W, kj_list, iZs_arr,
                              k_min, k_max,
                              title_root='', iter_label=''):
    """Build a 22-panel page (DZ-fit vs visit ordinal) for ONE iteration.

    Layout:
      panel 0     — (k=1, j=4) Focus alone (1 line)
      panel 1     — j=4, k=2..6 overlaid
      panels 2..21 — each remaining j in `iZs_arr`, k=1..6 overlaid

    A k=1..6 legend is drawn beneath the grid.
    """
    js_rest = [int(j) for j in iZs_arr if int(j) != 4]
    ncols = 4
    nrows = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 14),
                             layout='constrained', sharex=True)
    axes = axes.ravel()
    cmap_k = plt.get_cmap('viridis')

    def _kj_col(k_val, j_val):
        try:
            return kj_list.index((int(k_val), int(j_val)))
        except ValueError:
            return None

    n_v = W.shape[0]
    # Panel 0: (k=1, j=4) alone
    ax = axes[0]
    col = _kj_col(1, 4)
    if col is not None:
        ax.plot(np.arange(n_v), W[:, col], '.', ms=4, color=cmap_k(0.0))
    ax.set_title('(k=1, j=4) — Focus', fontsize=9)
    ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)

    # Panel 1: (k=2..6, j=4)
    ax = axes[1]
    for k_val in range(2, k_max + 1):
        col = _kj_col(k_val, 4)
        if col is None:
            continue
        c = cmap_k((k_val - k_min) / max(1, k_max - k_min))
        ax.plot(np.arange(n_v), W[:, col], '.', ms=3, color=c)
    ax.set_title('j=4, k=2..6', fontsize=9)
    ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)

    # Panels 2..21: each remaining j with k = 1..6
    for pidx, j_val in enumerate(js_rest, start=2):
        if pidx >= len(axes):
            break
        ax = axes[pidx]
        for ki, k_val in enumerate(range(k_min, k_max + 1)):
            col = _kj_col(k_val, j_val)
            if col is None:
                continue
            c = cmap_k(ki / max(1, k_max - k_min))
            ax.plot(np.arange(n_v), W[:, col], '.', ms=2.5, color=c)
        ax.set_title(f'j={int(j_val)}, k=1..6', fontsize=8)
        ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)

    for k in range(2 + len(js_rest), len(axes)):
        axes[k].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='', ms=6,
                       color=cmap_k(ki / max(1, k_max - k_min)),
                       label=f'k={k_min + ki}')
               for ki in range(k_max - k_min + 1)]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=k_max - k_min + 1,
               fontsize=9, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
    tag = f'  —  {iter_label}' if iter_label else ''
    fig.suptitle(f'{title_root}{tag}', fontsize=12)
    return fig


def plot_iter_stability_heatmap(W_prev, W_last, kj_list,
                                k_min, k_max, iZs_arr, title):
    """Heatmap of RMS(iter_last - iter_prev) per (k, j) cell."""
    n_k = k_max - k_min + 1
    n_j = len(iZs_arr)
    grid = np.full((n_k, n_j), np.nan)
    for m, (k, j) in enumerate(kj_list):
        diff = W_last[:, m] - W_prev[:, m]
        good = diff[np.isfinite(diff)]
        if len(good) < 3:
            continue
        ki = int(k) - k_min
        ji = int(np.where(iZs_arr == int(j))[0][0])
        grid[ki, ji] = float(np.sqrt(np.mean(good ** 2)))
    fig, ax = plt.subplots(figsize=(0.45 * n_j + 2.5, 0.6 * n_k + 1.8),
                           layout='constrained')
    vmax = float(np.nanmax(grid)) if np.any(np.isfinite(grid)) else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-3
    im = ax.imshow(grid, aspect='auto', cmap='magma_r',
                   vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n_j))
    ax.set_xticklabels([f'Z{j}' for j in iZs_arr], fontsize=8)
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([f'k={k_min + ki}' for ki in range(n_k)])
    ax.set_xlabel('pupil Noll j')
    ax.set_ylabel('focal Noll k')
    ax.set_title(title)
    for ki in range(n_k):
        for ji in range(n_j):
            v = grid[ki, ji]
            if not np.isfinite(v):
                continue
            ax.text(ji, ki, f'{v:.3f}', ha='center', va='center',
                    fontsize=6,
                    color='white' if v > 0.55 * vmax else 'black')
    plt.colorbar(im, ax=ax, label='RMS Δ  (μm)')
    return fig


def plot_example_visit_histograms(donut_df, residuals, fit_rows,
                                  j_list, iZidx, example_idx,
                                  hist_range=(-1.0, 1.0), n_bins=60):
    """Single page: residual histograms for one visit, one panel per j."""
    dobs_arr = np.asarray(donut_df['day_obs'])
    snum_arr = np.asarray(donut_df['seq_num'])
    good = [r for r in fit_rows if not r.get('bad_fit', False)]
    if not good:
        return None
    r = good[min(example_idx, len(good) - 1)]
    d = int(r['day_obs']); s = int(r['seq_num'])
    mask = (dobs_arr == d) & (snum_arr == s)

    ncols = 4
    n = len(j_list)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.4 * nrows),
                             layout='constrained')
    axes = np.atleast_2d(axes).ravel()
    for ji, j in enumerate(j_list):
        ax = axes[ji]
        res = residuals[mask][:, iZidx[j]]
        res = res[np.isfinite(res)]
        if len(res) >= 5:
            ax.hist(res, bins=n_bins, range=hist_range,
                    color='steelblue', alpha=0.85)
            sigma = float(np.std(res))
            mad = float(np.median(np.abs(res - np.median(res))))
            ax.set_title(f'Z{j}  σ={sigma:.3f}  σ_MAD={1.4826 * mad:.3f}',
                         fontsize=8)
        else:
            ax.set_title(f'Z{j}  (n<5)', fontsize=8)
        ax.axvline(0, color='k', lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=7)
    for k in range(n, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(f'Per-donut residual histograms  —  '
                 f'visit {d}/{s} (example_idx = {example_idx})',
                 fontsize=11)
    return fig


def plot_sigma_vs_visit_grid(sigmas, j_list, iZidx, visits_table,
                             which='sigma', title_root='σ vs visit'):
    """One page, 4×6 grid (24 panels, 21 used) of σ_j vs visit ordinal.

    Uses the shared intrinsics_lib marker scheme (elev = colour,
    rotator angle = shape, band = edge colour).  Pooled panel uses
    the quadrature-sum pool.
    """
    yvals_all = sigmas[which]
    pool      = (sigmas['sigma_pool']
                 if which == 'sigma' else sigmas['sigma_mad_pool'])
    n_v = yvals_all.shape[0]
    ordinal = sigmas['visit_ordinal']

    # Look up the standard marker style for every visit in the σ array.
    marker_lookup = build_visit_marker_lookup(visits_table)
    styles = []
    for d, s in zip(sigmas['day_obs'], sigmas['seq_num']):
        cls = marker_lookup.get((int(d), int(s)),
                                {'elev': None, 'rot': None, 'band': None})
        styles.append(visit_marker_style(elev=cls['elev'], rot=cls['rot'],
                                         band=cls['band'], base_size=5))

    ncols = 4
    nrows = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 14),
                             layout='constrained', sharex=True)
    axes = axes.ravel()
    # Pooled panel
    ax = axes[0]
    for i in range(n_v):
        y = pool[i]
        if not np.isfinite(y):
            continue
        ax.plot(ordinal[i], y, **styles[i])
    ax.set_title('pooled = sqrt( Σ σ_j² )  (quadrature sum)', fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    for ji, j in enumerate(j_list):
        ax = axes[ji + 1]
        for i in range(n_v):
            y = yvals_all[i, ji]
            if not np.isfinite(y):
                continue
            ax.plot(ordinal[i], y, **styles[i])
        ax.set_title(f'Z{j}', fontsize=8)
        ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    for k in range(len(j_list) + 1, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(title_root, fontsize=12)
    return fig


def plot_dof_vs_visit_pages(dof_array, dof_labels, dof_units,
                            visits_table, dobs, snum,
                            title_root, panels_per_page=25):
    """Two pages of 5×5 DOF panels showing DOF value vs visit ordinal.

    `dof_array` shape: (n_visits, n_dof).  `dobs`/`snum` are the
    per-visit (day_obs, seq_num) aligned 1:1 with the rows of
    `dof_array` (i.e. the fit-row order, ALL visits — good and bad).
    """
    n_v, n_dof = dof_array.shape
    ordinal = np.arange(n_v)
    dobs = np.asarray(dobs); snum = np.asarray(snum)

    marker_lookup = build_visit_marker_lookup(visits_table)
    styles = []
    for vi in range(n_v):
        d = int(dobs[vi]) if vi < len(dobs) else -1
        s = int(snum[vi]) if vi < len(snum) else -1
        cls = marker_lookup.get((d, s),
                                {'elev': None, 'rot': None, 'band': None})
        styles.append(visit_marker_style(elev=cls['elev'], rot=cls['rot'],
                                         band=cls['band'], base_size=5))

    figs = []
    n_pages = (n_dof + panels_per_page - 1) // panels_per_page
    for page in range(n_pages):
        lo = page * panels_per_page
        hi = min(lo + panels_per_page, n_dof)
        fig, axes = plt.subplots(5, 5, figsize=(15, 14),
                                 layout='constrained', sharex=True)
        axes = axes.ravel()
        for slot in range(25):
            dof_i = lo + slot
            ax = axes[slot]
            if dof_i >= hi:
                ax.set_visible(False); continue
            yvals = dof_array[:, dof_i]
            for i in range(n_v):
                y = yvals[i]
                if not np.isfinite(y):
                    continue
                ax.plot(ordinal[i], y, **styles[i])
            ax.set_title(f'{dof_labels[dof_i]}  ({dof_units[dof_i]})',
                         fontsize=8)
            ax.axhline(0, color='k', lw=0.4, alpha=0.5)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        fig.suptitle(f'{title_root}  —  page {page + 1}/{n_pages}',
                     fontsize=12)
        figs.append(fig)
    return figs


def plot_dof_median_summary(dof_per_iter, dof_labels, dof_units,
                            title='DOF median per iteration'):
    """4-panel layout (Hex Translations / Hex Rotations / M1M3 / M2)
    showing the per-visit median DOF for each iteration as separate colours.

    `dof_per_iter` is a dict {iter_idx_0based: (n_visits, n_dof)}.
    """
    n_iter = len(dof_per_iter)
    iters = sorted(dof_per_iter.keys())
    medians = {it: np.nanmedian(dof_per_iter[it], axis=0) for it in iters}

    # DOF index buckets
    hex_trans_idx = [0, 1, 2,  5, 6, 7]               # M2 z/x/y, Cam z/x/y
    hex_rot_idx   = [3, 4,     8, 9]                  # M2 rx/ry, Cam rx/ry
    m1m3_idx      = list(range(10, 30))
    m2_idx        = list(range(30, 50))

    fig, axes = plt.subplots(4, 1, figsize=(15, 14),
                             layout='constrained',
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.5, 1.5]))
    iter_colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, n_iter))

    n_iter_ = len(iters)
    # Spread iterations horizontally inside each DOF bin so points
    # don't overlap.  Total spread <= 0.7 * bin width.
    if n_iter_ > 1:
        offsets = (np.arange(n_iter_) - (n_iter_ - 1) / 2) * (0.7 / n_iter_)
    else:
        offsets = np.array([0.0])

    def _panel(ax, idx_list, title, y_unit):
        x = np.arange(len(idx_list))
        for xi in range(len(idx_list)):
            if xi % 2:
                ax.axvspan(xi - 0.5, xi + 0.5, color='black', alpha=0.05)
        for ci, it in enumerate(iters):
            ax.plot(x + offsets[ci], medians[it][idx_list], 'o',
                    ms=7, color=iter_colors[ci],
                    label=f'iter {it + 1}')
        ax.axhline(0, color='gray', lw=0.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([dof_labels[i] for i in idx_list],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'DOF Value ({y_unit})')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

    _panel(axes[0], hex_trans_idx, 'Hexapod Translations', 'μm')
    _panel(axes[1], hex_rot_idx,   'Hexapod Rotations',   'arcsec')
    _panel(axes[2], m1m3_idx,      'M1M3 Bending Modes',   'μm')
    _panel(axes[3], m2_idx,        'M2 Bending Modes',     'μm')
    axes[0].legend(loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=13)
    return fig


def plot_donut_fwhm_histogram(donut_df, residuals, fit_rows,
                              iZs, example_idx, n_bins=60,
                              hist_range=None, title_root='FWHM histogram'):
    """Single-panel histogram of per-donut FWHM (arcsec) for one visit.

    Annotates median, σ (std), and σ_MAD on the figure.
    """
    dobs_arr = np.asarray(donut_df['day_obs'])
    snum_arr = np.asarray(donut_df['seq_num'])
    good = [r for r in fit_rows if not r.get('bad_fit', False)]
    if not good:
        return None
    r = good[min(example_idx, len(good) - 1)]
    d = int(r['day_obs']); s = int(r['seq_num'])
    mask = (dobs_arr == d) & (snum_arr == s)
    res = residuals[mask]
    res = res[np.all(np.isfinite(res), axis=1)]
    fwhms = donut_fwhm_from_zk(res, iZs)
    fwhms = fwhms[np.isfinite(fwhms)]
    if len(fwhms) < 5:
        return None
    med = float(np.median(fwhms))
    sigma = float(np.std(fwhms))
    mad = float(np.median(np.abs(fwhms - med)))
    sigma_mad = 1.4826 * mad

    if hist_range is None:
        hi = float(np.nanpercentile(fwhms, 99.0)) * 1.1
        hist_range = (0.0, max(0.05, hi))
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.hist(fwhms, bins=n_bins, range=hist_range,
            color='steelblue', alpha=0.85, edgecolor='white')
    ax.axvline(med, color='k', lw=1.2, label=f'median = {med:.3f}″')
    ax.set_xlabel('per-donut FWHM-equivalent  (arcsec)')
    ax.set_ylabel('donut count')
    ax.set_title(f'{title_root}  —  visit {d}/{s}  '
                 f'(n = {len(fwhms)})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def plot_fwhm_vs_visit(fwhm_per_donut, dobs_per_donut, snum_per_donut,
                       sigmas_for_visit_order, visits_table, title_root):
    """Per-visit median per-donut FWHM-equivalent vs visit ordinal.

    For each visit `v` we plot
        median(per-donut FWHM over donuts in v)
    with ± σ_MAD error bars (1.4826 × MAD over the visit's donuts).
    Markers follow the standard intrinsics_lib elev / rot / band scheme.

    `fwhm_per_donut`, `dobs_per_donut`, `snum_per_donut` are 1-D arrays
    over the same set of donuts (already filtered to good visits).
    `sigmas_for_visit_order` provides the canonical visit ordering
    (its `day_obs`, `seq_num`, `visit_ordinal` arrays).
    """
    f = np.asarray(fwhm_per_donut, dtype=float)
    d = np.asarray(dobs_per_donut).astype(int)
    s = np.asarray(snum_per_donut).astype(int)

    # Per-visit aggregation
    n_v = len(sigmas_for_visit_order['day_obs'])
    med   = np.full(n_v, np.nan)
    sigma_mad = np.full(n_v, np.nan)
    for v in range(n_v):
        dv = int(sigmas_for_visit_order['day_obs'][v])
        sv = int(sigmas_for_visit_order['seq_num'][v])
        mask = (d == dv) & (s == sv)
        vals = f[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        med[v]       = float(np.median(vals))
        sigma_mad[v] = 1.4826 * float(np.median(np.abs(vals - med[v])))

    # Standard marker styles per visit
    marker_lookup = build_visit_marker_lookup(visits_table)
    styles = []
    for dv, sv in zip(sigmas_for_visit_order['day_obs'],
                      sigmas_for_visit_order['seq_num']):
        cls = marker_lookup.get((int(dv), int(sv)),
                                {'elev': None, 'rot': None, 'band': None})
        styles.append(visit_marker_style(elev=cls['elev'], rot=cls['rot'],
                                         band=cls['band'], base_size=6))

    fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
    ordinal = sigmas_for_visit_order['visit_ordinal']
    for v in range(n_v):
        if not np.isfinite(med[v]):
            continue
        # Vertical ± σ_MAD line — colour follows the marker style.
        col = styles[v].get('color', 'gray')
        ax.errorbar(ordinal[v], med[v], yerr=sigma_mad[v],
                    fmt='none', ecolor=col, elinewidth=0.8,
                    capsize=2, alpha=0.7)
        ax.plot(ordinal[v], med[v], **styles[v])

    ax.set_xlabel('Visit ordinal')
    ax.set_ylabel('median per-donut FWHM-equivalent  (arcsec)')
    ax.set_title(title_root, fontsize=12)
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    return fig


def plot_fwhm_focal_plane_map(thx_deg, thy_deg, fwhm_per_donut,
                              n_bins=49, fp_radius=1.8,
                              title='Focal-plane median FWHM-equivalent',
                              vmin_pct=5.0, vmax_pct=95.0):
    """Median per-donut FWHM-equivalent on a 2D focal-plane grid.

    Colour range defaults to the [`vmin_pct`, `vmax_pct`] percentile of
    the binned-median map (5-95 % by default) to suppress outlier bins
    with very few donuts.
    """
    from scipy.stats import binned_statistic_2d
    edges = np.linspace(-fp_radius, fp_radius, n_bins + 1)
    good = np.isfinite(fwhm_per_donut)
    stat, xe, ye, _ = binned_statistic_2d(
        thy_deg[good], thx_deg[good], fwhm_per_donut[good],
        statistic='median', bins=[edges, edges])
    finite = stat[np.isfinite(stat)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [vmin_pct, vmax_pct])
    else:
        vmin, vmax = None, None
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    im = ax.imshow(stat.T, origin='lower', cmap='viridis',
                   extent=[edges[0], edges[-1], edges[0], edges[-1]],
                   aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xlabel('thy [deg]')
    ax.set_ylabel('thx [deg]')
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax,
                      label=f'median FWHM  (arcsec, '
                            f'{int(vmin_pct)}-{int(vmax_pct)}% range)')
    return fig


def plot_fwhm_pooled_histogram(fwhm_per_donut, title_root,
                               n_bins=60, hist_range=None):
    """Single-panel histogram of per-donut FWHM-equivalent (arcsec).

    One entry per donut, pooled across every good donut in every good
    visit.  Annotated with median, std (σ), and 1.4826 × MAD (σ_MAD)
    of the distribution — for comparison against published donut /
    AOS WFE results.
    """
    f = np.asarray(fwhm_per_donut, dtype=float)
    f = f[np.isfinite(f)]
    if len(f) < 5:
        return None
    med = float(np.median(f))
    std = float(np.std(f))
    mad = float(np.median(np.abs(f - med)))
    sigma_mad = 1.4826 * mad
    if hist_range is None:
        hi = float(np.nanpercentile(f, 99.0)) * 1.1
        hist_range = (0.0, max(0.05, hi))
    fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')
    ax.hist(f, bins=n_bins, range=hist_range,
            color='steelblue', alpha=0.85, edgecolor='white')
    ax.axvline(med, color='k', lw=1.4,
               label=f'median = {med:.3f}″')
    ax.set_xlabel('per-donut FWHM-equivalent  (arcsec)')
    ax.set_ylabel('donut count')
    ax.set_title(f'{title_root}   (n = {len(f)} donuts)', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    return fig


def plot_intrinsic_vs_azimuth(thx_deg, thy_deg, intrinsic_2d, iZidx, iZs,
                              inner_edge_deg, outer_edge_deg=1.725,
                              n_radial_bins=3, azimuth_bin_deg=5.0,
                              title_root='', ncols=4):
    """Per-pupil-j measured-intrinsic median vs focal-plane azimuth.

    Azimuth = atan2(thy, thx) mapped to [0, 360).  One panel per pupil
    j; within a panel, one line per radial shell (equal bins from
    `inner_edge_deg` to `outer_edge_deg`).  A k=... style legend keyed
    by radial shell is drawn on the first panel.
    """
    r = np.hypot(thx_deg, thy_deg)
    az = np.degrees(np.arctan2(thy_deg, thx_deg)) % 360.0
    redges = np.linspace(inner_edge_deg, outer_edge_deg, n_radial_bins + 1)
    n_azimuth_bins = int(round(360.0 / float(azimuth_bin_deg)))
    azedges = np.linspace(0.0, 360.0, n_azimuth_bins + 1)
    azcent = 0.5 * (azedges[:-1] + azedges[1:])

    js = [int(j) for j in iZs]
    nrows = (len(js) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows),
                             layout='constrained', sharex=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap('viridis')
    shell_colors = [cmap(t) for t in np.linspace(0.1, 0.9, n_radial_bins)]

    from scipy.stats import binned_statistic
    for pidx, j in enumerate(js):
        ax = axes[pidx]
        col = iZidx[j]
        for b in range(n_radial_bins):
            m = (r >= redges[b]) & (r < redges[b + 1]) & np.isfinite(intrinsic_2d[:, col])
            if int(m.sum()) < 5:
                continue
            med, _, _ = binned_statistic(
                az[m], intrinsic_2d[m, col], statistic='median', bins=azedges)
            ax.plot(azcent, med, '.-', ms=3, lw=0.8,
                    color=shell_colors[b],
                    label=(f'{redges[b]:.3f}–{redges[b + 1]:.3f}°'
                           if pidx == 0 else None))
        ax.axhline(0, color='k', lw=0.4, alpha=0.5)
        ax.set_title(f'Z{j}', fontsize=9)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    for k in range(len(js), len(axes)):
        axes[k].set_visible(False)
    for ax in axes[max(0, len(js) - ncols):len(js)]:
        ax.set_xlabel('focal-plane azimuth [deg]', fontsize=8)
    if n_radial_bins:
        axes[0].legend(fontsize=6, loc='best', title='radial shell',
                       title_fontsize=6)
    fig.suptitle(f'{title_root}   '
                 f'(radial shells {inner_edge_deg:.3f}–{outer_edge_deg:.3f}°, '
                 f'{n_radial_bins} bins)', fontsize=12)
    return fig


def plot_z4_optical_page(z4_meas_ocs, z4_height_ccs, z4_optical_ocs,
                         xbins, ybins, plo=5.0, phi=95.0):
    """3-panel Z4 page: measured intrinsic (OCS), the CCD-height
    contribution as used in CCS (≈ the raw CCD height map in Z4 units),
    and Z4 optical = per-donut (measured − height) binned in OCS.
    Viridis, per-panel 5-95 % color scaling."""
    panels = [
        ('Z4 measured intrinsic (OCS)', z4_meas_ocs),
        ('Z4 height contribution (CCS)', z4_height_ccs),
        ('Z4 optical = measured − height (OCS)', z4_optical_ocs),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), layout='constrained')
    for ax, (title, g) in zip(axes, panels):
        if g is None or not np.any(np.isfinite(g)):
            ax.set_visible(False); continue
        finite = g[np.isfinite(g)]
        vmin, vmax = np.nanpercentile(finite, [plo, phi])
        im = ax.imshow(g.T, origin='lower',
                       extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                       cmap='viridis', interpolation='none', aspect='equal',
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label='μm')
        ax.set_xlabel('thy [deg]', fontsize=9)
        ax.set_ylabel('thx [deg]', fontsize=9)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Path A — Z4 intrinsic vs focal-plane height '
                 '(final iteration)', fontsize=12)
    return fig


def zk_cov_corr(resid_2d, iZs):
    """Covariance (μm²) + correlation of the per-donut residual Zernikes.

    `resid_2d` is (n_donut, n_zk); rows with any non-finite value are dropped.
    Returns ``(cov, corr, n, labels)`` — ``(None, None, n, labels)`` if fewer
    than 10 clean donuts.  Shared by :func:`plot_zk_cov_corr` and the build
    step's saved ``intrinsic_cov_edge.parquet`` so the plotted and written
    matrices are identical."""
    R = np.asarray(resid_2d, dtype=float)
    R = R[np.all(np.isfinite(R), axis=1)]
    labs = [f'Z{int(j)}' for j in iZs]
    if R.shape[0] < 10:
        return None, None, int(R.shape[0]), labs
    cov = np.cov(R, rowvar=False)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(sd, sd)
    return cov, corr, int(R.shape[0]), labs


def plot_zk_cov_corr(resid_2d, iZs, title_root):
    """Covariance + correlation color maps of the per-donut residual
    Zernikes (residual after the nDOF / n_keep adjustment).

    `resid_2d` is (n_donut, n_zk); rows with any non-finite value are
    dropped.  Left panel = covariance (μm²), right = correlation [-1, 1].
    """
    cov, corr, n, labs = zk_cov_corr(resid_2d, iZs)
    if cov is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    vmax = float(np.nanpercentile(np.abs(cov), 99)) or 1e-6
    im0 = axes[0].imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Covariance (μm²)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label='μm²')
    im1 = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Correlation')
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label='r')
    for ax in axes:
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=6)
    fig.suptitle(f'{title_root}   (n = {n} donuts)', fontsize=12)
    return fig


def _plot_modes_heatmap(A, title, clabel='Uᵀ w_raw (μm)',
                        ylabel='U-mode index (1-based)'):
    n_v, n_keep_ = A.shape
    finite = A[np.isfinite(A)]
    vmax = (float(np.nanpercentile(np.abs(finite), 98))
            if finite.size else 1e-3)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-3
    fig, ax = plt.subplots(figsize=(11.5, 5.0), layout='constrained')
    im = ax.imshow(A.T, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=+vmax,
                   extent=[-0.5, n_v - 0.5, n_keep_ - 0.5, -0.5])
    ax.set_xlabel('Visit ordinal')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    yt = np.arange(0, n_keep_, max(1, n_keep_ // 12))
    ax.set_yticks(yt)
    ax.set_yticklabels([str(int(p + 1)) for p in yt])
    plt.colorbar(im, ax=ax, label=clabel)
    return fig


def _plot_modes_lines_all(A, n_keep_, title):
    n_v = A.shape[0]
    ncols = 5
    nrows = (n_keep_ + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 1.7 * nrows),
                             layout='constrained', sharex=True)
    axes = np.atleast_2d(axes).ravel()
    cmap = plt.get_cmap('tab20')
    for mi in range(nrows * ncols):
        ax = axes[mi]
        if mi >= n_keep_:
            ax.axis('off'); continue
        ax.plot(np.arange(n_v), A[:, mi], '.-',
                ms=2, lw=0.6, color=cmap(mi % 20))
        ax.axhline(0, color='k', lw=0.4, alpha=0.5)
        ax.set_title(f'mode {mi + 1}', fontsize=7)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=11)
    return fig


def plot_iter_progression_for_j(j, original_grid, iter_grids,
                                tabulated_grid, xbins, ybins,
                                coord_sys, plo=5.0, phi=95.0,
                                path_tag=''):
    """Per-pupil-j N-panel page: Original | Iter 1 | ... | Iter n | Tabulated.

    `iter_grids` is a list of ``(label, grid)`` tuples, one per iteration
    in order.  Panel 1 (Original) uses its own 5-95 % range; all other
    panels share a single pooled 5-95 % range across the iteration
    grids and the Tabulated grid so they are directly comparable.
    """
    panels = [('Original median (raw zk)', original_grid)]
    panels.extend(iter_grids)
    panels.append(('Tabulated intrinsic', tabulated_grid))

    if original_grid is not None and np.any(np.isfinite(original_grid)):
        vmin0, vmax0 = np.nanpercentile(original_grid, [plo, phi])
    else:
        vmin0, vmax0 = -1.0, 1.0
    shared = [g for _, g in iter_grids] + [tabulated_grid]
    vmin_i, vmax_i = common_color_scale(shared, plo=plo, phi=phi)

    n_panels = len(panels)
    ncols = min(n_panels, 3) if n_panels > 4 else 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows),
                             layout='constrained')
    axes = np.asarray(axes).reshape(nrows, ncols).ravel()
    for idx, ax in enumerate(axes):
        if idx >= n_panels:
            ax.set_visible(False)
            continue
        title, grid = panels[idx]
        if grid is None or not np.any(np.isfinite(grid)):
            ax.set_visible(False)
            continue
        if idx == 0:
            vmin, vmax = vmin0, vmax0
            scale_note = f'5-95% own = [{vmin0:.3f}, {vmax0:.3f}] μm'
        else:
            vmin, vmax = vmin_i, vmax_i
            scale_note = f'shared 5-95% = [{vmin_i:.3f}, {vmax_i:.3f}] μm'
        im = ax.imshow(grid.T, origin='lower',
                       extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                       cmap='viridis', interpolation='none',
                       aspect='equal',
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label='μm')
        ax.set_xlabel(f'thy_{coord_sys} [deg]')
        ax.set_ylabel(f'thx_{coord_sys} [deg]')
        ax.set_title(f'{title}\n({scale_note})', fontsize=10)
    suptitle = (f'Pupil Z{j}  ({coord_sys}){path_tag}  '
                f'— Original uses own 5-95 %; iterations + Tabulated share '
                f'a single 5-95 %')
    fig.suptitle(suptitle, fontsize=12)
    return fig


def render_intrinsic_panel_pdf(grid_dict, xbins, ybins, iZs,
                                output_pdf_path, frame_label,
                                page_subtitle,
                                panels_per_page=4, ncols=2,
                                show_first=True):
    """Stream a multi-page PDF of pupil-j panels for an iter-final grid.

    Each page has up to `panels_per_page` panels in a `ncols`-wide grid;
    each panel uses its own 5-95 percentile color scale.  Pages are
    written one at a time and the figure is closed after savefig so
    memory stays bounded.

    Parameters
    ----------
    grid_dict : dict {pupil_j: 2D array}
    xbins, ybins : 1D arrays of bin edges
    iZs : list of pupil_j
    output_pdf_path : str / Path
    frame_label : 'OCS' or 'CCS' — used in axis labels
    page_subtitle : str — used in the per-page suptitle
    """
    n_total_pages = int(np.ceil(len(iZs) / max(1, int(panels_per_page))))
    n_pages = 0
    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf_path) as pdf:
        for start in range(0, len(iZs), panels_per_page):
            page_js = iZs[start:start + panels_per_page]
            nrows = (len(page_js) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(7.0 * ncols, 6.0 * nrows),
                layout='constrained')
            axes = np.atleast_1d(axes).ravel()
            for idx, j in enumerate(page_js):
                ax = axes[idx]
                grid = grid_dict.get(j)
                if grid is None or not np.any(np.isfinite(grid)):
                    ax.set_visible(False)
                    continue
                vmin, vmax = np.nanpercentile(grid, [5, 95])
                im = ax.imshow(
                    grid.T, origin='lower',
                    extent=[xbins[0], xbins[-1],
                            ybins[0], ybins[-1]],
                    cmap='viridis', interpolation='none', aspect='equal',
                    vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, shrink=0.8, label='μm')
                ax.set_xlabel(f'thy_{frame_label} [deg]')
                ax.set_ylabel(f'thx_{frame_label} [deg]')
                pname = PUPIL_NAMES.get(j, f'Z{j}')
                ax.set_title(
                    f'Z{j} ({pname})  —  iter-final measured intrinsic\n'
                    f'5-95% = [{vmin:+.3f}, {vmax:+.3f}] μm',
                    fontsize=11)
            for idx in range(len(page_js), len(axes)):
                axes[idx].set_visible(False)
            page_idx = start // panels_per_page + 1
            fig.suptitle(
                f'{page_subtitle}  ({page_idx}/{n_total_pages})',
                fontsize=12)
            if show_first and n_pages == 0:
                plt.show()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            n_pages += 1
    print(f'Wrote {n_pages} pages -> {output_pdf_path}')
    return n_pages
