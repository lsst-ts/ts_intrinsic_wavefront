"""Library functions for building intrinsic Zernike wavefront tables.

Extracts aggregate Zernike data from the LSST Butler, computes intrinsic
wavefront models, retrieves rotator angles and thermal data, and saves
the results as HDF5 files.

Usage from notebook:
    from lsst.ts.intrinsic.wavefront.intrinsics_lib import run_mktable, PARAM_SETS
    params = PARAM_SETS['fam_danish_triplets']
    aosTable, visit_info = await run_mktable(**params, coord_sys='OCS')

Usage from CLI:
    python run_mktable.py --param-set fam_danish_triplets
"""

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import LinearNDInterpolator
from astropy.table import QTable, vstack
from astropy.time import Time, TimeDelta
from tqdm import tqdm

# LSST imports (available on RSP)
from lsst.daf.butler import Butler, DatasetNotFoundError
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.task.estimateZernikesDanishTask import EstimateZernikesDanishTask
from lsst.ts.wep.task import CalcZernikesTask, CalcZernikesTaskConfig
from lsst.ts.wep.utils import getTaskInstrument, BandLabel
from lsst.summit.utils import ConsDbClient
from lsst.summit.utils.efdUtils import makeEfdClient

# Local utilities
from lsst.ts.intrinsic.wavefront.wcsutils import calc_rotator_from_visitinfo, pixel_to_focal

# Optional: M1M3 thermal analysis (may not be available in all environments)
try:
    from lsst.ts.m1m3.utils import ThermocoupleAnalysis
    HAS_M1M3_UTILS = True
except ImportError:
    HAS_M1M3_UTILS = False


async def _close_efd_client(efd_client):
    """Close an EfdClient's underlying aioinflux/aiohttp session so the run
    doesn't end with 'Unclosed client session' warnings.  Best-effort and
    version-tolerant (tries the aioinflux client's close(), then the raw
    aiohttp session's close())."""
    import inspect
    ic = getattr(efd_client, 'influx_client', None)
    closers = []
    if ic is not None:
        closers.append(getattr(ic, 'close', None))
        sess = getattr(ic, 'session', None)
        if sess is not None:
            closers.append(getattr(sess, 'close', None))
    for fn in closers:
        if fn is None:
            continue
        try:
            res = fn()
            if inspect.isawaitable(res):
                await res
            return
        except Exception:
            continue


# ============================================================
# Parameter Sets (loaded from param_sets.yaml)
# ============================================================

def load_param_sets(yaml_path=None):
    """Load parameter sets from param_sets.yaml.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Path to param_sets.yaml. Default: aos/param_sets.yaml
        (auto-detected relative to this file or cwd).

    Returns
    -------
    param_sets : dict
        Mapping of name -> dict with butler_repo, fam_collections, etc.
    """
    import yaml
    if yaml_path is None:
        # Try relative to this file first, then cwd
        candidates = [
            Path('param_sets.yaml'),
        ]
        for c in candidates:
            if c.exists():
                yaml_path = c
                break
        else:
            raise FileNotFoundError(
                "param_sets.yaml not found (expected in the working directory).")
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    # Strip 'description' key — not a pipeline parameter
    for name, cfg in data.items():
        cfg.pop('description', None)
    return data


# Load at import time for backward compatibility
try:
    PARAM_SETS = load_param_sets()
except FileNotFoundError:
    PARAM_SETS = {}


# ============================================================
# Defaults
# ============================================================

DEFAULT_ROTATOR_THRESHOLD = 90.0
DEFAULT_FP_RADIUS = 1.8
DEFAULT_FP_NSTEPS = 73
DEFAULT_MIN_VISITS_PER_DAY = 5
DEFAULT_TEMP_TIME_WINDOW_SEC = 0.2
DEFAULT_CONSDB_URL = "http://consdb-pq.consdb:8080/consdb"
DEFAULT_WEP_VER = 'wep_v16_8_0'
DEFAULT_DVIZ_VER = 'dviz_v3_5_0'

# Per-donut intra/extra centroid agreement (arcsec). The boolean
# `matched_intra_extra` column flips True when the offset is below this
# threshold in both axes. Default is None — cut is disabled and the
# column is True for every donut. The numeric offsets are still recorded
# in `intra_extra_offset_*_arcsec` so the cut can be re-applied later.
DEFAULT_MATCHED_THRESHOLD_ARCSEC = None

# ============================================================
# Visit marker scheme (shared by all per-visit AOS plots)
# ============================================================
#
# Color    = elevation bucket (centered on 30, 40, 50, 60, 70 deg ±5°)
# Shape    = chunky arrow pointing in the rotator direction, one of nine
#            buckets (-60, -45, -30, -15, 0, 15, 30, 45, 60 deg)
# Edge     = filter band; i-band uses edge = face (no visible outline);
#            other bands get a distinctive outline color so the few
#            non-i visits are easy to spot

ELEV_CENTERS  = (30, 40, 50, 60, 70)
ELEV_HALFWIDTH = 5
ROT_CENTERS   = (-60, -45, -30, -15, 0, 15, 30, 45, 60)

ELEV_COLORS = {
    30: 'tab:blue', 40: 'tab:green', 50: 'tab:orange',
    60: 'tab:red',  70: 'tab:purple',
}

# Band → marker edge color. None for i-band means "edge = face color"
# (no visible outline); all the other bands get a distinct outline.
BAND_EDGE_COLORS = {
    'u': 'magenta',
    'g': 'lime',
    'r': 'red',
    'i': None,
    'z': 'gold',
    'y': 'black',
}


def _arrow_marker(angle_deg, head_w=1.30, head_h=0.70, shaft_w=0.50):
    """Build a chunky-arrow MarkerStyle whose tip points at angle_deg
    (measured CCW from the +y axis, matching the rotator-angle convention).
    """
    import matplotlib.path as _mpath
    import matplotlib.markers as _mmarkers
    from matplotlib.transforms import Affine2D as _Affine2D
    sw = shaft_w / 2.0
    hw = head_w / 2.0
    verts = [
        (0.0,  1.0),
        ( hw,  1.0 - head_h),
        ( sw,  1.0 - head_h),
        ( sw, -1.0),
        (-sw, -1.0),
        (-sw,  1.0 - head_h),
        (-hw,  1.0 - head_h),
        (0.0,  1.0),
    ]
    codes = [_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * (len(verts) - 1)
    return _mmarkers.MarkerStyle(
        _mpath.Path(verts, codes).transformed(
            _Affine2D().rotate_deg(angle_deg)))


# Built lazily so importing this module doesn't drag in matplotlib up-front.
_ROT_MARKERS_CACHE = None


def _rot_markers():
    global _ROT_MARKERS_CACHE
    if _ROT_MARKERS_CACHE is None:
        _ROT_MARKERS_CACHE = {θ: _arrow_marker(θ) for θ in ROT_CENTERS}
    return _ROT_MARKERS_CACHE


def _alt_to_deg(alt_value):
    """Coerce an alt/elevation value to degrees, auto-detecting radians."""
    a = float(alt_value)
    return np.rad2deg(a) if abs(a) < 2.0 * np.pi + 1e-3 else a


def _classify_elev(alt_deg):
    """Return the nearest elevation bucket center, or None if outside all."""
    if not np.isfinite(alt_deg):
        return None
    for c in ELEV_CENTERS:
        if (c - ELEV_HALFWIDTH) <= alt_deg < (c + ELEV_HALFWIDTH):
            return c
    return None


def _classify_rot(rot_deg):
    """Return the nearest rotator bucket center."""
    if rot_deg is None or not np.isfinite(rot_deg):
        return None
    centers = np.array(ROT_CENTERS, dtype=float)
    return int(centers[int(np.argmin(np.abs(centers - rot_deg)))])


def classify_visit(alt_deg=None, rot_deg=None, band=None):
    """Classify a visit's elev/rot/band into the marker-scheme buckets.

    Parameters
    ----------
    alt_deg : float, optional
        Elevation in degrees (radians OK, auto-detected).
    rot_deg : float, optional
        Rotator angle in degrees.
    band : str, optional
        Filter band; first character is used (e.g. 'i_06' → 'i').

    Returns
    -------
    dict with keys 'elev', 'rot', 'band' — values are the bucket center
    (int) or None for missing/out-of-range, and a one-character band.
    """
    elev = _classify_elev(_alt_to_deg(alt_deg)) if alt_deg is not None else None
    rot = _classify_rot(rot_deg) if rot_deg is not None else None
    if band is None:
        b = None
    else:
        b = str(band).strip().lower()[:1] if str(band).strip() else None
    return {'elev': elev, 'rot': rot, 'band': b}


def visit_marker_style(elev=None, rot=None, band=None,
                       iter_=None, base_size=7,
                       fallback_marker='x', fallback_color='gray'):
    """Translate elev/rot/band buckets into matplotlib plot kwargs.

    iter_ in {None, 1, 2}: when given, controls the fill style — iter 1
    is filled, iter 2 is open. Otherwise the marker is filled.

    Returns a dict suitable for plt.plot(...): marker, color, mfc, mec,
    mew, markersize, linestyle.
    """
    if elev is None or rot is None:
        return dict(marker=fallback_marker, color=fallback_color,
                    markersize=base_size, linestyle='')
    color = ELEV_COLORS[elev]
    marker = _rot_markers()[rot]
    edge = BAND_EDGE_COLORS.get(band, None) if band is not None else None
    if edge is None:
        mec = color
        mew = 0.0
        size = base_size
    else:
        mec = edge
        mew = 1.5
        size = base_size + 1
    if iter_ == 2:
        mfc = 'none'
    else:
        mfc = color
    return dict(marker=marker, color=color, mfc=mfc, mec=mec, mew=mew,
                markersize=size, linestyle='')


def build_visit_marker_lookup(visit_info):
    """Build a (day_obs, seq_num) → classified-visit dict from a visits table.

    Reads `alt`, `rotator_angle`, and `band` columns when present. Useful
    for joining marker styles onto a fit table indexed by (day_obs, seq_num).
    """
    has_alt = 'alt' in visit_info.colnames
    has_rot = 'rotator_angle' in visit_info.colnames
    has_band = 'band' in visit_info.colnames
    out = {}
    for v in visit_info:
        d = int(v['day_obs'])
        s = int(v['seq_num'])
        alt = float(v['alt']) if has_alt else None
        rot = float(v['rotator_angle']) if has_rot else None
        band = str(v['band']) if has_band else None
        out[(d, s)] = classify_visit(alt_deg=alt, rot_deg=rot, band=band)
    return out


def markers_legend_figure(show_iter_distinction=False, show_band_legend=True,
                          figsize=(11, 8.5)):
    """Standalone matplotlib figure documenting the marker scheme.

    show_iter_distinction : add a third legend explaining iter1=filled,
        iter2=open (used on tracking PDFs).
    show_band_legend : show the band → edge-color legend (suppress if
        all visits in your data are i-band).
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_title('Marker scheme', fontsize=14, pad=20)

    # Rotator (shape)
    rot_handles = [plt.Line2D([], [], linestyle='', marker=m, markersize=10,
                              color='gray', mfc='none')
                   for m in _rot_markers().values()]
    rot_labels = [f'rot ≈ {θ:+d}°' for θ in ROT_CENTERS]
    leg1 = ax.legend(rot_handles, rot_labels,
                     loc='upper left', bbox_to_anchor=(0.05, 0.95),
                     title='rotator_angle (marker shape)',
                     fontsize=9, title_fontsize=10, frameon=True, ncol=1)
    ax.add_artist(leg1)

    # Elevation (color)
    elev_handles = [plt.Line2D([], [], linestyle='', marker='o',
                               markersize=10, color=c)
                    for c in ELEV_COLORS.values()]
    elev_labels = [f'alt ∈ [{c - ELEV_HALFWIDTH}, {c + ELEV_HALFWIDTH}) → {c}°'
                   for c in ELEV_COLORS]
    leg2 = ax.legend(elev_handles, elev_labels,
                     loc='upper right', bbox_to_anchor=(0.95, 0.95),
                     title='elevation (marker color)',
                     fontsize=9, title_fontsize=10, frameon=True)
    ax.add_artist(leg2)

    if show_band_legend:
        band_handles = []
        band_labels = []
        for b, edge in BAND_EDGE_COLORS.items():
            if edge is None:
                edge_disp = 'tab:purple'  # fill color shown for visibility
                lbl = f'{b} band — no distinct edge'
            else:
                edge_disp = edge
                lbl = f'{b} band'
            band_handles.append(plt.Line2D([], [], linestyle='', marker='o',
                                           markersize=10, color='tab:purple',
                                           mec=edge_disp, mew=1.5))
            band_labels.append(lbl)
        leg3 = ax.legend(band_handles, band_labels,
                         loc='lower right', bbox_to_anchor=(0.95, 0.05),
                         title='filter band (marker edge)',
                         fontsize=9, title_fontsize=10, frameon=True)
        ax.add_artist(leg3)

    if show_iter_distinction:
        h_it = [
            plt.Line2D([], [], linestyle='', marker='o', markersize=11,
                       mec='black', mfc='black'),
            plt.Line2D([], [], linestyle='', marker='o', markersize=11,
                       mec='black', mfc='none'),
        ]
        ax.legend(h_it, ['iter 1 (filled)', 'iter 2 (open)'],
                  loc='lower left', bbox_to_anchor=(0.05, 0.05),
                  fontsize=10, title='Iteration (fill style)',
                  title_fontsize=10, frameon=True)

    ax.text(0.5, 0.55,
            'Visits outside the elevation/rotator buckets are drawn as gray ✕.',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color='gray')
    return fig

# Per-visit quality cuts. Computed per visit during mktable, stored as
# numeric columns in visit_info, and re-applied at use time via
# quality_visit_mask(). A visit fails if any of these are violated.
DEFAULT_MIN_DONUTS_PER_VISIT = 500
DEFAULT_MIN_DONUTS_PER_DETECTOR = 3   # used to count "well-covered" detectors
DEFAULT_MIN_DETECTORS_PER_VISIT = 170  # at least this many CCDs with N+ donuts
DEFAULT_MAX_MEDIAN_BLUR_ARCSEC = 1.2


# ============================================================
# Zernike Utilities
# ============================================================

def infer_zernike_indices(nZk):
    """Infer the list of Noll Zernike indices from the number of terms."""
    if nZk == 25:
        return list(range(4, 29))
    elif nZk == 21:
        return list(range(4, 20)) + list(range(22, 27))
    elif nZk == 19:
        return list(range(4, 23))
    else:
        iZs = list(range(4, 4 + nZk))
        print(f"Warning: Unexpected number of Zernike terms ({nZk}), "
              f"assuming Z4-Z{3 + nZk}")
        return iZs


def derive_version_strings(collections):
    """Extract wep and dviz version strings from collection paths."""
    coll = collections[0]
    wep_match = re.search(r'(wep_v[\d_]+)', coll)
    dviz_match = re.search(r'(donut_viz_v[\d_]+)', coll)
    wep_ver = wep_match.group(1) if wep_match else DEFAULT_WEP_VER
    dviz_ver = (dviz_match.group(1).replace('donut_viz_', 'dviz_')
                if dviz_match else DEFAULT_DVIZ_VER)
    return wep_ver, dviz_ver


def parse_collection_info(collections, include_versions=False):
    """Parse collection name(s) to extract a filename phrase and date range.

    For a simple collection like 'aos_fam_danish_triplets':
        collection_phrase = 'aos_fam_danish_triplets', dates = (None, None)

    For 'u/brycek/aos_fam_danish_step2/wep_v16_5_0/donut_viz_v3_2_3/20251020_20251231':
        - Drop leading 'u/USERNAME/'
        - collection_phrase = 'aos_fam_danish_step2'
        - Optionally append version strings if include_versions=True
        - Parse trailing YYYYMMDD_YYYYMMDD as (day_obs_min, day_obs_max)

    Returns (collection_phrase, day_obs_min_or_None, day_obs_max_or_None).
    """
    coll = collections[0]
    parts = coll.split('/')

    # Check for u/USERNAME/... pattern
    if len(parts) >= 3 and parts[0] == 'u':
        # Drop u/USERNAME
        remaining = parts[2:]
        # First element is the collection phrase
        collection_phrase = remaining[0] if remaining else coll
        # Optionally include version strings
        if include_versions and len(remaining) > 1:
            version_parts = []
            for p in remaining[1:]:
                if re.match(r'wep_v[\d_]+', p) or re.match(r'donut_viz_v[\d_]+', p):
                    version_parts.append(p.replace('donut_viz_', 'dviz_'))
            if version_parts:
                collection_phrase += '_' + '_'.join(version_parts)
        # Check if last part is YYYYMMDD_YYYYMMDD or YYYYMMDD
        last = remaining[-1] if remaining else ''
        date_match = re.match(r'^(\d{8})_(\d{8})$', last)
        if date_match:
            return collection_phrase, int(date_match.group(1)), int(date_match.group(2))
        single_date = re.match(r'^(\d{8})$', last)
        if single_date:
            d = int(single_date.group(1))
            return collection_phrase, d, d
        return collection_phrase, None, None
    else:
        # Simple collection name
        return coll, None, None


# ============================================================
# Butler Data Extraction
# ============================================================

def get_aggregate_zernikes(butler, day_obs, seq_num, coord_sys, camera,
                           calc_focal_plane=False, calc_mean_zernike=False,
                           matched_threshold_arcsec=DEFAULT_MATCHED_THRESHOLD_ARCSEC):
    """Get aggregate Zernike table for a single visit.

    Parameters
    ----------
    calc_focal_plane : bool
        If True, compute intra/extra focal plane coordinates (fpx, fpy).
    calc_mean_zernike : bool
        If True, compute per-visit mean Zernike and add as column.
    matched_threshold_arcsec : float or None
        Boolean `matched_intra_extra` column flips True when both
        |Δthx| and |Δthy| are below this (in arcsec). Set to None to
        disable the cut (the column becomes True for every donut while
        the offsets are still recorded in `intra_extra_offset_*`).

    Returns (table, visit_meta_dict) or (None, None).
    """
    try:
        aosTable = butler.get('aggregateAOSVisitTableRaw',
                              day_obs=day_obs, seq_num=seq_num)
    except DatasetNotFoundError:
        print(f"DatasetNotFoundError: No data for day_obs={day_obs}, seq_num={seq_num}")
        return None, None
    except Exception as e:
        error_type = type(e).__name__
        if error_type == 'DimensionValueError':
            print(f"DimensionValueError for day_obs={day_obs}, seq_num={seq_num}")
        else:
            print(f"{error_type} for day_obs={day_obs}, seq_num={seq_num}: {e}")
        return None, None

    meta = aosTable.meta
    visit_meta = {
        'day_obs': day_obs,
        'seq_num': seq_num,
        'visit': meta.get('visit', None),
        'skyAngle': meta.get('rotAngle', np.nan),
        'ra': meta.get('ra', np.nan),
        'dec': meta.get('dec', np.nan),
        'az': meta.get('az', np.nan),
        'alt': meta.get('alt', np.nan),
        'band': meta.get('band', ''),
        'mjd': meta.get('mjd', np.nan),
        'nollIndices': meta.get('nollIndices', None),
    }

    # Extract blur (FWHM) from estimatorInfo metadata
    estimator_info = meta.get('estimatorInfo', {})
    fwhm = estimator_info.get('fwhm', None) if isinstance(estimator_info, dict) else None

    aosTable.meta = {}

    select = (aosTable['used'] == True)  # noqa: E712
    aosTable_sel = aosTable[select]

    if len(aosTable_sel) == 0:
        print(f"Warning: No 'used' donuts for day_obs={day_obs}, seq_num={seq_num}")
        return None, None

    aosTable_sel['seq_num'] = seq_num
    aosTable_sel['day_obs'] = day_obs

    # Add blur column
    if fwhm is not None:
        fwhm_arr = np.array(fwhm)
        # Prefer the selected-length match: when no donuts are dropped
        # len(aosTable) == len(aosTable_sel) and the array is already aligned to
        # the selection, so applying `select` again would be wrong.
        if len(fwhm_arr) == len(aosTable_sel):
            aosTable_sel['blur'] = fwhm_arr
        elif len(fwhm_arr) == len(aosTable):
            aosTable_sel['blur'] = fwhm_arr[select]
        else:
            print(f"Warning: fwhm length ({len(fwhm_arr)}) doesn't match "
                  f"table ({len(aosTable)}) or selected ({len(aosTable_sel)})")
            aosTable_sel['blur'] = np.nan
    else:
        aosTable_sel['blur'] = np.nan

    # Per-donut fit-quality / status from estimatorInfo (same length handling
    # as blur).  chi2 = donut-fit chi-square; lstsq_* come straight from
    # scipy.optimize.least_squares (status: 1-4 converged, 0=max-iter).
    # (fit_success / exception_status omitted: in the aggregate table they are
    # uniformly True / empty even for not-used donuts, so they carry no info.)
    def _add_estimator_col(out_name, ei_key, default=np.nan):
        v = (estimator_info.get(ei_key)
             if isinstance(estimator_info, dict) else None)
        if v is None:
            aosTable_sel[out_name] = default
            return
        arr = np.asarray(v)
        # Prefer the selected-length match (see blur handling above).
        if len(arr) == len(aosTable_sel):
            aosTable_sel[out_name] = arr
        elif len(arr) == len(aosTable):
            aosTable_sel[out_name] = arr[select]
        else:
            aosTable_sel[out_name] = default

    _add_estimator_col('chi2', 'chi_square')
    _add_estimator_col('lstsq_cost', 'lstsq_cost')
    _add_estimator_col('lstsq_optimality', 'lstsq_optimality')
    _add_estimator_col('lstsq_status', 'lstsq_status', default=-99)

    # Fitted model per donut: dx/dy = centering offset, flux per stamp.
    # Each is a 2-element array [stamp0, stamp1] for the intra/extra pair
    # (stored like the zk_* array columns).  Useful for centering and a
    # flux-based noise/SNR assessment.
    _add_estimator_col('model_dx', 'model_dx')
    _add_estimator_col('model_dy', 'model_dy')
    _add_estimator_col('model_flux', 'model_flux')

    # Focal plane coordinates (optional, expensive per-detector loop)
    if calc_focal_plane:
        nstars = len(aosTable_sel)
        intra_fpx = np.zeros(nstars)
        intra_fpy = np.zeros(nstars)
        extra_fpx = np.zeros(nstars)
        extra_fpy = np.zeros(nstars)

        for detector in camera:
            selone = (aosTable_sel['detector'] == detector.getName())
            if not np.any(selone):
                continue

            x_one = aosTable_sel[selone]['centroid_x_intra']
            y_one = aosTable_sel[selone]['centroid_y_intra']
            fpx_one, fpy_one = pixel_to_focal(x_one, y_one, detector)
            intra_fpx[selone] = fpx_one
            intra_fpy[selone] = fpy_one

            x_one = aosTable_sel[selone]['centroid_x_extra']
            y_one = aosTable_sel[selone]['centroid_y_extra']
            fpx_one, fpy_one = pixel_to_focal(x_one, y_one, detector)
            extra_fpx[selone] = fpx_one
            extra_fpy[selone] = fpy_one

        aosTable_sel['intra_fpx'] = intra_fpx
        aosTable_sel['intra_fpy'] = intra_fpy
        aosTable_sel['extra_fpx'] = extra_fpx
        aosTable_sel['extra_fpy'] = extra_fpy

    # Coordinate system column names
    zk_col = f'zk_{coord_sys}'
    thx_intra_col = f'thx_{coord_sys}_intra'
    thx_extra_col = f'thx_{coord_sys}_extra'
    thy_intra_col = f'thy_{coord_sys}_intra'
    thy_extra_col = f'thy_{coord_sys}_extra'

    thx_diff = np.abs(aosTable_sel[thx_intra_col] - aosTable_sel[thx_extra_col]) * 206265
    thy_diff = np.abs(aosTable_sel[thy_intra_col] - aosTable_sel[thy_extra_col]) * 206265
    if matched_threshold_arcsec is None:
        matched_intra_extra = np.ones(len(aosTable_sel), dtype=bool)
    else:
        matched_intra_extra = (thx_diff < matched_threshold_arcsec) & \
                              (thy_diff < matched_threshold_arcsec)
    aosTable_sel['matched_intra_extra'] = matched_intra_extra
    aosTable_sel['intra_extra_offset_x_arcsec'] = thx_diff
    aosTable_sel['intra_extra_offset_y_arcsec'] = thy_diff

    # Per-visit mean Zernike (optional)
    if calc_mean_zernike:
        zk_mean_col = f'zk_{coord_sys}_mean'
        values_array = np.stack(aosTable_sel[zk_col])
        mean_values = np.mean(values_array, axis=0)
        npts = len(aosTable_sel)
        aosTable_sel[zk_mean_col] = [mean_values for _ in range(npts)]

    return aosTable_sel, visit_meta


def _astropy_table_to_pyarrow(tbl):
    """Convert an astropy Table to a pyarrow Table, preserving list/array
    columns as pyarrow list<float> columns. Scalar columns pass through.
    """
    arrays = []
    names = []
    for col in tbl.colnames:
        data = tbl[col]
        arr = np.asarray(data)
        if arr.ndim == 2:
            # Per-row array column (e.g. zk_OCS shape (n, n_zk))
            # Pass as a list of per-row numpy arrays; pyarrow infers list<float>
            rows = [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]
            arrays.append(pa.array(rows))
        elif arr.dtype.kind == 'O':
            arrays.append(pa.array(list(arr)))
        else:
            arrays.append(pa.array(arr))
        names.append(col)
    return pa.Table.from_arrays(arrays, names=names)


def read_donuts_table(parquet_path, visit_pairs=None):
    """Load the donuts parquet table as a QTable.

    If visit_pairs (list of (day_obs, seq_num)) is provided, only those
    visits' row groups are loaded. Otherwise the full table is loaded.
    List columns (zk_OCS, etc.) are restacked into 2-D numpy arrays so
    downstream code sees the same shape as before.
    """
    pf = pq.ParquetFile(str(parquet_path))

    if visit_pairs is not None:
        wanted = set((int(d), int(s)) for d, s in visit_pairs)
        chosen = []
        for i in range(pf.num_row_groups):
            meta = pf.metadata.row_group(i)
            # Look up day_obs / seq_num in the row-group statistics
            stats = {}
            for ci in range(meta.num_columns):
                cmeta = meta.column(ci)
                name = cmeta.path_in_schema
                if name in ('day_obs', 'seq_num') and cmeta.statistics is not None:
                    stats[name] = cmeta.statistics.min  # min == max for single-visit
            key = (stats.get('day_obs'), stats.get('seq_num'))
            if key in wanted:
                chosen.append(i)
        tbls = [pf.read_row_group(i) for i in chosen]
        tbl_pa = pa.concat_tables(tbls) if tbls else pf.read_row_group(0).slice(0, 0)
    else:
        tbl_pa = pf.read()

    df = tbl_pa.to_pandas()

    # Reconstruct an astropy QTable, promoting list columns to 2-D arrays
    out = QTable()
    for col in df.columns:
        vals = df[col].values
        if len(vals) > 0 and isinstance(vals[0], (list, np.ndarray)) \
           and not isinstance(vals[0], (str, bytes)):
            try:
                out[col] = np.stack(vals)
                continue
            except ValueError as e:
                print(f"  WARNING: column {col!r} could not be stacked to a "
                      f"2-D array (ragged rows?), kept as object dtype: {e}")
        out[col] = vals
    return out


def read_donuts_for_visit(parquet_path, day_obs, seq_num):
    """Read the donuts for one visit (one row group) as a pandas DataFrame.

    Relies on the streaming writer putting each visit in its own row group,
    with min/max statistics on day_obs and seq_num so a full scan is avoided.
    Falls back to a filter if row-group stats don't match.
    """
    pf = pq.ParquetFile(str(parquet_path))
    day_obs = int(day_obs)
    seq_num = int(seq_num)
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
        if d == day_obs and s == seq_num:
            return pf.read_row_group(i).to_pandas()
    # Fallback: scan all row groups
    df = pf.read().to_pandas()
    return df[(df['day_obs'] == day_obs) & (df['seq_num'] == seq_num)].copy()


# ---------------------------------------------------------------------------
# Per-visit Zernike extraction — shared by the serial and parallel streaming
# paths.  The expensive part is get_aggregate_zernikes() (Butler dataset
# reads); everything downstream (quality metrics, column pruning, pyarrow
# conversion) is pure CPU and safe to run in a worker process.
# ---------------------------------------------------------------------------
def _compute_visit_payload(butler, pair, coord_sys, camera,
                           calc_focal_plane, calc_mean_zernike,
                           matched_threshold_arcsec, min_donuts_per_detector):
    """Extract + post-process one visit.

    Returns a picklable payload tuple
    ``(status, day_obs, seq_num, noll, visit_meta, tbl_pa)`` where ``status``
    is 'ok' or 'empty'.  Holds no shared state, so it runs unchanged inside a
    ProcessPoolExecutor worker.
    """
    day_obs_val, seq_num = pair
    agg_zern, visit_meta = get_aggregate_zernikes(
        butler, day_obs_val, seq_num, coord_sys, camera,
        calc_focal_plane=calc_focal_plane,
        calc_mean_zernike=calc_mean_zernike,
        matched_threshold_arcsec=matched_threshold_arcsec)
    if agg_zern is None:
        return ('empty', day_obs_val, seq_num, None, None, None)

    # Per-visit quality metrics computed BEFORE dropping columns, since
    # `blur` is one of the metrics we care about.
    visit_meta['n_donuts'] = int(len(agg_zern))
    if 'detector' in agg_zern.colnames:
        det_counts = Counter(np.asarray(agg_zern['detector']).tolist())
        visit_meta['n_detectors'] = int(len(det_counts))
        visit_meta['n_detectors_with_min_donuts'] = int(sum(
            1 for c in det_counts.values() if c >= min_donuts_per_detector))
    else:
        visit_meta['n_detectors'] = 0
        visit_meta['n_detectors_with_min_donuts'] = 0
    if 'blur' in agg_zern.colnames:
        with np.errstate(invalid='ignore'):
            visit_meta['median_blur_arcsec'] = float(
                np.nanmedian(np.asarray(agg_zern['blur'], dtype=float)))
    else:
        visit_meta['median_blur_arcsec'] = float('nan')

    # Drop unwanted columns per-visit
    drop_cols = [c for c in agg_zern.colnames
                 if '_W' in c or '_N' in c or '_NW' in c
                 or '_ra_' in c or '_dec_' in c]
    if drop_cols:
        agg_zern.remove_columns(drop_cols)

    tbl_pa = _astropy_table_to_pyarrow(agg_zern)
    noll = visit_meta.get('nollIndices', None)
    noll = list(noll) if noll is not None else None
    return ('ok', day_obs_val, seq_num, noll, visit_meta, tbl_pa)


# Per-process state for the parallel path.  Each worker builds its own Butler
# (a Butler is expensive to pickle and must not be shared across processes);
# the camera is rebuilt from LsstCam's class-level cache.
_VISIT_WORKER = {}


def _init_visit_worker(butler_repo, collections, coord_sys, calc_focal_plane,
                       calc_mean_zernike, matched_threshold_arcsec,
                       min_donuts_per_detector):
    global _VISIT_WORKER
    _VISIT_WORKER = dict(
        butler=Butler(butler_repo, instrument='LSSTCam', collections=collections),
        camera=LsstCam.getCamera(),
        coord_sys=coord_sys,
        calc_focal_plane=calc_focal_plane,
        calc_mean_zernike=calc_mean_zernike,
        matched_threshold_arcsec=matched_threshold_arcsec,
        min_donuts_per_detector=min_donuts_per_detector,
    )


def _visit_worker(pair):
    w = _VISIT_WORKER
    return _compute_visit_payload(
        w['butler'], pair, w['coord_sys'], w['camera'],
        w['calc_focal_plane'], w['calc_mean_zernike'],
        w['matched_threshold_arcsec'], w['min_donuts_per_detector'])


def _parallel_visit_payloads(visit_pairs, workers, init_args):
    """Yield per-visit payloads from a process pool.

    Keeps at most ~2*workers visits in flight so memory stays bounded, and
    yields in completion order — row-group order is irrelevant downstream and
    the noll reference is identical across visits, so order does not matter.
    A 'spawn' context is used (not fork) to avoid deadlocking on locks held by
    background threads inside the already-initialised LSST stack.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_visit_worker,
                             initargs=init_args) as ex:
        it = iter(visit_pairs)
        pending = set()
        for _ in range(max(1, workers * 2)):
            try:
                pending.add(ex.submit(_visit_worker, next(it)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                yield fut.result()
                try:
                    pending.add(ex.submit(_visit_worker, next(it)))
                except StopIteration:
                    pass


def stream_zernikes_to_parquet(visit_pairs, collections, butler_repo, coord_sys,
                               camera, output_file,
                               calc_focal_plane=False, calc_mean_zernike=False,
                               matched_threshold_arcsec=DEFAULT_MATCHED_THRESHOLD_ARCSEC,
                               min_donuts_per_detector=DEFAULT_MIN_DONUTS_PER_DETECTOR,
                               workers=1):
    """Stream aggregate Zernikes per visit to parquet (one row group per visit).

    List/array columns (zk_OCS, zk_intrinsic_OCS, etc.) are stored as
    native parquet list<float> — no explode/restack, so the file is
    readable by any parquet-aware tool.

    Per-visit quality metrics (n_donuts, n_detectors_with_min_donuts,
    median_blur_arcsec) are computed from each visit's table and stored
    in the returned visit_info QTable for downstream cuts.

    ``workers`` controls the per-visit extraction parallelism: 1 (default)
    keeps the original single-process behaviour; >1 fans the Butler reads out
    over a process pool while the parquet writing stays serial in this process.

    Returns visit_info (small QTable), or None on failure.
    """
    visit_meta_list = []
    success_count = 0
    error_count = 0
    ref_noll_indices = None
    noll_mismatch_count = 0
    total_rows = 0
    writer = None
    ref_schema = None

    n_workers = max(1, int(workers or 1))
    mode = 'serial' if n_workers == 1 else f'{n_workers} workers'
    print(f"\nStreaming Zernikes for {len(visit_pairs)} visits to {output_file} "
          f"({mode})...")

    # Overwrite existing file
    if Path(output_file).exists():
        Path(output_file).unlink()

    # Build the per-visit payload source: a single-process generator or a
    # bounded process pool.  Both yield identical payload tuples, so the
    # writer loop below is shared.
    if n_workers == 1:
        print(f"Initializing Butler with repo={butler_repo}, "
              f"collections: {collections}")
        butler = Butler(butler_repo, instrument='LSSTCam', collections=collections)
        payloads = (
            _compute_visit_payload(
                butler, pair, coord_sys, camera, calc_focal_plane,
                calc_mean_zernike, matched_threshold_arcsec,
                min_donuts_per_detector)
            for pair in visit_pairs)
    else:
        init_args = (butler_repo, collections, coord_sys, calc_focal_plane,
                     calc_mean_zernike, matched_threshold_arcsec,
                     min_donuts_per_detector)
        payloads = _parallel_visit_payloads(visit_pairs, n_workers, init_args)

    try:
        for status, day_obs_val, seq_num, noll, visit_meta, tbl_pa in tqdm(
                payloads, total=len(visit_pairs)):
            if status != 'ok':
                error_count += 1
                continue

            if noll is not None:
                if ref_noll_indices is None:
                    ref_noll_indices = noll
                elif noll != ref_noll_indices:
                    print(f"WARNING: nollIndices mismatch for day_obs={day_obs_val}, "
                          f"seq_num={seq_num}: {noll} != {ref_noll_indices} — skipping")
                    noll_mismatch_count += 1
                    error_count += 1
                    continue

            if writer is None:
                ref_schema = tbl_pa.schema
                writer = pq.ParquetWriter(str(output_file), ref_schema,
                                          compression='snappy')
            else:
                # Schema mismatch would corrupt the file — cast to reference
                try:
                    tbl_pa = tbl_pa.select(ref_schema.names).cast(ref_schema)
                except Exception as e:
                    print(f"WARNING: schema mismatch for day_obs={day_obs_val}, "
                          f"seq_num={seq_num}: {e} — skipping")
                    error_count += 1
                    continue

            writer.write_table(tbl_pa)  # one row group per visit
            visit_meta_list.append(visit_meta)
            success_count += 1
            total_rows += visit_meta['n_donuts']
    finally:
        if writer is not None:
            writer.close()

    print(f"\n{'='*60}")
    print("Extraction Summary:")
    print(f"  Total visits attempted: {len(visit_pairs)}")
    print(f"  Successful extractions: {success_count}")
    print(f"  Failed extractions:     {error_count}")
    if noll_mismatch_count > 0:
        print(f"  nollIndices mismatches: {noll_mismatch_count}")
    if ref_noll_indices is not None:
        print(f"  nollIndices: {ref_noll_indices}")
    print(f"  Total donut measurements: {total_rows}")
    print(f"{'='*60}\n")

    if success_count == 0:
        print(f"\nERROR: No Zernike data found for any visits!")
        print(f"Collections used: {collections}")
        return None

    # Build visit_info QTable (small, stays in memory)
    visit_info = QTable()
    visit_info['day_obs'] = [m['day_obs'] for m in visit_meta_list]
    visit_info['seq_num'] = [m['seq_num'] for m in visit_meta_list]
    visit_info['visit'] = [m['visit'] for m in visit_meta_list]
    visit_info['skyAngle'] = [m['skyAngle'] for m in visit_meta_list]
    visit_info['ra'] = [m['ra'] for m in visit_meta_list]
    visit_info['dec'] = [m['dec'] for m in visit_meta_list]
    visit_info['az'] = [m['az'] for m in visit_meta_list]
    visit_info['alt'] = [m['alt'] for m in visit_meta_list]
    visit_info['band'] = [m['band'] for m in visit_meta_list]
    visit_info['mjd'] = [m['mjd'] for m in visit_meta_list]
    visit_info['nollIndices'] = [m['nollIndices'] for m in visit_meta_list]
    # Per-visit quality metrics (used by quality_visit_mask downstream)
    visit_info['n_donuts'] = [m['n_donuts'] for m in visit_meta_list]
    visit_info['n_detectors'] = [m['n_detectors'] for m in visit_meta_list]
    visit_info['n_detectors_with_min_donuts'] = [
        m['n_detectors_with_min_donuts'] for m in visit_meta_list]
    visit_info['median_blur_arcsec'] = [
        m['median_blur_arcsec'] for m in visit_meta_list]
    # Stash the threshold used at extract time so users can interpret
    # the n_detectors_with_min_donuts column.
    visit_info.meta['min_donuts_per_detector'] = int(min_donuts_per_detector)
    visit_info.meta['matched_threshold_arcsec'] = (
        float(matched_threshold_arcsec) if matched_threshold_arcsec is not None
        else None)

    return visit_info


def get_zernikes_from_visits(visit_pairs, collections, butler_repo, coord_sys,
                             camera, calc_focal_plane=False,
                             calc_mean_zernike=False):
    """Get aggregate Zernikes for a list of (day_obs, seq_num) pairs.

    Returns (aosTable, visit_info_table) or (None, None).
    """
    print(f"Initializing Butler with repo={butler_repo}, collections: {collections}")
    butler = Butler(butler_repo, instrument='LSSTCam', collections=collections)

    agg_zernikes_list = []
    visit_meta_list = []
    success_count = 0
    error_count = 0
    ref_noll_indices = None
    noll_mismatch_count = 0

    print(f"\nExtracting Zernikes for {len(visit_pairs)} visits...")
    for day_obs_val, seq_num in tqdm(visit_pairs):
        agg_zern, visit_meta = get_aggregate_zernikes(
            butler, day_obs_val, seq_num, coord_sys, camera,
            calc_focal_plane=calc_focal_plane,
            calc_mean_zernike=calc_mean_zernike)
        if agg_zern is None:
            error_count += 1
            continue

        noll = visit_meta.get('nollIndices', None)
        if noll is not None:
            if ref_noll_indices is None:
                ref_noll_indices = list(noll)
            elif list(noll) != ref_noll_indices:
                print(f"WARNING: nollIndices mismatch for day_obs={day_obs_val}, "
                      f"seq_num={seq_num}: {list(noll)} != {ref_noll_indices} — skipping")
                noll_mismatch_count += 1
                error_count += 1
                continue

        agg_zernikes_list.append(agg_zern)
        visit_meta_list.append(visit_meta)
        success_count += 1

    print(f"\n{'='*60}")
    print(f"Extraction Summary:")
    print(f"  Total visits attempted: {len(visit_pairs)}")
    print(f"  Successful extractions: {success_count}")
    print(f"  Failed extractions:     {error_count}")
    if noll_mismatch_count > 0:
        print(f"  nollIndices mismatches: {noll_mismatch_count}")
    if ref_noll_indices is not None:
        print(f"  nollIndices: {ref_noll_indices}")

    if len(agg_zernikes_list) == 0:
        print(f"\nERROR: No Zernike data found for any visits!")
        print(f"Collections used: {collections}")
        return None, None

    agg_zernikes = vstack(agg_zernikes_list)
    print(f"  Total donut measurements: {len(agg_zernikes)}")
    print(f"{'='*60}\n")

    # Build visit-level info table
    visit_info = QTable()
    visit_info['day_obs'] = [m['day_obs'] for m in visit_meta_list]
    visit_info['seq_num'] = [m['seq_num'] for m in visit_meta_list]
    visit_info['visit'] = [m['visit'] for m in visit_meta_list]
    visit_info['skyAngle'] = [m['skyAngle'] for m in visit_meta_list]
    visit_info['ra'] = [m['ra'] for m in visit_meta_list]
    visit_info['dec'] = [m['dec'] for m in visit_meta_list]
    visit_info['az'] = [m['az'] for m in visit_meta_list]
    visit_info['alt'] = [m['alt'] for m in visit_meta_list]
    visit_info['band'] = [m['band'] for m in visit_meta_list]
    visit_info['mjd'] = [m['mjd'] for m in visit_meta_list]
    visit_info['nollIndices'] = [m['nollIndices'] for m in visit_meta_list]

    return agg_zernikes, visit_info


# ============================================================
# ConsDB Queries
# ============================================================

def get_visit_pairs_from_consdb(visits_df, programs, img_type='cwfs',
                                verify_pairing=True):
    """Extract (day_obs, seq_num) pairs from ConsDB dataframe."""
    # Coerce to string — ConsDB sometimes returns the column as float64
    # (all NaN) when no rows match yet, which breaks .str.contains.
    program_col = visits_df['science_program'].fillna('').astype(str)
    program_mask = program_col.str.contains(programs[0], na=False)
    for prog in programs[1:]:
        program_mask |= program_col.str.contains(prog, na=False)

    filtered = visits_df[program_mask & (visits_df['img_type'] == img_type)].copy()

    if len(filtered) == 0:
        print("Warning: No matching visits found in ConsDB")
        return []

    filtered = filtered.sort_values(['day_obs', 'seq_num'])

    if not verify_pairing:
        visit_pairs = list(zip(filtered['day_obs'], filtered['seq_num']))
        print(f"\nFound {len(visit_pairs)} matching visits (no pairing verification)")
        return visit_pairs

    visit_pairs = []
    unpaired_count = 0

    for day_obs_val, group in filtered.groupby('day_obs'):
        seq_nums = sorted(group['seq_num'].values)
        i = 0
        while i < len(seq_nums) - 1:
            if seq_nums[i+1] == seq_nums[i] + 1:
                visit_pairs.append((day_obs_val, seq_nums[i+1]))
                i += 2
            else:
                if unpaired_count < 5:
                    print(f"Warning: Unpaired image at day_obs={day_obs_val}, "
                          f"seq_num={seq_nums[i]}")
                unpaired_count += 1
                i += 1
        if i == len(seq_nums) - 1:
            if unpaired_count < 5:
                print(f"Warning: Unpaired image at day_obs={day_obs_val}, "
                      f"seq_num={seq_nums[i]}")
            unpaired_count += 1

    if unpaired_count > 5:
        print(f"... and {unpaired_count - 5} more unpaired images")

    print(f"\nFound {len(visit_pairs)} FAM image pairs (keeping second of each pair)")
    if unpaired_count > 0:
        print(f"Note: {unpaired_count} unpaired images were skipped")

    return visit_pairs


def print_band_counts_by_day(df, block_names, img_type_value):
    """Print a table showing band counts by day_obs for filtered data."""
    if isinstance(block_names, str):
        block_names = [block_names]

    # Coerce to string for the same reason as get_visit_pairs_from_consdb
    program_col = df['science_program'].fillna('').astype(str)
    block_mask = program_col.str.contains(block_names[0], na=False)
    for block_name in block_names[1:]:
        block_mask |= program_col.str.contains(block_name, na=False)

    filtered = df[block_mask & (df['img_type'] == img_type_value)].copy()
    print(f"Total rows matching {block_names} and img_type='{img_type_value}': "
          f"{len(filtered)}")

    if len(filtered) == 0:
        print("No matching rows found.")
        return

    filtered['band_first'] = filtered['band'].str[0]
    band_table = pd.crosstab(filtered['day_obs'], filtered['band_first'])
    desired_order = ['u', 'g', 'r', 'i', 'z', 'y']
    band_table = band_table.reindex(columns=desired_order, fill_value=0)
    totals = band_table.sum(axis=0)
    band_table.loc['TOTAL'] = totals

    print("\nBand counts by day_obs:")
    print(band_table)


# ============================================================
# Rotator Angle Functions
# ============================================================

def get_rotator_angles(visits_df, visit_pairs):
    """Get physical_rotator_angle from ConsDB (visit1_quicklook)."""
    visits_indexed = visits_df.set_index(['day_obs', 'seq_num'])

    records = []
    for day_obs_val, seq_num in visit_pairs:
        rec = {'day_obs': day_obs_val, 'seq_num': seq_num,
               'physical_rotator_angle': np.nan}
        try:
            row = visits_indexed.loc[(day_obs_val, seq_num)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            phys_rot = row.get('physical_rotator_angle', np.nan)
            if pd.notna(phys_rot):
                rec['physical_rotator_angle'] = float(phys_rot)
        except KeyError:
            pass
        records.append(rec)

    rotator_df = pd.DataFrame(records)
    n_phys = rotator_df['physical_rotator_angle'].notna().sum()
    print(f"\nConsDB physical_rotator_angle: {n_phys}/{len(rotator_df)} visits")
    return rotator_df


def get_visitinfo_rotator_angles(butler, visit_pairs):
    """Get RotPA and parallactic angle from Butler raw.visitInfo."""
    records = []
    print(f"Querying Butler visitInfo for {len(visit_pairs)} visits...")
    for day_obs_val, seq_num in tqdm(visit_pairs):
        rec = {'day_obs': day_obs_val, 'seq_num': seq_num,
               'visitinfo_rotpa': np.nan, 'visitinfo_par_angle': np.nan,
               'visitinfo_rotator_angle': np.nan}
        try:
            visinfo = butler.get('raw.visitInfo', instrument='LSSTCam',
                                 detector=4, day_obs=day_obs_val, seq_num=seq_num)
            rotpa = visinfo.getBoresightRotAngle().asDegrees()
            par = visinfo.getBoresightParAngle().asDegrees()
            rec['visitinfo_rotpa'] = rotpa
            rec['visitinfo_par_angle'] = par
            rec['visitinfo_rotator_angle'] = calc_rotator_from_visitinfo(par, rotpa)
        except Exception:
            try:
                raw = butler.get('raw', instrument='LSSTCam',
                                 detector=4, day_obs=day_obs_val, seq_num=seq_num)
                visinfo = raw.visitInfo
                rotpa = visinfo.getBoresightRotAngle().asDegrees()
                par = visinfo.getBoresightParAngle().asDegrees()
                rec['visitinfo_rotpa'] = rotpa
                rec['visitinfo_par_angle'] = par
                rec['visitinfo_rotator_angle'] = calc_rotator_from_visitinfo(par, rotpa)
            except Exception as e2:
                print(f"  visitInfo lookup failed for day_obs={day_obs_val} "
                      f"seq_num={seq_num} (rotator left NaN): "
                      f"{type(e2).__name__}: {e2}")
        records.append(rec)

    df = pd.DataFrame(records)
    n_ok = df['visitinfo_rotpa'].notna().sum()
    print(f"  Got visitInfo for {n_ok}/{len(visit_pairs)} visits")
    return df


async def async_getEfdData_exprecord(client, topic, expRecord, columns=None,
                                     prePadding=0, postPadding=0):
    """Async EFD query using a Butler exposure record for time range."""
    begin = Time(expRecord.timespan.begin, scale="tai")
    end = Time(expRecord.timespan.end, scale="tai")
    if prePadding:
        begin -= TimeDelta(prePadding, format="sec")
    if postPadding:
        end += TimeDelta(postPadding, format="sec")

    data = await client.select_time_series(topic, columns or [], begin.utc, end.utc)
    if data is None or len(data) == 0:
        return None
    return data


async def get_efd_rotator_angles(efd_client, butler, visit_pairs):
    """Get actual rotator position from EFD MTRotator telemetry."""
    records = []
    print(f"Querying EFD rotator positions for {len(visit_pairs)} visits...")
    for day_obs_val, seq_num in tqdm(visit_pairs):
        rec = {'day_obs': day_obs_val, 'seq_num': seq_num,
               'efd_rotator_angle': np.nan}
        try:
            results = butler.registry.queryDimensionRecords(
                'exposure',
                where="instrument='LSSTCam' AND exposure.day_obs=:day_obs "
                      "AND exposure.seq_num=:seq_num",
                bind={'day_obs': day_obs_val, 'seq_num': seq_num}
            )
            exprec = next(iter(results), None)
            if exprec is None:
                records.append(rec)
                continue

            rotData = await async_getEfdData_exprecord(
                efd_client, "lsst.sal.MTRotator.rotation",
                expRecord=exprec, columns=["actualPosition"],
            )
            if rotData is not None and 'actualPosition' in rotData.columns:
                rec['efd_rotator_angle'] = float(
                    rotData['actualPosition'].values.mean())
        except Exception:
            pass
        records.append(rec)

    df = pd.DataFrame(records)
    n_ok = df['efd_rotator_angle'].notna().sum()
    print(f"  Got EFD rotator for {n_ok}/{len(visit_pairs)} visits")
    return df


async def get_rotator_data(visits_df, visit_pairs, butler_repo,
                           rotator_threshold=DEFAULT_ROTATOR_THRESHOLD):
    """Get rotator angles from all sources and compute best value.

    Returns rotator_df with columns: day_obs, seq_num, physical_rotator_angle,
    efd_rotator_angle, visitinfo_rotator_angle, rotator_angle, rotator_flagged.
    """
    rotator_df = get_rotator_angles(visits_df, visit_pairs)

    missing_mask = rotator_df['physical_rotator_angle'].isna()
    missing_pairs = rotator_df.loc[missing_mask, ['day_obs', 'seq_num']].values.tolist()
    n_missing = len(missing_pairs)
    print(f"ConsDB physical_rotator_angle: "
          f"{len(rotator_df) - n_missing} present, {n_missing} missing")

    if n_missing > 0:
        # Summarize missing visits by day_obs for quick inspection
        missing_by_day = Counter(d for d, _ in missing_pairs)
        print(f"  Missing from ConsDB — falling back to EFD, then Butler visitInfo:")
        for d in sorted(missing_by_day):
            day_total = sum(1 for dv, _ in visit_pairs if dv == d)
            print(f"    day_obs={d}: {missing_by_day[d]}/{day_total} visits missing")
        # Show individual missing visits (up to 20, then summary)
        show_n = min(20, n_missing)
        print(f"  First {show_n} missing (day_obs, seq_num):")
        for d, s in missing_pairs[:show_n]:
            print(f"    {d} {s}")
        if n_missing > show_n:
            print(f"    ... and {n_missing - show_n} more")

        butler_rot = Butler(butler_repo, instrument='LSSTCam',
                            collections=['LSSTCam/raw/all'])

        # Step 1: try EFD first (faster — MTRotator.rotation queries)
        efd_df = None
        efd_client = None
        try:
            efd_client = makeEfdClient()
            efd_df = await get_efd_rotator_angles(efd_client, butler_rot, missing_pairs)
            rotator_df = rotator_df.merge(efd_df, on=['day_obs', 'seq_num'], how='left')
        except Exception as e:
            print(f"Warning: Could not access EFD: {e}")
            rotator_df['efd_rotator_angle'] = np.nan
        finally:
            if efd_client is not None:
                await _close_efd_client(efd_client)

        # Step 2: for visits still missing (EFD returned NaN too), fall back
        # to Butler visitInfo (much slower; only hit if needed)
        still_missing_mask = (rotator_df['physical_rotator_angle'].isna()
                              & rotator_df['efd_rotator_angle'].isna())
        still_missing_pairs = rotator_df.loc[
            still_missing_mask, ['day_obs', 'seq_num']].values.tolist()
        n_still_missing = len(still_missing_pairs)

        if n_still_missing > 0:
            print(f"  After EFD: {n_still_missing}/{n_missing} still missing, "
                  f"querying Butler visitInfo...")
            visitinfo_df = get_visitinfo_rotator_angles(
                butler_rot, still_missing_pairs)
            rotator_df = rotator_df.merge(
                visitinfo_df, on=['day_obs', 'seq_num'], how='left')
        else:
            print(f"  EFD filled all {n_missing} ConsDB-missing visits — "
                  f"skipping Butler visitInfo queries")
            rotator_df['visitinfo_rotator_angle'] = np.nan
            rotator_df['visitinfo_rotpa'] = np.nan
            rotator_df['visitinfo_par_angle'] = np.nan
    else:
        print("All visits have ConsDB physical_rotator_angle, "
              "skipping EFD/visitInfo queries")
        rotator_df['efd_rotator_angle'] = np.nan
        rotator_df['visitinfo_rotator_angle'] = np.nan
        rotator_df['visitinfo_rotpa'] = np.nan
        rotator_df['visitinfo_par_angle'] = np.nan

    # Best rotator: prefer ConsDB, then EFD, then visitInfo
    rotator_df['rotator_angle'] = (
        rotator_df['physical_rotator_angle']
        .fillna(rotator_df.get('efd_rotator_angle', np.nan))
        .fillna(rotator_df.get('visitinfo_rotator_angle', np.nan))
    )
    rotator_df['rotator_flagged'] = (
        rotator_df['rotator_angle'].abs() > rotator_threshold)

    n_flagged = rotator_df['rotator_flagged'].sum()
    n_phys = rotator_df['physical_rotator_angle'].notna().sum()
    n_efd = rotator_df['efd_rotator_angle'].notna().sum()
    n_vi = rotator_df['visitinfo_rotator_angle'].notna().sum()
    print(f"\nRotator angle sources: ConsDB={n_phys}, EFD={n_efd}, visitInfo={n_vi}")
    print(f"Flagged (|angle| > {rotator_threshold} deg): {n_flagged}")

    return rotator_df


# ============================================================
# Thermal Data Functions
# ============================================================

async def async_getEfdData_times(client, topic, obs_start, obs_end,
                                 columns=None, prePadding=0, postPadding=0,
                                 index=None):
    """Async EFD query using observation start/end times (TAI isot strings)."""
    begin = Time(obs_start, scale="tai")
    end = Time(obs_end, scale="tai")
    if prePadding:
        begin -= TimeDelta(prePadding, format="sec")
    if postPadding:
        end += TimeDelta(postPadding, format="sec")

    kwargs = {}
    if index is not None:
        kwargs["index"] = index
        kwargs["convert_influx_index"] = True

    data = await client.select_time_series(
        topic, columns or [], begin.utc, end.utc, **kwargs)
    if data is None or len(data) == 0:
        return None
    return data


async def get_ess_temperature(efd_client, obs_start, obs_end, index,
                              field="temperatureItem0",
                              post_padding=DEFAULT_TEMP_TIME_WINDOW_SEC):
    """Query a single ESS temperature sensor, return mean value."""
    data = await async_getEfdData_times(
        efd_client, "lsst.sal.ESS.temperature",
        obs_start, obs_end, columns=[field],
        postPadding=post_padding, index=index,
    )
    if data is not None and field in data.columns:
        return data[field].mean()
    return np.nan


def _interp_dataframe_to_times(df_in, data_times_int, t0):
    """Interpolate every column of a time-indexed DataFrame to a new time
    array (data_times_int, in nanoseconds since epoch). t0 is the reference
    time used for normalization. NaNs in the source are linearly bridged.
    Columns with object dtype (e.g. EFD per-thermocouple series mixing
    floats and None) are coerced to float; non-numeric → NaN.
    """
    src_times = pd.to_datetime(
        df_in.index, format="ISO8601", utc=True).astype("int64")
    src_times_norm = np.asarray((src_times - t0) / 1e9, dtype=float)
    target_times_norm = np.asarray((data_times_int - t0) / 1e9, dtype=float)
    n_target = len(target_times_norm)
    out = {}
    for col in df_in.columns:
        # Coerce to numeric float; non-numeric (e.g. None) → NaN
        values = pd.to_numeric(df_in[col], errors='coerce').to_numpy(dtype=float)
        if np.all(np.isnan(values)):
            out[col] = np.full(n_target, np.nan)
            continue
        val_interpolated = pd.Series(values).interpolate().to_numpy(dtype=float)
        # If the leading or trailing values were NaN, interpolate() leaves
        # them NaN — np.interp would then propagate that. Explicitly handle
        # by clipping target times to the valid src range.
        valid = ~np.isnan(val_interpolated)
        if not valid.any():
            out[col] = np.full(n_target, np.nan)
            continue
        out[col] = np.interp(
            target_times_norm,
            src_times_norm[valid], val_interpolated[valid])
    return out


def _thermocouple_metadata_table():
    """Return a DataFrame of static thermocouple metadata: name, x, y, z,
    core_location, scanner. Pulled from ThermocoupleTable in lsst.ts.xml.
    Empty DF if utils aren't available.
    """
    if not HAS_M1M3_UTILS:
        return pd.DataFrame()
    try:
        from lsst.ts.xml.tables.m1m3 import ThermocoupleTable
    except ImportError:
        return pd.DataFrame()
    rows = []
    for tc in ThermocoupleTable:
        rows.append({
            'name': getattr(tc, 'name', None),
            'x': getattr(tc, 'x_position', np.nan),
            'y': getattr(tc, 'y_position', np.nan),
            'z': getattr(tc, 'z_position', np.nan),
            'core_location': str(getattr(tc, 'core_location', '')),
            'scanner': int(getattr(tc, 'scanner', -1)) if getattr(tc, 'scanner', None) is not None else -1,
        })
    return pd.DataFrame(rows)


async def get_m1m3_data(efd_client, visit_table,
                        include_thermocouples=True,
                        include_cell_gradients=True):
    """Get M1M3 thermal data interpolated to observation times.

    Parameters
    ----------
    efd_client : EfdClient
    visit_table : DataFrame
        Must contain obs_start column (TAI isot strings).
    include_thermocouples : bool
        If True, include per-thermocouple temperature columns
        (prefix `m1m3_tc_<name>`, e.g. `m1m3_tc_MTC001B`).
    include_cell_gradients : bool
        If True, include per-cell front-back gradient columns
        (prefix `m1m3_dt_<name>`).

    Returns
    -------
    DataFrame indexed like visit_table, with at minimum the four
    columns x_gradient, y_gradient, z_gradient, radial_gradient.
    Plus optional per-thermocouple and per-cell-gradient columns.
    """
    base_cols = {'x_gradient': np.nan, 'y_gradient': np.nan,
                 'z_gradient': np.nan, 'radial_gradient': np.nan}
    if not HAS_M1M3_UTILS:
        print("Warning: lsst.ts.m1m3.utils not available, skipping M1M3 data")
        return pd.DataFrame(base_cols, index=visit_table.index)

    date_strings = Time(
        [str(x) for x in visit_table["obs_start"].values],
        format="isot", scale="tai"
    ).utc.isot
    data_times = pd.to_datetime(date_strings, format="ISO8601", utc=True)
    sorted_data_times = data_times.sort_values()
    start = Time(sorted_data_times[0])
    end = Time(sorted_data_times[-1])
    data_times_int = data_times.astype("int64")

    thermocouples = ThermocoupleAnalysis(efd_client)
    await thermocouples.load(start, end, time_bin=30)

    gradients = thermocouples.xyz_r_gradients
    grad_times = pd.to_datetime(
        gradients.index, format="ISO8601", utc=True
    ).astype("int64")
    t0 = grad_times[0]

    result = {}

    # xyz / r gradients
    for name in ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]:
        result[name] = _interp_dataframe_to_times(
            gradients[[name]], data_times_int, t0)[name]

    # Per-thermocouple temperatures (~100+ columns named after thermocouples)
    if include_thermocouples:
        tc_df = getattr(thermocouples, 'all_thermocouples_dataframe', None)
        if tc_df is not None and len(tc_df.columns) > 0:
            interp = _interp_dataframe_to_times(tc_df, data_times_int, t0)
            for col, vals in interp.items():
                result[f'm1m3_tc_{col}'] = vals
            print(f"  Added {len(tc_df.columns)} per-thermocouple temperature columns")

    # Per-cell vertical (front-back) gradients
    if include_cell_gradients:
        vg_df = getattr(thermocouples, 'vertical_cell_gradient_dataframe', None)
        if vg_df is not None and len(vg_df.columns) > 0:
            interp = _interp_dataframe_to_times(vg_df, data_times_int, t0)
            for col, vals in interp.items():
                result[f'm1m3_dt_{col}'] = vals
            print(f"  Added {len(vg_df.columns)} per-cell vertical gradient columns")

    return pd.DataFrame(result, index=visit_table.index)


# Backward-compatible alias — old name returns just the four scalar gradient
# columns (no per-thermocouple data) so existing callers keep working.
async def get_m1m3_gradients(efd_client, visit_table):
    """Backward-compatible wrapper; returns only the four xyz/r gradient
    columns. New code should use get_m1m3_data() for the per-thermocouple
    temperatures and per-cell gradients."""
    df = await get_m1m3_data(efd_client, visit_table,
                             include_thermocouples=False,
                             include_cell_gradients=False)
    return df[['x_gradient', 'y_gradient', 'z_gradient', 'radial_gradient']]


async def get_thermal_data(consdb_client, efd_client, visit_info,
                           temp_time_window_sec=DEFAULT_TEMP_TIME_WINDOW_SEC):
    """Retrieve all thermal data for visits and return as a DataFrame.

    Queries ESS temperatures, M1M3 gradients, and TMA truss temperatures
    from the EFD, keyed on (day_obs, seq_num).

    Returns DataFrame with 13 thermal columns.
    """
    # Get obs_start/obs_end from ConsDB
    day_obs_list = sorted(set(np.array(visit_info['day_obs'])))
    day_obs_str = ", ".join(str(d) for d in day_obs_list)

    query = f"""
        SELECT e.day_obs, e.seq_num, e.obs_start, e.obs_end
        FROM cdb_lsstcam.exposure e
        WHERE e.day_obs IN ({day_obs_str})
        ORDER BY e.day_obs, e.seq_num
    """
    consdb_df = consdb_client.query(query).to_pandas()
    print(f"ConsDB returned {len(consdb_df)} exposure records for obs times")

    # Build working DataFrame from visit_info
    vi_df = pd.DataFrame({
        'day_obs': np.array(visit_info['day_obs']),
        'seq_num': np.array(visit_info['seq_num']),
    })
    vi_df = vi_df.merge(
        consdb_df[['day_obs', 'seq_num', 'obs_start', 'obs_end']],
        on=['day_obs', 'seq_num'], how='left',
    )
    n_matched = vi_df['obs_start'].notna().sum()
    print(f"Matched obs times for {n_matched}/{len(vi_df)} visits")

    # ESS temperatures
    ess_sensors = {
        "cam_air_temp": 111,
        "m2_air_temp": 112,
        "m1m3_air_temp": 113,
        "outside_temp": 301,
    }

    valid = vi_df.dropna(subset=['obs_start', 'obs_end']).copy()
    unique_visits = valid[['day_obs', 'seq_num', 'obs_start', 'obs_end']].drop_duplicates()
    print(f"Querying EFD temperatures for {len(unique_visits)} visits...")

    temp_records = []
    for _, row in tqdm(unique_visits.iterrows(), total=len(unique_visits)):
        record = {"day_obs": row["day_obs"], "seq_num": row["seq_num"]}
        for name, index in ess_sensors.items():
            record[name] = await get_ess_temperature(
                efd_client, row["obs_start"], row["obs_end"], index,
                post_padding=temp_time_window_sec,
            )
        temp_records.append(record)

    temp_df = pd.DataFrame(temp_records)

    # Delta-T quantities
    temp_df["m2_delta_t"] = temp_df["m2_air_temp"] - temp_df["m1m3_air_temp"]
    temp_df["cam_m1m3_delta_t"] = temp_df["cam_air_temp"] - temp_df["m1m3_air_temp"]
    temp_df["dome_delta_t"] = temp_df["outside_temp"] - temp_df["m1m3_air_temp"]

    for col in ess_sensors:
        n_valid = temp_df[col].notna().sum()
        print(f"  {col}: {n_valid}/{len(temp_df)} valid")

    # M1M3 thermal data: gradients + per-thermocouple temperatures + per-cell
    # vertical gradients. Process per day_obs to avoid connection resets.
    gradient_parts = []
    for day_obs_val in sorted(unique_visits["day_obs"].unique()):
        day_visits = unique_visits[unique_visits["day_obs"] == day_obs_val].reset_index(drop=True)
        print(f"  M1M3 data for day_obs {day_obs_val}: {len(day_visits)} visits")
        try:
            gdf = await get_m1m3_data(efd_client, day_visits,
                                      include_thermocouples=True,
                                      include_cell_gradients=True)
            gdf["day_obs"] = day_visits["day_obs"].values
            gdf["seq_num"] = day_visits["seq_num"].values
            gradient_parts.append(gdf)
        except Exception as e:
            print(f"    WARNING: failed for {day_obs_val}: {e}")
            gdf = pd.DataFrame({
                "x_gradient": np.nan, "y_gradient": np.nan,
                "z_gradient": np.nan, "radial_gradient": np.nan,
                "day_obs": day_visits["day_obs"].values,
                "seq_num": day_visits["seq_num"].values,
            })
            gradient_parts.append(gdf)

    # Concat with sort=False; missing per-tc columns from a failed day become
    # NaN, which is what we want.
    gradient_df = pd.concat(gradient_parts, ignore_index=True, sort=False)

    # TMA truss temperatures
    truss_sensors = {
        "tma_truss_temp_pxpy": ("temperatureItem6", 122),
        "tma_truss_temp_mxmy": ("temperatureItem7", 122),
    }

    truss_records = []
    print(f"Querying TMA truss temperatures...")
    for _, row in tqdm(unique_visits.iterrows(), total=len(unique_visits)):
        record = {"day_obs": row["day_obs"], "seq_num": row["seq_num"]}
        for name, (field, index) in truss_sensors.items():
            record[name] = await get_ess_temperature(
                efd_client, row["obs_start"], row["obs_end"], index,
                field=field, post_padding=temp_time_window_sec,
            )
        truss_records.append(record)
    truss_df = pd.DataFrame(truss_records)

    # Merge all thermal data
    thermal_df = temp_df.merge(gradient_df, on=["day_obs", "seq_num"], how="left")
    thermal_df = thermal_df.merge(truss_df, on=["day_obs", "seq_num"], how="left")

    # Summarize the columns produced. Includes whatever per-thermocouple
    # and per-cell-gradient columns get_m1m3_data brought in.
    thermal_cols = [c for c in thermal_df.columns
                    if c not in ('day_obs', 'seq_num')]
    n_tc = sum(1 for c in thermal_cols if c.startswith('m1m3_tc_'))
    n_dt = sum(1 for c in thermal_cols if c.startswith('m1m3_dt_'))
    n_other = len(thermal_cols) - n_tc - n_dt
    print(f"\nThermal data: {len(thermal_cols)} columns for {len(thermal_df)} visits "
          f"({n_other} scalar, {n_tc} per-thermocouple, "
          f"{n_dt} per-cell-vertical-gradient)")

    return thermal_df


# ============================================================
# Intrinsic Wavefront Model
# ============================================================

def get_intrinsic_map(x, y, camera_id_map, band=BandLabel.LSST_I):
    """Get intrinsic wavefront Zernikes across a focal plane grid.

    Returns (X, Y, zkIntrinsics) where zkIntrinsics is [nZk x nPts] in meters.
    """
    config = CalcZernikesTaskConfig()
    config.estimateZernikes.retarget(EstimateZernikesDanishTask)
    config.donutStampSelector.maxSelect = 20
    config.donutStampSelector.maxFracBadPixels = 2.0e-4
    config.donutStampSelector.useCustomSnLimit = True
    config.donutStampSelector.minSignalToNoise = 100

    binFactor = 2
    config.estimateZernikes.binning = binFactor
    # Project convention: Noll 4-19, 22-26 (skip the spherical-defocus pair
    # 20, 21).  Must stay in lockstep with create_intrinsic_interpolators, which
    # maps these rows back to Noll number positionally.
    nollIndices = np.array(list(range(4, 20)) + list(range(22, 27)))
    config.estimateZernikes.nollIndices = list(nollIndices)
    config.estimateZernikes.lstsqKwargs = {
        'ftol': 1.0e-3, 'xtol': 1.0e-3, 'gtol': 1.0e-3}
    config.estimateZernikes.saveHistory = False

    task = CalcZernikesTask(config=config)

    camName = 'LSSTCam'
    extra_detector_id = 195
    extra_detector_name = camera_id_map[extra_detector_id].getName()
    instrument = getTaskInstrument(
        camName, extra_detector_name,
        task.estimateZernikes.config.instConfigFile,
    )

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    selpts = (R < 1.8)
    X = X[selpts].flatten()
    Y = Y[selpts].flatten()

    nZk = len(nollIndices)
    nPts = len(X)

    zkIntrinsics = np.zeros((nZk, nPts))
    for i in range(nPts):
        x_pt = float(X[i])
        y_pt = float(Y[i])
        zkIntrinsics[:, i] = instrument.getIntrinsicZernikes(
            xAngle=x_pt, yAngle=y_pt,
            defocalType=None,
            band=band, nollIndices=nollIndices,
        )

    return X, Y, zkIntrinsics


def create_intrinsic_interpolators(X, Y, zkIntrinsics):
    """Create interpolation functions for each Zernike term.

    ``iZs`` must match the Noll list and ordering used in get_intrinsic_map
    (Noll 4-19, 22-26, skipping 20/21) — the rows of ``zkIntrinsics`` are
    keyed back to Noll number positionally here.
    """
    iZs = np.array(list(range(4, 20)) + list(range(22, 27)))
    interpolators = {}
    points = np.column_stack([X, Y])
    for i, iZ in enumerate(iZs):
        values = zkIntrinsics[i, :]
        interpolators[iZ] = LinearNDInterpolator(points, values)
    return interpolators


def add_intrinsic_zernikes(aosTable, intrinsic_interpolators, coord_sys):
    """Add model intrinsic Zernikes to the data table and compute residuals."""
    if coord_sys != 'OCS':
        raise NotImplementedError(
            "add_intrinsic_zernikes evaluates an OCS-domain intrinsic "
            "interpolator at thx/thy_{coord_sys}_extra. With coord_sys='CCS' "
            "those are rotator-rotated camera-frame angles, so the "
            "interpolation would be silently wrong. Rotate (thx,thy) to OCS by "
            "-rotator_angle first, or build a CCS-domain interpolator, before "
            "enabling CCS here.")
    zk_col = f'zk_{coord_sys}'
    thx_extra_col = f'thx_{coord_sys}_extra'
    thy_extra_col = f'thy_{coord_sys}_extra'

    thx_deg = np.rad2deg(aosTable[thx_extra_col])
    thy_deg = np.rad2deg(aosTable[thy_extra_col])

    zk_data = np.stack(aosTable[zk_col])
    npts, nZk_data = zk_data.shape
    iZs_data = infer_zernike_indices(nZk_data)

    print(f"Data has {nZk_data} Zernike terms per donut")
    print(f"Zernike indices used: {iZs_data}")

    zk_intrinsic = np.zeros((npts, nZk_data))
    print(f"Interpolating intrinsic Zernikes for {npts} measurements...")
    for i, iZ in enumerate(iZs_data):
        if iZ in intrinsic_interpolators:
            interp_func = intrinsic_interpolators[iZ]
            zk_intrinsic[:, i] = interp_func(thx_deg, thy_deg)
        else:
            print(f"Warning: No interpolator for Z{iZ}, setting to zero")

    aosTable['zk_intrinsic'] = list(zk_intrinsic)
    zk_residual = zk_data - zk_intrinsic
    aosTable['zk_residual'] = list(zk_residual)

    print("Added columns: 'zk_intrinsic', 'zk_residual'")
    return aosTable


# ============================================================
# Merge Helpers
# ============================================================

def _rotator_columns_for(table, rotator_df, want_flagged):
    """Build per-row rotator_angle (and optionally rotator_flagged) arrays
    aligned to ``table``, by matching (day_obs, seq_num) against ``rotator_df``.

    Shared by merge_rotator_to_visit_info and merge_rotator_to_tables so the
    two stay in lockstep.  Returns (rot_angle_col, rot_flagged_col_or_None).
    """
    day_obs_arr = np.array(table['day_obs'])
    seq_num_arr = np.array(table['seq_num'])
    rot_angle_col = np.full(len(table), np.nan)
    rot_flagged_col = (np.zeros(len(table), dtype=bool)
                       if want_flagged else None)
    for _, r in rotator_df.iterrows():
        mask = (day_obs_arr == r['day_obs']) & (seq_num_arr == r['seq_num'])
        rot_angle_col[mask] = r['rotator_angle']
        if want_flagged:
            rot_flagged_col[mask] = r['rotator_flagged']
    return rot_angle_col, rot_flagged_col


def merge_rotator_to_visit_info(rotator_df, visit_info, rotator_threshold):
    """Add rotator_angle + rotator_flagged to visit_info only (no aosTable)."""
    rot_angle_col, rot_flagged_col = _rotator_columns_for(
        visit_info, rotator_df, want_flagged=True)
    visit_info['rotator_angle'] = rot_angle_col
    visit_info['rotator_flagged'] = rot_flagged_col

    n_flagged = int(np.sum(rot_flagged_col))
    print(f"Added rotator columns to visit_info. "
          f"Flagged: {n_flagged} (|angle| > {rotator_threshold} deg)")
    return visit_info


def quality_visit_mask(visit_info,
                       min_donuts_per_visit=DEFAULT_MIN_DONUTS_PER_VISIT,
                       min_detectors_per_visit=DEFAULT_MIN_DETECTORS_PER_VISIT,
                       max_median_blur_arcsec=DEFAULT_MAX_MEDIAN_BLUR_ARCSEC,
                       verbose=True):
    """Boolean per-visit mask for the standard quality cuts.

    The cuts are evaluated against the numeric metric columns recorded by
    `stream_zernikes_to_parquet`:
      * `n_donuts`                    >= min_donuts_per_visit
      * `n_detectors_with_min_donuts` >= min_detectors_per_visit
      * `median_blur_arcsec`          <= max_median_blur_arcsec

    Pass any threshold as None to disable that individual cut. The
    `min_donuts_per_detector` floor used to compute
    `n_detectors_with_min_donuts` is fixed at mktable time and cannot be
    re-tuned without re-running mktable; it's recorded in
    visit_info.meta['min_donuts_per_detector'] for documentation.

    Parameters
    ----------
    visit_info : astropy.table.QTable or pandas.DataFrame
        Visit-level table with the numeric quality metric columns.
    min_donuts_per_visit, min_detectors_per_visit : int or None
    max_median_blur_arcsec : float or None
    verbose : bool
        Print a one-line summary of how many visits each cut drops.

    Returns
    -------
    mask : ndarray of bool, shape (len(visit_info),)
    """
    n = len(visit_info)
    mask = np.ones(n, dtype=bool)

    def _col(name):
        return np.asarray(visit_info[name]) if name in visit_info.colnames \
            else None

    n_donuts = _col('n_donuts')
    n_dets   = _col('n_detectors_with_min_donuts')
    blur     = _col('median_blur_arcsec')

    drops = []
    if min_donuts_per_visit is not None and n_donuts is not None:
        m = n_donuts >= min_donuts_per_visit
        drops.append(('n_donuts >= %d' % min_donuts_per_visit,
                      int((~m).sum())))
        mask &= m
    if min_detectors_per_visit is not None and n_dets is not None:
        m = n_dets >= min_detectors_per_visit
        drops.append(('n_detectors_with_min_donuts >= %d'
                      % min_detectors_per_visit, int((~m).sum())))
        mask &= m
    if max_median_blur_arcsec is not None and blur is not None:
        m = (np.isfinite(blur)) & (blur <= max_median_blur_arcsec)
        drops.append(('median_blur_arcsec <= %.2f' % max_median_blur_arcsec,
                      int((~m).sum())))
        mask &= m

    if verbose:
        print(f"quality_visit_mask: {int(mask.sum())}/{n} visits pass")
        for label, n_dropped in drops:
            print(f"  cut {label}: drops {n_dropped}")

    return mask


def plot_visit_quality_diagnostics(visit_info, output_pdf=None,
                                   min_donuts_per_visit=DEFAULT_MIN_DONUTS_PER_VISIT,
                                   min_detectors_per_visit=DEFAULT_MIN_DETECTORS_PER_VISIT,
                                   max_median_blur_arcsec=DEFAULT_MAX_MEDIAN_BLUR_ARCSEC,
                                   title=None):
    """Three-panel validation plot of per-visit quality metrics.

    Plots, vs visit ordinal (sorted day_obs, seq_num):
      1. n_donuts                        + horizontal line at min_donuts_per_visit
      2. n_detectors_with_min_donuts     + horizontal line at min_detectors_per_visit
      3. median_blur_arcsec              + horizontal line at max_median_blur_arcsec

    Visits that pass all cuts are coloured blue; failing visits red.

    Parameters
    ----------
    visit_info : QTable or DataFrame
    output_pdf : str or Path, optional
        If given, saves the figure to this path.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n = len(visit_info)
    if n == 0:
        print("plot_visit_quality_diagnostics: empty visit_info — skipping")
        return None

    # Sort by (day_obs, seq_num) so the ordinal is monotonic in time-on-sky
    day_obs = np.asarray(visit_info['day_obs'])
    seq_num = np.asarray(visit_info['seq_num'])
    order = np.lexsort((seq_num, day_obs))

    def _ordered_col(name, default=np.nan):
        if name not in visit_info.colnames:
            return np.full(n, default)
        return np.asarray(visit_info[name])[order]

    n_donuts = _ordered_col('n_donuts')
    n_dets = _ordered_col('n_detectors_with_min_donuts')
    blur = _ordered_col('median_blur_arcsec')

    pass_mask = quality_visit_mask(
        visit_info,
        min_donuts_per_visit=min_donuts_per_visit,
        min_detectors_per_visit=min_detectors_per_visit,
        max_median_blur_arcsec=max_median_blur_arcsec,
        verbose=False)[order]

    # Build per-visit marker styles (color = elev, shape = rot, edge = band).
    # Visits failing the quality cuts are drawn at lower alpha so the
    # elev/rot encoding is preserved.
    has_alt = 'alt' in visit_info.colnames
    has_rot = 'rotator_angle' in visit_info.colnames
    has_band = 'band' in visit_info.colnames
    alt_arr = (np.asarray(visit_info['alt'])[order] if has_alt
               else np.full(n, np.nan))
    rot_arr = (np.asarray(visit_info['rotator_angle'])[order] if has_rot
               else np.full(n, np.nan))
    band_arr = (np.asarray(visit_info['band'])[order] if has_band
                else np.array([None] * n, dtype=object))

    classifications = [
        classify_visit(alt_deg=alt_arr[i], rot_deg=rot_arr[i],
                       band=band_arr[i] if has_band else None)
        for i in range(n)
    ]
    bands_seen = sorted({c['band'] for c in classifications if c['band']})
    show_band_legend = any(b for b in bands_seen if b != 'i')

    x = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    def _scatter(ax, y, label_min=None, label_max=None):
        for i in range(n):
            if not np.isfinite(y[i]):
                continue
            cls = classifications[i]
            style = visit_marker_style(elev=cls['elev'], rot=cls['rot'],
                                        band=cls['band'], base_size=7)
            alpha = 0.95 if pass_mask[i] else 0.30
            ax.plot(x[i], y[i], alpha=alpha, **style)

    _scatter(axes[0], n_donuts)
    if min_donuts_per_visit is not None:
        axes[0].axhline(min_donuts_per_visit, ls='--', color='k', lw=1,
                        label=f'min = {min_donuts_per_visit}')
        axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_ylabel('n_donuts per visit')
    axes[0].grid(alpha=0.3)

    _scatter(axes[1], n_dets)
    if min_detectors_per_visit is not None:
        axes[1].axhline(min_detectors_per_visit, ls='--', color='k', lw=1,
                        label=f'min = {min_detectors_per_visit}')
        axes[1].legend(loc='lower right', fontsize=9)
    min_n = visit_info.meta.get('min_donuts_per_detector',
                                DEFAULT_MIN_DONUTS_PER_DETECTOR) \
        if hasattr(visit_info, 'meta') else DEFAULT_MIN_DONUTS_PER_DETECTOR
    axes[1].set_ylabel(f'# detectors with ≥ {min_n} donuts')
    axes[1].grid(alpha=0.3)

    _scatter(axes[2], blur)
    if max_median_blur_arcsec is not None:
        axes[2].axhline(max_median_blur_arcsec, ls='--', color='k', lw=1,
                        label=f'max = {max_median_blur_arcsec:.2f}')
        axes[2].legend(loc='upper right', fontsize=9)
    axes[2].set_ylabel('median blur [arcsec]')
    axes[2].set_xlabel('visit ordinal (sorted by day_obs, seq_num)')
    axes[2].grid(alpha=0.3)

    n_pass = int(pass_mask.sum())
    suptitle = (title or '') + (
        f"  ({n_pass}/{n} pass quality cuts; "
        f"color = elev, shape = rotator, edge = band; "
        f"failing visits drawn at low alpha)")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # If we're saving a PDF, prepend a marker-scheme legend page so the
    # validation document is self-explanatory.
    if output_pdf is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(str(output_pdf)) as pdf:
            legend_fig = markers_legend_figure(
                show_iter_distinction=False,
                show_band_legend=show_band_legend)
            pdf.savefig(legend_fig)
            plt.close(legend_fig)
            pdf.savefig(fig)
        print(f"  Wrote validation plot: {output_pdf}")

    return fig


def merge_rotator_to_tables(rotator_df, aosTable, visit_info, rotator_threshold):
    """Add rotator_angle and rotator_flagged to aosTable and visit_info."""
    rot_angle_col, rot_flagged_col = _rotator_columns_for(
        aosTable, rotator_df, want_flagged=True)
    aosTable['rotator_angle'] = rot_angle_col
    aosTable['rotator_flagged'] = rot_flagged_col

    n_flagged = int(np.sum(rot_flagged_col))
    print(f"Added rotator columns to aosTable. "
          f"Flagged: {n_flagged} (|angle| > {rotator_threshold} deg)")

    # Add rotator_angle to visit_info (angle only, no flag — preserves prior
    # behaviour of this function).
    if visit_info is not None:
        vi_angle, _ = _rotator_columns_for(
            visit_info, rotator_df, want_flagged=False)
        visit_info['rotator_angle'] = vi_angle
        print("Added rotator_angle to visit_info")

    return aosTable, visit_info


def merge_program_reason_to_visit_info(visits_df, visit_info):
    """Merge `science_program` and `reason` from the ConsDB visits dataframe.

    The ConsDB query in `run_mktable` joins ``cdb_lsstcam.visit1`` with
    ``visit1_quicklook`` and pulls every column from `visit1` (`v1.*`).
    `science_program` is always present; `reason` is present in newer
    schema versions.  Whichever subset of those two columns is in
    `visits_df` is copied into `visit_info` keyed on
    `(day_obs, seq_num)`.  Missing rows get an empty string.
    """
    has_program = 'science_program' in visits_df.columns
    has_reason  = 'reason'           in visits_df.columns
    if not (has_program or has_reason):
        print('  (No science_program / reason columns in ConsDB visits — '
              'skipping)')
        return visit_info

    vi_day_obs = np.array(visit_info['day_obs']).astype(int)
    vi_seq_num = np.array(visit_info['seq_num']).astype(int)

    lookup = {}
    for _, row in visits_df.iterrows():
        try:
            d = int(row['day_obs']); s = int(row['seq_num'])
        except Exception:
            continue
        prog = str(row['science_program']) if has_program else ''
        reas = str(row['reason'])           if has_reason  else ''
        lookup[(d, s)] = (prog, reas)

    progs   = []
    reasons = []
    n_found = 0
    for d, s in zip(vi_day_obs, vi_seq_num):
        match = lookup.get((int(d), int(s)))
        if match is not None:
            progs.append(match[0])
            reasons.append(match[1])
            n_found += 1
        else:
            progs.append('')
            reasons.append('')

    added = []
    if has_program:
        visit_info['science_program'] = progs
        added.append('science_program')
    if has_reason:
        visit_info['reason'] = reasons
        added.append('reason')
    print(f'  Added {added} to visit_info: {n_found}/{len(vi_day_obs)} '
          f'visits matched in the ConsDB query result')
    return visit_info


def merge_thermal_to_visit_info(thermal_df, visit_info):
    """Merge thermal columns into visit_info QTable."""
    vi_day_obs = np.array(visit_info['day_obs'])
    vi_seq_num = np.array(visit_info['seq_num'])

    thermal_indexed = thermal_df.set_index(['day_obs', 'seq_num'])
    thermal_cols = [c for c in thermal_df.columns if c not in ('day_obs', 'seq_num')]

    for col in thermal_cols:
        arr = np.full(len(visit_info), np.nan)
        for idx, row in thermal_indexed.iterrows():
            mask = (vi_day_obs == idx[0]) & (vi_seq_num == idx[1])
            arr[mask] = row[col]
        visit_info[col] = arr

    print(f"Added {len(thermal_cols)} thermal columns to visit_info")
    return visit_info


def drop_unwanted_columns(aosTable):
    """Drop columns with _W, _N, _NW, _ra_, _dec_ in the name."""
    drop_cols = [c for c in aosTable.colnames
                 if '_W' in c or '_N' in c or '_NW' in c
                 or '_ra_' in c or '_dec_' in c]
    if drop_cols:
        aosTable.remove_columns(drop_cols)
        print(f"Dropped {len(drop_cols)} columns: {drop_cols}")
    return aosTable


# ============================================================
# Pipeline
# ============================================================

async def run_mktable(
    butler_repo,
    fam_collections,
    day_obs_min=None,
    day_obs_max=None,
    fam_programs=None,
    collection_phrase=None,
    include_versions=False,
    coord_sys='OCS',
    output_dir='output',
    rotator_threshold=DEFAULT_ROTATOR_THRESHOLD,
    fp_radius=DEFAULT_FP_RADIUS,
    fp_nsteps=DEFAULT_FP_NSTEPS,
    intrinsic_band=None,
    min_visits_per_day=DEFAULT_MIN_VISITS_PER_DAY,
    include_thermal=True,
    calc_mean_zernike=False,
    calc_focal_plane=False,
    temp_time_window_sec=DEFAULT_TEMP_TIME_WINDOW_SEC,
    consdb_url=DEFAULT_CONSDB_URL,
    overwrite=False,
    # Per-visit quality cut configuration
    matched_threshold_arcsec=DEFAULT_MATCHED_THRESHOLD_ARCSEC,
    min_donuts_per_visit=DEFAULT_MIN_DONUTS_PER_VISIT,
    min_donuts_per_detector=DEFAULT_MIN_DONUTS_PER_DETECTOR,
    min_detectors_per_visit=DEFAULT_MIN_DETECTORS_PER_VISIT,
    max_median_blur_arcsec=DEFAULT_MAX_MEDIAN_BLUR_ARCSEC,
    workers=1,
    # Legacy support
    prefix=None,
):
    """Run the full Zernike table-building pipeline.

    Parameters
    ----------
    collection_phrase : str, optional
        Override for output filename phrase. If None, auto-parsed from
        collection name.
    include_versions : bool
        If True, include wep/dviz version strings in the output filename.
    calc_mean_zernike : bool
        If True, compute per-visit mean Zernike columns.
    calc_focal_plane : bool
        If True, compute focal plane coordinates (fpx, fpy).
    prefix : str, optional
        Deprecated; use collection_phrase instead.

    Returns (aosTable, visit_info) or (None, None).
    """
    # Setup
    os.environ.setdefault("no_proxy", "")
    if ".consdb" not in os.environ["no_proxy"]:
        os.environ["no_proxy"] += ",.consdb"

    camera = LsstCam.getCamera()
    camera_id_map = camera.getIdMap()
    if consdb_url is None:
        consdb_url = DEFAULT_CONSDB_URL
    # If using an external URL, embed token from ~/.lsst/consdb_token
    if "@" not in consdb_url and "consdb-pq.consdb" not in consdb_url:
        token_file = Path.home() / ".lsst" / "consdb_token"
        if token_file.exists():
            token = token_file.read_text().strip()
            consdb_url = consdb_url.replace("://", f"://user:{token}@", 1)
    consdb_client = ConsDbClient(consdb_url)

    if intrinsic_band is None:
        intrinsic_band = BandLabel.LSST_I

    # Parse collection info for output naming and default date range
    parsed_phrase, parsed_min, parsed_max = parse_collection_info(
        fam_collections, include_versions=include_versions)

    # Resolve collection_phrase: explicit > legacy prefix > auto-parsed
    if collection_phrase is None:
        collection_phrase = prefix if prefix is not None else parsed_phrase

    # Resolve day_obs range: explicit > auto-parsed from collection
    if day_obs_min is None:
        day_obs_min = parsed_min
    if day_obs_max is None:
        day_obs_max = parsed_max
    if day_obs_min is None or day_obs_max is None:
        raise ValueError(
            "day_obs_min and day_obs_max must be specified (either explicitly "
            "or parseable from collection name)")

    # Build output filenames: donuts + visits parquet sidecar.
    # Static thermocouple metadata is shared across all chunks — written
    # once globally to output/m1m3_thermocouples.parquet (not per-chunk).
    os.makedirs(output_dir, exist_ok=True)
    stem = f'{collection_phrase}_{day_obs_min}_{day_obs_max}'
    output_file = f'{output_dir}/{stem}.parquet'
    visits_file = f'{output_dir}/{stem}_visits.parquet'
    tc_meta_file = f'{output_dir}/m1m3_thermocouples.parquet'

    # Refuse to clobber existing output unless explicitly asked
    # (the global thermocouples metadata is excluded — re-writing the same
    # static lookup table is harmless)
    existing = [p for p in (output_file, visits_file)
                if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output file(s) already exist — refusing to overwrite:\n  "
            + "\n  ".join(existing)
            + "\nPass overwrite=True (or --overwrite from the CLI) to replace them.")

    print(f"Pipeline: {collection_phrase} {coord_sys} {day_obs_min}-{day_obs_max}")
    print(f"  Butler: {butler_repo}")
    print(f"  Collections: {fam_collections}")
    print(f"  Options: mean_zk={calc_mean_zernike}, "
          f"fp_coords={calc_focal_plane}, thermal={include_thermal}")
    print(f"  Output: {output_file}")
    if existing:
        print(f"  Overwriting: {existing}")

    # Query ConsDB for visits
    instrument = 'lsstcam'
    visits_query = f'''
        SELECT v1.*, ql.physical_rotator_angle
        FROM cdb_{instrument}.visit1 v1
        LEFT JOIN cdb_{instrument}.visit1_quicklook ql
        ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs >= {day_obs_min} AND v1.day_obs <= {day_obs_max}
    '''
    visits = consdb_client.query(visits_query).to_pandas()
    print(f"Retrieved {len(visits)} visits from ConsDB")

    # Get visit pairs and filter sparse days
    print_band_counts_by_day(visits, fam_programs, 'cwfs')
    visit_pairs = get_visit_pairs_from_consdb(visits, fam_programs, img_type='cwfs')

    day_counts = Counter(d for d, s in visit_pairs)
    sparse_days = {d for d, n in day_counts.items() if n < min_visits_per_day}
    if sparse_days:
        n_before = len(visit_pairs)
        visit_pairs = [(d, s) for d, s in visit_pairs if d not in sparse_days]
        print(f"Removed {len(sparse_days)} day_obs with < {min_visits_per_day} "
              f"visit_pairs ({n_before - len(visit_pairs)} pairs dropped)")

    if len(visit_pairs) == 0:
        print("ERROR: No visit pairs remaining after filtering!")
        return None, None

    # Rotator angles
    rotator_df = await get_rotator_data(
        visits, visit_pairs, butler_repo, rotator_threshold)

    # Stream aggregate Zernikes per-visit to parquet (one row group per visit).
    # Only visit_info is returned in memory; donuts go straight to disk.
    visit_info = stream_zernikes_to_parquet(
        visit_pairs, fam_collections, butler_repo, coord_sys, camera,
        output_file=output_file,
        calc_focal_plane=calc_focal_plane,
        calc_mean_zernike=calc_mean_zernike,
        matched_threshold_arcsec=matched_threshold_arcsec,
        min_donuts_per_detector=min_donuts_per_detector,
        workers=workers)
    if visit_info is None:
        return None, None

    # Merge science_program / reason from the ConsDB visits dataframe so
    # downstream notebooks can filter by program / reason without having
    # to re-query ConsDB.
    visit_info = merge_program_reason_to_visit_info(visits, visit_info)

    # Merge rotator info into visit_info only
    visit_info = merge_rotator_to_visit_info(
        rotator_df, visit_info, rotator_threshold)

    # Thermal data (visit_info only)
    if include_thermal:
        efd_client = None
        try:
            efd_client = makeEfdClient()
            thermal_df = await get_thermal_data(
                consdb_client, efd_client, visit_info,
                temp_time_window_sec=temp_time_window_sec,
            )
            visit_info = merge_thermal_to_visit_info(thermal_df, visit_info)
        except Exception as e:
            print(f"Warning: Could not retrieve thermal data: {e}")
        finally:
            if efd_client is not None:
                await _close_efd_client(efd_client)

    # Compute per-visit quality flag using the standard cuts. Stored as
    # `visit_quality_pass` so downstream code can filter without
    # re-evaluating the metric thresholds.
    print("\n--- Per-visit quality cuts ---")
    pass_mask = quality_visit_mask(
        visit_info,
        min_donuts_per_visit=min_donuts_per_visit,
        min_detectors_per_visit=min_detectors_per_visit,
        max_median_blur_arcsec=max_median_blur_arcsec,
        verbose=True,
    )
    visit_info['visit_quality_pass'] = pass_mask
    visit_info.meta['min_donuts_per_visit'] = (
        int(min_donuts_per_visit) if min_donuts_per_visit is not None else None)
    visit_info.meta['min_detectors_per_visit'] = (
        int(min_detectors_per_visit) if min_detectors_per_visit is not None else None)
    visit_info.meta['max_median_blur_arcsec'] = (
        float(max_median_blur_arcsec) if max_median_blur_arcsec is not None
        else None)

    # Validation plot: n_donuts / n_detectors / median blur vs visit ordinal
    try:
        diag_pdf = f'{output_dir}/{stem}_visit_quality.pdf'
        plot_visit_quality_diagnostics(
            visit_info, output_pdf=diag_pdf,
            min_donuts_per_visit=min_donuts_per_visit,
            min_detectors_per_visit=min_detectors_per_visit,
            max_median_blur_arcsec=max_median_blur_arcsec,
            title=stem,
        )
    except Exception as e:
        print(f"Warning: failed to write quality diagnostics PDF: {e}")

    # Write the visits table as a small sidecar parquet
    visit_info.write(visits_file, format='parquet', overwrite=True)

    # Write static thermocouple metadata (name, x, y, z, scanner) once per
    # mktable run. Useful for spatial analyses of m1m3_tc_<name> columns.
    tc_meta = _thermocouple_metadata_table()
    if len(tc_meta) > 0:
        tc_meta.to_parquet(tc_meta_file, index=False)
        print(f"  thermocouples: {tc_meta_file} — {len(tc_meta)} thermocouples "
              f"(name, x, y, z, scanner)")

    print(f"\nSaved:")
    print(f"  donuts: {output_file} (streamed, one row group per visit)")
    print(f"  visits: {visits_file} — {len(visit_info)} rows, "
          f"{len(visit_info.columns)} columns")

    return None, visit_info
