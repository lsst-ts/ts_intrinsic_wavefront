"""CCD focal-plane height map helpers.

Reads the LSST focal-plane height-map FITS file produced by SLAC metrology
(per-sensor BinTableHDUs with X_CCS / Y_CCS / Z_CCS_MEASURED / Z_CCS_MODEL),
provides a KNN interpolator height(fpx, fpy), evaluates per-donut
focal-plane coordinates via cameraGeom, and converts a local piston
(height) into a defocus-Zernike (Z4) contribution using the empirical
`HEIGHT_TO_Z4_UM_PER_MM` = 15 μm/mm constant (Guillem's estimate).

These helpers were originally inline in ``aos/intrinsics_checkZ4.ipynb``;
they are extracted here so that other notebooks (e.g. the
build_measured_intrinsic flow) can import them.

Notes
-----
* The FITS reader applies the X<->Y swap from the CCS frame to the
  focal-plane convention: ``fpx = Y_CCS`` and ``fpy = X_CCS``.  This
  mirrors the convention used elsewhere in the repo (e.g.
  ``plot_Z_FAM_August-archive``).
* `get_height_interpolator` returns a callable that takes (fpx, fpy)
  arrays in mm and returns the height in whatever units `Z_CCS_*` stores
  in the FITS file.  Aaron's existing code labels that variable as
  ``height_mm`` and applies the 15 μm/mm constant directly, so we keep
  that interpretation here.
"""

import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


# Empirical conversion: μm of Z4 wavefront defocus per mm of local piston.
# Source: Guillem's estimate (≈ 0.15 μm Z4 per 10 μm of local height).
HEIGHT_TO_Z4_UM_PER_MM = 15.0

# Orientation mapping cameraGeom focal-plane mm -> the batoid_rubin per-detector
# Bicubic height-map frame, as (swap_xy, sign_x, sign_y).  ESTABLISHED (not guessed)
# by comparing batoid_rubin .sag against the LSST_FP_cold metrology surface over the
# 4 corner-WFS rafts + a science CCD (R22S11): swap=True,+1,+1 gave 12415/12415 points
# in-domain and the lowest RMS (1.0 μm).  See aos/wfs_ccd_height_compare.* .
BATOID_FP_ORIENTATION = (True, 1, 1)

# Corner-WFS nominal piston (mm): SW0 extra-focal +, SW1 intra-focal -.  Removed from
# the metrology surface so its height is the figure deviation (consistent with the
# at-focus science CCDs).  Verified by the same metrology-vs-batoid comparison.
WFS_DEFOCAL_MM = 1.5


# ----------------------------------------------------------------------
# Pixel -> focal-plane coordinates (per donut)
# ----------------------------------------------------------------------

def compute_fp_coords(donut_df, camera,
                      x_col='centroid_x_intra',
                      y_col='centroid_y_intra',
                      det_col='detector'):
    """Per-donut focal-plane (mm) coordinates via cameraGeom.

    Groups donuts by detector and calls ``common.camera_utils.pixel_to_focal``
    once per sensor (vectorised over all donuts on that sensor).

    Parameters
    ----------
    donut_df : pandas.DataFrame or astropy.QTable
        Must contain `x_col`, `y_col`, and `det_col`.
    camera : lsst.afw.cameraGeom.Camera
    x_col, y_col, det_col : str

    Returns
    -------
    fpx_mm, fpy_mm : ndarray
    """
    # Lazy import — the LSST stack is only available on RSP
    from lsst.ts.intrinsic.wavefront.common.camera_utils import pixel_to_focal

    det_names = np.asarray(donut_df[det_col]).astype(str)
    x_pix = np.asarray(donut_df[x_col], dtype=float)
    y_pix = np.asarray(donut_df[y_col], dtype=float)
    fpx = np.full_like(x_pix, np.nan)
    fpy = np.full_like(y_pix, np.nan)
    for det in camera:
        name = det.getName()
        mask = (det_names == name)
        if not np.any(mask):
            continue
        fx, fy = pixel_to_focal(x_pix[mask], y_pix[mask], det)
        fpx[mask] = fx
        fpy[mask] = fy
    return fpx, fpy


# ----------------------------------------------------------------------
# Height-map I/O + KNN interpolator
# ----------------------------------------------------------------------

def make_metrology_table(file, rsid=None, include_wfs=True):
    """Build a per-point focal-plane metrology table from the height map.

    Each per-sensor BinTableHDU is concatenated into one table with
    columns ``fpx, fpy, z_mod, z_meas, det``.  Note the x/y swap from
    CCS to focal-plane convention: ``fpx = Y_CCS``, ``fpy = X_CCS``.

    Parameters
    ----------
    file : str or Path
        FITS file path (e.g. ``LSST_FP_cold_b_measurement_4col_bysurface.fits``).
    rsid : str, optional
        Restrict to a single raft-sensor identifier (e.g. ``'R22S11'``).

    Returns
    -------
    astropy.table.Table
    """
    rows = []
    with fits.open(str(file)) as hdulist:
        for hdu in hdulist:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            extname = hdu.header.get('EXTNAME', '')
            if rsid is not None and extname != rsid:
                continue
            # science sensor 'R##S##'  or  corner-WFS 'R##WFS0/1' (det 'R##_SW0/1').
            # The corner WFS sit at the nominal +/-WFS_DEFOCAL_MM piston, removed here
            # so z is the figure deviation like the at-focus science CCDs.
            m_sci = re.fullmatch(r'R\d\dS\d\d', extname)
            m_wfs = re.fullmatch(r'R(\d\d)WFS([01])', extname)
            if m_sci:
                det_label, piston = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname), 0.0
            elif include_wfs and m_wfs:
                det_label = f'R{m_wfs.group(1)}_SW{m_wfs.group(2)}'
                piston = +WFS_DEFOCAL_MM if m_wfs.group(2) == '0' else -WFS_DEFOCAL_MM
            elif rsid is not None:           # explicit rsid that isn't sci/wfs -> take as-is
                det_label, piston = extname, 0.0
            else:
                continue
            tab = Table(hdu.data)
            for x, y, z_mod, z_meas in zip(
                tab['X_CCS'], tab['Y_CCS'],
                tab['Z_CCS_MODEL'], tab['Z_CCS_MEASURED'],
            ):
                rows.append([y, x, z_mod - piston, z_meas - piston, det_label])
    return Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])


def get_height_interpolator(metrology_table, k=3,
                            weight_type='distance',
                            ztype='measured'):
    """KNN interpolator for focal-plane height.

    Returns ``interp_func(fpx, fpy)`` which accepts ndarrays and returns
    the height (in the FITS file's native Z_CCS unit; see module
    docstring).
    """
    from sklearn.neighbors import KNeighborsRegressor  # local import

    x = np.column_stack((metrology_table['fpx'], metrology_table['fpy']))
    if ztype == 'measured':
        y = np.asarray(metrology_table['z_meas'], dtype=float)
    elif ztype == 'model':
        y = np.asarray(metrology_table['z_mod'], dtype=float)
    else:
        raise ValueError("ztype must be 'measured' or 'model'")
    knn = KNeighborsRegressor(n_neighbors=k, weights=weight_type)
    knn.fit(x, y)

    def interp_func(fpx, fpy):
        fpx = np.atleast_1d(np.asarray(fpx, dtype=float))
        fpy = np.atleast_1d(np.asarray(fpy, dtype=float))
        pts = np.column_stack((fpx, fpy))
        return knn.predict(pts)

    return interp_func


def height_to_z4(height_mm, factor=HEIGHT_TO_Z4_UM_PER_MM):
    """Convert local CCD piston (mm) to a defocus-Z4 contribution (μm)."""
    return factor * np.asarray(height_mm, dtype=float)


# ----------------------------------------------------------------------
# Per-CCD x<->y transpose helpers (used to mirror the pipeline's
# 'intrinsic_transpose_bug' when computing the height contribution that
# matches the *intrinsic* Zernike tabulation rather than the data.)
# ----------------------------------------------------------------------

def ccd_centers_fp(camera):
    """Return {detector_name: (cx_mm, cy_mm)} for every detector in `camera`."""
    from lsst.ts.intrinsic.wavefront.common.camera_utils import pixel_to_focal

    centers = {}
    for det in camera:
        c = det.getBBox().getCenter()
        cx, cy = pixel_to_focal(np.array([c.getX()]),
                                np.array([c.getY()]), det)
        centers[det.getName()] = (float(cx[0]), float(cy[0]))
    return centers


def transpose_around_ccd_centers(fpx_mm, fpy_mm, det_names, camera):
    """Per-CCD x<->y transpose around each detector's center.

    Mirrors the pipeline's per-CCD intrinsic Zernike calculation bug:
    relative to the detector center, the x and y offsets are swapped.

    Parameters
    ----------
    fpx_mm, fpy_mm : ndarray (n_donuts,)
        Focal-plane coordinates (already in DVCS mm) for every donut.
    det_names : array-like (n_donuts,) of str
    camera : lsst.afw.cameraGeom.Camera

    Returns
    -------
    fpx_swap, fpy_swap : ndarray (n_donuts,)
    """
    centers = ccd_centers_fp(camera)
    fpx_arr = np.asarray(fpx_mm, dtype=float)
    fpy_arr = np.asarray(fpy_mm, dtype=float)
    fpx_swap = np.full_like(fpx_arr, np.nan)
    fpy_swap = np.full_like(fpy_arr, np.nan)
    det_arr = np.asarray(det_names).astype(str)
    for name, (cx, cy) in centers.items():
        mask = (det_arr == name)
        if not np.any(mask):
            continue
        dx = fpx_arr[mask] - cx
        dy = fpy_arr[mask] - cy
        fpx_swap[mask] = cx + dy
        fpy_swap[mask] = cy + dx
    return fpx_swap, fpy_swap


# ----------------------------------------------------------------------
# batoid_rubin CCD-height source (ts_wep's current source)
# ----------------------------------------------------------------------

def _resolve_ccd_height_map_dir(height_map_dir):
    """Resolve the directory that directly contains ccd_height_map.fits.gz.

    Accepts ``~`` paths.  When `height_map_dir` is None, falls back to
    ``batoid_rubin.utils.ensure_data_dir('ccd_height_map')``.  If the
    given dir doesn't hold the file directly, also tries a
    ``ccd_height_map`` subdirectory (the layout produced by
    ``download_rubin_data.py --outdir <dir> ccd_height_map``).
    """
    if height_map_dir is None:
        from batoid_rubin.utils import ensure_data_dir
        return str(ensure_data_dir('ccd_height_map'))
    p = Path(height_map_dir).expanduser()
    if (p / 'ccd_height_map.fits.gz').exists():
        return str(p)
    if (p / 'ccd_height_map' / 'ccd_height_map.fits.gz').exists():
        return str(p / 'ccd_height_map')
    return str(p)   # as given — let det_height_maps raise a clear error


def _sag_oriented(maps, fpx_mm, fpy_mm, det_names, swap, sx, sy):
    """Evaluate per-detector batoid Bicubic .sag (height mm) at an
    oriented focal-plane position.  `(swap, sx, sy)` map cameraGeom's
    focal-plane mm into the batoid map frame: optional x<->y swap and
    per-axis sign.  Returns NaN where the detector has no map or the
    point is outside the map domain."""
    if swap:
        u, v = sx * fpy_mm, sy * fpx_mm
    else:
        u, v = sx * fpx_mm, sy * fpy_mm
    h = np.full(len(det_names), np.nan, dtype=float)
    for det in np.unique(det_names):
        if det not in maps:
            continue
        m = (det_names == det)
        h[m] = np.asarray(maps[det].sag(u[m] * 1e-3, v[m] * 1e-3),
                          dtype=float) * 1e3        # m -> mm
    return h


def compute_ccd_heights(donut_df, camera, source='batoid_rubin',
                        height_map_dir=None, metrology_fits=None,
                        factor=HEIGHT_TO_Z4_UM_PER_MM, det_col='detector'):
    """Single entry point for per-donut CCD heights + the Z4 contribution.

    The focal-plane position is computed from the **intra-** and
    **extra-focal centroids** via cameraGeom, the chosen height model is
    evaluated at each, and the two are averaged (NaN-aware).

    Parameters
    ----------
    donut_df : DataFrame/QTable with centroid_x/y_intra, centroid_x/y_extra,
        and `det_col`.
    camera : lsst.afw.cameraGeom.Camera
    source : {'batoid_rubin', 'metrology'}
        'batoid_rubin' uses ``batoid_rubin.builder.det_height_maps`` (the
        per-detector Bicubic maps ts_wep applies); 'metrology' uses the
        LSST_FP_cold_b metrology FITS via the KNN interpolator.
    height_map_dir : str or None
        ccd_height_map dir for the batoid source (None -> ensure_data_dir).
    metrology_fits : str or None
        FITS path for the metrology source.
    factor : float
        μm of Z4 per mm of height.

    Returns
    -------
    dict of equal-length ndarrays:
        ccd_height_intra, ccd_height_extra, ccd_height_mean  [mm]
        Z4_height  =  factor * ccd_height_mean               [μm]
    """
    fpx_i, fpy_i = compute_fp_coords(donut_df, camera,
                                     'centroid_x_intra', 'centroid_y_intra',
                                     det_col)
    fpx_e, fpy_e = compute_fp_coords(donut_df, camera,
                                     'centroid_x_extra', 'centroid_y_extra',
                                     det_col)
    det_names = np.asarray(donut_df[det_col]).astype(str)

    if source == 'metrology':
        if not metrology_fits:
            raise ValueError("source='metrology' requires metrology_fits")
        interp = get_height_interpolator(make_metrology_table(metrology_fits))
        h_i = np.asarray(interp(fpx_i, fpy_i), dtype=float)
        h_e = np.asarray(interp(fpx_e, fpy_e), dtype=float)
    elif source == 'batoid_rubin':
        from batoid_rubin.builder import det_height_maps
        d = _resolve_ccd_height_map_dir(height_map_dir)
        print(f'  ccd_height_map dir: {d}')
        maps = det_height_maps(d)
        missing = sorted(set(det_names) - {str(k) for k in maps})
        if missing:
            print(f'  (no batoid height map for {len(missing)} detector(s): '
                  f'{missing[:5]}{"..." if len(missing) > 5 else ""})')
        swap, sx, sy = BATOID_FP_ORIENTATION       # established by metrology comparison
        h_i = _sag_oriented(maps, fpx_i, fpy_i, det_names, swap, sx, sy)
        h_e = _sag_oriented(maps, fpx_e, fpy_e, det_names, swap, sx, sy)
    else:
        raise ValueError(f'unknown height source {source!r}')

    with np.errstate(invalid='ignore'):
        h_mean = np.nanmean(np.vstack([h_i, h_e]), axis=0)
    z4 = factor * h_mean
    n_fin = int(np.isfinite(z4).sum())
    print(f'  CCD heights ({source}): intra/extra average -> '
          f'Z4_height n_finite={n_fin}/{len(z4)}  '
          f'(factor={factor:g} μm/mm)')
    return {'ccd_height_intra': h_i, 'ccd_height_extra': h_e,
            'ccd_height_mean': h_mean, 'Z4_height': z4}


def add_ccd_height_to_parquet(in_path, out_path=None, source='batoid_rubin',
                              camera=None, height_map_dir=None,
                              metrology_fits=None,
                              factor=HEIGHT_TO_Z4_UM_PER_MM):
    """Read a donut parquet, add ccd_height_intra/extra/mean + Z4_height
    columns, and write it back out (default: ``<stem>_heights.parquet``).

    Reads/writes a flat table — does NOT preserve the per-visit
    row-group layout — so use the result where whole-table loads are
    fine (e.g. build_measured_intrinsic), not the streaming consumers.
    Returns the output path.
    """
    import pandas as pd
    if camera is None:
        from lsst.obs.lsst import LsstCam
        camera = LsstCam.getCamera()
    df = pd.read_parquet(in_path)
    cols = compute_ccd_heights(df, camera, source=source,
                               height_map_dir=height_map_dir,
                               metrology_fits=metrology_fits, factor=factor)
    for k, v in cols.items():
        df[k] = v
    if out_path is None:
        p = Path(in_path)
        out_path = str(p.with_name(p.stem + '_heights' + p.suffix))
    df.to_parquet(out_path)
    print(f'Wrote {out_path}  (+{len(cols)} height columns)')
    return out_path


def batoid_rubin_height_per_donut(donut_df, camera, height_map_dir=None,
                                  det_col='detector'):
    """Back-compat thin wrapper: the intra/extra-averaged batoid height
    (mm) from :func:`compute_ccd_heights`."""
    return compute_ccd_heights(
        donut_df, camera, source='batoid_rubin',
        height_map_dir=height_map_dir, det_col=det_col)['ccd_height_mean']


# ----------------------------------------------------------------------
# Corner-WFS field-angle coverage (for radial-shell definitions)
# ----------------------------------------------------------------------

def wfs_field_radius_range(camera,
                           sensors=('R00_SW0', 'R04_SW0',
                                    'R40_SW0', 'R44_SW0')):
    """Return (r_min_deg, r_max_deg): the inner/outer field-angle radius
    spanned by the given corner-WFS sensors, from cameraGeom.

    Uses each sensor's PIXELS -> FIELD_ANGLE transform on its bbox
    corners.  ``SW0`` are the inner halves of the corner CCDs, so the
    minimum corner radius over the default SW0 sensors is the inner
    edge of the wavefront-sensor field coverage.

    Requires the LSST stack (lsst.afw.cameraGeom) — RSP only.
    """
    from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
    rmins, rmaxs = [], []
    for name in sensors:
        det = camera[name]
        tr = det.getTransform(PIXELS, FIELD_ANGLE)
        rad = []
        for c in det.getCorners(PIXELS):
            fa = tr.applyForward(c)
            rad.append(float(np.hypot(fa.getX(), fa.getY())))
        rmins.append(min(rad))
        rmaxs.append(max(rad))
    return float(np.degrees(min(rmins))), float(np.degrees(max(rmaxs)))
