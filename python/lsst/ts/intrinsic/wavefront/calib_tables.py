# This file is part of ts_intrinsic_wavefront.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Build ``lsst.ip.isr.IntrinsicZernikes`` *source* tables from MIW maps.

The ``intrinsic_split_maps.parquet`` handoff (columns ``thx_deg``, ``thy_deg``,
``Z{j}_OCS``, ``Z{j}_CCS`` in µm on the rot≈0 disk grid) is the measured
intrinsic wavefront (MIW).  ``lsst.ip.isr.IntrinsicZernikes`` consumes
per-coordinate-system astropy ``Table`` objects with columns ``x``, ``y``
(angular) and ``Z{j}`` (length) plus a ``coord_sys`` (``"CCS"``/``"OCS"``)
``meta`` key — see ``IntrinsicZernikes.fromTable`` / ``_unpackTable`` in
``ip_isr``.

The two systems are split by their physical nature, which also matches how
``IntrinsicZernikes.getIntrinsicZernikes`` queries them:

* **OCS** — the telescope-fixed intrinsics, defined in optical coordinates.  At
  query time the field point is *rotated* by the rotator angle before the OCS
  interpolation, so the query can land anywhere on the focal plane → the OCS
  table must hold the **whole** map.  It is detector- *and* filter-independent,
  so it is written **once per instrument**.
* **CCS** — the camera-fixed focal-plane contribution: the smooth camera field
  ``C`` plus the per-CCD focal-plane **height** (a Z4 piston, ``ccd_height``).
  The CCS query point is never rotated — it stays within the detector — so each
  detector only needs its own value, and the height makes it genuinely
  **per-detector**.  One CCS table is written per detector.

This module builds the tables from in-memory arrays (astropy + numpy only).  The
per-CCD height piston itself is computed by the caller (it needs ``cameraGeom``
and the metrology map; see ``bin.src/run_make_calib_tables.py``) and passed in
here as a scalar µm offset added to the CCS ``Z4`` column.
"""

__all__ = [
    "COORD_SYS",
    "LSSTCAM_PHYSICAL_FILTERS",
    "maps_noll_indices",
    "ocs_source_table",
    "ccs_source_table",
    "ccs_source_table_on_grid",
]

import re

import numpy as np
from astropy import units as u
from astropy.table import Table

# LSSTCam on-sky band -> physical_filter, from obs_lsst LSSTCAM_FILTER_DEFINITIONS
# (python/lsst/obs/lsst/filters.py).  Used at ingest time to replicate the
# filter-independent tables across the physical_filter dataId.
LSSTCAM_PHYSICAL_FILTERS = {
    "u": "u_24",
    "g": "g_6",
    "r": "r_57",
    "i": "i_39",
    "z": "z_20",
    "y": "y_10",
}

COORD_SYS = ("CCS", "OCS")


def maps_noll_indices(maps):
    """Return the sorted Noll indices present in both coordinate systems.

    ``IntrinsicZernikes`` requires the CCS and OCS systems to describe the same
    Noll indices, so only indices that appear as **both** ``Z{j}_OCS`` and
    ``Z{j}_CCS`` columns are usable.

    Parameters
    ----------
    maps : `astropy.table.Table`
        ``intrinsic_split_maps.parquet`` contents.

    Returns
    -------
    `list` [`int`]
        Sorted shared Noll indices.
    """
    js_ocs = {int(m.group(1)) for c in maps.colnames
              for m in [re.match(r"Z(\d+)_OCS$", c)] if m}
    js_ccs = {int(m.group(1)) for c in maps.colnames
              for m in [re.match(r"Z(\d+)_CCS$", c)] if m}
    return sorted(js_ocs & js_ccs)


def _resolve_js(maps, noll_list):
    js = maps_noll_indices(maps)
    if noll_list is None:
        if not js:
            raise ValueError("no shared Z{j}_OCS / Z{j}_CCS columns found in maps")
        return js
    want = [int(j) for j in noll_list]
    missing = [j for j in want if j not in js]
    if missing:
        raise ValueError(
            f"requested Noll indices not present in both systems: {missing}")
    return want


def _xy_finite(maps):
    thx = np.asarray(maps["thx_deg"], dtype=float)
    thy = np.asarray(maps["thy_deg"], dtype=float)
    return thx, thy


def _build_table(thx, thy, columns, js, coord_sys, base_meta, extra_meta=None):
    """Assemble a source table from a {j: values} dict, dropping NaN rows.

    A row is kept only if all of its retained ``Z{j}`` values (and x, y) are
    finite — a NaN sample would corrupt the calibration's interpolator.
    """
    vals = np.column_stack([np.asarray(columns[j], dtype=float) for j in js])
    keep = np.isfinite(thx) & np.isfinite(thy) & np.isfinite(vals).all(axis=1)
    data = {"x": thx[keep] * u.deg, "y": thy[keep] * u.deg}
    for k, j in enumerate(js):
        data[f"Z{j}"] = vals[keep, k] * u.um
    table = Table(data)
    meta = dict(base_meta)
    meta["coord_sys"] = coord_sys
    if extra_meta:
        meta.update(extra_meta)
    table.meta = meta
    return table


def ocs_source_table(maps, noll_list=None, instrument="LSSTCam"):
    """Build the single per-instrument OCS source table from the maps.

    Parameters
    ----------
    maps : `astropy.table.Table`
        MIW maps table (uses the ``Z{j}_OCS`` columns).
    noll_list : `list` [`int`], optional
        Subset of Noll indices; default all shared (see `maps_noll_indices`).
    instrument : `str`, optional
        Recorded in ``meta``.

    Returns
    -------
    `astropy.table.Table`
        Columns ``x`` (deg), ``y`` (deg), ``Z{j}`` (µm); ``meta["coord_sys"]``
        == ``"OCS"``.
    """
    js = _resolve_js(maps, noll_list)
    thx, thy = _xy_finite(maps)
    columns = {j: maps[f"Z{j}_OCS"] for j in js}
    return _build_table(thx, thy, columns, js, "OCS", dict(maps.meta),
                        extra_meta={"instrument": instrument})


def ccs_source_table(maps, detector=None, piston_z4_um=0.0, noll_list=None,
                     instrument="LSSTCam", zero_camera_field=False):
    """Build a per-detector CCS source table from the maps.

    The CCS values are the smooth camera field ``C`` (the ``Z{j}_CCS`` columns)
    with the per-CCD focal-plane height added to ``Z4`` as a piston offset.

    Parameters
    ----------
    maps : `astropy.table.Table`
        MIW maps table (uses the ``Z{j}_CCS`` columns).
    detector : `int`, optional
        Detector id; recorded in ``meta`` for traceability.
    piston_z4_um : `float`, optional
        Height→Z4 piston (µm) for this detector, added to the ``Z4`` column.
        ``0`` leaves the smooth camera field unchanged (e.g. when no metrology
        map is available).
    noll_list : `list` [`int`], optional
        Subset of Noll indices; default all shared.
    instrument : `str`, optional
        Recorded in ``meta``.
    zero_camera_field : `bool`, optional
        If ``True``, zero the smooth camera field ``C`` and keep only the ``Z4``
        height piston (all other Noll become 0).  Used for the field-edge CCDs
        that sit past the data: there is no measured camera field for them, so
        it is more honest to store the height-only term than a whole-field
        placeholder.  Default ``False``.

    Returns
    -------
    `astropy.table.Table`
        Columns ``x`` (deg), ``y`` (deg), ``Z{j}`` (µm); ``meta["coord_sys"]``
        == ``"CCS"``.
    """
    js = _resolve_js(maps, noll_list)
    thx, thy = _xy_finite(maps)
    if zero_camera_field:
        columns = {j: np.zeros(len(thx), dtype=float) for j in js}
    else:
        columns = {j: np.asarray(maps[f"Z{j}_CCS"], dtype=float) for j in js}
    if piston_z4_um:
        if 4 not in columns:
            raise ValueError("piston_z4_um given but Z4 is not among the "
                             "selected Noll indices")
        columns[4] = columns[4] + float(piston_z4_um)
    extra = {"instrument": instrument, "piston_z4_um": float(piston_z4_um),
             "camera_field": (not zero_camera_field)}
    if detector is not None:
        extra["detector"] = int(detector)
    return _build_table(thx, thy, columns, js, "CCS", dict(maps.meta),
                        extra_meta=extra)


# Per-(maps, noll) CCS interpolators are expensive to build (a Delaunay
# triangulation of the whole field grid per Noll) and identical for every
# detector, so cache them.  The maps object is held in the value both to key on
# identity and to stop its id() being recycled while cached.
_CCS_INTERP_CACHE = {}


def _ccs_interpolators(maps, js):
    """Build (and cache) a linear interpolator per ``Z{j}_CCS`` map column."""
    from scipy.interpolate import LinearNDInterpolator

    key = (id(maps), tuple(js))
    cached = _CCS_INTERP_CACHE.get(key)
    if cached is not None and cached[0] is maps:
        return cached[1]
    thx, thy = _xy_finite(maps)
    good = np.isfinite(thx) & np.isfinite(thy)
    pts = np.column_stack([thx[good], thy[good]])
    interps = {}
    for j in js:
        v = np.asarray(maps[f"Z{j}_CCS"], dtype=float)[good]
        m = np.isfinite(v)
        interps[j] = LinearNDInterpolator(pts[m], v[m])
    _CCS_INTERP_CACHE[key] = (maps, interps)
    return interps


def _interp_ccs_map(maps, js, x_deg, y_deg):
    """Interpolate the smooth CCS camera field ``Z{j}_CCS`` onto (x, y) points.

    Uses a linear (barycentric) interpolator over the maps' own field grid.
    Points that fall **outside** the maps' convex hull (i.e. past the data
    support) get NaN — never an extrapolated value — so the caller drops them
    rather than inventing a camera field where there was no data.

    Parameters
    ----------
    maps : `astropy.table.Table`
        MIW maps table (``thx_deg``, ``thy_deg``, ``Z{j}_CCS``).
    js : `list` [`int`]
        Noll indices to interpolate.
    x_deg, y_deg : `numpy.ndarray`
        Target field positions in degrees, in the **CCS** frame (same
        convention as ``thx_deg``/``thy_deg``).

    Returns
    -------
    `dict` [`int`, `numpy.ndarray`]
        ``{j: values at (x, y)}`` (µm), NaN outside the map support.
    """
    interps = _ccs_interpolators(maps, js)
    xi = np.column_stack([np.asarray(x_deg, float), np.asarray(y_deg, float)])
    return {j: interps[j](xi) for j in js}


def ccs_source_table_on_grid(maps, x_deg, y_deg, height_z4_um=None,
                             detector=None, noll_list=None, instrument="LSSTCam"):
    """Build a per-detector CCS source table sampled on an explicit point grid.

    Unlike `ccs_source_table` (which reuses the whole-focal-plane maps grid and
    adds a single scalar Z4 piston), this samples the smooth camera field ``C``
    at the caller-supplied ``(x, y)`` points — a per-detector footprint grid —
    and adds a **per-point** Z4 height, so the CCD's intra-detector height
    structure is preserved instead of collapsed to its centre value.

    Points outside the maps' data support interpolate to NaN and are dropped
    (no extrapolation).

    Parameters
    ----------
    maps : `astropy.table.Table`
        MIW maps table (uses the ``Z{j}_CCS`` columns).
    x_deg, y_deg : `array-like`
        Sample positions in degrees, in the **CCS** frame.
    height_z4_um : `array-like`, optional
        Per-point Z4 height contribution (µm), same length as ``x_deg``; added
        to the ``Z4`` column.  ``None`` leaves the smooth field unchanged.
    detector : `int`, optional
        Detector id; recorded in ``meta``.
    noll_list : `list` [`int`], optional
        Subset of Noll indices; default all shared.
    instrument : `str`, optional
        Recorded in ``meta``.

    Returns
    -------
    `astropy.table.Table`
        Columns ``x`` (deg), ``y`` (deg), ``Z{j}`` (µm); ``meta["coord_sys"]``
        == ``"CCS"``.  Rows with any non-finite value are dropped.
    """
    js = _resolve_js(maps, noll_list)
    x = np.asarray(x_deg, dtype=float)
    y = np.asarray(y_deg, dtype=float)
    columns = _interp_ccs_map(maps, js, x, y)
    if height_z4_um is not None:
        if 4 not in columns:
            raise ValueError("height_z4_um given but Z4 is not among the "
                             "selected Noll indices")
        columns[4] = columns[4] + np.asarray(height_z4_um, dtype=float)
    extra = {"instrument": instrument}
    if detector is not None:
        extra["detector"] = int(detector)
    if height_z4_um is not None:
        # height is applied PER POINT here (not a scalar piston); record only the
        # mean as a diagnostic summary of the CCD's height field.
        extra["mean_z4_height_um"] = float(np.nanmean(np.asarray(height_z4_um, float)))
    return _build_table(x, y, columns, js, "CCS", dict(maps.meta),
                        extra_meta=extra)
