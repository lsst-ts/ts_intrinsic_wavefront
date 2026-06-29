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
                     instrument="LSSTCam"):
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

    Returns
    -------
    `astropy.table.Table`
        Columns ``x`` (deg), ``y`` (deg), ``Z{j}`` (µm); ``meta["coord_sys"]``
        == ``"CCS"``.
    """
    js = _resolve_js(maps, noll_list)
    thx, thy = _xy_finite(maps)
    columns = {j: np.asarray(maps[f"Z{j}_CCS"], dtype=float) for j in js}
    if piston_z4_um:
        if 4 not in columns:
            raise ValueError("piston_z4_um given but Z4 is not among the "
                             "selected Noll indices")
        columns[4] = columns[4] + float(piston_z4_um)
    extra = {"instrument": instrument, "piston_z4_um": float(piston_z4_um)}
    if detector is not None:
        extra["detector"] = int(detector)
    return _build_table(thx, thy, columns, js, "CCS", dict(maps.meta),
                        extra_meta=extra)
