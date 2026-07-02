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

"""Batoid design-intrinsic OCS Zernikes to backfill Noll indices the MIW lacks.

The measured intrinsic wavefront (MIW) maps only carry the Noll indices the
donut wavefront estimation solved for.  The telescope-fixed **OCS** intrinsic
for the *unmeasured* Noll indices is well described by the batoid optical design,
so this module evaluates that design intrinsic on the MIW field grid (via
``batoid_rubin``'s ``LSSTBuilder`` and the Rubin **v3.14** as-built optical
model) and backfills the missing ``Z{j}_OCS`` columns.

The camera-fixed **CCS** term for those Noll is left at **zero** -- there is no
measured camera field for them -- which both matches the pipeline's existing
OCS-only convention (``Z{j}_CCS`` present but zero) and keeps the OCS and CCS
systems on the *identical* Noll set that ``lsst.ip.isr.IntrinsicZernikes``
requires (it rejects tables whose Noll indices disagree).

The recipe mirrors ts_aos_analysis ``generateIntrinsics`` (lsst-sitcom PR 87):
field angles are passed to batoid as ``theta_x = thx_deg``, ``theta_y = thy_deg``
(the MIW map convention, ``thx`` = ``FIELD_ANGLE`` y), a gnomonic projection
(DM's ``FIELD_ANGLE`` convention), and the Zernike *waves* are scaled by the band
wavelength to a physical length.  Unlike ``generateIntrinsics`` -- which builds a
per-detector telescope on a per-detector footprint grid -- the OCS map is
telescope-fixed and detector-independent, so a single ``build()`` is evaluated on
the whole-focal-plane MIW grid.
"""

__all__ = [
    "BAND_WAVELENGTH_M",
    "DEFAULT_OPTICAL_MODEL",
    "batoid_ocs_zernikes",
    "backfill_ocs_from_batoid",
]

import numpy as np

# obs_lsst effective band wavelengths (m); same values as ts_aos_analysis
# generateIntrinsics.  The OCS table is written once (filter-independent), so a
# single band is chosen for the design backfill.
BAND_WAVELENGTH_M = {
    "u": 0.365e-6,
    "g": 0.480e-6,
    "r": 0.620e-6,
    "i": 0.754e-6,
    "z": 0.868e-6,
    "y": 0.973e-6,
}

# batoid optical-model yaml stem.  'Rubin_v3.14' -> Rubin_v3.14_{band}.yaml (the
# latest as-built model); 'LSST' -> LSST_{band}.yaml (the nominal design).
DEFAULT_OPTICAL_MODEL = "Rubin_v3.14"

# batoid.zernike sampling parameters (ts_aos_analysis generateIntrinsics values).
DEFAULT_EPS = 0.612            # pupil obscuration fraction
DEFAULT_NX = 63               # ray grid per side
DEFAULT_PROJECTION = "gnomonic"  # DM's FIELD_ANGLE convention


def batoid_ocs_zernikes(thx_deg, thy_deg, noll_list, band="i",
                        optical_model=DEFAULT_OPTICAL_MODEL, jmax=None,
                        eps=DEFAULT_EPS, nx=DEFAULT_NX,
                        projection=DEFAULT_PROJECTION):
    """Design OCS intrinsic Zernikes (µm) at the given field points.

    Parameters
    ----------
    thx_deg, thy_deg : array-like
        Field angles in degrees in the MIW / OCS map convention (``thx`` =
        ``FIELD_ANGLE`` y, ``thy`` = ``FIELD_ANGLE`` x).  Passed to batoid as
        ``theta_x = thx``, ``theta_y = thy`` (matching generateIntrinsics).
    noll_list : `list` [`int`]
        Noll indices to return.
    band : `str`
        Band whose effective wavelength sets the batoid wavelength (see
        `BAND_WAVELENGTH_M`).
    optical_model : `str`
        Batoid yaml stem; ``Rubin_v3.14`` -> ``Rubin_v3.14_{band}.yaml``
        (as-built), ``LSST`` -> ``LSST_{band}.yaml`` (nominal design).
    jmax : `int`, optional
        Max Noll index traced by batoid; default ``max(noll_list)``.  Must be
        >= the largest requested Noll.
    eps, nx : `float`, `int`
        ``batoid.zernike`` pupil obscuration and ray-grid sampling.
    projection : `str`
        ``batoid.zernike`` field-angle projection.

    Returns
    -------
    `dict` [`int`, `numpy.ndarray`]
        ``{j: values (µm)}`` aligned with the input points; NaN at any point
        where batoid could not evaluate the wavefront.
    """
    import batoid
    from batoid_rubin import LSSTBuilder

    js = [int(j) for j in noll_list]
    if not js:
        return {}
    if band not in BAND_WAVELENGTH_M:
        raise ValueError(f"unknown band {band!r}; known {list(BAND_WAVELENGTH_M)}")
    wavelength = BAND_WAVELENGTH_M[band]
    if jmax is None:
        jmax = max(js)
    elif jmax < max(js):
        raise ValueError(f"jmax={jmax} < largest requested Noll {max(js)}")

    fid = batoid.Optic.fromYaml(f"{optical_model}_{band}.yaml")
    telescope = LSSTBuilder(
        fid,
        dof_coord_system="OCS",
        flip_m2_bending_modes=False,
        dof_angle_units="degree",
    ).build()

    thx = np.radians(np.asarray(thx_deg, dtype=float))
    thy = np.radians(np.asarray(thy_deg, dtype=float))
    out = {j: np.full(thx.size, np.nan, dtype=float) for j in js}
    for i in range(thx.size):
        if not (np.isfinite(thx[i]) and np.isfinite(thy[i])):
            continue
        try:
            z = batoid.zernike(
                telescope,
                theta_x=float(thx[i]), theta_y=float(thy[i]),
                wavelength=wavelength, projection=projection,
                jmax=jmax, eps=eps, nx=nx,
            )
        except Exception:
            continue
        z_um = np.asarray(z, dtype=float) * wavelength * 1e6  # waves -> m -> µm
        for j in js:
            out[j][i] = z_um[j] if j < z_um.size else np.nan
    return out


def backfill_ocs_from_batoid(maps, fill_noll, band="i",
                             optical_model=DEFAULT_OPTICAL_MODEL, jmax=None,
                             eps=DEFAULT_EPS, nx=DEFAULT_NX,
                             projection=DEFAULT_PROJECTION):
    """Add batoid design OCS columns for the Noll the MIW maps are missing.

    For every ``j`` in ``fill_noll`` that is **not** already a shared OCS/CCS
    Noll of ``maps`` (see ``calib_tables.maps_noll_indices``), add a
    ``Z{j}_OCS`` column (the batoid design intrinsic sampled on the maps'
    ``thx_deg``/``thy_deg`` grid, in µm) and a zero ``Z{j}_CCS`` column (no
    measured camera field for these -- OCS-only).  Noll already present in the
    maps are left untouched.

    A copy of ``maps`` is returned (the input is not mutated) with a
    ``batoid_ocs_backfill`` record added to ``.meta``.

    Parameters
    ----------
    maps : `astropy.table.Table`
        MIW ``intrinsic_split_maps`` table (``thx_deg``, ``thy_deg``,
        ``Z{j}_OCS``/``Z{j}_CCS``).
    fill_noll : iterable of `int`
        Candidate Noll indices to backfill; only those absent from ``maps`` are
        actually added.
    band, optical_model, jmax, eps, nx, projection
        Passed to `batoid_ocs_zernikes`.

    Returns
    -------
    augmented_maps : `astropy.table.Table`
        Copy of ``maps`` with the backfilled columns.
    filled_js : `list` [`int`]
        The Noll indices that were actually added (empty if none were missing).
    """
    # Local import: avoid a hard calib_tables <-> batoid_intrinsic import cycle
    # and keep calib_tables free of any batoid dependency.
    from lsst.ts.intrinsic.wavefront.calib_tables import maps_noll_indices

    have = set(maps_noll_indices(maps))
    fill_js = sorted({int(j) for j in fill_noll} - have)
    if not fill_js:
        return maps, []

    thx = np.asarray(maps["thx_deg"], dtype=float)
    thy = np.asarray(maps["thy_deg"], dtype=float)
    vals = batoid_ocs_zernikes(thx, thy, fill_js, band=band,
                               optical_model=optical_model, jmax=jmax,
                               eps=eps, nx=nx, projection=projection)

    out = maps.copy()
    zeros = np.zeros(len(out), dtype=float)
    for j in fill_js:
        out[f"Z{j}_OCS"] = vals[j]
        out[f"Z{j}_CCS"] = zeros.copy()   # OCS-only: camera field left empty
    meta = dict(out.meta)
    meta["batoid_ocs_backfill"] = dict(
        optical_model=optical_model,
        band=band,
        filled_noll=[int(j) for j in fill_js],
        jmax=int(jmax if jmax is not None else max(fill_js)),
        eps=float(eps),
        nx=int(nx),
        projection=projection,
    )
    out.meta = meta
    return out, fill_js
