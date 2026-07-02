#!/usr/bin/env python3
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

"""Generate ``IntrinsicZernikes`` *source* tables from MIW maps.

Reads an ``intrinsic_split_maps.parquet`` (thx_deg, thy_deg, Z{j}_OCS, Z{j}_CCS
in µm) and writes, under ``<out-root>/<version>/``:

* ``intrinsic_aberrations_OCS.parquet`` — the telescope-fixed intrinsics, the
  full focal-plane map.  Written **once**: it is detector- and
  filter-independent (the OCS query point is rotated by the rotator, so each
  detector needs the whole map).  With ``--fill-ocs-from-batoid`` the Noll
  indices the MIW never measured are backfilled here from the batoid design
  intrinsic (Rubin v3.14) on the same grid; their CCS camera field is left
  empty (zero), keeping OCS and CCS on the identical Noll set ip_isr requires.
* ``intrinsic_aberrations_CCS_det<NNN>.parquet`` — one **per detector**: the
  smooth camera field ``C`` sampled on a grid of points across the CCD footprint,
  with the CCD focal-plane height (``ccd_height``; 15 µm/mm) added to ``Z4`` at
  **each point** (so the intra-CCD height structure is preserved).  The CCS query
  point is never rotated, so each detector only needs its own footprint.  The
  ~16 field-edge CCDs whose footprints fall past the data are stored height-only
  (``Z4`` = mean height, camera field zeroed).
* ``provenance.yaml`` — map source, git state, detectors, height source, the
  list of height-only fallback detectors, and the maps' own ``.meta``.

Filters are handled at **ingest** time (the tables are filter-independent; the
``physical_filter`` is set in the dataId, replicating across bands), so nothing
filter-specific is written here.  The Butler dataset type stays per
``(detector, physical_filter)`` — no ts_wep change.

Locate the maps with ``--maps PATH`` or, mirroring ``calibration/stage_miw.py``,
via ``--param-set``/``--mi-name`` (``--output-root`` defaults to
``pipelines/output``).  Needs the LSST stack (cameraGeom + obs_lsst) and the
batoid_rubin / metrology height map.

Example
-------
Full run, all LSSTCam detectors, batoid_rubin heights, version v4::

    run_make_calib_tables.py --param-set <ps> --mi-name <mi> --version v4 \
        --height-source batoid_rubin --height-map-dir <ccd_height_map>
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from astropy.table import Table

from lsst.ts.intrinsic.wavefront import calib_tables as ct

# Repo root (this file is <repo>/bin.src/run_make_calib_tables.py).
REPO = Path(__file__).resolve().parent.parent

DEFAULT_OUT_ROOT = "/sdf/group/rubin/repo/aos_imsim/gmegias/intrinsic_zernikes"


def _git(*args):
    """Best-effort ``git`` query against the repo; ``None`` on any failure."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _resolve_maps(args):
    if args.maps:
        return Path(args.maps)
    if not (args.param_set and args.mi_name):
        sys.exit("ERROR: give either --maps PATH or both --param-set and --mi-name.")
    return (Path(args.output_root) / args.param_set / args.mi_name
            / "intrinsic_split_maps.parquet")


def _get_camera(instrument):
    if instrument == "LSSTCam":
        from lsst.obs.lsst import LsstCam
        return LsstCam().getCamera()
    if instrument == "LSSTComCam":
        from lsst.obs.lsst import LsstComCam
        return LsstComCam().getCamera()
    raise ValueError(f"don't know the camera for instrument {instrument!r}")


def _detector_footprints(camera, detectors, n_side):
    """Per-detector footprint sample points for the refined CCS table.

    For each detector, lay an ``n_side`` x ``n_side`` grid across its pixel
    bounding box and map every node to a field position in the **CCS map
    frame** (cameraGeom ``FIELD_ANGLE`` with the x<->y transpose the intrinsic
    maps use, ``thx = FA_y``, ``thy = FA_x``).

    Returns ``{detector_id: dict(thx_deg, thy_deg, pix_x, pix_y, det_name)}``.
    The pixel coords are kept so the CCD height can be sampled at the very same
    points (via ``compute_ccd_heights``).
    """
    import numpy as np
    import lsst.geom as geom
    from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS

    want = {int(d) for d in detectors}
    out = {}
    for det in camera:
        did = int(det.getId())
        if did not in want:
            continue
        bb = det.getBBox()
        xs = np.linspace(bb.getMinX(), bb.getMaxX(), n_side)
        ys = np.linspace(bb.getMinY(), bb.getMaxY(), n_side)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        tr = det.getTransform(PIXELS, FIELD_ANGLE)
        fax = np.empty(gx.size)
        fay = np.empty(gx.size)
        for i in range(gx.size):
            p = tr.applyForward(geom.Point2D(float(gx[i]), float(gy[i])))
            fax[i] = np.degrees(p.getX())
            fay[i] = np.degrees(p.getY())
        out[did] = dict(thx_deg=fay, thy_deg=fax, pix_x=gx, pix_y=gy,
                        det_name=det.getName())
    return out


def _footprint_heights(footprints, camera, source, height_map_dir,
                       metrology_fits, factor):
    """Per-point Z4 height (µm) for every footprint sample of every detector.

    Runs the pipeline's ``compute_ccd_heights`` **once** over all points (so the
    batoid height maps load a single time), then splits the result back per
    detector.  Returns ``{detector_id: ndarray of Z4_height (µm)}`` aligned with
    each detector's footprint order.
    """
    import numpy as np
    import pandas as pd

    from lsst.ts.intrinsic.wavefront import ccd_height as ch

    order = list(footprints)
    det_names, px, py = [], [], []
    for did in order:
        fp = footprints[did]
        n = fp["pix_x"].size
        det_names.extend([fp["det_name"]] * n)
        px.append(fp["pix_x"])
        py.append(fp["pix_y"])
    df = pd.DataFrame({"detector": det_names,
                       "centroid_x_intra": np.concatenate(px),
                       "centroid_y_intra": np.concatenate(py)})
    # a footprint node is a single point, so intra == extra centroid there
    df["centroid_x_extra"] = df["centroid_x_intra"]
    df["centroid_y_extra"] = df["centroid_y_intra"]
    out = ch.compute_ccd_heights(df, camera, source=source,
                                 height_map_dir=height_map_dir,
                                 metrology_fits=metrology_fits, factor=factor)
    z4 = np.asarray(out["Z4_height"], dtype=float)
    z4 = np.where(np.isfinite(z4), z4, 0.0)
    res, pos = {}, 0
    for did in order:
        n = footprints[did]["pix_x"].size
        res[did] = z4[pos:pos + n]
        pos += n
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("map source (give --maps OR --param-set/--mi-name)")
    src.add_argument("--maps", default=None, help="path to intrinsic_split_maps.parquet")
    src.add_argument("--param-set", default=None)
    src.add_argument("--mi-name", default=None)
    src.add_argument("--output-root", default=str(REPO / "pipelines" / "output"),
                     help="pipeline output root for --param-set/--mi-name lookup")

    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help=f"root for generated tables (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--version", required=True,
                    help="version label; tables go in <out-root>/<version>/")
    ap.add_argument("--instrument", default="LSSTCam")
    ap.add_argument("--detectors", type=int, nargs="+", default=None,
                    help="detector ids (default: every camera detector)")
    ap.add_argument("--noll-list", type=int, nargs="+", default=None,
                    help="subset of Noll indices (default: all shared by OCS & CCS)")

    b = ap.add_argument_group(
        "OCS backfill from the batoid optical model",
        "Fill Noll indices the MIW never measured with the batoid design "
        "intrinsic (OCS only; the CCS camera field for those is left empty). "
        "The design intrinsic is evaluated on the SAME field grid as the OCS "
        "table (the maps' thx/thy points).")
    b.add_argument("--fill-ocs-from-batoid", action="store_true",
                   help="enable the batoid OCS backfill")
    b.add_argument("--batoid-optical-model", default="Rubin_v3.14",
                   help="batoid yaml stem: 'Rubin_v3.14' -> Rubin_v3.14_<band>.yaml "
                        "(latest as-built, default); 'LSST' -> nominal design")
    b.add_argument("--batoid-band", default="i",
                   help="band whose wavelength sets the design intrinsic; the OCS "
                        "table is filter-independent so one band is chosen "
                        "(default: i, the MIW measurement band)")
    b.add_argument("--batoid-fill-jmax", type=int, default=78,
                   help="fill every missing Noll in 4..JMAX (default: 78); "
                        "ignored if --batoid-fill-noll is given")
    b.add_argument("--batoid-fill-noll", type=int, nargs="+", default=None,
                   help="explicit Noll indices to backfill (overrides "
                        "--batoid-fill-jmax); only those absent from the maps "
                        "are added")

    h = ap.add_argument_group("per-detector CCS heights")
    h.add_argument("--height-source", default="batoid_rubin",
                   choices=["batoid_rubin", "metrology"])
    h.add_argument("--height-map-dir", default=None,
                   help="ccd_height_map dir for the batoid_rubin source "
                        "(None -> batoid_rubin ensure_data_dir)")
    h.add_argument("--metrology-fits", default=None,
                   help="FITS path for the metrology source")
    h.add_argument("--height-to-z4-factor", type=float, default=15.0,
                   help="µm of Z4 per mm of CCD height (default: 15)")
    h.add_argument("--ccs-footprint-points", type=int, default=50,
                   help="approx. sample points per detector for the CCS table: "
                        "a round(sqrt(N)) x round(sqrt(N)) grid across each "
                        "detector footprint, with the CCD height sampled at "
                        "each point (preserves intra-CCD height structure). "
                        "(default: 50)")
    h.add_argument("--ccs-footprint-min-points", type=int, default=3,
                   help="if a detector has fewer than this many in-support "
                        "footprint points (i.e. it sits at the field edge, past "
                        "the data), store it height-only: Z4 = mean height, "
                        "camera field zeroed. (default: 3)")

    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing <out-root>/<version>/ directory")
    args = ap.parse_args()

    maps_path = _resolve_maps(args)
    if not maps_path.exists():
        sys.exit(f"ERROR: maps not found: {maps_path}")
    maps = Table.read(str(maps_path), format="parquet")

    # ---- optional OCS backfill from the batoid design intrinsic ----
    # Add Z{j}_OCS (batoid design, on the maps grid) + zero Z{j}_CCS for every
    # requested Noll the MIW never measured, so the downstream OCS/CCS builders
    # pick them up as ordinary shared-Noll columns (CCS left empty for them).
    backfilled = []
    if args.fill_ocs_from_batoid:
        from lsst.ts.intrinsic.wavefront import batoid_intrinsic as bi
        fill_noll = (args.batoid_fill_noll if args.batoid_fill_noll is not None
                     else list(range(4, args.batoid_fill_jmax + 1)))
        jmax = (max(fill_noll) if args.batoid_fill_noll is not None
                else args.batoid_fill_jmax)
        print(f"[make_calib_tables] batoid OCS backfill: model "
              f"{args.batoid_optical_model}, band {args.batoid_band}, "
              f"candidate Noll {min(fill_noll)}..{max(fill_noll)}")
        maps, backfilled = bi.backfill_ocs_from_batoid(
            maps, fill_noll, band=args.batoid_band,
            optical_model=args.batoid_optical_model, jmax=jmax)
        print(f"  filled {len(backfilled)} missing Noll from batoid (CCS zeroed): "
              f"{backfilled}")

    js = ct.maps_noll_indices(maps)
    if args.noll_list is not None:
        js = [j for j in args.noll_list if j in js]
    print(f"[make_calib_tables] maps: {maps_path}  ({len(maps)} field points, "
          f"Noll {js})")

    # ---- detectors + per-detector footprint heights ----
    # Sample the CCD height at an n_side x n_side grid across each detector
    # footprint, so the intra-CCD height structure is preserved (added to Z4 per
    # point below).  ``mean_height`` is only used for the height-only field-edge
    # CCDs, whose footprints fall past the data.
    camera = _get_camera(args.instrument)
    all_ids = [int(det.getId()) for det in camera]
    detectors = list(args.detectors) if args.detectors is not None else all_ids
    n_side = max(2, int(round(args.ccs_footprint_points ** 0.5)))
    print(f"[make_calib_tables] sampling CCS footprints "
          f"({n_side}x{n_side}={n_side * n_side} pts/detector; "
          f"source={args.height_source}, factor={args.height_to_z4_factor})")
    footprints = _detector_footprints(camera, detectors, n_side)
    fp_heights = _footprint_heights(
        footprints, camera, args.height_source, args.height_map_dir,
        args.metrology_fits, args.height_to_z4_factor)
    mean_height = {d: float(fp_heights[d].mean()) for d in detectors}

    dest = Path(args.out_root) / args.version
    if dest.exists() and any(dest.iterdir()) and not args.force:
        sys.exit(f"ERROR: {dest} already exists and is not empty (use --force).")
    dest.mkdir(parents=True, exist_ok=True)

    # ---- OCS table (once) ----
    ocs = ct.ocs_source_table(maps, noll_list=args.noll_list,
                              instrument=args.instrument)
    ocs_name = "intrinsic_aberrations_OCS.parquet"
    ocs.write(str(dest / ocs_name), format="parquet", overwrite=True)
    print(f"  wrote {ocs_name}  ({len(ocs)} pts x {len(js)} Zernikes)")

    # ---- per-detector CCS tables ----
    ccs_files = []
    n_rows = []
    fell_back = []
    for det in detectors:
        fp = footprints[det]
        ccs = ct.ccs_source_table_on_grid(
            maps, fp["thx_deg"], fp["thy_deg"],
            height_z4_um=fp_heights[det], detector=det,
            noll_list=args.noll_list, instrument=args.instrument)
        if len(ccs) < args.ccs_footprint_min_points:
            # field-edge CCD past the data: too few in-support footprint points to
            # interpolate a camera field, and no measured field exists there.
            # Store it height-only (Z4 = mean height, camera field zeroed).
            ccs = ct.ccs_source_table(maps, detector=det,
                                      piston_z4_um=mean_height[det],
                                      noll_list=args.noll_list,
                                      instrument=args.instrument,
                                      zero_camera_field=True)
            fell_back.append(det)
        name = f"intrinsic_aberrations_CCS_det{det:03d}.parquet"
        ccs.write(str(dest / name), format="parquet", overwrite=True)
        ccs_files.append(name)
        n_rows.append(len(ccs))
    fp_rows = [n for d, n in zip(detectors, n_rows) if d not in fell_back]
    print(f"  wrote {len(ccs_files)} per-detector CCS tables "
          f"({len(fell_back)} field-edge CCDs stored height-only; footprint "
          f"pts/detector {min(fp_rows) if fp_rows else 0}.."
          f"{max(fp_rows) if fp_rows else 0}; mean-height range "
          f"{min(mean_height.values()):+.4f}..{max(mean_height.values()):+.4f} µm)")
    if fell_back:
        print(f"    height-only (camera field zeroed): detectors {fell_back}")

    # ---- provenance ----
    prov = dict(
        version=args.version,
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        maps_source=(str(maps_path.relative_to(REPO))
                     if maps_path.is_relative_to(REPO) else str(maps_path)),
        param_set=args.param_set,
        mi_name=args.mi_name,
        out_dir=str(dest),
        git_sha=_git("rev-parse", "HEAD"),
        git_describe=_git("describe", "--tags", "--always", "--dirty"),
        instrument=args.instrument,
        n_detectors=len(detectors),
        detectors=[int(d) for d in detectors],
        noll_indices=[int(j) for j in js],
        ocs_file=ocs_name,
        ccs_files=ccs_files,
        batoid_ocs_backfill=(dict(maps.meta["batoid_ocs_backfill"])
                             if backfilled else None),
        heights=dict(
            source=args.height_source,
            height_map_dir=args.height_map_dir,
            metrology_fits=args.metrology_fits,
            height_to_z4_factor=args.height_to_z4_factor,
            ccs_footprint_points=args.ccs_footprint_points,
            ccs_footprint_fallback_detectors=sorted(int(d) for d in fell_back),
        ),
        maps_meta={k: v for k, v in dict(maps.meta).items()},
    )
    prov_path = dest / "provenance.yaml"
    with open(prov_path, "w") as fh:
        yaml.safe_dump({k: v for k, v in prov.items() if v is not None}, fh,
                       sort_keys=False, default_flow_style=False)
    print(f"[make_calib_tables] wrote OCS + {len(ccs_files)} CCS tables + "
          f"provenance.yaml -> {dest}")
    print(f"  git: {prov['git_describe']}")
    print("Next: review the tables, then ingest with bin.src/ingest_calib_tables.py "
          "(not run automatically).")


if __name__ == "__main__":
    main()
