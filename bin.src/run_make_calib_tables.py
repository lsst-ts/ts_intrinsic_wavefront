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
  detector needs the whole map).
* ``intrinsic_aberrations_CCS_det<NNN>.parquet`` — one **per detector**: the
  smooth camera field ``C`` plus that CCD's focal-plane height as a Z4 piston
  (``ccd_height``; 15 µm/mm).  The CCS query point is never rotated, so each
  detector only needs its own value, and the height makes it detector-specific.
* ``provenance.yaml`` — map source, git state, detectors, height source, the
  per-detector pistons, and the maps' own ``.meta``.

Filters are handled at **ingest** time (the tables are filter-independent; the
``physical_filter`` is set in the dataId, replicating across bands), so nothing
filter-specific is written here.  The Butler dataset type stays per
``(detector, physical_filter)`` — no ts_wep change.

Locate the maps with ``--maps PATH`` or, mirroring ``calibration/stage_miw.py``,
via ``--param-set``/``--mi-name`` (``--output-root`` defaults to
``pipelines/output``).

The per-detector heights need the LSST stack (cameraGeom + obs_lsst) and the
batoid_rubin / metrology height map.  Use ``--no-heights`` for a quick,
stack-free run (CCS = smooth camera field only, Z4 piston 0); combine with an
explicit ``--detectors`` list to avoid needing the camera at all.

Examples
--------
Full run, all LSSTCam detectors, batoid_rubin heights, version v2::

    run_make_calib_tables.py \
        --maps calibration/miw/intrinsic_split_maps_v1.parquet --version v2

Quick stack-free smoke test for two detectors::

    run_make_calib_tables.py --maps <maps> --version test \
        --no-heights --detectors 90 91
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
    raise ValueError(f"don't know the camera for instrument {instrument!r}; "
                     f"pass --detectors and --no-heights")


def _per_detector_piston_z4(camera, source, height_map_dir, metrology_fits, factor):
    """Return {detector_id: Z4 piston (µm)} from the CCD height at each CCD centre.

    Reuses the pipeline's ``ccd_height.compute_ccd_heights`` path (same source,
    orientation and factor) evaluated at every detector's bbox centre, so the
    piston is consistent with how the build step applies heights.
    """
    import numpy as np
    import pandas as pd

    from lsst.ts.intrinsic.wavefront import ccd_height as ch

    ids, rows = [], []
    for det in camera:
        c = det.getBBox().getCenter()
        ids.append(int(det.getId()))
        rows.append((det.getName(), float(c.getX()), float(c.getY())))
    df = pd.DataFrame(rows, columns=["detector", "cx", "cy"])
    # compute_ccd_heights averages the intra- and extra-focal centroids; at a
    # CCD centre both are the same point.
    df["centroid_x_intra"] = df["cx"]
    df["centroid_y_intra"] = df["cy"]
    df["centroid_x_extra"] = df["cx"]
    df["centroid_y_extra"] = df["cy"]
    out = ch.compute_ccd_heights(df, camera, source=source,
                                 height_map_dir=height_map_dir,
                                 metrology_fits=metrology_fits, factor=factor)
    z4 = np.asarray(out["Z4_height"], dtype=float)
    return {ids[i]: (float(z4[i]) if np.isfinite(z4[i]) else 0.0)
            for i in range(len(ids))}


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

    h = ap.add_argument_group("per-detector CCS heights")
    h.add_argument("--no-heights", action="store_true",
                   help="skip the CCD height piston (CCS = smooth camera field only)")
    h.add_argument("--height-source", default="batoid_rubin",
                   choices=["batoid_rubin", "metrology"])
    h.add_argument("--height-map-dir", default=None,
                   help="ccd_height_map dir for the batoid_rubin source "
                        "(None -> batoid_rubin ensure_data_dir)")
    h.add_argument("--metrology-fits", default=None,
                   help="FITS path for the metrology source")
    h.add_argument("--height-to-z4-factor", type=float, default=15.0,
                   help="µm of Z4 per mm of CCD height (default: 15)")

    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing <out-root>/<version>/ directory")
    args = ap.parse_args()

    maps_path = _resolve_maps(args)
    if not maps_path.exists():
        sys.exit(f"ERROR: maps not found: {maps_path}")
    maps = Table.read(str(maps_path), format="parquet")
    js = ct.maps_noll_indices(maps)
    if args.noll_list is not None:
        js = [j for j in args.noll_list if j in js]
    print(f"[make_calib_tables] maps: {maps_path}  ({len(maps)} field points, "
          f"Noll {js})")

    # ---- detectors + per-detector Z4 piston ----
    if args.detectors is not None and args.no_heights:
        detectors = list(args.detectors)        # no camera needed
        pistons = {d: 0.0 for d in detectors}
    else:
        camera = _get_camera(args.instrument)
        all_ids = [int(det.getId()) for det in camera]
        detectors = list(args.detectors) if args.detectors is not None else all_ids
        if args.no_heights:
            pistons = {d: 0.0 for d in detectors}
        else:
            print(f"[make_calib_tables] computing CCD height pistons "
                  f"(source={args.height_source}, factor={args.height_to_z4_factor})")
            pistons = _per_detector_piston_z4(
                camera, args.height_source, args.height_map_dir,
                args.metrology_fits, args.height_to_z4_factor)
            pistons = {d: pistons.get(d, 0.0) for d in detectors}

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
    for det in detectors:
        ccs = ct.ccs_source_table(maps, detector=det, piston_z4_um=pistons[det],
                                  noll_list=args.noll_list,
                                  instrument=args.instrument)
        name = f"intrinsic_aberrations_CCS_det{det:03d}.parquet"
        ccs.write(str(dest / name), format="parquet", overwrite=True)
        ccs_files.append(name)
    print(f"  wrote {len(ccs_files)} per-detector CCS tables "
          f"(Z4 piston range {min(pistons.values()):+.4f}..{max(pistons.values()):+.4f} µm)")

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
        heights=dict(
            applied=not args.no_heights,
            source=(None if args.no_heights else args.height_source),
            height_map_dir=(None if args.no_heights else args.height_map_dir),
            metrology_fits=(None if args.no_heights else args.metrology_fits),
            height_to_z4_factor=args.height_to_z4_factor,
            piston_z4_um={int(d): round(pistons[d], 6) for d in detectors},
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
