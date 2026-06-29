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

"""Ingest generated MIW source tables into the Butler as ``IntrinsicZernikes``.

Companion to ``run_make_calib_tables.py``.  Reads the single OCS table
(``intrinsic_aberrations_OCS.parquet``) and the per-detector CCS tables
(``intrinsic_aberrations_CCS_det<NNN>.parquet``) it produced, builds one
``lsst.ip.isr.IntrinsicZernikes`` **per detector** (the shared OCS system +
that detector's CCS system), and ``butler.put``\ s it for **every requested
physical_filter** (the tables are filter-independent; the filter lives only in
the dataId).

The ``intrinsicZernikes`` dataset type stays dimensioned ``(instrument,
detector, physical_filter)`` — matching ts_wep's ``CalcZernikesTask`` lookup —
so no ts_wep change is needed.

This is the OCS-aware counterpart to ts_wep's ``ingestIntrinsicZernikes`` (which
builds a CCS-only calibration from single-table sources already in a Butler
collection).  Each built calibration is verified by default before any write
(Noll indices present, CCS/OCS interpolators built, ``getIntrinsicZernikes``
finite at interior sample points incl. the OCS rotation); a failure aborts the
run.  Pass ``--no-verify`` to skip it.

Defaults to ``--dry-run`` — it resolves, verifies and logs the plan but writes
nothing unless ``--execute`` is given.

Example (after reviewing the generated tables)::

    ingest_calib_tables.py \
        -b /repo/main \
        --tables-dir /sdf/group/rubin/repo/aos_imsim/gmegias/intrinsic_zernikes/v2 \
        --output-run LSSTCam/calib/DM-55048/intrinsicZernikes.v2/run.20260628 \
        --certify-into LSSTCam/calib/DM-55048/intrinsicZernikes.v2 \
        --execute

Needs the LSST stack (lsst.daf.butler, lsst.ip.isr, lsst.obs.lsst).
"""
import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

from lsst.daf.butler import Butler, CollectionType, DatasetType, Timespan
from lsst.ip.isr import IntrinsicZernikes

from lsst.ts.intrinsic.wavefront import calib_tables as ct

OCS_NAME = "intrinsic_aberrations_OCS.parquet"
_CCS_RE = re.compile(r"^intrinsic_aberrations_CCS_det(\d+)\.parquet$")


def _discover_tables(tables_dir):
    """Return (ocs_path, {detector_id: ccs_path}) from a generated tables dir."""
    d = Path(tables_dir)
    ocs = d / OCS_NAME
    if not ocs.exists():
        raise RuntimeError(f"missing OCS table {ocs}")
    ccs = {}
    for p in sorted(d.glob("intrinsic_aberrations_CCS_det*.parquet")):
        m = _CCS_RE.match(p.name)
        if m:
            ccs[int(m.group(1))] = p
    if not ccs:
        raise RuntimeError(f"no per-detector CCS tables in {tables_dir}")
    return ocs, ccs


def _verify_calib(calib, has_ocs):
    """Sanity-check a freshly built ``IntrinsicZernikes``; return problem list."""
    problems = []
    nz = np.asarray(calib.noll_indices)
    if nz.size == 0:
        problems.append("no Noll indices")
    if calib.interpolator is None:
        problems.append("CCS interpolator is None")
    if has_ocs and calib.interpolator_ocs is None:
        problems.append("OCS table supplied but OCS interpolator is None")

    fx = np.asarray(calib.field_x, dtype=float)
    fy = np.asarray(calib.field_y, dtype=float)
    if fx.size == 0:
        problems.append("no CCS sample points")
        return problems

    # Interior points (r <= 0.6 * r_max): rotating the query for the OCS term
    # keeps them inside the hull, so a NaN here is a real defect.
    r = np.hypot(fx, fy)
    interior = np.where(r <= 0.6 * r.max())[0]
    if interior.size == 0:
        interior = np.array([int(np.argmin(r))])
    pick = interior[np.linspace(0, interior.size - 1,
                                min(5, interior.size)).astype(int)]
    for rot in (0.0, 30.0):
        vals = np.asarray(calib.getIntrinsicZernikes(
            field_x=fx[pick], field_y=fy[pick], rotTelPos=rot))
        if vals.shape[-1] != nz.size:
            problems.append(f"value width {vals.shape[-1]} != {nz.size} Noll "
                            f"indices (rotTelPos={rot:g})")
        n_bad = int((~np.isfinite(vals)).sum())
        if n_bad:
            problems.append(f"{n_bad} non-finite interpolated value(s) at "
                            f"interior sample points (rotTelPos={rot:g})")
    return problems


def main():
    tz = time.strftime("%z")
    logging.basicConfig(
        format="%(levelname)s %(asctime)s.%(msecs)03d" + tz + " - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-b", "--butler-config", required=True, metavar="REPO",
                    help="butler/registry config (e.g. /repo/main)")
    ap.add_argument("--tables-dir", required=True,
                    help="dir with intrinsic_aberrations_OCS.parquet + "
                         "intrinsic_aberrations_CCS_det<NNN>.parquet")
    ap.add_argument("--instrument", default="LSSTCam")
    ap.add_argument("--output-dataset-type", default="intrinsicZernikes")
    ap.add_argument("--output-run", required=True,
                    help="RUN collection to write into")
    ap.add_argument("--certify-into", default=None,
                    help="if set, certify into this CALIBRATION collection "
                         "(unbounded timespan)")
    ap.add_argument("--bands", nargs="+", default=list(ct.LSSTCAM_PHYSICAL_FILTERS),
                    help="bands to write the calib for (default: u g r i z y)")
    ap.add_argument("--physical-filters", nargs="+", default=None,
                    help="explicit physical_filter list (overrides --bands)")
    ap.add_argument("--detectors", type=int, nargs="+", default=None,
                    help="detector ids to ingest (default: all CCS tables found)")
    ap.add_argument("--execute", action="store_true",
                    help="actually write/certify (default: dry-run only)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-detector calibration sanity check "
                         "(verification is on by default)")
    args = ap.parse_args()

    dry_run = not args.execute
    verify = not args.no_verify

    if args.physical_filters is not None:
        physical_filters = list(args.physical_filters)
    else:
        unknown = [b for b in args.bands if b not in ct.LSSTCAM_PHYSICAL_FILTERS]
        if unknown:
            raise RuntimeError(f"unknown band(s) {unknown}; known: "
                               f"{list(ct.LSSTCAM_PHYSICAL_FILTERS)} "
                               f"(or pass --physical-filters)")
        physical_filters = [ct.LSSTCAM_PHYSICAL_FILTERS[b] for b in args.bands]

    ocs_path, ccs_paths = _discover_tables(args.tables_dir)
    if args.detectors is not None:
        missing = [d for d in args.detectors if d not in ccs_paths]
        if missing:
            raise RuntimeError(f"no CCS table for detector(s) {missing}")
        ccs_paths = {d: ccs_paths[d] for d in args.detectors}
    detectors = sorted(ccs_paths)
    n_put = len(detectors) * len(physical_filters)
    logger.info(f"{len(detectors)} detectors x {len(physical_filters)} filters "
                f"= {n_put} calibrations to write "
                f"({'DRY-RUN' if dry_run else 'EXECUTE'}); OCS={ocs_path.name}")

    ocs = Table.read(str(ocs_path), format="parquet")

    # Build + verify every detector's calibration before touching the Butler, so
    # a calibration that fails verification aborts before anything is written.
    calibs = {}
    for det in detectors:
        ccs = Table.read(str(ccs_paths[det]), format="parquet")
        calib = IntrinsicZernikes(table=ccs, table_ocs=ocs)
        if verify:
            problems = _verify_calib(calib, has_ocs=True)
            if problems:
                for p in problems:
                    logger.error(f"verify detector={det}: {p}")
                raise RuntimeError(
                    f"calibration for detector {det} failed verification "
                    f"({len(problems)} problem(s)); fix the source tables or pass "
                    f"--no-verify to bypass")
        calibs[det] = calib
    logger.info(f"Built {len(calibs)} per-detector calibrations"
                + (" (verified)" if verify else "")
                + f"; Noll {calibs[detectors[0]].noll_indices.tolist()}")

    butler = Butler.from_config(args.butler_config, writeable=not dry_run)

    outputDatasetType = DatasetType(
        args.output_dataset_type,
        ("instrument", "detector", "physical_filter"),
        "IsrCalib",
        universe=butler.dimensions,
        isCalibration=True,
    )
    if not dry_run:
        logger.info(f"Registering dataset type {args.output_dataset_type!r} (if absent)")
        butler.registry.registerDatasetType(outputDatasetType)
        logger.info(f"Registering output RUN collection {args.output_run!r}")
        butler.registry.registerCollection(args.output_run, CollectionType.RUN)

    outputRefs = []
    for det in detectors:
        calib = calibs[det]
        for pf in physical_filters:
            calib.updateMetadata(
                INSTRUME=args.instrument, DETECTOR=int(det), FILTER=str(pf)
            )
            dataId = dict(instrument=args.instrument, detector=int(det),
                          physical_filter=pf)
            if dry_run:
                continue
            outputRefs.append(
                butler.put(calib, args.output_dataset_type, dataId=dataId,
                           run=args.output_run)
            )

    if dry_run:
        logger.info(f"[dry-run] would write {n_put} calibrations into "
                    f"{args.output_run!r}"
                    + (f" and certify into {args.certify_into!r}"
                       if args.certify_into else ""))
        logger.info("Re-run with --execute to write.")
        return

    logger.info(f"Wrote {len(outputRefs)} calibrations into {args.output_run!r}")
    if args.certify_into is None:
        logger.info("Done. No certification requested.")
        return

    logger.info(f"Registering CALIBRATION collection {args.certify_into!r}")
    butler.registry.registerCollection(args.certify_into, CollectionType.CALIBRATION)
    logger.info(f"Certifying {len(outputRefs)} refs into {args.certify_into!r} "
                f"(unbounded timespan)")
    butler.registry.certify(args.certify_into, outputRefs, Timespan(None, None))
    logger.info("Done.")


if __name__ == "__main__":
    main()
