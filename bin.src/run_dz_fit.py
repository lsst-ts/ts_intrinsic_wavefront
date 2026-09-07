#!/usr/bin/env python3
"""Run Double Zernike focal-plane fits on a donut wavefront table.

Usage:
    python run_dz_fit.py input.hdf5
    python run_dz_fit.py input.hdf5 --coord-sys CCS
    python run_dz_fit.py input.hdf5 --output output_fits.parquet
"""

import argparse

from lsst.ts.intrinsic.wavefront.dz_fitting import run_double_zernike_fits


def main():
    parser = argparse.ArgumentParser(
        description='Run Double Zernike focal-plane fits on donut wavefront data.')
    parser.add_argument('input_file',
                        help='Input HDF5 file with donuts+visits tables '
                             '(from intrinsics_mktable)')
    parser.add_argument('--output', default=None,
                        help='Output fit parquet file '
                             '(default: {input_stem}_fits.parquet)')
    parser.add_argument('--visits', default=None,
                        help='Visits sidecar parquet (default: auto — '
                             '{input_stem}_visits.parquet, else visits.parquet '
                             'in the same directory)')
    parser.add_argument('--intrinsic-sidecar', default=None,
                        help='Measured-intrinsic sidecar parquet '
                             '(zk_intrinsic.parquet from run_make_intrinsic_sidecar). '
                             'If given, the fit subtracts the measured intrinsic '
                             'instead of the tabulated zk_intrinsic.')
    parser.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'],
                        help='Coordinate system (default: OCS)')
    parser.add_argument('--bad-fit-threshold', type=float, default=2.0,
                        help='Flag fits with |coeff| > threshold μm (default: 2.0)')
    parser.add_argument('--min-donuts', type=int, default=200,
                        help='Flag fits with fewer donuts (default: 200)')
    parser.add_argument('--min-detectors', type=int, default=None,
                        help='Opt-in override of the per-visit quality selection: '
                             'keep visits with n_detectors_with_min_donuts >= this, '
                             'relaxing ONLY the CCD-count cut (n_donuts/blur cuts '
                             'unchanged) and bypassing the precomputed '
                             'visit_quality_pass. Default: None (unchanged behavior).')
    parser.add_argument('--no-quality-cut', action='store_true',
                        help='Fit EVERY visit and apply no per-visit quality cut here; '
                             'the metric columns still travel in the output so each '
                             'consumer can cut as needed ("fit all, cut at use"). '
                             'Default: off (unchanged behavior).')

    args = parser.parse_args()

    run_double_zernike_fits(
        input_file=args.input_file,
        coord_sys=args.coord_sys,
        output_file=args.output,
        bad_fit_threshold=args.bad_fit_threshold,
        min_donuts=args.min_donuts,
        visits_file=args.visits,
        intrinsic_sidecar=args.intrinsic_sidecar,
        min_detectors=args.min_detectors,
        no_quality_cut=args.no_quality_cut,
    )


if __name__ == '__main__':
    main()
