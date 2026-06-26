#!/usr/bin/env python3
"""Build Zernike wavefront tables from Rubin AOS FAM observations.

Usage:
    # Use a named parameter set (from param_sets.yaml)
    python run_mktable.py --param-set fam_danish_triplets

    # Override date range for a large collection
    python run_mktable.py --param-set fam_danish_triplets --day-obs-min 20260315 --day-obs-max 20260316

    # Full manual specification
    python run_mktable.py --butler-repo /repo/embargo \
        --collections aos_fam_danish_triplets \
        --day-obs-min 20260315 --day-obs-max 20260317 \
        --programs T278 T381 T492 T539 T614

    # Enable optional computations
    python run_mktable.py --param-set fam_danish_triplets --calc-focal-plane
"""

import argparse
import asyncio

from lsst.ts.intrinsic.wavefront.intrinsics_lib import run_mktable, load_param_sets


def main():
    parser = argparse.ArgumentParser(
        description='Build Zernike wavefront tables from FAM observations.')

    # Option A: use a named parameter set from param_sets.yaml
    parser.add_argument('--param-set', default=None,
                        help='Use a named parameter set from param_sets.yaml '
                             '(e.g. fam_danish_triplets)')

    # Option B: specify everything directly
    parser.add_argument('--butler-repo', default=None,
                        help='Butler repository path (e.g. /repo/embargo)')
    parser.add_argument('--collections', nargs='+', default=None,
                        help='Butler collection(s)')
    parser.add_argument('--day-obs-min', type=int, default=None,
                        help='Minimum day_obs (e.g. 20260315); auto-parsed '
                             'from collection if not specified')
    parser.add_argument('--day-obs-max', type=int, default=None,
                        help='Maximum day_obs (e.g. 20260317); auto-parsed '
                             'from collection if not specified')
    parser.add_argument('--programs', nargs='+', default=None,
                        help='Science program names (e.g. T278 T381)')

    # Output naming
    parser.add_argument('--collection-phrase', default=None,
                        help='Override output filename phrase (auto-parsed '
                             'from collection if not specified)')
    parser.add_argument('--include-versions', action='store_true',
                        help='Include wep/dviz versions in output filename')

    # Optional computations (all off by default)
    parser.add_argument('--calc-mean-zernike', action='store_true',
                        help='Compute per-visit mean Zernike columns')
    parser.add_argument('--calc-focal-plane', action='store_true',
                        help='Compute focal plane coordinates (fpx, fpy)')

    # Other options
    parser.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'],
                        help='Coordinate system (default: OCS)')
    default_output_dir = 'output'
    parser.add_argument('--output-dir', default=default_output_dir,
                        help=f'Output directory (default: {default_output_dir})')
    parser.add_argument('--rotator-threshold', type=float, default=90.0,
                        help='Rotator angle flagging threshold in degrees')
    parser.add_argument('--fp-radius', type=float, default=1.8,
                        help='Focal plane radius in degrees')
    parser.add_argument('--fp-nsteps', type=int, default=73,
                        help='Grid points per axis for intrinsic model')
    parser.add_argument('--min-visits-per-day', type=int, default=5,
                        help='Minimum visit pairs per day_obs to include')
    parser.add_argument('--no-thermal', action='store_true',
                        help='Skip thermal data retrieval')
    parser.add_argument('--temp-time-window', type=float, default=0.2,
                        help='EFD temperature query post-padding (seconds)')
    parser.add_argument('--consdb-url',
                        default='https://usdf-rsp.slac.stanford.edu/consdb',
                        help='ConsDB URL (default: external USDF URL; '
                             'token read from ~/.lsst/consdb_token)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output parquet files '
                             '(default: refuse to clobber)')

    # Per-visit quality cut configuration
    parser.add_argument('--matched-threshold-arcsec', type=float, default=0.0,
                        help='intra/extra centroid agreement threshold for the '
                             'matched_intra_extra column (default 0 = disabled; '
                             'pass a positive value in arcsec to enable).')
    parser.add_argument('--min-donuts-per-visit', type=int, default=500,
                        help='Minimum donuts per visit (default 500). '
                             'Pass 0 to disable.')
    parser.add_argument('--min-donuts-per-detector', type=int, default=3,
                        help='Per-detector donut floor used to count "covered" '
                             'detectors (default 3).')
    parser.add_argument('--min-detectors-per-visit', type=int, default=170,
                        help='Minimum detectors with at least min-donuts-per-detector '
                             'donuts per visit (default 170). Pass 0 to disable.')
    parser.add_argument('--max-median-blur-arcsec', type=float, default=1.2,
                        help='Maximum median donut FWHM per visit (default 1.2). '
                             'Pass 0 or a negative number to disable.')

    args = parser.parse_args()

    # Resolve parameters
    if args.param_set is not None:
        param_sets = load_param_sets()
        if args.param_set not in param_sets:
            parser.error(f"Unknown param-set: {args.param_set}. "
                         f"Available: {sorted(param_sets.keys())}")
        params = dict(param_sets[args.param_set])
        # Honor the param_set's collection_phrase override so the output
        # filename is the clean phrase (e.g. aos_fam_danish_1_0_wep17_3_0_bin2x)
        # rather than the slashed collection path.  An explicit
        # --collection-phrase on the CLI still wins.
        if args.collection_phrase is None:
            args.collection_phrase = params.get('collection_phrase')
        # Allow CLI overrides
        if args.butler_repo is not None:
            params['butler_repo'] = args.butler_repo
        if args.collections is not None:
            params['fam_collections'] = args.collections
        if args.day_obs_min is not None:
            params['day_obs_min'] = args.day_obs_min
        if args.day_obs_max is not None:
            params['day_obs_max'] = args.day_obs_max
        if args.programs is not None:
            params['fam_programs'] = args.programs
    else:
        if not all([args.butler_repo, args.collections, args.programs]):
            parser.error("Must specify --param-set OR at least: "
                         "--butler-repo, --collections, --programs "
                         "(dates can be auto-parsed from collection name)")
        params = dict(
            butler_repo=args.butler_repo,
            fam_collections=args.collections,
            fam_programs=args.programs,
        )
        if args.day_obs_min is not None:
            params['day_obs_min'] = args.day_obs_min
        if args.day_obs_max is not None:
            params['day_obs_max'] = args.day_obs_max

    # Translate "0 or negative" sentinel values to None to disable the cut
    def _none_if_disabled(val):
        return None if (val is None or val <= 0) else val

    asyncio.run(run_mktable(
        butler_repo=params['butler_repo'],
        fam_collections=params['fam_collections'],
        fam_programs=params['fam_programs'],
        day_obs_min=params.get('day_obs_min'),
        day_obs_max=params.get('day_obs_max'),
        collection_phrase=args.collection_phrase,
        include_versions=args.include_versions,
        coord_sys=args.coord_sys,
        output_dir=args.output_dir,
        rotator_threshold=args.rotator_threshold,
        fp_radius=args.fp_radius,
        fp_nsteps=args.fp_nsteps,
        min_visits_per_day=args.min_visits_per_day,
        include_thermal=not args.no_thermal,
        calc_mean_zernike=args.calc_mean_zernike,
        calc_focal_plane=args.calc_focal_plane,
        temp_time_window_sec=args.temp_time_window,
        consdb_url=args.consdb_url,
        overwrite=args.overwrite,
        matched_threshold_arcsec=_none_if_disabled(args.matched_threshold_arcsec),
        min_donuts_per_visit=_none_if_disabled(args.min_donuts_per_visit),
        min_donuts_per_detector=args.min_donuts_per_detector,
        min_detectors_per_visit=_none_if_disabled(args.min_detectors_per_visit),
        max_median_blur_arcsec=_none_if_disabled(args.max_median_blur_arcsec),
    ))


if __name__ == '__main__':
    main()
