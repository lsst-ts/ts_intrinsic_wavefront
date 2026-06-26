#!/usr/bin/env python3
"""Stage a Measured-Intrinsic-Wavefront (MIW) calibration product.

Copies the OCS/CCS maps parquet from a pipeline output into the versioned
calibration store and writes a provenance record, so the calibration is frozen,
tracked, and self-describing — independent of later pipeline reruns.

    python stage_miw.py --param-set fam_danish_1_0_wep17_3_0_bin2x \
        --mi-name pathA_50_34_i_5rot --version v1

The version label is in the FILENAME (not a subdir) so it travels with the file
wherever it is deployed. Writes into calibration/miw/ :
    intrinsic_split_maps_<version>.parquet           (the handoff maps)
    intrinsic_split_maps_<version>.provenance.yaml   (source + git + meta)
    intrinsic_split_decomp_<version>.parquet         (only with --with-decomp)

After staging, commit the new files and tag the repo state, e.g.:
    git add calibration/miw/intrinsic_split_maps_<version>.*
    git commit -m "MIW calibration <version>: <param_set>/<mi_name>"
    git tag -a miw-<version> -m "MIW calibration <version>"

Pure-python (no LSST stack); needs astropy to read the parquet metadata.
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # <pkg>/calibration/
REPO = HERE.parent                               # ts_intrinsic_wavefront/ (git root)


def _git(*args):
    try:
        return subprocess.check_output(['git', '-C', str(REPO), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--version', required=True, help='calibration version label, e.g. v1')
    ap.add_argument('--output-root', default=str(REPO / 'pipelines' / 'output'))
    ap.add_argument('--with-decomp', action='store_true',
                    help='also stage the (larger) reconstruction decomp parquet')
    ap.add_argument('--force', action='store_true', help='overwrite an existing version')
    args = ap.parse_args()

    src_dir = Path(args.output_root) / args.param_set / args.mi_name
    src_maps = src_dir / 'intrinsic_split_maps.parquet'
    if not src_maps.exists():
        sys.exit(f'ERROR: {src_maps} not found (run the pipeline first).')

    # version is in the FILENAME (not a subdir) so it travels with the file
    dest = HERE / 'miw'
    dest.mkdir(parents=True, exist_ok=True)
    v = args.version
    maps_name = f'intrinsic_split_maps_{v}.parquet'
    if (dest / maps_name).exists() and not args.force:
        sys.exit(f'ERROR: {dest / maps_name} already exists (use --force to overwrite).')

    shutil.copy2(src_maps, dest / maps_name)
    staged = [maps_name]
    if args.with_decomp:
        src_dec = src_dir / 'intrinsic_split_decomp.parquet'
        if src_dec.exists():
            dec_name = f'intrinsic_split_decomp_{v}.parquet'
            shutil.copy2(src_dec, dest / dec_name)
            staged.append(dec_name)

    # provenance: source, git state, staging time, and the maps' own .meta
    try:
        from astropy.table import Table
        maps_meta = dict(Table.read(str(src_maps), format='parquet').meta)
    except Exception as e:
        maps_meta = {'(unread)': str(e)}
    prov = dict(
        version=args.version, param_set=args.param_set, mi_name=args.mi_name,
        source=str(src_maps.relative_to(REPO)) if src_maps.is_relative_to(REPO) else str(src_maps),
        staged_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        git_sha=_git('rev-parse', 'HEAD'),
        git_describe=_git('describe', '--tags', '--always', '--dirty'),
        staged_files=staged,
        maps_meta={k: v for k, v in maps_meta.items()},
    )
    prov_name = f'intrinsic_split_maps_{v}.provenance.yaml'
    with open(dest / prov_name, 'w') as fh:
        yaml.safe_dump(prov, fh, sort_keys=False, default_flow_style=False)
    staged.append(prov_name)

    print(f'staged {staged} -> {dest.relative_to(REPO)}')
    print(f'  source git: {prov["git_describe"]}')
    print('Next: git add these files, commit, and tag (see this script\'s docstring).')


if __name__ == '__main__':
    main()
