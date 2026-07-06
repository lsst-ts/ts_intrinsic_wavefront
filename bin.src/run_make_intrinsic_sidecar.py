#!/usr/bin/env python3
"""Build the per-donut measured-intrinsic sidecar — Phase 2 step 3.

For every donut in output/<param_set>/donuts.parquet, evaluate the measured
intrinsic Zernike vector from the OCS/CCS spin decomposition (step 2) plus the
per-donut CCD-height Z4, and write a **row-aligned** sidecar

    output/<param_set>/<mi_name>/zk_intrinsic.parquet

with one row per donut (same order as donuts.parquet), a list column
``zk_intrinsic_MI`` (ordered by ``nollIndices``), and key columns
(day_obs, seq_num, detector, centroid_x_intra, centroid_y_intra) so the refit
(step 4) can join robustly.

Reconstruction (per donut, at its visit's rotator angle theta) uses
intrinsic_split.reconstruct_at — the exact inverse of the decomposition:
``A(theta) = O_pol + exp(i*n_spin*s*theta)*roll(C_pol, shift)`` — so the
camera-fixed (CCS) component, including the azimuthal spin phase for
astig/coma/trefoil, is rotated back into OCS before interpolating at the
donut's OCS field position.  Z4 uses the height-removed optical intrinsic
(from step 2's z4_optical) plus the per-donut camera-height Z4.  RSP-only.
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import Delaunay

from lsst.ts.intrinsic.wavefront import intrinsic_split as isp
from lsst.ts.intrinsic.wavefront import mi_config as mc


def barycentric_weights(tri, pts):
    """Per-point simplex index + 3 barycentric weights on ``tri``.
    Points outside the hull get simplex -1 (-> NaN interpolation)."""
    simp = tri.find_simplex(pts)
    T = tri.transform[simp]
    b01 = np.einsum('nij,nj->ni', T[:, :2, :], pts - T[:, 2, :])
    bary = np.column_stack([b01, 1.0 - b01.sum(axis=1)])
    verts = tri.simplices[simp]
    return simp, verts, bary


def interp_field(field_ravel, simp, verts, bary):
    """Linear-interpolate a raveled field at precomputed barycentric pts."""
    out = np.where(simp >= 0,
                   (field_ravel[verts] * bary).sum(axis=1), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--coord-sys', default='OCS')
    ap.add_argument('--donuts', default=None,
                    help='donuts parquet (default <ps>/donuts.parquet); point at '
                         'wfs/donuts.parquet to build a corner-WFS MIW sidecar')
    ap.add_argument('--output', default=None,
                    help='sidecar output path (default <mi>/zk_intrinsic.parquet)')
    ap.add_argument('--wfs-corner-height', action='store_true',
                    help='corner-WFS Z4 height = mean(h(SW1,intra), h(SW0,extra)) — the two '
                         'correct half-sensors — instead of both centroids on one detector')
    ap.add_argument('--height-map-dir', default=None, help='override batoid_rubin height-map dir')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    b = cfg.get('build', {})
    coord = cfg.get('coord_sys', args.coord_sys)
    base = Path(args.output_root) / args.param_set
    mi_dir = base / args.mi_name

    # ---- decomposition (step 2): all-parquet astropy Table ----
    # One row per (j, part); O/C complex polar fields are flattened to
    # O_re/O_im/C_re/C_im (len n_r*n_az, C-order).  Polar grid A/X/Y in .meta.
    from astropy.table import Table
    dt = Table.read(str(mi_dir / 'intrinsic_split_decomp.parquet'), format='parquet')
    n_r = int(dt.meta['n_r'])
    n_az = int(dt.meta['n_az'])
    A = np.asarray(dt.meta['A'], dtype=float)
    X = np.asarray(dt.meta['X'], dtype=float).reshape(n_r, n_az)
    Y = np.asarray(dt.meta['Y'], dtype=float).reshape(n_r, n_az)
    jvals = [int(j) for j in dt['j']]
    nZk = len(jvals)

    def _fld(row, re_col, im_col):
        return (np.asarray(dt[re_col][row], dtype=float)
                + 1j * np.asarray(dt[im_col][row], dtype=float)).reshape(n_r, n_az)
    O_pol = np.stack([_fld(i, 'O_re', 'O_im') for i in range(nZk)])
    C_pol = np.stack([_fld(i, 'C_re', 'C_im') for i in range(nZk)])
    n_spin = np.asarray(dt['n_spin'], dtype=int)
    s_arr = np.asarray(dt['s'], dtype=int)
    part = np.asarray(dt['part'], dtype=int)
    j4_col = jvals.index(4) if 4 in jvals else None
    print(f'[sidecar] {args.param_set}/{args.mi_name}: {nZk} Zernikes {jvals}')

    # ---- donut light columns (file order) ----
    cols = ['day_obs', 'seq_num', 'detector',
            f'thx_{coord}', f'thy_{coord}',
            'centroid_x_intra', 'centroid_y_intra',
            'centroid_x_extra', 'centroid_y_extra']
    donuts_path = Path(args.donuts) if args.donuts else base / 'donuts.parquet'
    # prefer a per-donut rotator_angle column (present in wfs/donuts.parquet,
    # whose seq_num is the in-focus seq and won't match the FAM-keyed visits)
    if 'rotator_angle' in pq.read_schema(str(donuts_path)).names:
        cols.append('rotator_angle')
    dd = pq.read_table(str(donuts_path), columns=cols).to_pandas()
    rot_col = dd['rotator_angle'].to_numpy(float) if 'rotator_angle' in dd.columns else None
    N = len(dd)
    print(f'  donuts: {N}')

    # ---- rotator angle per visit (deg) from visits.parquet ----
    vt = pq.read_table(str(base / 'visits.parquet'),
                       columns=['day_obs', 'seq_num', 'rotator_angle']).to_pandas()
    rot_lut = {(int(r.day_obs), int(r.seq_num)): float(r.rotator_angle)
               for r in vt.itertuples()}

    # ---- per-donut camera-height Z4 (rotation-independent), once ----
    z4_height = np.full(N, np.nan)
    if j4_col is not None:
        try:
            from lsst.obs.lsst import LsstCam
            from lsst.ts.intrinsic.wavefront.ccd_height import (
                compute_ccd_heights, HEIGHT_TO_Z4_UM_PER_MM)
            cam = LsstCam.getCamera()
            fac = b.get('height_to_z4_factor') or HEIGHT_TO_Z4_UM_PER_MM
            hkw = dict(source=b.get('height_source', 'batoid_rubin'),
                       height_map_dir=(args.height_map_dir or b.get('batoid_rubin_height_map_dir')),
                       metrology_fits=b.get('height_map_fits'), factor=fac)
            if args.wfs_corner_height:
                # corner rafts: extra donut on <raft>_SW0, intra on <raft>_SW1;
                # evaluate each half on its own sensor and average the heights.
                dfe = dd.copy()
                dfi = dd.copy()
                dfi['detector'] = dd['detector'].astype(str).str.replace('SW0', 'SW1')
                he = np.asarray(compute_ccd_heights(dfe, cam, **hkw)['ccd_height_extra'], float)
                hi = np.asarray(compute_ccd_heights(dfi, cam, **hkw)['ccd_height_intra'], float)
                with np.errstate(invalid='ignore'):
                    z4_height = fac * np.nanmean(np.vstack([hi, he]), axis=0)
                print(f'  Z4_height (WFS corner, SW1-intra/SW0-extra): mean={np.nanmean(z4_height):+.4f} μm')
            else:
                z4_height = np.asarray(compute_ccd_heights(dd, cam, **hkw)['Z4_height'], float)
                print(f'  Z4_height: mean={np.nanmean(z4_height):+.4f} μm')
        except Exception as e:
            print(f'  Z4 height skipped: {type(e).__name__}: {e}')

    # ---- precompute interpolation weights (donut OCS field deg) once ----
    thx_deg = np.rad2deg(np.asarray(dd[f'thx_{coord}'], dtype=float))
    thy_deg = np.rad2deg(np.asarray(dd[f'thy_{coord}'], dtype=float))
    tri = Delaunay(np.column_stack([X.ravel(), Y.ravel()]))
    simp, verts, bary = barycentric_weights(tri, np.column_stack([thx_deg, thy_deg]))

    # ---- reconstruct per visit (fixed theta), fill row-aligned zk array ----
    zk_int = np.full((N, nZk), np.nan)
    dobs = np.asarray(dd['day_obs']).astype(int)
    snum = np.asarray(dd['seq_num']).astype(int)
    order = np.lexsort((snum, dobs))
    # group contiguous (dobs, snum)
    keys = list(zip(dobs[order], snum[order]))
    n_missing_rot = 0
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            idx = order[start:i]
            d, snv = keys[start]
            theta = float(rot_col[idx[0]]) if rot_col is not None else rot_lut.get((d, snv))
            if theta is not None and np.isfinite(theta):
                th = np.deg2rad(theta)
                for ij in range(nZk):
                    dec = {'O_pol': O_pol[ij], 'C_pol': C_pol[ij],
                           'n_spin': int(n_spin[ij]), 's': int(s_arr[ij])}
                    recon = isp.reconstruct_at(dec, th, A)
                    fld = recon.real if part[ij] == 0 else recon.imag
                    zk_int[idx, ij] = interp_field(
                        np.ascontiguousarray(fld).ravel(),
                        simp[idx], verts[idx], bary[idx])
            else:
                n_missing_rot += len(idx)
            start = i
    if j4_col is not None:
        # add CCD-height Z4 back, but don't let a missing (out-of-domain)
        # height NaN-poison an otherwise-finite optical Z4
        finite_h = np.isfinite(z4_height)
        zk_int[:, j4_col] = np.where(finite_h, zk_int[:, j4_col] + z4_height,
                                     zk_int[:, j4_col])
        n_missing_h = int((~finite_h).sum())
        if n_missing_h:
            print(f'  CCD-height Z4 missing for {n_missing_h} donuts '
                  f'(Z4 left optical-only there)')
    n_ok = int(np.isfinite(zk_int).all(axis=1).sum())
    print(f'  reconstructed {n_ok}/{N} donuts with a finite intrinsic '
          f'({n_missing_rot} donuts missing rotator)')

    # ---- write row-aligned sidecar ----
    out_tbl = pa.table({
        'day_obs': pa.array(dobs), 'seq_num': pa.array(snum),
        'detector': pa.array(dd['detector'].astype(str)),
        'centroid_x_intra': pa.array(np.asarray(dd['centroid_x_intra'], dtype=float)),
        'centroid_y_intra': pa.array(np.asarray(dd['centroid_y_intra'], dtype=float)),
        # Extra-focal centroids included so the per-donut join key (in
        # dz_fitting.load_intrinsic_lookup) is unique even if two donuts in the
        # same visit+detector share the same rounded intra centroid.
        'centroid_x_extra': pa.array(np.asarray(dd['centroid_x_extra'], dtype=float)),
        'centroid_y_extra': pa.array(np.asarray(dd['centroid_y_extra'], dtype=float)),
        'zk_intrinsic_MI': pa.array(list(zk_int)),
    })
    meta = {b'nollIndices': np.array(jvals, dtype=int).tobytes(),
            b'mi_name': args.mi_name.encode(), b'coord_sys': coord.encode(),
            b'n_rows': str(N).encode()}
    out_tbl = out_tbl.replace_schema_metadata(meta)
    out_path = Path(args.output) if args.output else mi_dir / 'zk_intrinsic.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, str(out_path), compression='snappy')
    print(f'  wrote {out_path} ({N} rows, zk_intrinsic_MI[{nZk}] ordered by {jvals})')


if __name__ == '__main__':
    main()
