#!/usr/bin/env python3
"""Decompose the per-rotator-bin measured intrinsic into OCS (telescope-fixed)
and CCS (camera-fixed) components — Phase 2 step 2.

Script port of the decomposition core of intrinsic_camera_telescope_split.ipynb
(which stays for interactive diagnostics).  Reads the per-rotator-bin grids
written by run_build_intrinsic.py, runs the spin-aware OCS/CCS decomposition
(intrinsic_split.py) per Zernike group, and writes three all-parquet (astropy
Table) products under output/<param_set>/<mi_name>/:

    intrinsic_split_maps.parquet    AOS handoff: thx_deg, thy_deg, Z{j}_OCS,
                                    Z{j}_CCS on the rot~0 regular disk-grid (um)
    intrinsic_split_decomp.parquet  load-bearing: complex polar O/C fields per
                                    (j, part) for per-donut reconstruction (read
                                    by run_make_intrinsic_sidecar.py)
    intrinsic_split_rms.parquet     per-Zernike telescope/camera/residual RMS

Config knobs (mi_config.yaml `split:` block, per entry):
  * rotator_select  — subset of rotator bins to decompose (drop an out-of-family
                      epoch that sits at distinct rotator angles); no rebuild.
  * split_js        — Noll j to keep a FULL OCS/CCS split; all other j become
                      OCS-only (camera term forced to 0 in the outputs).
  * ocs_only        — explicit Noll j to force OCS-only.
  * build_from      — reuse a parent entry's already-built grids (no rebuild).
RSP-only (numpy/scipy + intrinsic_split).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lsst.ts.intrinsic.wavefront import intrinsic_split as isp
from lsst.ts.intrinsic.wavefront import mi_config as mc

DEFAULT_NOLL = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                22, 23, 24, 25, 26]
DEFAULT_SPLIT = dict(rotation_sign=1, spin_sign=1, m_max=12,
                     degen_assignment='ocs', ridge=1e-3, hole_dist=0.06,
                     r_min=0.06, r_max=1.75, n_r=80, n_az=180,
                     r_lim_lo=0.1, r_lim_hi=1.6, use_z4_optical=True)


def _map_rms(vals_pol, R, r_lim):
    # Ring-limited RMS over a (..., n_r, n_az) polar field.  R is the 1-D radial
    # array, so the mask applies to the radial axis (-2), not axis 0 — the data
    # field is stacked per rotator bin (n_bins, n_r, n_az).  Delegate to the lib
    # helper, which masks [..., m, :] and ignores NaNs.
    return isp.residual_rms(np.real(vals_pol), R, r_lim)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    sp = {**DEFAULT_SPLIT, **(cfg.get('split') or {})}
    noll_list = cfg.get('noll_list') or DEFAULT_NOLL
    alt_range = (cfg.get('alt_min_deg'), cfg.get('alt_max_deg'))

    # Optional rotator-bin subset: decompose only these of the entry's bins (the
    # build grids for ALL bins still exist; e.g. to drop an out-of-family epoch
    # that occupies distinct rotator angles).  Shared helper validates the subset.
    sel = mc.rotator_select(cfg)
    rot_bins = mc.selected_rotator_bins(cfg)
    if sel is not None:
        print(f'[intrinsic_split] rotator_select: using {len(rot_bins)}/'
              f'{len(mc.rotator_bins(cfg))} bins {rot_bins}')

    # Per-Zernike OCS-only set: force the camera (CCS) term to zero in the outputs.
    ocs_only = mc.ocs_only_js(cfg, noll_list)
    if ocs_only:
        print(f'[intrinsic_split] OCS-only (C forced to 0): {sorted(ocs_only)}')

    # Read the per-rotator-bin grids from the BUILD SOURCE entry (build_from lets
    # a derived entry reuse a parent's already-built grids — no rebuild).
    src_mi = mc.build_source(cfg, args.mi_name)
    base = Path(args.output_root) / args.param_set / args.mi_name
    src_base = Path(args.output_root) / args.param_set / src_mi
    if src_mi != args.mi_name:
        print(f'[intrinsic_split] reading build grids from source mi={src_mi!r}')
        # the bins we decompose must have been built by the source entry
        src_bins = {(round(lo, 3), round(hi, 3))
                    for lo, hi in mc.rotator_bins(mc.load_mi_config(args.param_set, src_mi))}
        extra = {(round(lo, 3), round(hi, 3)) for lo, hi in rot_bins} - src_bins
        if extra:
            raise ValueError(f'build_from={src_mi!r}: rotator bins {sorted(extra)} are not '
                             f'among the source entry\'s built bins')
    fits_path = Path(args.output_root) / args.param_set / 'fits.parquet'
    fits_table = (pd.read_parquet(fits_path,
                                  columns=['rotator_angle', 'alt', 'day_obs', 'seq_num'])
                  if fits_path.exists() else None)

    # ---- load each rotator-bin grid + its mean rotator angle ----
    dsets, thetas, labels = [], [], []
    for lo, hi in rot_bins:
        gp = src_base / 'build' / f'rot_{lo:g}_{hi:g}' / 'intrinsic_grid.parquet'
        if not gp.exists():
            raise FileNotFoundError(gp)
        df = pd.read_parquet(gp)
        zk = np.vstack(df['zk'].values)
        nolls = [int(j) for j in np.asarray(df['nollIndices'][0]).tolist()]
        rec = {'thx': df['thx_deg'].values, 'thy': df['thy_deg'].values}
        if 'z4_optical_OCS' in df.columns:
            rec['z4opt'] = df['z4_optical_OCS'].values
        for j in noll_list:
            rec[f'z{j}'] = zk[:, nolls.index(j)]
        dsets.append(rec)
        center = 0.5 * (lo + hi)
        if fits_table is not None:
            mr, *_ = isp.mean_rotator(fits_table, (lo, hi), alt_range)
            theta = mr if np.isfinite(mr) else center
        else:
            theta = center
        thetas.append(theta)
        labels.append(f'rot{theta:+.0f}')
    thetas = np.array(thetas)
    th_rad = np.deg2rad(thetas)
    # distinct rotator angles drive the O/C separability — a full split needs >=2
    n_rot_distinct = int(np.unique(np.round(thetas, 1)).size)
    print(f'[intrinsic_split] {args.param_set}/{args.mi_name}: '
          f'{len(dsets)} rotator bins, thetas={np.round(thetas,2).tolist()} '
          f'({n_rot_distinct} distinct)')

    R, A, X, Y = isp.make_polar_grid(r_min=sp['r_min'], r_max=sp['r_max'],
                                     n_r=sp['n_r'], n_az=sp['n_az'])
    r_lim = (sp['r_lim_lo'], sp['r_lim_hi'])

    def sample_field(key):
        maps = []
        for d in dsets:
            v = d[key]
            f = np.isfinite(v)
            maps.append((d['thx'][f], d['thy'][f], v[f]))
        Z, valid, _ = isp.sample_maps_polar(maps, X, Y, hole_dist=sp['hole_dist'])
        return np.nan_to_num(Z), valid

    # Field-rotation sense s (fixed; see split.rotation_sign in mi_config.yaml).
    #
    # s=+1 is the empirically-correct OCS->CCS convention, determined directly
    # from the per-donut field angles in donuts.parquet (which come straight
    # from the Butler aggregateAOSVisitTableRaw): at a nonzero rotator angle
    # theta, the OCS and CCS field angles of the same donut satisfy
    #     (thx,thy)_CCS = R(-theta) . (thx,thy)_OCS,  R(-th)=[[c, s],[-s, c]]
    # i.e. the CCS azimuth phi_CCS = phi_OCS - theta (verified at theta=+/-60 deg:
    # residual ~6e-5 rad for R(-theta) vs ~3.5e-2 rad for R(+theta)).  The
    # decomposition models the camera term as C(r, phi - s*theta) (CCS azimuth
    # psi = phi - s*theta), so phi_CCS = phi_OCS - s*theta matches the data at
    # s=+1.  No sign search is needed; this is the convention in use.
    s = int(sp['rotation_sign'])
    print(f's (field rotation) = {s:+d} (fixed, OCS->CCS = R(-theta))')

    # ---- per-group decomposition ----
    groups = isp.group_zernikes(noll_list)
    dec_by_j, metrics = {}, []
    def _decomp(Zc, V, n_spin, ocs):
        return isp.decompose_spin_lsq(Zc, V, th_rad, A, n_spin=n_spin, s=s,
                                      m_max=sp['m_max'], ridge=sp['ridge'],
                                      degen_assignment=sp['degen_assignment'],
                                      ocs_only=ocs)

    for grp in groups:
        n_spin = sp['spin_sign'] * grp['spin']
        if grp['kind'] == 'single':
            ocs = grp['j'] in ocs_only
            key = ('z4opt' if (grp['j'] == 4 and sp['use_z4_optical']
                               and 'z4opt' in dsets[0]) else f"z{grp['j']}")
            Z, V = sample_field(key)
            Zc = Z.astype(complex)
            comps = [(grp['j'], np.real)]
        elif grp['kind'] == 'pair':
            # a doublet shares one dec; OCS-only if either member is listed
            ocs = (grp['j_cos'] in ocs_only) or (grp['j_sin'] in ocs_only)
            Zc_, Vc_ = sample_field(f"z{grp['j_cos']}")
            Zs_, Vs_ = sample_field(f"z{grp['j_sin']}")
            Zc = Zc_ + 1j * Zs_
            V = Vc_ & Vs_
            comps = [(grp['j_cos'], np.real), (grp['j_sin'], np.imag)]
        else:
            print(f"  (skipping unpaired {grp['label']})")
            continue

        # A full O/C split needs >=2 distinct rotator angles to separate the
        # telescope and camera terms; with fewer the system is degenerate (the
        # ridge would pick an arbitrary split).  Force OCS-only and warn.
        if not ocs and n_rot_distinct < 2:
            print(f"  WARNING: {grp['label']} requested a full OCS/CCS split but only "
                  f"{n_rot_distinct} distinct rotator angle(s) — degenerate; forcing OCS-only")
            ocs = True

        dec = _decomp(Zc, V, n_spin, ocs)
        # Diagnostic camera RMS: for OCS-only j the output C is constrained to 0,
        # so also run the full split (when non-degenerate) purely to report the
        # camera RMS that was given up — informs whether OCS-only is justified.
        c_dec = _decomp(Zc, V, n_spin, False) if (ocs and n_rot_distinct >= 2) else dec

        for j, part in comps:
            dec_by_j[j] = (dec, part, Zc, ocs)
            metrics.append(dict(
                Zernike=f'Z{j}', j=j, spin=abs(dec['n_spin']),
                dataRMS=_map_rms(part(Zc), R, r_lim),
                O_tel=_map_rms(part(dec['O_pol']), R, r_lim),
                # C_cam is the camera RMS the *full* split yields (diagnostic);
                # for OCS-only j the OUTPUT C is 0 but this shows what was given up.
                C_cam=_map_rms(part(c_dec['C_pol']), R, r_lim),
                residRMS=_map_rms(part(dec['res']), R, r_lim),
                ocs_only=bool(ocs)))

    metrics_df = pd.DataFrame(metrics).sort_values('j').reset_index(drop=True)
    print(f"  overall |O| tel RMS={np.sqrt((metrics_df['O_tel']**2).mean()):.4f}  "
          f"|C| cam RMS={np.sqrt((metrics_df['C_cam']**2).mean()):.4f} um")

    base.mkdir(parents=True, exist_ok=True)
    i0 = int(np.argmin(np.abs(thetas)))
    thx0, thy0 = _complete_disk_grid(dsets[i0]['thx'], dsets[i0]['thy'])
    meta = dict(rotation_sign=int(s), degen_assignment=str(sp['degen_assignment']),
                thetas_deg=[float(t) for t in np.round(thetas, 3)],
                n_bins_used=int(len(thetas)),
                rotator_select=([list(b) for b in sel] if sel is not None else None),
                ocs_only=sorted(int(j) for j in ocs_only),
                build_from=(src_mi if src_mi != args.mi_name else None))
    _write_outputs(base, dec_by_j, noll_list, metrics_df, A, X, Y,
                   thx0, thy0, meta)

    # ---- diagnostic plots (RMS summary + per-Zernike O/C/residual maps) ----
    _write_split_pdf(base / 'intrinsic_split.pdf', metrics_df, dec_by_j,
                     noll_list, thetas, labels, X, Y, R, r_lim, s)
    print('[intrinsic_split] done.')


# ----------------------------------------------------------------------
# output tables (all-parquet astropy Tables)
# ----------------------------------------------------------------------
def _complete_disk_grid(thx, thy):
    """Rebuild the full Cartesian lattice (masked to the field radius) from a
    hole-punched build grid.

    The build grid is a regular square lattice masked to a disk, but with cells
    dropped wherever a CCD had no donuts.  This reconstructs the underlying
    lattice from its own spacing/extent and re-applies only the radius mask, so
    the no-donut cells are put back as output sample points (their OCS/CCS values
    are then filled by evaluating the decomposition there).  Returns the flat
    ``(thx0, thy0)`` disk grid.
    """
    thx = np.asarray(thx, dtype=float)
    thy = np.asarray(thy, dtype=float)
    ux = np.unique(np.round(thx, 4))
    step = float(np.median(np.diff(ux)))
    lo = min(float(ux.min()), float(np.round(thy, 4).min()))
    hi = max(float(ux.max()), float(np.round(thy, 4).max()))
    n = int(round((hi - lo) / step)) + 1
    axis = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(axis, axis)
    R = float(np.hypot(thx, thy).max())
    keep = np.hypot(gx, gy) <= R + 1e-9
    return gx[keep].ravel(), gy[keep].ravel()


def _write_outputs(base, dec_by_j, noll_list, metrics_df, A, X, Y,
                   thx0, thy0, meta):
    """Write the three parquet products (all astropy Tables):

      intrinsic_split_maps.parquet   — AOS handoff: thx_deg, thy_deg and per j
          Z{j}_OCS / Z{j}_CCS sampled on the rot~0 regular disk-grid (um).
          For OCS-only j the camera term is zero (C was constrained out of the
          fit, so O is the C=0 telescope map).  meta carries the split params.
      intrinsic_split_decomp.parquet — load-bearing (sidecar reconstruction):
          one row per (j, part) with the complex polar O/C fields flattened to
          O_re/O_im/C_re/C_im list-columns (len n_r*n_az, C-order), plus
          j, n_spin, s, part, n_r, n_az.  Polar grid A/X/Y in meta.  C is zero
          for OCS-only j (so reconstruct_at yields O-only).
      intrinsic_split_rms.parquet    — per-Zernike diagnostic RMS (was the .csv).
    """
    from astropy.table import Table

    n_r, n_az = np.asarray(X).shape
    jorder = [j for j in noll_list if j in dec_by_j]

    # ---- maps table (rot~0 bin's regular disk-grid field points) ----
    # dec['C_pol'] is already zero for OCS-only j (constrained fit), so we sample
    # O and C straight from the decomposition with no special-casing here.
    thx0 = np.asarray(thx0, dtype=np.float64)
    thy0 = np.asarray(thy0, dtype=np.float64)
    cols = {'thx_deg': thx0, 'thy_deg': thy0}
    for j in jorder:
        dec, part, _, is_ocs = dec_by_j[j]
        O_pts = isp.polar_field_to_points(part(dec['O_pol']), X, Y, thx0, thy0)
        C_pts = isp.polar_field_to_points(part(dec['C_pol']), X, Y, thx0, thy0)
        cols[f'Z{j}_OCS'] = np.asarray(O_pts, dtype=np.float64)
        cols[f'Z{j}_CCS'] = np.asarray(C_pts, dtype=np.float64)
    maps = Table(cols)
    maps.meta.update({k: v for k, v in meta.items() if v is not None})
    maps.write(str(base / 'intrinsic_split_maps.parquet'),
               format='parquet', overwrite=True)

    # ---- decomp table (complex polar fields, per (j, part)) ----
    rows = dict(j=[], n_spin=[], s=[], part=[], n_r=[], n_az=[],
                O_re=[], O_im=[], C_re=[], C_im=[])
    for j in jorder:
        dec, part, _, is_ocs = dec_by_j[j]
        O = np.asarray(dec['O_pol'])
        C = np.asarray(dec['C_pol'])   # already zero for OCS-only j (constrained)
        rows['j'].append(int(j))
        rows['n_spin'].append(int(dec.get('n_spin', 0)))
        rows['s'].append(int(dec['s']))
        rows['part'].append(0 if part is np.real else 1)
        rows['n_r'].append(int(n_r)); rows['n_az'].append(int(n_az))
        rows['O_re'].append(O.real.ravel().astype(np.float64))
        rows['O_im'].append(O.imag.ravel().astype(np.float64))
        rows['C_re'].append(C.real.ravel().astype(np.float64))
        rows['C_im'].append(C.imag.ravel().astype(np.float64))
    decomp = Table(rows)
    # Polar grid is identical for every row -> store once in meta (lists so the
    # astropy parquet writer round-trips them via pyarrow metadata).
    decomp.meta.update(dict(
        A=[float(a) for a in np.asarray(A).ravel()],
        X=[float(x) for x in np.asarray(X).ravel()],
        Y=[float(y) for y in np.asarray(Y).ravel()],
        n_r=int(n_r), n_az=int(n_az),
        rotation_sign=int(meta['rotation_sign'])))
    decomp.write(str(base / 'intrinsic_split_decomp.parquet'),
                 format='parquet', overwrite=True)

    # ---- rms diagnostic table ----
    Table.from_pandas(metrics_df).write(
        str(base / 'intrinsic_split_rms.parquet'),
        format='parquet', overwrite=True)

    print(f'  wrote intrinsic_split_maps.parquet ({len(maps)} rows x '
          f'{len(maps.colnames)} cols), intrinsic_split_decomp.parquet '
          f'({len(decomp)} (j,part) records), intrinsic_split_rms.parquet')


# ----------------------------------------------------------------------
# diagnostic plots (port of intrinsic_camera_telescope_split.ipynb maps)
# ----------------------------------------------------------------------
_SPLIT_DPI = 120        # rasterized-contour resolution (keeps the PDF small)


def _plot_field(ax, X, Y, vals, title, vlim, levels=21, cmap='RdBu_r'):
    """Filled-contour map of a polar-grid field on its (X, Y) points.  The
    filled contours are rasterized so the multi-panel PDF stays small."""
    import numpy as np
    xr, yr, vr = X.ravel(), Y.ravel(), np.asarray(vals).ravel()
    fin = np.isfinite(vr)
    im = ax.tricontourf(xr[fin], yr[fin], vr[fin],
                        levels=np.linspace(-vlim, vlim, levels),
                        cmap=cmap, extend='both')
    im.set_rasterized(True)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=8)
    ax.set_xlabel('thx [deg]', fontsize=7); ax.set_ylabel('thy [deg]', fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def _abs_vlim(Z, R, r_lim, pct=98):
    import numpy as np
    m = (R >= r_lim[0]) & (R <= r_lim[1])
    return max(float(np.nanpercentile(np.abs(np.asarray(Z)[..., m, :]), pct)), 1e-6)


def _maps_page(fields, titles, suptitle, X, Y, vlim, unit, ncols=3):
    """A grid of polar-field maps (one per entry) with a shared colorbar."""
    import numpy as np
    import matplotlib.pyplot as plt
    n = len(fields)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.5 * nrows),
                            layout='constrained', squeeze=False)
    flat = axs.ravel()
    im = None
    for p in range(len(flat)):
        if p < n:
            im = _plot_field(flat[p], X, Y, fields[p], titles[p], vlim)
        else:
            flat[p].set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label=unit)
    fig.suptitle(suptitle, fontsize=13)
    return fig


def _zernike_pages(part, dec, Zc, thetas, labels, X, Y, R, r_lim, zlabel):
    """Three pages per Zernike: (1) all per-rotator-bin data maps, (2) the
    OCS / CCS decomposition + the rot~0 model, (3) all per-rotator-bin
    residual maps.  Every panel title carries its in-window map RMS."""
    import numpy as np
    O = part(dec['O_pol']); C = part(dec['C_pol'])
    data = part(np.asarray(Zc)); resid = part(dec['res'])
    n = len(labels)
    i0 = int(np.argmin(np.abs(np.asarray(thetas))))      # rot~0 bin
    model0 = data[i0] - resid[i0]
    vlim_d = _abs_vlim(data, R, r_lim)
    vlim_r = _abs_vlim(resid, R, r_lim)                  # residuals scaled to self
    spin = f'(spin n={dec["n_spin"]}, s={dec["s"]:+d})'
    unit = f'{zlabel} [μm]'

    dtitles = [f'{labels[i]}  rot={thetas[i]:+.0f}\nRMS={_map_rms(data[i], R, r_lim):.4f}'
               for i in range(n)]
    pg1 = _maps_page([data[i] for i in range(n)], dtitles,
                     f'{zlabel} data — per rotator bin {spin}', X, Y, vlim_d, unit)

    btitles = [f'O telescope (OCS)\n|O| RMS={_map_rms(O, R, r_lim):.4f}',
               f'C camera (CCS)\n|C| RMS={_map_rms(C, R, r_lim):.4f}',
               f'model rot={thetas[i0]:+.0f} (data-res)\n'
               f'RMS={_map_rms(model0, R, r_lim):.4f}']
    pg2 = _maps_page([O, C, model0], btitles,
                     f'{zlabel}: OCS telescope + CCS camera + rot~0 model {spin}',
                     X, Y, vlim_d, unit, ncols=3)

    rtitles = [f'{labels[i]}  rot={thetas[i]:+.0f}\nRMS={_map_rms(resid[i], R, r_lim):.4f}'
               for i in range(n)]
    pg3 = _maps_page([resid[i] for i in range(n)], rtitles,
                     f'{zlabel} residual — per rotator bin {spin}  '
                     f'(colour ±{vlim_r:.3f} μm)', X, Y, vlim_r, unit)
    return [pg1, pg2, pg3]


def _write_split_pdf(path, metrics_df, dec_by_j, noll_list, thetas, labels,
                     X, Y, R, r_lim, s):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    n_pages = 0
    with PdfPages(str(path)) as pdf:
        # telescope (O) vs camera (C) map RMS per Zernike
        d = metrics_df
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(16, 5.5), layout='constrained')
        ax.bar(x - 0.21, d['O_tel'], 0.4, label='|O| telescope', color='steelblue')
        ax.bar(x + 0.21, d['C_cam'], 0.4, label='|C| camera', color='indianred')
        ax.plot(x, d['dataRMS'], 'k_', ms=12, label='data RMS')
        ax.plot(x, d['residRMS'], 'kx', ms=6, label='residual RMS')
        ax.set_xticks(x); ax.set_xticklabels(d['Zernike'], rotation=45)
        ax.set_ylabel('map RMS [μm]')
        ax.set_title('Telescope (O) vs camera (C) map RMS per Zernike  '
                     f'(in-window r={r_lim[0]:.1f}..{r_lim[1]:.1f} deg)')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig); n_pages += 1

        for j in noll_list:
            if j not in dec_by_j:
                continue
            dec, part, Zc, _ = dec_by_j[j]
            for fig in _zernike_pages(part, dec, Zc, thetas, labels,
                                      X, Y, R, r_lim, f'Z{j}'):
                pdf.savefig(fig, bbox_inches='tight', dpi=_SPLIT_DPI)
                plt.close(fig); n_pages += 1
    print(f'  wrote intrinsic_split.pdf ({n_pages} pages)')


if __name__ == '__main__':
    main()
