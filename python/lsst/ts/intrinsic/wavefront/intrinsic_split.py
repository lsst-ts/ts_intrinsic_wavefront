"""Decompose a rotator-dependent intrinsic field map into camera-fixed (CCS)
and telescope-fixed (OCS) components.

A measured intrinsic map (e.g. ``z4_optical_OCS``) sampled at OCS field
position ``(r, phi)`` from data taken at rotator angle ``theta`` is modelled
as the sum of a telescope-fixed map ``O`` (OCS) and a camera-fixed map ``C``
that rotates rigidly with the rotator (CCS)::

    Z_theta(r, phi) = O(r, phi) + C(r, phi - s*theta)

Rotation is diagonal in the azimuthal Fourier basis, so per radius ``r`` and
azimuthal mode ``m``::

    A_{d,m}(r) = O_m(r) + C_m(r) * exp(-i m s theta_d)

Given several rotator datasets this is a 2-unknown complex least-squares
solve per ``(r, m)``.  ``m = 0`` (the axisymmetric profile) is degenerate —
an axisymmetric pattern is identical in both frames — so it is assigned by a
flag (``'ocs'``, ``'ccs'`` or ``'split'``).

Intended for scalar (rotationally-invariant) Zernikes such as Z4 defocus,
whose coefficient value does not pick up phase under field rotation.

numpy + scipy only.
"""
from __future__ import annotations

import numpy as np


# Noll index -> (radial n, signed azimuthal m).  Even-|m| split into a
# cosine (m>0) and sine (m<0) partner; the spin of a term is |m|.
NOLL_NM = {
    4: (2, 0), 5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
    9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2), 13: (4, -2),
    14: (4, 4), 15: (4, -4), 16: (5, 1), 17: (5, -1), 18: (5, 3),
    19: (5, -3), 20: (5, 5), 21: (5, -5), 22: (6, 0), 23: (6, -2),
    24: (6, 2), 25: (6, -4), 26: (6, 4),
}


def group_zernikes(noll_list):
    """Group a list of Noll indices into decomposition units.

    Returns a list of dicts.  An m=0 term is a scalar singlet
    ``{'kind':'single','spin':0,'j':j,'label':'Zj'}``; a (cos,sin) doublet
    with the same (n, |m|) is ``{'kind':'pair','spin':|m|,'j_cos':jc,
    'j_sin':js,'label':'Zjc/Zjs'}`` — the complex field is
    ``Z_cos + i*Z_sin`` (vertical + i*oblique), which rotates as spin |m|.
    Unpaired non-zero-m terms are skipped with a note in the dict.
    """
    s = set(int(j) for j in noll_list)
    groups, used = [], set()
    for j in sorted(s):
        if j in used:
            continue
        n, m = NOLL_NM[j]
        if m == 0:
            groups.append({'kind': 'single', 'spin': 0, 'j': j,
                           'label': f'Z{j}'})
            used.add(j)
            continue
        # find the opposite-sign partner (same n, same |m|)
        partner = next((k for k in s if k not in used and k != j
                        and NOLL_NM[k] == (n, -m)), None)
        if partner is None:
            groups.append({'kind': 'orphan', 'spin': abs(m), 'j': j,
                           'label': f'Z{j}'})
            used.add(j)
            continue
        j_cos, j_sin = (j, partner) if m > 0 else (partner, j)
        groups.append({'kind': 'pair', 'spin': abs(m),
                       'j_cos': j_cos, 'j_sin': j_sin,
                       'label': f'Z{j_cos}/Z{j_sin}'})
        used.update((j, partner))
    return groups


def decompose_spin_fft(Zc, thetas_rad, A, n_spin=0, s=1,
                       degen_assignment='ocs'):
    """Spin-aware camera/telescope split of a (possibly complex) field.

    Model (azimuthal Fourier, per radius):
        A_{d,m} = O_m + C_m * exp(i (n_spin - m) s theta_d)
    where the spin ``n_spin`` is the Zernike azimuthal order (0 for a
    scalar like Z4 — then this reduces to :func:`decompose_polar`; |m| for
    an astig/coma/... doublet whose complex field rotates as that spin).
    The degenerate mode is now ``m = n_spin`` (where the phase is 1 for all
    datasets) and is assigned by ``degen_assignment``.

    ``Zc`` is (n_set, n_r, n_az) real (scalar) or complex (doublet,
    ``Z_cos + i Z_sin``).  Returns dict with complex ``O_pol``, ``C_pol``
    (n_r, n_az), per-dataset complex ``res``, ``n_spin``, ``s``.
    """
    Zc = np.asarray(Zc, complex)
    n_set, n_r, n_az = Zc.shape
    thetas_rad = np.asarray(thetas_rad, float)
    F = np.fft.fft(Zc, axis=2)
    mk = np.rint(np.fft.fftfreq(n_az, d=1.0 / n_az)).astype(int)
    O = np.zeros((n_r, n_az), complex)
    C = np.zeros_like(O)
    for k in range(n_az):
        m = int(mk[k])
        if m == int(n_spin):
            tot = F[:, :, k].mean(0)
            if degen_assignment == 'ccs':
                C[:, k] = tot
            elif degen_assignment == 'split':
                O[:, k] = tot / 2
                C[:, k] = tot / 2
            else:
                O[:, k] = tot
            continue
        ph = np.exp(1j * (int(n_spin) - m) * s * thetas_rad)
        D = np.stack([np.ones_like(ph), ph], axis=1)
        sol, *_ = np.linalg.lstsq(D, F[:, :, k], rcond=None)
        O[:, k] = sol[0]
        C[:, k] = sol[1]
    O_pol = np.fft.ifft(O, axis=1)
    C_pol = np.fft.ifft(C, axis=1)
    dphi = A[1] - A[0]
    res = np.empty_like(Zc)
    for d, th in enumerate(thetas_rad):
        shift = int(np.round(s * th / dphi))
        res[d] = Zc[d] - (O_pol + np.exp(1j * int(n_spin) * s * th)
                          * np.roll(C_pol, shift, axis=1))
    return {'O_pol': O_pol, 'C_pol': C_pol, 'res': res,
            'n_spin': int(n_spin), 's': int(s),
            'degen_assignment': degen_assignment}


def make_polar_grid(r_min=0.06, r_max=1.70, n_r=40, n_az=180):
    """Common polar sampling grid.  Returns (R, A, X, Y) where X/Y are the
    (n_r, n_az) Cartesian field coords of each polar node."""
    R = np.linspace(r_min, r_max, n_r)
    A = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    RR, AA = np.meshgrid(R, A, indexing='ij')
    return R, A, RR * np.cos(AA), RR * np.sin(AA)


def sample_maps_polar(maps, X, Y, hole_dist=None):
    """Interpolate each scattered map onto the polar grid (X, Y).

    `maps` is a list of (thx, thy, val) arrays (finite values).  Returns
    ``(Z, valid, n_nan)``: ``Z`` (n_set, n_r, n_az) is the interpolated
    value (NaN outside each map's hull); ``valid`` is a boolean mask of the
    same shape that is True where the node has real support.

    If ``hole_dist`` is given (deg), a node is invalid when the nearest
    actual grid point is farther than ``hole_dist`` — i.e. it sits in a
    dead-detector hole and the interpolated value spans the gap.  This lets
    the decomposition fit only real samples (so the telescope map O is not
    corrupted by holes), while the camera map C still gets a smooth
    reconstruction across the camera-fixed dead regions.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    Zs, Vs = [], []
    for thx, thy, val in maps:
        pts = np.column_stack([thx, thy])
        z = LinearNDInterpolator(pts, val)(X, Y)
        v = np.isfinite(z)
        if hole_dist is not None:
            d, _ = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]))
            v = v & (d.reshape(X.shape) <= hole_dist)
        Zs.append(z); Vs.append(v)
    Z = np.stack(Zs)
    return Z, np.stack(Vs), int(np.isnan(Z).sum())


def decompose_polar_lsq(Z, valid, thetas_rad, A, s=1, m0_assignment='ocs',
                        m_max=12, ridge=1e-3):
    """Hole-aware decomposition: per radius, fit O and C jointly by
    least-squares over the *valid* (dataset, azimuth) samples only.

    Same model as :func:`decompose_polar` but masked, so dead-detector
    holes never enter the fit.  Because the holes are camera-fixed and
    rotate through OCS, every OCS azimuth is sampled by some dataset, so
    the reconstructed telescope map ``O`` is hole-free; the camera-dead
    CCS regions are filled by the smooth Fourier reconstruction of ``C``.

    `m_max` caps the azimuthal order (per ring there are 1 + 4*m_max real
    unknowns).  `ridge` is a small Tikhonov factor (relative to the mean
    diagonal of DᵀD) that damps azimuthal modes the ring's coverage cannot
    constrain — prevents Fourier ringing on partially-covered outer rings
    while leaving well-determined modes essentially unchanged.  Returns the
    same dict as :func:`decompose_polar`, with ``res`` NaN at invalid nodes.
    """
    n_set, n_r, n_az = Z.shape
    thetas_rad = np.asarray(thetas_rad, float)
    M = int(m_max)
    ms = np.arange(1, M + 1)
    O_pol = np.full((n_r, n_az), np.nan)
    C_pol = np.full((n_r, n_az), np.nan)
    res = np.full_like(Z, np.nan)
    # azimuth grids per dataset for reconstruction
    for k in range(n_r):
        rows, rhs, smp = [], [], []   # design rows, data, (d, j)
        for d in range(n_set):
            jj = np.nonzero(valid[d, k, :])[0]
            if jj.size == 0:
                continue
            phi = A[jj]
            psi = phi - s * thetas_rad[d]          # CCS azimuth
            cols = [np.ones_like(phi)]             # const (degenerate m=0)
            for m in ms:
                cols += [np.cos(m * phi), np.sin(m * phi)]      # O
            for m in ms:
                cols += [np.cos(m * psi), np.sin(m * psi)]      # C
            rows.append(np.column_stack(cols))
            rhs.append(Z[d, k, jj]); smp.append((d, jj))
        if not rows:
            continue
        D = np.vstack(rows); y = np.concatenate(rhs)
        if D.shape[0] < D.shape[1]:                # under-determined ring
            continue
        DtD = D.T @ D
        lam = ridge * np.trace(DtD) / DtD.shape[0]
        coef = np.linalg.solve(DtD + lam * np.eye(DtD.shape[0]), D.T @ y)
        const = coef[0]
        Ocs = coef[1:1 + 2 * M].reshape(M, 2)
        Ccs = coef[1 + 2 * M:1 + 4 * M].reshape(M, 2)
        if m0_assignment == 'ccs':
            cO, cC = 0.0, const
        elif m0_assignment == 'split':
            cO = cC = const / 2
        else:
            cO, cC = const, 0.0
        Orow = cO + sum(Ocs[i, 0] * np.cos(ms[i] * A) + Ocs[i, 1] * np.sin(ms[i] * A)
                        for i in range(M))
        Crow = cC + sum(Ccs[i, 0] * np.cos(ms[i] * A) + Ccs[i, 1] * np.sin(ms[i] * A)
                        for i in range(M))
        O_pol[k] = Orow; C_pol[k] = Crow
        dphi = A[1] - A[0]
        for (d, jj) in smp:
            shift = int(np.round(s * thetas_rad[d] / dphi))
            pred = Orow + np.roll(Crow, shift)
            res[d, k, jj] = Z[d, k, jj] - pred[jj]
    return {'O_pol': O_pol, 'C_pol': C_pol, 'res': res,
            's': int(s), 'm0_assignment': m0_assignment, 'm_max': M}


def decompose_polar(Z, thetas_rad, A, s=1, m0_assignment='ocs', m_max=None):
    """Camera/telescope decomposition on a polar grid.

    Parameters
    ----------
    Z : ndarray (n_set, n_r, n_az)
        Maps sampled on the polar grid (NaNs treated as 0).
    thetas_rad : ndarray (n_set,)
        Rotator angle of each dataset (radians).
    A : ndarray (n_az,)
        Azimuth samples (used only for the roll step).
    s : int
        Rotation sense (+1 or -1).
    m0_assignment : {'ocs', 'ccs', 'split'}
        Where to put the degenerate axisymmetric (m=0) component.
    m_max : int or None
        If set, azimuthal modes above this are assigned entirely to ``O``
        (treated as rotation-independent) rather than fit — suppresses
        high-m camera noise.

    Returns
    -------
    dict with O_pol, C_pol (n_r, n_az), res (n_set, n_r, n_az), s,
    m0_assignment.
    """
    n_nan = int(np.isnan(Z).sum())
    if n_nan:
        print(f'  WARNING: decompose_polar (FFT path) zero-filled {n_nan} NaN '
              f'samples — NOT hole-aware; use method="lsq" for holey maps')
    Zc = np.nan_to_num(Z, nan=0.0)
    n_set, n_r, n_az = Zc.shape
    Afft = np.fft.rfft(Zc, axis=2)                 # (n_set, n_r, M1)
    M1 = Afft.shape[2]
    O = np.zeros((n_r, M1), complex)
    C = np.zeros_like(O)
    thetas_rad = np.asarray(thetas_rad, float)
    for m in range(M1):
        if m == 0:
            tot = Afft[:, :, 0].mean(0)            # only the sum is determined
            if m0_assignment == 'ocs':
                O[:, 0] = tot
            elif m0_assignment == 'ccs':
                C[:, 0] = tot
            else:
                O[:, 0] = tot / 2
                C[:, 0] = tot / 2
            continue
        if m_max is not None and m > m_max:
            O[:, m] = Afft[:, :, m].mean(0)
            continue
        D = np.stack([np.ones_like(thetas_rad),
                      np.exp(-1j * m * s * thetas_rad)], axis=1)   # (n_set, 2)
        sol, *_ = np.linalg.lstsq(D, Afft[:, :, m], rcond=None)    # (2, n_r)
        O[:, m] = sol[0]
        C[:, m] = sol[1]
    O_pol = np.fft.irfft(O, n=n_az, axis=1)
    C_pol = np.fft.irfft(C, n=n_az, axis=1)

    dphi = A[1] - A[0]
    res = np.empty_like(Zc)
    for i, th in enumerate(thetas_rad):
        shift = int(np.round(s * th / dphi))
        res[i] = Zc[i] - (O_pol + np.roll(C_pol, shift, axis=1))
    return {'O_pol': O_pol, 'C_pol': C_pol, 'res': res,
            's': int(s), 'm0_assignment': m0_assignment}


def decompose_spin_lsq(Zc, valid, thetas_rad, A, n_spin=0, s=1, m_max=12,
                       ridge=1e-3, degen_assignment='ocs', ocs_only=False):
    """Hole-aware, spin-aware decomposition (unifies the scalar LSQ and the
    spin FFT).  Per radius, fit complex O and C by least squares over the
    valid samples only:

        A_d(phi) = sum_m O_m e^{i m phi}
                 + sum_m C_m e^{i (n_spin - m) s theta_d} e^{i m phi}

    `Zc` is (n_set, n_r, n_az) real (scalar, n_spin=0) or complex (doublet
    Z_cos + i Z_sin).  The degenerate mode m = n_spin is assigned by
    ``degen_assignment``.  `m_max` caps the azimuthal order; `ridge` damps
    modes the ring coverage cannot constrain.  Reduces to
    :func:`decompose_polar_lsq` at n_spin=0 and to :func:`decompose_spin_fft`
    when rings are fully sampled.  Returns the same dict shape as
    :func:`decompose_spin_fft`, with ``res`` NaN at invalid nodes.

    ``ocs_only=True`` fits the O (telescope) basis ALONE — the camera columns are
    dropped from the design matrix, so O absorbs the rotator-averaged field and
    ``C_pol`` is identically zero (a constrained fit, NOT a joint O+C fit with C
    discarded afterward — the two differ when the joint fit puts signal in C).
    The degenerate-mode redistribution is skipped (there is no C to assign to).
    """
    Zc = np.asarray(Zc, complex)
    n_set, n_r, n_az = Zc.shape
    thetas_rad = np.asarray(thetas_rad, float)
    ms = np.arange(-int(m_max), int(m_max) + 1)
    nm = ms.size
    keepC = ms != int(n_spin)                       # drop degenerate C column
    EA = np.exp(1j * np.outer(A, ms))               # (n_az, nm) reconstruction
    O_pol = np.full((n_r, n_az), np.nan, complex)
    C_pol = np.full((n_r, n_az), np.nan, complex)
    res = np.full_like(Zc, np.nan)
    for k in range(n_r):
        phi, yk, thk, idx = [], [], [], []
        for d in range(n_set):
            jj = np.nonzero(valid[d, k, :])[0]
            if jj.size:
                phi.append(A[jj]); yk.append(Zc[d, k, jj])
                thk.append(np.full(jj.size, thetas_rad[d])); idx.append((d, jj))
        if not phi:
            continue
        phi = np.concatenate(phi); yk = np.concatenate(yk)
        thk = np.concatenate(thk)
        E = np.exp(1j * np.outer(phi, ms))                          # O basis
        if ocs_only:
            D = E                                                   # O basis only
        else:
            Cc = E * np.exp(1j * (int(n_spin) - ms[None, :]) * s * thk[:, None])
            D = np.hstack([E, Cc[:, keepC]])                        # (N, 2nm-1)
        if D.shape[0] < D.shape[1]:
            continue
        DhD = D.conj().T @ D
        lam = ridge * np.trace(DhD).real / DhD.shape[0]
        coef = np.linalg.solve(DhD + lam * np.eye(DhD.shape[0]),
                               D.conj().T @ yk)
        Ofull = coef[:nm]
        if ocs_only:
            Cfull = np.zeros(nm, complex)        # constrained: no camera term
        else:
            Cfull = np.zeros(nm, complex); Cfull[keepC] = coef[nm:]
            # degenerate sum currently sits in O_{n_spin}; redistribute by flag
            i_n = int(np.nonzero(ms == int(n_spin))[0][0])
            if degen_assignment == 'ccs':
                Cfull[i_n] = Ofull[i_n]; Ofull[i_n] = 0
            elif degen_assignment == 'split':
                Ofull[i_n] *= 0.5; Cfull[i_n] = Ofull[i_n]
        O_pol[k] = EA @ Ofull
        C_pol[k] = EA @ Cfull
        dphi = A[1] - A[0]
        for (d, jj) in idx:
            shift = int(np.round(s * thetas_rad[d] / dphi))
            pred = O_pol[k] + np.exp(1j * int(n_spin) * s * thetas_rad[d]) \
                * np.roll(C_pol[k], shift)
            res[d, k, jj] = Zc[d, k, jj] - pred[jj]
    return {'O_pol': O_pol, 'C_pol': C_pol, 'res': res,
            'n_spin': int(n_spin), 's': int(s),
            'degen_assignment': degen_assignment, 'm_max': int(m_max)}


def reconstruct_at(dec, theta_rad, A):
    """Reconstruct the complex polar field at rotator angle ``theta_rad`` from a
    decomposition dict (``O_pol``, ``C_pol``, ``n_spin``, ``s``).

    Exactly the model that :func:`decompose_spin_lsq` / :func:`decompose_spin_fft`
    use to form ``res`` (the per-dataset prediction)::

        A(theta) = O_pol + exp(i*n_spin*s*theta) * roll(C_pol, round(s*theta/dphi))

    ``O_pol``/``C_pol`` are (n_r, n_az).  Returns a complex (n_r, n_az) field;
    take ``.real`` for a scalar/single Zernike, and ``.real``/``.imag`` for the
    cos/sin members of a spin doublet.  ``A`` is the azimuth-sample array used in
    the decomposition (its spacing sets ``dphi``).
    """
    O_pol = np.asarray(dec['O_pol'])
    C_pol = np.asarray(dec['C_pol'])
    n_spin = int(dec.get('n_spin', 0))
    s = int(dec['s'])
    dphi = A[1] - A[0]
    shift = int(np.round(s * float(theta_rad) / dphi))
    return O_pol + np.exp(1j * n_spin * s * float(theta_rad)) * \
        np.roll(C_pol, shift, axis=1)


# NOTE: a former decompose_auto_sign() searched s=+1 vs s=-1 by residual.  It
# was removed: the rotation sense is fixed at s=+1, the empirically-verified
# OCS->CCS convention (CCS = R(-theta).OCS, i.e. phi_CCS = phi_OCS - theta;
# see the derivation in run_intrinsic_split.py).  There is one source of truth
# for the sign (split.rotation_sign in mi_config.yaml) and no search.


def _rmask(R, r_lim):
    return (R >= r_lim[0]) & (R <= r_lim[1])


def residual_rms(field, R, r_lim=(0.1, 1.6)):
    """RMS of a (..., n_r, n_az) field over rings within ``r_lim``.
    NaN entries (e.g. masked holes) are ignored."""
    m = _rmask(R, r_lim)
    return float(np.sqrt(np.nanmean(np.asarray(field)[..., m, :] ** 2)))


def explained_variance(Z, res, R, r_lim=(0.1, 1.6)):
    """Fraction of the data variance the 2-component model explains."""
    d = residual_rms(Z, R, r_lim)
    return 1.0 - (residual_rms(res, R, r_lim) / d) ** 2 if d > 0 else np.nan


def azimuthal_amplitude(field_pol, R, r_lim=(0.1, 1.6), n_m=9):
    """RMS azimuthal-mode amplitude (m=0..n_m-1) of a polar field, averaged
    over rings within ``r_lim`` (rfft scaling)."""
    m = _rmask(R, r_lim)
    F = np.fft.rfft(field_pol, axis=1)
    return np.sqrt((np.abs(F[m]) ** 2).mean(0))[:n_m]


def polar_field_to_points(field_pol, X, Y, thx, thy):
    """Interpolate a polar field (sampled on X, Y) onto scattered points
    (thx, thy).  Used to resample O/C onto a focal-plane grid."""
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(
        np.column_stack([X.ravel(), Y.ravel()]), field_pol.ravel())
    return interp(np.asarray(thx), np.asarray(thy))


def mean_rotator(fits_table, rot_range, alt_range=None,
                 rot_col='rotator_angle', alt_col='alt'):
    """Actual mean rotator angle (deg) of the visits a dataset was built
    from: rows of ``fits_table`` within ``rot_range`` (and ``alt_range``).
    Returns (mean, n, lo, hi).  ``alt`` auto-detected radians vs degrees.
    """
    import numpy as _np
    rot = _np.asarray(fits_table[rot_col], float)
    keep = (rot >= rot_range[0]) & (rot <= rot_range[1])
    if alt_range is not None and alt_col in getattr(fits_table, 'colnames',
                                                    fits_table):
        alt = _np.asarray(fits_table[alt_col], float)
        altd = _np.rad2deg(alt) if _np.nanmax(_np.abs(alt)) < 2 * _np.pi + 1e-3 else alt
        # guard each end independently so alt_range=(None, None) or a one-sided
        # bound doesn't raise on the None comparison
        if alt_range[0] is not None:
            keep &= altd >= alt_range[0]
        if alt_range[1] is not None:
            keep &= altd <= alt_range[1]
    r = rot[keep]
    if r.size == 0:
        return float('nan'), 0, float('nan'), float('nan')
    return float(r.mean()), int(r.size), float(r.min()), float(r.max())
