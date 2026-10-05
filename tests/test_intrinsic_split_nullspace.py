"""Field-edge null-space handling of the OCS/CCS decomposition
(``intrinsic_split.decompose_spin_lsq`` / ``solve_ring_nullspace``).

numpy + scipy only (no LSST stack needed).
"""
import numpy as np
import pytest

from lsst.ts.intrinsic.wavefront import intrinsic_split as isp


def _synthetic(seed=1, n_r=30, n_az=120, noise=0.002):
    """Known telescope (O) + camera (C) maps at five rotator angles, with full
    azimuth coverage inside r<1.3, camera-fixed partial wedges outside, and an
    almost empty outermost ring (10 nodes in total)."""
    rng = np.random.default_rng(seed)
    R, A, X, Y = isp.make_polar_grid(0.06, 1.75, n_r, n_az)
    # rotator angles are multiples of the 3 deg azimuth step so the integer
    # roll used by reconstruct_at / res is exact
    th = np.deg2rad([-60.0, -15.0, 0.0, 15.0, 60.0])
    RR, AA = R[:, None], A[None, :]
    O_true = 0.03 * RR ** 2 * np.cos(2 * AA) + 0.01 * np.sin(3 * AA) * RR
    C_true = 0.02 * RR * np.cos(AA) + 0.015 * np.cos(4 * AA) * RR ** 2
    Z = np.empty((len(th), n_r, n_az))
    for d, t in enumerate(th):
        psi = AA - t
        Z[d] = O_true + 0.02 * RR * np.cos(psi) + 0.015 * np.cos(4 * psi) * RR ** 2
    Z += rng.normal(0.0, noise, Z.shape)
    valid = np.ones(Z.shape, bool)
    for k in range(n_r):
        if R[k] < 1.3:
            continue
        for d, t in enumerate(th):
            psi = (A - t) % (2 * np.pi)
            width = np.deg2rad(60.0 * (1.0 - (R[k] - 1.3) / 0.5))
            valid[d, k] = (psi % (np.pi / 2)) < width
    valid[:, -1] = False
    valid[0, -1, :5] = True
    valid[3, -1, n_az // 2:n_az // 2 + 5] = True
    return R, A, th, Z, valid, O_true, C_true


def _no_mean(F):
    return F - F.mean(axis=1, keepdims=True)      # drop the degenerate m=0 term


def test_ring_coverage():
    v = np.zeros((2, 3, 10), bool)
    v[0, 0, :5] = True
    v[1, 0, 3:8] = True
    v[0, 1, :] = True
    cov = isp.ring_coverage(v)
    assert np.allclose(cov, [0.8, 1.0, 0.0])


def test_solve_ring_nullspace_matches_ridge_when_full_rank():
    rng = np.random.default_rng(0)
    N, P, n_o = 200, 21, 11
    D = rng.normal(size=(N, P)) + 1j * rng.normal(size=(N, P))
    y = rng.normal(size=N) + 1j * rng.normal(size=N)
    DhD = D.conj().T @ D
    lam = 1e-3 * np.trace(DhD).real / P
    ref = np.linalg.solve(DhD + lam * np.eye(P), D.conj().T @ y)
    x, rank, null_dim = isp.solve_ring_nullspace(D, y, n_o, ridge=1e-3, null_tol=1e-2)
    assert rank == P and null_dim == 0
    assert np.allclose(x, ref, atol=1e-10)


def test_solve_ring_nullspace_assigns_null_direction():
    """A design with an exact O/C degeneracy: the data fix only O_1 + C_1."""
    rng = np.random.default_rng(2)
    N = 50
    phi = rng.uniform(0, 2 * np.pi, N)
    basis = np.exp(1j * phi)
    # columns: O_1, O_2 (independent), C_1 == O_1 (degenerate), C_2 independent
    D = np.column_stack([basis, np.exp(2j * phi), basis, np.exp(3j * phi)])
    truth = np.array([0.5, 0.2, 0.3, -0.1])
    y = D @ truth
    for assign in ('ocs', 'ccs', 'split'):
        x, rank, null_dim = isp.solve_ring_nullspace(
            D, y, n_o=2, ridge=0.0, null_tol=1e-8, degen_assignment=assign,
            null_weight=1e-2)
        assert rank == 3 and null_dim == 1
        assert np.allclose(D @ x, y, atol=1e-8)            # fit unchanged
        assert np.isclose(x[0] + x[2], 0.8, atol=1e-8)     # only the sum is known
        assert np.allclose(x[[1, 3]], [0.2, -0.1], atol=1e-8)
        if assign == 'ocs':
            assert abs(x[2]) < 0.02 * 0.8                  # camera block ~empty
            assert abs(x[0]) > 0.98 * 0.8
        elif assign == 'ccs':
            assert abs(x[0]) < 0.02 * 0.8
        else:
            assert np.isclose(abs(x[0]), abs(x[2]), atol=1e-8)


def test_under_determined_ring_is_bounded():
    """Fewer samples than unknowns: no blow-up along the null directions."""
    rng = np.random.default_rng(3)
    N, P, n_o = 20, 49, 25
    D = rng.normal(size=(N, P)) + 1j * rng.normal(size=(N, P))
    y = rng.normal(size=N)
    x, rank, null_dim = isp.solve_ring_nullspace(D, y, n_o, ridge=1e-3, null_tol=1e-2)
    assert rank <= N and null_dim >= P - N
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x) < 10 * np.linalg.norm(y)


@pytest.mark.parametrize('assign', ['ocs', 'split', 'ccs'])
def test_decompose_edge_has_no_nan_and_fits_data(assign):
    R, A, th, Z, valid, O_true, C_true = _synthetic()
    dec = isp.decompose_spin_lsq(Z.astype(complex), valid, th, A, n_spin=0, s=1,
                                 m_max=12, ridge=1e-3, degen_assignment=assign)
    O = dec['O_pol'].real
    C = dec['C_pol'].real
    # every ring with data is defined (the sparse outermost one by continuation)
    assert np.isfinite(O).all() and np.isfinite(C).all()
    assert dec['ring_filled'][-1] and not dec['ring_filled'][:-1].any()
    assert dec['ring_rank'][-1] == -1
    # the outer, partially covered rings have null directions; the inner ones none
    null = dec['ring_null_dim']
    assert (null[R < 1.3] == 0).all()
    assert (null[(R >= 1.4) & (R < 1.74)] > 0).any()
    # the model still fits the sampled data at the noise level
    res = dec['res'].real
    assert np.sqrt(np.nanmean(res[valid] ** 2)) < 0.004
    # interior recovery (full coverage): exact up to the m=0 degeneracy
    inner = R < 1.3
    assert np.sqrt(np.mean(_no_mean(O - O_true)[inner] ** 2)) < 0.002
    assert np.sqrt(np.mean(_no_mean(C - C_true)[inner] ** 2)) < 0.004


def test_assignment_moves_unconstrained_signal_but_not_the_fit():
    R, A, th, Z, valid, _, _ = _synthetic()
    out = {}
    for assign in ('ocs', 'split', 'ccs'):
        dec = isp.decompose_spin_lsq(Z.astype(complex), valid, th, A, n_spin=0,
                                     s=1, m_max=12, ridge=1e-3,
                                     degen_assignment=assign)
        null = dec['ring_null_dim'] > 0
        pred = np.stack([isp.reconstruct_at(dec, t, A).real for t in th])
        # sampled nodes on FITTED rings (the continued outermost ring is not a
        # fit, so its prediction legitimately depends on the assignment)
        fitted = valid & ~dec['ring_filled'][None, :, None]
        out[assign] = (np.sqrt(np.mean(dec['C_pol'].real[null] ** 2)),
                       np.sqrt(np.mean(dec['O_pol'].real[null] ** 2)),
                       pred[fitted])
    # same prediction on the fitted, sampled nodes whatever the assignment: the
    # null directions carry singular values below null_tol x the largest, so
    # moving along them changes the fitted data by a few percent of its RMS at most
    scale = np.sqrt(np.mean(out['split'][2] ** 2))
    assert np.max(np.abs(out['ocs'][2] - out['split'][2])) < 0.1 * scale
    assert np.max(np.abs(out['ccs'][2] - out['split'][2])) < 0.1 * scale
    # 'ocs' leaves the least in the camera map on null rings, 'ccs' the most
    assert out['ocs'][0] < out['split'][0] < out['ccs'][0]


def test_interior_identical_to_plain_ridge():
    """Full-coverage rings: the rank-aware solve reproduces the old ridge solve."""
    R, A, th, Z, valid, _, _ = _synthetic()
    Zc = Z.astype(complex)
    dec = isp.decompose_spin_lsq(Zc, valid, th, A, n_spin=0, s=1, m_max=12,
                                 ridge=1e-3, degen_assignment='ocs')
    ms = np.arange(-12, 13)
    nm = ms.size
    keepC = ms != 0
    EA = np.exp(1j * np.outer(A, ms))
    k = 5
    phi = np.tile(A, len(th))
    thk = np.repeat(th, len(A))
    y = Zc[:, k, :].ravel()
    E = np.exp(1j * np.outer(phi, ms))
    Cc = E * np.exp(1j * (0 - ms[None, :]) * thk[:, None])
    D = np.hstack([E, Cc[:, keepC]])
    DhD = D.conj().T @ D
    lam = 1e-3 * np.trace(DhD).real / DhD.shape[0]
    coef = np.linalg.solve(DhD + lam * np.eye(DhD.shape[0]), D.conj().T @ y)
    O_ref = (EA @ coef[:nm]).real
    assert np.allclose(dec['O_pol'][k].real, O_ref, atol=1e-9)


def test_polar_field_to_points_rim_fill():
    R, A, X, Y = isp.make_polar_grid(0.06, 1.75, 20, 60)
    F = np.hypot(X, Y)                                   # field = radius
    step = R[1] - R[0]
    thx = np.array([1.70, 1.7512, 1.75 + 2 * step])
    thy = np.zeros(3)
    plain = isp.polar_field_to_points(F, X, Y, thx, thy)
    assert np.isfinite(plain[0]) and np.isnan(plain[1]) and np.isnan(plain[2])
    filled = isp.polar_field_to_points(F, X, Y, thx, thy, nearest_within=step)
    assert np.isclose(filled[1], 1.75, atol=1e-9)        # nearest node on the rim
    assert np.isnan(filled[2])                           # farther out: still NaN
