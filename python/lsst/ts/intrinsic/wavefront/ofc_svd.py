"""OFC sensitivity-matrix SVD and degree-of-freedom recovery.

Shared machinery for projecting per-visit Double-Zernike vectors onto the
OFC sensitivity-matrix singular modes (u-modes / v-modes) and recovering
physical degrees of freedom (DOF).  Mirrors the construction used in
``build_measured_intrinsic`` (Path A): the sensitivity matrix is sliced to
a focal-``k`` x pupil-``j`` block, right-multiplied by the geom_mean
normalization, and SVD'd; ``n_keep`` u-modes are retained.

Importing this module is cheap — it only needs numpy.  The actual SVD build
(:func:`build_ofc_svd`) imports ``lsst.ts.ofc`` + ``yaml`` lazily, so callers
on non-RSP machines can import the constants and :func:`recover_dof_per_visit`
without the LSST stack present.

Conventions
-----------
* The OFC sensitivity matrix axis index equals the Noll index: axis 0 is the
  unused Noll-0 placeholder, axis 4 is Z4, etc.  Slicing the focal axis with
  ``[k_min:k_max+1]`` keeps focal Noll ``k_min..k_max`` inclusive; the pupil
  axis is advanced-indexed with the explicit ``iZs`` list (so it skips j=20,
  j=21 when absent).
* Row layout of the flattened ``S`` (and hence the column order of the raw
  DZ vector ``w``) is ``row = (k - k_min) * n_j + j_idx`` — ``k`` outer, ``j``
  inner — captured in :attr:`OFCSvd.kj_grid`.
* v-mode amplitude ``c_i = a_i / sigma_i`` (geom-mean-normalized DOF coords);
  physical DOF = ``N @ V_eff @ diag(1/sigma) @ a``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    'LABELS_50DOF', 'DOF_UNITS_50', 'DOF_GROUPS', 'DEFAULT_NORM_YAML',
    'find_ofc_normalization_yaml', 'load_normalization_weights',
    'recover_dof_per_visit', 'OFCSvd', 'build_ofc_svd',
    'dz_table_to_W', 'project_dz_table', 'vmode_fwhm_scale',
    'NORM_SCHEMES', 'FP_ZERNIKE_NAMES', 'compute_normalization_components',
    'normalization_weights_by_scheme', 'focal_zernike_at_points',
    'make_dz_basis_vector', 'decompose_into_dz',
]

# 50-DOF state vector: M2 hexapod (5), Camera hexapod (5),
# M1M3 bending (20), M2 bending (20).
LABELS_50DOF = [
    'M2_dz', 'M2_dx', 'M2_dy', 'M2_rx', 'M2_ry',
    'Cam_dz', 'Cam_dx', 'Cam_dy', 'Cam_rx', 'Cam_ry',
    'B1_1', 'B1_2', 'B1_3', 'B1_4', 'B1_5',
    'B1_6', 'B1_7', 'B1_8', 'B1_9', 'B1_10',
    'B1_11', 'B1_12', 'B1_13', 'B1_14', 'B1_15',
    'B1_16', 'B1_17', 'B1_18', 'B1_19', 'B1_20',
    'B2_1', 'B2_2', 'B2_3', 'B2_4', 'B2_5',
    'B2_6', 'B2_7', 'B2_8', 'B2_9', 'B2_10',
    'B2_11', 'B2_12', 'B2_13', 'B2_14', 'B2_15',
    'B2_16', 'B2_17', 'B2_18', 'B2_19', 'B2_20',
]
DOF_UNITS_50 = (
    ['μm', 'μm', 'μm', 'arcsec', 'arcsec'] +   # M2 hex
    ['μm', 'μm', 'μm', 'arcsec', 'arcsec'] +   # Cam hex
    ['μm'] * 20 +                                        # M1M3 bending
    ['μm'] * 20                                          # M2 bending
)

# Generic 0-based index groups into the 50-DOF vector.  ``hex_piston`` /
# ``hex_decenter`` / ``hex_tiptilt`` interleave Camera-then-M2 for plotting.
DOF_GROUPS = {
    'm2_hex':        list(range(0, 5)),
    'cam_hex':       list(range(5, 10)),
    'm1m3_bending':  list(range(10, 30)),
    'm2_bending':    list(range(30, 50)),
    'hex_piston':    [5, 0],
    'hex_decenter':  [6, 7, 1, 2],
    'hex_tiptilt':   [8, 9, 3, 4],
}

DEFAULT_NORM_YAML = 'range0.5_fwhm-0.15.yaml'   # geom_mean normalization


def find_ofc_normalization_yaml(user_path=None, name=DEFAULT_NORM_YAML):
    """Locate the OFC normalization-weights yaml inside ts_config_mttcs.

    Path is ``$TS_CONFIG_MTTCS_DIR/MTAOS/ofc/normalization_weights/<name>``.
    An explicit ``user_path`` wins when it exists.
    """
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(
                f'user-supplied ofc normalization yaml {p} does not exist')
        return p
    root = os.environ.get('TS_CONFIG_MTTCS_DIR')
    if not root:
        raise RuntimeError(
            'TS_CONFIG_MTTCS_DIR is not set; setup ts_config_mttcs (eups) '
            'or pass an explicit normalization yaml path.')
    target = Path(root) / 'MTAOS' / 'ofc' / 'normalization_weights' / name
    if not target.exists():
        raise FileNotFoundError(
            f'OFC normalization yaml not found at {target} '
            '(try `git pull` in ts_config_mttcs).')
    return target


def load_normalization_weights(user_path=None, name=DEFAULT_NORM_YAML):
    """Return the OFC normalization weights (1-D array) from yaml."""
    import yaml
    with open(find_ofc_normalization_yaml(user_path, name)) as fp:
        return np.array(yaml.safe_load(fp), dtype=float)


def recover_dof_per_visit(A_modes, V, Sigma, N_diag, n_keep):
    """Recover physical DOF estimates per visit from u-mode amplitudes.

    With ``S = S_orig @ N`` (right-side normalized), ``SVD = U Σ Vᵀ``::

        A     = U_eff^T @ w_raw            (μm, mode amps)
        dof_n = V_eff @ diag(1/σ_eff) @ A  (normalized)
        dof   = N @ dof_n                  (physical units)

    Parameters
    ----------
    A_modes : ndarray (n_visits, n_keep)
        u-mode amplitudes per visit (= ``U_effᵀ @ w_raw``).
    V : ndarray (n_dof, n_dof)
    Sigma : ndarray (n_dof,)
    N_diag : ndarray (n_dof,)
        OFC normalization weights (diagonal of N).
    n_keep : int or sequence of int
        Number of leading modes (0..n_keep-1) or an explicit list of mode
        indices to retain (to drop individual v-modes).

    Returns
    -------
    dof : ndarray (n_visits, n_dof)
        Physical DOF in the mixed units of :data:`DOF_UNITS_50`.
    """
    keep = list(range(int(n_keep))) if np.isscalar(n_keep) else list(n_keep)
    V_eff = V[:, keep]
    inv_sig = 1.0 / Sigma[keep]
    A = np.where(np.isfinite(A_modes), A_modes, 0.0)
    dof_norm = (V_eff * inv_sig[None, :]) @ A.T          # (n_dof, n_visits)
    dof_phys = N_diag[:, None] * dof_norm
    return dof_phys.T                                    # (n_visits, n_dof)


@dataclass
class OFCSvd:
    """Result of :func:`build_ofc_svd`: the kept SVD plus the (k, j) layout.

    Use :meth:`project_amplitudes` (raw DZ vectors -> u-mode amps ``A``),
    :meth:`vmodes` (``A`` -> v-mode amplitudes ``c_i = a_i/σ_i``) and
    :meth:`dof` (``A`` -> physical DOF).
    """
    U_eff: np.ndarray
    V: np.ndarray
    Sigma: np.ndarray
    normalization_weights: np.ndarray
    kj_grid: list
    n_keep_eff: int
    k_min: int
    k_max: int
    iZs: list
    n_dof: int
    vmode_labels: list = field(default_factory=list)
    dof_idx: list = field(default_factory=list)   # DOF indices (0-49) in the SVD
    keep_idx: list = field(default_factory=list)   # mode indices retained

    def _keep(self):
        return self.keep_idx if self.keep_idx else list(range(self.n_keep_eff))

    def project_amplitudes(self, W):
        """Raw DZ matrix ``W`` (n_visits, n_kj) in :attr:`kj_grid` order ->
        u-mode amplitudes ``A`` (n_visits, n_keep).  NaNs are treated as 0."""
        return np.where(np.isfinite(W), W, 0.0) @ self.U_eff

    def vmodes(self, A_modes):
        """u-mode amps -> v-mode amplitudes ``c_i = a_i/σ_i``."""
        return np.asarray(A_modes) / self.Sigma[self._keep()][None, :]

    def dof(self, A_modes):
        """u-mode amps -> physical DOF (n_visits, n_dof_sub) over dof_idx."""
        return recover_dof_per_visit(
            A_modes, self.V, self.Sigma, self.normalization_weights,
            self._keep())

    def dof_labels(self):
        """DOF labels for the dof_idx subset (full 50 if dof_idx empty)."""
        idx = self.dof_idx if self.dof_idx else list(range(len(LABELS_50DOF)))
        return [LABELS_50DOF[i] for i in idx], [DOF_UNITS_50[i] for i in idx]


def build_ofc_svd(iZs, k_min, k_max, n_keep, n_dof=None,
                  ofc_normalization_yaml=None,
                  instrument='lsst', norm_yaml_name=DEFAULT_NORM_YAML):
    """Build the OFC sensitivity-matrix SVD for a focal-k x pupil-j slice.

    ``n_dof`` selects the DOF columns: None -> all 50; an int ``n`` -> DOF
    0..n-1; an explicit list -> exactly those DOF indices.  ``n_keep`` selects
    the modes: an int ``n`` -> modes 0..n-1; a list -> exactly those mode
    indices (to drop individual v-modes).  Imports ``lsst.ts.ofc`` lazily.
    Returns an :class:`OFCSvd`.
    """
    from lsst.ts.ofc import OFCData

    nw_full = load_normalization_weights(ofc_normalization_yaml, norm_yaml_name)
    S_full = np.asarray(OFCData(instrument).sensitivity_matrix)
    iZs_arr = np.asarray(iZs, dtype=int)
    n_k = int(k_max) - int(k_min) + 1
    n_j = len(iZs_arr)
    S_slab = S_full[int(k_min):int(k_max) + 1, iZs_arr, :]   # (n_k, n_j, n_dof_full)
    n_dof_full = S_slab.shape[-1]

    if n_dof is None:
        dof_idx = list(range(n_dof_full))
    elif np.isscalar(n_dof):
        dof_idx = list(range(int(n_dof)))
    else:
        dof_idx = sorted(int(d) for d in n_dof)
    norm_sub = nw_full[dof_idx]
    S = S_slab.reshape(-1, n_dof_full)[:, dof_idx] @ np.diag(norm_sub)
    kj_grid = [(int(k_min + ki), int(iZs_arr[ji]))
               for ki in range(n_k) for ji in range(n_j)]

    U, Sigma, Vh = np.linalg.svd(S, full_matrices=False)
    V = Vh.T
    n_modes = U.shape[1]
    if np.isscalar(n_keep):
        keep_idx = list(range(min(int(n_keep), n_modes)))
    else:
        keep_idx = sorted(k for k in (int(x) for x in n_keep) if k < n_modes)
    n_keep_eff = len(keep_idx)
    return OFCSvd(
        U_eff=U[:, keep_idx], V=V, Sigma=Sigma,
        normalization_weights=norm_sub, kj_grid=kj_grid,
        n_keep_eff=n_keep_eff, k_min=int(k_min), k_max=int(k_max),
        iZs=[int(j) for j in iZs_arr], n_dof=len(dof_idx),
        vmode_labels=[f'v{m + 1}' for m in keep_idx],   # 1-based
        dof_idx=dof_idx, keep_idx=keep_idx)


def dz_table_to_W(fit_table, prefix, kj_grid):
    """Pack per-visit raw DZ vectors into ``W`` (n_visits, n_kj) in
    ``kj_grid`` column order.  Missing columns stay NaN.  Works with any
    table exposing ``colnames`` and column access (astropy QTable / Table)."""
    n = len(fit_table)
    W = np.full((n, len(kj_grid)), np.nan)
    for ci, (k, j) in enumerate(kj_grid):
        col = f'{prefix}_z{j}_c{k}'
        if col in fit_table.colnames:
            W[:, ci] = np.asarray(fit_table[col], dtype=float)
    return W


def project_dz_table(fit_table, prefix, svd):
    """Project a per-visit DZ fit table onto the SVD modes.

    Returns ``(vmodes, dof, A_modes, W)`` where ``vmodes`` is
    (n_visits, n_keep), ``dof`` is (n_visits, n_dof), ``A_modes`` is the
    u-mode amplitudes and ``W`` the raw DZ matrix.
    """
    W = dz_table_to_W(fit_table, prefix, svd.kj_grid)
    A = svd.project_amplitudes(W)
    return svd.vmodes(A), svd.dof(A), A, W


# ----------------------------------------------------------------------
# Normalization-weight schemes (Ã = A @ diag(n_j))
# ----------------------------------------------------------------------
# Each scheme sets n_j; only some are invariant under x_j -> alpha*x_j (a
# change of physical units for DOF j).  See the V-mode normalization study.
#   'default'   — stored OFC weights as-is            (NOT unit-invariant)
#   'rf'        — computed r_j * f_j                   (NOT unit-invariant)
#   'r_only'    — range weights only:    n_j = r_j     (unit-invariant)
#   'inv_f'     — inverse FWHM weights:  n_j = 1/f_j   (unit-invariant)
#   'geom_mean' — geometric mean: n_j = sqrt(r_j/f_j)  (unit-invariant; default)
#   'tunable'   — n_j = r_j**a * f_j**(a-1), a=`a`     (unit-invariant any a)
NORM_SCHEMES = ('default', 'rf', 'r_only', 'inv_f', 'geom_mean', 'tunable')

# Focal-plane (double-Zernike) Noll index names, k = 1..6.
FP_ZERNIKE_NAMES = {1: 'piston', 2: 'x-tilt', 3: 'y-tilt',
                    4: 'defocus', 5: 'astig-45', 6: 'astig-0'}


def compute_normalization_components(ofc_data, dz_sensitivity_matrix,
                                     field_angles):
    """Range (r_j) and FWHM (f_j) normalization-weight components.

    Mirrors ts_ofc ``generate_normalization_weights.compute_normalization_weights``:
    ``r_j`` from hexapod stroke / bending-mode force range over the max
    rotation-matrix element; ``f_j`` from the RSS of the FWHM sensitivity
    (``convertZernikesToPsfWidth``) across field positions.  Imports
    ``lsst.ts.ofc`` / ``lsst.ts.wep`` lazily.  Returns ``(range_weights,
    fwhm_weights)``, each length ``n_dof`` (50).
    """
    from lsst.ts.ofc import BendModeToForce
    from lsst.ts.wep.utils import convertZernikesToPsfWidth

    m1m3_bending_range = ofc_data.m1m3_force_range / 20
    m2_bending_range = ofc_data.m2_force_range / 20
    m1m3_bmf = BendModeToForce('M1M3', ofc_data)
    m2_bmf = BendModeToForce('M2', ofc_data)
    range_weights = np.concatenate((
        ofc_data.rb_stroke,
        m1m3_bending_range / np.max(np.abs(m1m3_bmf.rot_mat), axis=0),
        m2_bending_range / np.max(np.abs(m2_bmf.rot_mat), axis=0),
    ))

    sens = dz_sensitivity_matrix.evaluate(field_angles, rotation_angle=0.0)
    sens = sens[:, ofc_data.zn_idx, :]
    n_dof = sens.shape[2]
    fwhm = np.zeros(sens.shape)
    for idy in range(sens.shape[0]):
        fwhm[idy, ...] = convertZernikesToPsfWidth(sens[idy, ...].T).T
    fwhm_2d = fwhm.reshape((-1, n_dof))
    fwhm_weights = np.sqrt(np.sum(np.square(fwhm_2d), axis=0))
    return range_weights, fwhm_weights


def normalization_weights_by_scheme(scheme, range_weights=None,
                                    fwhm_weights=None, *, stored=None, a=0.5):
    """Normalization vector ``n_j`` for a named :data:`NORM_SCHEMES` scheme.

    ``range_weights`` / ``fwhm_weights`` come from
    :func:`compute_normalization_components`.  ``scheme='default'`` returns the
    ``stored`` OFC weights (which must be passed); ``'tunable'`` uses exponent
    ``a`` (``a=0`` → ``1/f``, ``a=0.5`` → ``sqrt(r/f)``, ``a=1`` → ``r_only``).
    """
    if scheme == 'default':
        if stored is None:
            raise ValueError("scheme 'default' requires stored= weights")
        return np.asarray(stored, dtype=float)
    if range_weights is None or fwhm_weights is None:
        raise ValueError(f'scheme {scheme!r} needs range_weights and fwhm_weights')
    r = np.asarray(range_weights, dtype=float)
    f = np.asarray(fwhm_weights, dtype=float)
    if scheme == 'rf':
        return r * f
    if scheme == 'r_only':
        return r
    if scheme == 'inv_f':
        return 1.0 / f
    if scheme == 'geom_mean':
        return np.sqrt(r / f)
    if scheme == 'tunable':
        return (r ** a) * (f ** (a - 1.0))
    raise ValueError(f'Unknown normalization scheme {scheme!r}; '
                     f'expected one of {NORM_SCHEMES}')


# ----------------------------------------------------------------------
# Double-Zernike basis vectors (pupil Noll j x focal-plane Noll k)
# ----------------------------------------------------------------------
def focal_zernike_at_points(k_noll, rho, theta):
    """Noll focal-plane Zernike ``Z_k`` (k = 1..6) at field points ``(rho, theta)``."""
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if k_noll == 1:
        return np.ones_like(rho)
    if k_noll == 2:
        return 2.0 * rho * np.cos(theta)
    if k_noll == 3:
        return 2.0 * rho * np.sin(theta)
    if k_noll == 4:
        return np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0)
    if k_noll == 5:
        return np.sqrt(6.0) * rho ** 2 * np.sin(2.0 * theta)
    if k_noll == 6:
        return np.sqrt(6.0) * rho ** 2 * np.cos(2.0 * theta)
    raise ValueError(f'Noll index {k_noll} not implemented (only 1-6)')


def make_dz_basis_vector(j_noll, k_fp, zn_array, fp_zernike_dict, n_fp, n_zernike):
    """Double-Zernike basis vector: pupil Noll ``j_noll`` × focal-plane ``k_fp``.

    Places the focal-plane values ``fp_zernike_dict[k_fp]`` (one per field
    position) at the ``j_noll`` slot of each field position's Zernike block, in a
    flat vector of length ``n_fp * n_zernike`` (field-position outer, Zernike
    inner).  Returns None if ``j_noll`` is not in ``zn_array``.
    """
    match = np.where(np.asarray(zn_array) == j_noll)[0]
    if len(match) == 0:
        return None
    p = int(match[0])
    fp_vals = fp_zernike_dict[k_fp]
    vec = np.zeros(n_fp * n_zernike)
    for s in range(n_fp):
        vec[s * n_zernike + p] = fp_vals[s]
    return vec


def decompose_into_dz(z_vec, dz_vectors_dict):
    """Project ``z_vec`` onto each double-Zernike basis vector.

    DZ basis vectors for distinct ``(j, k_fp)`` have non-overlapping support, so
    each projection is ``(z·d)/(d·d)``.  Returns ``{(j, k_fp): coeff}``.
    """
    coeffs = {}
    for key, d in dz_vectors_dict.items():
        norm2 = float(np.dot(d, d))
        if norm2 > 0:
            coeffs[key] = float(np.dot(z_vec, d) / norm2)
    return coeffs


def vmode_fwhm_scale(svd):
    """Per-v-mode FWHM scale ``g_i`` (arcsec per unit ``c_i``) for the
    geom_mean normalization, or None if ``lsst.ts.ofc`` lacks the pieces.

    ``fwhm_i = c_i * g_i``, ``g_i = ||(r_j/n_j) ⊙ v_i||`` with range
    weights ``r_j`` and the loaded normalization ``n_j``; for geom_mean
    ``f_j n_j = sqrt(r_j f_j) = r_j/n_j``.
    """
    try:
        from lsst.ts.ofc import BendModeToForce, OFCData
        ofc = OFCData('lsst')
        m1m3 = ofc.m1m3_force_range / 20
        m2 = ofc.m2_force_range / 20
        range_weights = np.concatenate((
            ofc.rb_stroke,
            m1m3 / np.max(np.abs(BendModeToForce('M1M3', ofc).rot_mat), axis=0),
            m2 / np.max(np.abs(BendModeToForce('M2', ofc).rot_mat), axis=0),
        ))
        rf_sqrt = range_weights / np.asarray(svd.normalization_weights, float)
        return np.sqrt(((rf_sqrt[:, None] * svd.V[:, :svd.n_keep_eff]) ** 2)
                       .sum(axis=0))
    except Exception:
        return None
