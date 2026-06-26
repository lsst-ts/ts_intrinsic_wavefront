"""Canonical Noll-indexed Zernike polynomial names and formulas.

This module is the single source of truth for Zernike-mode names used in
the Rubin AOS notebooks (`aos/`).  The same names apply whether the
Zernike index labels a *pupil-plane* mode (often called ``j``) or a
*focal-plane* mode (often called ``k``) — Noll's convention is universal.

Sources / conventions
---------------------
* Noll, R. J. (1976). *Zernike polynomials and atmospheric turbulence.*
  J. Opt. Soc. Am., 66(3), 207-211.  Establishes the j-indexing,
  cos/sin parity (even j = cos(mφ), odd j = sin(mφ)), and the
  orthonormalization factors.
* The "primary / secondary / tertiary" verbal names (e.g. "secondary
  spherical", "secondary astigmatism") follow standard optics
  textbooks and are also the convention used by Wikipedia
  (https://en.wikipedia.org/wiki/Zernike_polynomials), Wyant &
  Creath (1992), and the LSST AOS / `ts_wep` codebase.

Naming convention used here
---------------------------
* For ``m = 0`` (radial-only) terms: bare name (e.g. "Defocus",
  "Spherical", "2nd Spherical").
* For ``m > 0``: pair of cos/sin terms with suffix ``_x`` (cos, even j)
  vs ``_y`` (sin, odd j).  This matches the existing AOS code
  convention where "Astig0" is cos(2φ) and "Astig45" is sin(2φ),
  "Coma_x" is cos(φ) and "Coma_y" is sin(φ), etc.
* "Primary" terms get the bare radial-family name; subsequent
  appearances of the same `m` at higher `n` get a "2nd"/"3rd" prefix.
  So ``j=12,13`` is the **secondary** astigmatism (n=4, m=2),
  ``j=23,24`` is the **tertiary** astigmatism (n=6, m=2), etc.

Output bindings
---------------
``NOLL_TERMS``   — full table: ``j -> {n, m, parity, name, formula}``
``NOLL_NAMES``   — convenience: ``j -> name`` (string)
``NOLL_FORMULAS``— convenience: ``j -> formula`` (Unicode string;
                   uses ρ / φ for radial / azimuthal coordinates)

For backwards compatibility with code that previously had
inline ``PUPIL_NAMES`` and ``FOCAL_NAMES`` dicts, both aliases are
exported and point at the same name table:

    PUPIL_NAMES = FOCAL_NAMES = NOLL_NAMES

Usage
-----

    from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES, NOLL_FORMULAS

    >>> NOLL_NAMES[6]
    'Astig0'
    >>> NOLL_FORMULAS[11]
    '√5 (6ρ⁴ − 6ρ² + 1)'
"""

from __future__ import annotations


# ---------------------------------------------------------------------
# Canonical table — indexed by Noll j.
# ---------------------------------------------------------------------
#
# Notes on the higher-order radial-zero terms:
#   j= 4  Defocus              (n=2, m=0)  — primary
#   j=11  Spherical            (n=4, m=0)  — primary
#   j=22  2nd Spherical        (n=6, m=0)  — secondary
#   j=37  3rd Spherical        (n=8, m=0)  — tertiary  (added for completeness)
#
# Higher-order azimuthal families pick up the "2nd / 3rd / ..." prefix
# the second / third / ... time the same |m| appears as n increases:
#   m=1: 7,8 -> Coma         (n=3) ; 16,17 -> 2nd Coma    (n=5)
#   m=2: 5,6 -> Astig        (n=2) ; 12,13 -> 2nd Astig   (n=4) ;
#                                    23,24 -> 3rd Astig    (n=6)
#   m=3: 9,10-> Trefoil      (n=3) ; 18,19 -> 2nd Trefoil (n=5)
#   m=4: 14,15-> Tetrafoil   (n=4) ; 25,26 -> 2nd Tetrafoil (n=6)
#   m=5: 20,21-> Pentafoil   (n=5)
#   m=6: 27,28-> Hexafoil    (n=6)

NOLL_TERMS: dict[int, dict] = {
    # ----- n = 0 -----
    1:  {'n': 0, 'm': 0, 'parity': 'radial', 'name': 'Piston',
         'formula': '1'},
    # ----- n = 1 -----
    2:  {'n': 1, 'm': 1, 'parity': 'cos',    'name': 'Tilt_x',
         'formula': '2ρ cos(φ)'},
    3:  {'n': 1, 'm': 1, 'parity': 'sin',    'name': 'Tilt_y',
         'formula': '2ρ sin(φ)'},
    # ----- n = 2 -----
    4:  {'n': 2, 'm': 0, 'parity': 'radial', 'name': 'Defocus',
         'formula': '√3 (2ρ² − 1)'},
    5:  {'n': 2, 'm': 2, 'parity': 'sin',    'name': 'Astig45',
         'formula': '√6 ρ² sin(2φ)'},
    6:  {'n': 2, 'm': 2, 'parity': 'cos',    'name': 'Astig0',
         'formula': '√6 ρ² cos(2φ)'},
    # ----- n = 3 -----
    7:  {'n': 3, 'm': 1, 'parity': 'sin',    'name': 'Coma_y',
         'formula': '√8 (3ρ³ − 2ρ) sin(φ)'},
    8:  {'n': 3, 'm': 1, 'parity': 'cos',    'name': 'Coma_x',
         'formula': '√8 (3ρ³ − 2ρ) cos(φ)'},
    9:  {'n': 3, 'm': 3, 'parity': 'sin',    'name': 'Trefoil_y',
         'formula': '√8 ρ³ sin(3φ)'},
    10: {'n': 3, 'm': 3, 'parity': 'cos',    'name': 'Trefoil_x',
         'formula': '√8 ρ³ cos(3φ)'},
    # ----- n = 4 -----
    11: {'n': 4, 'm': 0, 'parity': 'radial', 'name': 'Spherical',
         'formula': '√5 (6ρ⁴ − 6ρ² + 1)'},
    12: {'n': 4, 'm': 2, 'parity': 'cos',    'name': '2ndAstig0',
         'formula': '√10 (4ρ⁴ − 3ρ²) cos(2φ)'},
    13: {'n': 4, 'm': 2, 'parity': 'sin',    'name': '2ndAstig45',
         'formula': '√10 (4ρ⁴ − 3ρ²) sin(2φ)'},
    14: {'n': 4, 'm': 4, 'parity': 'cos',    'name': 'Tetrafoil_x',
         'formula': '√10 ρ⁴ cos(4φ)'},
    15: {'n': 4, 'm': 4, 'parity': 'sin',    'name': 'Tetrafoil_y',
         'formula': '√10 ρ⁴ sin(4φ)'},
    # ----- n = 5 -----
    16: {'n': 5, 'm': 1, 'parity': 'cos',    'name': '2ndComa_x',
         'formula': '√12 (10ρ⁵ − 12ρ³ + 3ρ) cos(φ)'},
    17: {'n': 5, 'm': 1, 'parity': 'sin',    'name': '2ndComa_y',
         'formula': '√12 (10ρ⁵ − 12ρ³ + 3ρ) sin(φ)'},
    18: {'n': 5, 'm': 3, 'parity': 'cos',    'name': '2ndTrefoil_x',
         'formula': '√12 (5ρ⁵ − 4ρ³) cos(3φ)'},
    19: {'n': 5, 'm': 3, 'parity': 'sin',    'name': '2ndTrefoil_y',
         'formula': '√12 (5ρ⁵ − 4ρ³) sin(3φ)'},
    20: {'n': 5, 'm': 5, 'parity': 'cos',    'name': 'Pentafoil_x',
         'formula': '√12 ρ⁵ cos(5φ)'},
    21: {'n': 5, 'm': 5, 'parity': 'sin',    'name': 'Pentafoil_y',
         'formula': '√12 ρ⁵ sin(5φ)'},
    # ----- n = 6 -----
    22: {'n': 6, 'm': 0, 'parity': 'radial', 'name': '2ndSpherical',
         'formula': '√7 (20ρ⁶ − 30ρ⁴ + 12ρ² − 1)'},
    23: {'n': 6, 'm': 2, 'parity': 'sin',    'name': '3rdAstig45',
         'formula': '√14 (15ρ⁶ − 20ρ⁴ + 6ρ²) sin(2φ)'},
    24: {'n': 6, 'm': 2, 'parity': 'cos',    'name': '3rdAstig0',
         'formula': '√14 (15ρ⁶ − 20ρ⁴ + 6ρ²) cos(2φ)'},
    25: {'n': 6, 'm': 4, 'parity': 'sin',    'name': '2ndTetrafoil_y',
         'formula': '√14 (6ρ⁶ − 5ρ⁴) sin(4φ)'},
    26: {'n': 6, 'm': 4, 'parity': 'cos',    'name': '2ndTetrafoil_x',
         'formula': '√14 (6ρ⁶ − 5ρ⁴) cos(4φ)'},
    27: {'n': 6, 'm': 6, 'parity': 'sin',    'name': 'Hexafoil_y',
         'formula': '√14 ρ⁶ sin(6φ)'},
    28: {'n': 6, 'm': 6, 'parity': 'cos',    'name': 'Hexafoil_x',
         'formula': '√14 ρ⁶ cos(6φ)'},
}


# ---------------------------------------------------------------------
# Convenience views.
# ---------------------------------------------------------------------

NOLL_NAMES:    dict[int, str] = {j: t['name']    for j, t in NOLL_TERMS.items()}
NOLL_FORMULAS: dict[int, str] = {j: t['formula'] for j, t in NOLL_TERMS.items()}

# Backwards-compatibility aliases — the same Noll table is the right
# answer whether the index is a pupil-plane Noll j or a focal-plane
# Noll k.
PUPIL_NAMES = NOLL_NAMES
FOCAL_NAMES = NOLL_NAMES


def name(j: int) -> str:
    """Return the Noll-indexed Zernike name (or 'Z{j}' if unknown)."""
    return NOLL_NAMES.get(int(j), f'Z{int(j)}')


def formula(j: int) -> str:
    """Return the Noll-indexed Zernike formula string (or '' if unknown)."""
    return NOLL_FORMULAS.get(int(j), '')


def info(j: int) -> dict:
    """Return the full info dict ``{n, m, parity, name, formula}`` for j."""
    return dict(NOLL_TERMS.get(int(j), {}))
