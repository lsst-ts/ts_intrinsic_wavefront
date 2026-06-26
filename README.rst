######################
ts_intrinsic_wavefront
######################

Rubin AOS measured-intrinsic wavefront calibration: tools to build the
telescope-fixed (OCS) and camera-fixed (CCS) Measured Intrinsic Wavefront (MIW)
maps from FAM observations, and to read/deploy them in the AOS online system.

Layout
======

- ``python/lsst/ts/intrinsic/wavefront/`` — the library (DZ fitting, measured
  intrinsic build, OCS/CCS split, OFC SVD, CCD heights, config loaders).
- ``bin.src/`` — pipeline entry points (``run_mktable``, ``run_dz_fit``,
  ``run_build_intrinsic``, ``run_intrinsic_split``, ``combine_parquets``).
- ``pipelines/`` — Snakemake driver + configs to generate the calibration.
  See ``pipelines/README.md``.
- ``calibration/`` — versioned, frozen MIW map products + ``stage_miw.py``.
  See ``calibration/README.md``.
- ``notebooks/aos_miw_ocs_ccs_maps.ipynb`` — standalone OCS/CCS map reader
  (numpy / matplotlib / astropy / scipy only; no LSST stack).

The calibration-generation pipeline is RSP-only (Butler + ``ts_wep`` /
``ts_ofc``); the map-reader notebook runs anywhere.
