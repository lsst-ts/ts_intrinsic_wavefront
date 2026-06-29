######################
ts_intrinsic_wavefront
######################

Rubin AOS measured-intrinsic wavefront calibration: tools to build the
telescope-fixed (OCS) and camera-fixed (CCS) Measured Intrinsic Wavefront (MIW)
maps from Full Array Mode (FAM) observations, and to read/deploy them in the AOS
online system.

Layout
======

- ``python/lsst/ts/intrinsic/wavefront/`` — the library (DZ fitting, measured
  intrinsic build, OCS/CCS split, OFC SVD, CCD heights, config loaders), plus
  the ``common/`` subpackage.
- ``bin.src/`` — pipeline entry points (``run_mktable``, ``run_dz_fit``,
  ``run_build_intrinsic``, ``run_intrinsic_split``, ``combine_parquets``), plus
  the calibration-table tools ``run_make_calib_tables`` (MIW maps →
  ``IntrinsicZernikes`` source tables) and ``ingest_calib_tables`` (those tables
  → Butler).
- ``pipelines/`` — Snakemake driver + configs to generate the calibration.
  Quickstart: ``pipelines/README.md``.
- ``calibration/`` — versioned, frozen MIW map products + ``stage_miw.py``.
  Details: ``calibration/README.md``.
- ``notebooks/aos_miw_ocs_ccs_maps.ipynb`` — standalone OCS/CCS map reader
  (numpy / matplotlib / astropy / scipy only; no LSST stack);
  ``notebooks/plot_calib_tables.ipynb`` — view the generated per-filter
  ``IntrinsicZernikes`` source tables; ``notebooks/compare_calib_tables_vs_v1.ipynb``
  — diff new calib tables against an old frozen map.

The calibration-generation pipeline is RSP-only (Butler + ``ts_wep`` /
``ts_ofc``); the map-reader notebook runs anywhere.

Pipeline overview
=================

The Snakemake pipeline (``pipelines/Snakefile``) processes each ``param_set``
(a FAM Butler collection + processing variant, e.g.
``fam_danish_1_0_wep17_3_0_bin2x``) into the per-``mi_name`` MIW maps::

    per param_set                          per param_set × mi_name
    ─────────────────────────              ──────────────────────────────────
    mktable ──► fit          (per chunk)   build_intrinsic   (per rotator bin)
       │         │                              │
       ▼         ▼                              ▼
    combine_{donuts,fits,visits} ─────────► intrinsic_split  (OCS + CCS maps)

.. list-table::
   :header-rows: 1
   :widths: 14 22 28 36

   * - Step
     - Granularity
     - Entry point
     - Short description
   * - ``mktable``
     - per chunk
     - ``bin.src/run_mktable.py``
     - Butler → per-donut Zernike table + per-visit table
   * - ``fit``
     - per chunk
     - ``bin.src/run_dz_fit.py``
     - Double-Zernike fit of (data − batoid intrinsic) per visit
   * - ``combine_*``
     - per param_set
     - ``bin.src/combine_parquets.py``
     - Concatenate chunks → one donuts/fits/visits table each
   * - ``build_intrinsic``
     - per (ps, mi, rotator bin)
     - ``bin.src/run_build_intrinsic.py``
     - Measured-intrinsic focal-plane grid (Path-A U-mode constrained)
   * - ``intrinsic_split``
     - per (ps, mi)
     - ``bin.src/run_intrinsic_split.py``
     - Decompose the grids into telescope-fixed (OCS) + camera-fixed (CCS) maps

Library modules live under ``python/lsst/ts/intrinsic/wavefront/``; the entry
points are ``bin.src/`` scripts, on PATH after ``setup -r . && scons``.

Pipeline steps in detail
========================

``mktable`` — donut tables (per chunk)
--------------------------------------
``bin.src/run_mktable.py`` (library: ``intrinsics_lib``). Per date chunk:
queries ConsDB for FAM visits, extracts per-donut Zernikes via the Butler,
attaches OCS/CCS field angles and the tabulated batoid intrinsic
(``zk_intrinsic_{OCS,CCS}``), and writes
``output/<ps>/chunks/<dmin>_<dmax>/{donuts,visits}.parquet``. The expensive
Butler step — deliberately *not* re-triggered by code edits (see Snakefile
comments). Requires RSP (Butler + ConsDB).

``fit`` — Double-Zernike fits (per chunk)
-----------------------------------------
``bin.src/run_dz_fit.py`` (library: ``dz_fitting``). Per chunk: robust (Huber)
Double-Zernike fit of the per-donut residual ``zk_data − zk_intrinsic_<coord>``
for each visit, producing per-visit DZ coefficients, errors, and quality flags
in ``chunks/<d>_<d>/fits.parquet``. ``coord_sys`` (OCS default) is set per
param_set in ``snake_config.yaml``.

``combine_donuts`` / ``combine_fits`` / ``combine_visits`` (per param_set)
--------------------------------------------------------------------------
``bin.src/combine_parquets.py``. Concatenate the chunk tables into one
param_set-level table each: ``output/<ps>/{donuts,fits,visits}.parquet``.
**All downstream steps use the combined tables.** Adding data = adding/editing a
chunk in ``snake_config.yaml``; Snakemake re-runs combine + everything
downstream automatically.

``build_intrinsic`` — measured-intrinsic grid (per rotator bin)
---------------------------------------------------------------
``bin.src/run_build_intrinsic.py`` (libraries: ``measured_intrinsic``,
``intrinsic_build_plots``). Per rotator bin: builds the empirical focal-plane
intrinsic Zernike grid from the FAM donuts via the Path-A U-mode-constrained
method (iterated DZ removal of the reachable wavefront), with CCD-height Z4
handling → ``output/<ps>/<mi>/build/rot_<lo>_<hi>/intrinsic_grid.parquet`` +
validation plots. Each ``mi_name`` entry in ``mi_config.yaml`` (e.g.
``pathA_50_34_i``) defines one build: path, ``n_dof``/``n_keep``,
band/program/elevation selection, rotator bins, and build/split parameters.
RSP-only (needs ``lsst.ts.ofc``/``wep``, ``$TS_CONFIG_MTTCS_DIR``, batoid height
maps).

``intrinsic_split`` — OCS/CCS decomposition (per param_set × mi_name)
---------------------------------------------------------------------
``bin.src/run_intrinsic_split.py`` (library: ``intrinsic_split``). Decomposes
the per-rotator-bin grids into a telescope-fixed component **O** (OCS frame) and
a camera-fixed component **C** (CCS frame, rotating with the rotator) for every
Noll Zernike in use (4–19, 22–26). Spin-aware: astig/coma/trefoil doublets are
combined as ``Z_cos + i·Z_sin`` and decomposed with the spin model; hole-aware
least-squares keeps **O** hole-free. Writes, under ``output/<ps>/<mi>/``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Product
     - Contents
   * - ``intrinsic_split_maps.parquet``
     - **the MIW calibration** — ``thx_deg``, ``thy_deg``, ``Z{j}_OCS``,
       ``Z{j}_CCS`` on the rot≈0 disk grid (µm)
   * - ``intrinsic_split_decomp.parquet``
     - complex polar O/C fields per (j, part) for per-donut reconstruction
   * - ``intrinsic_split_rms.parquet``
     - per-Zernike telescope/camera/residual RMS
   * - ``intrinsic_split.pdf``
     - decomposition diagnostics

The ``intrinsic_split_maps.parquet`` is what ``calibration/stage_miw.py``
freezes into the versioned calibration store, and what the standalone reader
notebook ``notebooks/aos_miw_ocs_ccs_maps.ipynb`` plots.

Calibration tables for the Butler
=================================

``bin.src/run_make_calib_tables.py`` (library: ``calib_tables``) turns an
``intrinsic_split_maps.parquet`` into the **source tables** that
``lsst.ip.isr.IntrinsicZernikes`` ingests (astropy tables with columns ``x``,
``y`` in deg and ``Z{j}`` in µm plus a ``coord_sys`` (``CCS``/``OCS``) ``meta``
key).  The two systems are split by their physical nature, which matches how
``IntrinsicZernikes.getIntrinsicZernikes`` queries them — the OCS point is
rotated by the rotator before interpolation, the CCS point is not:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Product (under ``<out-root>/<version>/``)
     - Contents
   * - ``intrinsic_aberrations_OCS.parquet``
     - the telescope-fixed intrinsics — the **whole** focal-plane map, written
       **once** (detector- and filter-independent; the rotated OCS query can
       land anywhere on the focal plane)
   * - ``intrinsic_aberrations_CCS_det<NNN>.parquet``
     - **per detector**: the smooth camera field ``C`` plus that CCD's
       focal-plane height as a ``Z4`` piston (``ccd_height``; 15 µm/mm).  The
       CCS query stays within the detector, and the height makes it
       detector-specific
   * - ``provenance.yaml``
     - map source, git state, detectors, height source, per-detector pistons,
       the maps' own ``.meta``

Filters are handled at **ingest** time (the tables are filter-independent; the
``physical_filter`` lives only in the dataId), so the Butler dataset type stays
per ``(detector, physical_filter)`` and **no ts_wep change is needed**.

.. code-block:: bash

    # all LSSTCam detectors, batoid_rubin heights, from the frozen v1 maps
    run_make_calib_tables.py \
        --maps calibration/miw/intrinsic_split_maps_v1.parquet --version v2
    # or from a fresh pipeline run
    run_make_calib_tables.py \
        --param-set fam_danish_1_2_0_wep17_6_1_bin2x \
        --mi-name pathA_50_34_i_5rot --version v2
    # quick stack-free smoke test (CCS = smooth field only, no height piston)
    run_make_calib_tables.py --maps <maps> --version test \
        --no-heights --detectors 90 91

Default ``--out-root`` is
``/sdf/group/rubin/repo/aos_imsim/gmegias/intrinsic_zernikes``.  The OCS table
needs only astropy/numpy; the per-detector height piston additionally needs the
LSST stack (cameraGeom + obs_lsst) and the batoid_rubin / metrology height map
(``--no-heights`` skips it).

``bin.src/ingest_calib_tables.py`` is the (separate, **not run automatically**)
ingest step.  It builds one ``IntrinsicZernikes`` **per detector** (the shared
OCS system + that detector's CCS system) and ``butler.put``\ s it for every
requested ``physical_filter``.  Each calibration is verified by default before
any write (interpolators built; ``getIntrinsicZernikes`` finite at interior
points, including through the OCS rotation) — a failure aborts the run;
``--no-verify`` skips it.  It defaults to ``--dry-run``; pass ``--execute`` to
write.  This is the OCS-aware counterpart to ts_wep's ``ingestIntrinsicZernikes``
(which builds a CCS-only calibration from single-table sources already in a
Butler collection).

Configuration
=============

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - File (in ``pipelines/``)
     - Contents
   * - ``param_sets.yaml``
     - Butler repo / FAM collection definitions per param_set (referenced by name)
   * - ``snake_config.yaml``
     - Which param_sets to build, their date chunks, and ``coord_sys``
   * - ``mi_config.yaml``
     - Measured-intrinsic entries per param_set: path, ``n_dof``/``n_keep``
       (scalar or explicit index list), band/program/elevation selection,
       rotator bins, build + split parameters. ``defaults:`` block applies to
       every entry

Rules consume the **resolved per-entry config as a Snakemake** ``params``
**value**, not the config *file* as an input. The ``params`` rerun-trigger then
fires only when *that* entry's resolved settings change — adding or editing one
param_set never invalidates another's cached outputs via the shared file's
mtime. (Editing a shared ``defaults:`` block still propagates to every entry, as
it should.)

Running
=======

See ``pipelines/README.md`` for setup and launch details. In brief, from the
``pipelines/`` directory on the RSP:

.. code-block:: bash

    snakemake -n                              # dry-run: show what is stale
    snakemake -j4 --resources mem_mb=14000    # build
    ./run_snake.sh                            # detached run (survives dropped SSH)
    ./run_snake.sh --mode batch               # submit to Slurm from an s3df node
    snakemake --dag | dot -Tpng > dag.png

Output layout
=============

::

    pipelines/output/<param_set>/
      chunks/<dmin>_<dmax>/ {donuts,fits,visits}.parquet     # per chunk
      {donuts,fits,visits}.parquet                           # combined (downstream input)
      <mi_name>/
        build/rot_<lo>_<hi>/intrinsic_grid.parquet           # per rotator bin
        intrinsic_split_{maps,decomp,rms}.parquet  intrinsic_split.pdf   # MIW maps

``pipelines/output/`` is gitignored and overwritten on rerun; the frozen,
versioned calibration lives in ``calibration/miw/`` instead.

Other docs
==========
- ``pipelines/README.md`` — pipeline quickstart (setup + launch).
- ``calibration/README.md`` — the versioned calibration store + staging.
