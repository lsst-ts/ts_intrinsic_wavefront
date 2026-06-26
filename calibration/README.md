# AOS calibration products

Versioned, **frozen** copies of calibration parquets — currently the Measured
Intrinsic Wavefront (MIW) OCS/CCS maps — kept under version control so a
calibration used online or in a tech note is stable and reproducible, decoupled
from pipeline reruns (which overwrite `aos/output/...`).

```
calibration/
  README.md
  stage_miw.py                              # copy a maps parquet from output/ + write provenance
  miw/
    intrinsic_split_maps_v1.parquet         # OCS/CCS MIW maps (thx_deg, thy_deg, Z{j}_OCS/_CCS)
    intrinsic_split_maps_v1.provenance.yaml # param_set, mi_name, source, git sha/tag, time, maps .meta
    intrinsic_split_maps_v2.parquet         # next version, side by side
    ...
```

## Why here (and not `output/`)
`*/output/*` is gitignored and is overwritten on every rerun. Calibration products
must be **tracked and immutable per version**, so they live here. They are small
(the maps table is one disk-grid × a few dozen Zernike columns), so committing them
is the deliberate exception to the "no parquet in git" convention — the same spirit
as committing figures in `notes/`. (The larger reconstruction `decomp` parquet is
staged only with `--with-decomp`; keep an eye on its size.)

## Versioning
- **Version in the filename** (`intrinsic_split_maps_v1.parquet`) is the primary,
  self-describing handle — it travels with the file when it is deployed online, so a
  consumer always knows which calibration they hold. Bump it (`v2`, …) when the
  calibration changes (new data, new method, different `rotator_select`). Versions sit
  side by side in `miw/` so old and new are both available.
- **Git tag** the repo state that produced it (`miw-v1`) for full provenance;
  the per-file `*.provenance.yaml` records the source path, the git sha/describe, the
  staging time, and the maps' own `.meta` (rotator angles used, `rotator_select`,
  `ocs_only`, …).

## Staging a new version
After a clean pipeline run:
```bash
python calibration/stage_miw.py \
    --param-set fam_danish_1_0_wep17_3_0_bin2x \
    --mi-name pathA_50_34_i_5rot --version v1
git add calibration/miw/intrinsic_split_maps_v1.*
git commit -m "MIW calibration v1: fam_danish_1_0 / pathA_50_34_i_5rot (March-only)"
git tag -a miw-v1 -m "MIW calibration v1"
```

## Reading it
`notebooks/aos_miw_ocs_ccs_maps.ipynb` reads `miw/intrinsic_split_maps_<version>.parquet`
(set the `version` parameter). For online/AOS use, point its `maps_path` at the
deployed copy of that file.
