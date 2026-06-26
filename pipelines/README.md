# Measured-intrinsic calibration pipeline

Snakemake driver that builds the **Measured Intrinsic Wavefront (MIW) OCS/CCS
maps** — the calibration consumed by the AOS online system and read by
[`notebooks/aos_miw_ocs_ccs_maps.ipynb`](../notebooks/aos_miw_ocs_ccs_maps.ipynb).

Chain: `mktable → fit → combine → build_intrinsic → intrinsic_split`, producing
`output/<param_set>/<mi_name>/intrinsic_split_maps.parquet`.

This is the minimal calibration-generation subset of Aaron Roodman's
`rubin-work/aos` pipeline; the analysis/study/WFS rules are intentionally not
included here.

## Setup (once, in an LSST stack shell)

```bash
cd <path to>/ts_intrinsic_wavefront
setup -k -r .          # puts the package on PYTHONPATH and bin/ on PATH
scons                  # generates python/.../version.py and installs bin/ scripts
```

`setup` is required so `lsst.ts.intrinsic.wavefront` imports and the `run_*.py`
entry points resolve by bare name. RSP-only: the build needs `ts_wep`,
`ts_ofc`, `obs_lsst`, `summit_utils`, Butler access, and batoid height maps.

## Run

Run from **this directory** (`pipelines/`) so the configs and `output/` resolve
relative to it:

```bash
cd pipelines
snakemake -n                      # dry-run: show the plan
snakemake -j4 --resources mem_mb=14000   # build (RSP terminal)
./run_snake.sh                    # detached local run (nohup)
./run_snake.sh --mode batch       # submit to Slurm (from an s3df node)
```

Configuration:
- `param_sets.yaml` — Butler repo / collections / programs per param_set
- `snake_config.yaml` — param_set → date chunks, coord_sys
- `mi_config.yaml` — measured-intrinsic entries (rotator bins, split knobs)

## Freeze a calibration

After a clean run, copy the maps into the versioned, tracked calibration store:

```bash
python ../calibration/stage_miw.py \
    --param-set fam_danish_1_0_wep17_3_0_bin2x \
    --mi-name pathA_50_34_i_5rot --version v1
```

See [`../calibration/README.md`](../calibration/README.md).
