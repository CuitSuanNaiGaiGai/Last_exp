# Aligned Data Quality and Baseline Experiment

## Goal

Under `code/data/scripts/aligned_test/`, validate the existing ERA5/RADKLIM-YW
aligned files, create a spatially fair RADKLIM-derived synthetic low-resolution
input, make chronological train/validation/test splits, and compare two
same-configuration U-Nets for hourly RADKLIM precipitation reconstruction.

## Existing data facts

- There are 24 aligned monthly NetCDF files and 17,543 hourly samples.
- Each file has a `1021 × 900` RADKLIM grid. The stored map increases northward
  by row and eastward by column; the domain-overlap mask contains 876,532 cells.
- Both precipitation variables use millimetres. In 48 exploratory hourly
  samples, ERA5 had no missing cells inside the overlap mask; RADKLIM had about
  36–43% missing cells there. Validation and model metrics must distinguish
  outside-domain cells from in-domain RADKLIM missing observations.
- The native ERA5 grid is `35 × 57` at 0.25 degrees. The stored mapping
  `era5_to_radklim_bilinear.npz` defines the current ERA5-to-RADKLIM interpolation.
- The first ERA5 time, 2019-01-01 00:00 UTC, is absent. The remaining timeline is
  continuous at one-hour intervals.

## Data flow

### 1. Validate aligned inputs

`01_validate_aligned.py` checks every file's schema and time axis, then samples
40 distinct hours using a fixed seed. It checks time continuity, coordinate
orientation and range, precipitation units and value ranges, and NaNs both
inside and outside the geographic overlap mask. It writes a machine-readable
JSON summary, one CSV row per sampled hour, and a map preview under
`outputs/qc/`.

### 2. Build the synthetic perfect-model input

`02_make_synthetic_lr.py` uses hourly RADKLIM target fields from the aligned
files. It conservatively remaps each field from the native RADKLIM projected
grid to the exact native ERA5 `35 × 57` latitude/longitude grid using
area-weighted average resampling, preserving NoData. It records the valid
source-area fraction for each coarse cell and marks cells below the configured
0.8 support threshold as unavailable. This threshold is measured over source
area contributing to a coarse cell; it is not a claim that RADKLIM covers the
entire geographic ERA5 cell at the archive boundary.

The resulting hourly coarse fields are stored by month. At training and
evaluation time, the synthetic coarse fields are mapped to requested RADKLIM
patches/tiles with the same saved ERA5-to-RADKLIM indices and bilinear weights
used for the real ERA5 input. No full-resolution synthetic time series is stored. The synthetic
and real inputs therefore share the same coarse coordinates, target grid,
interpolation weights, time samples, and geographic mask.

### 3. Make chronological splits

`03_make_splits.py` partitions the 731 UTC calendar dates in order: 512 train,
110 validation, and 109 test days. The ranges are 2019-01-01–2020-05-26,
2020-05-27–2020-09-13, and 2020-09-14–2020-12-31. The missing first hour stays
absent; expected counts are 12,287 train, 2,640 validation, and 2,616 test
hours. It writes date lists, timestamp lists, and a summary under
`outputs/splits/`.

### 4. Train and compare the U-Nets

`04_train_compare.py` trains separate copies of one small U-Net, one for the
real ERA5 input and one for the synthetic low-resolution input. Both runs use
the same seed, architecture, optimizer, samples, and training settings. The
input has precipitation and availability-mask channels; precipitation uses
`log1p(mm)`. The target is hourly RADKLIM precipitation in the same transform.
Masked L1 loss ignores missing targets. Both runs train on the same spatial and
temporal common-support mask, use 128 × 128 patches, batch size 2, and up to
50 epochs. Early stopping monitors overall validation masked L1 in
`log1p(mm/h)` space with patience 5; the best validation checkpoint is retained.
Each training epoch samples one eligible patch per training hour, with
the same patch coordinates used by both runs. The model has four encoder levels
with 16, 32, 64, and 128 channels, GroupNorm, and a single regression output.
Adam uses a learning rate of 0.001. Validation uses one fixed eligible patch
per validation hour. Each epoch records overall masked L1 and masked L1 for
target bins `<0.1`, `0.1–<1`, and `>=1 mm/h` (bins defined in physical units),
prediction mean/P95/P99/max and CSI at 0.1 and 1 mm/h. Prediction statistics
and CSI use inverse-transformed predictions in physical mm/h. Test inference
predicts 256 × 256 output cores with a 112-pixel context halo, then reports MAE, RMSE,
bias, and CSI at 1 mm/h in physical millimetres; inverse-transformed
predictions are clipped at zero. It also reports common support as the fraction
of observed target pixels in the overlap domain that both inputs can provide.

Training uses an isolated PyTorch environment, prefers Apple MPS when
available, and falls back to CPU. The existing base environment is not modified.

## Files and outputs

- `01_validate_aligned.py`: schema and sampled-data checks.
- `02_make_synthetic_lr.py`: conservative coarse-grid creation and recorded
  coverage.
- `03_make_splits.py`: chronological split manifests.
- `04_train_compare.py`: common model-training and comparison entry point.
- `model.py`, `common.py`, `config.json`, `requirements-ml.txt`, and `README.md`:
  shared architecture, paths/mapping helpers, fixed settings, dependencies, and
  run instructions.
- `outputs/qc/`: validation summary, sampled CSV, and preview map.
- `outputs/synthetic_lr/`: monthly ERA5-grid synthetic precipitation and coverage.
- `outputs/splits/`: chronological date and hour manifests.
- `outputs/models/` and `outputs/metrics/`: checkpoints, training logs, and
  comparison results.

Generated outputs and model weights stay under this experiment folder and are
excluded from Git. No train/validation/test data are randomly mixed by frame.

## Known execution constraint

The arm64 machine has 16 GB total memory. PyTorch is installed only in the
experiment's isolated `.venv`; MPS is unavailable in this runtime, so training
and inference use CPU. The implementation streams data and avoids full-resolution
synthetic copies.
