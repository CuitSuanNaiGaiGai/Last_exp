# Aligned Test Experiment Implementation Plan

> **For agentic workers:** Execute this plan inline and in order. These steps share generated artifacts and must not run concurrently.

**Goal:** Validate the 2019–2020 aligned precipitation data, create a RADKLIM-derived ERA5-grid synthetic input, split time chronologically, and compare two identical small U-Nets.

**Architecture:** Keep all experiment code and outputs under `code/data/scripts/aligned_test/`. Read the existing monthly aligned NetCDF files and saved bilinear map directly. Store only the small 35×57 synthetic hourly fields; bilinearly map them to 128×128 model patches on demand. Train and evaluate both methods on identical temporal and spatial support.

**Tech Stack:** Python 3.13, NumPy, netCDF4, Rasterio/GDAL `Resampling.average`, Matplotlib, PyTorch (isolated venv; MPS if available, otherwise CPU).

## Global Constraints

- Do not modify or duplicate the existing raw, processed, or aligned source datasets.
- All code, configs, manifests, reports, checkpoints, and logs belong under `code/data/scripts/aligned_test/`.
- Use 40 reproducible sampled hours for the QC report and retain checks over all 24 files' schemas and time axes.
- Synthetic fields use area-weighted remapping to the exact ERA5 35×57 grid, then the existing saved ERA5-to-RADKLIM bilinear mapping.
- Use the agreed coarse-cell support threshold of 0.8 and report its definition and coverage.
- Split whole UTC dates chronologically into 512/110/109 days; never randomly split individual hours.
- Train the same seeded U-Net configuration for both methods: 128×128 patches, batch size 2, up to 50 epochs, and early-stopping patience 5 on overall validation masked L1.
- Mask missing targets and compare both methods on their common valid support.
- Never materialize a full-resolution synthetic dataset or commit generated datasets/checkpoints.
- Do not create or run a software test suite; verify with the requested data QA and artifact inspection commands.

---

### Task 1: Experiment scaffold and shared readers

**Files:**
- Create: `code/data/scripts/aligned_test/.gitignore`
- Create: `code/data/scripts/aligned_test/config.json`
- Create: `code/data/scripts/aligned_test/common.py`
- Create: `code/data/scripts/aligned_test/requirements-ml.txt`
- Create: `code/data/scripts/aligned_test/README.md`
- Modify: `code/data/scripts/aligned_test/DESIGN.md`

- [x] Set experiment input/output paths relative to the repository root; ignore `outputs/`, `.venv/`, `__pycache__/`, and checkpoint files.
- [x] Define shared configuration for seed 20260927, 40 QC samples, 0.8 source-footprint coverage threshold, 128-pixel patches, batch size 2, up to 50 epochs with early-stopping patience 5, Adam at 0.001, and CSI threshold 1 mm/h.
- [x] Implement one monthly NetCDF LRU reader for aligned and synthetic files, one helper to apply the saved bilinear map to a target-grid crop, one `log1p(mm)` transform helper, and one common-support mask helper.
- [x] Document the script run order and environment setup without copying the source data.
- [x] Check Python syntax and confirm all configured input paths exist.

### Task 2: Aligned data quality report

**Files:**
- Create: `code/data/scripts/aligned_test/01_validate_aligned.py`
- Create at runtime: `code/data/scripts/aligned_test/outputs/qc/aligned_qc.json`
- Create at runtime: `code/data/scripts/aligned_test/outputs/qc/sampled_hours.csv`
- Create at runtime: `code/data/scripts/aligned_test/outputs/qc/map_preview.png`

- [x] Open all 24 aligned files and verify expected month-hour counts, strict one-hour time steps, time uniqueness, matching paired-variable shapes, coordinate consistency, and `mm` units.
- [x] Check the overlap mask and coordinate orientation: row/latitude and projected y increase northward; column/longitude and projected x increase eastward; report geographic bounds and valid-cell count.
- [x] Sample 40 distinct timestamps reproducibly. For each, record ERA5/RADKLIM NaN fractions separately inside and outside the overlap mask, finite min/median/95th/99th/max, and nonnegative-value checks.
- [x] Save JSON/CSV and a map preview with north/east axes labeled.
- [x] Run `python code/data/scripts/aligned_test/01_validate_aligned.py` and inspect the report and preview.

### Task 3: Conservative synthetic low-resolution fields

**Files:**
- Create: `code/data/scripts/aligned_test/02_make_synthetic_lr.py`
- Create at runtime: `code/data/scripts/aligned_test/outputs/synthetic_lr/synthetic_lr_YYYY_MM.nc` for 24 months.

- [x] Read each hourly `precipitation_radklim` target and its RADKLIM crop transform/CRS; flip source rows to north-up for GDAL while preserving the model-facing row order.
- [x] Reproject to the exact native ERA5 35×57 coordinate grid using weighted `Resampling.average`, with source NaNs treated as NoData.
- [x] Reproject a valid-source indicator with average resampling; record the fraction of valid RADKLIM footprint in each coarse cell and set cells below 0.8 to NoData.
- [x] Preserve identical hourly timestamps and millimetre units; write monthly files atomically and support skipping complete outputs.
- [x] Run the generator and check dimensions, time counts, coverage statistics, finite-value range, and output sizes.

### Task 4: Chronological split manifests

**Files:**
- Create: `code/data/scripts/aligned_test/03_make_splits.py`
- Create at runtime: `outputs/splits/train_dates.txt`, `val_dates.txt`, `test_dates.txt`
- Create at runtime: `outputs/splits/train_timestamps.txt`, `val_timestamps.txt`, `test_timestamps.txt`
- Create at runtime: `outputs/splits/split_summary.json`

- [x] Assign 2019-01-01–2020-05-26 to train, 2020-05-27–2020-09-13 to validation, and 2020-09-14–2020-12-31 to test.
- [x] Derive hourly manifests from aligned NetCDF times, retaining the absent 2019-01-01 00:00 hour as absent.
- [x] Check that hour manifests are ordered, pairwise disjoint, and total 12,287/2,640/2,616 hours (17,543 total).
- [x] Run `python code/data/scripts/aligned_test/03_make_splits.py` and inspect the summary.

### Task 5: Shared U-Net and fair comparison

**Files:**
- Create: `code/data/scripts/aligned_test/model.py`
- Create: `code/data/scripts/aligned_test/04_train_compare.py`
- Create at runtime: `outputs/models/{era5,synthetic}/`
- Create at runtime: `outputs/metrics/comparison.json` and `per_hour_metrics.csv`
- Create at runtime: `outputs/training/`

- [x] Implement the same four-level 16/32/64/128-channel GroupNorm U-Net for both input sources, with two input channels (transformed precipitation and availability mask) and one `log1p(mm)` regression output.
- [x] Use deterministic common patch coordinates: one eligible 128×128 patch per training hour each epoch; use batch size 2, Adam at 0.001, masked L1 loss, up to 50 epochs, and early-stopping patience 5.
- [x] Record each epoch's overall and target-intensity validation masked L1, prediction distribution, and CSI at 0.1/1 mm/h.
- [x] Train separate copies from the same initialization seed; validate with one fixed patch per validation hour and retain the lower-validation-loss checkpoint.
- [x] Stream test inference over 256×256 output cores with a 112-pixel context halo; exclude missing targets and pixels where either baseline input is unavailable.
- [x] Report MAE, RMSE, bias, and CSI at 1 mm/h in physical units for each method, along with evaluated counts and common-support fraction.
- [x] Use the isolated PyTorch environment, select Apple MPS if available, and otherwise use CPU; do not modify the base environment.
- [x] Run the smoke training chain and inspect both checkpoints, logs, and comparison metrics.

> User requested stopping the full training run after confirming that the chain
> worked. The completed smoke run uses 2 train hours, 2 validation hours, 1
> test hour, and 1 epoch per model. Full-period training remains unrun.

### Task 6: End-to-end artifact audit

**Files:**
- Modify: `code/data/scripts/aligned_test/README.md`
- Create at runtime: `outputs/run_summary.json`

- [x] Confirm all 24 synthetic files and all three split manifests exist, output timelines match aligned times, and the two runs used identical model/configuration/split identifiers.
- [x] Record final paths, split counts, source support, run device, and metric paths in `outputs/smoke_test/run_summary.json`.
- [x] Check that generated outputs remain ignored by Git and source datasets are unchanged.
