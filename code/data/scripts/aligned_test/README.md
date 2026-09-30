# Aligned data experiment

All experiment code and artifacts for aligned-data validation, a conservative
RADKLIM-derived synthetic low-resolution input, chronological splits, and a
small U-Net comparison live in this directory.

On the `new` GPU server, activate the existing environment and run from the
repository root:

```bash
conda activate downscaling
python code/data/scripts/aligned_test/01_validate_aligned.py
python code/data/scripts/aligned_test/02_make_synthetic_lr.py
python code/data/scripts/aligned_test/03_make_splits.py
python code/data/scripts/aligned_test/04_train_compare.py
```

The `downscaling` environment on `new` provides NumPy, netCDF4, Rasterio,
Matplotlib, and CUDA-enabled PyTorch. To confirm CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run a small end-to-end training-chain check (two train hours, two validation
hours, one test hour, one epoch per method) with:

```bash
python code/data/scripts/aligned_test/04_train_compare.py --smoke-test
```

This writes checkpoints and metrics only under `outputs/smoke_test/`; those
numbers verify the pipeline and are not full-period comparison results. To run
the full configured experiment, omit `--smoke-test`. Training selects CUDA when
available, then Apple MPS, and otherwise uses CPU. Generated NetCDF, reports, plots, logs,
and model weights are written below `outputs/` and ignored by Git. Raw and
processed source datasets are never copied.

After full-run checkpoints and `outputs/metrics/comparison.json` exist, generate
test-period input/prediction distributions, intensity-stratified scores, and
representative hourly maps with:

```bash
python code/data/scripts/aligned_test/05_make_diagnostics.py
```

The diagnostic figures, CSV tables, and selection record are written under
`outputs/diagnostics/`. The five-panel hourly maps compare RADKLIM truth, both
raw inputs, and both U-Net predictions using a shared color scale per hour.
