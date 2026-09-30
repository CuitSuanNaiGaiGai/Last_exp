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

## Cross-product gap decomposition

To refresh the model evaluation on the exact common test mask without retraining,
then calculate raw-input diagnostics and the gap decomposition for the 2,616
chronological test hours, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=3 python -u code/data/scripts/aligned_test/04_train_compare.py --mode all --skip-training
CUDA_VISIBLE_DEVICES=3 python -u code/data/scripts/aligned_test/05_diagnose_real_vs_synthetic.py --year 2020 --split test --raw-only
CUDA_VISIBLE_DEVICES=3 python -u code/data/scripts/aligned_test/06_decompose_cross_product_gap.py --split test
```

These commands do not train models. The first uses existing `best.pt` files and
refreshes `outputs/metrics/comparison.json`; the second refreshes raw ERA5 and
Synthetic LR metrics on the shared finite RADKLIM/ERA5/Synthetic support. The
third adds 1/5/10/25 km block metrics, circular neighborhood CSI/FSS, and a
train-only ERA5 quantile-mapping comparison. The three summaries include the
ordered timestamp-list SHA-256, common-mask SHA-256, and common pixel-hour count
so their evaluation populations can be compared directly. Outputs are written
under `outputs/gap_decomposition/test/`.

The QM table reports exact mean, standard deviation, and maximum, while P95,
P99, and P99.9 are deterministic estimates from up to 2,048 shared-support
pixels sampled uniformly per test hour.

Run the support and circular-kernel regression checks with:

```bash
python code/data/scripts/aligned_test/tests/test_support_contract.py -v
```
