# Test-period diagnostics figure contract

## Claim and scope

The figures compare raw and U-Net precipitation estimates against RADKLIM-YW
truth on the common valid pixels of the fixed chronological test period. They
are descriptive diagnostics for the current 2-epoch checkpoints, not a claim of
generalization across years, models, or random seeds.

- **Archetypes:** quantitative comparison panels plus representative geospatial
  image plates.
- **Backend:** Python / Matplotlib only.
- **Data:** all 2,616 test hours; 1,278,647,605 common pixel-hours in the current
  completed full run, subject to being recomputed from the current checkpoints.
- **Common support:** RADKLIM target, ERA5 input, and synthetic input must all be
  finite at the pixel and hour. This identical support is used for raw inputs,
  predictions, and intensity strata.
- **Intensity bins:** dry means target exactly 0; `0–0.1` means `0 < target <
  0.1`; `0.1–1` means `0.1 ≤ target < 1`; `1–5` means `1 ≤ target < 5`; and
  `5 mm/h` means target `≥ 5 mm/h`.

## Panel map

1. `raw_input_vs_target`: ERA5 and synthetic raw input versus RADKLIM hexbin
   density scatter plots. The seeded sample is capped at 256 paired pixels per
   hour for display only; the displayed MAE/RMSE/bias and CSV scores use all
   common-support test pixels.
2. `prediction_distribution`: all-pixel hourly distributions for ERA5 U-Net,
   synthetic U-Net, and RADKLIM truth. The plot uses logarithmic positive-value
   bins; exact zeros and positive values below 0.001 mm/h are reported
   separately in the CSV/JSON.
3. `intensity_stratified_metrics`: MAE, RMSE, and bias for both raw inputs and
   both trained models in the five disjoint target-intensity bins. Counts are
   shown per bin and retained in CSV.
4. `typical_*.png/.pdf/.svg/.tiff`: one distinct test hour per intensity bin,
   selected as the hour with the largest within-bin area fraction on common
   support. Map display samples every fourth RADKLIM pixel for rendering only;
   score calculations always use the full-resolution grid. A shared 99.5th
   percentile-based color maximum is used across the five fields within an hour.

The hexbin sample is actual observed data selected by a fixed RNG seed; no
simulated values are used. Mapping input values below zero is not expected from
the nonnegative precipitation source. The log-axis plot only receives positive
histogram bins; exact zero values are annotated separately.

## Output set

- `outputs/diagnostics/diagnostics_summary.json`
- `outputs/diagnostics/overall_metrics.csv`
- `outputs/diagnostics/intensity_stratified_metrics.csv`
- `outputs/diagnostics/prediction_distribution.csv`
- `outputs/diagnostics/typical_hours.json`
- PNG, editable-text PDF/SVG, and 600 dpi TIFF exports for the three summary
  figures and each selected hourly field plate.

Wide five-panel hourly plates are intended for direct visual inspection rather
than a single-column manuscript placement. The plotted rainfall values remain
in mm/h; values above a plate's shared color maximum are saturated, with that
maximum recorded in the figure title.
