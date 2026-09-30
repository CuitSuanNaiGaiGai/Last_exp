#!/usr/bin/env python3
"""Compare raw and model precipitation fields against RADKLIM-YW."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset
from tqdm import tqdm

from common import (
    ALIGNED_DIR,
    CONFIG,
    DatasetCache,
    OUTPUT_DIR,
    bilinear_to_radklim,
    common_input_support,
    load_bilinear_map,
    read_field,
    require_finite_on_support,
    timestamp_list_sha256,
    update_support_sha256,
)


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("train_compare", HERE / "04_train_compare.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)

CSI_THRESHOLDS = (("0.1", 0.1), ("1", 1.0), ("5", 5.0))
FIELDS = (
    "RADKLIM truth",
    "ERA5 raw",
    "Synthetic LR raw",
    "ERA5 prediction",
    "Synthetic prediction",
)
METRIC_METHODS = FIELDS[1:]
QUANTILE_SEED = int(CONFIG["seed"]) + 20260929


def load_manifest_timestamps(name):
    path = OUTPUT_DIR / "splits" / (name + "_timestamps.txt")
    if not path.is_file():
        return set()
    return set(int(line.strip()) for line in path.read_text().splitlines() if line.strip())


def read_year_records(year, split_filter):
    """Return aligned timestamps with matching synthetic files and time axes."""
    split_sets = {name: load_manifest_timestamps(name) for name in ("train", "val", "test")}
    records = []
    aligned_months = []
    available_synthetic_months = []
    missing_synthetic_months = []

    for aligned_path in sorted(ALIGNED_DIR.glob("aligned_*.nc")):
        try:
            _, year_text, month_text = aligned_path.stem.split("_")
            file_year = int(year_text)
            int(month_text)
        except (TypeError, ValueError):
            continue
        if file_year != year:
            continue
        aligned_months.append(aligned_path.stem)
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / aligned_path.name.replace(
            "aligned_", "synthetic_lr_"
        )
        if not synthetic_path.is_file():
            missing_synthetic_months.append(aligned_path.stem.replace("aligned_", ""))
            continue
        available_synthetic_months.append(aligned_path.stem.replace("aligned_", ""))

        with Dataset(str(aligned_path)) as aligned, Dataset(str(synthetic_path)) as synthetic:
            aligned_times = np.asarray(aligned.variables["time"][:], dtype=np.int64)
            synthetic_times = np.asarray(synthetic.variables["time"][:], dtype=np.int64)
            if not np.array_equal(aligned_times, synthetic_times):
                raise ValueError(
                    "Time axes differ between {} and {}".format(aligned_path.name, synthetic_path.name)
                )
            if aligned.variables["precipitation_era5"].shape != aligned.variables[
                "precipitation_radklim"
            ].shape:
                raise ValueError("ERA5 and RADKLIM grids differ in {}".format(aligned_path.name))
            for index, value in enumerate(aligned_times.tolist()):
                timestamp = int(value)
                if split_filter != "all" and timestamp not in split_sets[split_filter]:
                    continue
                records.append((timestamp, aligned_path, index))

    if split_filter != "all" and not split_sets[split_filter]:
        raise FileNotFoundError(
            "Missing split manifest for '{}'; run 03_make_splits.py first.".format(split_filter)
        )
    if not records:
        raise RuntimeError(
            "No {} timestamps with aligned and synthetic files for year {}. "
            "Aligned months: {}; missing synthetic months: {}".format(
                split_filter, year, aligned_months, missing_synthetic_months
            )
        )

    if split_filter != "all":
        expected_timestamps = sorted(
            timestamp
            for timestamp in split_sets[split_filter]
            if datetime.fromtimestamp(timestamp, timezone.utc).year == year
        )
        actual_timestamps = [record[0] for record in records]
        if actual_timestamps != expected_timestamps:
            missing = sorted(set(expected_timestamps) - set(actual_timestamps))
            unexpected = sorted(set(actual_timestamps) - set(expected_timestamps))
            raise RuntimeError(
                "Refusing partial split evaluation for {}: expected {} timestamps, got {}; "
                "missing={}, unexpected={}".format(
                    split_filter,
                    len(expected_timestamps),
                    len(actual_timestamps),
                    len(missing),
                    len(unexpected),
                )
            )

    split_membership = {}
    for timestamp, _, _ in records:
        matches = [name for name, values in split_sets.items() if timestamp in values]
        if len(matches) > 1:
            raise ValueError("Timestamp {} appears in overlapping splits: {}".format(timestamp, matches))
        split_membership[timestamp] = matches[0] if matches else "unclassified"
    return records, split_membership, aligned_months, available_synthetic_months, missing_synthetic_months


def preflight_synthetic(records):
    """Check one selected timestamp per month before a long model-inference run."""
    first_record_by_month = {}
    for record in records:
        timestamp, aligned_path, index = record
        first_record_by_month.setdefault(aligned_path, (timestamp, aligned_path, index))

    cache = DatasetCache(maximum_open=2)
    rows = []
    for timestamp, aligned_path, index in first_record_by_month.values():
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / aligned_path.name.replace(
            "aligned_", "synthetic_lr_"
        )
        synthetic = cache.get(synthetic_path)
        coarse = read_field(synthetic.variables["precipitation"], index)
        rows.append({
            "month": aligned_path.stem.replace("aligned_", ""),
            "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "finite_coarse_cells": int(np.isfinite(coarse).sum()),
        })
    cache.close()
    return rows


def load_model(mode, device):
    checkpoint_path = OUTPUT_DIR / "models" / mode / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Missing {} checkpoint: {}. Run 04_train_compare.py after rebuilding valid synthetic LR files.".format(
                mode, checkpoint_path
            )
        )
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = TRAIN.SmallUNet(base_channels=CONFIG["model"]["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def new_metric_accumulator():
    return {
        "n_pixels": 0,
        "valid_hours": 0,
        "abs_error": 0.0,
        "squared_error": 0.0,
        "error": 0.0,
        "hits": {label: 0 for label, _ in CSI_THRESHOLDS},
        "false_alarms": {label: 0 for label, _ in CSI_THRESHOLDS},
        "misses": {label: 0 for label, _ in CSI_THRESHOLDS},
    }


def update_metrics(accumulator, estimate, truth):
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if not estimate.size:
        return
    error = estimate - truth
    accumulator["n_pixels"] += int(error.size)
    accumulator["valid_hours"] += 1
    accumulator["abs_error"] += float(np.abs(error).sum())
    accumulator["squared_error"] += float(np.square(error).sum())
    accumulator["error"] += float(error.sum())
    for label, threshold in CSI_THRESHOLDS:
        hits = (estimate >= threshold) & (truth >= threshold)
        false_alarms = (estimate >= threshold) & (truth < threshold)
        misses = (estimate < threshold) & (truth >= threshold)
        accumulator["hits"][label] += int(np.count_nonzero(hits))
        accumulator["false_alarms"][label] += int(np.count_nonzero(false_alarms))
        accumulator["misses"][label] += int(np.count_nonzero(misses))


def finish_metrics(accumulator):
    count = int(accumulator["n_pixels"])
    result = {
        "n_pixels": count,
        "valid_hours": int(accumulator["valid_hours"]),
        "mae_mm_h": accumulator["abs_error"] / count if count else None,
        "rmse_mm_h": math.sqrt(accumulator["squared_error"] / count) if count else None,
        "bias_mm_h": accumulator["error"] / count if count else None,
    }
    for label, _ in CSI_THRESHOLDS:
        hits = int(accumulator["hits"][label])
        false_alarms = int(accumulator["false_alarms"][label])
        misses = int(accumulator["misses"][label])
        denominator = hits + false_alarms + misses
        result["csi_at_{}_mm_h".format(label)] = hits / denominator if denominator else None
        result["hits_at_{}_mm_h".format(label)] = hits
        result["false_alarms_at_{}_mm_h".format(label)] = false_alarms
        result["misses_at_{}_mm_h".format(label)] = misses
    return result


def new_distribution_accumulator():
    return {
        "n_pixels": 0,
        "valid_hours": 0,
        "sum": 0.0,
        "sum_squared": 0.0,
        "max": None,
        "wet_pixels": 0,
        "above": {label: 0 for label, _ in CSI_THRESHOLDS},
        "samples": [],
        "sample_pixels": 0,
    }


def update_distribution(accumulator, values, rng):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return
    values64 = values.astype(np.float64, copy=False)
    accumulator["n_pixels"] += int(values.size)
    accumulator["valid_hours"] += 1
    accumulator["sum"] += float(values64.sum())
    accumulator["sum_squared"] += float(np.square(values64).sum())
    maximum = float(values64.max())
    if accumulator["max"] is None or maximum > accumulator["max"]:
        accumulator["max"] = maximum
    accumulator["wet_pixels"] += int(np.count_nonzero(values > 0.0))
    for label, threshold in CSI_THRESHOLDS:
        accumulator["above"][label] += int(np.count_nonzero(values > threshold))

    # Bounded deterministic pixel sampling keeps multi-year quantiles in memory.
    sample_count = min(values.size, 2048, max(128, int(values.size // 1024)))
    if sample_count == values.size:
        sample = values.copy()
    else:
        sample = values[rng.choice(values.size, sample_count, replace=False)]
    accumulator["samples"].append(sample)
    accumulator["sample_pixels"] += int(sample.size)


def finish_distribution(accumulator):
    count = int(accumulator["n_pixels"])
    if not count:
        return {
            "n_pixels": 0,
            "valid_hours": 0,
            "mean_mm_h": None,
            "std_mm_h": None,
            "max_mm_h": None,
            "p90_mm_h_estimate": None,
            "p95_mm_h_estimate": None,
            "p99_mm_h_estimate": None,
            "p99_9_mm_h_estimate": None,
            "wet_fraction": None,
            "fraction_gt_0.1_mm_h": None,
            "fraction_gt_1_mm_h": None,
            "fraction_gt_5_mm_h": None,
            "quantile_sample_pixels": 0,
        }
    mean = accumulator["sum"] / count
    variance = max(accumulator["sum_squared"] / count - mean * mean, 0.0)
    sample = np.concatenate(accumulator["samples"]) if accumulator["samples"] else np.empty(0)
    result = {
        "n_pixels": count,
        "valid_hours": int(accumulator["valid_hours"]),
        "mean_mm_h": mean,
        "std_mm_h": math.sqrt(variance),
        "max_mm_h": accumulator["max"],
        "p90_mm_h_estimate": float(np.percentile(sample, 90)) if sample.size else None,
        "p95_mm_h_estimate": float(np.percentile(sample, 95)) if sample.size else None,
        "p99_mm_h_estimate": float(np.percentile(sample, 99)) if sample.size else None,
        "p99_9_mm_h_estimate": float(np.percentile(sample, 99.9)) if sample.size else None,
        "wet_fraction": float(accumulator["wet_pixels"]) / count,
        "quantile_sample_pixels": int(accumulator["sample_pixels"]),
    }
    for label, _ in CSI_THRESHOLDS:
        result["fraction_gt_{}_mm_h".format(label)] = float(accumulator["above"][label]) / count
    return result


def _one_hour_metrics(estimate, truth, mask):
    accumulator = new_metric_accumulator()
    update_metrics(accumulator, estimate[mask], truth[mask])
    return finish_metrics(accumulator)


def select_typical_hours(candidates):
    """Select a dry hour and wet hours near the 25th, 50th, 90th and peak levels."""
    if not candidates:
        return []
    selected = []
    dry = [item for item in candidates if item["target_max_mm_h"] <= 0.0]
    if dry:
        selected.append(("dry", max(dry, key=lambda item: item["common_pixels"])))
    else:
        selected.append(("driest_available", min(candidates, key=lambda item: item["wet_fraction"])) )

    wet = [item for item in candidates if item["target_max_mm_h"] > 0.0]
    if wet:
        maxima = np.asarray([item["target_max_mm_h"] for item in wet], dtype=np.float64)
        for quantile, label in ((25, "light_wet_p25"), (50, "typical_wet_p50"), (90, "heavy_wet_p90")):
            target = float(np.percentile(maxima, quantile))
            selected.append((label, min(wet, key=lambda item: abs(item["target_max_mm_h"] - target))))
        selected.append(("peak", max(wet, key=lambda item: item["target_max_mm_h"])))

    merged = {}
    order = []
    for label, item in selected:
        timestamp = item["timestamp"]
        if timestamp not in merged:
            merged[timestamp] = {"labels": [], "item": item}
            order.append(timestamp)
        merged[timestamp]["labels"].append(label)
    return [merged[timestamp] for timestamp in order[:5]]


def render_typical_figures(selected, records_by_timestamp, fields, mapping, models, device, output_dir):
    if not selected or not models:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "typical_hours"
    figure_dir.mkdir(parents=True, exist_ok=True)
    overlap = fields.overlap
    outputs = []

    panel_names = (
        "RADKLIM truth",
        "ERA5 raw",
        "ERA5 prediction",
        "Synthetic LR raw",
        "Synthetic prediction",
    )
    for entry in selected:
        item = entry["item"]
        timestamp = int(item["timestamp"])
        _, aligned_path, _ = records_by_timestamp[timestamp]
        target, era, coarse, _ = fields.get(timestamp)
        synthetic, synthetic_valid = bilinear_to_radklim(coarse, mapping)
        era_prediction = TRAIN.infer_tiled(models["era5"], "era5", era, coarse, mapping, device)
        synthetic_prediction = TRAIN.infer_tiled(
            models["synthetic"], "synthetic", era, coarse, mapping, device
        )
        common = (
            overlap
            & np.isfinite(target)
            & np.isfinite(era)
            & synthetic_valid
            & np.isfinite(synthetic)
            & np.isfinite(era_prediction)
            & np.isfinite(synthetic_prediction)
        )
        if not common.any():
            continue

        arrays = [target.copy(), era.copy(), era_prediction.copy(), synthetic.copy(), synthetic_prediction.copy()]
        for array in arrays:
            array[~common] = np.nan
        sample = np.concatenate([array[common] for array in arrays]).astype(np.float32, copy=False)
        vmax = float(np.percentile(sample, 99.5)) if sample.size else 1.0
        vmax = max(vmax, 0.1)
        with Dataset(str(aligned_path)) as ds:
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)

        fig, axes = plt.subplots(1, 5, figsize=(23, 5.2), sharex=True, sharey=True)
        image = None
        for ax, name, array in zip(axes, panel_names, arrays):
            image = ax.pcolormesh(
                lon,
                lat,
                np.ma.masked_invalid(array),
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(name, fontsize=10)
            ax.set_xlim(2.0, 16.0)
            ax.set_ylim(46.5, 55.0)
            ax.set_xlabel("Longitude (°E)")
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("Latitude (°N)")
        fig.colorbar(image, ax=list(axes), shrink=0.82, pad=0.02, label="Precipitation (mm/h)")
        time_text = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        labels = "+".join(entry["labels"])
        fig.suptitle(
            "{} | {} | RADKLIM max={:.2f} mm/h, wet fraction={:.3f} | shared vmax={:.2f}".format(
                time_text, labels, item["target_max_mm_h"], item["wet_fraction"], vmax
            ),
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        filename = "{}_{}.png".format(labels, datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y%m%dT%H%MZ"))
        figure_path = figure_dir / filename
        fig.savefig(str(figure_path), dpi=160)
        plt.close(fig)
        outputs.append(str(figure_path))
    return outputs


def write_metrics_csv(path, metrics, support_descriptions):
    metric_keys = (
        "valid_hours",
        "n_pixels",
        "mae_mm_h",
        "rmse_mm_h",
        "bias_mm_h",
        "csi_at_0.1_mm_h",
        "csi_at_1_mm_h",
        "csi_at_5_mm_h",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "support") + metric_keys)
        writer.writeheader()
        for method in metrics:
            row = {"method": method, "support": support_descriptions[method]}
            row.update({key: metrics[method].get(key) for key in metric_keys})
            writer.writerow(row)


def write_distribution_csv(path, distributions, support_descriptions):
    keys = (
        "valid_hours",
        "n_pixels",
        "mean_mm_h",
        "std_mm_h",
        "max_mm_h",
        "p90_mm_h_estimate",
        "p95_mm_h_estimate",
        "p99_mm_h_estimate",
        "p99_9_mm_h_estimate",
        "wet_fraction",
        "fraction_gt_0.1_mm_h",
        "fraction_gt_1_mm_h",
        "fraction_gt_5_mm_h",
        "quantile_sample_pixels",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("field", "support") + keys)
        writer.writeheader()
        for name in FIELDS:
            row = {"field": name, "support": support_descriptions[name]}
            row.update({key: distributions[name].get(key) for key in keys})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2019, help="Calendar year to analyze (default: 2019).")
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Optional chronological split filter (default: all hours in the selected year).",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Skip model inference; still report the raw input comparisons and distributions.",
    )
    parser.add_argument(
        "--plot-count",
        type=int,
        default=5,
        choices=(3, 4, 5),
        help="Maximum number of representative five-panel hourly maps to write (default: 5).",
    )
    args = parser.parse_args()

    records, split_membership, aligned_months, synthetic_months, missing_months = read_year_records(
        args.year, args.split
    )
    device = TRAIN.get_device()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    mapping = load_bilinear_map()
    synthetic_preflight = preflight_synthetic(records)
    zero_preflight_months = [row["month"] for row in synthetic_preflight if row["finite_coarse_cells"] == 0]
    positive_preflight_months = [row for row in synthetic_preflight if row["finite_coarse_cells"] > 0]
    print(
        "Year={} split={} candidate_hours={} device={} synthetic_preflight={}/{} sampled months have valid cells".format(
            args.year, args.split, len(records), device, len(positive_preflight_months), len(synthetic_preflight)
        ),
        flush=True,
    )
    if not args.raw_only and synthetic_preflight and not positive_preflight_months:
        raise RuntimeError(
            "Every sampled synthetic LR month has zero finite coarse cells. The current synthetic files "
            "cannot support training or diagnostics. Rebuild all months on the server with:\n"
            "  python -u code/data/scripts/aligned_test/02_make_synthetic_lr.py --overwrite\n"
            "Then check each [done] line reports a positive coarse valid-cell count. Existing 04 checkpoints "
            "may have trained with zero eligible patches and should be checked/retrained."
        )

    models = {}
    checkpoints = {}
    if not args.raw_only:
        for mode in ("era5", "synthetic"):
            models[mode], checkpoints[mode] = load_model(mode, device)

    fields = TRAIN.Fields(mapping)
    overlap = fields.overlap
    rng = np.random.RandomState(QUANTILE_SEED)
    metric_accumulators = {method: new_metric_accumulator() for method in METRIC_METHODS}
    raw_common_metrics = {
        name: new_metric_accumulator() for name in ("ERA5 raw", "Synthetic LR raw")
    }
    common_model_metrics = {name: new_metric_accumulator() for name in ("ERA5 prediction", "Synthetic prediction")}
    distribution_accumulators = {name: new_distribution_accumulator() for name in FIELDS}
    common_distribution_accumulators = {name: new_distribution_accumulator() for name in FIELDS}
    per_hour_rows = []
    plot_candidates = []
    record_by_timestamp = {timestamp: record for record in records for timestamp in (record[0],)}
    observed_target_pixels = 0
    common_support_hours = 0
    common_support_pixel_hours = 0
    hours_by_split = {name: 0 for name in ("train", "val", "test", "unclassified")}
    common_hours_by_split = {name: 0 for name in ("train", "val", "test", "unclassified")}
    support_digest = hashlib.sha256()

    support_descriptions = {
        "RADKLIM truth": "shared paired metric support: finite target, ERA5, and mapped Synthetic LR inside overlap",
        "ERA5 raw": "shared paired metric support: finite target, ERA5, and mapped Synthetic LR inside overlap",
        "Synthetic LR raw": "shared paired metric support: finite target, ERA5, and mapped Synthetic LR inside overlap",
        "ERA5 prediction": "shared paired metric support; model output is required to be finite there",
        "Synthetic prediction": "shared paired metric support; model output is required to be finite there",
    }

    progress = tqdm(
        records,
        total=len(records),
        desc="diagnose {} {}".format(args.year, args.split),
        unit="hour",
        dynamic_ncols=True,
    )
    for position, record in enumerate(progress):
        timestamp, _, _ = record
        target, era, coarse, _ = fields.get(timestamp)
        target_valid = np.isfinite(target) & overlap
        observed_target_pixels += int(target_valid.sum())
        synthetic_raw, synthetic_valid = bilinear_to_radklim(coarse, mapping)
        raw_common_mask = common_input_support(
            target,
            era,
            synthetic_valid & np.isfinite(synthetic_raw),
            overlap,
        )
        require_finite_on_support("Synthetic LR raw", synthetic_raw, raw_common_mask)
        update_support_sha256(support_digest, timestamp, raw_common_mask)
        common_support_pixel_hours += int(raw_common_mask.sum())
        if raw_common_mask.any():
            update_metrics(metric_accumulators["ERA5 raw"], era[raw_common_mask], target[raw_common_mask])
            update_metrics(
                metric_accumulators["Synthetic LR raw"],
                synthetic_raw[raw_common_mask],
                target[raw_common_mask],
            )
            update_distribution(distribution_accumulators["ERA5 raw"], era[raw_common_mask], rng)
            update_distribution(distribution_accumulators["Synthetic LR raw"], synthetic_raw[raw_common_mask], rng)
            update_distribution(common_distribution_accumulators["RADKLIM truth"], target[raw_common_mask], rng)
            update_distribution(common_distribution_accumulators["ERA5 raw"], era[raw_common_mask], rng)
            update_distribution(
                common_distribution_accumulators["Synthetic LR raw"], synthetic_raw[raw_common_mask], rng
            )
            update_metrics(raw_common_metrics["ERA5 raw"], era[raw_common_mask], target[raw_common_mask])
            update_metrics(
                raw_common_metrics["Synthetic LR raw"],
                synthetic_raw[raw_common_mask],
                target[raw_common_mask],
            )
        if target_valid.any():
            update_distribution(distribution_accumulators["RADKLIM truth"], target[target_valid], rng)

        era_prediction = None
        synthetic_prediction = None
        era_prediction_mask = np.zeros(target.shape, dtype=bool)
        synthetic_prediction_mask = np.zeros(target.shape, dtype=bool)
        if models:
            if raw_common_mask.any():
                era_prediction = TRAIN.infer_tiled(
                    models["era5"], "era5", era, coarse, mapping, device
                )
                synthetic_prediction = TRAIN.infer_tiled(
                    models["synthetic"], "synthetic", era, coarse, mapping, device
                )
                require_finite_on_support("ERA5 prediction", era_prediction, raw_common_mask)
                require_finite_on_support("Synthetic prediction", synthetic_prediction, raw_common_mask)
                era_prediction_mask = raw_common_mask.copy()
                synthetic_prediction_mask = raw_common_mask.copy()
            if raw_common_mask.any():
                update_metrics(
                    metric_accumulators["ERA5 prediction"],
                    era_prediction[raw_common_mask],
                    target[raw_common_mask],
                )
                update_distribution(
                    distribution_accumulators["ERA5 prediction"],
                    era_prediction[raw_common_mask],
                    rng,
                )
                update_metrics(
                    metric_accumulators["Synthetic prediction"],
                    synthetic_prediction[raw_common_mask],
                    target[raw_common_mask],
                )
                update_distribution(
                    distribution_accumulators["Synthetic prediction"],
                    synthetic_prediction[raw_common_mask],
                    rng,
                )
                update_distribution(
                    common_distribution_accumulators["ERA5 prediction"],
                    era_prediction[raw_common_mask],
                    rng,
                )
                update_distribution(
                    common_distribution_accumulators["Synthetic prediction"],
                    synthetic_prediction[raw_common_mask],
                    rng,
                )
                update_metrics(
                    common_model_metrics["ERA5 prediction"],
                    era_prediction[raw_common_mask],
                    target[raw_common_mask],
                )
                update_metrics(
                    common_model_metrics["Synthetic prediction"],
                    synthetic_prediction[raw_common_mask],
                    target[raw_common_mask],
                )

        method_masks = {
            "ERA5 raw": raw_common_mask,
            "Synthetic LR raw": raw_common_mask,
            "ERA5 prediction": era_prediction_mask,
            "Synthetic prediction": synthetic_prediction_mask,
        }
        row = {
            "time_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "split": split_membership.get(timestamp, "unclassified"),
            "target_valid_pixels": int(target_valid.sum()),
        }
        for method in METRIC_METHODS:
            mask = method_masks[method]
            estimate = {
                "ERA5 raw": era,
                "Synthetic LR raw": synthetic_raw,
                "ERA5 prediction": era_prediction,
                "Synthetic prediction": synthetic_prediction,
            }[method]
            row[method + "_valid_pixels"] = int(mask.sum())
            if estimate is None or not mask.any():
                metrics = finish_metrics(new_metric_accumulator())
            else:
                metrics = _one_hour_metrics(estimate, target, mask)
            for key in (
                "mae_mm_h",
                "rmse_mm_h",
                "bias_mm_h",
                "csi_at_0.1_mm_h",
                "csi_at_1_mm_h",
                "csi_at_5_mm_h",
            ):
                row[method + "_" + key] = metrics[key]

        shared_mask = raw_common_mask
        split_name = split_membership.get(timestamp, "unclassified")
        if shared_mask.any():
            common_support_hours += 1
            hours_by_split[split_name] += 1
            common_hours_by_split[split_name] += 1
            truth_values = target[shared_mask]
            if models:
                plot_candidates.append({
                    "timestamp": int(timestamp),
                    "common_pixels": int(shared_mask.sum()),
                    "target_max_mm_h": float(truth_values.max()),
                    "wet_fraction": float(np.count_nonzero(truth_values > 0.1)) / int(truth_values.size),
                })
        per_hour_rows.append(row)
        progress.set_postfix(
            era_pixels=int(raw_common_mask.sum()),
            synthetic_pixels=int(raw_common_mask.sum()),
            shared_pixels=int(shared_mask.sum()),
            refresh=False,
        )

    fields.close()
    metrics = {name: finish_metrics(value) for name, value in metric_accumulators.items()}
    raw_common_metrics_result = {
        name: finish_metrics(value) for name, value in raw_common_metrics.items()
    }
    model_common_metrics_result = {name: finish_metrics(value) for name, value in common_model_metrics.items()}
    expected_shared_pixels = int(common_support_pixel_hours)
    for name in ("ERA5 raw", "Synthetic LR raw"):
        if metrics[name]["n_pixels"] != expected_shared_pixels:
            raise RuntimeError("{} did not use the full shared common support".format(name))
        if raw_common_metrics_result[name]["n_pixels"] != expected_shared_pixels:
            raise RuntimeError("{} paired baseline denominator differs from shared support".format(name))
    if models:
        for name in ("ERA5 prediction", "Synthetic prediction"):
            if metrics[name]["n_pixels"] != expected_shared_pixels:
                raise RuntimeError("{} did not use the full shared common support".format(name))
            if model_common_metrics_result[name]["n_pixels"] != expected_shared_pixels:
                raise RuntimeError("{} paired model denominator differs from shared support".format(name))
    distributions = {name: finish_distribution(value) for name, value in distribution_accumulators.items()}
    common_distributions = {
        name: finish_distribution(value) for name, value in common_distribution_accumulators.items()
    }

    typical = select_typical_hours(plot_candidates)[: args.plot_count] if models else []
    output_dir = OUTPUT_DIR / "diagnostics_real_vs_synthetic" / "{}_{}".format(args.year, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Re-open the small monthly cache for the selected figures.
    figure_fields = TRAIN.Fields(mapping) if typical and models else None
    figure_warning = None
    figures = []
    if figure_fields is not None:
        try:
            figures = render_typical_figures(
                typical,
                record_by_timestamp,
                figure_fields,
                mapping,
                models,
                device,
                output_dir,
            )
        except Exception as exc:
            # Keep the hourly statistics even if optional map rendering fails.
            figure_warning = "Typical-hour figure generation failed: {}: {}".format(
                type(exc).__name__, exc
            )
            print("[warning] " + figure_warning, flush=True)
        finally:
            figure_fields.close()

    era_prediction_stats = distributions["ERA5 prediction"]
    summary = {
        "year": int(args.year),
        "split_filter": args.split,
        "hours_in_time_axis": int(len(records)),
        "timestamp_list_sha256": timestamp_list_sha256([record[0] for record in records]),
        "common_support_sha256": support_digest.hexdigest(),
        "common_support_pixel_hours": int(common_support_pixel_hours),
        "common_support_hours_with_valid_pixels": int(common_support_hours),
        "observed_RADKLIM_pixels_in_overlap_total": int(observed_target_pixels),
        "metric_supports": {
            "ERA5 raw": support_descriptions["ERA5 raw"],
            "Synthetic LR raw": support_descriptions["Synthetic LR raw"],
            "ERA5 prediction": support_descriptions["ERA5 prediction"],
            "Synthetic prediction": support_descriptions["Synthetic prediction"],
            "shared_common": "same pixels with finite RADKLIM, ERA5 raw, and mapped Synthetic LR inside overlap; predictions must be finite on it",
        },
        "raw_input_vs_RADKLIM": {
            "ERA5 raw": metrics["ERA5 raw"],
            "Synthetic LR raw": metrics["Synthetic LR raw"],
            "comparison_uses_pairwise_support": False,
            "comparison_uses_three_field_common_support": True,
            "both_inputs_same_common_support": raw_common_metrics_result,
            "mae_winner": (
                "ERA5 raw"
                if raw_common_metrics_result["ERA5 raw"]["mae_mm_h"] is not None
                and raw_common_metrics_result["Synthetic LR raw"]["mae_mm_h"] is not None
                and raw_common_metrics_result["ERA5 raw"]["mae_mm_h"]
                < raw_common_metrics_result["Synthetic LR raw"]["mae_mm_h"]
                else "Synthetic LR raw"
                if raw_common_metrics_result["ERA5 raw"]["mae_mm_h"] is not None
                and raw_common_metrics_result["Synthetic LR raw"]["mae_mm_h"] is not None
                and raw_common_metrics_result["Synthetic LR raw"]["mae_mm_h"]
                < raw_common_metrics_result["ERA5 raw"]["mae_mm_h"]
                else "tie_or_unavailable"
            ),
        },
        "model_prediction_vs_RADKLIM": {
            "ERA5 prediction": metrics["ERA5 prediction"],
            "Synthetic prediction": metrics["Synthetic prediction"],
            "both_models_same_common_support": model_common_metrics_result,
        },
        "distributions": distributions,
        "distributions_on_same_shared_support": common_distributions,
        "ERA5_prediction_collapse_check": {
            "valid_hours": era_prediction_stats["valid_hours"],
            "n_pixels": era_prediction_stats["n_pixels"],
            "pred_max_mm_h": era_prediction_stats["max_mm_h"],
            "pred_P99_mm_h_estimate": era_prediction_stats["p99_mm_h_estimate"],
            "pred_P99_9_mm_h_estimate": era_prediction_stats["p99_9_mm_h_estimate"],
            "fraction_pred_gt_0.1": era_prediction_stats["fraction_gt_0.1_mm_h"],
            "fraction_pred_gt_1.0": era_prediction_stats["fraction_gt_1_mm_h"],
            "fraction_pred_gt_5.0": era_prediction_stats["fraction_gt_5_mm_h"],
        },
        "typical_hours": [
            {
                "labels": entry["labels"],
                "time_utc": datetime.fromtimestamp(entry["item"]["timestamp"], timezone.utc).isoformat(),
                "RADKLIM_max_mm_h": entry["item"]["target_max_mm_h"],
                "RADKLIM_fraction_gt_0.1": entry["item"]["wet_fraction"],
                "common_pixels": entry["item"]["common_pixels"],
            }
            for entry in typical
        ],
        "typical_figure_paths": figures,
        "synthetic_preflight_by_month": synthetic_preflight,
        "zero_finite_synthetic_preflight_months": zero_preflight_months,
        "missing_synthetic_month_files": missing_months,
        "aligned_month_files": len(aligned_months),
        "synthetic_month_files_used": len(synthetic_months),
        "checkpoint_epoch": {mode: int(checkpoints[mode].get("epoch", -1)) for mode in checkpoints},
        "device": str(device),
        "quantile_method": (
            "deterministic per-hour random pixel sample; means/std/max/wet fractions are exact over each field's valid pixels"
        ),
        "definitions": {
            "bias": "estimate minus RADKLIM truth; positive means overestimation",
            "CSI": "hits / (hits + false alarms + misses), with both estimate and truth >= threshold",
            "wet_fraction": "fraction of valid pixels with precipitation > 0 mm/h",
            "distribution_support": "RADKLIM truth's standalone distribution uses all observed overlap pixels; raw inputs and model predictions use the shared three-field support, also repeated in the shared-support table",
            "units": "hourly accumulation in mm, numerically equivalent to mm/h for these hourly samples",
        },
        "warnings": [],
    }
    if figure_warning:
        summary["warnings"].append(figure_warning)
    if args.raw_only:
        summary["warnings"].append("Model inference was skipped because --raw-only was specified.")
    if zero_preflight_months:
        summary["warnings"].append(
            "Some sampled synthetic LR months have zero finite coarse cells; their synthetic/model support is unavailable."
        )
    if not common_support_hours and not args.raw_only:
        summary["warnings"].append(
            "No hour had all five fields on common finite support; typical five-panel maps were not created."
        )
    if missing_months:
        summary["warnings"].append("Months without synthetic files were excluded from diagnostics.")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_metrics_csv(output_dir / "metrics_by_method.csv", metrics, support_descriptions)
    write_metrics_csv(
        output_dir / "raw_common_support_metrics.csv",
        raw_common_metrics_result,
        {
            "ERA5 raw": "same pixels with finite RADKLIM, ERA5 raw, and Synthetic LR raw",
            "Synthetic LR raw": "same pixels with finite RADKLIM, ERA5 raw, and Synthetic LR raw",
        },
    )
    write_distribution_csv(output_dir / "distributions.csv", distributions, support_descriptions)
    write_distribution_csv(
        output_dir / "distributions_common_support.csv",
        common_distributions,
        {name: "identical three-field input common pixels; prediction fields use that same support when models are loaded" for name in FIELDS},
    )

    with (output_dir / "per_hour_metrics.csv").open("w", newline="") as stream:
        base_keys = ("time_utc", "split", "target_valid_pixels")
        method_keys = []
        for method in METRIC_METHODS:
            method_keys.append(method + "_valid_pixels")
            method_keys.extend(
                method + "_" + key
                for key in (
                    "mae_mm_h",
                    "rmse_mm_h",
                    "bias_mm_h",
                    "csi_at_0.1_mm_h",
                    "csi_at_1_mm_h",
                    "csi_at_5_mm_h",
                )
            )
        writer = csv.DictWriter(stream, fieldnames=base_keys + tuple(method_keys))
        writer.writeheader()
        writer.writerows(per_hour_rows)

    with (output_dir / "typical_hours.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("labels", "time_utc", "RADKLIM_max_mm_h", "RADKLIM_fraction_gt_0.1", "common_pixels"),
        )
        writer.writeheader()
        for entry in summary["typical_hours"]:
            writer.writerow({**entry, "labels": "+".join(entry["labels"])})

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "candidate_hours": len(records),
                "raw_input_vs_RADKLIM": summary["raw_input_vs_RADKLIM"],
                "model_prediction_vs_RADKLIM": summary["model_prediction_vs_RADKLIM"],
                "distributions": summary["distributions"],
                "ERA5_prediction_collapse_check": summary["ERA5_prediction_collapse_check"],
                "typical_hours": summary["typical_hours"],
                "typical_figure_paths": figures,
                "warnings": summary["warnings"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
