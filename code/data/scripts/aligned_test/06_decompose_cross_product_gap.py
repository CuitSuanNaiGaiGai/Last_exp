#!/usr/bin/env python3
"""Decompose ERA5/RADKLIM cross-product errors on the chronological test split.

ERA5 here means the aligned ERA5 precipitation field, already bilinearly mapped
onto the native 1 km RADKLIM-YW grid. The script does not train or modify either
U-Net. It fits a deterministic, global empirical quantile map from the train
split, then evaluates raw inputs and the existing best checkpoints on the
selected split (test by default).

Outputs are written to outputs/gap_decomposition/<split>/.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import time

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
    load_bilinear_map,
    read_field,
)


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("gap_decomposition_train", HERE / "04_train_compare.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)

THRESHOLDS = (("0.1", 0.1), ("1", 1.0), ("5", 5.0))
SCALES_KM = (1, 5, 10, 25)
ESTIMATE_NAMES = (
    "ERA5 raw",
    "Synthetic LR raw",
    "ERA5 prediction",
    "Synthetic prediction",
)
RAW_NAMES = ("ERA5 raw", "Synthetic LR raw")
INTENSITY_BINS = (
    ("target_lt_0.1", "RADKLIM < 0.1 mm/h"),
    ("target_0.1_to_lt_1", "0.1 <= RADKLIM < 1 mm/h"),
    ("target_1_to_lt_5", "1 <= RADKLIM < 5 mm/h"),
    ("target_5_to_10", "5 <= RADKLIM <= 10 mm/h"),
    ("target_gt_10", "RADKLIM > 10 mm/h"),
)


class FieldReader:
    """Read monthly aligned fields and their matching synthetic coarse fields."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.overlap = mapping["overlap_mask"].astype(bool)
        self.aligned_cache = DatasetCache(maximum_open=3)
        self.synthetic_cache = DatasetCache(maximum_open=3)
        self.index = {}
        self.checked_synthetic_files = set()
        for path in sorted(ALIGNED_DIR.glob("aligned_*.nc")):
            with Dataset(str(path)) as dataset:
                times = np.asarray(dataset.variables["time"][:], dtype=np.int64)
                for position, timestamp in enumerate(times.tolist()):
                    timestamp = int(timestamp)
                    if timestamp in self.index:
                        raise ValueError("Duplicate aligned timestamp {}".format(timestamp))
                    self.index[timestamp] = (path, int(position))

    def get_base(self, timestamp):
        try:
            path, index = self.index[int(timestamp)]
        except KeyError:
            raise KeyError("Timestamp {} is absent from aligned NetCDF files".format(timestamp))
        dataset = self.aligned_cache.get(path)
        target = read_field(dataset.variables["precipitation_radklim"], index)
        era = read_field(dataset.variables["precipitation_era5"], index)
        if target.shape != self.overlap.shape or era.shape != self.overlap.shape:
            raise ValueError("Aligned field shape does not match saved RADKLIM overlap mask")
        target[~self.overlap] = np.nan
        era[~self.overlap] = np.nan
        return target, era, path, index

    def get_synthetic(self, timestamp, aligned_path, index):
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / aligned_path.name.replace(
            "aligned_", "synthetic_lr_"
        )
        synthetic = self.synthetic_cache.get(synthetic_path)
        if synthetic_path not in self.checked_synthetic_files:
            aligned = self.aligned_cache.get(aligned_path)
            aligned_times = np.asarray(aligned.variables["time"][:], dtype=np.int64)
            synthetic_times = np.asarray(synthetic.variables["time"][:], dtype=np.int64)
            if not np.array_equal(aligned_times, synthetic_times):
                raise ValueError(
                    "Time axes differ between {} and {}".format(
                        aligned_path.name, synthetic_path.name
                    )
                )
            self.checked_synthetic_files.add(synthetic_path)
        return read_field(synthetic.variables["precipitation"], index)

    def close(self):
        self.aligned_cache.close()
        self.synthetic_cache.close()


def split_timestamps(name):
    return TRAIN.read_manifest(name)


def load_model(mode, device):
    path = OUTPUT_DIR / "models" / mode / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing {} checkpoint: {}. Run 04_train_compare.py first.".format(mode, path)
        )
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)
    model = TRAIN.SmallUNet(base_channels=CONFIG["model"]["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def fit_empirical_quantile_map(train_times, reader, samples_per_hour, seed):
    """Fit a deterministic global marginal ERA5-to-RADKLIM quantile mapping.

    Each UTC hour contributes at most ``samples_per_hour`` uniformly sampled
    common-support pixels. Source and target empirical quantiles are paired by
    rank. Duplicate source quantiles (notably the dry-value mass) are collapsed
    by averaging their corresponding target quantiles, yielding a monotone,
    deterministic lookup table.
    """
    if samples_per_hour < 1:
        raise ValueError("--qm-samples-per-hour must be positive")
    rng = np.random.default_rng(seed)
    source_chunks = []
    target_chunks = []
    sampled_hours = 0
    sampled_pixel_hours = 0
    valid_pixel_hours = 0

    for timestamp in tqdm(
        train_times,
        total=len(train_times),
        desc="Fit train-only ERA5 quantile map",
        unit="hour",
        dynamic_ncols=True,
    ):
        target, era, _, _ = reader.get_base(timestamp)
        support = reader.overlap & np.isfinite(target) & np.isfinite(era)
        flat_indices = np.flatnonzero(support)
        valid_pixel_hours += int(flat_indices.size)
        if not flat_indices.size:
            continue
        take = min(int(flat_indices.size), int(samples_per_hour))
        if flat_indices.size > take:
            flat_indices = rng.choice(flat_indices, size=take, replace=False)
        source_chunks.append(era.ravel()[flat_indices].astype(np.float32, copy=True))
        target_chunks.append(target.ravel()[flat_indices].astype(np.float32, copy=True))
        sampled_hours += 1
        sampled_pixel_hours += int(take)

    if not source_chunks:
        raise RuntimeError("No finite ERA5/RADKLIM train pixels were available for quantile mapping")

    source_sample = np.concatenate(source_chunks).astype(np.float64, copy=False)
    target_sample = np.concatenate(target_chunks).astype(np.float64, copy=False)
    levels = np.linspace(0.0, 1.0, 2049, dtype=np.float64)
    source_knots = np.quantile(source_sample, levels)
    target_knots = np.quantile(target_sample, levels)

    unique_source, inverse = np.unique(source_knots, return_inverse=True)
    target_sums = np.bincount(inverse, weights=target_knots, minlength=unique_source.size)
    target_counts = np.bincount(inverse, minlength=unique_source.size)
    unique_target = target_sums / np.maximum(target_counts, 1)
    unique_target = np.maximum.accumulate(np.maximum(unique_target, 0.0))

    fit = {
        "method": "global deterministic empirical marginal quantile mapping",
        "fit_split": "train",
        "fit_hours_available": int(len(train_times)),
        "fit_hours_sampled": int(sampled_hours),
        "common_train_pixel_hours": int(valid_pixel_hours),
        "sampled_pixel_hours": int(sampled_pixel_hours),
        "max_samples_per_hour": int(samples_per_hour),
        "quantile_knots_before_duplicate_collapse": int(levels.size),
        "lookup_knots_after_duplicate_collapse": int(unique_source.size),
        "source_dry_fraction_sampled": float(np.mean(source_sample <= 0.0)),
        "target_dry_fraction_sampled": float(np.mean(target_sample <= 0.0)),
        "tie_policy": "duplicate ERA5 quantile values map to the mean target quantile over their tied rank interval",
        "random_seed": int(seed),
    }
    return (unique_source.astype(np.float64), unique_target.astype(np.float64)), fit


def apply_quantile_map(values, lookup):
    values = np.asarray(values, dtype=np.float32)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.any():
        source_knots, target_knots = lookup
        mapped = np.interp(
            values[finite].astype(np.float64),
            source_knots,
            target_knots,
            left=float(target_knots[0]),
            right=float(target_knots[-1]),
        )
        result[finite] = np.maximum(mapped, 0.0).astype(np.float32)
    return result


def new_point_accumulator():
    return {
        "n_pixels": 0,
        "valid_hours": 0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "error_sum": 0.0,
        "hits": {label: 0 for label, _ in THRESHOLDS},
        "false_alarms": {label: 0 for label, _ in THRESHOLDS},
        "misses": {label: 0 for label, _ in THRESHOLDS},
        "fss_squared_error_sum": {label: 0.0 for label, _ in THRESHOLDS},
        "fss_denominator_sum": {label: 0.0 for label, _ in THRESHOLDS},
    }


def update_point_metrics(accumulator, estimate, truth):
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if not estimate.size:
        return
    error = estimate - truth
    accumulator["n_pixels"] += int(error.size)
    accumulator["valid_hours"] += 1
    accumulator["absolute_error_sum"] += float(np.abs(error).sum())
    accumulator["squared_error_sum"] += float(np.square(error).sum())
    accumulator["error_sum"] += float(error.sum())
    for label, threshold in THRESHOLDS:
        forecast_event = estimate >= threshold
        observed_event = truth >= threshold
        accumulator["hits"][label] += int(np.count_nonzero(forecast_event & observed_event))
        accumulator["false_alarms"][label] += int(np.count_nonzero(forecast_event & ~observed_event))
        accumulator["misses"][label] += int(np.count_nonzero(~forecast_event & observed_event))
        accumulator["fss_squared_error_sum"][label] += float(
            np.count_nonzero(forecast_event != observed_event)
        )
        accumulator["fss_denominator_sum"][label] += float(
            np.count_nonzero(forecast_event) + np.count_nonzero(observed_event)
        )


def finish_point_metrics(accumulator):
    count = int(accumulator["n_pixels"])
    result = {
        "n_pixel_hours": count,
        "valid_hours": int(accumulator["valid_hours"]),
        "mae_mm_h": accumulator["absolute_error_sum"] / count if count else None,
        "rmse_mm_h": math.sqrt(accumulator["squared_error_sum"] / count) if count else None,
        "bias_mm_h": accumulator["error_sum"] / count if count else None,
        "csi": {},
        "fss_one_cell": {},
    }
    for label, _ in THRESHOLDS:
        hits = int(accumulator["hits"][label])
        false_alarms = int(accumulator["false_alarms"][label])
        misses = int(accumulator["misses"][label])
        denominator = hits + false_alarms + misses
        fss_denom = float(accumulator["fss_denominator_sum"][label])
        result["csi"][label] = {
            "score": hits / denominator if denominator else None,
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
        }
        result["fss_one_cell"][label] = (
            1.0 - accumulator["fss_squared_error_sum"][label] / fss_denom
            if fss_denom
            else None
        )
    return result


def aggregate_common_blocks(target, estimates, support, block_km, min_support_fraction):
    """Area-average common-support pixels into square native-grid blocks."""
    height, width = support.shape
    out_h, out_w = height // block_km, width // block_km
    crop_h, crop_w = out_h * block_km, out_w * block_km
    start_y, start_x = (height - crop_h) // 2, (width - crop_w) // 2
    row_slice = slice(start_y, start_y + crop_h)
    col_slice = slice(start_x, start_x + crop_w)
    block_support = support[row_slice, col_slice].reshape(out_h, block_km, out_w, block_km)
    valid_counts = block_support.sum(axis=(1, 3), dtype=np.int32)
    minimum_count = int(math.ceil(min_support_fraction * block_km * block_km))
    valid_blocks = valid_counts >= minimum_count

    def block_mean(field):
        values = np.asarray(field[row_slice, col_slice], dtype=np.float32)
        values = np.where(support[row_slice, col_slice], values, 0.0)
        values = values.reshape(out_h, block_km, out_w, block_km)
        totals = values.sum(axis=(1, 3), dtype=np.float64)
        mean = np.full((out_h, out_w), np.nan, dtype=np.float32)
        np.divide(
            totals.astype(np.float32),
            valid_counts.astype(np.float32),
            out=mean,
            where=valid_counts > 0,
        )
        mean[~valid_blocks] = np.nan
        return mean

    result = {"RADKLIM truth": block_mean(target)}
    for name in ESTIMATE_NAMES:
        result[name] = block_mean(estimates[name])
    block_support_fraction = valid_counts[valid_blocks].astype(np.float64) / float(block_km * block_km)
    return result, valid_blocks, block_support_fraction


def new_neighborhood_accumulator():
    return {
        "valid_centers": 0,
        "valid_hours": 0,
        "fractional_hits": 0.0,
        "fractional_false_alarms": 0.0,
        "fractional_misses": 0.0,
        "fss_squared_error_sum": 0.0,
        "fss_denominator_sum": 0.0,
    }


def update_neighborhood_metrics(accumulator, forecast_fraction, observed_fraction, valid_centers):
    if not valid_centers.any():
        return
    forecast = forecast_fraction[valid_centers].astype(np.float64, copy=False)
    observed = observed_fraction[valid_centers].astype(np.float64, copy=False)
    accumulator["valid_centers"] += int(forecast.size)
    accumulator["valid_hours"] += 1
    accumulator["fractional_hits"] += float(np.sum(forecast * observed))
    accumulator["fractional_false_alarms"] += float(np.sum(forecast * (1.0 - observed)))
    accumulator["fractional_misses"] += float(np.sum((1.0 - forecast) * observed))
    accumulator["fss_squared_error_sum"] += float(np.sum(np.square(forecast - observed)))
    accumulator["fss_denominator_sum"] += float(np.sum(np.square(forecast) + np.square(observed)))


def finish_neighborhood_metrics(accumulator):
    hits = accumulator["fractional_hits"]
    false_alarms = accumulator["fractional_false_alarms"]
    misses = accumulator["fractional_misses"]
    csi_denom = hits + false_alarms + misses
    fss_denom = accumulator["fss_denominator_sum"]
    return {
        "valid_centers": int(accumulator["valid_centers"]),
        "valid_hours": int(accumulator["valid_hours"]),
        "neighborhood_csi_fractional": hits / csi_denom if csi_denom else None,
        "fractional_hits": hits,
        "fractional_false_alarms": false_alarms,
        "fractional_misses": misses,
        "fss": (
            1.0 - accumulator["fss_squared_error_sum"] / fss_denom
            if fss_denom
            else None
        ),
    }


def event_integral_stack(truth, estimates, support):
    """Build integral images for support and threshold events for all methods."""
    maps = [support.astype(np.int32)]
    channel_names = [(None, None)]
    for method in ESTIMATE_NAMES:
        values = estimates[method]
        for label, threshold in THRESHOLDS:
            maps.append((support & np.isfinite(values) & (values >= threshold)).astype(np.int32))
            channel_names.append((method, label))
    for label, threshold in THRESHOLDS:
        # Observations are stored after estimates so each forecast channel is easy to index.
        maps.append((support & np.isfinite(truth) & (truth >= threshold)).astype(np.int32))
        channel_names.append(("RADKLIM truth", label))

    event_maps = np.stack(maps, axis=0)
    channels, height, width = event_maps.shape
    integral = np.zeros((channels, height + 1, width + 1), dtype=np.int32)
    integral[:, 1:, 1:] = np.cumsum(
        np.cumsum(event_maps, axis=1, dtype=np.int32), axis=2, dtype=np.int32
    )
    channel_index = {key: index for index, key in enumerate(channel_names)}
    return integral, channel_index


def integral_window_sums(integral, window):
    return (
        integral[:, window:, window:]
        - integral[:, :-window, window:]
        - integral[:, window:, :-window]
        + integral[:, :-window, :-window]
    )


def compute_intensity_masks(target):
    return {
        "target_lt_0.1": target < 0.1,
        "target_0.1_to_lt_1": (target >= 0.1) & (target < 1.0),
        "target_1_to_lt_5": (target >= 1.0) & (target < 5.0),
        "target_5_to_10": (target >= 5.0) & (target <= 10.0),
        "target_gt_10": target > 10.0,
    }


def new_error_accumulator():
    return {"n_pixel_hours": 0, "absolute_error_sum": 0.0, "squared_error_sum": 0.0, "error_sum": 0.0}


def update_error_accumulator(accumulator, estimate, target, mask):
    if not mask.any():
        return
    error = estimate[mask].astype(np.float64) - target[mask].astype(np.float64)
    accumulator["n_pixel_hours"] += int(error.size)
    accumulator["absolute_error_sum"] += float(np.abs(error).sum())
    accumulator["squared_error_sum"] += float(np.square(error).sum())
    accumulator["error_sum"] += float(error.sum())


def finish_error_accumulator(accumulator):
    count = int(accumulator["n_pixel_hours"])
    return {
        "n_pixel_hours": count,
        "mae_mm_h": accumulator["absolute_error_sum"] / count if count else None,
        "rmse_mm_h": math.sqrt(accumulator["squared_error_sum"] / count) if count else None,
        "bias_mm_h": accumulator["error_sum"] / count if count else None,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_delta(left, right):
    if left is None or right is None:
        return None
    return float(left - right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Evaluation split (default: test; expected 2,616 hours).",
    )
    parser.add_argument(
        "--qm-samples-per-hour",
        type=int,
        default=512,
        help="Maximum common-support train pixels sampled per hour to fit the empirical quantile map.",
    )
    parser.add_argument(
        "--min-support-fraction",
        type=float,
        default=0.5,
        help="Minimum common-support fraction for a spatial aggregate/window (default: 0.5).",
    )
    args = parser.parse_args()
    if not (0.0 < args.min_support_fraction <= 1.0):
        parser.error("--min-support-fraction must be in (0, 1]")

    started = time.time()
    mapping = load_bilinear_map()
    reader = FieldReader(mapping)
    train_times = split_timestamps("train")
    eval_times = split_timestamps(args.split)
    if not train_times:
        reader.close()
        raise RuntimeError("Train split manifest is empty; run 03_make_splits.py first")
    if not eval_times:
        reader.close()
        raise RuntimeError("{} split manifest is empty; run 03_make_splits.py first".format(args.split))
    missing_timestamps = [timestamp for timestamp in eval_times if timestamp not in reader.index]
    if missing_timestamps:
        reader.close()
        raise RuntimeError("{} evaluation timestamps are missing from aligned files".format(len(missing_timestamps)))
    for mode in ("era5", "synthetic"):
        checkpoint = OUTPUT_DIR / "models" / mode / "best.pt"
        if not checkpoint.is_file():
            reader.close()
            raise FileNotFoundError("Missing model checkpoint: {}".format(checkpoint))

    for timestamp in eval_times:
        aligned_path, _ = reader.index[timestamp]
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / aligned_path.name.replace(
            "aligned_", "synthetic_lr_"
        )
        if not synthetic_path.is_file():
            reader.close()
            raise FileNotFoundError("Missing synthetic LR file required for evaluation: {}".format(synthetic_path))

    output_dir = OUTPUT_DIR / "gap_decomposition" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(CONFIG["seed"]) + 20260930
    print(
        "Gap decomposition: split={} hours={} train_hours={} device pending; ERA5 is already mapped to the RADKLIM grid".format(
            args.split, len(eval_times), len(train_times)
        ),
        flush=True,
    )

    try:
        lookup, qm_fit = fit_empirical_quantile_map(
            train_times, reader, args.qm_samples_per_hour, seed
        )
        device = TRAIN.get_device()
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        models = {}
        checkpoints = {}
        for mode in ("era5", "synthetic"):
            models[mode], checkpoints[mode] = load_model(mode, device)
        print("Evaluation device: {}".format(device), flush=True)

        multiscale_accumulators = {
            (scale, name): new_point_accumulator()
            for scale in SCALES_KM
            for name in ESTIMATE_NAMES
        }
        multiscale_support = {
            scale: {"valid_cells": 0, "possible_cells": 0, "accepted_block_support_sum": 0.0, "accepted_blocks": 0}
            for scale in SCALES_KM
        }
        neighborhood_accumulators = {
            (window, name, label): new_neighborhood_accumulator()
            for window in SCALES_KM
            for name in ESTIMATE_NAMES
            for label, _ in THRESHOLDS
        }
        quantile_accumulators = {
            name: new_point_accumulator()
            for name in ("ERA5 raw", "ERA5 quantile mapped")
        }
        intensity_accumulators = {
            (bin_key, name): new_error_accumulator()
            for bin_key, _ in INTENSITY_BINS
            for name in ESTIMATE_NAMES
        }
        support_totals = {
            "overlap_pixels": int(reader.overlap.sum()),
            "target_valid_pixel_hours": 0,
            "raw_three_way_pixel_hours": 0,
            "five_field_common_pixel_hours": 0,
        }

        progress = tqdm(
            eval_times,
            total=len(eval_times),
            desc="Decompose {} gap".format(args.split),
            unit="hour",
            dynamic_ncols=True,
        )
        for position, timestamp in enumerate(progress):
            target, era, aligned_path, index = reader.get_base(timestamp)
            coarse = reader.get_synthetic(timestamp, aligned_path, index)
            synthetic, synthetic_valid = bilinear_to_radklim(coarse, mapping)
            raw_support = (
                reader.overlap
                & np.isfinite(target)
                & np.isfinite(era)
                & synthetic_valid
                & np.isfinite(synthetic)
            )

            era_prediction = TRAIN.infer_tiled(
                models["era5"], "era5", era, coarse, mapping, device
            )
            synthetic_prediction = TRAIN.infer_tiled(
                models["synthetic"], "synthetic", era, coarse, mapping, device
            )
            estimates = {
                "ERA5 raw": era,
                "Synthetic LR raw": synthetic,
                "ERA5 prediction": era_prediction,
                "Synthetic prediction": synthetic_prediction,
            }
            common = raw_support.copy()
            for values in estimates.values():
                common &= np.isfinite(values)
            support_totals["target_valid_pixel_hours"] += int(
                (reader.overlap & np.isfinite(target)).sum()
            )
            support_totals["raw_three_way_pixel_hours"] += int(raw_support.sum())
            support_totals["five_field_common_pixel_hours"] += int(common.sum())

            # 1. Multiscale point metrics: area-average the same five-way common
            # support into nominal 1/5/10/25 km square cells.
            for scale in SCALES_KM:
                aggregated, valid_blocks, block_support_fraction = aggregate_common_blocks(
                    target, estimates, common, scale, args.min_support_fraction
                )
                support_row = multiscale_support[scale]
                support_row["valid_cells"] += int(valid_blocks.sum())
                support_row["possible_cells"] += int(valid_blocks.size)
                support_row["accepted_block_support_sum"] += float(block_support_fraction.sum())
                support_row["accepted_blocks"] += int(block_support_fraction.size)
                truth_coarse = aggregated["RADKLIM truth"]
                for name in ESTIMATE_NAMES:
                    update_point_metrics(
                        multiscale_accumulators[(scale, name)],
                        aggregated[name][valid_blocks],
                        truth_coarse[valid_blocks],
                    )

            # 2. Spatial tolerance: fractions of event pixels in each moving
            # square window, with identical valid support for all five fields.
            integral, event_channels = event_integral_stack(target, estimates, common)
            for window in SCALES_KM:
                sums = integral_window_sums(integral, window)
                support_count = sums[event_channels[(None, None)]]
                valid_centers = support_count >= int(
                    math.ceil(args.min_support_fraction * window * window)
                )
                denominator = np.maximum(support_count.astype(np.float32), 1.0)
                for label, _ in THRESHOLDS:
                    observed_fraction = (
                        sums[event_channels[("RADKLIM truth", label)]].astype(np.float32)
                        / denominator
                    )
                    for name in ESTIMATE_NAMES:
                        forecast_fraction = (
                            sums[event_channels[(name, label)]].astype(np.float32)
                            / denominator
                        )
                        update_neighborhood_metrics(
                            neighborhood_accumulators[(window, name, label)],
                            forecast_fraction,
                            observed_fraction,
                            valid_centers,
                        )

            # 3. ERA5-only train-fitted marginal correction; raw and mapped
            # ERA5 are scored against target on exactly the same pair support.
            quantile_support = reader.overlap & np.isfinite(target) & np.isfinite(era)
            calibrated_era = apply_quantile_map(era, lookup)
            update_point_metrics(
                quantile_accumulators["ERA5 raw"],
                era[quantile_support],
                target[quantile_support],
            )
            calibrated_support = quantile_support & np.isfinite(calibrated_era)
            update_point_metrics(
                quantile_accumulators["ERA5 quantile mapped"],
                calibrated_era[calibrated_support],
                target[calibrated_support],
            )

            # 4. Per-target-intensity error for raw inputs and both predictions.
            bin_masks = compute_intensity_masks(target)
            for bin_key, _ in INTENSITY_BINS:
                target_bin = common & bin_masks[bin_key]
                for name in ESTIMATE_NAMES:
                    update_error_accumulator(
                        intensity_accumulators[(bin_key, name)],
                        estimates[name],
                        target,
                        target_bin,
                    )

            progress.set_postfix(
                raw_pixels="{:.3g}".format(float(raw_support.sum())),
                shared_pixels="{:.3g}".format(float(common.sum())),
                refresh=False,
            )

        # Convert streaming accumulators to JSON/CSV-ready results.
        multiscale_results = {}
        multiscale_rows = []
        for scale in SCALES_KM:
            multiscale_results[str(scale)] = {}
            support_row = multiscale_support[scale]
            for name in ESTIMATE_NAMES:
                result = finish_point_metrics(multiscale_accumulators[(scale, name)])
                multiscale_results[str(scale)][name] = result
                multiscale_rows.append({
                    "scale_km": scale,
                    "method": name,
                    "n_pixel_hours": result["n_pixel_hours"],
                    "valid_hours": result["valid_hours"],
                    "valid_cells_fraction": (
                        support_row["valid_cells"] / support_row["possible_cells"]
                        if support_row["possible_cells"] else None
                    ),
                    "mean_common_support_fraction_in_accepted_blocks": (
                        support_row["accepted_block_support_sum"] / support_row["accepted_blocks"]
                        if support_row["accepted_blocks"] else None
                    ),
                    "mae_mm_h": result["mae_mm_h"],
                    "rmse_mm_h": result["rmse_mm_h"],
                    "bias_mm_h": result["bias_mm_h"],
                    "csi_at_0.1": result["csi"]["0.1"]["score"],
                    "csi_at_1": result["csi"]["1"]["score"],
                    "csi_at_5": result["csi"]["5"]["score"],
                    "fss_at_0.1_one_cell": result["fss_one_cell"]["0.1"],
                    "fss_at_1_one_cell": result["fss_one_cell"]["1"],
                    "fss_at_5_one_cell": result["fss_one_cell"]["5"],
                })

        neighborhood_results = {}
        neighborhood_rows = []
        for window in SCALES_KM:
            neighborhood_results[str(window)] = {}
            for name in ESTIMATE_NAMES:
                neighborhood_results[str(window)][name] = {}
                for label, threshold in THRESHOLDS:
                    result = finish_neighborhood_metrics(
                        neighborhood_accumulators[(window, name, label)]
                    )
                    neighborhood_results[str(window)][name][label] = result
                    neighborhood_rows.append({
                        "neighborhood_km": window,
                        "window_pixels": window,
                        "method": name,
                        "threshold_mm_h": threshold,
                        "valid_centers": result["valid_centers"],
                        "valid_hours": result["valid_hours"],
                        "neighborhood_csi_fractional": result["neighborhood_csi_fractional"],
                        "fss": result["fss"],
                        "fractional_hits": result["fractional_hits"],
                        "fractional_false_alarms": result["fractional_false_alarms"],
                        "fractional_misses": result["fractional_misses"],
                    })

        quantile_results = {
            name: finish_point_metrics(accumulator)
            for name, accumulator in quantile_accumulators.items()
        }
        quantile_rows = []
        for name, result in quantile_results.items():
            quantile_rows.append({
                "method": name,
                "support": "finite ERA5/RADKLIM pair support inside overlap",
                "n_pixel_hours": result["n_pixel_hours"],
                "valid_hours": result["valid_hours"],
                "mae_mm_h": result["mae_mm_h"],
                "rmse_mm_h": result["rmse_mm_h"],
                "bias_mm_h": result["bias_mm_h"],
                "csi_at_0.1": result["csi"]["0.1"]["score"],
                "csi_at_1": result["csi"]["1"]["score"],
                "csi_at_5": result["csi"]["5"]["score"],
            })

        intensity_results = {}
        intensity_rows = []
        for bin_key, bin_label in INTENSITY_BINS:
            intensity_results[bin_key] = {"label": bin_label, "methods": {}}
            for name in ESTIMATE_NAMES:
                result = finish_error_accumulator(intensity_accumulators[(bin_key, name)])
                intensity_results[bin_key]["methods"][name] = result
                intensity_rows.append({"target_bin": bin_key, "target_bin_label": bin_label, "method": name, **result})

        raw_vs_prediction_by_bin = {}
        for bin_key, _ in INTENSITY_BINS:
            raw_result = intensity_results[bin_key]["methods"]["ERA5 raw"]
            prediction_result = intensity_results[bin_key]["methods"]["ERA5 prediction"]
            raw_vs_prediction_by_bin[bin_key] = {
                "era5_prediction_minus_raw_mae_mm_h": safe_delta(
                    prediction_result["mae_mm_h"], raw_result["mae_mm_h"]
                ),
                "era5_prediction_minus_raw_rmse_mm_h": safe_delta(
                    prediction_result["rmse_mm_h"], raw_result["rmse_mm_h"]
                ),
                "raw_era5": raw_result,
                "era5_prediction": prediction_result,
            }

        scale_comparison = {}
        for name in RAW_NAMES:
            one_km = multiscale_results["1"][name]
            twenty_five_km = multiscale_results["25"][name]
            scale_comparison[name] = {
                "mae_25km_minus_1km_mm_h": safe_delta(
                    twenty_five_km["mae_mm_h"], one_km["mae_mm_h"]
                ),
                "rmse_25km_minus_1km_mm_h": safe_delta(
                    twenty_five_km["rmse_mm_h"], one_km["rmse_mm_h"]
                ),
                "csi_at_1_25km_minus_1km": safe_delta(
                    twenty_five_km["csi"]["1"]["score"], one_km["csi"]["1"]["score"]
                ),
            }

        spatial_recovery = {}
        for name in RAW_NAMES:
            spatial_recovery[name] = {}
            for label, _ in THRESHOLDS:
                one = neighborhood_results["1"][name][label]
                twenty_five = neighborhood_results["25"][name][label]
                spatial_recovery[name][label] = {
                    "fractional_csi_25km_minus_1km": safe_delta(
                        twenty_five["neighborhood_csi_fractional"],
                        one["neighborhood_csi_fractional"],
                    ),
                    "fss_25km_minus_1km": safe_delta(twenty_five["fss"], one["fss"]),
                }

        summary = {
            "experiment": "Gap Decomposition Experiment",
            "evaluation_split": args.split,
            "evaluation_hours": int(len(eval_times)),
            "evaluation_first_timestamp_utc": datetime.fromtimestamp(
                int(eval_times[0]), timezone.utc
            ).isoformat(),
            "evaluation_last_timestamp_utc": datetime.fromtimestamp(
                int(eval_times[-1]), timezone.utc
            ).isoformat(),
            "device": str(device),
            "elapsed_seconds": float(time.time() - started),
            "era5_definition": "aligned ERA5 precipitation, already bilinearly interpolated to the native RADKLIM-YW 1 km grid",
            "precipitation_units": "mm/h (hourly precipitation amount interpreted as hourly rate)",
            "model_checkpoints": {
                mode: {
                    "path": str(OUTPUT_DIR / "models" / mode / "best.pt"),
                    "checkpoint_epoch": int(checkpoints[mode].get("epoch", -1)),
                }
                for mode in ("era5", "synthetic")
            },
            "support": {
                "overlap_grid_pixels": support_totals["overlap_pixels"],
                "target_valid_pixel_hours": support_totals["target_valid_pixel_hours"],
                "raw_three_way_pixel_hours": support_totals["raw_three_way_pixel_hours"],
                "five_field_common_pixel_hours": support_totals["five_field_common_pixel_hours"],
                "multiscale_and_neighborhood": "same hourly support across RADKLIM truth, both raw inputs, and both predictions; all inside saved overlap mask",
                "quantile_mapping": "finite ERA5/RADKLIM pair support inside saved overlap mask; raw and calibrated ERA5 use identical pixels",
            },
            "definitions": {
                "multiscale": "spatial means over centered, non-overlapping N by N native grid blocks; N=1/5/10/25 cells for nominal 1/5/10/25 km; block cells require the configured common-support fraction; partial outer rows/columns are symmetrically cropped",
                "block_validity": "mean precipitation over available five-field common-support pixels within each block; default minimum support fraction 0.5",
                "multiscale_fss": "standard FSS computed at one coarse-grid cell per aggregated field and threshold",
                "spatial_tolerance": "moving square windows with the exact requested side lengths in native 1 km cells; a window is scored if its common support reaches the configured fraction",
                "neighborhood_csi": "fractional neighborhood CSI: H=sum(f*o), FA=sum(f*(1-o)), M=sum((1-f)*o), where f/o are forecast/observed event fractions in the same window; at 1 km this equals ordinary CSI",
                "fss": "FSS=1-sum((f-o)^2)/sum(f^2+o^2), using event fractions in the same valid neighborhood windows",
                "quantile_map": "global empirical one-dimensional ERA5-to-RADKLIM marginal map fitted on sampled train-only common-support pixel-hours; no spatial conditioning; applied only to ERA5 raw on evaluation hours",
                "intensity_bins": "RADKLIM target bins: <0.1, [0.1,1), [1,5), [5,10], >10 mm/h; error metrics use the five-field common support",
                "bias": "estimate minus RADKLIM target",
            },
            "configuration": {
                "minimum_support_fraction": float(args.min_support_fraction),
                "qm_samples_per_hour": int(args.qm_samples_per_hour),
                "seed": int(seed),
                "scales_km": list(SCALES_KM),
                "thresholds_mm_h": {label: value for label, value in THRESHOLDS},
            },
            "quantile_mapping_fit": qm_fit,
            "multiscale_metrics": multiscale_results,
            "multiscale_coverage": {
                str(scale): {
                    "valid_block_fraction": (
                        row["valid_cells"] / row["possible_cells"] if row["possible_cells"] else None
                    ),
                    "mean_common_support_fraction_in_accepted_blocks": (
                        row["accepted_block_support_sum"] / row["accepted_blocks"]
                        if row["accepted_blocks"] else None
                    ),
                }
                for scale, row in multiscale_support.items()
            },
            "spatial_tolerance_metrics": neighborhood_results,
            "raw_vs_quantile_mapped_era5": quantile_results,
            "intensity_stratified_errors": intensity_results,
            "derived_deltas": {
                "raw_input_25km_vs_1km": scale_comparison,
                "raw_input_spatial_recovery_25km_vs_1km": spatial_recovery,
                "era5_prediction_minus_raw_by_target_bin": raw_vs_prediction_by_bin,
                "era5_quantile_mapped_minus_raw": {
                    "mae_mm_h": safe_delta(
                        quantile_results["ERA5 quantile mapped"]["mae_mm_h"],
                        quantile_results["ERA5 raw"]["mae_mm_h"],
                    ),
                    "rmse_mm_h": safe_delta(
                        quantile_results["ERA5 quantile mapped"]["rmse_mm_h"],
                        quantile_results["ERA5 raw"]["rmse_mm_h"],
                    ),
                    "csi_at_0.1": safe_delta(
                        quantile_results["ERA5 quantile mapped"]["csi"]["0.1"]["score"],
                        quantile_results["ERA5 raw"]["csi"]["0.1"]["score"],
                    ),
                    "csi_at_1": safe_delta(
                        quantile_results["ERA5 quantile mapped"]["csi"]["1"]["score"],
                        quantile_results["ERA5 raw"]["csi"]["1"]["score"],
                    ),
                    "csi_at_5": safe_delta(
                        quantile_results["ERA5 quantile mapped"]["csi"]["5"]["score"],
                        quantile_results["ERA5 raw"]["csi"]["5"]["score"],
                    ),
                },
            },
            "interpretation_limit": "These diagnostics indicate which error signatures are consistent with scale, displacement, or marginal-distribution effects; they do not uniquely identify causal mechanisms.",
            "output_files": {
                "summary": str(output_dir / "summary.json"),
                "multiscale_csv": str(output_dir / "multiscale_metrics.csv"),
                "spatial_tolerance_csv": str(output_dir / "spatial_tolerance_metrics.csv"),
                "quantile_mapping_csv": str(output_dir / "quantile_mapping_metrics.csv"),
                "intensity_stratified_csv": str(output_dir / "intensity_stratified_metrics.csv"),
            },
        }

        write_csv(
            output_dir / "multiscale_metrics.csv",
            multiscale_rows,
            (
                "scale_km", "method", "n_pixel_hours", "valid_hours", "valid_cells_fraction",
                "mean_common_support_fraction_in_accepted_blocks", "mae_mm_h", "rmse_mm_h",
                "bias_mm_h", "csi_at_0.1", "csi_at_1", "csi_at_5",
                "fss_at_0.1_one_cell", "fss_at_1_one_cell", "fss_at_5_one_cell",
            ),
        )
        write_csv(
            output_dir / "spatial_tolerance_metrics.csv",
            neighborhood_rows,
            (
                "neighborhood_km", "window_pixels", "method", "threshold_mm_h", "valid_centers",
                "valid_hours", "neighborhood_csi_fractional", "fss", "fractional_hits",
                "fractional_false_alarms", "fractional_misses",
            ),
        )
        write_csv(
            output_dir / "quantile_mapping_metrics.csv",
            quantile_rows,
            (
                "method", "support", "n_pixel_hours", "valid_hours", "mae_mm_h", "rmse_mm_h",
                "bias_mm_h", "csi_at_0.1", "csi_at_1", "csi_at_5",
            ),
        )
        write_csv(
            output_dir / "intensity_stratified_metrics.csv",
            intensity_rows,
            (
                "target_bin", "target_bin_label", "method", "n_pixel_hours", "mae_mm_h",
                "rmse_mm_h", "bias_mm_h",
            ),
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            "[done] Gap decomposition outputs: {}".format(output_dir),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "summary": str(output_dir / "summary.json"),
                    "evaluation_hours": len(eval_times),
                    "raw_three_way_pixel_hours": support_totals["raw_three_way_pixel_hours"],
                    "five_field_common_pixel_hours": support_totals["five_field_common_pixel_hours"],
                    "ERA5 raw MAE": quantile_results["ERA5 raw"]["mae_mm_h"],
                    "ERA5 quantile-mapped MAE": quantile_results["ERA5 quantile mapped"]["mae_mm_h"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        reader.close()


if __name__ == "__main__":
    main()
