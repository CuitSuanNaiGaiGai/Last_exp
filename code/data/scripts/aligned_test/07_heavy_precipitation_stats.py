#!/usr/bin/env python3
"""Count pixel-hour extremes, hourly maxima, and sampled training-patch extremes."""

from __future__ import print_function

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

from common import (
    ALIGNED_DIR, CONFIG, DatasetCache, OUTPUT_DIR, bilinear_valid_mask,
    load_bilinear_map, read_field,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_SIZE = int(CONFIG["model"]["patch_size"])
PATCH_STEP = 32
MAX_EPOCHS = int(CONFIG["model"]["epochs"])
BASE_SEED = int(CONFIG["seed"])
MINIMUM_PATCH_SUPPORT = float(CONFIG["model"]["minimum_patch_support"])
INTENSITY_BINS = (
    ("lt_0.1", "<0.1", lambda x: x < 0.1),
    ("0.1_to_lt_1", "0.1-<1", lambda x: (x >= 0.1) & (x < 1.0)),
    ("1_to_lt_5", "1-<5", lambda x: (x >= 1.0) & (x < 5.0)),
    ("5_to_lt_10", "5-<10", lambda x: (x >= 5.0) & (x < 10.0)),
    ("10_to_20_inclusive", "10-20 (inclusive)", lambda x: (x >= 10.0) & (x <= 20.0)),
    ("gt_20", ">20", lambda x: x > 20.0),
)
MAX_THRESHOLDS = (1.0, 5.0, 10.0, 20.0, 50.0)
PATCH_THRESHOLDS = (1.0, 5.0, 10.0)


def aligned_files():
    files = sorted(ALIGNED_DIR.glob("aligned_*.nc"))
    if len(files) != 24:
        raise RuntimeError("Expected 24 aligned monthly files under {}; found {}".format(ALIGNED_DIR, len(files)))
    return files


def to_utc(timestamp):
    return datetime.fromtimestamp(int(timestamp), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_source_counts():
    return {
        "pixel_hours_by_bin": {name: 0 for name, _, _ in INTENSITY_BINS},
        "negative_pixel_hours": 0,
        "overall_max_mm_h": None,
        "hours_max_gt_threshold": {str(int(value)): 0 for value in MAX_THRESHOLDS},
    }


def calculate_pixel_hour_and_hourly_maxima():
    """Use identical finite ERA5/RADKLIM pixels for both source summaries."""
    sources = {"ERA5_aligned": empty_source_counts(), "RADKLIM_YW": empty_source_counts()}
    pixel_hour_denominator = 0
    valid_hours = 0
    hourly_rows = []
    total_hours = 0
    previous_timestamp = None

    for path in aligned_files():
        with Dataset(path) as dataset:
            times = np.asarray(dataset.variables["time"][:], dtype=np.int64)
            era_variable = dataset.variables["precipitation_era5"]
            rad_variable = dataset.variables["precipitation_radklim"]
            if era_variable.shape != rad_variable.shape or era_variable.shape[0] != len(times):
                raise ValueError("Aligned time or field dimensions mismatch in {}".format(path))

            block_count = (len(times) + 11) // 12
            for block_number, block_start in enumerate(tqdm(
                range(0, len(times), 12), desc=path.stem, unit="block", leave=False
            ), 1):
                block_end = min(block_start + 12, len(times))
                era_masked = np.ma.asarray(era_variable[block_start:block_end], dtype=np.float32)
                rad_masked = np.ma.asarray(rad_variable[block_start:block_end], dtype=np.float32)
                era = np.asarray(np.ma.filled(era_masked, np.nan), dtype=np.float32)
                rad = np.asarray(np.ma.filled(rad_masked, np.nan), dtype=np.float32)
                common = np.isfinite(era) & np.isfinite(rad)
                counts = common.sum(axis=(1, 2), dtype=np.int64)
                pixel_hour_denominator += int(counts.sum())

                era_hour_max = np.max(np.where(common, era, -np.inf), axis=(1, 2))
                rad_hour_max = np.max(np.where(common, rad, -np.inf), axis=(1, 2))
                era_hour_max[counts == 0] = np.nan
                rad_hour_max[counts == 0] = np.nan

                for source_name, values, maxima in (
                    ("ERA5_aligned", era, era_hour_max),
                    ("RADKLIM_YW", rad, rad_hour_max),
                ):
                    stats = sources[source_name]
                    selected = values[common]
                    stats["negative_pixel_hours"] += int(np.count_nonzero(selected < 0.0))
                    for name, _, condition in INTENSITY_BINS:
                        stats["pixel_hours_by_bin"][name] += int(np.count_nonzero(condition(selected)))
                    finite_maxima = maxima[np.isfinite(maxima)]
                    if finite_maxima.size:
                        old_max = stats["overall_max_mm_h"]
                        batch_max = float(np.max(finite_maxima))
                        stats["overall_max_mm_h"] = batch_max if old_max is None else max(old_max, batch_max)
                    for threshold in MAX_THRESHOLDS:
                        stats["hours_max_gt_threshold"][str(int(threshold))] += int(
                            np.count_nonzero(finite_maxima > threshold)
                        )

                for offset, timestamp in enumerate(times[block_start:block_end]):
                    timestamp = int(timestamp)
                    if previous_timestamp is not None and timestamp <= previous_timestamp:
                        raise ValueError("Aligned timestamps are duplicated or out of order near {}".format(path))
                    previous_timestamp = timestamp
                    total_hours += 1
                    count = int(counts[offset])
                    if count:
                        valid_hours += 1
                    hourly_rows.append({
                        "time_utc": to_utc(timestamp),
                        "common_valid_pixels": count,
                        "ERA5_max_mm_h": float(era_hour_max[offset]) if count else "",
                        "RADKLIM_YW_max_mm_h": float(rad_hour_max[offset]) if count else "",
                    })

                if block_number == 1 or block_number % 10 == 0 or block_number == block_count:
                    tqdm.write("[{}] blocks {}/{}".format(path.stem, block_number, block_count))

    for stats in sources.values():
        stats["pixel_hour_fraction_by_bin"] = {
            name: (count / float(pixel_hour_denominator) if pixel_hour_denominator else None)
            for name, count in stats["pixel_hours_by_bin"].items()
        }
        stats["hours_with_valid_pixels"] = valid_hours

    return {
        "hours": total_hours,
        "hours_with_common_valid_pixels": valid_hours,
        "common_valid_pixel_hours": pixel_hour_denominator,
        "sources": sources,
        "hourly_rows": hourly_rows,
    }


def patch_grid(shape):
    height, width = shape
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError("The RADKLIM grid is smaller than one configured patch")
    rows = np.unique(np.append(np.arange(0, height - PATCH_SIZE + 1, PATCH_STEP), height - PATCH_SIZE)).astype(np.int32)
    cols = np.unique(np.append(np.arange(0, width - PATCH_SIZE + 1, PATCH_STEP), width - PATCH_SIZE)).astype(np.int32)
    return rows, cols


def eligible_patch_indices(support, rows, cols):
    integral = np.pad(support.astype(np.int32), ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64)
    row_end = rows + PATCH_SIZE
    col_end = cols + PATCH_SIZE
    counts = (
        integral[row_end[:, None], col_end[None, :]]
        - integral[rows[:, None], col_end[None, :]]
        - integral[row_end[:, None], cols[None, :]]
        + integral[rows[:, None], cols[None, :]]
    )
    minimum_pixels = int(math.ceil(MINIMUM_PATCH_SUPPORT * PATCH_SIZE * PATCH_SIZE))
    return np.argwhere(counts >= minimum_pixels)


class TrainingFields:
    """Read the same RADKLIM/ERA5/synthetic support used by 04_train_compare.py."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.overlap = mapping["overlap_mask"].astype(bool)
        self.aligned_cache = DatasetCache(maximum_open=3)
        self.synthetic_cache = DatasetCache(maximum_open=3)
        self.index = {}
        for path in aligned_files():
            with Dataset(path) as dataset:
                for index, timestamp in enumerate(np.asarray(dataset.variables["time"][:], dtype=np.int64)):
                    self.index[int(timestamp)] = (path, index)

    def get(self, timestamp):
        path, index = self.index[int(timestamp)]
        aligned = self.aligned_cache.get(path)
        target = read_field(aligned.variables["precipitation_radklim"], index)
        era = read_field(aligned.variables["precipitation_era5"], index)
        era[~self.overlap] = np.nan
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / path.name.replace("aligned_", "synthetic_lr_")
        synthetic = self.synthetic_cache.get(synthetic_path)
        coarse = read_field(synthetic.variables["precipitation"], index)
        synthetic_available = bilinear_valid_mask(coarse, self.mapping)
        support = np.isfinite(target) & np.isfinite(era) & synthetic_available & self.overlap
        return target, support

    def close(self):
        self.aligned_cache.close()
        self.synthetic_cache.close()


def choose_patch_for_epoch(eligible, rows, cols, timestamp, epoch):
    if not len(eligible):
        return None
    seed = BASE_SEED + epoch * 100_000 + int(timestamp)
    rng = np.random.default_rng(seed)
    choice = int(rng.integers(len(eligible)))
    row_index, col_index = eligible[choice]
    return int(rows[row_index]), int(cols[col_index])


def count_training_patch_extremes():
    timestamp_path = OUTPUT_DIR / "splits" / "train_timestamps.txt"
    if not timestamp_path.is_file():
        raise FileNotFoundError("Run 03_make_splits.py first; missing {}".format(timestamp_path))
    train_times = [int(value) for value in timestamp_path.read_text().splitlines() if value.strip()]
    mapping = load_bilinear_map()
    rows, cols = patch_grid(mapping["overlap_mask"].shape)
    totals_by_epoch = [
        {"eligible_patches": 0, "max_gt_1": 0, "max_gt_5": 0, "max_gt_10": 0}
        for _ in range(MAX_EPOCHS)
    ]
    skipped_hours = 0
    fields = TrainingFields(mapping)
    try:
        progress = tqdm(train_times, desc="sampled training patches", unit="hour", dynamic_ncols=True)
        for position, timestamp in enumerate(progress, 1):
            target, support = fields.get(timestamp)
            eligible = eligible_patch_indices(support, rows, cols)
            if not len(eligible):
                skipped_hours += 1
                continue

            for epoch in range(MAX_EPOCHS):
                coordinate = choose_patch_for_epoch(eligible, rows, cols, timestamp, epoch)
                if coordinate is None:
                    continue
                row, col = coordinate
                patch_target = target[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
                patch_support = support[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
                valid_target = patch_target[patch_support]
                if not valid_target.size:
                    continue
                patch_max = float(np.max(valid_target))
                counts = totals_by_epoch[epoch]
                counts["eligible_patches"] += 1
                for threshold in PATCH_THRESHOLDS:
                    if patch_max > threshold:
                        counts["max_gt_{}".format(int(threshold))] += 1

            progress.set_postfix(skipped_hours=skipped_hours, refresh=False)
    finally:
        fields.close()

    cumulative = {"eligible_patch_exposures": 0, "max_gt_1": 0, "max_gt_5": 0, "max_gt_10": 0}
    for counts in totals_by_epoch:
        cumulative["eligible_patch_exposures"] += counts["eligible_patches"]
        for name in ("max_gt_1", "max_gt_5", "max_gt_10"):
            cumulative[name] += counts[name]

    return {
        "target": "RADKLIM-YW precipitation on the exact shared training support used by 04_train_compare.py",
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
        "patch_step_for_candidates": PATCH_STEP,
        "minimum_patch_support_fraction": MINIMUM_PATCH_SUPPORT,
        "sampling": "one deterministic eligible patch per train timestamp per epoch, using the 04_train_compare.py seed schedule",
        "train_timestamps": len(train_times),
        "hours_without_an_eligible_patch": skipped_hours,
        "max_epochs_in_config": MAX_EPOCHS,
        "cumulative_counts_are_patch_exposures_not_unique_geographic_patches": True,
        "per_epoch": [dict(epoch=index + 1, **counts) for index, counts in enumerate(totals_by_epoch)],
        "planned_cumulative_up_to_max_epochs": cumulative,
    }


def main():
    pixel_stats = calculate_pixel_hour_and_hourly_maxima()
    patch_stats = count_training_patch_extremes()
    output_dir = OUTPUT_DIR / "heavy_precipitation_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_csv = output_dir / "hourly_maxima_2019_2020.csv"
    with hourly_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "time_utc", "common_valid_pixels", "ERA5_max_mm_h", "RADKLIM_YW_max_mm_h",
        ))
        writer.writeheader()
        writer.writerows(pixel_stats.pop("hourly_rows"))

    patch_csv = output_dir / "training_patch_counts_by_epoch.csv"
    with patch_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "eligible_patches", "max_gt_1", "max_gt_5", "max_gt_10"))
        writer.writeheader()
        writer.writerows(patch_stats["per_epoch"])

    summary = {
        "period_utc": "2019-01-01T01:00:00Z through 2020-12-31T23:00:00Z, as present in aligned files",
        "units": "aligned hourly precipitation amounts in mm per one-hour interval, reported as mm/h",
        "pixel_hour_and_hourly_max_support": "finite ERA5_aligned and RADKLIM_YW values at the same pixel and hour",
        "intensity_bins": [
            {"name": "<0.1", "interval_mm_h": "value < 0.1"},
            {"name": "0.1-<1", "interval_mm_h": "0.1 <= value < 1"},
            {"name": "1-<5", "interval_mm_h": "1 <= value < 5"},
            {"name": "5-<10", "interval_mm_h": "5 <= value < 10"},
            {"name": "10-20 inclusive", "interval_mm_h": "10 <= value <= 20"},
            {"name": ">20", "interval_mm_h": "value > 20"},
        ],
        "pixel_hour_distribution": {
            "common_valid_pixel_hours": pixel_stats["common_valid_pixel_hours"],
            "sources": pixel_stats["sources"],
        },
        "hourly_maxima": {
            "hours": pixel_stats["hours"],
            "hours_with_common_valid_pixels": pixel_stats["hours_with_common_valid_pixels"],
            "threshold_comparison": "strictly greater than each threshold",
            "sources": {
                name: {
                    "overall_max_mm_h": stats["overall_max_mm_h"],
                    "hours_max_gt_threshold": stats["hours_max_gt_threshold"],
                }
                for name, stats in pixel_stats["sources"].items()
            },
        },
        "training_patch_extremes": patch_stats,
        "outputs": {
            "hourly_maxima_csv": str(hourly_csv.resolve()),
            "training_patch_counts_by_epoch_csv": str(patch_csv.resolve()),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("[done] summary: {}".format(summary_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
