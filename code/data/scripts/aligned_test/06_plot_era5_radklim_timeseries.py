#!/usr/bin/env python3
"""Plot daily area-mean precipitation for aligned ERA5 and RADKLIM-YW."""

from __future__ import print_function

import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[4]
ALIGNED_DIR = PROJECT_ROOT / "code" / "data" / "processed" / "aligned"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "timeseries"
START_DATE = date(2019, 1, 1)
END_DATE = date(2020, 12, 31)
ROLLING_DAYS = 30
BATCH_HOURS = 12


def read_float_block(variable, first, last):
    values = np.ma.asarray(variable[first:last, :, :], dtype=np.float32)
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)


def rolling_mean(values, window):
    """Centered rolling mean; keep only complete calendar-day windows."""
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < window:
        return output
    left_width = window // 2
    for index in range(left_width, len(values) - (window - left_width) + 1):
        block = values[index - left_width:index - left_width + window]
        finite = np.isfinite(block)
        if finite.all():
            output[index] = float(np.mean(block, dtype=np.float64))
    return output


def discover_files():
    files = []
    for path in sorted(ALIGNED_DIR.glob("aligned_*.nc")):
        if path.name.startswith("aligned_2019_") or path.name.startswith("aligned_2020_"):
            files.append(path)
    if len(files) != 24:
        raise RuntimeError("Expected 24 aligned monthly files for 2019-2020 under {}; found {}".format(
            ALIGNED_DIR, len(files)
        ))
    return files


def collect_daily_means():
    """Compute each hour's spatial mean on the shared finite grid, then daily means."""
    daily_values = {}
    total_hours = 0
    first_timestamp = None
    last_timestamp = None
    previous_timestamp = None

    for path in discover_files():
        with Dataset(path) as dataset:
            required = ("time", "precipitation_era5", "precipitation_radklim")
            missing = [name for name in required if name not in dataset.variables]
            if missing:
                raise KeyError("{} is missing variables: {}".format(path, ", ".join(missing)))

            time_values = np.asarray(dataset.variables["time"][:], dtype=np.int64)
            era_variable = dataset.variables["precipitation_era5"]
            radklim_variable = dataset.variables["precipitation_radklim"]
            if era_variable.shape != radklim_variable.shape:
                raise ValueError("ERA5 and RADKLIM shapes differ in {}".format(path))
            if era_variable.shape[0] != len(time_values):
                raise ValueError("Time axis and precipitation dimensions differ in {}".format(path))

            block_count = (len(time_values) + BATCH_HOURS - 1) // BATCH_HOURS
            for block_number, block_start in enumerate(range(0, len(time_values), BATCH_HOURS), 1):
                block_end = min(block_start + BATCH_HOURS, len(time_values))
                era = read_float_block(era_variable, block_start, block_end)
                radklim = read_float_block(radklim_variable, block_start, block_end)
                common = np.isfinite(era) & np.isfinite(radklim)
                pixel_counts = common.sum(axis=(1, 2), dtype=np.int64)
                era_sums = np.where(common, era, np.float32(0.0)).sum(axis=(1, 2), dtype=np.float64)
                radklim_sums = np.where(common, radklim, np.float32(0.0)).sum(axis=(1, 2), dtype=np.float64)

                for offset, timestamp in enumerate(time_values[block_start:block_end]):
                    timestamp = int(timestamp)
                    if previous_timestamp is not None and timestamp <= previous_timestamp:
                        raise ValueError("Timestamps are duplicated or out of order at {}".format(path))
                    previous_timestamp = timestamp
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp
                    total_hours += 1

                    hour = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    day = hour.date()
                    if day < START_DATE or day > END_DATE:
                        continue
                    count = int(pixel_counts[offset])
                    if day not in daily_values:
                        daily_values[day] = {"era5": [], "radklim": [], "valid_pixels": [], "time_steps": 0}
                    entry = daily_values[day]
                    entry["time_steps"] += 1
                    if count:
                        entry["era5"].append(float(era_sums[offset] / count))
                        entry["radklim"].append(float(radklim_sums[offset] / count))
                        entry["valid_pixels"].append(count)
                if block_number == 1 or block_number % 10 == 0 or block_number == block_count:
                    print("[{}] blocks {}/{} ({}/{} hours)".format(
                        path.stem, block_number, block_count, block_end, len(time_values)
                    ), flush=True)

    days = []
    current = START_DATE
    while current <= END_DATE:
        days.append(current)
        current += timedelta(days=1)

    era_daily = np.full(len(days), np.nan, dtype=np.float64)
    radklim_daily = np.full(len(days), np.nan, dtype=np.float64)
    valid_hours = np.zeros(len(days), dtype=np.int32)
    mean_valid_pixels = np.full(len(days), np.nan, dtype=np.float64)
    for index, day in enumerate(days):
        entry = daily_values.get(day)
        if not entry:
            continue
        valid_hours[index] = len(entry["era5"])
        if valid_hours[index]:
            era_daily[index] = float(np.mean(entry["era5"], dtype=np.float64))
            radklim_daily[index] = float(np.mean(entry["radklim"], dtype=np.float64))
            mean_valid_pixels[index] = float(np.mean(entry["valid_pixels"], dtype=np.float64))

    return {
        "days": days,
        "era5_daily": era_daily,
        "radklim_daily": radklim_daily,
        "valid_hours": valid_hours,
        "mean_valid_pixels": mean_valid_pixels,
        "total_hours": total_hours,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def main():
    series = collect_daily_means()
    era_smooth = rolling_mean(series["era5_daily"], ROLLING_DAYS)
    radklim_smooth = rolling_mean(series["radklim_daily"], ROLLING_DAYS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "era5_radklim_daily_timeseries_2019_2020.csv"
    figure_path = OUTPUT_DIR / "era5_radklim_daily_timeseries_2019_2020.png"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "date_utc", "era5_daily_area_mean_mm_per_hour", "radklim_daily_area_mean_mm_per_hour",
            "valid_hours", "mean_common_pixels_per_valid_hour",
            "era5_centered_30day_mean_mm_per_hour", "radklim_centered_30day_mean_mm_per_hour",
        ))
        for index, day in enumerate(series["days"]):
            writer.writerow((
                day.isoformat(),
                series["era5_daily"][index],
                series["radklim_daily"][index],
                int(series["valid_hours"][index]),
                series["mean_valid_pixels"][index],
                era_smooth[index],
                radklim_smooth[index],
            ))

    dates = [datetime.combine(day, datetime.min.time()) for day in series["days"]]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(dates, series["era5_daily"], color="#2878B5", linewidth=0.75, alpha=0.28, label="ERA5 daily")
    ax.plot(dates, series["radklim_daily"], color="#D9534F", linewidth=0.75, alpha=0.28, label="RADKLIM-YW daily")
    ax.plot(dates, era_smooth, color="#2878B5", linewidth=2.0, label="ERA5 30-day mean")
    ax.plot(dates, radklim_smooth, color="#D9534F", linewidth=2.0, label="RADKLIM-YW 30-day mean")
    ax.set_title("Daily Area-Mean Hourly Precipitation: ERA5 vs RADKLIM-YW (2019-2020)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Area-mean hourly precipitation (mm/h)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(datetime(2019, 1, 1), datetime(2021, 1, 1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.text(
        0.01, 0.01,
        "Each hour uses the same finite ERA5/RADKLIM cells; daily values average hourly spatial means. "
        "Thick lines are centered 30-day means. First available hour: 2019-01-01 01:00 UTC.",
        ha="left", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(str(figure_path), dpi=180)
    plt.close(fig)

    print("[done] aligned hourly records: {}".format(series["total_hours"]))
    print("[done] date range: {} to {} UTC".format(series["days"][0], series["days"][-1]))
    print("[done] days with valid common pixels: {}".format(int(np.count_nonzero(series["valid_hours"]))))
    print("[done] daily CSV: {}".format(csv_path.resolve()))
    print("[done] figure: {}".format(figure_path.resolve()))


if __name__ == "__main__":
    main()
