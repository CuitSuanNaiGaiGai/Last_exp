"""Audit aligned precipitation files and reproducibly sample 40 hourly maps."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

_experiment_root = Path(__file__).resolve().parent
_cache_dir = _experiment_root / "outputs" / "matplotlib_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_experiment_root / "outputs" / "cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from common import ALIGNED_DIR, CONFIG, OUTPUT_DIR, TIME_UNITS, load_bilinear_map, read_field


def expected_timestamps() -> np.ndarray:
    start = int(datetime(2019, 1, 1, 1, tzinfo=timezone.utc).timestamp())
    stop = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
    return np.arange(start, stop, 3600, dtype=np.int64)


def value_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "finite_count": int(finite.size),
        "nan_fraction": float(np.isnan(values).mean()),
        "negative_count": int(np.count_nonzero(finite < -1e-7)),
        "min_mm": float(np.min(finite)) if finite.size else None,
        "median_mm": float(np.median(finite)) if finite.size else None,
        "p95_mm": float(np.quantile(finite, 0.95)) if finite.size else None,
        "p99_mm": float(np.quantile(finite, 0.99)) if finite.size else None,
        "max_mm": float(np.max(finite)) if finite.size else None,
    }


def main() -> None:
    output = OUTPUT_DIR / "qc"
    output.mkdir(parents=True, exist_ok=True)
    mapping = load_bilinear_map()
    overlap = mapping["overlap_mask"].astype(bool)
    expected = expected_timestamps()
    files = sorted(ALIGNED_DIR.glob("aligned_*.nc"))
    if len(files) != 24:
        raise RuntimeError(f"Expected 24 aligned monthly files, found {len(files)}")

    months = []
    all_times = []
    reference = None
    for path in files:
        with Dataset(path) as ds:
            for name in ("time", "x", "y", "lat", "lon", "crs", "precipitation_era5", "precipitation_radklim"):
                if name not in ds.variables:
                    raise ValueError(f"{path.name} is missing variable {name!r}")
            times = np.asarray(ds.variables["time"][:], dtype=np.int64)
            if len(times) == 0 or (len(times) > 1 and not np.all(np.diff(times) == 3600)):
                raise ValueError(f"Non-hourly or empty time axis in {path.name}")
            if getattr(ds.variables["time"], "units", None) != TIME_UNITS:
                raise ValueError(f"Unexpected time units in {path.name}")
            for variable_name in ("precipitation_era5", "precipitation_radklim"):
                variable = ds.variables[variable_name]
                if variable.dimensions != ("time", "y", "x"):
                    raise ValueError(f"Unexpected dimensions for {variable_name} in {path.name}: {variable.dimensions}")
                if variable.units != "mm":
                    raise ValueError(f"Unexpected precipitation unit {variable.units!r} in {path.name}")
            if ds.variables["precipitation_era5"].shape != ds.variables["precipitation_radklim"].shape:
                raise ValueError(f"ERA5/RADKLIM shapes differ in {path.name}")
            coords = {name: np.asarray(ds.variables[name][:]) for name in ("x", "y", "lat", "lon")}
            if coords["lat"].shape != overlap.shape or coords["lon"].shape != overlap.shape:
                raise ValueError(f"Coordinate shape differs from saved overlap map in {path.name}")
            if reference is None:
                reference = coords
                if not np.all(np.diff(coords["x"]) > 0) or not np.all(np.diff(coords["y"]) > 0):
                    raise ValueError("Projected x/y coordinates must both increase")
                if float(np.nanmedian(np.diff(coords["lat"], axis=0))) <= 0:
                    raise ValueError("RADKLIM rows do not increase northward")
                if float(np.nanmedian(np.diff(coords["lon"], axis=1))) <= 0:
                    raise ValueError("RADKLIM columns do not increase eastward")
                south, north, west, east = map(float, mapping["bbox"])
                if not np.all((coords["lat"][overlap] >= south) & (coords["lat"][overlap] <= north)):
                    raise ValueError("Overlap mask contains a latitude outside the requested bbox")
                if not np.all((coords["lon"][overlap] >= west) & (coords["lon"][overlap] <= east)):
                    raise ValueError("Overlap mask contains a longitude outside the requested bbox")
            else:
                for name in reference:
                    if not np.allclose(reference[name], coords[name], rtol=0, atol=1e-8, equal_nan=True):
                        raise ValueError(f"Coordinate {name} changes between aligned months")
            all_times.extend(times.tolist())
            months.append({"file": path.name, "count": int(len(times)), "first_epoch": int(times[0]), "last_epoch": int(times[-1])})

    all_times = np.asarray(all_times, dtype=np.int64)
    if len(np.unique(all_times)) != len(all_times):
        raise ValueError("Aligned timestamps contain duplicates")
    if not np.array_equal(all_times, expected):
        raise ValueError(f"Aligned timeline differs from expected hourly coverage ({len(all_times)} vs {len(expected)})")

    rng = np.random.default_rng(int(CONFIG["seed"]))
    picks: list[tuple[Path, int]] = []
    by_month = []
    for path in files:
        with Dataset(path) as ds:
            n = len(ds.dimensions["time"])
        idx = int(rng.integers(0, n))
        picks.append((path, idx))
        by_month.extend((path, i) for i in range(n) if (path, i) not in picks)
    remaining = int(CONFIG["qc_sample_hours"]) - len(picks)
    if remaining < 0:
        raise ValueError("QC sample count must be at least the number of months")
    candidates = [(path, idx) for path, idx in by_month]
    extra_indices = rng.choice(len(candidates), size=remaining, replace=False)
    picks.extend(candidates[int(i)] for i in extra_indices)
    if len({(str(p), i) for p, i in picks}) != len(picks):
        raise AssertionError("QC sampling unexpectedly selected a duplicate hour")

    rows = []
    map_candidate = None
    map_score = -1.0
    for path, index in picks:
        with Dataset(path) as ds:
            epoch = int(ds.variables["time"][index])
            fields = {
                "era5": read_field(ds.variables["precipitation_era5"], index),
                "radklim": read_field(ds.variables["precipitation_radklim"], index),
            }
        row = {"file": path.name, "index": index, "time_utc": datetime.fromtimestamp(epoch, timezone.utc).isoformat()}
        score = 0.0
        for name, field in fields.items():
            row[f"{name}_inside_overlap"] = value_stats(field[overlap])
            row[f"{name}_outside_overlap"] = value_stats(field[~overlap])
            if row[f"{name}_inside_overlap"]["negative_count"]:
                raise ValueError(f"Negative precipitation values found in {path.name} at index {index} ({name})")
            if name == "radklim":
                score = float(np.nansum(field[overlap]))
        rows.append(row)
        if score > map_score:
            map_candidate = (epoch, fields)
            map_score = score

    qc_summary = {
        "sample_seed": int(CONFIG["seed"]),
        "sample_hours": len(rows),
        "all_months_represented": len({r["file"] for r in rows}) == 24,
        "aligned_hours": int(len(all_times)),
        "expected_hours": int(len(expected)),
        "time_start_utc": datetime.fromtimestamp(int(all_times[0]), timezone.utc).isoformat(),
        "time_end_utc": datetime.fromtimestamp(int(all_times[-1]), timezone.utc).isoformat(),
        "excluded_time_utc": "2019-01-01T00:00:00+00:00",
        "monthly_counts": months,
        "grid": {
            "shape_y_x": list(overlap.shape),
            "valid_overlap_cells": int(overlap.sum()),
            "outside_overlap_cells": int((~overlap).sum()),
            "x_increasing_eastward": bool(np.all(np.diff(reference["x"]) > 0)),
            "y_increasing_northward": bool(np.all(np.diff(reference["y"]) > 0)),
            "lat_increasing_northward_by_row": bool(np.nanmedian(np.diff(reference["lat"], axis=0)) > 0),
            "lon_increasing_eastward_by_column": bool(np.nanmedian(np.diff(reference["lon"], axis=1)) > 0),
            "valid_lat_range": [float(np.min(reference["lat"][overlap])), float(np.max(reference["lat"][overlap]))],
            "valid_lon_range": [float(np.min(reference["lon"][overlap])), float(np.max(reference["lon"][overlap]))],
        },
        "variables": {"era5": {"units": "mm"}, "radklim": {"units": "mm"}},
        "sample_aggregate_ranges_inside_overlap": {
            name: {
                "nan_fraction_min": float(min(r[f"{name}_inside_overlap"]["nan_fraction"] for r in rows)),
                "nan_fraction_max": float(max(r[f"{name}_inside_overlap"]["nan_fraction"] for r in rows)),
                "sampled_max_mm_range": [float(min(r[f"{name}_inside_overlap"]["max_mm"] for r in rows if r[f"{name}_inside_overlap"]["max_mm"] is not None)),
                                          float(max(r[f"{name}_inside_overlap"]["max_mm"] for r in rows if r[f"{name}_inside_overlap"]["max_mm"] is not None))],
            }
            for name in ("era5", "radklim")
        },
        "map_preview_timestamp_utc": datetime.fromtimestamp(map_candidate[0], timezone.utc).isoformat(),
        "map_preview_radklim_total_mm": map_score,
        "status": "passed",
    }
    (output / "aligned_qc.json").write_text(json.dumps(qc_summary, indent=2, ensure_ascii=False) + "\n")
    with (output / "sampled_hours.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "index", "time_utc", "era5_inside_overlap", "era5_outside_overlap", "radklim_inside_overlap", "radklim_outside_overlap"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if key.endswith(("overlap", "overlap")) and isinstance(row[key], dict) else row[key] for key in writer.fieldnames})

    epoch, fields = map_candidate
    lat, lon = reference["lat"], reference["lon"]
    finite_values = np.concatenate([field[overlap & np.isfinite(field)] for field in fields.values()])
    vmax = max(float(np.quantile(finite_values, 0.995)), 0.1)
    levels = np.linspace(0.0, vmax, 17)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, name, title in zip(axes, ("era5", "radklim"), ("ERA5 interpolated (mm/hour)", "RADKLIM-YW target (mm/hour)")):
        image = ax.contourf(lon, lat, np.ma.masked_invalid(fields[name]), levels=levels, cmap="viridis", extend="max")
        ax.set(xlim=(2, 16), ylim=(46.5, 55), xlabel="Longitude (°E)", ylabel="Latitude (°N)", title=title)
        ax.grid(alpha=0.2)
        figure.colorbar(image, ax=ax, label="mm/hour")
    figure.suptitle(f"Aligned precipitation map check — {datetime.fromtimestamp(epoch, timezone.utc):%Y-%m-%d %H:%M UTC}")
    figure.savefig(output / "map_preview.png", dpi=150)
    plt.close(figure)
    print(f"QC passed: {len(files)} months, {len(all_times)} hours, {len(rows)} sampled hours")
    print(f"Grid {overlap.shape}; overlap pixels={int(overlap.sum())}")
    print(f"Report: {output / 'aligned_qc.json'}")
    print(f"Map: {output / 'map_preview.png'}")


if __name__ == "__main__":
    main()
