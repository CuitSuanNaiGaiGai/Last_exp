"""Create perfect-model inputs using RADKLIM -> ERA5 grid -> RADKLIM mapping."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Optional

import numpy as np
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as transform_coordinates

from common import ALIGNED_DIR, CONFIG, DATA_ROOT, DatasetCache, OUTPUT_DIR, read_field


def _transform(axis: np.ndarray, y_axis: Optional[np.ndarray] = None):
    if y_axis is None:
        step = float(np.median(np.diff(axis)))
        return step
    dx = float(np.median(np.diff(axis)))
    dy = float(np.median(np.diff(y_axis)))
    return dx, dy


def _geolocation_error(crs: CRS, x: np.ndarray, y: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Compare a projected CRS against the aligned file's authoritative 2-D lon/lat."""
    x_indices = np.unique(np.linspace(0, len(x) - 1, min(5, len(x)), dtype=int))
    y_indices = np.unique(np.linspace(0, len(y) - 1, min(5, len(y)), dtype=int))
    sample_x, sample_y = np.meshgrid(x[x_indices], y[y_indices])
    projected_lon, projected_lat = transform_coordinates(
        crs,
        "EPSG:4326",
        sample_x.ravel().tolist(),
        sample_y.ravel().tolist(),
    )
    expected_lat = lat[np.ix_(y_indices, x_indices)].ravel()
    expected_lon = lon[np.ix_(y_indices, x_indices)].ravel()
    projected_lat = np.asarray(projected_lat, dtype=np.float64)
    projected_lon = np.asarray(projected_lon, dtype=np.float64)
    valid = (
        np.isfinite(projected_lat)
        & np.isfinite(projected_lon)
        & np.isfinite(expected_lat)
        & np.isfinite(expected_lon)
    )
    if not valid.any():
        return float("inf"), float("inf")
    longitude_error = (projected_lon[valid] - expected_lon[valid] + 180.0) % 360.0 - 180.0
    latitude_error = projected_lat[valid] - expected_lat[valid]
    east_west_error = longitude_error * np.cos(np.deg2rad(expected_lat[valid]))
    error = np.hypot(latitude_error, east_west_error)
    return float(np.median(error)), float(np.max(error))


def _resolve_source_crs(crs_wkt: str, x: np.ndarray, y: np.ndarray, lat: np.ndarray, lon: np.ndarray, x_units: str, y_units: str):
    """Resolve malformed RADKLIM WKT unit scale using stored grid coordinates.

    The RADKLIM x/y variables declare metres, while this dataset's WKT says one
    ``metre`` is 1000 metres (kilometres). Test the original and unit-normalized
    CRS against the aligned file's 2-D geolocation before choosing either one.
    """
    candidates = [("stored WKT", CRS.from_wkt(crs_wkt))]
    units = {str(x_units).strip().lower(), str(y_units).strip().lower()}
    if units.issubset({"m", "metre", "meter", "metres", "meters"}):
        normalized_wkt = re.sub(
            r'LENGTHUNIT\["metre",\s*1000(?:\.0+)?\]',
            'LENGTHUNIT["metre",1]',
            crs_wkt,
        )
        if normalized_wkt != crs_wkt:
            candidates.append(("WKT length units normalized to metres", CRS.from_wkt(normalized_wkt)))

    scored = []
    for description, candidate in candidates:
        median_error, maximum_error = _geolocation_error(candidate, x, y, lat, lon)
        scored.append((median_error, maximum_error, description, candidate))
    median_error, maximum_error, description, candidate = min(scored, key=lambda row: row[0])
    if median_error > 0.02 or maximum_error > 0.05:
        details = "; ".join(
            "{}: median={:.6f}°, max={:.6f}°".format(name, median, maximum)
            for median, maximum, name, _ in scored
        )
        raise ValueError(
            "RADKLIM projected CRS does not match the aligned x/y and 2-D lat/lon grid ({}). "
            "Refusing to create an all-invalid synthetic LR file.".format(details)
        )
    if description != "stored WKT":
        print(
            "[crs] corrected RADKLIM WKT unit scale using x/y units={!r}/{!r}; "
            "geolocation error median={:.6f}°, max={:.6f}°".format(
                x_units, y_units, median_error, maximum_error
            ),
            flush=True,
        )
    return candidate, description, median_error, maximum_error


def create_month(month_path: Path, output_path: Path, minimum_coverage: float, overwrite: bool):
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path.name} exists; use --overwrite to rebuild", flush=True)
        return
    source_cache = DatasetCache(maximum_open=1)
    ds = source_cache.get(month_path)
    times = np.asarray(ds.variables["time"][:], dtype=np.int64)
    x = np.asarray(ds.variables["x"][:], dtype=np.float64)
    y = np.asarray(ds.variables["y"][:], dtype=np.float64)
    target = ds.variables["precipitation_radklim"]
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    x_units = getattr(ds.variables["x"], "units", "")
    y_units = getattr(ds.variables["y"], "units", "")
    with Dataset(DATA_ROOT / "metadata" / "grid" / "source_grids.nc") as ds:
        src_crs_wkt = ds.variables["radklim_crs"].crs_wkt
    src_crs, crs_resolution, crs_median_error, crs_max_error = _resolve_source_crs(
        src_crs_wkt, x, y, lat, lon, x_units, y_units
    )
    era_path = DATA_ROOT / "processed" / "era5" / f"era5_{datetime.fromtimestamp(int(times[0]), timezone.utc):%Y_%m}.nc"
    with Dataset(era_path) as ds:
        era_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        era_lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)

    dx, dy = _transform(x, y)
    dlat, dlon = _transform(era_lat), _transform(era_lon)
    src_transform = from_origin(float(x[0] - dx / 2), float(y[-1] + dy / 2), dx, dy)
    dst_transform = from_origin(float(era_lon[0] - dlon / 2), float(era_lat[-1] + dlat / 2), dlon, dlat)
    shape = (len(era_lat), len(era_lon))
    low = np.full((len(times), *shape), np.nan, dtype=np.float32)
    coverage = np.zeros((len(times), *shape), dtype=np.float32)
    destination = np.full(shape, np.nan, dtype=np.float32)
    coverage_north = np.zeros(shape, dtype=np.float32)
    indicator = np.empty((len(y), len(x)), dtype=np.float32)

    for index in range(len(times)):
        field = read_field(target, index)
        source_north = np.flipud(field)
        destination.fill(np.nan)
        coverage_north.fill(0.0)
        reproject(
            source=source_north,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=2,
        )
        indicator[:] = np.isfinite(source_north).astype(np.float32)
        reproject(
            source=indicator,
            destination=coverage_north,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=0.0,
            resampling=Resampling.average,
            num_threads=2,
        )
        # GDAL returns north-to-south rows; ERA5 latitude is stored south-to-north.
        fraction = coverage_north[::-1, :].copy()
        values = destination[::-1, :].copy()
        values[fraction < minimum_coverage] = np.nan
        low[index] = values
        coverage[index] = fraction
        if (index + 1) % 100 == 0 or index + 1 == len(times):
            print(f"[{output_path.stem}] synthetic {index + 1}/{len(times)} hours", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with Dataset(temporary, "w", format="NETCDF4") as out:
            out.createDimension("time", len(times))
            out.createDimension("lat", len(era_lat))
            out.createDimension("lon", len(era_lon))
            tv = out.createVariable("time", "i8", ("time",))
            tv[:] = times
            tv.units = "seconds since 1970-01-01 00:00:00"
            tv.calendar = "standard"
            la = out.createVariable("lat", "f8", ("lat",))
            la[:] = era_lat
            la.units = "degrees_north"
            lo = out.createVariable("lon", "f8", ("lon",))
            lo[:] = era_lon
            lo.units = "degrees_east"
            precip = out.createVariable("precipitation", "f4", ("time", "lat", "lon"), zlib=True, complevel=2, fill_value=np.float32(9.96921e36), chunksizes=(1, len(era_lat), len(era_lon)))
            precip[:] = np.ma.masked_invalid(low)
            precip.units = "mm"
            precip.long_name = "RADKLIM target remapped to the native ERA5 grid using GDAL average resampling"
            cov = out.createVariable("source_valid_fraction", "f4", ("time", "lat", "lon"), zlib=True, complevel=2, chunksizes=(1, len(era_lat), len(era_lon)))
            cov[:] = coverage
            cov.units = "1"
            cov.long_name = "fraction of contributing RADKLIM source footprint with valid precipitation"
            out.minimum_source_coverage = float(minimum_coverage)
            out.regridding = "RADKLIM projected grid to ERA5 geographic native grid; GDAL Resampling.average"
            out.source_crs = src_crs.to_wkt()
            out.source_crs_resolution = crs_resolution
            out.source_crs_geolocation_median_error_degrees = crs_median_error
            out.source_crs_geolocation_max_error_degrees = crs_max_error
            out.original_source_crs_wkt = src_crs_wkt
            out.valid_fraction_semantics = "valid RADKLIM fraction of the covered source footprint, not full ERA5-cell footprint"
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    valid = np.isfinite(low)
    print(f"[done] {output_path.name}: {len(times)} hours; coarse valid cells/hour={valid.sum(axis=(1, 2)).min()}–{valid.sum(axis=(1, 2)).max()}", flush=True)
    source_cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--month", help="Optional YYYY_MM selector, e.g. 2019_01")
    args = parser.parse_args()
    out_dir = OUTPUT_DIR / "synthetic_lr"
    months = sorted(ALIGNED_DIR.glob("aligned_*.nc"))
    if args.month:
        months = [p for p in months if p.stem[len("aligned_"):] == args.month]
    if not months:
        raise SystemExit(f"No aligned monthly files found under {ALIGNED_DIR}")
    for path in months:
        create_month(path, out_dir / path.name.replace("aligned_", "synthetic_lr_"), float(CONFIG["minimum_source_coverage"]), args.overwrite)


if __name__ == "__main__":
    main()
