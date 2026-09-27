"""Aggregate exact RADKLIM-YW hourly windows and align ERA5 to the YW grid."""

import argparse
import calendar
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
BBOX = (46.5, 55.0, 2.0, 16.0)  # south, north, west, east


def json_scalar(value):
    """Convert NumPy scalar metadata to native JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def mapping(lat: np.ndarray, lon: np.ndarray, era_lat: np.ndarray, era_lon: np.ndarray):
    south, north, west, east = BBOX
    inside = np.isfinite(lat) & np.isfinite(lon) & (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    if not inside.any():
        raise ValueError("No RADKLIM grid cells lie inside the requested geographic area")
    rows, cols = np.where(inside)
    ys = slice(rows.min(), rows.max() + 1)
    xs = slice(cols.min(), cols.max() + 1)
    target_lat, target_lon = lat[ys, xs], lon[ys, xs]
    mask = inside[ys, xs]
    if np.any(np.diff(era_lat) <= 0) or np.any(np.diff(era_lon) <= 0):
        raise ValueError("Processed ERA5 latitude/longitude coordinates must be strictly ascending")
    i0 = np.searchsorted(era_lat, target_lat, side="right") - 1
    j0 = np.searchsorted(era_lon, target_lon, side="right") - 1
    i0 = np.clip(i0, 0, len(era_lat) - 2)
    j0 = np.clip(j0, 0, len(era_lon) - 2)
    fy = (target_lat - era_lat[i0]) / (era_lat[i0 + 1] - era_lat[i0])
    fx = (target_lon - era_lon[j0]) / (era_lon[j0 + 1] - era_lon[j0])
    valid = mask & (target_lat >= era_lat[0]) & (target_lat <= era_lat[-1]) & (target_lon >= era_lon[0]) & (target_lon <= era_lon[-1])
    weights = ((1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx)
    return ys, xs, target_lat, target_lon, valid, i0, j0, weights


def interpolate(field, valid, i0, j0, weights):
    a = field[i0, j0]
    b = field[i0, j0 + 1]
    c = field[i0 + 1, j0]
    d = field[i0 + 1, j0 + 1]
    result = a * weights[0] + b * weights[1] + c * weights[2] + d * weights[3]
    result = np.asarray(result, dtype=np.float32)
    result[~valid | ~np.isfinite(result)] = np.nan
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data = args.root.resolve() / "code" / "data"
    era_dir, yw_dir = data / "processed" / "era5", data / "processed" / "radklim"
    out_dir, meta_dir = data / "processed" / "aligned", data / "metadata" / "grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    with Dataset(era_dir / "era5_2019_01.nc") as era0, Dataset(meta_dir / "source_grids.nc") as grids:
        era_lat, era_lon = np.asarray(era0.variables["lat"][:]), np.asarray(era0.variables["lon"][:])
        lat = np.asarray(grids.variables["radklim_lat"][:])
        lon = np.asarray(grids.variables["radklim_lon"][:])
        x = np.asarray(grids.variables["radklim_x"][:])
        y = np.asarray(grids.variables["radklim_y"][:])
        crs_attrs = {name: getattr(grids.variables["radklim_crs"], name) for name in grids.variables["radklim_crs"].ncattrs()}
    ys, xs, crop_lat, crop_lon, valid, i0, j0, weights = mapping(lat, lon, era_lat, era_lon)
    mapping_path = meta_dir / "era5_to_radklim_bilinear.npz"
    partial_map = mapping_path.with_suffix(".npz.part")
    with partial_map.open("wb") as stream:
        np.savez_compressed(stream, radklim_y_slice=np.array([ys.start, ys.stop]), radklim_x_slice=np.array([xs.start, xs.stop]),
                            lat=crop_lat, lon=crop_lon, overlap_mask=valid, era_lat_index=i0, era_lon_index=j0,
                            weight_00=weights[0], weight_01=weights[1], weight_10=weights[2], weight_11=weights[3],
                            bbox=np.array(BBOX), method=np.array("bilinear"))
    partial_map.replace(mapping_path)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
              "area_bbox_south_north_west_east": list(BBOX), "spatial_method": "bilinear ERA5 to cropped RADKLIM-YW grid",
              "time_rule": "Sum exactly 12 YW five-minute start times in [ERA5 time - 3600 s, ERA5 time).",
              "excluded_timestamps": ["2019-01-01T00:00:00Z"], "months": {}, "grid_crop_shape": list(crop_lat.shape),
              "grid_valid_pixels": int(valid.sum()), "grid_crop_source_slices": {"y": [ys.start, ys.stop], "x": [xs.start, xs.stop]}}
    for year in (2019, 2020):
        for month in range(1, 13):
            stem = f"{year}_{month:02d}"
            target = out_dir / f"aligned_{stem}.nc"
            if target.is_file() and not args.overwrite:
                with Dataset(target) as existing:
                    count = len(existing.dimensions["time"])
                report["months"][stem] = {"status": "existing_skipped", "time_count": count}
                continue
            with Dataset(era_dir / f"era5_{stem}.nc") as era:
                times = np.asarray(era.variables["time"][:], dtype=np.int64)
                expected_hours = calendar.monthrange(year, month)[1] * 24
                if len(times) != expected_hours or not np.all(np.diff(times) == 3600):
                    raise ValueError(f"ERA5 time axis is incomplete or irregular in {stem}")
                start_index = 1 if (year, month) == (2019, 1) else 0
                output_times = times[start_index:]
                partial = target.with_suffix(".nc.part")
                partial.unlink(missing_ok=True)
                with Dataset(partial, "w", format="NETCDF4") as dst:
                    nt, ny, nx = len(output_times), crop_lat.shape[0], crop_lat.shape[1]
                    dst.createDimension("time", nt); dst.createDimension("y", ny); dst.createDimension("x", nx)
                    t = dst.createVariable("time", "i8", ("time",)); t.units = TIME_UNITS; t.calendar = "proleptic_gregorian"; t.standard_name = "time"; t[:] = output_times
                    for name, values, units, standard, axis in (("x", x[xs], "m", "projection_x_coordinate", "X"), ("y", y[ys], "m", "projection_y_coordinate", "Y")):
                        var = dst.createVariable(name, "f8", (name,)); var[:] = values; var.units = units; var.standard_name = standard; var.axis = axis
                    for name, values, units, standard in (("lat", crop_lat, "degrees_north", "latitude"), ("lon", crop_lon, "degrees_east", "longitude")):
                        var = dst.createVariable(name, "f8", ("y", "x"), zlib=True, complevel=2); var[:] = values; var.units = units; var.standard_name = standard
                    crs = dst.createVariable("crs", "i4", ())
                    for name, value in crs_attrs.items(): setattr(crs, name, value)
                    a = dst.createVariable("precipitation_era5", "f4", ("time", "y", "x"), zlib=True, complevel=3, shuffle=True, chunksizes=(1, min(256,ny), min(256,nx)), fill_value=np.float32(np.nan))
                    b = dst.createVariable("precipitation_radklim", "f4", ("time", "y", "x"), zlib=True, complevel=3, shuffle=True, chunksizes=(1, min(256,ny), min(256,nx)), fill_value=np.float32(np.nan))
                    for var, label in ((a, "ERA5"), (b, "RADKLIM-YW")):
                        var.units = "mm"; var.standard_name = "precipitation_amount"; var.grid_mapping = "crs"; var.coordinates = "lat lon"; var.long_name = f"Hourly precipitation amount from {label}"
                    a.cell_methods = b.cell_methods = "time: sum (interval: 1 hour)"
                    dst.title = "Hourly ERA5 and RADKLIM-YW precipitation on common grid"
                    dst.temporal_alignment = "Strict matching of ERA5 accumulation end time and YW sum over the preceding 60 minutes."
                    dst.spatial_alignment = "ERA5 bilinearly interpolated to native RADKLIM-YW grid; outside bbox stored as NaN."
                    dst.area = "46.5–55.0 N, 2.0–16.0 E"
                    yw_cache = {}
                    for oi, time_value in enumerate(output_times):
                        window_start = int(time_value) - 3600
                        first_date = datetime.fromtimestamp(window_start, tz=timezone.utc)
                        last_date = datetime.fromtimestamp(int(time_value) - 300, tz=timezone.utc)
                        for d in {first_date.date(), last_date.date()}:
                            ym = (d.year, d.month)
                            if ym not in yw_cache:
                                if len(yw_cache) == 2:
                                    old = next(iter(yw_cache)); yw_cache.pop(old).close()
                                yw_cache[ym] = Dataset(yw_dir / f"radklim_{ym[0]}_{ym[1]:02d}.nc")
                        yd = yw_cache[(first_date.year, first_date.month)]
                        ytime = yd.variables["time"]
                        pos = int(np.searchsorted(ytime[:], window_start))
                        window = np.asarray(ytime[pos:pos + 12], dtype=np.int64)
                        wanted = np.arange(window_start, int(time_value), 300, dtype=np.int64)
                        if len(window) != 12 or not np.array_equal(window, wanted):
                            raise ValueError(f"YW source window does not exactly match ERA5 timestamp {int(time_value)}")
                        rain = np.ma.asarray(yd.variables["precipitation"][pos:pos + 12, :, :], dtype=np.float32)
                        vals = np.ma.filled(rain, np.nan)
                        total = np.sum(vals, axis=0, dtype=np.float32)
                        total[np.any(~np.isfinite(vals), axis=0)] = np.nan
                        total = total[ys, xs]
                        ev = np.asarray(era.variables["precipitation"][start_index + oi, :, :], dtype=np.float32)
                        a[oi] = interpolate(ev, valid, i0, j0, weights)
                        b[oi] = total
                        if (oi + 1) % 100 == 0:
                            print(f"[{stem}] aligned {oi + 1}/{nt} hours", flush=True)
                    for ds in yw_cache.values(): ds.close()
                partial.replace(target)
            report["months"][stem] = {"status": "written", "time_count": len(output_times), "first_time_epoch": int(output_times[0]), "last_time_epoch": int(output_times[-1])}
            print(f"[done] {target.name}: {len(output_times)} matched hours", flush=True)
    report["total_aligned_hours"] = sum(v["time_count"] for v in report["months"].values())
    report["bilinear_map"] = str(mapping_path.relative_to(data))
    (meta_dir / "alignment_coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_scalar) + "\n"
    )
    print(f"Total aligned hours: {report['total_aligned_hours']}", flush=True)


if __name__ == "__main__":
    main()
