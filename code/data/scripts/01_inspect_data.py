"""Inspect ERA5 and RADKLIM-YW source metadata and write grid inventory files."""

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "code" / "data"
YEARS = (2019, 2020)
YW_MEMBER = re.compile(r"(?:^|/)YW_2017\.002_(20\d{2})(\d{2})(\d{2})\.nc$")


def iso_time(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def time_bounds(variable) -> tuple[str, str, float | None]:
    values = variable[:]
    dates = num2date(
        [values[0], values[-1]],
        variable.units,
        getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    step = float(values[1] - values[0]) if len(values) > 1 else None
    return iso_time(dates[0]), iso_time(dates[1]), step


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def inspect_era5() -> tuple[list[dict], dict | None]:
    root = DATA_ROOT / "raw" / "era5"
    records, grid = [], None
    for year in YEARS:
        for month in range(1, 13):
            path = root / f"era5_tp_{year}_{month:02d}.nc"
            if not path.is_file():
                records.append({"file": path.name, "status": "missing"})
                continue
            with Dataset(path) as ds:
                time_name = "valid_time" if "valid_time" in ds.variables else "time"
                precip_name = "tp" if "tp" in ds.variables else "total_precipitation"
                lat_name = "latitude" if "latitude" in ds.variables else "lat"
                lon_name = "longitude" if "longitude" in ds.variables else "lon"
                time, precip = ds.variables[time_name], ds.variables[precip_name]
                start, end, step = time_bounds(time)
                records.append({
                    "file": path.name,
                    "status": "ok",
                    "size_bytes": path.stat().st_size,
                    "time_count": len(time),
                    "time_start": start,
                    "time_end": end,
                    "time_step_seconds": step,
                    "variable": precip_name,
                    "variable_dimensions": list(precip.dimensions),
                    "variable_shape": list(precip.shape),
                    "units": getattr(precip, "units", None),
                    "fill_value": scalar(getattr(precip, "_FillValue", None)),
                })
                if grid is None:
                    grid = {
                        "lat": np.asarray(ds.variables[lat_name][:], dtype=np.float64),
                        "lon": np.asarray(ds.variables[lon_name][:], dtype=np.float64),
                    }
    return records, grid


def inspect_yw_sample(tf, member, year: int) -> tuple[dict, dict]:
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        shutil.copyfileobj(tf.extractfile(member), tmp)
        tmp.flush()
        with Dataset(tmp.name) as ds:
            time = ds.variables["time"]
            precip_name = "RR" if "RR" in ds.variables else "precipitation"
            precip = ds.variables[precip_name]
            start, end, step = time_bounds(time)
            sample = {
                "member": member.name,
                "size_bytes": member.size,
                "time_count_per_day": len(time),
                "time_start": start,
                "time_end": end,
                "time_step_seconds": step,
                "variable": precip_name,
                "variable_dimensions": list(precip.dimensions),
                "variable_shape": list(precip.shape),
                "units": getattr(precip, "units", None),
                "fill_value": scalar(getattr(precip, "_FillValue", None)),
                "crs_attributes": {
                    name: scalar(getattr(ds.variables["crs"], name))
                    for name in ds.variables["crs"].ncattrs()
                },
            }
            grid = {
                "x": np.asarray(ds.variables["x"][:], dtype=np.float64),
                "y": np.asarray(ds.variables["y"][:], dtype=np.float64),
                "lat": np.asarray(ds.variables["lat"][:], dtype=np.float64),
                "lon": np.asarray(ds.variables["lon"][:], dtype=np.float64),
                "crs": {
                    name: getattr(ds.variables["crs"], name)
                    for name in ds.variables["crs"].ncattrs()
                },
            }
    return sample, grid


def expected_days(year: int) -> set[date]:
    day = date(year, 1, 1)
    stop = date(year + 1, 1, 1)
    result = set()
    while day < stop:
        result.add(day)
        day += timedelta(days=1)
    return result


def inspect_yw() -> tuple[dict, dict | None]:
    root = DATA_ROOT / "raw" / "radklim"
    years_report, grid = {}, None
    for year in YEARS:
        archive = root / f"YW2017.002_{year}_netcdf.tar.gz"
        if not archive.is_file():
            years_report[str(year)] = {"status": "missing_archive", "archive": archive.name}
            continue
        seen, monthly_counts, sample_info = set(), {}, None
        with tarfile.open(archive, mode="r|gz") as tf:
            for member in tf:
                match = YW_MEMBER.search(member.name)
                if not member.isfile() or not match:
                    continue
                y, month, day = map(int, match.groups())
                if y != year:
                    raise ValueError(f"Unexpected YW member {member.name} in {archive.name}")
                current_date = date(y, month, day)
                if current_date in seen:
                    raise ValueError(f"Duplicate RADKLIM-YW date: {current_date}")
                seen.add(current_date)
                monthly_counts[f"{month:02d}"] = monthly_counts.get(f"{month:02d}", 0) + 1
                if sample_info is None:
                    sample_info, grid = inspect_yw_sample(tf, member, year)

        missing = sorted(d.isoformat() for d in expected_days(year) - seen)
        unexpected = sorted(d.isoformat() for d in seen - expected_days(year))
        years_report[str(year)] = {
            "archive": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "daily_files_found": len(seen),
            "daily_files_expected": len(expected_days(year)),
            "first_day": min(seen).isoformat() if seen else None,
            "last_day": max(seen).isoformat() if seen else None,
            "daily_files_by_month": monthly_counts,
            "missing_days": missing,
            "unexpected_days": unexpected,
            "sample_daily_file": sample_info,
            "status": "ok" if not missing and not unexpected else "incomplete",
        }
    return years_report, grid


def write_grids(path: Path, era_grid: dict | None, rad_grid: dict | None) -> None:
    if era_grid is None or rad_grid is None:
        return
    partial = path.with_suffix(".nc.part")
    partial.unlink(missing_ok=True)
    with Dataset(partial, "w", format="NETCDF4") as ds:
        ds.createDimension("era_lat", len(era_grid["lat"]))
        ds.createDimension("era_lon", len(era_grid["lon"]))
        ds.createDimension("rad_y", len(rad_grid["y"]))
        ds.createDimension("rad_x", len(rad_grid["x"]))
        for name, dim, units in (("era5_lat", "era_lat", "degrees_north"), ("era5_lon", "era_lon", "degrees_east")):
            var = ds.createVariable(name, "f8", (dim,))
            var.units = units
            var[:] = era_grid["lat" if name.endswith("lat") else "lon"]
        for name, dim, units in (("radklim_x", "rad_x", "m"), ("radklim_y", "rad_y", "m")):
            var = ds.createVariable(name, "f8", (dim,))
            var.units = units
            var[:] = rad_grid["x" if name.endswith("_x") else "y"]
        for name, units in (("radklim_lat", "degrees_north"), ("radklim_lon", "degrees_east")):
            var = ds.createVariable(name, "f8", ("rad_y", "rad_x"), zlib=True, complevel=2)
            var.units = units
            var[:] = rad_grid["lat" if name.endswith("lat") else "lon"]
        crs = ds.createVariable("radklim_crs", "i4", ())
        for name, value in rad_grid["crs"].items():
            setattr(crs, name, scalar(value))
        ds.title = "Native ERA5 and RADKLIM-YW source grids"
        ds.era5_grid = "Regular 0.25 degree latitude-longitude grid"
        ds.radklim_grid = "1 km RADOLAN polar stereographic grid"
        ds.spatial_overlap_bbox = "46.5–55.0 N, 2.0–16.0 E"
        ds.temporal_alignment = (
            "ERA5 accumulation ending at HH:00 matches the 12 RADKLIM-YW five-minute "
            "amounts whose interval starts run from (HH-1):00 through (HH-1):55."
        )
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    args = parser.parse_args()
    global DATA_ROOT
    DATA_ROOT = args.root.resolve() / "code" / "data"

    era5, era_grid = inspect_era5()
    yw, rad_grid = inspect_yw()
    metadata_dir = DATA_ROOT / "metadata" / "grid"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "years": list(YEARS),
        "era5_months": era5,
        "radklim_yw": yw,
        "radklim_rw_archives_retained": [
            {"file": f"RW2017.002_{year}_netcdf.tar.gz", "present": (DATA_ROOT / "raw" / "radklim" / f"RW2017.002_{year}_netcdf.tar.gz").is_file()}
            for year in YEARS
        ],
        "yw_time_convention": "Each timestamp is the start of its five-minute measurement period (DWD format specification).",
        "strict_hourly_window_match": True,
        "hourly_aggregation": "Sum the 12 YW values whose timestamps lie in [ERA5 valid_time - 1 hour, ERA5 valid_time).",
        "first_era5_timestamp_without_yw_context": "2019-01-01T00:00:00Z",
        "first_hour_note": "This one ERA5 hour needs 2018-12-31 YW values, which are outside the downloaded 2019-2020 YW archives.",
        "radklim_yw_doi": "10.5676/DWD/RADKLIM_YW_V2017.002",
        "expected_era5_months": 24,
        "found_era5_months": sum(record.get("status") == "ok" for record in era5),
    }
    (metadata_dir / "data_inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_grids(metadata_dir / "source_grids.nc", era_grid, rad_grid)
    print(f"ERA5: {report['found_era5_months']}/24 monthly files")
    for year, details in yw.items():
        print(f"RADKLIM-YW {year}: {details.get('daily_files_found', 0)}/{details.get('daily_files_expected', 0)} daily files ({details.get('status')})")
    print("Exact hourly-window aggregation: possible; the first 2019 ERA5 hour lacks preceding YW data")
    print(f"Inventory: {metadata_dir / 'data_inventory.json'}")
    print(f"Source grids: {metadata_dir / 'source_grids.nc'}")


if __name__ == "__main__":
    main()
