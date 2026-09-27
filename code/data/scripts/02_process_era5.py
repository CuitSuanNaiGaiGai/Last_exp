"""Normalize monthly ERA5 total precipitation to millimetres and CF time."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num, num2date


PROJECT_ROOT = Path(__file__).resolve().parents[3]
YEARS = (2019, 2020)


def process_month(source: Path, target: Path, overwrite: bool = False) -> None:
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        print(f"[skip] {target.name} already exists")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".nc.part")
    partial.unlink(missing_ok=True)
    try:
        with Dataset(source) as src:
            time_name = "valid_time" if "valid_time" in src.variables else "time"
            precip_name = "tp" if "tp" in src.variables else "total_precipitation"
            lat_name = "latitude" if "latitude" in src.variables else "lat"
            lon_name = "longitude" if "longitude" in src.variables else "lon"
            time_src = src.variables[time_name]
            precip_src = src.variables[precip_name]
            lat = np.asarray(src.variables[lat_name][:], dtype=np.float32)
            lon = np.asarray(src.variables[lon_name][:], dtype=np.float32)
            flip_lat = len(lat) > 1 and lat[0] > lat[-1]
            if flip_lat:
                lat = lat[::-1].copy()

            raw_units = getattr(precip_src, "units", "").strip().lower()
            if raw_units in {"m", "meter", "metre", "meters", "metres"}:
                unit_scale = 1000.0
                conversion = "ERA5 total precipitation converted from m to mm (x 1000)."
            elif raw_units in {"mm", "kg m-2", "kg m^-2"}:
                unit_scale = 1.0
                conversion = "Source precipitation is already expressed as millimetres water equivalent."
            else:
                raise ValueError(f"Unsupported precipitation unit {raw_units!r} in {source}")

            times = num2date(
                time_src[:],
                time_src.units,
                getattr(time_src, "calendar", "proleptic_gregorian"),
                only_use_cftime_datetimes=False,
            )
            time_units = "seconds since 1970-01-01 00:00:00"
            time_values = date2num(times, time_units, calendar="proleptic_gregorian").astype(np.int64)

            with Dataset(partial, "w", format="NETCDF4") as dst:
                dst.createDimension("time", len(time_values))
                dst.createDimension("lat", len(lat))
                dst.createDimension("lon", len(lon))

                time = dst.createVariable("time", "i8", ("time",))
                time.units = time_units
                time.calendar = "proleptic_gregorian"
                time.standard_name = "time"
                time[:] = time_values

                lat_var = dst.createVariable("lat", "f4", ("lat",))
                lon_var = dst.createVariable("lon", "f4", ("lon",))
                lat_var.units = "degrees_north"
                lat_var.standard_name = "latitude"
                lon_var.units = "degrees_east"
                lon_var.standard_name = "longitude"
                lat_var[:] = lat
                lon_var[:] = lon

                precipitation = dst.createVariable(
                    "precipitation",
                    "f4",
                    ("time", "lat", "lon"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=(min(24, len(time_values)), len(lat), len(lon)),
                    fill_value=np.float32(np.nan),
                )
                precipitation.units = "mm"
                precipitation.standard_name = "precipitation_amount"
                precipitation.long_name = "hourly total precipitation"
                precipitation.cell_methods = "time: sum (interval: 1 hour)"
                precipitation.comment = conversion
                for start in range(0, len(time_values), 24):
                    stop = min(start + 24, len(time_values))
                    values = np.ma.asarray(precip_src[start:stop, :, :], dtype=np.float32)
                    values = np.ma.filled(values, np.nan) * unit_scale
                    if flip_lat:
                        values = values[:, ::-1, :]
                    precipitation[start:stop, :, :] = values

                dst.title = "ERA5 hourly total precipitation, normalized"
                dst.source_file = source.name
                dst.processing_history = f"Processed {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC"
                dst.time_reference = "ERA5 valid_time denotes the end of the hourly accumulation period (UTC)."
                dst.area = "46.5–55.0 N, 2.0–16.0 E"
        partial.replace(target)
        print(f"[done] {source.name} -> {target}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing processed files")
    args = parser.parse_args()
    data_root = args.root.resolve() / "code" / "data"
    source_dir = data_root / "raw" / "era5"
    target_dir = data_root / "processed" / "era5"
    for year in YEARS:
        for month in range(1, 13):
            stem = f"{year}_{month:02d}"
            process_month(
                source_dir / f"era5_tp_{stem}.nc",
                target_dir / f"era5_{stem}.nc",
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
