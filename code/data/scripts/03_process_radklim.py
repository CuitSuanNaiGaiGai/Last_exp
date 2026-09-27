"""Normalize daily RADKLIM-YW files into monthly five-minute NetCDF files."""

import argparse
import calendar
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num, num2date

PROJECT_ROOT = Path(__file__).resolve().parents[3]
YEARS = (2019, 2020)
MEMBER = re.compile(r"(?:^|/)YW_2017\.002_(20\d{2})(\d{2})(\d{2})\.nc$")
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
STEPS_PER_DAY = 288


def make_month_file(path: Path, src: Dataset, year: int, month: int) -> Dataset:
    days = calendar.monthrange(year, month)[1]
    ny, nx = len(src.dimensions["y"]), len(src.dimensions["x"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", days * STEPS_PER_DAY)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)
    time = ds.createVariable("time", "i8", ("time",))
    time.units = TIME_UNITS
    time.calendar = "proleptic_gregorian"
    time.standard_name = "time"
    for name, standard, axis in (("x", "projection_x_coordinate", "X"), ("y", "projection_y_coordinate", "Y")):
        var = ds.createVariable(name, "f8", (name,))
        var.units = getattr(src.variables[name], "units", "m")
        var.standard_name, var.axis = standard, axis
        var[:] = src.variables[name][:]
    for name, standard, units in (("lat", "latitude", "degrees_north"), ("lon", "longitude", "degrees_east")):
        var = ds.createVariable(name, "f8", ("y", "x"), zlib=True, complevel=2)
        var.standard_name, var.units = standard, units
        var[:] = src.variables[name][:]
    crs = ds.createVariable("crs", "i4", ())
    for name in src.variables["crs"].ncattrs():
        setattr(crs, name, getattr(src.variables["crs"], name))
    rain = ds.createVariable(
        "precipitation", "f4", ("time", "y", "x"), zlib=True, complevel=4, shuffle=True,
        chunksizes=(12, min(256, ny), min(256, nx)), fill_value=np.float32(np.nan),
    )
    rain.units = "mm"
    rain.standard_name = "precipitation_amount"
    rain.long_name = "RADKLIM-YW five-minute precipitation amount"
    rain.cell_methods = "time: sum (interval: 5 minutes)"
    rain.grid_mapping = "crs"
    rain.coordinates = "lat lon"
    rain.comment = "Source kg m-2 equals mm water equivalent; fill value 999 is stored as NaN."
    ds.title = "RADKLIM-YW five-minute precipitation, normalized"
    ds.source_product = "DWD RADKLIM-YW V2017.002"
    ds.time_reference = "Each timestamp is the start of its five-minute measurement period (UTC)."
    ds.processing_history = f"Processed {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC"
    ds.sync()
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data = args.root.resolve() / "code" / "data"
    source, output = data / "raw" / "radklim", data / "processed" / "radklim"
    for year in YEARS:
        archive = source / f"YW2017.002_{year}_netcdf.tar.gz"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        expected = {(year, m, d) for m in range(1, 13) for d in range(1, calendar.monthrange(year, m)[1] + 1)}
        seen: set[tuple[int, int, int]] = set()
        outputs: dict[int, Dataset] = {}
        partials: dict[int, Path] = {}
        skipped: set[int] = set()
        grid_ref = None
        try:
            with tarfile.open(archive, "r|gz") as tf:
                for member in tf:
                    match = MEMBER.search(member.name)
                    if not member.isfile() or not match:
                        continue
                    y, month, day = map(int, match.groups())
                    key = (y, month, day)
                    if y != year or key in seen or key not in expected:
                        raise ValueError(f"Unexpected or duplicate YW member: {member.name}")
                    seen.add(key)
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        raise OSError(f"Cannot read archive member {member.name}")
                    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                        shutil.copyfileobj(extracted, tmp, length=8 * 1024 * 1024)
                        tmp.flush()
                        with Dataset(tmp.name) as src:
                            rain = src.variables["RR"]
                            if rain.shape != (STEPS_PER_DAY, len(src.dimensions["y"]), len(src.dimensions["x"])):
                                raise ValueError(f"Unexpected RR dimensions/shape in {member.name}: {rain.shape}")
                            units = getattr(rain, "units", "").strip().lower().replace("^", "")
                            if units not in {"kg m-2", "mm"}:
                                raise ValueError(f"Unexpected YW precipitation units {units!r} in {member.name}")
                            time_var = src.variables["time"]
                            dates = num2date(time_var[:], time_var.units, getattr(time_var, "calendar", "standard"), only_use_cftime_datetimes=False)
                            epoch = date2num(dates, TIME_UNITS, calendar="proleptic_gregorian").astype(np.int64)
                            expected_start = int(date2num(datetime(y, month, day), TIME_UNITS, calendar="proleptic_gregorian"))
                            if len(epoch) != STEPS_PER_DAY or epoch[0] != expected_start or not np.all(np.diff(epoch) == 300):
                                raise ValueError(f"YW timestamps are not a complete 00:00–23:55 sequence: {member.name}")
                            current_grid = {n: np.asarray(src.variables[n][:]) for n in ("x", "y", "lat", "lon")}
                            current_grid["crs"] = getattr(src.variables["crs"], "crs_wkt", "")
                            if grid_ref is None:
                                grid_ref = current_grid
                            else:
                                for n in ("x", "y", "lat", "lon"):
                                    if current_grid[n].shape != grid_ref[n].shape or not np.allclose(current_grid[n], grid_ref[n], rtol=0, atol=1e-8, equal_nan=True):
                                        raise ValueError(f"Inconsistent {n} grid in {member.name}")
                                if current_grid["crs"] != grid_ref["crs"]:
                                    raise ValueError(f"Inconsistent CRS in {member.name}")
                            target = output / f"radklim_{year}_{month:02d}.nc"
                            if target.is_file() and not args.overwrite:
                                skipped.add(month)
                                continue
                            if month not in outputs:
                                partial = target.with_suffix(".nc.part")
                                partial.unlink(missing_ok=True)
                                partials[month] = partial
                                outputs[month] = make_month_file(partial, src, year, month)
                            dst = outputs[month]
                            start = (day - 1) * STEPS_PER_DAY
                            dst.variables["time"][start:start + STEPS_PER_DAY] = epoch
                            for i in range(0, STEPS_PER_DAY, 12):
                                values = np.ma.asarray(rain[i:i + 12, :, :], dtype=np.float32)
                                values = np.ma.filled(values, np.nan)
                                fill = getattr(rain, "_FillValue", None)
                                if fill is not None:
                                    values[values == fill] = np.nan
                                dst.variables["precipitation"][start + i:start + i + 12, :, :] = values
                    if len(seen) % 30 == 0:
                        print(f"[{year}] validated {len(seen)}/{len(expected)} daily members", flush=True)
            missing = sorted(expected - seen)
            if missing:
                raise RuntimeError(f"Missing RADKLIM-YW days in {archive.name}: {missing[:10]} ({len(missing)} total)")
            for month, ds in outputs.items():
                ds.close()
                partials[month].replace(output / f"radklim_{year}_{month:02d}.nc")
                print(f"[done] RADKLIM-YW {year}-{month:02d}", flush=True)
            for month in sorted(skipped - outputs.keys()):
                print(f"[skip] radklim_{year}_{month:02d}.nc exists; archive dates and grids validated", flush=True)
        except Exception:
            for ds in outputs.values():
                try:
                    ds.close()
                except Exception:
                    pass
            for partial in partials.values():
                partial.unlink(missing_ok=True)
            raise
        print(f"Year {year}: validated {len(seen)}/{len(expected)} daily members", flush=True)


if __name__ == "__main__":
    main()
