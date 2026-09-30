"""Shared paths, metadata, monthly NetCDF access, and bilinear mapping helpers."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / "code" / "data"
ALIGNED_DIR = DATA_ROOT / "processed" / "aligned"
ERA5_DIR = DATA_ROOT / "processed" / "era5"
RADKLIM_DIR = DATA_ROOT / "processed" / "radklim"
GRID_DIR = DATA_ROOT / "metadata" / "grid"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
CONFIG = json.loads((HERE / "config.json").read_text())
TIME_UNITS = "seconds since 1970-01-01 00:00:00"


def month_stem(timestamp: int) -> str:
    value = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    return f"{value.year}_{value.month:02d}"


def load_bilinear_map() -> dict[str, np.ndarray]:
    path = GRID_DIR / "era5_to_radklim_bilinear.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ERA5-to-RADKLIM map: {path}")
    with np.load(path) as saved:
        return {key: saved[key].copy() for key in saved.files}


class DatasetCache:
    """Small LRU cache of open NetCDF files; call close() when finished."""

    def __init__(self, maximum_open: int = 3):
        self.maximum_open = maximum_open
        self._datasets: OrderedDict[Path, Dataset] = OrderedDict()

    def get(self, path: Path) -> Dataset:
        path = path.resolve()
        if path in self._datasets:
            self._datasets.move_to_end(path)
            return self._datasets[path]
        if len(self._datasets) >= self.maximum_open:
            _, old = self._datasets.popitem(last=False)
            old.close()
        if not path.is_file():
            raise FileNotFoundError(path)
        ds = Dataset(path)
        self._datasets[path] = ds
        return ds

    def close(self) -> None:
        while self._datasets:
            _, ds = self._datasets.popitem(last=False)
            ds.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def read_field(variable, time_index: int) -> np.ndarray:
    """Read one 2-D field as float32, representing both NetCDF masks and NaNs."""
    values = np.ma.asarray(variable[int(time_index)], dtype=np.float32)
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)


def bilinear_to_radklim(coarse: np.ndarray, mapping: dict[str, np.ndarray], region=None) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact saved ERA5-to-RADKLIM bilinear weights to a coarse field.

    Returns mapped precipitation and its availability mask. Any missing source
    corner makes the bilinear result unavailable, matching the strict map use.
    """
    if coarse.ndim != 2:
        raise ValueError(f"Expected a 2-D ERA5-grid field, received shape {coarse.shape}")
    if region is None:
        y_slice, x_slice = slice(None), slice(None)
    else:
        y_slice, x_slice = region
    i = mapping["era_lat_index"][y_slice, x_slice]
    j = mapping["era_lon_index"][y_slice, x_slice]
    weights = [mapping[name][y_slice, x_slice] for name in ("weight_00", "weight_01", "weight_10", "weight_11")]
    corners = (coarse[i, j], coarse[i, j + 1], coarse[i + 1, j], coarse[i + 1, j + 1])
    valid = mapping["overlap_mask"][y_slice, x_slice].astype(bool, copy=True)
    for corner in corners:
        valid &= np.isfinite(corner)
    for weight in weights:
        valid &= np.isfinite(weight)
    result = np.zeros(valid.shape, dtype=np.float32)
    for corner, weight in zip(corners, weights):
        result += np.where(valid, corner, 0).astype(np.float32) * weight.astype(np.float32)
    result[~valid] = np.nan
    return result, valid


def bilinear_valid_mask(coarse: np.ndarray, mapping: dict[str, np.ndarray]) -> np.ndarray:
    """Return the exact availability mask of the strict four-corner map."""
    i = mapping["era_lat_index"]
    j = mapping["era_lon_index"]
    valid = mapping["overlap_mask"].astype(bool, copy=True)
    for corner in (coarse[i, j], coarse[i, j + 1], coarse[i + 1, j], coarse[i + 1, j + 1]):
        valid &= np.isfinite(corner)
    for name in ("weight_00", "weight_01", "weight_10", "weight_11"):
        valid &= np.isfinite(mapping[name])
    return valid


def common_support(target: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
    """Pixels eligible for a fair paired comparison at one time."""
    if any(field.shape != target.shape for field in inputs):
        raise ValueError("Target and model input fields must share the RADKLIM grid")
    valid = np.isfinite(target)
    for field in inputs:
        valid &= np.isfinite(field)
    return valid


def common_input_support(
    target: np.ndarray,
    era: np.ndarray,
    synthetic_available: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    """Return the shared raw-input support used by every paired metric.

    ``synthetic_available`` is the RADKLIM-grid mask from the saved strict
    four-corner bilinear map. The mapped synthetic field must be finite on
    these cells wherever it is materialized by the caller.
    """
    arrays = (era, synthetic_available, overlap)
    if any(np.shape(array) != np.shape(target) for array in arrays):
        raise ValueError("Target, ERA5, synthetic availability, and overlap must share one grid")
    return (
        np.asarray(overlap, dtype=bool)
        & np.isfinite(target)
        & np.isfinite(era)
        & np.asarray(synthetic_available, dtype=bool)
    )


def circular_neighborhood_kernel(radius_cells: int) -> np.ndarray:
    """Return a binary Euclidean disk on a square native-grid neighborhood."""
    radius_cells = int(radius_cells)
    if radius_cells < 0:
        raise ValueError("Neighborhood radius must be nonnegative")
    offsets = np.arange(-radius_cells, radius_cells + 1, dtype=np.int32)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    return (xx * xx + yy * yy <= radius_cells * radius_cells).astype(np.float32)


def require_finite_on_support(name: str, field: np.ndarray, support: np.ndarray) -> None:
    """Fail loudly instead of silently changing a paired metric denominator."""
    if np.shape(field) != np.shape(support):
        raise ValueError("{} and common support must share one grid".format(name))
    missing = np.asarray(support, dtype=bool) & ~np.isfinite(field)
    count = int(np.count_nonzero(missing))
    if count:
        raise ValueError(
            "{} has {} non-finite values on common support; refusing to shrink metric denominator".format(
                name, count
            )
        )


def timestamp_list_sha256(timestamps) -> str:
    """Stable fingerprint of an ordered Unix-second timestamp list."""
    values = np.asarray([int(value) for value in timestamps], dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def update_support_sha256(digest, timestamp: int, support: np.ndarray) -> None:
    """Add one timestamp and its packed row-major mask to a running digest."""
    mask = np.asarray(support, dtype=bool)
    digest.update(np.asarray([int(timestamp)], dtype="<i8").tobytes())
    digest.update(np.asarray(mask.shape, dtype="<i8").tobytes())
    digest.update(np.packbits(mask.reshape(-1)).tobytes())


def log_precipitation(values: np.ndarray) -> np.ndarray:
    """Apply the agreed log1p(mm) transform after clipping tiny negatives."""
    clipped = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return np.log1p(clipped).astype(np.float32)
