# Precipitation Data Preparation Implementation Plan

**Goal:** Inspect, normalize, temporally aggregate, and spatially align ERA5 and RADKLIM-YW data for 2019–2020.

**Architecture:** Inspect the 24 ERA5 files and yearly YW archives; normalize ERA5 to mm and YW five-minute amounts to a common schema; aggregate the 12 YW five-minute periods beginning in each ERA5 one-hour accumulation window; bilinearly interpolate ERA5 onto the RADKLIM grid over the geographic overlap. Keep all stages in monthly NetCDF files and record the interpolation map and temporal coverage.

**Tech Stack:** Python standard library, `netCDF4`, and NumPy.

## Global Constraints

- Preserve all original ERA5, RADKLIM-RW, and RADKLIM-YW raw files.
- Write ERA5, RADKLIM-YW, and aligned data to their respective `code/data/processed/` directories.
- Use `time` and `precipitation` as normalized names; precipitation is in millimetres.
- ERA5 hourly timestamps are accumulation end times; RADKLIM-YW timestamps start their five-minute periods.
- Strict hourly aggregation sums YW timestamps in `[ERA5 valid_time - 1 hour, ERA5 valid_time)`.
- Keep the 1 km RADKLIM grid and bilinearly interpolate ERA5 over the spatial overlap.
- The 2019-01-01 00:00 ERA5 timestamp needs YW from 2018-12-31; omit this one timestamp and record it.
- Do not generate train/val/test splits or patches.
- Validate by running the data scripts and checking produced metadata, coverage, values, and spatial maps; do not add a test suite.

---

### Task 1: Source inspection and grid metadata

**Files:**
- Create: `code/data/scripts/01_inspect_data.py`
- Create: `code/data/metadata/grid/data_inventory.json`
- Create: `code/data/metadata/grid/source_grids.nc`

- [x] Inventory ERA5 monthly files and the daily RADKLIM-YW members and inspect representative metadata.
- [x] Save source coordinates, RADKLIM CRS, archive coverage, and exact time-window rule.
- [x] Run the updated inspection script and verify 24 ERA5 months and all 2019–2020 YW days.

### Task 2: ERA5 normalization

**Files:**
- Create: `code/data/scripts/02_process_era5.py`
- Create: 24 monthly files in `code/data/processed/era5/`

- [x] Rename `valid_time` to `time`, `latitude`/`longitude` to `lat`/`lon`, and `tp` to `precipitation`.
- [x] Convert precipitation from metres to millimetres and write compressed monthly output.
- [x] Verify timestep counts, coordinate order, units, and sampled converted values.

### Task 3: RADKLIM-YW normalization

**Files:**
- Modify: `code/data/scripts/03_process_radklim.py`
- Create: 24 monthly files in `code/data/processed/radklim/`
- Preserve prior RW results in: `code/data/processed/radklim_rw/`

- [x] Stream all daily members from the yearly YW archives into monthly NetCDF files.
- [x] Preserve 5-minute timestamps, convert fill 999 to NaN, standardize precipitation to mm, and retain native coordinates and CRS.
- [x] Verify all 731 days, exact five-minute cadence, grid consistency, units, and sample values.

### Task 4: Exact temporal and spatial alignment

**Files:**
- Create: `code/data/scripts/04_align_data.py`
- Create: monthly NetCDF files in `code/data/processed/aligned/`
- Create: `code/data/metadata/grid/era5_to_radklim_bilinear.npz`
- Create: `code/data/metadata/grid/alignment_coverage.json`

- [x] Aggregate each exact 60-minute YW window ending at the ERA5 valid time.
- [x] Bilinearly interpolate ERA5 to the RADKLIM grid and retain only the geographic intersection.
- [x] Omit the single 2019-01-01 00:00 ERA5 time lacking 2018 YW context and record it.
- [x] Verify aligned hourly timestamps, paired array shapes, precipitation units, overlap mask, and sampled temporal/spatial values.
