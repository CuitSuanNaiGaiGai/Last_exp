# Last_exp

Research code for precipitation downscaling. The current completed data stage is
`preprocess_raw_align`: inspect sources, normalize ERA5 and RADKLIM-YW, and
produce monthly files with strict temporal and spatial alignment.

## Data preparation

Run from the repository root after placing the ERA5 and RADKLIM-YW source files
in `code/data/raw/`:

```bash
python code/data/scripts/01_inspect_data.py
python code/data/scripts/02_process_era5.py
python code/data/scripts/03_process_radklim.py
python code/data/scripts/04_align_data.py
```

The scripts use Python, NumPy, and netCDF4. `download_era5_tp.py` additionally
requires `cdsapi` and five CDS profiles under `~/.cds_profiles/account1` through
`account5`.

ERA5 hourly precipitation is paired with the sum of the 12 RADKLIM-YW
five-minute amounts whose start times fall in `[ERA5 valid time - 1 hour,
ERA5 valid time)`. ERA5 is bilinearly interpolated to the RADKLIM-YW grid over
46.5–55.0°N, 2.0–16.0°E. The 2019-01-01 00:00 hour is omitted because its
preceding RADKLIM-YW window is outside the downloaded archive period.

The aligned monthly outputs contain 17,543 paired hours across 2019–2020.
Generated NetCDF/NPZ datasets and raw archives are intentionally excluded from
Git; the scripts can recreate them locally. Lightweight JSON inventories and
coverage metadata are tracked under `code/data/metadata/grid/`.
