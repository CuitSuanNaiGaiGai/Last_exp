"""Download monthly ERA5 total precipitation for the RADKLIM study domain.

The 24 months are assigned round-robin to account1..account5 in
``~/.cds_profiles``. Existing complete targets are skipped, so the script can
be rerun after an interrupted download.
"""

import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "code" / "data" / "raw" / "era5"
PROFILE_DIR = Path.home() / ".cds_profiles"
YEARS = (2019, 2020)
ACCOUNTS = tuple(f"account{i}" for i in range(1, 6))

# CDS area order: North, West, South, East.
AREA = [55.0, 2.0, 46.5, 16.0]
TIMES = [f"{hour:02d}:00" for hour in range(24)]
DATASET = "reanalysis-era5-single-levels"


def read_profile(profile_path: Path) -> tuple[str, str]:
    """Read a CDS profile containing ``url:`` and ``key:`` entries."""
    values = {}
    for line in profile_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip().lower()] = value.strip()

    url, key = values.get("url"), values.get("key")
    if not url or not key:
        raise ValueError(f"Profile {profile_path.name} must define non-empty url and key fields")
    return url, key


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for account in ACCOUNTS:
        profile_path = PROFILE_DIR / account
        if not profile_path.is_file():
            raise FileNotFoundError(f"Missing CDS profile: {profile_path}")
        profiles[account] = read_profile(profile_path)

    months_by_account = {account: [] for account in ACCOUNTS}
    months = [(year, month) for year in YEARS for month in range(1, 13)]
    for index, month_key in enumerate(months):
        months_by_account[ACCOUNTS[index % len(ACCOUNTS)]].append(month_key)

    def download_account(account: str) -> None:
        url, key = profiles[account]
        client = cdsapi.Client(url=url, key=key)
        for year, month in months_by_account[account]:
            target = OUT_DIR / f"era5_tp_{year}_{month:02d}.nc"
            if target.is_file() and target.stat().st_size > 0:
                print(f"[skip] {target.name} already exists ({account})", flush=True)
                continue

            days = [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)]
            request = {
                "product_type": ["reanalysis"],
                "variable": ["total_precipitation"],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": TIMES,
                "area": AREA,
                "grid": [0.25, 0.25],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }

            partial = target.with_suffix(".nc.part")
            partial.unlink(missing_ok=True)
            print(f"[download] {year}-{month:02d} using {account}", flush=True)
            client.retrieve(DATASET, request, str(partial))
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError(f"CDS returned no data for {year}-{month:02d}")
            partial.replace(target)

    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        futures = {pool.submit(download_account, account): account for account in ACCOUNTS}
        failures = []
        for future in as_completed(futures):
            account = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((account, error))
                print(f"[error] {account}: {error}", flush=True)
        if failures:
            failed_accounts = ", ".join(account for account, _ in failures)
            raise RuntimeError(f"Downloads failed for: {failed_accounts}") from failures[0][1]

    print(f"Done. Monthly files are in {OUT_DIR}")


if __name__ == "__main__":
    main()
