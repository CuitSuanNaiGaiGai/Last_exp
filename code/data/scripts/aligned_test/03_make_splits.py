"""Create reproducible chronological train/val/test date and timestamp splits."""

from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
from netCDF4 import Dataset

from common import ALIGNED_DIR, CONFIG, OUTPUT_DIR


def main():
    start = date(2019, 1, 1)
    stop = date(2021, 1, 1)
    split_days = CONFIG["split_days"]
    dates = [start + timedelta(days=i) for i in range((stop - start).days)]
    expected = sum(int(split_days[name]) for name in ("train", "val", "test"))
    if len(dates) != expected:
        raise ValueError(f"Configured split days sum to {expected}, but timeline has {len(dates)} days")
    cuts = [int(split_days["train"]), int(split_days["train"]) + int(split_days["val"])]
    date_sets = {"train": dates[:cuts[0]], "val": dates[cuts[0]:cuts[1]], "test": dates[cuts[1]:]}

    all_times = []
    for path in sorted(ALIGNED_DIR.glob("aligned_*.nc")):
        with Dataset(path) as ds:
            all_times.extend(np.asarray(ds.variables["time"][:], dtype=np.int64).tolist())
    all_times = np.asarray(all_times, dtype=np.int64)
    epoch_dates = np.array([datetime.fromtimestamp(int(t), timezone.utc).date() for t in all_times], dtype=object)
    report = {"method": "chronological UTC calendar-day split; no random frame splitting", "configured_days": split_days, "hourly_samples": int(len(all_times)), "periods": {}}
    root = OUTPUT_DIR / "splits"
    root.mkdir(parents=True, exist_ok=True)
    for name, selected_dates in date_sets.items():
        date_strings = [d.isoformat() for d in selected_dates]
        selected = np.isin(epoch_dates, selected_dates)
        timestamps = all_times[selected]
        if len(timestamps) == 0:
            raise ValueError(f"Split {name} unexpectedly has no aligned timestamps")
        (root / f"{name}_dates.txt").write_text("\n".join(date_strings) + "\n")
        (root / f"{name}_timestamps.txt").write_text("\n".join(map(str, timestamps.tolist())) + "\n")
        report["periods"][name] = {
            "date_count": len(selected_dates),
            "first_date_utc": date_strings[0],
            "last_date_utc": date_strings[-1],
            "hour_count": int(len(timestamps)),
            "first_timestamp_utc": datetime.fromtimestamp(int(timestamps[0]), timezone.utc).isoformat(),
            "last_timestamp_utc": datetime.fromtimestamp(int(timestamps[-1]), timezone.utc).isoformat(),
        }
    sets = [set(date_sets[k]) for k in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Date splits overlap")
    report["disjoint"] = True
    report["total_split_hours"] = sum(report["periods"][k]["hour_count"] for k in date_sets)
    report["first_aligned_hour_exclusion"] = "2019-01-01T00:00:00+00:00 is absent from aligned data"
    (root / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
