"""Summarize full test-period inputs/predictions and render representative fields."""

from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "outputs" / "matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / "outputs" / "cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset

from common import ALIGNED_DIR, CONFIG, OUTPUT_DIR, bilinear_to_radklim, load_bilinear_map
from model import SmallUNet


SPEC = importlib.util.spec_from_file_location("train_compare", HERE / "04_train_compare.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)
# Match the formal evaluation tile shape exactly. GroupNorm aggregates spatial
# statistics, so changing inference tile size changes predictions.
TRAIN.CONFIG["model"]["inference_core_size"] = 256
TRAIN.CONFIG["model"]["inference_batch_size"] = 4

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

METHODS = ("ERA5 raw", "Synthetic raw", "ERA5 model", "Synthetic model")
COLORS = {
    "ERA5 raw": "#0072B2",
    "Synthetic raw": "#E69F00",
    "ERA5 model": "#56B4E9",
    "Synthetic model": "#D55E00",
    "RADKLIM truth": "#333333",
}
STRATA = (
    ("dry (0)", lambda y: y == 0),
    ("0–0.1 (wet)", lambda y: (y > 0) & (y < 0.1)),
    ("0.1–1", lambda y: (y >= 0.1) & (y < 1.0)),
    ("1–5", lambda y: (y >= 1.0) & (y < 5.0)),
    ("≥5 mm/h", lambda y: y >= 5.0),
)
MAP_LABELS = ("RADKLIM truth", "ERA5 raw", "Synthetic raw", "ERA5 model", "Synthetic model")


def save_figure(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


def load_models(device):
    result = {}
    for mode in ("era5", "synthetic"):
        checkpoint_path = OUTPUT_DIR / "models" / mode / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing full-run checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = SmallUNet(base_channels=CONFIG["model"]["base_channels"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        result[mode] = (model, checkpoint)
    return result


def metric_accumulator():
    return {key: 0.0 for key in ("abs", "sq", "bias", "n")}


def add_metrics(acc, prediction, target):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    acc["abs"] += float(np.abs(error).sum())
    acc["sq"] += float(np.square(error).sum())
    acc["bias"] += float(error.sum())
    acc["n"] += int(error.size)


def finish_metrics(acc):
    n = int(acc["n"])
    return {
        "n_pixels": n,
        "mae_mm_h": acc["abs"] / n if n else None,
        "rmse_mm_h": float(np.sqrt(acc["sq"] / n)) if n else None,
        "bias_mm_h": acc["bias"] / n if n else None,
    }


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    mapping = load_bilinear_map()
    device = TRAIN.get_device()
    models = load_models(device)
    test_times = TRAIN.read_manifest("test")
    fields = TRAIN.Fields(mapping)
    rng = np.random.default_rng(int(CONFIG["seed"]) + 73)

    histogram_floor = 1e-3
    hist_edges = np.concatenate(([histogram_floor], np.geomspace(histogram_floor, 300.0, 121)[1:], [np.inf]))
    hist_values = {name: np.zeros(len(hist_edges) - 1, dtype=np.int64) for name in ("RADKLIM truth", "ERA5 model", "Synthetic model")}
    below_floor_count = {name: 0 for name in hist_values}
    overall = {name: metric_accumulator() for name in METHODS}
    by_stratum = {label: {name: metric_accumulator() for name in METHODS} for label, _ in STRATA}
    stratum_pixels = {label: 0 for label, _ in STRATA}
    zero_counts = {name: 0 for name in ("RADKLIM truth", "ERA5 model", "Synthetic model")}
    reservoir = {name: [] for name in ("RADKLIM truth", "ERA5 raw", "Synthetic raw", "ERA5 model", "Synthetic model")}
    candidates = {label: [] for label, _ in STRATA}
    total_common = 0
    total_target_observed = 0
    sample_per_hour = 256

    print(f"Diagnostics device={device}; test hours={len(test_times)}", flush=True)
    for position, timestamp in enumerate(test_times):
        target, era, coarse, common = fields.get(timestamp)
        synthetic, synthetic_valid = bilinear_to_radklim(coarse, mapping)
        common &= synthetic_valid
        common &= np.isfinite(target) & np.isfinite(era) & np.isfinite(synthetic)
        if not common.any():
            continue
        era_prediction = TRAIN.infer_tiled(models["era5"][0], "era5", era, coarse, mapping, device)
        synthetic_prediction = TRAIN.infer_tiled(models["synthetic"][0], "synthetic", era, coarse, mapping, device)
        truth = target[common]
        values = {
            "ERA5 raw": era[common],
            "Synthetic raw": synthetic[common],
            "ERA5 model": era_prediction[common],
            "Synthetic model": synthetic_prediction[common],
        }
        total_common += int(common.sum())
        total_target_observed += int((np.isfinite(target) & fields.overlap).sum())
        for name, prediction in values.items():
            add_metrics(overall[name], prediction, truth)
        for label, mask_fn in STRATA:
            mask = mask_fn(truth)
            stratum_pixels[label] += int(mask.sum())
            for name, prediction in values.items():
                add_metrics(by_stratum[label][name], prediction[mask], truth[mask])

        fields_for_distribution = {"RADKLIM truth": truth, "ERA5 model": values["ERA5 model"], "Synthetic model": values["Synthetic model"]}
        for name, arr in fields_for_distribution.items():
            positive = arr[arr > 0]
            below_floor_count[name] += int(np.count_nonzero(positive < histogram_floor))
            hist_values[name] += np.histogram(positive[positive >= histogram_floor], bins=hist_edges)[0]
            zero_counts[name] += int(np.count_nonzero(arr == 0))

        indices = np.flatnonzero(common)
        take = min(sample_per_hour, len(indices))
        chosen = rng.choice(indices, size=take, replace=False)
        flattened = {
            "RADKLIM truth": target.ravel()[chosen],
            "ERA5 raw": era.ravel()[chosen],
            "Synthetic raw": synthetic.ravel()[chosen],
            "ERA5 model": era_prediction.ravel()[chosen],
            "Synthetic model": synthetic_prediction.ravel()[chosen],
        }
        for name, arr in flattened.items():
            reservoir[name].append(np.asarray(arr, dtype=np.float32))

        fractions = {label: float(mask_fn(truth).mean()) for label, mask_fn in STRATA}
        iso = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        for label, _ in STRATA:
            candidates[label].append((fractions[label], int(timestamp), iso))
        if (position + 1) % 25 == 0 or position + 1 == len(test_times):
            print(f"diagnostics {position+1}/{len(test_times)} hours", flush=True)

    fields.close()
    overall_result = {name: finish_metrics(value) for name, value in overall.items()}
    stratified_result = []
    for label, _ in STRATA:
        row = {"intensity_bin": label, "n_pixels": stratum_pixels[label]}
        for name in METHODS:
            row[name] = finish_metrics(by_stratum[label][name])
        stratified_result.append(row)

    distribution_summary = {}
    for name, count in zero_counts.items():
        n_pixels = int(hist_values[name].sum()) + count + below_floor_count[name]
        distribution_summary[name] = {
            "n_pixels": n_pixels,
            "zero_count": int(count),
            "zero_fraction": count / max(n_pixels, 1),
            "positive_below_0.001_mm_h_count": int(below_floor_count[name]),
        }
    summary = {
        "test_period_utc": [datetime.fromtimestamp(test_times[0], timezone.utc).isoformat(), datetime.fromtimestamp(test_times[-1], timezone.utc).isoformat()],
        "test_hours": len(test_times),
        "common_valid_pixels": total_common,
        "observed_target_pixels_in_overlap": total_target_observed,
        "common_support_fraction_of_observed_target": total_common / total_target_observed if total_target_observed else None,
        "pixel_rule": "only pixels with finite RADKLIM target, ERA5 input, and synthetic input; same pixels for all methods",
        "strata_rule": "dry = target exactly 0; 0–0.1 = 0 < target < 0.1; 0.1–1 = 0.1 ≤ target < 1; 1–5 = 1 ≤ target < 5; ≥5 = target ≥ 5 mm/h",
        "overall_input_and_prediction_metrics": overall_result,
        "prediction_distribution": distribution_summary,
        "stratified_metrics": stratified_result,
        "typical_hour_selection": "for each target-intensity bin, choose the test hour with the largest fraction of common-support pixels in that bin; ties go to the earliest hour; selected hours are kept distinct",
        "device": str(device),
        "checkpoint_epoch": {name: int(value[1]["epoch"]) for name, value in models.items()},
    }
    (OUTPUT / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    with (OUTPUT / "overall_metrics.csv").open("w", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=("method", "n_pixels", "mae_mm_h", "rmse_mm_h", "bias_mm_h"))
        writer.writeheader()
        for name, metric in overall_result.items():
            writer.writerow({"method": name, **metric})
    with (OUTPUT / "intensity_stratified_metrics.csv").open("w", newline="") as stream:
        import csv
        names = ("ERA5 raw", "Synthetic raw", "ERA5 model", "Synthetic model")
        columns = ("intensity_bin", "n_pixels", *(f"{n}_{m}" for n in names for m in ("mae_mm_h", "rmse_mm_h", "bias_mm_h")))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in stratified_result:
            output_row = {"intensity_bin": row["intensity_bin"], "n_pixels": row["n_pixels"]}
            for name in names:
                for metric in ("mae_mm_h", "rmse_mm_h", "bias_mm_h"):
                    output_row[f"{name}_{metric}"] = row[name][metric]
            writer.writerow(output_row)

    distribution_methods = ("RADKLIM truth", "ERA5 model", "Synthetic model")
    with (OUTPUT / "prediction_distribution.csv").open("w", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=("bin_lower_mm_h", "bin_upper_mm_h", *(f"{n}_count" for n in distribution_methods)))
        writer.writeheader()
        counts_by_name = {name: hist_values[name] for name in distribution_methods}
        for special, lower, upper in (("exact_zero", 0.0, 0.0), ("positive_below_0.001", 0.0, histogram_floor)):
            row = {"bin_lower_mm_h": lower, "bin_upper_mm_h": upper}
            for name in distribution_methods:
                row[f"{name}_count"] = int(zero_counts[name] if special == "exact_zero" else below_floor_count[name])
            writer.writerow(row)
        for index in range(len(hist_edges) - 1):
            row = {"bin_lower_mm_h": float(hist_edges[index]), "bin_upper_mm_h": float(hist_edges[index + 1])}
            for name in distribution_methods:
                row[f"{name}_count"] = int(counts_by_name[name][index])
            writer.writerow(row)

    colors = {"RADKLIM truth": COLORS["RADKLIM truth"], "ERA5 model": COLORS["ERA5 model"], "Synthetic model": COLORS["Synthetic model"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    scatter_sample_count = len(reservoir["RADKLIM truth"]) * sample_per_hour
    scatter_sample_count = min(scatter_sample_count, sum(len(values) for values in reservoir["RADKLIM truth"]))
    for ax, method in zip(axes, ("ERA5 raw", "Synthetic raw")):
        x = np.concatenate(reservoir[method]); y = np.concatenate(reservoir["RADKLIM truth"])
        hb = ax.hexbin(x, y, gridsize=65, bins="log", mincnt=1, cmap="viridis", linewidths=0)
        lim = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)), 0.1)
        ax.plot([0, lim], [0, lim], color="#777777", lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
        ax.set(xlim=(-0.001, lim), ylim=(-0.001, lim), xlabel=f"{method} (mm/h)", ylabel="RADKLIM target (mm/h)")
        metric = overall_result[method]
        ax.text(0.04, 0.96, f"MAE {metric['mae_mm_h']:.3f}\nRMSE {metric['rmse_mm_h']:.3f}\nbias {metric['bias_mm_h']:.3f} mm/h",
                transform=ax.transAxes, va="top", ha="left", fontsize=7, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.colorbar(hb, ax=axes, label="sampled pixel count per hexagon (log scale)", shrink=0.82)
    fig.suptitle(f"Raw inputs versus RADKLIM target — {scatter_sample_count:,} seeded sample pairs; metrics use all common pixels")
    save_figure(fig, OUTPUT / "raw_input_vs_target")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    finite_upper = np.where(np.isfinite(hist_edges[1:]), hist_edges[1:], hist_edges[-2] * 10)
    centers = np.sqrt(hist_edges[:-1] * finite_upper)
    for name in distribution_methods:
        counts = hist_values[name]
        fraction = counts / max(int(hist_values[name].sum()), 1)
        ax.step(centers, fraction, where="mid", lw=1.8, color=colors[name], label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel="Hourly precipitation (mm/h; logarithmic bins)", ylabel="Fraction of all common-support pixels per bin",
           title=f"Prediction and truth distributions — {total_common:,} common test pixel-hours")
    text = "\n".join(f"{name} exact-zero: {distribution_summary[name]['zero_fraction']:.1%}" for name in distribution_methods)
    ax.text(0.98, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"})
    ax.legend(loc="lower left")
    save_figure(fig, OUTPUT / "prediction_distribution")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
    metric_keys = (("mae_mm_h", "MAE (mm/h)"), ("rmse_mm_h", "RMSE (mm/h)"), ("bias_mm_h", "Bias (mm/h)"))
    x = np.arange(len(STRATA)); width = 0.19
    for ax, (key, title) in zip(axes, metric_keys):
        for series_idx, method in enumerate(METHODS):
            values = [row[method][key] if row[method]["n_pixels"] else np.nan for row in stratified_result]
            ax.bar(x + (series_idx - 1.5) * width, values, width=width, color=COLORS[method], label=method)
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{label.replace(' mm/h', '')}\nn={stratum_pixels[label]:,}" for label, _ in STRATA], rotation=25, ha="right")
        ax.set_title(title); ax.set_ylabel(title)
    axes[0].legend(fontsize=6, ncol=2, loc="upper left")
    fig.suptitle("Errors stratified by RADKLIM target intensity — shared test pixels")
    save_figure(fig, OUTPUT / "intensity_stratified_metrics")
    plt.close(fig)

    chosen = []
    used = set()
    for label, _ in STRATA:
        ranked = sorted(candidates[label], key=lambda item: (-item[0], item[1]))
        pick = next((item for item in ranked if item[0] > 0 and item[1] not in used), None)
        if pick is not None:
            used.add(pick[1])
            chosen.append({"intensity_bin": label, "bin_fraction": pick[0], "timestamp": pick[1], "time_utc": pick[2]})
    (OUTPUT / "typical_hours.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + "\n")
    coordinate_file = sorted(ALIGNED_DIR.glob("aligned_*.nc"))[0]
    with Dataset(coordinate_file) as ds:
        latitude = np.asarray(ds.variables["lat"][::4, ::4], dtype=np.float32)
        longitude = np.asarray(ds.variables["lon"][::4, ::4], dtype=np.float32)
    for item in chosen:
        timestamp = item["timestamp"]
        # Fields was closed after the streaming pass; use a fresh reader for each selected scene.
        one = TRAIN.Fields(mapping)
        target, era, coarse, common = one.get(timestamp)
        synthetic, valid = bilinear_to_radklim(coarse, mapping)
        common &= valid
        era_model = TRAIN.infer_tiled(models["era5"][0], "era5", era, coarse, mapping, device)
        synthetic_model = TRAIN.infer_tiled(models["synthetic"][0], "synthetic", era, coarse, mapping, device)
        one.close()
        layers = (target, era, synthetic, era_model, synthetic_model)
        finite = np.concatenate([layer[common & np.isfinite(layer)][::16] for layer in layers if np.any(common & np.isfinite(layer))])
        vmax = max(float(np.quantile(finite, 0.995)), 0.1)
        fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.3), constrained_layout=True, sharex=True, sharey=True)
        image = None
        for ax, layer, label in zip(axes, layers, MAP_LABELS):
            sampled = layer[::4, ::4]
            mask = common[::4, ::4] & np.isfinite(sampled)
            image = ax.contourf(longitude, latitude, np.ma.masked_where(~mask, sampled),
                                levels=np.linspace(0.0, vmax, 21), cmap="YlGnBu", vmin=0, vmax=vmax, extend="max")
            ax.set(xlim=(2, 16), ylim=(46.5, 55), title=label, xlabel="Longitude (°E)")
            ax.grid(alpha=0.15, lw=0.4)
        axes[0].set_ylabel("Latitude (°N)")
        fig.colorbar(image, ax=axes, label="Hourly precipitation (mm/h)", shrink=0.82)
        utc = datetime.fromtimestamp(timestamp, timezone.utc)
        fig.suptitle(f"{item['intensity_bin']} representative hour — {utc:%Y-%m-%d %H:%M UTC}; common-support fraction={item['bin_fraction']:.1%}; shared color max={vmax:.2f} mm/h")
        safe = item["intensity_bin"].replace("≥", "ge").replace("–", "-").replace("<", "lt").replace(" ", "_").replace("/", "-")
        save_figure(fig, OUTPUT / f"typical_{safe}")
        plt.close(fig)

    print(f"Diagnostics written to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
