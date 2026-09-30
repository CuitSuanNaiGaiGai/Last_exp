"""Train two paired U-Nets and evaluate on shared held-out support."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import random
import time

import numpy as np
import torch
from netCDF4 import Dataset
from tqdm import tqdm

from common import (
    ALIGNED_DIR, CONFIG, DatasetCache, OUTPUT_DIR, bilinear_to_radklim,
    bilinear_valid_mask, load_bilinear_map, log_precipitation, read_field,
)
from model import SmallUNet


PATCH = int(CONFIG["model"]["patch_size"])
BASE_SEED = int(CONFIG["seed"])


def read_manifest(name: str) -> list[int]:
    path = OUTPUT_DIR / "splits" / f"{name}_timestamps.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Run 03_make_splits.py first; missing {path}")
    return [int(line) for line in path.read_text().splitlines() if line.strip()]


def integral_patch_counts(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64)
    r2, c2 = rows + PATCH, cols + PATCH
    return integral[r2[:, None], c2[None, :]] - integral[rows[:, None], c2[None, :]] - integral[r2[:, None], cols[None, :]] + integral[rows[:, None], cols[None, :]]


def candidate_grid(shape: tuple[int, int]):
    h, w = shape
    rows = np.unique(np.append(np.arange(0, h - PATCH + 1, 32), h - PATCH)).astype(np.int32)
    cols = np.unique(np.append(np.arange(0, w - PATCH + 1, 32), w - PATCH)).astype(np.int32)
    return rows, cols


class Fields:
    def __init__(self, mapping):
        self.aligned_cache = DatasetCache(maximum_open=3)
        self.synthetic_cache = DatasetCache(maximum_open=3)
        self.mapping = mapping
        self.overlap = mapping["overlap_mask"].astype(bool)
        self.index = {}
        for path in sorted(ALIGNED_DIR.glob("aligned_*.nc")):
            with Dataset(path) as ds:
                for i, timestamp in enumerate(np.asarray(ds.variables["time"][:], dtype=np.int64)):
                    self.index[int(timestamp)] = (path, i)

    def get(self, timestamp: int):
        path, index = self.index[int(timestamp)]
        aligned = self.aligned_cache.get(path)
        target = read_field(aligned.variables["precipitation_radklim"], index)
        era = read_field(aligned.variables["precipitation_era5"], index)
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / path.name.replace("aligned_", "synthetic_lr_")
        synthetic_ds = self.synthetic_cache.get(synthetic_path)
        coarse = read_field(synthetic_ds.variables["precipitation"], index)
        era[~self.overlap] = np.nan
        synthetic_available = bilinear_valid_mask(coarse, self.mapping)
        support = np.isfinite(target) & np.isfinite(era) & synthetic_available & self.overlap
        return target, era, coarse, support

    def close(self):
        self.aligned_cache.close()
        self.synthetic_cache.close()


def choose_patch(support: np.ndarray, rows: np.ndarray, cols: np.ndarray, seed: int):
    counts = integral_patch_counts(support, rows, cols)
    threshold = int(math.ceil(float(CONFIG["model"]["minimum_patch_support"]) * PATCH * PATCH))
    eligible = np.argwhere(counts >= threshold)
    if not len(eligible):
        return None
    rng = np.random.default_rng(seed)
    index = int(rng.integers(len(eligible)))
    ri, ci = eligible[index]
    return int(rows[ri]), int(cols[ci])


def make_patch(field: np.ndarray, support: np.ndarray, row: int, col: int, size: int = PATCH):
    end_y, end_x = min(row + size, field.shape[0]), min(col + size, field.shape[1])
    data = field[row:end_y, col:end_x]
    mask = support[row:end_y, col:end_x]
    h, w = data.shape
    x_patch = make_input_patch(field, row, col, size)
    target_mask = np.zeros((size, size), dtype=np.float32)
    target_mask[:h, :w] = mask.astype(np.float32)
    return x_patch, target_mask


def make_source_patch(mode: str, era: np.ndarray, coarse: np.ndarray, mapping, row: int, col: int, size: int = PATCH):
    if mode == "era5":
        return make_input_patch(era, row, col, size)
    y_end = min(row + size, mapping["overlap_mask"].shape[0])
    x_end = min(col + size, mapping["overlap_mask"].shape[1])
    mapped, _ = bilinear_to_radklim(coarse, mapping, (slice(row, y_end), slice(col, x_end)))
    return make_input_patch(mapped, 0, 0, size)


def make_context_tile(mode: str, era: np.ndarray, coarse: np.ndarray, mapping, top: int, left: int, core_size: int, halo: int):
    """Build a coarse-on-demand input tile with an explicit context halo."""
    height, width = era.shape
    tile_size = core_size + 2 * halo
    y0, x0 = max(0, top - halo), max(0, left - halo)
    y1, x1 = min(height, top + core_size + halo), min(width, left + core_size + halo)
    if mode == "era5":
        values = era[y0:y1, x0:x1]
    else:
        values, _ = bilinear_to_radklim(coarse, mapping, (slice(y0, y1), slice(x0, x1)))
    h, w = values.shape
    tile = np.zeros((2, tile_size, tile_size), dtype=np.float32)
    tile[0, :h, :w] = np.where(np.isfinite(values), log_precipitation(values), 0.0)
    tile[1, :h, :w] = np.isfinite(values).astype(np.float32)
    core_y, core_x = top - y0, left - x0
    core_h, core_w = min(core_size, height - top), min(core_size, width - left)
    return tile, (core_y, core_x, core_h, core_w)


def make_input_patch(field: np.ndarray, row: int, col: int, size: int = PATCH):
    end_y, end_x = min(row + size, field.shape[0]), min(col + size, field.shape[1])
    data = field[row:end_y, col:end_x]
    h, w = data.shape
    val = np.zeros((size, size), dtype=np.float32)
    avail = np.zeros_like(val)
    val[:h, :w] = np.where(np.isfinite(data), log_precipitation(data), 0.0)
    avail[:h, :w] = np.isfinite(data).astype(np.float32)
    return np.stack((val, avail))


def make_target_mm_patch(field: np.ndarray, row: int, col: int, size: int = PATCH):
    """Return a padded target patch in physical mm/h for validation bins and CSI."""
    end_y, end_x = min(row + size, field.shape[0]), min(col + size, field.shape[1])
    data = field[row:end_y, col:end_x]
    h, w = data.shape
    patch = np.zeros((size, size), dtype=np.float32)
    patch[:h, :w] = np.where(np.isfinite(data), np.maximum(data, 0.0), 0.0)
    return patch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_l1(prediction, target, mask):
    errors = torch.abs(prediction[:, 0] - target)
    return (errors * mask).sum() / mask.sum().clamp_min(1.0)


def new_validation_accumulator():
    return {
        "overall_abs_error_sum": 0.0,
        "valid_pixels": 0,
        "target_bins": {
            "target_lt_0.1_mm_h": {"abs_error_sum": 0.0, "pixels": 0},
            "target_0.1_to_lt_1_mm_h": {"abs_error_sum": 0.0, "pixels": 0},
            "target_ge_1_mm_h": {"abs_error_sum": 0.0, "pixels": 0},
        },
        "prediction_chunks": [],
        "prediction_nonfinite_pixels": 0,
        "csi": {
            "0.1_mm_h": {"hits": 0, "false_alarms": 0, "misses": 0},
            "1_mm_h": {"hits": 0, "false_alarms": 0, "misses": 0},
        },
    }


def update_validation_accumulator(accumulator, prediction_log, target_log, target_mm, mask):
    """Accumulate validation loss groups, physical prediction values, and CSI counts."""
    valid = mask > 0.0
    absolute_error = torch.abs(prediction_log - target_log)
    valid_count = int(valid.sum().item())
    accumulator["overall_abs_error_sum"] += float((absolute_error * mask).sum().item())
    accumulator["valid_pixels"] += valid_count

    target_conditions = {
        "target_lt_0.1_mm_h": target_mm < 0.1,
        "target_0.1_to_lt_1_mm_h": (target_mm >= 0.1) & (target_mm < 1.0),
        "target_ge_1_mm_h": target_mm >= 1.0,
    }
    for name, condition in target_conditions.items():
        group_mask = valid & condition
        group = accumulator["target_bins"][name]
        group["pixels"] += int(group_mask.sum().item())
        group["abs_error_sum"] += float(absolute_error[group_mask].sum().item())

    prediction_mm = torch.clamp(torch.expm1(prediction_log), min=0.0)
    finite_prediction = torch.isfinite(prediction_mm)
    usable = valid & finite_prediction
    accumulator["prediction_nonfinite_pixels"] += int((valid & ~finite_prediction).sum().item())
    if usable.any():
        values = prediction_mm[usable].detach().cpu().numpy().astype(np.float32, copy=True)
        accumulator["prediction_chunks"].append(values)

    for name, threshold in (("0.1_mm_h", 0.1), ("1_mm_h", 1.0)):
        predicted_event = prediction_mm >= threshold
        observed_event = target_mm >= threshold
        event_mask = usable
        counts = accumulator["csi"][name]
        counts["hits"] += int((event_mask & predicted_event & observed_event).sum().item())
        counts["false_alarms"] += int((event_mask & predicted_event & ~observed_event).sum().item())
        counts["misses"] += int((event_mask & ~predicted_event & observed_event).sum().item())


def finish_validation_accumulator(accumulator):
    valid_pixels = accumulator["valid_pixels"]
    target_bins = {}
    for name, values in accumulator["target_bins"].items():
        pixels = values["pixels"]
        target_bins[name] = {
            "masked_log1p_l1": values["abs_error_sum"] / pixels if pixels else None,
            "valid_pixels": pixels,
        }

    if accumulator["prediction_chunks"]:
        predictions = np.concatenate(accumulator["prediction_chunks"])
        prediction_stats = {
            "valid_pixels": int(predictions.size),
            "nonfinite_pixels": int(accumulator["prediction_nonfinite_pixels"]),
            "mean_mm_h": float(np.mean(predictions, dtype=np.float64)),
            "p95_mm_h": float(np.percentile(predictions, 95)),
            "p99_mm_h": float(np.percentile(predictions, 99)),
            "max_mm_h": float(np.max(predictions)),
        }
    else:
        prediction_stats = {
            "valid_pixels": 0,
            "nonfinite_pixels": int(accumulator["prediction_nonfinite_pixels"]),
            "mean_mm_h": None,
            "p95_mm_h": None,
            "p99_mm_h": None,
            "max_mm_h": None,
        }

    csi = {}
    for name, counts in accumulator["csi"].items():
        denominator = counts["hits"] + counts["false_alarms"] + counts["misses"]
        csi[name] = {
            "csi": counts["hits"] / denominator if denominator else None,
            "hits": counts["hits"],
            "false_alarms": counts["false_alarms"],
            "misses": counts["misses"],
        }
    return {
        "overall_masked_log1p_l1": (
            accumulator["overall_abs_error_sum"] / valid_pixels if valid_pixels else None
        ),
        "overall_valid_pixels": valid_pixels,
        "target_bins_mm_h": target_bins,
        "prediction_mm_h": prediction_stats,
        "csi": csi,
    }


def evaluate_validation_batch(model, batch_x, batch_y, batch_mask, batch_target_mm, device, accumulator):
    """Evaluate one validation batch and update its per-epoch diagnostics."""
    xb = torch.from_numpy(np.stack(batch_x)).to(device)
    yb = torch.from_numpy(np.stack(batch_y)).to(device)
    mb = torch.from_numpy(np.stack(batch_mask)).to(device)
    target_mm = torch.from_numpy(np.stack(batch_target_mm)).to(device)
    prediction = model(xb)
    loss = masked_l1(prediction, yb, mb)
    update_validation_accumulator(accumulator, prediction[:, 0], yb, target_mm, mb)
    return float(loss.detach().cpu()), float(mb.sum().detach().cpu()), len(batch_x)


def run_training(mode: str, train_times, val_times, mapping, device, artifact_root=OUTPUT_DIR, epochs=None):
    seed_everything(BASE_SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = SmallUNet(base_channels=CONFIG["model"]["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["model"]["learning_rate"]))
    fields = Fields(mapping)
    h, w = mapping["overlap_mask"].shape
    rows, cols = candidate_grid((h, w))
    log_path = artifact_root / "training" / f"{mode}_training.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_root / "models" / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = None
    patience = int(CONFIG["model"].get("early_stopping_patience", 5))
    epochs_without_improvement = 0
    stopped_early = False
    epoch_logs = []
    batch_size = int(CONFIG["model"]["batch_size"])
    started = time.time()

    epoch_count = int(epochs if epochs is not None else CONFIG["model"]["epochs"])

    def persist_training_log(status):
        best_val_json = best_val if np.isfinite(best_val) else None
        payload = {
            "status": status,
            "mode": mode,
            "device": str(device),
            "seed": BASE_SEED,
            "model": CONFIG["model"],
            "max_epochs": epoch_count,
            "epochs_requested": epoch_count,
            "epochs_executed": len(epoch_logs),
            "early_stopping": {
                "monitor": "validation.overall_masked_log1p_l1",
                "patience": patience,
                "epochs_without_improvement": epochs_without_improvement,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_json,
                "stopped_early": stopped_early,
            },
            "train_hours": len(train_times),
            "validation_hours": len(val_times),
            "elapsed_seconds": time.time() - started,
            "validation_metric_definitions": {
                "overall_and_target_bin_l1": "masked mean absolute error in log1p(mm/h) space",
                "target_bins": "raw RADKLIM target intensity in physical mm/h",
                "prediction_distribution": "inverse-transformed mm/h, clipped below at zero, over finite common validation support",
                "csi": "physical mm/h event thresholds over finite common validation support",
            },
            "epochs": epoch_logs,
        }
        temporary_path = log_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary_path.replace(log_path)
        return payload

    persist_training_log("running")
    for epoch in range(epoch_count):
        model.train()
        order_rng = np.random.default_rng(BASE_SEED + 1000 + epoch)
        order = order_rng.permutation(len(train_times))
        batch_x, batch_y, batch_mask = [], [], []
        loss_sum, valid_pixel_count, batch_count, eligible_hours = 0.0, 0, 0, 0
        epoch_start = time.time()
        train_progress = tqdm(
            enumerate(order),
            total=len(order),
            desc="{} train epoch {}/{}".format(mode, epoch + 1, epoch_count),
            unit="hour",
            dynamic_ncols=True,
        )
        for position, idx in train_progress:
            timestamp = train_times[int(idx)]
            target, era, coarse, support = fields.get(timestamp)
            patch_coord = choose_patch(support, rows, cols, BASE_SEED + epoch * 100_000 + timestamp)
            if patch_coord is None:
                continue
            row, col = patch_coord
            x_patch = make_source_patch(mode, era, coarse, mapping, row, col)
            mask_patch = support[row:row + PATCH, col:col + PATCH].astype(np.float32)
            y_patch, _ = make_patch(target, support, row, col)
            batch_x.append(x_patch)
            batch_y.append(y_patch[0])
            batch_mask.append(mask_patch)
            eligible_hours += 1
            if len(batch_x) == batch_size or position + 1 == len(order):
                xb = torch.from_numpy(np.stack(batch_x)).to(device)
                yb = torch.from_numpy(np.stack(batch_y)).to(device)
                mb = torch.from_numpy(np.stack(batch_mask)).to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = masked_l1(pred, yb, mb)
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.detach().cpu())
                batch_pixels = float(mb.sum().detach().cpu())
                loss_sum += batch_loss * batch_pixels
                valid_pixel_count += int(batch_pixels)
                batch_count += 1
                train_progress.set_postfix(loss="{:.4f}".format(batch_loss), batches=batch_count, refresh=False)
                batch_x.clear(); batch_y.clear(); batch_mask.clear()
        if batch_x:
            xb = torch.from_numpy(np.stack(batch_x)).to(device)
            yb = torch.from_numpy(np.stack(batch_y)).to(device)
            mb = torch.from_numpy(np.stack(batch_mask)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_l1(model(xb), yb, mb)
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach().cpu())
            batch_pixels = float(mb.sum().detach().cpu())
            loss_sum += batch_loss * batch_pixels
            valid_pixel_count += int(batch_pixels)
            batch_count += 1
        train_loss = loss_sum / max(valid_pixel_count, 1)

        model.eval()
        val_sum, val_pixels, val_count = 0.0, 0, 0
        batch_x, batch_y, batch_mask = [], [], []
        batch_target_mm = []
        validation_accumulator = new_validation_accumulator()
        with torch.no_grad():
            val_progress = tqdm(
                enumerate(val_times),
                total=len(val_times),
                desc="{} validation epoch {}/{}".format(mode, epoch + 1, epoch_count),
                unit="hour",
                dynamic_ncols=True,
            )
            for position, timestamp in val_progress:
                target, era, coarse, support = fields.get(timestamp)
                patch_coord = choose_patch(support, rows, cols, BASE_SEED + 500_000 + timestamp)
                if patch_coord is None:
                    continue
                row, col = patch_coord
                x_patch = make_source_patch(mode, era, coarse, mapping, row, col)
                mask_patch = support[row:row + PATCH, col:col + PATCH].astype(np.float32)
                y_patch, _ = make_patch(target, support, row, col)
                batch_x.append(x_patch); batch_y.append(y_patch[0]); batch_mask.append(mask_patch)
                batch_target_mm.append(make_target_mm_patch(target, row, col))
                if len(batch_x) == batch_size or position + 1 == len(val_times):
                    batch_loss, batch_pixels, batch_size_actual = evaluate_validation_batch(
                        model, batch_x, batch_y, batch_mask, batch_target_mm, device, validation_accumulator
                    )
                    val_sum += batch_loss * batch_pixels
                    val_pixels += int(batch_pixels); val_count += batch_size_actual
                    val_progress.set_postfix(loss="{:.4f}".format(batch_loss), patches=val_count, refresh=False)
                    batch_x.clear(); batch_y.clear(); batch_mask.clear(); batch_target_mm.clear()
            if batch_x:
                batch_loss, batch_pixels, batch_size_actual = evaluate_validation_batch(
                    model, batch_x, batch_y, batch_mask, batch_target_mm, device, validation_accumulator
                )
                val_sum += batch_loss * batch_pixels
                val_pixels += int(batch_pixels); val_count += batch_size_actual
        val_loss = val_sum / max(val_pixels, 1)
        validation_metrics = finish_validation_accumulator(validation_accumulator)
        if val_pixels and not np.isclose(val_loss, validation_metrics["overall_masked_log1p_l1"], rtol=1e-5, atol=1e-7):
            raise RuntimeError("Validation loss disagrees with accumulated masked L1 metrics")

        improved = bool(np.isfinite(val_loss) and val_loss < best_val)
        if improved:
            best_val = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "mode": mode, "epoch": epoch + 1,
                        "validation_loss": val_loss, "seed": BASE_SEED, "config": CONFIG["model"]}, checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        stopped_early = epochs_without_improvement >= patience and epoch + 1 < epoch_count

        epoch_log = {"epoch": epoch + 1, "train_masked_log1p_l1": train_loss, "val_masked_log1p_l1": val_loss,
                     "validation": validation_metrics,
                     "best_epoch_so_far": best_epoch,
                     "epochs_without_improvement": epochs_without_improvement,
                     "improved_validation": improved,
                     "train_batches": batch_count, "train_eligible_hours": eligible_hours,
                     "val_patches": val_count, "elapsed_seconds": time.time() - epoch_start}
        epoch_logs.append(epoch_log)
        persist_training_log("early_stopped" if stopped_early else "running")
        tqdm.write("[{}] epoch {}/{} train={:.6f} val_L1(log1p)={:.6f}; val_bin_L1(log1p): <0.1={} 0.1-<1={} >=1={}; pred(mm/h): mean={} P95={} P99={} max={}; CSI@0.1={} CSI@1={}".format(
            mode, epoch + 1, epoch_count, train_loss, val_loss,
            validation_metrics["target_bins_mm_h"]["target_lt_0.1_mm_h"]["masked_log1p_l1"],
            validation_metrics["target_bins_mm_h"]["target_0.1_to_lt_1_mm_h"]["masked_log1p_l1"],
            validation_metrics["target_bins_mm_h"]["target_ge_1_mm_h"]["masked_log1p_l1"],
            validation_metrics["prediction_mm_h"]["mean_mm_h"],
            validation_metrics["prediction_mm_h"]["p95_mm_h"],
            validation_metrics["prediction_mm_h"]["p99_mm_h"],
            validation_metrics["prediction_mm_h"]["max_mm_h"],
            validation_metrics["csi"]["0.1_mm_h"]["csi"],
            validation_metrics["csi"]["1_mm_h"]["csi"],
        ))
        if stopped_early:
            tqdm.write("[{}] early stopping after {} epochs without validation improvement (best epoch {})".format(
                mode, patience, best_epoch
            ))
            break

    fields.close()
    payload = persist_training_log("early_stopped" if stopped_early else "complete")
    return payload


def infer_tiled(model, mode: str, era: np.ndarray, coarse: np.ndarray, mapping, device):
    h, w = era.shape
    prediction = np.full((h, w), np.nan, dtype=np.float32)
    model.eval()
    tile_batch_size = int(CONFIG["model"].get("inference_batch_size", 8))
    halo = int(CONFIG["model"].get("inference_halo", 32))
    core_size = int(CONFIG["model"].get("inference_core_size", 256))
    with torch.no_grad():
        tiles = [(top, left) for top in range(0, h, core_size) for left in range(0, w, core_size)]
        for start in range(0, len(tiles), tile_batch_size):
            batch_tiles = tiles[start:start + tile_batch_size]
            prepared = [make_context_tile(mode, era, coarse, mapping, top, left, core_size, halo) for top, left in batch_tiles]
            xb = torch.from_numpy(np.stack([item[0] for item in prepared])).to(device)
            outputs = model(xb)[:, 0].detach().cpu().numpy()
            for output, (top, left), (core_y, core_x, core_h, core_w) in zip(outputs, batch_tiles, [item[1] for item in prepared]):
                prediction[top:top + core_h, left:left + core_w] = output[core_y:core_y + core_h, core_x:core_x + core_w]
    return np.maximum(np.expm1(prediction), 0.0, dtype=np.float32)


def evaluate(mode: str, test_times, mapping, device, artifact_root=OUTPUT_DIR):
    checkpoint = torch.load(artifact_root / "models" / mode / "best.pt", map_location=device, weights_only=False)
    model = SmallUNet(base_channels=CONFIG["model"]["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    fields = Fields(mapping)
    totals = {"abs_error": 0.0, "squared_error": 0.0, "error": 0.0, "pixels": 0, "target_pixels": 0, "hits": 0, "false_alarms": 0, "misses": 0}
    rows = []
    threshold = float(CONFIG["model"]["csi_threshold_mm_per_hour"])
    started = time.time()
    test_progress = tqdm(
        enumerate(test_times),
        total=len(test_times),
        desc="{} test evaluation".format(mode),
        unit="hour",
        dynamic_ncols=True,
    )
    for position, timestamp in test_progress:
        target, era, coarse, common = fields.get(timestamp)
        pred = infer_tiled(model, mode, era, coarse, mapping, device)
        valid = common & np.isfinite(target) & np.isfinite(pred)
        target_valid = np.isfinite(target) & fields.overlap
        truth = target[valid].astype(np.float64)
        estimate = pred[valid].astype(np.float64)
        errors = estimate - truth
        pixels = int(len(truth))
        hits = int(np.count_nonzero((estimate >= threshold) & (truth >= threshold)))
        fa = int(np.count_nonzero((estimate >= threshold) & (truth < threshold)))
        miss = int(np.count_nonzero((estimate < threshold) & (truth >= threshold)))
        totals["abs_error"] += float(np.abs(errors).sum())
        totals["squared_error"] += float(np.square(errors).sum())
        totals["error"] += float(errors.sum())
        totals["pixels"] += pixels; totals["hits"] += hits; totals["false_alarms"] += fa; totals["misses"] += miss
        totals["target_pixels"] += int(target_valid.sum())
        rows.append({"time_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), "valid_pixels": pixels,
                     "common_support_fraction": float(pixels / target_valid.sum()) if target_valid.any() else None,
                     "mae_mm": float(np.abs(errors).mean()) if pixels else None,
                     "rmse_mm": float(np.sqrt(np.square(errors).mean())) if pixels else None,
                     "bias_mm": float(errors.mean()) if pixels else None, "hits": hits, "false_alarms": fa, "misses": miss})
        running_mae = totals["abs_error"] / totals["pixels"] if totals["pixels"] else 0.0
        test_progress.set_postfix(mae="{:.4f}".format(running_mae), refresh=False)
    fields.close()
    n = totals["pixels"]
    denom = totals["hits"] + totals["false_alarms"] + totals["misses"]
    summary = {"mode": mode, "timestamps": len(test_times), "valid_pixels": n,
               "target_observed_pixels_within_overlap": totals["target_pixels"],
               "common_support_fraction_of_observed_target": n / totals["target_pixels"] if totals["target_pixels"] else None,
               "mae_mm": totals["abs_error"] / n if n else None,
               "rmse_mm": math.sqrt(totals["squared_error"] / n) if n else None,
               "bias_mm": totals["error"] / n if n else None,
               "csi_threshold_mm_per_hour": threshold,
               "csi": totals["hits"] / denom if denom else None,
               "hits": totals["hits"], "false_alarms": totals["false_alarms"], "misses": totals["misses"],
               "elapsed_seconds": time.time() - started, "checkpoint_epoch": checkpoint["epoch"]}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("era5", "synthetic", "all"), default="all")
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing best.pt checkpoints only")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short end-to-end pipeline check; outputs are isolated under outputs/smoke_test/")
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    train_times, val_times, test_times = (read_manifest(name) for name in ("train", "val", "test"))
    artifact_root = OUTPUT_DIR / "smoke_test" if args.smoke_test else OUTPUT_DIR
    if args.smoke_test:
        train_times, val_times, test_times = train_times[:2], val_times[:2], test_times[:1]
        if args.skip_training:
            raise ValueError("--smoke-test checks training; do not combine it with --skip-training")
    epochs = 1 if args.smoke_test else int(CONFIG["model"]["epochs"])
    mapping = load_bilinear_map()
    device = get_device()
    print(f"Training/evaluation device: {device}", flush=True)
    modes = ("era5", "synthetic") if args.mode == "all" else (args.mode,)
    training = {}
    if not args.skip_training:
        for mode in modes:
            training[mode] = run_training(mode, train_times, val_times, mapping, device, artifact_root, epochs)
    summaries, all_rows = {}, {}
    for mode in modes:
        summaries[mode], all_rows[mode] = evaluate(mode, test_times, mapping, device, artifact_root)
    metrics_dir = artifact_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "comparison.json").write_text(json.dumps({"training": training, "test": summaries,
        "paired_protocol": {"seed": BASE_SEED, "train_hours": len(train_times), "validation_hours": len(val_times),
                            "test_hours": len(test_times), "patch_size": PATCH, "batch_size": CONFIG["model"]["batch_size"],
                            "max_epochs": epochs,
                            "early_stopping_patience": CONFIG["model"].get("early_stopping_patience", 5),
                            "smoke_test": bool(args.smoke_test),
                            "inference_core_size": CONFIG["model"].get("inference_core_size", 256),
                            "inference_halo": CONFIG["model"].get("inference_halo", 112),
                            "shared_common_support": True}}, indent=2) + "\n")
    with (metrics_dir / "per_hour_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("mode", "time_utc", "valid_pixels", "common_support_fraction", "mae_mm", "rmse_mm", "bias_mm", "hits", "false_alarms", "misses"))
        writer.writeheader()
        for mode in modes:
            for row in all_rows[mode]:
                writer.writerow({"mode": mode, **row})
    synthetic_files = sorted((OUTPUT_DIR / "synthetic_lr").glob("synthetic_lr_*.nc"))
    if len(synthetic_files) != 24:
        raise RuntimeError(f"Expected 24 synthetic monthly files, found {len(synthetic_files)}")
    synthetic_hours = 0
    for aligned_path in sorted(ALIGNED_DIR.glob("aligned_*.nc")):
        synthetic_path = OUTPUT_DIR / "synthetic_lr" / aligned_path.name.replace("aligned_", "synthetic_lr_")
        with Dataset(aligned_path) as aligned, Dataset(synthetic_path) as synthetic:
            if not np.array_equal(aligned.variables["time"][:], synthetic.variables["time"][:]):
                raise ValueError(f"Synthetic time axis differs from aligned source: {synthetic_path.name}")
            if synthetic.variables["precipitation"].shape[1:] != (35, 57):
                raise ValueError(f"Unexpected synthetic ERA5 grid shape in {synthetic_path.name}")
            synthetic_hours += len(synthetic.dimensions["time"])
    if len(summaries) == 2 and summaries["era5"]["valid_pixels"] != summaries["synthetic"]["valid_pixels"]:
        raise ValueError("Paired test methods were not evaluated on the same number of pixels")
    split_summary = json.loads((OUTPUT_DIR / "splits" / "split_summary.json").read_text())
    run_summary = {
        "status": "smoke_test_passed" if args.smoke_test else "complete",
        "qc_report": str(OUTPUT_DIR / "qc" / "aligned_qc.json"),
        "synthetic_files": len(synthetic_files),
        "synthetic_hours": synthetic_hours,
        "split_summary": split_summary,
        "device": str(device),
        "shared_model_seed": BASE_SEED,
        "model_configuration": {
            **CONFIG["model"],
            "epochs_executed_by_mode": {
                mode: training[mode]["epochs_executed"] for mode in training
            },
        },
        "test_metrics": summaries,
        "artifact_root": str(artifact_root),
        "comparison_json": str(metrics_dir / "comparison.json"),
        "per_hour_metrics_csv": str(metrics_dir / "per_hour_metrics.csv"),
        "checkpoints": {mode: str(artifact_root / "models" / mode / "best.pt") for mode in modes},
        "training_logs": {mode: str(artifact_root / "training" / f"{mode}_training.json") for mode in modes},
    }
    (artifact_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
