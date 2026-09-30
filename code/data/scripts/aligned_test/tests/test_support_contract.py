import sys
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    bilinear_to_radklim,
    bilinear_valid_mask,
    circular_neighborhood_kernel,
    common_input_support,
    require_finite_on_support,
)

try:
    import torch
except ImportError:
    torch = None

GAP = None
if torch is not None:
    GAP_SPEC = importlib.util.spec_from_file_location(
        "gap_decomposition_test", SCRIPT_DIR / "06_decompose_cross_product_gap.py"
    )
    GAP = importlib.util.module_from_spec(GAP_SPEC)
    GAP_SPEC.loader.exec_module(GAP)


class SupportContractTests(unittest.TestCase):
    def test_common_support_intersects_all_three_inputs_and_overlap(self):
        target = np.ones((2, 3), dtype=np.float32)
        era = np.ones((2, 3), dtype=np.float32)
        synthetic_available = np.ones((2, 3), dtype=bool)
        overlap = np.ones((2, 3), dtype=bool)
        target[0, 1] = np.nan
        era[1, 1] = np.nan
        synthetic_available[0, 2] = False
        overlap[1, 2] = False

        support = common_input_support(target, era, synthetic_available, overlap)

        expected = np.array([[True, False, False], [True, False, False]])
        np.testing.assert_array_equal(support, expected)

    def test_prediction_must_be_finite_on_shared_support(self):
        support = np.array([[True, False], [True, True]])
        prediction = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "1 non-finite values on common support"):
            require_finite_on_support("ERA5 prediction", prediction, support)

    def test_prediction_nonfinite_outside_support_is_ignored(self):
        support = np.array([[True, False], [False, False]])
        prediction = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)

        require_finite_on_support("ERA5 prediction", prediction, support)

    def test_disk_radius_uses_euclidean_cell_distance(self):
        self.assertEqual(int(circular_neighborhood_kernel(1).sum()), 5)
        self.assertEqual(int(circular_neighborhood_kernel(2).sum()), 13)

    def test_fast_and_materialized_synthetic_support_are_identical(self):
        coarse = np.array(
            [[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        ones = np.ones((2, 2), dtype=np.float32)
        mapping = {
            "era_lat_index": np.array([[0, 0], [1, 1]], dtype=np.int32),
            "era_lon_index": np.array([[0, 1], [0, 1]], dtype=np.int32),
            "overlap_mask": np.ones((2, 2), dtype=np.uint8),
            "weight_00": ones.copy(),
            "weight_01": ones.copy(),
            "weight_10": ones.copy(),
            "weight_11": ones.copy(),
        }
        mapping["weight_11"][0, 1] = np.nan

        mapped, materialized_support = bilinear_to_radklim(coarse, mapping)
        fast_support = bilinear_valid_mask(coarse, mapping)

        np.testing.assert_array_equal(fast_support, materialized_support)
        np.testing.assert_array_equal(fast_support, np.isfinite(mapped))

    @unittest.skipIf(GAP is None, "PyTorch is unavailable in this runtime")
    def test_circular_event_counts_use_shared_support(self):
        target = np.zeros((7, 7), dtype=np.float32)
        target[3, 3] = 1.0
        support = np.ones((7, 7), dtype=bool)
        estimates = {name: target.copy() for name in GAP.ESTIMATE_NAMES}

        counts, channels, disk_cells = GAP.circle_event_counts(
            target, estimates, support, 1, torch.device("cpu")
        )

        self.assertEqual(disk_cells, 5)
        self.assertEqual(counts[channels[("support", None)], 3, 3], 5.0)
        self.assertEqual(counts[channels[("RADKLIM truth", "1")], 3, 3], 1.0)


if __name__ == "__main__":
    unittest.main()
