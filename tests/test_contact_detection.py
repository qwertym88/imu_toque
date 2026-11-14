"""Tests for temporal contact inference and Y13 sample isolation."""

from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from contact_detection import fit_temporal_hmm, validate_pairing


class ContactDetectionTests(unittest.TestCase):
    def test_persistence_model_recovers_temporal_states_and_respects_invalid_gap(self):
        rng = np.random.default_rng(7)
        count, dim = 1200, 3
        truth = np.zeros(count, dtype=np.int8)
        truth[200:440] = 1
        truth[440:720] = 0
        truth[720:1000] = 1
        features = rng.normal(size=(count, dim))
        features[truth == 1] += 2.5
        valid = np.ones(count, dtype=bool)
        valid[580:620] = False
        result = fit_temporal_hmm(features, valid, sample_rate=100, expected_state_duration_s=0.5)
        predicted = result["state"]
        # Lower-dynamic-activity state is the contact-like state, state id zero.
        expected_contact_state = truth == 0
        self.assertGreater(np.mean((predicted[valid] == 0) == expected_contact_state[valid]), 0.95)
        self.assertTrue(np.isnan(result["probability_contact"][~valid]).all())
        self.assertTrue((predicted[~valid] == -1).all())

    def test_y13_pairing_rejects_legacy_xsens_and_unrelated_force_plate(self):
        fit_metadata = {
            "source_xlsx": str(ROOT / "data/y13_MVN_3_8_01.xlsx"),
            "source_metadata": {"Suit Label": "20261106_Y13_Xu_Hanyang"},
            "source_xlsx_sha256": hashlib.sha256(
                (ROOT / "data/y13_MVN_3_8_01.xlsx").read_bytes()
            ).hexdigest(),
            "model_sha256": "model-hash",
        }
        result = validate_pairing(ROOT / "data/y13_MVN_3_8_01.xlsx", fit_metadata)
        self.assertEqual(result["sample_id"], "y13")
        self.assertFalse(result["force_plate_used"])
        self.assertIsNone(result["force_plate_path"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_pairing(ROOT / "data/xsense_raw.xlsx", fit_metadata)
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "y13_stairs.xlsx"
            other.touch()
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_pairing(other, fit_metadata)

    def test_inference_needs_enough_valid_time(self):
        with self.assertRaisesRegex(ValueError, "Not enough valid"):
            fit_temporal_hmm(np.zeros((20, 3)), np.ones(20, dtype=bool), sample_rate=60)


if __name__ == "__main__":
    unittest.main()
