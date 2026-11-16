"""Boundary tests for the small public STRENGTH prototype helpers."""

import unittest

from h4_leakage_guard import H4RandomHoldoutSplitter
from strength_ns_engine import StrengthIntentNSEngine


class HoldoutSplitterTests(unittest.TestCase):
    def test_seeded_partition_is_repeatable_and_complete(self):
        first = H4RandomHoldoutSplitter(total_networks=10, seed=7)
        second = H4RandomHoldoutSplitter(total_networks=10, seed=7)
        train_a, holdout_a = first.generate_random_holdout_split()
        train_b, holdout_b = second.generate_random_holdout_split()

        self.assertEqual(train_a.tolist(), train_b.tolist())
        self.assertEqual(holdout_a.tolist(), holdout_b.tolist())
        self.assertEqual(set(train_a) | set(holdout_a), set(range(10)))
        self.assertFalse(set(train_a) & set(holdout_a))

    def test_rejects_invalid_population_and_seed(self):
        for count in (True, 1, 2.5):
            with self.subTest(count=count):
                with self.assertRaises((TypeError, ValueError)):
                    H4RandomHoldoutSplitter(total_networks=count)
        for seed in (True, -1, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    H4RandomHoldoutSplitter(seed=seed)

    def test_rejects_ratios_that_are_invalid_or_empty_a_partition(self):
        splitter = H4RandomHoldoutSplitter(total_networks=10)
        for ratio in (True, "0.8", 0, 1, float("nan"), float("inf"), 0.01):
            with self.subTest(ratio=ratio):
                with self.assertRaises((TypeError, ValueError)):
                    splitter.generate_random_holdout_split(train_ratio=ratio)


class ScalarBoundTests(unittest.TestCase):
    def setUp(self):
        self.engine = StrengthIntentNSEngine()

    def test_valid_result_states_arithmetic_and_authentication_boundary(self):
        result = self.engine.audit_state(velocity=12500.5, strain=1420.0)
        self.assertEqual(result["payload"]["status"], "BOUNDS_CHECKED_ONLY")
        self.assertEqual(result["payload"]["arithmetic"], "Python float (binary64)")
        self.assertEqual(result["authentication"], "none")
        self.assertEqual(len(result["integrity_digest_sha256"]), 64)
        int(result["integrity_digest_sha256"], 16)

    def test_rejects_non_numeric_boolean_non_finite_negative_and_overflow(self):
        invalid_values = (True, "1", None, float("nan"), float("inf"), -1, 10**400)
        for name in ("velocity", "strain"):
            for value in invalid_values:
                with self.subTest(name=name, value=repr(value)):
                    arguments = {"velocity": 1.0, "strain": 1.0}
                    arguments[name] = value
                    with self.assertRaises((TypeError, ValueError)):
                        self.engine.audit_state(**arguments)

    def test_rejects_both_configured_bounds(self):
        with self.assertRaisesRegex(ValueError, "Strain"):
            self.engine.audit_state(velocity=1.0, strain=self.engine.strain_threshold + 1)
        with self.assertRaisesRegex(ValueError, "Contracted value"):
            self.engine.audit_state(velocity=1_000_000.0, strain=1.0)


if __name__ == "__main__":
    unittest.main()

