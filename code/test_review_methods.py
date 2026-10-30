"""Small deterministic checks for the revised statistical implementations."""

import unittest

import numpy as np
import pandas as pd

from survey_jackknife import allocation, capture_contrast, interval, replicate_multipliers
from final_submission_analysis import fractional_capture, ordered_scores
from w8_measurement_invariance import Group, ITEMS, Model


class ReviewMethodTests(unittest.TestCase):
    def test_replicates_preserve_full_sample_model_tie_order(self):
        rng = np.random.default_rng(20261030)
        scores = rng.integers(0, 5, size=100).astype(float)
        counts = np.ones(100)
        counts[::7] = 0
        weights = np.linspace(1, 2, 100) * counts
        y = (np.arange(100) % 3 == 0).astype(float)
        full_order = np.argsort(-scores)
        ranks = np.empty(100)
        ranks[full_order] = -np.arange(100)
        for population in (False, True):
            self.assertEqual(
                allocation(y, scores, weights, population=population, counts=counts),
                allocation(y, ranks, weights, population=population, counts=counts),
            )

    def test_tie_order_does_not_depend_on_input_row_order(self):
        values, ids = np.array([.7, .6, .7]), np.array([3, 2, 1])
        scores = ordered_scores(values, ids)
        perm = np.array([2, 0, 1])
        np.testing.assert_array_equal(
            scores[perm], ordered_scores(values[perm], ids[perm]))
        self.assertEqual(list(ids[np.argsort(-scores)]), [2, 1, 3])

    def test_fractional_ties_split_capacity_equally(self):
        y = np.array([1., 0., 1., 0.])
        scores, weights = np.ones(4), np.array([1., 2., 3., 4.])
        self.assertAlmostEqual(fractional_capture(y, scores, weights, np.ones(4), .5), .5)
        counts = np.array([0., 2., 1., 1.])
        self.assertAlmostEqual(fractional_capture(y, scores, weights * counts, counts, .5), .5)

    def test_jackknife_allows_three_psus_and_matches_total_variance(self):
        frame = pd.DataFrame({"SDMVSTRA": [1, 1, 2, 2, 2], "SDMVPSU": [1, 2, 1, 2, 3]})
        multipliers, factors, df = replicate_multipliers(frame)
        y = np.array([0., 2., 4., 6., 8.])
        result = interval(y.sum(), multipliers @ y, factors, df)
        self.assertEqual(df, 3)
        self.assertAlmostEqual(result["se"] ** 2, 16.)
        self.assertTrue(np.all(multipliers >= 0))

    def test_identical_models_have_zero_contrast(self):
        frame = pd.DataFrame({"SDMVSTRA": [1] * 6, "SDMVPSU": [1, 1, 2, 2, 3, 3]})
        p = np.linspace(.1, .9, 6)
        result = capture_contrast(frame, np.ones(6), p, p, np.ones(6), share=.5)
        self.assertEqual((result["point"], result["lo"], result["hi"]), (0., 0., 0.))

    def test_population_capacity_is_exact_and_rank_preserving(self):
        y, p, w = np.array([1, 0, 1]), np.array([.9, .7, .1]), np.array([4., 3., 3.])
        original = allocation(y, p, w, population=True)
        transformed = allocation(y, p ** .4, w, population=True)
        self.assertAlmostEqual(original["reviewed_population"], .1)
        self.assertEqual(original, transformed)

    def test_irt_parameter_counts(self):
        frame = pd.DataFrame(np.zeros((3, 9)), columns=ITEMS)
        groups = [Group(str(index), frame, False) for index in range(6)]
        self.assertEqual(Model(groups, "configural").n_parameters(), 216)
        self.assertEqual(Model(groups, "metric").n_parameters(), 176)
        self.assertEqual(Model(groups, "scalar").n_parameters(), 46)
        self.assertEqual(Model(groups, "partial", ("DPQ010",)).n_parameters(), 61)


if __name__ == "__main__":
    unittest.main()
