"""用手算和异常样本核对分类含义，避免平盘/概率/原始涨幅混用。"""

import json
import math
import sys
import unittest
from pathlib import Path
import numpy as np

from next_day_up.domain import up_labels, ClassificationSettings
from daily_return.samples.domain import ROW_DTYPE
from next_day_up.evaluation import binary_metrics, probability_bins, evaluate_classification_fold
from daily_return.experiments import ValidationError, FoldDescription


class ClassificationTests(unittest.TestCase):
    def test_positive_negative_flat_and_missing_have_distinct_meaning(self):
        np.testing.assert_array_equal(up_labels([.03, .0001, 0, -.02]), [1, 1, 0, 0])
        with self.assertRaises(ValidationError):
            up_labels([.02, np.nan])

    def test_hand_computed_confusion_losses_and_tie_threshold(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([.8, .7, .5, .1])
        m = binary_metrics(labels, scores)
        self.assertEqual([m[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 1])
        self.assertEqual((m["accuracy"], m["balanced_accuracy"]), (.5, .5))
        self.assertAlmostEqual(m["logloss"], -math.log(.8*.3*.5*.9)/4)
        self.assertAlmostEqual(m["brier"], (.04+.49+.25+.01)/4)
        self.assertAlmostEqual(m["auc"], .75)

    def test_auc_ties_single_class_empty_and_raw_regression(self):
        self.assertEqual(binary_metrics([1, 0, 1, 0], [.5]*4)["auc"], .5)
        single = binary_metrics([1, 1], [.4, .6])
        self.assertIsNone(single["auc"])
        self.assertIsNone(single["balanced_accuracy"])
        self.assertIsNone(binary_metrics([], [])["accuracy"])
        raw = binary_metrics([1, 0, 1], [.03, -.02, 0], probability=False, threshold=0)
        self.assertAlmostEqual(raw["accuracy"], 2/3)
        self.assertEqual(raw["auc"], 1)
        self.assertIsNone(raw["logloss"])
        self.assertIsNone(raw["brier"])
        for bad in ([1.1], [np.nan]):
            with self.assertRaises(ValidationError):
                binary_metrics([1], bad)
        json.dumps(binary_metrics([0, 1], [1, 0]), allow_nan=False)

    def test_fixed_calibration_bins_keep_empty_and_include_one(self):
        bins = probability_bins(np.array([0, 1, 0, 1]), np.array([0, .1, .9, 1]))
        self.assertEqual([b["n"] for b in bins], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(bins[-1]["observed_up_rate"], .5)

    def test_paused_predictions_prior_and_daily_single_class_coverage(self):
        rows = np.zeros(5, dtype=ROW_DTYPE)
        rows["target_idx"] = [0, 1, 1, 1, 1]
        d = FoldDescription("F01", {"first_target_date":"2025-01-02", "last_target_date":"2025-01-03"}, (), (), 0,
                            ("2025-01-02", "2025-01-03"), (), rows, np.array([True, True, True, True, False]), {}, {})
        m = evaluate_classification_fold(d, np.array([.01, -.01, 0, .02, np.nan]), np.array([.9, .1, .2, .8, .6]),
                                         np.array([.1, .2, .4, np.nan, np.nan]), np.array([1, 0, 1, 0, 0], dtype=bool), (.2, .3, .5), .75)
        self.assertEqual((m["prediction_count"], m["score_count"], m["unscorable_count"]), (5, 4, 1))
        self.assertEqual(m["actual_flat_count"], 1)
        self.assertEqual(m["overall"]["lightgbm"]["accuracy"], 1)
        self.assertEqual(m["overall"]["training_prior"]["accuracy"], .5)
        self.assertEqual(m["day_equal"]["lightgbm"]["auc_days"], 1)
        self.assertEqual(m["nonflat_diagnostic"]["lightgbm"]["n"], 3)
        for grouping in ("training_volatility", "trend_availability", "industry"):
            self.assertEqual(sum(r["lightgbm"]["n"] for r in m["groups"] if r["grouping"] == grouping), 4)
        json.dumps(m, allow_nan=False)

    def test_classification_configs_change_only_patience(self):
        root = Path(__file__).resolve().parents[1]/"configs"
        hundred = json.loads((root/"p3-classification-patience-100.json").read_text(encoding="utf-8"))
        two = json.loads((root/"p3-classification.json").read_text(encoding="utf-8"))
        self.assertEqual(ClassificationSettings.from_mapping(two).target_id, "next-market-day-up-v1")
        self.assertEqual(hundred, {**two, "early_stopping_rounds":100})
        two["model_params"]["metric"] = "rmse"
        with self.assertRaises(ValidationError):
            ClassificationSettings.from_mapping(two)
