"""用已知误差和不同日期权重验证评分含义。"""

import json
import math
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from daily_return.samples.domain import ROW_DTYPE
from daily_return.experiments.validation_domain import ValidationSettings
from daily_return.experiments.evaluation_domain import ValidationError, FoldDescription, validate_fold_roles, fit_volatility_boundaries
from daily_return.experiments.evaluation import compare_predictions, evaluate_fold
from experiments.test_split_rules import small_population


class EvaluationTests(unittest.TestCase):
    def test_exact_errors_zero_reference_and_three_way_direction(self):
        result = compare_predictions(np.array([.02,-.01,0]), np.array([.01,.01,0]))
        self.assertAlmostEqual(result["lightgbm"]["rmse"], math.sqrt(.0005/3))
        self.assertAlmostEqual(result["lightgbm"]["mae"], .01)
        self.assertAlmostEqual(result["lightgbm"]["mean_error"], .01/3)
        self.assertAlmostEqual(result["lightgbm"]["direction_accuracy"], 2/3)
        self.assertAlmostEqual(result["zero"]["direction_accuracy"], 1/3)
        self.assertAlmostEqual(result["delta_rmse"],0)

    def test_zero_error_baseline_empty_groups_and_nonfinite_predictions(self):
        self.assertIsNone(compare_predictions([0,0],[0,.1])["relative_rmse_improvement"])
        self.assertIsNone(compare_predictions([],[])["lightgbm"]["rmse"])
        with self.assertRaises(ValidationError): compare_predictions([0],[np.nan])

    def test_daily_equal_weights_and_missing_groups_keep_coverage(self):
        rows=np.zeros(5,dtype=ROW_DTYPE)
        rows["target_idx"]=[0,1,1,1,1]
        description=FoldDescription("F01",{"first_target_date":"2025-01-02","last_target_date":"2025-01-03"},(),(),0,
                                    ("2025-01-02","2025-01-03"),(),rows,np.array([True,True,True,True,False]),{}, {})
        result=evaluate_fold(description,np.array([1,3,3,3,np.nan]),np.zeros(5),np.array([.1,.2,.4,np.nan,np.nan]),
                             np.array([True,False,True,False,False]),(.2,.3,.5))
        self.assertAlmostEqual(result["overall"]["lightgbm"]["rmse"],math.sqrt(7))
        self.assertAlmostEqual(result["day_equal"]["lightgbm"]["rmse"],math.sqrt(5))
        self.assertEqual(result["day_equal"]["lightgbm"]["mae"],2)
        self.assertEqual((result["prediction_count"],result["score_count"],result["unscorable_count"]),(5,4,1))
        for grouping in ("training_volatility","trend_availability","industry"):
            groups=[g for g in result["groups"] if g["grouping"]==grouping]
            self.assertEqual(sum(g["prediction_count"] for g in groups),5)
            self.assertEqual(sum(g["lightgbm"]["n"] for g in groups),4)
        json.dumps(result,allow_nan=False)

    def test_volatility_thresholds_are_derived_from_supplied_training_values(self):
        self.assertEqual(fit_volatility_boundaries(np.array([0,1,2,3,np.nan])),(.75,1.5,2.25))
        self.assertEqual(fit_volatility_boundaries(np.full(3,np.nan)),())

    def test_fold_boundary_and_late_labels_cannot_enter_training(self):
        p,_=small_population()
        window={"kind":"validation","first_target_date":"2026-08-17","last_target_date":"2026-08-19","fit_cutoff_date":"2026-08-14"}
        roles={"train":np.array([0,1],dtype=np.int64),"predict":np.array([2,3,4],dtype=np.int64),"score":np.array([2,3,4],dtype=np.int64)}
        self.assertTrue(validate_fold_roles(p.rows,roles,window,p.calendar,"2026-08-25").all())
        p.rows[1]["label_available_idx"]=3
        with self.assertRaisesRegex(ValidationError,"成熟"):
            validate_fold_roles(p.rows,roles,window,p.calendar,"2026-08-25")
        with self.assertRaisesRegex(ValidationError,"留出"):
            validate_fold_roles(p.rows,roles,{**window,"kind":"holdout"},p.calendar,"2026-08-25")

    def test_configuration_has_user_requested_cap_without_duplicate_alias(self):
        path=Path(__file__).resolve().parents[2]/"configs/p3-validation.json"
        config=json.loads(path.read_text(encoding="utf-8"))
        parsed=ValidationSettings.from_mapping(config)
        self.assertEqual((parsed.rounds,parsed.patience,parsed.feature_count),(10000,200,49))
        config["model_params"]["num_iterations"]=5
        with self.assertRaises(ValidationError):ValidationSettings.from_mapping(config)
