"""最终轮数、时点隔离以及没有验证Dataset的真实训练。"""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import lightgbm as lgb

from next_day_up.domain import freeze_classification_plan
from daily_return.experiments import validate_holdout_roles
from daily_return.experiments import ValidationError
from next_day_up.lightgbm_model import LightGBMTrainer
from experiments.test_lightgbm import Telemetry
from experiments.test_split_rules import small_population


def completed_records():
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/p3-classification.json").read_text(encoding="utf-8"))
    provenance = {"preparation_id":"prepared", "split_id":"split", "holdout_identity_sha256":"a"*64}
    return [{"fold":f"F{i:02}", "configuration":copy.deepcopy(config), "provenance":dict(provenance), "best_iteration":best}
            for i,best in enumerate((6,26,1,1,67,4,120), 1)]


class HoldoutRuleTests(unittest.TestCase):
    def test_final_rounds_are_median_not_best_fold_and_no_early_stopping(self):
        settings, config = freeze_classification_plan(completed_records())
        self.assertEqual((settings.rounds, settings.patience), (6,0))
        self.assertEqual(config["class_threshold"], .5)
        self.assertEqual(config["development_best_iterations"], [6,26,1,1,67,4,120])

    def test_missing_duplicate_and_mismatched_development_records_are_rejected(self):
        for variant in ("missing", "duplicate", "parameter", "identity", "iteration"):
            records = completed_records()
            if variant == "missing": records.pop()
            elif variant == "duplicate": records[1]["fold"] = "F01"
            elif variant == "parameter": records[1]["configuration"]["model_params"]["num_leaves"] = 63
            elif variant == "identity": records[1]["provenance"]["holdout_identity_sha256"] = "b"*64
            else: records[1]["best_iteration"] = 0
            with self.subTest(variant=variant), self.assertRaises(ValidationError):
                freeze_classification_plan(records)

    def test_holdout_roles_reject_boundary_or_immature_training_labels(self):
        p,_ = small_population()
        window = {"kind":"holdout", "first_target_date":"2026-08-17", "last_target_date":"2026-08-19", "fit_cutoff_date":"2026-08-14"}
        roles = {"train":np.array([0,1],dtype=np.int64), "predict":np.array([2,3,4],dtype=np.int64), "score":np.array([2,3,4],dtype=np.int64)}
        self.assertTrue(validate_holdout_roles(p.rows, roles, window, p.calendar, "2026-08-17").all())
        with self.assertRaises(ValidationError):
            validate_holdout_roles(p.rows, roles, {**window,"fit_cutoff_date":"2026-08-17"}, p.calendar, "2026-08-17")
        p.rows[1]["label_available_idx"] = 3
        with self.assertRaises(ValidationError):
            validate_holdout_roles(p.rows, roles, window, p.calendar, "2026-08-17")

    def test_real_fixed_training_does_not_receive_validation_or_early_stopping(self):
        records = completed_records()
        for record in records:
            record["configuration"]["model_params"].update(num_threads=1, min_data_in_leaf=5)
        settings,_ = freeze_classification_plan(records)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(240,49)).astype(np.float32)
        y = (x[:,0] > 0).astype(np.uint8)
        names = tuple(f"f{i}" for i in range(49))
        trainer = LightGBMTrainer(Telemetry(), lambda e:None)
        with patch("next_day_up.lightgbm_model.lgb.train", wraps=lgb.train) as fit:
            model = trainer.fit_fixed(x, y, names, settings)
        self.assertNotIn("valid_sets", fit.call_args.kwargs)
        self.assertEqual(model.summary["rounds_run"], 6)
        self.assertFalse(model.summary["early_stopping_enabled"])
        self.assertFalse(model.summary["validation_dataset_supplied"])
        prediction = model.predict(x)
        self.assertGreater(float(np.std(prediction)), 0)
        self.assertEqual(trainer.check_reload(model.export_text(), names, x, prediction)["maximum_absolute_difference"], 0)
