"""真实分类入口验证原始目标保留、暂停不填零及归档单位。"""

import json
import unittest
from fixtures import hashes, write_json
from classification_fixture import ClassificationFixture, compressed_rows


class ClassificationIntegrationTests(ClassificationFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config["target_id"] = "next-market-day-up-v1"
        self.config["model_params"].update(objective="binary", metric="binary_logloss")
        write_json(self.config_path, self.config)

    def test_classification_keeps_source_targets_and_uses_training_prior(self):
        dataset = json.loads((self.p1/"dataset.json").read_text(encoding="utf-8"))
        before = hashes(self.module/dataset["cache_directory"])
        process, result, run = self.validate()
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr+(run/"issues.csv").read_text(encoding="utf-8"))
        self.assertEqual(before, hashes(self.module/dataset["cache_directory"]))
        self.assertEqual(result["task"], "binary_classification")
        self.assertFalse(result["holdout_evaluation_performed"])
        predicted, scored = compressed_rows(run/"predictions.csv.gz"), compressed_rows(run/"predictions.scored.csv.gz")
        self.assertEqual(len(predicted), 2)
        self.assertNotIn("y_pred", predicted[0])
        self.assertTrue(all(0 <= float(r["p_up"]) <= 1 for r in predicted))
        paused = next(r for r in scored if r["score_eligible"] == "False")
        self.assertEqual((paused["actual_class"], paused["actual_return"], paused["unscorable_reason"]), ("", "", "target_suspended"))
        flat = next(r for r in scored if r["score_eligible"] == "True")
        self.assertEqual((float(flat["actual_return"]), flat["actual_class"]), (0, "0"))
        prior = result["training_prior"]
        self.assertEqual(prior["up_probability"], prior["up_count"]/prior["n"])
        self.assertEqual(json.loads((run/"model.json").read_text(encoding="utf-8"))["prediction_unit"], "up_probability")
        self.assertTrue((run/"calibration_bins.csv").exists())

    def test_binary_cannot_claim_the_original_return_target_id(self):
        self.config["target_id"] = "next-market-day-quote-return-v1"
        write_json(self.config_path, self.config)
        process, result, run = self.validate()
        self.assertEqual(process.returncode, 2)
        self.assertIsNone(result)
        self.assertIsNone(run)

    def test_classification_runs_and_fingerprints_belong_to_method(self):
        process, result, run = self.validate()
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        self.assertTrue(run.is_relative_to(self.method_root/"runs"))
        identity = json.loads((run/"inputs.json").read_text(encoding="utf-8"))["identity"]
        hashes = identity["calculation_code_sha256"]
        self.assertIn("methods/next-day-up/src/next_day_up/domain.py", hashes)
        self.assertIn("experiments/evaluation_domain.py", hashes)
        self.assertEqual(result["provenance"]["source_target_id"], "next-market-day-quote-return-v1")

    def test_common_regression_entry_rejects_binary_configuration(self):
        write_json(self.module/"configs/p3-validation.json", self.config)
        process, run = self.command("validate", "--fold", "F01")
        self.assertEqual(process.returncode, 2)
        self.assertIsNone(run)

    def test_classification_entry_rejects_regression_configuration(self):
        self.config["model_params"].update(objective="regression", metric="rmse")
        write_json(self.config_path, self.config)
        process, result, run = self.validate()
        self.assertEqual(process.returncode, 2)
        self.assertIsNone(run)
