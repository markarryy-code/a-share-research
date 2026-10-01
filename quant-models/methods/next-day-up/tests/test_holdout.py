"""用临时P1/P2走留出入口，核对不可提前读取目标及使用登记。"""

import copy
import json
import sys
import unittest
from unittest.mock import patch
from fixtures import SRC, write_json, read_csv, write_csv, hashes
from classification_fixture import ClassificationFixture

sys.path.insert(0, str(SRC))
from daily_return.serialization import file_hash
from next_day_up.development_source import load_classification_plan
from next_day_up.domain import UP_TARGET_SPEC, up_labels
from daily_return.experiments.validation_source import PreparedFoldSource
from daily_return.experiments import ValidationError, FoldInputSpec
from daily_return.samples.prepared_reader import NumpyPreparedReader


class HoldoutIntegrationTests(ClassificationFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.make_selection()

    def make_selection(self):
        split = json.loads((self.p2/"splits.json").read_text(encoding="utf-8"))
        config = copy.deepcopy(self.config)
        config.update(split_run=self.p2.relative_to(self.module).as_posix(), target_id="next-market-day-up-v1", num_boost_round=10000)
        config["model_params"].update(objective="binary", metric="binary_logloss")
        references = {}
        for i,best in enumerate((6,26,1,1,67,4,120), 1):
            fold = f"F{i:02}"
            directory = self.module/"runs"/f"development-{fold}"
            write_json(directory/"configuration.json", config)
            write_json(directory/"model.json", {"best_iteration":best, "completion_state":"fold-validated"})
            write_json(directory/"experiment.json", {"execution_status":"completed"})
            write_json(directory/"validation.json", {"fold":fold, "state":"fold-validated", "task":"binary_classification", "inputs_unchanged":True,
                       "best_iteration":best, "provenance":{k:split[k] for k in ("preparation_id","split_id","holdout_identity_sha256")},
                       "artifacts":{name:{"sha256":file_hash(directory/name)} for name in ("configuration.json","model.json")}})
            references[fold] = directory.relative_to(self.module).as_posix()
        self.selection = self.module/"runs/development-selection.json"
        write_json(self.selection, {"classification_runs":references})

    def holdout(self):
        process,run = self.method_command("evaluate-holdout", "--development-selection", self.selection.relative_to(self.module).as_posix())
        result = json.loads((run/"evaluation.json").read_text(encoding="utf-8")) if run else None
        return process,result,run

    def test_public_fixed_model_precedes_scoring_and_second_evaluation_is_blocked(self):
        p1 = json.loads((self.p1/"dataset.json").read_text(encoding="utf-8"))
        before = hashes(self.module/p1["cache_directory"])
        process,result,run = self.holdout()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr+(run/"issues.csv").read_text(encoding="utf-8"))
        self.assertEqual(result["state"], "holdout-evaluated")
        self.assertEqual(result["max_rounds"], 6)
        self.assertFalse(result["early_stopping_enabled"])
        self.assertEqual(before, hashes(self.module/p1["cache_directory"]))
        model = json.loads((run/"model.json").read_text(encoding="utf-8"))
        self.assertEqual(model["dataset_rows"]["validation"], 0)
        usage = json.loads((run/"holdout_usage.json").read_text(encoding="utf-8"))["attempts"][-1]
        self.assertTrue(usage["exposure_started"])
        self.assertEqual(usage["predictions_sha256"], file_hash(run/"predictions.csv.gz"))
        self.assertEqual(usage["model_sha256"], file_hash(run/"model_artifacts/model.txt"))
        self.assertLess(usage["reserved_at_utc"], usage["exposure_started_at_utc"])
        process,second,second_run = self.holdout()
        self.assertEqual(process.returncode,2)
        self.assertFalse(second["model_trained"])
        self.assertIn("已经开始过", (second_run/"issues.csv").read_text(encoding="utf-8"))

    def test_source_rejects_holdout_as_validation_or_targets_before_sealing(self):
        settings,_,inputs = load_classification_plan(self.module,self.selection)
        source = PreparedFoldSource(self.module,NumpyPreparedReader(self.module),FoldInputSpec(settings.split_run, settings.feature_set, settings.feature_count, UP_TARGET_SPEC["source_target_id"]),"holdout",self.selection,file_hash(self.selection),kind="holdout",input_records=inputs)
        try:
            source.describe()
            with self.assertRaises(ValidationError): source.validation_data()
            with self.assertRaises(ValidationError): source.prediction_targets()
            train_x,train_y = source.training_data()
            self.assertEqual(len(train_x),len(train_y))
        finally:
            source.close()

    def run_in_process(self):
        from next_day_up.bootstrap import evaluate_classification_holdout
        return evaluate_classification_holdout(self.module,self.selection)

    def test_failure_before_exposure_is_recorded_and_may_retry(self):
        with patch("next_day_up.lightgbm_model.LightGBMTrainer.fit_fixed", side_effect=RuntimeError("fixture failure")):
            self.assertEqual(self.run_in_process(),1)
        path = next((self.module/"runs/holdout-usage").glob("*.json"))
        first = json.loads(path.read_text(encoding="utf-8"))["attempts"]
        self.assertFalse(first[0]["exposure_started"])
        process,result,run = self.holdout()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))["attempts"]),2)

    def test_failure_after_exposure_cannot_reset_the_holdout(self):
        with patch("daily_return.experiments.validation_source.PreparedFoldSource.prediction_targets", side_effect=ValueError("fixture score failure")):
            self.assertEqual(self.run_in_process(),1)
        path = next((self.module/"runs/holdout-usage").glob("*.json"))
        self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["attempts"][0]["exposure_started"])
        process,result,run = self.holdout()
        self.assertEqual(process.returncode,2)
        self.assertFalse(result["model_trained"])

    def test_changed_holdout_quotes_do_not_change_final_training(self):
        process,result,first = self.holdout()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        text = (first/"model_artifacts/model.txt").read_text(encoding="utf-8")
        changes = {"日线":{"open":"11","high":"12","low":"10","close":"11","change_pct":"10"},
                   "日线派生涨跌":{"provider_change_pct":"10","inferred_reference_close_yuan":"10"},
                   "网页股本估值":{"CLOSE_PRICE":"11","CHANGE_RATE":"10"}, "状态估值":{"close":"11","preclose":"10"}}
        for folder,values in changes.items():
            path = self.data/folder/"SZ000001.csv"
            records = read_csv(path)
            records[-1].update(values)
            write_csv(path,records)
            if folder != "状态估值": write_csv(self.month/folder/path.name,records[-3:])
        self.update_provenance()
        self.p1,self.p2 = self.build_completed_inputs()
        self.make_selection()
        settings,_,inputs = load_classification_plan(self.module,self.selection)
        source = PreparedFoldSource(self.module,NumpyPreparedReader(self.module),FoldInputSpec(settings.split_run, settings.feature_set, settings.feature_count, UP_TARGET_SPEC["source_target_id"]),"holdout",self.selection,file_hash(self.selection),kind="holdout",input_records=inputs)
        from next_day_up.lightgbm_model import LightGBMTrainer
        from experiments.test_lightgbm import Telemetry
        try:
            description = source.describe()
            x,y = source.training_data()
            model = LightGBMTrainer(Telemetry(),lambda e:None).fit_fixed(x,up_labels(y),description.feature_names,settings)
            self.assertEqual(text,model.export_text())
        finally:
            source.close()

    def test_existing_workspace_holdout_usage_is_visible_to_new_method(self):
        split = json.loads((self.p2/"splits.json").read_text(encoding="utf-8"))
        identity = split["holdout_identity_sha256"]
        path = self.module/"runs/holdout-usage"/(identity+".json")
        previous = {"schema_version": 1, "holdout_identity_sha256": identity,
                    "attempts": [{"run": "runs/prior-method-result", "state": "holdout-evaluated", "exposure_started": True}]}
        write_json(path, previous)
        before = path.read_bytes()
        with patch("next_day_up.lightgbm_model.LightGBMTrainer.fit_fixed", side_effect=AssertionError("must not fit")):
            self.assertEqual(self.run_in_process(), 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.method_root/"runs/holdout-usage").exists())
