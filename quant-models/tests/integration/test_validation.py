"""单折CLI覆盖实际LightGBM、不可评分预测和版本失败。"""

import csv
import gzip
import json
import math
import sys
import unittest
from fixtures import hashes, read_csv, write_csv, write_json, SRC
from validation_fixtures import ValidationFixture

sys.path.insert(0,str(SRC))


def compressed_rows(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


class ValidationIntegrationTests(ValidationFixture,unittest.TestCase):
    def test_public_f01_is_readonly_keeps_predictions_and_scores_original_targets(self):
        p1=json.loads((self.p1/"dataset.json").read_text(encoding="utf-8"))
        before=hashes(self.data),hashes(self.module/p1["cache_directory"]),hashes(self.p1),hashes(self.p2)
        process,result,run=self.validate()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr+(run/"issues.csv").read_text(encoding="utf-8"))
        self.assertEqual(result["state"],"fold-validated")
        self.assertFalse(result["p3_complete"])
        self.assertFalse(result["holdout_evaluation_performed"])
        self.assertEqual(result["completed_folds"],["F01"])
        self.assertEqual(before,(hashes(self.data),hashes(self.module/p1["cache_directory"]),hashes(self.p1),hashes(self.p2)))
        predicted=compressed_rows(run/"predictions.csv.gz")
        scored=compressed_rows(run/"predictions.scored.csv.gz")
        self.assertEqual(len(predicted),2)
        self.assertNotIn("y_true",predicted[0])
        valid=[r for r in scored if r["score_eligible"]=="True"]
        self.assertEqual(len(valid),1)
        self.assertEqual(next(r for r in scored if r["score_eligible"]=="False")["unscorable_reason"],"target_suspended")
        expected=math.sqrt(sum((float(r["y_pred"])-float(r["y_true"]))**2 for r in valid)/len(valid))
        self.assertAlmostEqual(result["metrics"]["lightgbm"]["rmse"],expected)
        self.assertEqual(result["reload_check"]["maximum_absolute_difference"],0)

    def test_wrong_feature_count_and_damaged_index_fail_without_a_model(self):
        self.config["feature_count"]=44
        write_json(self.module/"configs/p3-validation.json",self.config)
        process,result,run=self.validate()
        self.assertEqual(process.returncode,2)
        self.assertFalse((run/"model_artifacts/model.txt").exists())
        self.config["feature_count"]=49
        write_json(self.module/"configs/p3-validation.json",self.config)
        split=json.loads((self.p2/"splits.json").read_text(encoding="utf-8"))
        with (self.module/split["cache_directory"]/"F01.train.npy").open("ab") as stream:stream.write(b"corrupt")
        process,result,run=self.validate()
        self.assertEqual(process.returncode,2)
        self.assertIn("指纹不符",(run/"issues.csv").read_text(encoding="utf-8"))

    def test_holdout_is_not_a_validate_command_choice(self):
        process,run=self.command("validate","--fold","holdout")
        self.assertEqual(process.returncode,2)
        self.assertIsNone(run)

    def run_with_event(self,event):
        from daily_return.execution import RunRecord
        from daily_return.resources import ResourceMonitor
        from daily_return.serialization import file_hash
        from daily_return.samples.prepared_reader import NumpyPreparedReader
        from daily_return.experiments.validation_domain import ValidationSettings
        from daily_return.experiments.validation_source import PreparedFoldSource
        from daily_return.experiments.validation_archive import LocalValidationArchive
        from daily_return.experiments.lightgbm_model import LightGBMTrainer
        from daily_return.experiments.validation import validate_fold,VALIDATION_MODULES
        run=RunRecord(self.module,"P3")
        telemetry=ResourceMonitor(self.module)
        settings=ValidationSettings.from_mapping(self.config)
        path=self.module/"configs/p3-validation.json"
        source=PreparedFoldSource(self.module,NumpyPreparedReader(self.module),settings,"F01",path,file_hash(path))
        archive=LocalValidationArchive(self.module,run.directory,run.context,{n:run.context.code_sha256[n] for n in VALIDATION_MODULES},run.record["environment"])
        trainer=LightGBMTrainer(telemetry,event)
        outcome=validate_fold(source,trainer,archive,telemetry,settings,self.config,"F01",run.context)
        run.finish(outcome.status)
        return outcome,run.directory

    def test_interruption_preserves_failure_and_releases_sample_maps(self):
        def interrupt(event):raise KeyboardInterrupt()
        outcome,run=self.run_with_event(interrupt)
        self.assertEqual(outcome.exit_code,130)
        self.assertEqual(outcome.result["state"],"interrupted")
        self.assertFalse(outcome.result["p3_complete"])

    def test_midrun_config_change_cannot_mark_fold_completed(self):
        def change(event):
            with (self.module/"configs/p3-validation.json").open("a",encoding="utf-8") as stream:stream.write(" ")
        outcome,run=self.run_with_event(change)
        self.assertEqual(outcome.exit_code,2)
        self.assertEqual(outcome.result["state"],"failed")
        self.assertFalse(outcome.result["inputs_unchanged"])

    def test_changed_holdout_quote_does_not_change_f01_model_or_predictions(self):
        process,_,first=self.validate()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        changes={"日线":{"open":"11","high":"12","low":"10","close":"11","change_pct":"10"},
                 "日线派生涨跌":{"provider_change_pct":"10","inferred_reference_close_yuan":"10"},
                 "网页股本估值":{"CLOSE_PRICE":"11","CHANGE_RATE":"10"},
                 "状态估值":{"close":"11","preclose":"10"}}
        for folder,values in changes.items():
            path=self.data/folder/"SZ000001.csv"
            records=read_csv(path)
            records[-1].update(values)
            write_csv(path,records)
            if folder!="状态估值":write_csv(self.month/folder/path.name,records[-3:])
        self.update_provenance()
        self.p1,self.p2=self.build_completed_inputs()
        self.config["split_run"]=self.p2.relative_to(self.module).as_posix()
        write_json(self.module/"configs/p3-validation.json",self.config)
        process,_,second=self.validate()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertEqual((first/"model_artifacts/model.txt").read_bytes(),(second/"model_artifacts/model.txt").read_bytes())
        before=compressed_rows(first/"predictions.csv.gz")
        after=compressed_rows(second/"predictions.csv.gz")
        self.assertEqual([r["y_pred"] for r in before],[r["y_pred"] for r in after])
