"""真实P0/P1产物接入公开P2命令，检查归档、只读和失败状态。"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from fixtures import DataFixture, SRC, hashes, read_csv, write_json


class SplitIntegrationTests(DataFixture, unittest.TestCase):
    def command(self, *args):
        env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, "-B", "-m", "daily_return", "prepare", "--module-root", str(self.module), *args],
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        last = json.loads(result.stdout.strip().splitlines()[-1])
        run = Path(last["run_directory"]) if "run_directory" in last else None
        return result, run

    def setUp(self):
        super().setUp()
        process, _, p0 = self.run_command()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.config = {"schema_version":1,"feature_set":"price-volume-v2","target_id":"next-market-day-quote-return-v1",
                       "source_acceptance":{"inspection_run":str(p0.relative_to(self.module)),"supplements":[]},
                       "split_plan":{"version":"expanding-target-date-v1", "folds":[
                           {"id":"F01","train_through":"2026-08-21","validation_start":"2026-08-22","validation_end":"2026-08-24"}],
                           "holdout":{"start":"2026-08-25","end":"2026-08-27"}}}
        write_json(self.module/"configs/baseline.json", self.config)
        process, self.parent = self.command()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.parent_result = json.loads((self.parent/"dataset.json").read_text(encoding="utf-8"))
        self.samples = self.module/self.parent_result["cache_directory"]

    def prepare_split(self):
        process, run = self.command("--prepared-run", str(self.parent.relative_to(self.module)))
        result = json.loads((run/"splits.json").read_text(encoding="utf-8")) if run else None
        return process, result, run

    def successful_split(self):
        process, result, run = self.prepare_split()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr+(run/"issues.csv").read_text(encoding="utf-8"))
        return result, run, self.module/result["cache_directory"]

    def test_public_command_exact_roles_and_readonly_inputs(self):
        before = (hashes(self.data), hashes(self.samples), hashes(self.parent))
        result, run, cache = self.successful_split()
        self.assertEqual(before,(hashes(self.data),hashes(self.samples),hashes(self.parent)))
        self.assertEqual(result["state"],"split-ready")
        self.assertTrue(result["splits_prepared"])
        self.assertFalse(result["model_trained"])
        self.assertFalse(result["holdout_evaluation_performed"])
        expected = {"F01.train":[5],"F01.predict":[0,6],"F01.score":[0],"holdout.train":[0,5],"holdout.predict":[1,2,3,8,9],"holdout.score":[1,2,3,8,9]}
        for name, indices in expected.items():
            self.assertEqual(np.load(cache/(name+".npy"),allow_pickle=False).tolist(),indices)
        self.assertTrue(all(r["unchanged"] == "True" for r in read_csv(run/"source_files.csv")))
        self.assertEqual(result["parent_run"],self.parent.relative_to(self.module).as_posix())

    def test_reuse_does_not_rewrite_cache_and_keeps_separate_run(self):
        first, old_run, cache = self.successful_split()
        before = hashes(cache)
        second, run, reused = self.successful_split()
        self.assertEqual(cache,reused)
        self.assertNotEqual(run,old_run)
        self.assertFalse(first["cache_reused"])
        self.assertTrue(second["cache_reused"])
        self.assertEqual(before,hashes(cache))

    def test_changed_completed_sample_is_not_hidden_by_split_cache(self):
        self.successful_split()
        with (self.samples/"rows.npy").open("r+b") as stream:
            stream.seek(-1,2)
            stream.write(b"\x7f")
        process,result,run = self.prepare_split()
        self.assertEqual(process.returncode,2)
        self.assertEqual(result["state"],"failed")
        self.assertIn("缓存文件指纹不符",(run/"issues.csv").read_text(encoding="utf-8"))

    def test_damaged_or_incomplete_split_cache_fails(self):
        _,_,cache = self.successful_split()
        with (cache/"F01.train.npy").open("ab") as stream:
            stream.write(b"changed")
        process,result,run = self.prepare_split()
        self.assertEqual(process.returncode,2)
        self.assertIn("时间索引缓存指纹不符",(run/"issues.csv").read_text(encoding="utf-8"))
        meta = json.loads((cache/"split-ready.json").read_text(encoding="utf-8"))
        meta["state"] = "writing"
        write_json(cache/"split-ready.json",meta)
        process,result,run = self.prepare_split()
        self.assertEqual(process.returncode,2)
        self.assertIn("缓存未完成",(run/"issues.csv").read_text(encoding="utf-8"))

    def test_parent_must_be_completed_and_reference_same_manifest(self):
        execution = json.loads((self.parent/"experiment.json").read_text(encoding="utf-8"))
        execution["execution_status"] = "failed"
        write_json(self.parent/"experiment.json",execution)
        process,_,run = self.prepare_split()
        self.assertEqual(process.returncode,2)
        self.assertIn("归档尚未完成",(run/"issues.csv").read_text(encoding="utf-8"))
        execution["execution_status"] = "completed"
        write_json(self.parent/"experiment.json",execution)
        self.parent_result["cache_manifest_sha256"] = "0"*64
        write_json(self.parent/"dataset.json",self.parent_result)
        process,_,run = self.prepare_split()
        self.assertEqual(process.returncode,2)
        self.assertIn("清单指纹不一致",(run/"issues.csv").read_text(encoding="utf-8"))

    def test_plan_cannot_move_holdout_month(self):
        self.config["split_plan"]["holdout"]["start"] = "2026-08-24"
        write_json(self.module/"configs/baseline.json",self.config)
        process,result,run = self.prepare_split()
        self.assertEqual(process.returncode,2)
        self.assertEqual(result["state"],"failed")
        self.assertIn("冻结月份不符",(run/"issues.csv").read_text(encoding="utf-8"))

    def test_feature_and_target_versions_must_match_named_samples(self):
        for key, other in (("target_id", "different-target"), ("feature_set", "price-volume-v1")):
            original = self.config[key]
            self.config[key] = other
            write_json(self.module/"configs/baseline.json",self.config)
            process,result,run = self.prepare_split()
            self.assertEqual(process.returncode,2)
            self.assertIn("版本与实验配置不一致",(run/"issues.csv").read_text(encoding="utf-8"))
            self.config[key] = original

    def run_with_progress(self, callback):
        from dataclasses import replace
        from daily_return.execution import RunRecord
        from daily_return.samples.prepared_reader import NumpyPreparedReader
        from daily_return.experiments.sample_source import PreparedIndexSource
        from daily_return.experiments.numpy_store import NumpySplitStore
        from daily_return.experiments.application import prepare_splits, SPLIT_MODULES
        from daily_return.serialization import file_hash
        run = RunRecord(self.module,"P2")
        context = replace(run.context,progress=callback)
        reader = NumpyPreparedReader(self.module)
        source = PreparedIndexSource(reader,self.parent.relative_to(self.module).as_posix(),("price-volume-v2","next-market-day-quote-return-v1"))
        config_path = self.module/"configs/baseline.json"
        store = NumpySplitStore(self.module,run.directory,context,{n:context.code_sha256[n] for n in SPLIT_MODULES},config_path,file_hash(config_path))
        outcome = prepare_splits(source,store,self.config["split_plan"],context)
        run.finish(outcome.status)
        return outcome, run.directory

    def test_interrupt_keeps_partial_and_cannot_publish_ready_indices(self):
        def interrupt(event):
            raise KeyboardInterrupt()
        outcome,run = self.run_with_progress(interrupt)
        self.assertEqual(outcome.exit_code,130)
        self.assertEqual(outcome.result["state"],"interrupted")
        cache = self.module/".cache/splits"
        self.assertEqual(list(cache.rglob("split-ready.json")),[])
        self.assertEqual(len(list(cache.glob("*.partial"))),1)

    def test_inputs_changed_during_selection_cannot_publish_indices(self):
        def change(event):
            with (self.samples/"rows.npy").open("ab") as stream:
                stream.write(b"changed-during-split")
        outcome,run = self.run_with_progress(change)
        self.assertEqual(outcome.exit_code,2)
        self.assertEqual(outcome.result["state"],"failed")
        self.assertEqual(list((self.module/".cache/splits").rglob("split-ready.json")),[])
        self.assertIn("P2执行期间完成样本发生变化",(run/"issues.csv").read_text(encoding="utf-8"))
