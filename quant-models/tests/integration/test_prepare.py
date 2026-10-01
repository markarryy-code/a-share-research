"""P0准入到P1公开命令的行为测试，全部来源位于临时目录。"""

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from fixtures import DataFixture, SRC, hashes, read_csv, write_csv, write_json


class PrepareTest(DataFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        process, _, self.p0_run = self.run_command()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.baseline = {"schema_version": 1, "feature_set": "price-volume-v2", "target_id": "next-market-day-quote-return-v1",
                         "source_acceptance": {"inspection_run": str(self.p0_run.relative_to(self.module)), "supplements": []}}
        write_json(self.module / "configs/baseline.json", self.baseline)

    def prepare(self):
        env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        process = subprocess.run([sys.executable, "-B", "-m", "daily_return", "prepare", "--module-root", str(self.module)], env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        last = json.loads(process.stdout.strip().splitlines()[-1])
        if "run_directory" not in last:
            return process, last, None
        run = Path(last["run_directory"])
        result = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
        return process, result, run

    def successful_prepare(self):
        process, result, run = self.prepare()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr + (run / "issues.csv").read_text(encoding="utf-8"))
        return result, run, self.module / result["cache_directory"]

    def test_full_public_path_is_read_only_and_keeps_pause_and_unknown_labels(self):
        before = hashes(self.data)
        result, run, cache = self.successful_prepare()
        self.assertEqual(before, hashes(self.data))
        self.assertEqual(result["shape"], [11, 49])
        self.assertEqual(result["counts"]["prediction_eligible"], 10)
        self.assertEqual(result["counts"]["supervised_candidates"], 7)
        self.assertEqual(result["counts"]["eligible_target_suspended"], 1)
        self.assertEqual(result["counts"]["eligible_target_calendar_unknown"], 2)
        self.assertFalse(result["splits_prepared"])
        self.assertFalse(result["model_trained"])
        X, y, rows = (np.load(cache / (name + ".npy"), allow_pickle=False) for name in ("X", "y", "rows"))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.isnan(X[:, 44:]).all())  # 小样本不足20日，仍保留全部预测/目标行。
        self.assertEqual(y.dtype, np.float64)
        stocks = json.loads((cache / "stocks.json").read_text(encoding="utf-8"))
        sz = next(item["stock_idx"] for item in stocks if item["stock_id"] == "SZ000001")
        friday = np.flatnonzero((rows["stock_idx"] == sz) & (rows["as_of_idx"] == 1))[0]
        self.assertTrue(rows["prediction_eligible"][friday])
        self.assertEqual(rows["target_idx"][friday], 2)
        self.assertTrue(np.isnan(y[friday]))
        paused = np.flatnonzero((rows["stock_idx"] == sz) & (rows["as_of_idx"] == 2))[0]
        self.assertFalse(rows["prediction_eligible"][paused])
        self.assertAlmostEqual(y[paused], -.01)
        with (run / "source_files.csv").open(encoding="utf-8", newline="") as stream:
            self.assertTrue(all(row["unchanged"] == "True" for row in csv.DictReader(stream)))

    def test_cache_reuse_retains_new_execution_record(self):
        first, first_run, cache = self.successful_prepare()
        before = hashes(cache)
        second, second_run, second_cache = self.successful_prepare()
        self.assertFalse(first["cache_reused"])
        self.assertTrue(second["cache_reused"])
        self.assertEqual(cache, second_cache)
        self.assertNotEqual(first_run, second_run)
        self.assertEqual(before, hashes(cache))

    def test_invalid_acceptance_config_fails_before_creating_run(self):
        variants = [[], {key: value for key, value in self.baseline.items() if key != "source_acceptance"},
                    {**self.baseline, "source_acceptance": None},
                    {**self.baseline, "source_acceptance": {"inspection_run": ""}},
                    {**self.baseline, "source_acceptance": {"inspection_run": "p0", "supplements": "not-a-list"}}]
        for baseline in variants:
            with self.subTest(baseline=baseline):
                write_json(self.module / "configs/baseline.json", baseline)
                before = hashes(self.module)
                process, result, run = self.prepare()
                self.assertEqual(process.returncode, 2)
                self.assertFalse(result["output_created"])
                self.assertIsNone(run)
                self.assertEqual(before, hashes(self.module))

    def test_source_change_cannot_hide_behind_existing_cache(self):
        self.successful_prepare()
        path = self.data / "日线/SZ000001.csv"
        records = read_csv(path)
        records[0]["turnover_pct"] = "9"
        write_csv(path, records)
        process, result, run = self.prepare()
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result["state"], "failed")
        self.assertIn("内容已变化", (run / "issues.csv").read_text(encoding="utf-8"))

    def test_corrupt_or_unfinished_cache_is_rejected(self):
        _, _, cache = self.successful_prepare()
        path = cache / "X.npy"
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            old = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([old[0] ^ 1]))
        process, result, run = self.prepare()
        self.assertEqual(process.returncode, 2)
        self.assertIn("缓存文件指纹不符", (run / "issues.csv").read_text(encoding="utf-8"))
        metadata = json.loads((cache / "prepared.json").read_text(encoding="utf-8"))
        metadata["state"] = "writing"
        write_json(cache / "prepared.json", metadata)
        process, result, run = self.prepare()
        self.assertEqual(process.returncode, 2)
        self.assertIn("缓存未完成", (run / "issues.csv").read_text(encoding="utf-8"))

    def test_failed_primary_and_conflicting_supplement_are_not_accepted(self):
        path = self.p0_run / "dataset.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["admission_status"] = "blocked"
        write_json(path, metadata)
        process, _, run = self.prepare()
        self.assertEqual(process.returncode, 2)
        self.assertIn("完整通过", (run / "issues.csv").read_text(encoding="utf-8"))
        metadata["admission_status"] = "passed_with_limitations"
        write_json(path, metadata)
        supplement = self.module / "runs/supplement"
        write_json(supplement / "verification.json", {"previous_p0_run": self.p0_run.name, "errors": 0, "checked_inputs_unchanged": True,
                   "prior_p0_market_source_hashes_match": True, "markets": {symbol: {"monthly_copy": "verified"} for symbol in ("sh000001", "sz399001")}})
        record = next(row for row in read_csv(self.p0_run / "source_files.csv") if row["path"].endswith("冻结配置.json"))
        record["sha256"] = record["end_sha256"] = "0" * 64
        write_csv(supplement / "source_files.csv", [record])
        self.baseline["source_acceptance"]["supplements"] = [str(supplement.relative_to(self.module))]
        write_json(self.module / "configs/baseline.json", self.baseline)
        process, _, run = self.prepare()
        self.assertEqual(process.returncode, 2)
        self.assertIn("指纹冲突", (run / "issues.csv").read_text(encoding="utf-8"))

    def test_future_changed_source_changes_target_but_not_past_feature_rows(self):
        _, _, original_cache = self.successful_prepare()
        original_X = np.load(original_cache / "X.npy", allow_pickle=False)
        original_y = np.load(original_cache / "y.npy", allow_pickle=False)
        original_rows = np.load(original_cache / "rows.npy", allow_pickle=False)
        changes = {
            "日线": {"open": "11", "high": "12", "low": "10", "close": "11", "change_pct": "10"},
            "日线派生涨跌": {"provider_change_pct": "10", "inferred_reference_close_yuan": "10"},
            "网页股本估值": {"CLOSE_PRICE": "11", "CHANGE_RATE": "10"},
            "状态估值": {"close": "11", "preclose": "10"},
        }
        for folder, values in changes.items():
            path = self.data / folder / "SZ000001.csv"
            records = read_csv(path)
            records[-1].update(values)
            write_csv(path, records)
            if folder != "状态估值":
                write_csv(self.month / folder / path.name, records[-3:])
        self.update_provenance()
        process, _, new_p0 = self.run_command()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.baseline["source_acceptance"]["inspection_run"] = str(new_p0.relative_to(self.module))
        write_json(self.module / "configs/baseline.json", self.baseline)
        _, _, new_cache = self.successful_prepare()
        self.assertNotEqual(original_cache, new_cache)
        new_X = np.load(new_cache / "X.npy", allow_pickle=False)
        new_y = np.load(new_cache / "y.npy", allow_pickle=False)
        past = original_rows["as_of_idx"] < 5
        np.testing.assert_array_equal(original_X[past], new_X[past])
        stocks = json.loads((original_cache / "stocks.json").read_text(encoding="utf-8"))
        sz = next(item["stock_idx"] for item in stocks if item["stock_id"] == "SZ000001")
        prior_day = np.flatnonzero((original_rows["stock_idx"] == sz) & (original_rows["as_of_idx"] == 4))[0]
        self.assertEqual(original_y[prior_day], 0)
        self.assertAlmostEqual(new_y[prior_day], .1)


if __name__ == "__main__":
    unittest.main()
