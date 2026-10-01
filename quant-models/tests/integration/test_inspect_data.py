"""从公开命令核验P0行为与读取边界，复用受控的临时原始资料。"""

import sys
import unittest
from decimal import Decimal as D
from pathlib import Path

from fixtures import DataFixture, SRC, hashes, read_csv, write_csv, write_json


class InspectDataTests(DataFixture, unittest.TestCase):
    def test_success_is_read_only_and_units_cases_are_visible(self):
        before = hashes(self.data)
        process, dataset, run = self.run_command()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(before, hashes(self.data))
        self.assertEqual(dataset["admission_status"], "passed_with_limitations")
        self.assertEqual(dataset["counts"]["daily_rows"], 10)
        self.assertEqual(dataset["counts"]["suspended_rows"], 1)
        self.assertEqual(dataset["counts"]["next_day_pair_candidates"], 7)
        self.assertEqual(dataset["counts"]["next_day_suspended"], 1)
        self.assertEqual(dataset["counts"]["calendar_tail_without_next_day"], 2)
        self.assertFalse(dataset["model_trained"])
        samples = read_csv(run / "label_samples.csv")
        negative = next(row for row in samples if row["case"] == "known_ex_dividend_clue")
        self.assertEqual(D(negative["label_fraction"]), D("-0.01"))
        ipo = next(row for row in samples if row["case"] == "listing_day")
        self.assertEqual(D(ipo["label_fraction"]), D("1"))
        self.assertTrue(all(row["unchanged"] == "True" for row in read_csv(run / "source_files.csv")))

    def test_optional_monthly_market_copy_is_explicit_warning(self):
        (self.month / "市场行情/sh000001.csv").unlink()
        process, dataset, _ = self.run_command()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(dataset["markets"]["sh000001"]["monthly_copy"], "not_present")
        self.assertIn("market_month_copy_missing", [item["code"] for item in dataset["issues"]])

    def test_duplicate_date_blocks_admission(self):
        path = self.data / "日线/SZ000001.csv"
        rows = read_csv(path)
        write_csv(path, rows + rows[-1:])
        _, dataset, run = self.run_command()
        self.assertEqual(dataset["admission_status"], "blocked")
        self.assertTrue(any("重复证券日期" in row["detail"] for row in read_csv(run / "issues.csv")))

    def test_wrong_or_nonfinite_label_cannot_use_legacy_fallback(self):
        path = self.data / "日线派生涨跌/SZ000001.csv"
        original = read_csv(path)
        for value in ("999", "NaN", "Infinity", ""):
            with self.subTest(value=value):
                rows = [dict(row) for row in original]
                rows[0]["provider_change_pct"] = value
                write_csv(path, rows)
                process, dataset, run = self.run_command()
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(dataset["admission_status"], "blocked")
                self.assertIn("invalid_trading_row", [row["code"] for row in read_csv(run / "issues.csv")])

    def test_month_slice_conflict_is_not_repaired(self):
        path = self.month / "日线/SZ000001.csv"
        rows = read_csv(path)
        rows[0]["close"] = "123"
        write_csv(path, rows)
        before = path.read_bytes()
        _, dataset, _ = self.run_command()
        self.assertEqual(dataset["admission_status"], "blocked")
        self.assertEqual(path.read_bytes(), before)

    def test_required_field_and_calendar_gap_fail(self):
        path = self.data / "交易状态/逐日交易状态/SZ000001.csv"
        original = read_csv(path)
        write_csv(path, original[1:])
        _, dataset, run = self.run_command()
        self.assertEqual(dataset["admission_status"], "blocked")
        self.assertTrue(any("上市后状态日期不符" in row["detail"] for row in read_csv(run / "issues.csv")))
        write_csv(path, original)
        path = self.data / "日线/SH600001.csv"
        rows = read_csv(path)
        for row in rows:
            del row["volume_shares"]
        write_csv(path, rows)
        _, dataset, run = self.run_command()
        self.assertEqual(dataset["admission_status"], "blocked")
        self.assertTrue(any("缺少必要列" in row["detail"] and "volume_shares" in row["detail"] for row in read_csv(run / "issues.csv")))

    def test_provenance_change_is_reported(self):
        path = self.data / "日线/SZ000001.csv"
        rows = read_csv(path)
        rows[0]["turnover_pct"] = "2"
        write_csv(path, rows)
        _, dataset, _ = self.run_command()
        self.assertIn("provenance_hash_mismatch", [item["code"] for item in dataset["issues"]])

    def test_unexpected_metadata_failure_cannot_report_pass(self):
        write_json(self.data / "日线派生涨跌验收.json", {"assumed_max_percentage_error_points": "bad"})
        process, dataset, run = self.run_command()
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(dataset["admission_status"], "blocked")
        self.assertTrue((run / "report.md").exists())
        self.assertTrue((run / "coverage.csv").exists())

    def test_runs_are_preserved_and_output_cannot_overlap_source(self):
        _, _, first = self.run_command()
        _, _, second = self.run_command()
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists())
        write_json(self.module / "configs/data_sources.json", {"paths_relative_to": "module_root", "history_directory": ".", "holdout_directory": "../data/month"})
        before = hashes(self.module)
        process, result, _ = self.run_command()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(result["output_created"])
        self.assertEqual(before, hashes(self.module))

    def test_content_change_detected_even_with_same_size(self):
        sys.path.insert(0, str(SRC))
        from daily_return.market_data.local_source import FileLedger
        path = Path(self.temporary.name) / "changing.txt"
        path.write_bytes(b"first")
        ledger = FileLedger(self.module)
        ledger.read(path, "fixture")
        path.write_bytes(b"other")
        self.assertEqual(len(ledger.verify_unchanged()), 1)


if __name__ == "__main__":
    unittest.main()
