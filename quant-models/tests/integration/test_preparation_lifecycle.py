"""真实读写会话的中断、来源变化与不可变边界，不需要全池测试资料。"""

import sys
import unittest
from dataclasses import fields

from fixtures import DataFixture, SRC

sys.path.insert(0, str(SRC))
from daily_return.execution import RunRecord
from daily_return.market_data import resolve_admission
from daily_return.market_data.local_source import FileLedger, LocalMarketSource
from daily_return.market_data.archive import LocalAdmissionArchive
from daily_return.samples.application import prepare_samples
from daily_return.samples.history_reader import AdmittedHistorySource
from daily_return.samples.numpy_store import NumpyPreparedStore


class PreparationLifecycleTests(DataFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        process, _, run = self.run_command()
        self.assertEqual(process.returncode, 0)
        self.selection = {"inspection_run": str(run.relative_to(self.module)), "supplements": []}

    def readers(self):
        ledger = FileLedger(self.module)
        config = ledger.json(self.module / "configs/data_sources.json", "data_config")
        source = LocalMarketSource(self.module, config, ledger)
        archive = LocalAdmissionArchive(self.module, source.history, source.holdout, ledger)
        return source, archive

    def run_direct(self, reader_class):
        source, archive = self.readers()
        run = RunRecord(self.module, "P1")
        reader = reader_class(source, archive, self.selection)
        store = NumpyPreparedStore(self.module, run.directory, run.context)
        outcome = prepare_samples(reader, store, run.context)
        run.finish(outcome.status)
        return outcome, run

    def test_admission_snapshot_is_deeply_read_only_and_has_no_io(self):
        source, archive = self.readers()
        snapshot = resolve_admission(source, archive, self.selection)
        self.assertFalse({"files", "module_root", "history", "holdout"} & {f.name for f in fields(snapshot)})
        with self.assertRaises(TypeError):
            snapshot.expected_hashes["new"] = "bad"
        with self.assertRaises(TypeError):
            snapshot.core.stocks[0]["code"] = "999999"
        with self.assertRaises(TypeError):
            snapshot.core.frozen["count"] = 99

    def test_interruption_keeps_partial_and_closes_windows_handles(self):
        class InterruptedHistory(AdmittedHistorySource):
            def iter_stocks(self):
                iterator = super().iter_stocks()
                yield next(iterator)
                raise KeyboardInterrupt

        outcome, run = self.run_direct(InterruptedHistory)
        self.assertEqual(outcome.exit_code, 130)
        self.assertEqual(outcome.result["state"], "interrupted")
        directories = list((self.module / ".cache").iterdir())
        self.assertEqual(len(directories), 1)
        partial = directories[0]
        self.assertTrue(partial.name.endswith(".partial"))
        self.assertFalse((partial / "prepared.json").exists())
        renamed = partial.with_name(partial.name + ".closed")
        partial.rename(renamed)
        self.assertTrue(renamed.exists())
        self.assertTrue((run.directory / "report.md").exists())

    def test_source_changed_after_read_cannot_publish_cache(self):
        changed = self.data / "日线/SZ000001.csv"

        class ChangingHistory(AdmittedHistorySource):
            def iter_stocks(self):
                yield from super().iter_stocks()
                # 修改已读的临时原件，验证结束复核，而不是伪造一个失败返回值。
                changed.write_bytes(changed.read_bytes() + b"\n")

        outcome, run = self.run_direct(ChangingHistory)
        self.assertEqual(outcome.exit_code, 2)
        self.assertFalse(outcome.result["inputs_unchanged"])
        self.assertTrue(all(p.name.endswith(".partial") for p in (self.module / ".cache").iterdir()))
        self.assertIn("来源发生变化", (run.directory / "issues.csv").read_text(encoding="utf-8"))
