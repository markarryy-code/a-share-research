"""准入规则直接消费值数据，测试不建立研究目录或报告文件。"""

import sys
import unittest
from fixtures import SRC

sys.path.insert(0, str(SRC))
from daily_return.market_data.domain import (
    AdmissionEvidence, DataError, IssueCollector, MarketEvidence, Table,
    approve_evidence, immutable, review_market,
)


class AdmissionRulesTests(unittest.TestCase):
    def test_missing_monthly_market_copy_is_a_warning_value(self):
        calendar = ("2026-08-24", "2026-08-25")
        rows = immutable([dict(date=d, symbol="sh000001", open="100", close="100", high="101", low="99") for d in calendar])
        table = Table(tuple(rows[0]), rows)
        evidence = MarketEvidence("sh000001", table, None, "market-history", "market-month")
        issues = IssueCollector()
        result = review_market(evidence, calendar, "2026-08-25", issues)
        problems = issues.drain()
        self.assertEqual(result, {"rows": 2, "monthly_rows_in_history": 1, "monthly_copy": "not_present"})
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].severity, "warning")
        self.assertEqual(problems[0].code, "market_month_copy_missing")
        self.assertEqual(issues.errors, 0)

    def test_conflicting_approval_is_rejected_without_source_io(self):
        primary = immutable(dict(schema_version=1, stage="P0", admission_status="passed_with_limitations", inputs_unchanged=True))
        execution = immutable(dict(execution_status="completed", run_id="p0"))
        row = dict(path="prices", sha256="a" * 64, end_sha256="a" * 64, unchanged="True")
        supplement = {"record": {"previous_p0_run": "p0", "errors": 0, "checked_inputs_unchanged": True,
                                "prior_p0_market_source_hashes_match": True,
                                "markets": {s: {"monthly_copy": "verified"} for s in ("sh000001", "sz399001")}},
                      "manifest": [{**row, "sha256": "b" * 64, "end_sha256": "b" * 64}], "allowed_new": ()}
        evidence = AdmissionEvidence(primary, execution, "p0", "p0", True, immutable([row]), immutable([supplement]), immutable({}))
        with self.assertRaisesRegex(DataError, "指纹冲突"):
            approve_evidence(evidence)
