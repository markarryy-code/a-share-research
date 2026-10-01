"""资料检查和准入解析用例：通过接口读取与归档，领域裁定规则。"""

from collections import Counter
from dataclasses import dataclass
from .domain import DataError, IssueCollector, AdmissionSnapshot, approve_evidence, admitted_snapshot, inspection_header, review_market, review_stock_evidence
from .ports import MarketSource, AdmissionArchive
from ..execution import RunContext


def resolve_admission(source: MarketSource, archive: AdmissionArchive, selection) -> AdmissionSnapshot:
    evidence = archive.read_evidence(selection)
    expected = approve_evidence(evidence)
    source.enforce(expected)
    core = source.read_core()
    return admitted_snapshot(evidence, core, expected, source.input_refs(core))


class AdmittedReadSession:
    """显式读取会话；快照本身始终不持有I/O。"""
    def __init__(self, snapshot: AdmissionSnapshot, source: MarketSource):
        self.snapshot, self._source = snapshot, source
        source.enforce(snapshot.expected_hashes)

    def read_market_facts(self, symbol):
        return self._source.read_market(symbol).history

    def iter_stock_facts(self):
        for stock in self.snapshot.core.stocks:
            yield stock, self._source.read_stock(stock)

    def verify_current(self):
        self._source.verify_current(self.snapshot)

    def verify_unchanged(self):
        return self._source.verify_unchanged()


def open_admitted_read(snapshot: AdmissionSnapshot, source: MarketSource):
    return AdmittedReadSession(snapshot, source)


@dataclass(frozen=True)
class InspectionOutcome:
    result: dict
    status: str
    exit_code: int


def inspect_market_data(source: MarketSource, archive: AdmissionArchive, context: RunContext):
    issues = IssueCollector()
    archive.initialize()
    dataset, status, exit_code = {}, "running", 0
    try:
        inputs = source.inspection_inputs()
        core, frozen, calendar = inputs.core, inputs.core.frozen, inputs.core.calendar
        month_start = frozen["month_start"]
        start, end = frozen["start"], frozen["end"]
        tolerance, clues, provenance = inspection_header(inputs, issues)
        markets = {symbol: review_market(source.read_market(symbol, monthly=True), calendar, month_start, issues)
                   for symbol in ("sh000001", "sz399001")}
        archive.append_issues(issues.drain())
        totals, coverage, samples, sample_keys = Counter(), [], [], set()
        for index, stock in enumerate(core.stocks, 1):
            symbol = stock["exchange"] + stock["code"]
            try:
                evidence = source.read_stock(stock, monthly=True)
                if "reference" in evidence.tables:
                    totals["independent_reference_stocks"] += 1
                summary = review_stock_evidence(stock, evidence, calendar, month_start, issues, tolerance, clues, provenance, samples, sample_keys)
                coverage.append(summary)
                totals.update({key: value for key, value in summary.items() if key != "stock_id"})
                totals["stocks_checked"] += 1
            except (DataError, ValueError, KeyError) as error:
                issues.add("stock_contract_failed", str(error), symbol)
            archive.append_issues(issues.drain())
            if index % 100 == 0 or index == len(core.stocks):
                context.progress({"stage": "stock_audit", "stocks_processed": index, "stocks_total": len(core.stocks),
                                  "daily_rows": totals["daily_rows"], "errors": issues.errors})
        expectations = {"stocks_checked": len(core.stocks), "daily_rows": inputs.daily_audit["daily_rows"],
                        "monthly_daily_rows": inputs.derived_audit["monthly_rows"], "legacy_difference_records_checked": len(clues)}
        for key, expected in expectations.items():
            if totals[key] != expected:
                issues.add("source_count_mismatch", f"{key}实际{totals[key]}，来源声明/名单为{expected}")
        archive.write_coverage(coverage, samples)
        dataset = {
            "history_directory": inputs.history_key, "holdout_directory": inputs.holdout_key,
            "range": {"start": start, "end": end, "month_start": month_start, "first_market_day": calendar[0],
                      "last_market_day": calendar[-1], "market_days": len(calendar), "holdout_market_days": len(inputs.monthly_calendar)},
            "source_frozen_at": frozen.get("frozen_at"), "stock_count": len(core.stocks), "counts": dict(totals), "markets": markets,
            "label": {"column": "provider_change_pct", "source_column": "CHANGE_RATE", "source_unit": "percentage_points",
                      "internal_unit": "fraction", "conversion": "source / 100", "inferred_reference_is_original": False,
                      "inference_tolerance_percentage_points": str(tolerance)},
            "units": {"ohlc": "CNY", "volume_lots": "100_shares", "volume_shares": "shares", "amount_yuan": "CNY", "turnover_pct": "percentage_points"},
            "source_scope_complete": frozen.get("scope_complete"), "survivorship_bias": frozen.get("survivorship_bias"),
            "price_selection_bias": frozen.get("price_selection_bias"),
        }
        status = "completed"
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
        issues.add("inspection_interrupted", "用户或系统中断，保留已消费文件记录")
    except Exception as error:
        status, exit_code = "failed", 1
        issues.add("inspection_failed", f"{type(error).__name__}: {error}")
    finally:
        context.progress({"stage": "verify_source_hashes"})
        receipt = source.verify_unchanged()
        for key in receipt.changed:
            issues.add("source_changed_during_inspection", "检查期间源文件内容改变或不可再读", source=key)
        dataset.update(schema_version=1, stage="P0", admission_status="blocked" if issues.errors else "passed_with_limitations",
                       source_fingerprint=receipt.fingerprint, source_file_count=len(receipt.records), source_manifest="source_files.csv",
                       inputs_unchanged=not receipt.changed, issues=issues.summary(),
                       features_prepared=False, splits_prepared=False, model_trained=False)
        archive.append_issues(issues.drain())
        archive.finish(dataset, receipt, {"run_id": context.run_id, "execution_status": status})
    return InspectionOutcome(dataset, status, exit_code or (2 if issues.errors else 0))
