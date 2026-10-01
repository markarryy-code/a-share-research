"""准入证据与P0报告的文件实现；不裁定资料是否合格。"""

from dataclasses import asdict
from pathlib import Path
from .domain import AdmissionEvidence, immutable
from .local_source import FileLedger
from ..serialization import save_json, write_csv

COVERAGE_FIELDS = ("stock_id", "daily_rows", "state_rows", "suspended_rows", "web_rows", "monthly_daily_rows", "valid_trading_rows",
                   "independent_reference_rows", "legacy_difference_records_checked", "next_day_pair_candidates",
                   "calendar_tail_without_next_day", "next_day_suspended", "next_day_invalid_label",
                   "case_ordinary", "case_listing_day", "case_resumed_after_pause", "case_known_ex_dividend_clue",
                   "case_reference_differs_from_last_close")
LIMITATIONS = [
    "股票池按期末存续与不超过70元价格筛选，属于固定池条件研究，存在前视选择偏差。",
    "涨跌幅统一取网页CHANGE_RATE；反推参考价依赖精度假设，不是交易所原始前收价。",
    "数据源一致、代数关系成立及已有子集交叉核验，不等于全量外部真实性认证。",
    "历史母表仍含留出月；P0只核对资料，特征加工和目标日隔离在P1/P2实施。",
    "日级信息可用时间采用收盘资料已可得的研究约定，没有虚构精确发布时间。",
    "行业、财务、估值、完整交易特殊制度未纳入首版输入准入；网页表只核对收盘和涨跌原值。",
]



def report_text(dataset, experiment):
    counts = dataset.get("counts", {})
    status = "首版量价资料检查通过，保留已声明限制" if dataset["admission_status"] == "passed_with_limitations" else "资料检查未通过，须先处理所列问题"
    lines = ["# P0资料准入报告", "", f"**{status}。**", "", f"运行：`{experiment['run_id']}`；执行状态：`{experiment['execution_status']}`。",
             "", "本次只执行资料与环境检查，未生成预测特征、拟合模型或计算预测成绩。", "",
             "| 检查项目 | 实际结果 |", "|---|---|",
             f"| 已检查股票 | {counts.get('stocks_checked', 0)} |", f"| 成交日线 | {counts.get('daily_rows', 0)} |",
             f"| 通过行级核验 | {counts.get('valid_trading_rows', 0)} |", f"| 暂停股票日 | {counts.get('suspended_rows', 0)} |",
             f"| 月度成交日线 | {counts.get('monthly_daily_rows', 0)} |",
             f"| 已有独立前收价交叉核验 | {counts.get('independent_reference_rows', 0)} |",
             f"| 既有涨跌口径差异逐条核对 | {counts.get('legacy_difference_records_checked', 0)} |",
             f"| 下一市场日有合格行情的候选对 | {counts.get('next_day_pair_candidates', 0)} |",
             f"| 信息日合格、下一市场日暂停 | {counts.get('next_day_suspended', 0)} |",
             f"| 日历末端无下一市场日 | {counts.get('calendar_tail_without_next_day', 0)} |",
             f"| 已读取输入文件 | {dataset['source_file_count']} |",
             f"| 已读取输入检查前后哈希一致 | {dataset['inputs_unchanged']} |", "",
             "候选对仅是日历和行情可用性计数，不是完成特征、时点和切分后的训练样本数。", "",
             "## 问题与限制", ""]
    for item in dataset["issues"]:
        lines.append(f"- `{item['severity']}/{item['code']}`：{item['count']}条，细节见[问题表](issues.csv)。")
    lines.extend("- " + item for item in LIMITATIONS)
    lines.extend(["", "## 文件入口", "", "- [数据清单与准入](dataset.json)、[逐文件指纹](source_files.csv)。",
                  "- [逐股覆盖](coverage.csv)、[标签核验示例](label_samples.csv)、[运行环境与代码版本](experiment.json)。",
                  "", "市场基准月度副本是否存在及核对结果见dataset.json的markets；缺副本不修改研究原件。",
                  "P0仅使用Python标准库。训练依赖的当前状态记录在experiment.json，未安装或升级依赖。", ""])
    return "\n".join(lines)



SAMPLE_FIELDS = ("case", "stock_id", "date", "provider_change_pct", "label_fraction", "close", "inferred_reference", "independent_reference")


class LocalAdmissionArchive:
    def __init__(self, root: Path, history: Path, holdout: Path, ledger: FileLedger, run_dir: Path | None = None):
        self.root, self.history, self.holdout, self.ledger, self.run_dir = root, history, holdout, ledger, run_dir
        self._issue_stream = None

    def read_evidence(self, selection):
        base = (self.root / selection["inspection_run"]).resolve()
        primary = self.ledger.json(base / "dataset.json", "acceptance_record")
        execution = self.ledger.json(base / "experiment.json", "acceptance_execution")
        primary_manifest = base / primary["source_manifest"]

        def manifest(path):
            table = self.ledger.csv(path, "acceptance_manifest", ("path", "sha256", "end_sha256", "unchanged"))
            return immutable([{**row, "path": self.ledger.relative((self.root / row["path"]).resolve())} for row in table.rows])

        records = manifest(primary_manifest)
        supplements = []
        for name in selection.get("supplements", ()):
            folder = (self.root / name).resolve()
            record = self.ledger.json(folder / "verification.json", "supplemental_acceptance")
            allowed = {primary_manifest.resolve(), self.history / "冻结配置.json", self.history / "交易日期.json"}
            for symbol in ("sh000001", "sz399001"):
                allowed.update({self.history / "市场行情" / (symbol + ".csv"), self.holdout / "市场行情" / (symbol + ".csv")})
            supplements.append({"key": self.ledger.relative(folder), "record": record,
                                "manifest": manifest(folder / "source_files.csv"),
                                "allowed_new": tuple(self.ledger.relative(p) for p in allowed)})
        matched = self.history == (self.root / primary["history_directory"]).resolve() and self.holdout == (self.root / primary["holdout_directory"]).resolve()
        prior_reads = {entry["path"]: entry["sha256"] for entry in self.ledger.entries.values()}
        return AdmissionEvidence(immutable(primary), immutable(execution), base.name, self.ledger.relative(base), matched,
                                 records, immutable(supplements), immutable(prior_reads))

    def initialize(self):
        import csv
        self._issue_stream = (self.run_dir / "issues.csv").open("w", encoding="utf-8", newline="")
        self._issue_writer = csv.DictWriter(self._issue_stream, fieldnames=("severity", "code", "stock_id", "date", "source", "detail"), lineterminator="\n")
        self._issue_writer.writeheader()
        self.write_coverage([], [])

    def append_issues(self, issues):
        for issue in issues:
            self._issue_writer.writerow(asdict(issue))
        self._issue_stream.flush()

    def write_coverage(self, coverage, samples):
        write_csv(self.run_dir / "coverage.csv", COVERAGE_FIELDS, coverage)
        write_csv(self.run_dir / "label_samples.csv", SAMPLE_FIELDS, samples)

    def finish(self, result, receipt, execution):
        self._issue_stream.close()
        result["limitations"] = LIMITATIONS
        write_csv(self.run_dir / "source_files.csv", ("path", "role", "bytes", "sha256", "end_sha256", "unchanged"), receipt.records)
        save_json(self.run_dir / "dataset.json", result)
        (self.run_dir / "report.md").write_text(report_text(result, execution), encoding="utf-8", newline="\n")
