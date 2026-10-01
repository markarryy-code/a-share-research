"""NumPy缓存和P1归档实现：唯一拥有数组文件与发布目录的模块。"""

import json
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
from numpy.lib.format import open_memmap
from .domain import DataError, StoredSamples, ROW_DTYPE, planned_rows
from ..serialization import save_json, write_csv, file_hash, fingerprint
from ..execution import RunContext


CACHE_FILES = ("X.npy", "y.npy", "rows.npy", "stocks.json", "calendar.json", "features.json", "coverage.csv", "feature_quality.csv")
COVERAGE_FIELDS = ("stock_id", "grid_rows", "prediction_eligible", "information_day_paused", "target_observed",
                   "target_suspended", "target_calendar_unknown", "supervised_candidates", "eligible_target_suspended",
                   "eligible_target_calendar_unknown", "observed_zero_labels", "rows_with_nan_features")
QUALITY_FIELDS = ("name", "finite_rows", "nan_rows", "eligible_finite_rows", "minimum", "maximum")
SOURCE_FIELDS = ("path", "role", "bytes", "sha256", "end_sha256", "unchanged")


def readonly_description(value):
    """缓存返回的是只读描述，不把解析后的可变JSON树泄漏给用例。"""
    if isinstance(value, Mapping):
        return MappingProxyType({key: readonly_description(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(readonly_description(item) for item in value)
    return value

def validate_cache(directory: Path, preparation_id: str) -> dict:
    metadata = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    if metadata.get("state") != "prepared" or metadata.get("preparation_id") != preparation_id:
        raise DataError("缓存未完成或与当前准备版本不同")
    if fingerprint(metadata["identity"]) != preparation_id:
        raise DataError("缓存规则与准备标识不一致")
    for name in CACHE_FILES:
        if file_hash(directory / name) != metadata["files"][name]["sha256"]:
            raise DataError(f"缓存文件指纹不符：{name}")
    for name, shape, dtype in (("X", metadata["shape"], np.dtype("float32")), ("y", metadata["shape"][:1], np.dtype("float64")),
                               ("rows", metadata["shape"][:1], ROW_DTYPE)):
        array = np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        if list(array.shape) != shape or array.dtype != dtype:
            raise DataError(f"缓存数组契约不符：{name}")
        del array
    return metadata



def report_text(result: dict) -> str:
    counts = result.get("counts", {})
    lines = ["# P1样本与特征准备报告", "", f"状态：**{result['state']}**。", "",
             "本阶段生成独立的特征、目标和日期索引；尚未划分七折与留出角色，也没有训练模型。", "",
             "| 项目 | 数量 |", "|---|---|",
             f"| 上市后股票日网格 | {counts.get('grid_rows', 0)} |",
             f"| 信息日具备预测资格 | {counts.get('prediction_eligible', 0)} |",
             f"| 信息日暂停 | {counts.get('information_day_paused', 0)} |",
             f"| 具备预测资格且下一市场日有目标的候选对 | {counts.get('supervised_candidates', 0)} |",
             f"| 具备预测资格但下一市场日暂停 | {counts.get('eligible_target_suspended', 0)} |",
             f"| 具备预测资格但日历尚无下一日 | {counts.get('eligible_target_calendar_unknown', 0)} |",
             f"| 特征列数 | {result.get('shape', [0, 0])[1]} |", "",
             "候选对不是最终训练样本数。P2仍需按目标日和标签成熟条件隔离训练、验证、留出。", "",
             "- [数据与缓存清单](dataset.json)", "- [特征公式与顺序](features.json)",
             "- [逐股覆盖](coverage.csv)", "- [各列有效覆盖](feature_quality.csv)",
             "- [实际读取来源指纹](source_files.csv)", "- [问题记录](issues.csv)", "- [执行环境](experiment.json)", "",
             "股票池仍有期末存续和价格筛选偏差；行业、财务和当前复权绝对价未进入特征。",
             "NaN表示按规格缺少历史或分母不适用，0仍是合法数值；预测资格独立于事后目标状态。", ""]
    if result.get("cache_directory"):
        lines.extend([f"缓存位置（相对模型目录）：`{result['cache_directory']}`。", "",
                      "数组保存在本机.cache；迁移需复制相应缓存和来源，单独克隆Git不能恢复这些数组。", ""])
    return "\n".join(lines)




class NumpyPreparedStore:
    def __init__(self, root: Path, run_dir: Path, context: RunContext):
        self.root, self.run_dir, self.context = root, run_dir, context
        self.cache_root = root / ".cache"

    def initialize(self, feature_manifest):
        save_json(self.run_dir / "features.json", feature_manifest)
        write_csv(self.run_dir / "coverage.csv", COVERAGE_FIELDS, [])
        write_csv(self.run_dir / "feature_quality.csv", QUALITY_FIELDS, [])

    def _reference(self, directory, metadata):
        return StoredSamples(metadata["preparation_id"], directory.relative_to(self.root).as_posix(),
                             file_hash(directory / "prepared.json"), readonly_description(metadata))

    def find_complete(self, identity):
        key = fingerprint(identity)
        directory = self.cache_root / key
        if not directory.exists():
            return None
        return self._reference(directory, validate_cache(directory, key))

    def begin(self, identity, description, feature_manifest):
        self.cache_root.mkdir(exist_ok=True)
        return NumpyWriteSession(self, identity, description, feature_manifest)

    def finish(self, result, receipt, issues, stored):
        if stored is not None:
            directory = self.root / stored.cache_key
            for name in ("coverage.csv", "feature_quality.csv"):
                (self.run_dir / name).write_bytes((directory / name).read_bytes())
        write_csv(self.run_dir / "source_files.csv", SOURCE_FIELDS, receipt.records)
        write_csv(self.run_dir / "issues.csv", ("code", "detail"), issues)
        save_json(self.run_dir / "dataset.json", result)
        (self.run_dir / "report.md").write_text(report_text(result), encoding="utf-8", newline="\n")


class NumpyWriteSession:
    """写会话只追加已算好的块；发布前关闭Windows映射句柄。"""
    def __init__(self, store, identity, description, feature_manifest):
        self.store, self.identity, self.description, self.feature_manifest = store, identity, description, feature_manifest
        self.key = fingerprint(identity)
        self.partial = store.cache_root / ("." + self.key + "." + store.context.run_id + ".partial")
        self.partial.mkdir(exist_ok=False)
        self.offset, self.arrays = 0, {}
        count = planned_rows(description)
        try:
            for name, dtype, shape in (("X", "float32", (count, feature_manifest["count"])), ("y", "float64", (count,)), ("rows", ROW_DTYPE, (count,))):
                self.arrays[name] = open_memmap(self.partial / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        except BaseException:
            self.close()
            raise

    def append(self, chunk):
        end = self.offset + len(chunk.rows)
        self.arrays["X"][self.offset:end] = chunk.values
        self.arrays["y"][self.offset:end] = chunk.labels
        self.arrays["rows"][self.offset:end] = chunk.rows
        self.offset = end

    def close(self):
        # 不把memmap切片泄漏给用例；因此可以在本会话显式释放全部映射。
        for array in self.arrays.values():
            array.flush()
            array._mmap.close()
        self.arrays.clear()

    def publish(self, summary, receipt):
        metadata = summary.complete(self.description)
        if self.offset != metadata["shape"][0] or receipt.changed:
            raise DataError("写入数量或来源凭证不满足发布条件")
        self.close()
        stocks = [{"stock_idx": i, "stock_id": s["exchange"] + s["code"], "code": s["code"], "exchange": s["exchange"], "list_date": s["list_date"]}
                  for i, s in enumerate(self.description.stocks)]
        save_json(self.partial / "stocks.json", stocks)
        save_json(self.partial / "calendar.json", list(self.description.calendar))
        save_json(self.partial / "features.json", self.feature_manifest)
        write_csv(self.partial / "coverage.csv", COVERAGE_FIELDS, summary.coverage)
        write_csv(self.partial / "feature_quality.csv", QUALITY_FIELDS, summary.feature_quality())
        descriptions = {name: {"bytes": (self.partial / name).stat().st_size, "sha256": file_hash(self.partial / name)} for name in CACHE_FILES}
        metadata.update(schema_version=1, state="prepared", preparation_id=self.key, created_by_run=self.store.context.run_id,
                        identity=self.identity, files=descriptions, row_dtype=ROW_DTYPE.descr,
                        target_statuses=self.feature_manifest["target"]["statuses"], unknown_date_index=-1,
                        prediction_eligibility="trade_state == 1 at information date; independent of target status",
                        source_fingerprint=fingerprint(self.identity["input_sha256"]))
        save_json(self.partial / "prepared.json", metadata)
        published = self.store.cache_root / self.key
        self.partial.rename(published)
        return self.store._reference(published, metadata)
