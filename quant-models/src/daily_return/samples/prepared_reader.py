"""具名完成样本的文件读取实现；提供索引视图及按行选择的只读矩阵会话。"""

import json
from pathlib import Path
import numpy as np

from .domain import DataError, PreparedIndex, StoredSamples, ReadReceipt
from .numpy_store import validate_cache, readonly_description
from ..serialization import file_hash, fingerprint


class NumpyPreparedReader:
    def __init__(self, root: Path):
        self.root = root
        self.records = {}

    def _record(self, path, digest=None):
        digest = digest or file_hash(path)
        if path in self.records and self.records[path]["sha256"] != digest:
            raise DataError(f"完成样本再次读取时发生变化：{path}")
        self.records[path] = {"path": path.relative_to(self.root).as_posix(), "role": "prepared_input",
                              "bytes": path.stat().st_size, "sha256": digest}

    def _json(self, path):
        data = path.read_bytes()
        # JSON按实际读取字节记账，结束凭证会核对文件是否变化。
        import hashlib
        self._record(path, hashlib.sha256(data).hexdigest())
        return json.loads(data.decode("utf-8"))

    def read_index(self, reference):
        try:
            run = (self.root / reference).resolve()
            if not run.is_relative_to(self.root / "runs"):
                raise DataError("必须指定模型目录内的具名P1运行")
            dataset = self._json(run / "dataset.json")
            execution = self._json(run / "experiment.json")
            if (dataset.get("stage"), dataset.get("state"), dataset.get("inputs_unchanged")) != ("P1", "prepared", True):
                raise DataError("指定运行不是已完成且来源未变的P1")
            if execution.get("stage") != "P1" or execution.get("run_id") != run.name or execution.get("execution_status") != "completed":
                raise DataError("P1运行归档尚未完成或标识不符")
            key = dataset["preparation_id"]
            directory = (self.root / dataset["cache_directory"]).resolve()
            if directory.parent != self.root / ".cache" or directory.name != key:
                raise DataError("P1必须引用已发布的样本缓存")
            metadata = self._json(directory / "prepared.json")
            if self.records[directory / "prepared.json"]["sha256"] != dataset["cache_manifest_sha256"]:
                raise DataError("P1运行与缓存清单指纹不一致")
            if metadata["identity"].get("schema_version") != 2:
                raise DataError("只读取现行DDD样本身份格式")
            validate_cache(directory, key)
            if metadata["shape"] != dataset["shape"] or metadata["counts"] != dataset["counts"]:
                raise DataError("P1运行与缓存样本统计不一致")
            for name, item in metadata["files"].items():
                self._record(directory / name, item["sha256"])
            rows = np.load(directory / "rows.npy", allow_pickle=False)
            rows.flags.writeable = False
            stocks = self._json(directory / "stocks.json")
            calendar = tuple(self._json(directory / "calendar.json"))
            if len(stocks) != metadata["stock_count"] or len(calendar) != metadata["market_days"]:
                raise DataError("样本身份或日历字典长度不符")
            if [s["stock_idx"] for s in stocks] != list(range(len(stocks))) or len({s["stock_id"] for s in stocks}) != len(stocks):
                raise DataError("股票字典序号或身份重复")
            stored = StoredSamples(key, dataset["cache_directory"], dataset["cache_manifest_sha256"], readonly_description(metadata))
            return PreparedIndex(stored, run.relative_to(self.root).as_posix(), calendar, readonly_description(stocks),
                                 readonly_description(dataset["range"]),
                                 readonly_description({k: dataset[k] for k in ("survivorship_bias", "price_selection_bias")}), rows)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DataError(f"读取完成样本失败：{error}") from error

    def verify_unchanged(self):
        records, changed = [], []
        for path, item in sorted(self.records.items()):
            try:
                actual = file_hash(path)
            except OSError:
                actual = "unreadable"
            stable = actual == item["sha256"]
            records.append({**item, "end_sha256": actual, "unchanged": stable})
            if not stable:
                changed.append(item["path"])
        return ReadReceipt(tuple(readonly_description(r) for r in records), tuple(changed),
                           fingerprint({r["path"]: r["sha256"] for r in records}))

    def open_matrices(self, index):
        return NumpyMatrixSession(self.root / index.stored.cache_key, index)


class NumpyMatrixSession:
    """只读映射不外泄；高级索引生成独立副本，可在释放映射后使用。"""
    def __init__(self, directory, index):
        self.index, self.arrays = index, {}
        try:
            for name in ("X", "y"):
                self.arrays[name] = np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        except BaseException:
            self.close()
            raise

    def features_at(self, indices):
        return self.arrays["X"][indices]

    def targets_at(self, indices):
        return self.arrays["y"][indices]

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()
