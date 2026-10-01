"""P2索引与运行归档；完成索引另存版本，原样本缓存只读。"""

import hashlib
import json
from pathlib import Path
import numpy as np

from .domain import SplitError, plan_description
from ..serialization import save_json, write_csv, file_hash, fingerprint


COUNT_FIELDS = ("period_rows", "information_ineligible", "prediction_eligible", "target_unavailable", "label_not_mature", "selected")
ROLE_FIELDS = ("window", "role", *COUNT_FIELDS)
DAILY_FIELDS = ("window", "role", "target_date", *COUNT_FIELDS)


def sample_key_hash(rows, indices):
    digest = hashlib.sha256()
    for start in range(0, len(indices), 65536):
        block = rows[indices[start:start+65536]]
        keys = np.column_stack([block[name] for name in ("stock_idx", "as_of_idx", "target_idx")]).astype("<i4", copy=False)
        digest.update(keys.tobytes(order="C"))
    return digest.hexdigest()


class NumpySplitStore:
    def __init__(self, root: Path, run_dir: Path, context, calculation_hashes, config_path: Path, config_hash: str):
        self.root, self.run_dir, self.context = root, run_dir, context
        self.calculation_hashes = calculation_hashes
        self.config_path, self.config_hash = config_path, config_hash
        self.cache_root = root / ".cache" / "splits"

    def identity(self, population, plan):
        return {"schema_version": 1, "preparation_id": population.preparation_id,
                "prepared_manifest_sha256": population.manifest_sha256,
                "source_fingerprint": population.source_fingerprint, "feature_set": population.feature_set,
                "target_id": population.target_id, "population_rows": len(population.rows),
                "plan": plan_description(plan, population), "index_dtype": "int64",
                "python": self.context.python_version, "numpy": self.context.numpy_version,
                "calculation_code_sha256": self.calculation_hashes}

    def _verify_config(self):
        if file_hash(self.config_path) != self.config_hash:
            raise SplitError("P2执行期间时间方案配置发生变化")

    def _reference(self, directory, metadata):
        return {"split_id": metadata["split_id"], "cache_directory": directory.relative_to(self.root).as_posix(),
                "cache_manifest_sha256": file_hash(directory / "split-ready.json"),
                "preparation_id": metadata["identity"]["preparation_id"],
                "plan": metadata["identity"]["plan"], "roles": metadata["roles"],
                "holdout_identity": metadata["holdout_identity"], "holdout_identity_sha256": metadata["holdout_identity_sha256"]}

    def find_complete(self, identity):
        key = fingerprint(identity)
        directory = self.cache_root / key
        if not directory.exists():
            return None
        try:
            metadata = json.loads((directory / "split-ready.json").read_text(encoding="utf-8"))
            if metadata.get("state") != "split-ready" or metadata.get("split_id") != key or fingerprint(metadata["identity"]) != key:
                raise SplitError("时间索引缓存未完成或身份不符")
            expected = [(w["id"], role) for w in identity["plan"]["windows"] for role in ("train", "predict", "score")]
            if [(r["window"], r["role"]) for r in metadata["roles"]] != expected:
                raise SplitError("时间索引缓存缺少规定窗口或角色")
            if set(metadata["files"]) != {w + "." + role + ".npy" for w, role in expected} | {"role_coverage.csv", "daily_coverage.csv"}:
                raise SplitError("时间索引缓存文件集合不完整")
            for name, item in metadata["files"].items():
                if Path(name).name != name or file_hash(directory / name) != item["sha256"]:
                    raise SplitError(f"时间索引缓存指纹不符：{name}")
            for role in metadata["roles"]:
                indices = np.load(directory / role["file"], allow_pickle=False)
                if (indices.dtype != np.dtype("<i8") or indices.shape != (role["counts"]["selected"],)
                        or np.any(indices < 0) or np.any(indices >= identity["population_rows"])
                        or np.any(indices[1:] <= indices[:-1])):
                    raise SplitError("时间索引格式、顺序或范围不符")
            self._verify_config()
            return self._reference(directory, metadata)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SplitError(f"读取时间索引缓存失败：{error}") from error

    def begin(self, population, plan, identity):
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return NumpySplitWriter(self, population, plan, identity)

    def finish(self, result, receipt, issues):
        if result["state"] == "split-ready":
            directory = self.root / result["cache_directory"]
            for name in ("role_coverage.csv", "daily_coverage.csv"):
                (self.run_dir / name).write_bytes((directory / name).read_bytes())
        else:
            write_csv(self.run_dir / "role_coverage.csv", ROLE_FIELDS, [])
            write_csv(self.run_dir / "daily_coverage.csv", DAILY_FIELDS, [])
        write_csv(self.run_dir / "source_files.csv", ("path", "role", "bytes", "sha256", "end_sha256", "unchanged"), receipt)
        write_csv(self.run_dir / "issues.csv", ("code", "detail"), issues)
        save_json(self.run_dir / "splits.json", {**result, "configuration": {"path": str(self.config_path.relative_to(self.root)), "sha256": self.config_hash}})
        (self.run_dir / "report.md").write_text(report_text(result), encoding="utf-8", newline="\n")


class NumpySplitWriter:
    def __init__(self, store, population, plan, identity):
        self.store, self.population, self.identity = store, population, identity
        self.key = fingerprint(identity)
        self.partial = store.cache_root / ("." + self.key + "." + store.context.run_id + ".partial")
        self.partial.mkdir(exist_ok=False)
        self.roles, self.coverage, self.daily, self.files = [], [], [], {}
        self.expected_windows = tuple(w.name for w in plan.windows)

    def append(self, selection):
        for role in selection.roles:
            name = selection.window.name + "." + role.name + ".npy"
            np.save(self.partial / name, role.indices, allow_pickle=False)
            self.roles.append({"window": selection.window.name, "role": role.name, "file": name,
                               "counts": dict(role.counts), "sample_key_sha256": sample_key_hash(self.population.rows, role.indices)})
            self.coverage.append({"window": selection.window.name, "role": role.name, **role.counts})
            self.daily.extend({"window": selection.window.name, "role": role.name, **day} for day in role.daily)

    def publish(self, receipt):
        self.store._verify_config()
        expected = [(window, role) for window in self.expected_windows for role in ("train", "predict", "score")]
        if [(r["window"], r["role"]) for r in self.roles] != expected or not receipt:
            raise SplitError("时间索引角色或读取凭证不完整")
        write_csv(self.partial / "role_coverage.csv", ROLE_FIELDS, self.coverage)
        write_csv(self.partial / "daily_coverage.csv", DAILY_FIELDS, self.daily)
        names = [r["file"] for r in self.roles] + ["role_coverage.csv", "daily_coverage.csv"]
        files = {name: {"sha256": file_hash(self.partial / name), "bytes": (self.partial / name).stat().st_size} for name in names}
        holdout = {"stock_pool_sha256": fingerprint(self.population.stock_ids), "target_id": self.population.target_id,
                   "start": self.population.period["month_start"], "end": self.population.period["end"]}
        metadata = {"schema_version": 1, "state": "split-ready", "split_id": self.key, "created_by_run": self.store.context.run_id,
                    "identity": self.identity, "files": files, "roles": self.roles,
                    "sample_key_encoding": "little-endian int32(stock_idx,as_of_idx,target_idx); dictionaries belong to preparation_id",
                    "holdout_identity": holdout, "holdout_identity_sha256": fingerprint(holdout)}
        save_json(self.partial / "split-ready.json", metadata)
        directory = self.store.cache_root / self.key
        self.partial.rename(directory)
        return self.store._reference(directory, metadata)


def report_text(result):
    lines = ["# P2 时间切分与归档", "", f"状态：**{result['state']}**。", "",
             "按目标交易日生成扩展窗口；拟合只用截止日已成熟标签。所有索引引用原P1数组，原样本缓存不写入。", "",
             "| 时间窗口 | 训练行 | 应预测行 | 可评分行 |", "|---|---:|---:|---:|"]
    roles = result.get("roles", [])
    for window in dict.fromkeys(r["window"] for r in roles):
        sizes = {r["role"]: r["counts"]["selected"] for r in roles if r["window"] == window}
        lines.append(f"| {window} | {sizes['train']} | {sizes['predict']} | {sizes['score']} |")
    lines.extend(["", "holdout的train是最终重训候选索引，标签目标日严格早于留出月。不同折训练期可以扩展重叠，验证目标日期互不重叠。", "",
                  "本运行没有拟合模型、选择参数或进行留出效果评价；这不是对该月份全部历史使用情况的重置。", "",
                  "- [方案、父运行、角色计数与样本键摘要](splits.json)", "- [角色覆盖与排除原因](role_coverage.csv)",
                  "- [逐日角色覆盖](daily_coverage.csv)", "- [完成产物读取指纹](source_files.csv)",
                  "- [执行环境](experiment.json)", "- [问题记录](issues.csv)", "",
                  "标签可得采用目标日收盘资料已可得的日级约定，未认证精确发布时间；股票池的期末存续与价格筛选偏差仍保留。", ""])
    if "cache_directory" in result:
        lines.append("本机索引缓存（相对模型目录）：`" + result["cache_directory"] + "`。\n")
    return "\n".join(lines)
