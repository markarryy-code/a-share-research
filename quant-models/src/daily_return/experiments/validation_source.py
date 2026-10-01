"""从具名P2取得单折行号，经samples公开接口读取对应矩阵。"""

import hashlib
import json
from pathlib import Path
import numpy as np

from ..samples import PreparedReader, open_prepared_matrices, DataError
from ..serialization import file_hash, fingerprint
from .numpy_store import sample_key_hash
from .evaluation_domain import (ValidationError, FoldInputSpec, FoldDescription, PredictionBatch,
                                validate_fold_roles, validate_holdout_roles)


class PreparedFoldSource:
    def __init__(self, root: Path, reader: PreparedReader, settings: FoldInputSpec, fold, config_path, config_hash, *, kind="validation", input_records=()):
        self.root, self.reader, self.settings, self.fold = root, reader, settings, fold
        self.kind, self.scoring_enabled = kind, kind == "validation"
        self.session, self.description, self.roles = None, None, {}
        self.records = {config_path: {"path": config_path.relative_to(root).as_posix(), "role": "training_config",
                                     "bytes": config_path.stat().st_size, "sha256": config_hash}}
        self.records.update({root/r["path"]:dict(r) for r in input_records})

    def _bytes(self, path, role, expected=None):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        prior = self.records.get(path)
        if (expected is not None and digest != expected) or (prior and prior["sha256"] != digest):
            raise ValidationError(f"P3输入指纹不符：{path}")
        self.records[path] = {"path": path.relative_to(self.root).as_posix(), "role": role, "bytes": len(data), "sha256": digest}
        return data

    def _json(self, path, role, expected=None):
        return json.loads(self._bytes(path, role, expected).decode("utf-8"))

    def describe(self):
        try:
            run = (self.root / self.settings.split_run).resolve()
            if not run.is_relative_to(self.root / "runs"):
                raise ValidationError("P3须引用模型目录内的具名P2")
            split = self._json(run / "splits.json", "split_run")
            execution = self._json(run / "experiment.json", "split_execution")
            if split.get("state") != "split-ready" or split.get("stage") != "P2" or execution.get("execution_status") != "completed" or execution.get("run_id") != run.name:
                raise ValidationError("P2运行未完成")
            directory = (self.root / split["cache_directory"]).resolve()
            if directory.parent != self.root / ".cache/splits" or directory.name != split["split_id"]:
                raise ValidationError("P2没有引用已发布的索引缓存")
            meta = self._json(directory / "split-ready.json", "split_manifest", split["cache_manifest_sha256"])
            if meta.get("state") != "split-ready" or meta["split_id"] != split["split_id"] or fingerprint(meta["identity"]) != split["split_id"]:
                raise ValidationError("切分身份不一致")
            if split["plan"] != meta["identity"]["plan"] or split["roles"] != meta["roles"]:
                raise ValidationError("P2归档与完成缓存方案不一致")
            if self.kind not in ("validation", "holdout"):
                raise ValidationError("未定义的评价阶段")
            windows = [w for w in split["plan"]["windows"] if w["id"] == self.fold and w["kind"] == self.kind]
            if len(windows) != 1:
                raise ValidationError("请求的窗口与评价阶段不匹配，开发期入口不能使用留出")
            self.session = open_prepared_matrices(self.reader, split["parent_run"])
            index = self.session.index
            prepared = index.stored.metadata
            identity = prepared["identity"]
            if index.stored.preparation_id != split["preparation_id"] or meta["identity"]["preparation_id"] != index.stored.preparation_id or meta["identity"]["prepared_manifest_sha256"] != index.stored.manifest_sha256:
                raise ValidationError("P2与P1准备身份不匹配")
            feature_spec, target_spec = identity["features"], identity["target"]
            source_target = self.settings.target_id
            if (feature_spec["feature_set"], feature_spec["count"], target_spec["target_id"]) != (self.settings.feature_set, self.settings.feature_count, source_target):
                raise ValidationError("P3要求的特征数量、版本或目标与完成样本不符")
            if meta["identity"]["feature_set"] != self.settings.feature_set or meta["identity"]["target_id"] != source_target:
                raise ValidationError("P2特征/目标与P3配置不符")
            names = tuple(f["name"] for f in feature_spec["features"])
            if len(names) != self.settings.feature_count or len(set(names)) != len(names):
                raise ValidationError("特征列顺序清单不完整")
            counts, hashes = {}, {}
            for role in ("train", "predict", "score"):
                selected = [r for r in split["roles"] if r["window"] == self.fold and r["role"] == role]
                if len(selected) != 1 or selected[0]["file"] != f"{self.fold}.{role}.npy":
                    raise ValidationError("单折索引角色或文件名不符")
                record = selected[0]
                path = directory / record["file"]
                self._bytes(path, "fold_" + role, meta["files"][record["file"]]["sha256"])
                indices = np.load(path, allow_pickle=False)
                if indices.shape != (record["counts"]["selected"],):
                    raise ValidationError("索引长度与P2计数不符")
                # 先验证行号范围，再在业务边界核对时点；避免无效行号参与索引摘要。
                if indices.dtype != np.dtype("int64") or not len(indices) or np.any(indices < 0) or np.any(indices >= len(index.rows)):
                    raise ValidationError("单折行号格式或范围不符")
                if sample_key_hash(index.rows, indices) != record["sample_key_sha256"]:
                    raise ValidationError("单折样本键摘要不符")
                indices.flags.writeable = False
                self.roles[role], counts[role], hashes[role] = indices, dict(record["counts"]), record["sample_key_sha256"]
            validator = validate_holdout_roles if self.kind == "holdout" else validate_fold_roles
            mask = validator(index.rows, self.roles, windows[0], index.calendar, index.period["month_start"])
            trend = tuple(i for i, f in enumerate(feature_spec["features"]) if f["family"] == "quote_index_trend")
            if len(trend) != 5:
                raise ValidationError("趋势v2必须具备五个趋势列")
            prediction_rows = index.rows[self.roles["predict"]]
            prediction_rows.flags.writeable = False
            provenance = {"split_run": run.relative_to(self.root).as_posix(), "split_id": split["split_id"],
                          "split_manifest_sha256": split["cache_manifest_sha256"], "parent_run": split["parent_run"],
                          "preparation_id": index.stored.preparation_id, "prepared_manifest_sha256": index.stored.manifest_sha256,
                          "source_fingerprint": prepared["source_fingerprint"], "feature_set": self.settings.feature_set,
                          "target_id": self.settings.target_id, "source_target_id": source_target,
                          "target_spec": dict(target_spec),
                          "feature_spec": feature_spec, "sample_key_sha256": hashes,
                          "holdout_identity_sha256": split["holdout_identity_sha256"], "limitations": index.limitations,
                          "holdout_start": index.period["month_start"]}
            self.description = FoldDescription(self.fold, windows[0], names, trend, names.index("return_std_20"), index.calendar,
                                               tuple(s["stock_id"] for s in index.stocks), prediction_rows, mask, counts, provenance)
            return self.description
        except (OSError, ValueError, KeyError, TypeError, IndexError, DataError) as error:
            raise ValidationError(f"单折输入读取失败：{error}") from error

    def training_data(self):
        return self._supervised_data("train")

    def validation_data(self):
        if self.kind == "holdout":
            raise ValidationError("留出数据不得作为早停或训练验证集")
        return self._supervised_data("score")

    def _supervised_data(self, role):
        returns = self.session.targets_at(self.roles[role])
        return self.session.features_at(self.roles[role]), returns

    def prediction_batches(self, size):
        for offset in range(0, len(self.roles["predict"]), size):
            ids = self.roles["predict"][offset:offset+size]
            yield PredictionBatch(offset, self.session.features_at(ids), self.description.prediction_rows[offset:offset+size])

    def prediction_targets(self):
        # 评分保留原始涨幅和不可评分的NaN，不修改P1数组。
        if not self.scoring_enabled:
            raise ValidationError("留出预测尚未封存和登记使用，不能读取目标数值")
        return self.session.targets_at(self.roles["predict"])

    def enable_holdout_scoring(self):
        self.scoring_enabled = True

    def verify_unchanged(self):
        receipt = self.reader.verify_unchanged()
        records, changed = list(receipt.records), list(receipt.changed)
        for path, item in self.records.items():
            try:
                digest = file_hash(path)
            except OSError:
                digest = "unreadable"
            records.append({**item, "end_sha256": digest, "unchanged": digest == item["sha256"]})
            if digest != item["sha256"]:
                changed.append(item["path"])
        return tuple(records), tuple(changed)

    def close(self):
        if self.session is not None:
            self.session.close()
