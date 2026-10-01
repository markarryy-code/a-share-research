"""所有研究方法共用的留出使用登记；按P2数据身份追踪，不按方法重置。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from .evaluation_domain import ValidationError
from ..serialization import save_json, fingerprint, file_hash


class LocalHoldoutUsage:
    """先独占登记，再接触目标；更换模型或标签版本不能清零留出历史。"""
    def __init__(self, root: Path, run_dir: Path):
        self.root, self.run_dir = root, run_dir
        self.lock = self.lock_path = self.path = self.record = self.entry = None

    def _save(self):
        temporary = self.path.with_suffix(".partial")
        save_json(temporary, self.record)
        temporary.replace(self.path)

    def reserve(self, description, configuration):
        identity = description.provenance["holdout_identity_sha256"]
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValidationError("留出身份必须为SHA256")
        directory = self.root/"runs/holdout-usage"
        directory.mkdir(exist_ok=True)
        self.path, self.lock_path = directory/(identity+".json"), directory/(identity+".lock")
        try:
            self.lock = self.lock_path.open("x", encoding="utf-8")
        except FileExistsError as error:
            raise ValidationError("同一留出已有运行占用；须核对原运行，不能并行开新评价") from error
        self.record = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "schema_version":1, "holdout_identity_sha256":identity, "window":dict(description.window), "attempts":[]}
        if self.record["holdout_identity_sha256"] != identity:
            raise ValidationError("留出登记身份不一致")
        if any(a["exposure_started"] for a in self.record["attempts"]):
            raise ValidationError("该留出月已经开始过效果评价；请复核原产物，不能再次作为未见测试")
        if any(a["state"] == "reserved" for a in self.record["attempts"]):
            raise ValidationError("留出存在未结束记录，先核对中断状态")
        self.entry = {"run":self.run_dir.relative_to(self.root).as_posix(), "configuration_sha256":fingerprint(configuration),
                      "reserved_at_utc":datetime.now(timezone.utc).isoformat(), "state":"reserved", "exposure_started":False}
        self.record["attempts"].append(self.entry)
        self._save()
        save_json(self.run_dir/"holdout_usage.json", self.record)

    def mark_exposed(self):
        if self.entry is None:
            raise ValidationError("留出尚未预约")
        # 即使随后的读取或评分失败，也保留已开始评价的事实。
        self.entry.update(exposure_started=True, exposure_started_at_utc=datetime.now(timezone.utc).isoformat(),
                          predictions_sha256=file_hash(self.run_dir/"predictions.csv.gz"),
                          model_sha256=file_hash(self.run_dir/"model_artifacts/model.txt"))
        self._save()
        save_json(self.run_dir/"holdout_usage.json", self.record)

    def finish(self, state):
        try:
            if self.entry is not None:
                self.entry.update(state=state, finished_at_utc=datetime.now(timezone.utc).isoformat())
                self._save()
                save_json(self.run_dir/"holdout_usage.json", self.record)
        finally:
            if self.lock is not None:
                self.lock.close()
                self.lock_path.unlink()
                self.lock = None
