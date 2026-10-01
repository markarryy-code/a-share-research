"""读取本方法具名开发期证据；留出使用登记由共用实验模块拥有。"""

import hashlib
import json
from .domain import freeze_classification_plan
from daily_return.experiments import ValidationError


def load_classification_plan(root, selection_path):
    inputs = {}
    def read(path, role, expected=None):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValidationError("开发期引用必须位于模型目录")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValidationError("开发期证据指纹不一致："+str(path))
        inputs[path] = {"path":path.relative_to(root).as_posix(), "role":role, "bytes":len(data), "sha256":digest}
        return json.loads(data.decode("utf-8"))
    selection = read(selection_path, "development_selection")
    references = selection["classification_runs"]
    if set(references) != {f"F{i:02}" for i in range(1, 8)} or len(set(references.values())) != 7:
        raise ValidationError("必须选择七个独立完成的分类运行")
    records = []
    for fold in sorted(references):
        directory = root/references[fold]
        validation = read(directory/"validation.json", "development_result")
        execution = read(directory/"experiment.json", "development_execution")
        if (validation["fold"] != fold or validation["state"] != "fold-validated"
                or validation["task"] != "binary_classification" or not validation["inputs_unchanged"]
                or execution["execution_status"] != "completed"):
            raise ValidationError("开发期运行未完成或任务不符")
        configuration = read(directory/"configuration.json", "development_configuration", validation["artifacts"]["configuration.json"]["sha256"])
        model = read(directory/"model.json", "development_model", validation["artifacts"]["model.json"]["sha256"])
        if model["best_iteration"] != validation["best_iteration"] or model["completion_state"] != "fold-validated":
            raise ValidationError("开发期模型和执行记录不一致")
        records.append({"fold":fold, "configuration":configuration, "provenance":validation["provenance"], "best_iteration":model["best_iteration"]})
    settings, configuration = freeze_classification_plan(records)
    configuration.update(development_selection=selection_path.relative_to(root).as_posix(), selected_development_runs=references,
                         development_evidence_sha256={r["path"]:r["sha256"] for r in inputs.values()})
    return settings, configuration, tuple(inputs.values())
