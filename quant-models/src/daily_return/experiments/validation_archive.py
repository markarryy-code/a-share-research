"""单折模型、不可变预测、独立评分与失败记录的文件归档。"""

import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path
from ..serialization import save_json, write_csv, fingerprint, file_hash
from ..resources import summarize_resources


KEY_FIELDS = ("stock_id", "as_of_date", "target_date")
METRIC_FIELDS = ("prediction_count", "score_count", "lightgbm_rmse_pp", "zero_rmse_pp", "delta_rmse_pp",
                 "lightgbm_mae_pp", "zero_mae_pp", "delta_mae_pp", "mean_error_pp", "direction_accuracy")
def flat_metrics(record):
    model, zero = record["lightgbm"], record["zero"]
    def points(value):
        return value*100 if value is not None else None
    return {"prediction_count": record.get("prediction_count"), "score_count": model["n"],
            "lightgbm_rmse_pp": points(model["rmse"]), "zero_rmse_pp": points(zero["rmse"]), "delta_rmse_pp": points(record["delta_rmse"]),
            "lightgbm_mae_pp": points(model["mae"]), "zero_mae_pp": points(zero["mae"]), "delta_mae_pp": points(record["delta_mae"]),
            "mean_error_pp": points(model["mean_error"]), "direction_accuracy": model["direction_accuracy"]}


class LocalValidationArchive:
    def __init__(self, root: Path, run_dir: Path, context, code_hashes, environment):
        self.root, self.directory, self.context = root, run_dir, context
        self.code_hashes, self.environment = code_hashes, environment
        self.stream, self.description, self.model_record = None, None, None
        self.artifacts = {}

    def begin(self, description, configuration):
        self.description = description
        identity = {"schema_version": 1, "configuration": configuration, "input": description.provenance,
                    "calculation_code_sha256": self.code_hashes,
                    "python": self.context.python_version, "packages": self.environment["installed_training_packages"]}
        self.study_id = fingerprint(identity)
        save_json(self.directory / "configuration.json", configuration)
        save_json(self.directory / "inputs.json", {"study_id": self.study_id, "identity": identity,
                                                  "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                                                  "fold": description.fold, "window": description.window, "counts": description.counts})
        save_json(self.directory / "features.json", description.provenance["feature_spec"])
        self.model_directory = self.directory / "model_artifacts"
        self.model_directory.mkdir(exist_ok=False)

    def fitted(self, model, boundaries):
        path = self.model_directory / "model.txt"
        path.write_text(model.export_text(), encoding="utf-8", newline="\n")
        self.model_record = {k: v for k, v in model.summary.items() if k != "learning_curve"}
        self.model_record.update(model_file="model_artifacts/model.txt", model_sha256=file_hash(path),
                                 study_id=self.study_id, fold=self.description.fold, feature_names=list(self.description.feature_names),
                                 fit_cutoff_date=self.description.window["fit_cutoff_date"],
                                 preparation_id=self.description.provenance["preparation_id"],
                                 split_id=self.description.provenance["split_id"],
                                 target_id=self.description.provenance["target_id"],
                                 prediction_unit="return_fraction",
                                 volatility_boundaries=list(boundaries), completion_state="fitted_pending_validation")
        save_json(self.directory / "model.json", self.model_record)
        write_csv(self.directory / "learning_curve.csv", ("iteration", "validation_" + model.summary["metric_name"]), model.summary["learning_curve"])
        usage = [{**r, "new_trend_feature": i in self.description.trend_columns} for i, r in enumerate(model.feature_usage())]
        write_csv(self.directory / "feature_usage.csv", ("name", "gain", "split", "new_trend_feature"), usage)
        self.stream = gzip.open(self.directory / "predictions.csv.gz", "wt", encoding="utf-8", newline="")
        values = ("y_pred",)
        self.writer = csv.DictWriter(self.stream, fieldnames=(*KEY_FIELDS, "fold", *values, "prediction_eligible", "generated_at_utc", "mode"), lineterminator="\n")
        self.writer.writeheader()

    def _key(self, row):
        d = self.description
        return {"stock_id": d.stock_ids[int(row["stock_idx"])], "as_of_date": d.calendar[int(row["as_of_idx"])],
                "target_date": d.calendar[int(row["target_idx"])]}

    def predictions(self, rows, values):
        timestamp = datetime.now(timezone.utc).isoformat()
        self.writer.writerows({**self._key(row), "fold": self.description.fold,
                               "y_pred": float(value),
                               "prediction_eligible": True, "generated_at_utc": timestamp, "mode": "historical_replay"}
                              for row, value in zip(rows, values, strict=True))

    def predictions_finished(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def evaluated(self, truth, prediction, metrics):
        save_json(self.directory / "metrics.json", metrics)
        write_csv(self.directory / "daily_metrics.csv", ("target_date", *METRIC_FIELDS),
                  [{"target_date": r["target_date"], **flat_metrics(r)} for r in metrics["daily"]])
        write_csv(self.directory / "group_metrics.csv", ("grouping", "group", *METRIC_FIELDS),
                  [{"grouping": r["grouping"], "group": r["group"], **flat_metrics(r)} for r in metrics["groups"]])
        fields = (*KEY_FIELDS, "y_pred", "y_true", "score_eligible", "error", "zero_error", "unscorable_reason")
        with gzip.open(self.directory / "predictions.scored.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row, actual, predicted, scored in zip(self.description.prediction_rows, truth, prediction, self.description.score_mask, strict=True):
                writer.writerow({**self._key(row), "y_pred": float(predicted), "y_true": float(actual) if scored else "",
                                 "score_eligible": bool(scored), "error": float(predicted-actual) if scored else "",
                                 "zero_error": float(-actual) if scored else "",
                                 "unscorable_reason": "" if scored else ("target_suspended" if row["target_status"] == 2 else "label_not_mature")})

    def finish(self, result, receipt, issues, resources):
        self.predictions_finished()
        resource_summary = summarize_resources(resources)
        save_json(self.directory / "resources.json", resource_summary)
        write_csv(self.directory / "source_files.csv", ("path", "role", "bytes", "sha256", "end_sha256", "unchanged"), receipt)
        write_csv(self.directory / "issues.csv", ("code", "detail"), issues)
        if self.model_record is not None:
            self.model_record["completion_state"] = result["state"]
            save_json(self.directory / "model.json", self.model_record)
        names = ["configuration.json", "inputs.json", "features.json", "model.json", "model_artifacts/model.txt",
                 "learning_curve.csv", "feature_usage.csv", "predictions.csv.gz", "predictions.scored.csv.gz",
                 "metrics.json", "daily_metrics.csv", "group_metrics.csv", "resources.json", "source_files.csv", "issues.csv"]
        if result["stage"] == "P4":
            names.append("holdout_usage.json")
        result["artifacts"] = {name: {"sha256": file_hash(self.directory / name), "bytes": (self.directory / name).stat().st_size}
                               for name in names if (self.directory / name).exists()}
        result["resources"] = {k: v for k, v in resource_summary.items() if k != "checkpoints"}
        save_json(self.directory / ("evaluation.json" if result["stage"] == "P4" else "validation.json"), result)
        (self.directory / "report.md").write_text(report_text(result, issues), encoding="utf-8", newline="\n")


def report_text(result, issues):
    lines = ["# P3 单折开发期验证", "", f"折：**{result['fold']}**；状态：**{result['state']}**。", "",
             "本次只执行具名开发期折，七折尚未全部完成，留出月未进行效果评价。", ""]
    if result["state"] == "fold-validated":
        comparison = result["metrics"]
        model, zero = comparison["lightgbm"], comparison["zero"]
        conclusion = "本折RMSE低于恒0参照" if comparison["delta_rmse"] < 0 else "本折未显示相对恒0参照的RMSE改善"
        lines.extend([f"**{conclusion}。** 这是用于早停的开发期成绩，当前固定股票池含期末存续与价格筛选偏差。", "",
                      "| 指标 | LightGBM | 恒0参照 |", "|---|---:|---:|",
                      f"| RMSE（百分点） | {model['rmse']*100:.6f} | {zero['rmse']*100:.6f} |",
                      f"| MAE（百分点） | {model['mae']*100:.6f} | {zero['mae']*100:.6f} |",
                      f"| 平均偏差（百分点） | {model['mean_error']*100:.6f} | {zero['mean_error']*100:.6f} |",
                      f"| 涨/平/跌方向一致率 | {model['direction_accuracy']:.4%} | {zero['direction_accuracy']:.4%} |", "",
                      f"最大轮数{result['max_rounds']}；实际完成{result['rounds_run']}轮验证，最佳轮数{result['best_iteration']}。",
                      f"训练行{result['counts']['train']['selected']}，应预测{result['counts']['predict']['selected']}，可评分{result['counts']['score']['selected']}。", ""])
    if issues:
        lines.extend(["本次问题：", "", *["- " + item["code"] + "：" + item["detail"] for item in issues], ""])
    links = (("model.json", "模型及最佳轮数"), ("metrics.json", "指标与分组"), ("daily_metrics.csv", "逐日结果"),
             ("group_metrics.csv", "分组结果"), ("feature_usage.csv", "特征使用情况"), ("learning_curve.csv", "学习曲线"),
             ("resources.json", "资源记录"), ("source_files.csv", "实际输入指纹"), ("issues.csv", "问题记录"))
    lines.extend(["- [完整执行结果与产物哈希](validation.json)",
                  *[f"- [{label}]({name})" for name, label in links if name in result.get("artifacts", {})], "",
                  "预测明细和模型保存在本机；原预测与评分分开记录。特征使用统计不等于新增五项的增量效果证明。", ""])
    return "\n".join(lines)
