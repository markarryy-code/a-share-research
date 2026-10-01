"""单折模型、不可变预测、独立评分与失败记录的文件归档。"""

import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path
from daily_return.serialization import save_json, write_csv, fingerprint, file_hash
from daily_return.resources import summarize_resources


KEY_FIELDS = ("stock_id", "as_of_date", "target_date")
CLASS_METRICS = ("accuracy", "balanced_accuracy", "up_recall", "not_up_recall", "up_precision",
                 "predicted_up_rate", "actual_up_rate", "logloss", "brier", "auc")
CLASS_FIELDS = ("prediction_count", "score_count", *(model + "_" + metric for model in ("lightgbm", "training_prior") for metric in CLASS_METRICS))


def flat_classification(record):
    return {"prediction_count": record.get("prediction_count"), "score_count": record["lightgbm"]["n"],
            **{model + "_" + metric: record[model][metric] for model in ("lightgbm", "training_prior") for metric in CLASS_METRICS}}


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
                                 prediction_unit="up_probability",
                                 volatility_boundaries=list(boundaries), completion_state="fitted_pending_validation")
        save_json(self.directory / "model.json", self.model_record)
        write_csv(self.directory / "learning_curve.csv", ("iteration", "validation_" + model.summary["metric_name"]), model.summary["learning_curve"])
        usage = [{**r, "new_trend_feature": i in self.description.trend_columns} for i, r in enumerate(model.feature_usage())]
        write_csv(self.directory / "feature_usage.csv", ("name", "gain", "split", "new_trend_feature"), usage)
        self.stream = gzip.open(self.directory / "predictions.csv.gz", "wt", encoding="utf-8", newline="")
        values = ("p_up", "predicted_class")
        self.writer = csv.DictWriter(self.stream, fieldnames=(*KEY_FIELDS, "fold", *values, "prediction_eligible", "generated_at_utc", "mode"), lineterminator="\n")
        self.writer.writeheader()

    def _key(self, row):
        d = self.description
        return {"stock_id": d.stock_ids[int(row["stock_idx"])], "as_of_date": d.calendar[int(row["as_of_idx"])],
                "target_date": d.calendar[int(row["target_idx"])]}

    def predictions(self, rows, values):
        timestamp = datetime.now(timezone.utc).isoformat()
        self.writer.writerows({**self._key(row), "fold": self.description.fold,
                               "p_up": float(value), "predicted_class": int(value > 0.5),
                               "prediction_eligible": True, "generated_at_utc": timestamp, "mode": "historical_replay"}
                              for row, value in zip(rows, values, strict=True))

    def predictions_finished(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def evaluated(self, truth, prediction, metrics):
        save_json(self.directory / "metrics.json", metrics)

        write_csv(self.directory / "daily_metrics.csv", ("target_date", *CLASS_FIELDS),
                  [{"target_date": r["target_date"], **flat_classification(r)} for r in metrics["daily"]])
        write_csv(self.directory / "group_metrics.csv", ("grouping", "group", *CLASS_FIELDS),
                  [{"grouping": r["grouping"], "group": r["group"], **flat_classification(r)} for r in metrics["groups"]])
        write_csv(self.directory / "calibration_bins.csv", ("lower", "upper", "upper_inclusive", "n", "mean_probability", "observed_up_rate"), metrics["calibration_bins"])
        fields = (*KEY_FIELDS, "p_up", "predicted_class", "actual_return", "actual_class", "training_up_prior", "score_eligible", "unscorable_reason")
        with gzip.open(self.directory / "predictions.scored.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row, actual, predicted, scored in zip(self.description.prediction_rows, truth, prediction, self.description.score_mask, strict=True):
                writer.writerow({**self._key(row), "p_up": float(predicted), "predicted_class": int(predicted > 0.5),
                                 "actual_return": float(actual) if scored else "", "actual_class": int(actual > 0) if scored else "",
                                 "training_up_prior": metrics["prior_probability"], "score_eligible": bool(scored),
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
        names.append("calibration_bins.csv")
        if result["stage"] == "P4":
            names.append("holdout_usage.json")
        result["artifacts"] = {name: {"sha256": file_hash(self.directory / name), "bytes": (self.directory / name).stat().st_size}
                               for name in names if (self.directory / name).exists()}
        result["resources"] = {k: v for k, v in resource_summary.items() if k != "checkpoints"}
        save_json(self.directory / ("evaluation.json" if result["stage"] == "P4" else "validation.json"), result)
        (self.directory / "report.md").write_text(classification_report(result, issues), encoding="utf-8", newline="\n")


def classification_report(result, issues):
    final = result["stage"] == "P4"
    title = "# P4 次日上涨二分类：当前固定股票池的历史留出评价" if final else "# P3 次日上涨二分类：单折开发期验证"
    interpretation = ("最终轮数在读取留出目标前按七折中位数冻结；固定模型，没有验证Dataset或早停。留出评价已开启时，该月不能再称未见测试。"
                      if final else "本折用于早停，属于开发期成绩；最近一个月未评价。")
    lines = [title, "", f"折：**{result['fold']}**；状态：**{result['state']}**。", "",
             "上涨为真实行情涨幅>0；不上涨包含下跌和平盘；暂停/缺失不填零。概率>0.5才预测上涨。", "",
             interpretation + "概率不是涨幅，也不是已证明的交易胜率。", ""]
    if result["state"] in ("fold-validated", "holdout-evaluated"):
        model, prior = result["metrics"]["lightgbm"], result["metrics"]["training_prior"]
        def display(value, percent=False):
            return "不可计算" if value is None else (f"{value:.4%}" if percent else f"{value:.8f}")
        lines.extend(["| 指标 | LightGBM分类 | 训练期常数概率/多数类 |", "|---|---:|---:|"])
        for name, label, percent in (("accuracy", "准确率", True), ("balanced_accuracy", "平衡准确率", True),
                                     ("up_recall", "上涨识别率", True), ("not_up_recall", "不上涨识别率", True),
                                     ("logloss", "Logloss", False), ("brier", "Brier", False), ("auc", "AUC", False)):
            lines.append(f"| {label} | {display(model[name], percent)} | {display(prior[name], percent)} |")
        lines.extend(["", f"训练期上涨比例：{result['training_prior']['up_probability']:.6%}。",
                      (f"开发期冻结轮数{result['max_rounds']}；实际训练{result['rounds_run']}轮，未在留出期选轮数。" if final
                       else f"最大{result['max_rounds']}轮；实际{result['rounds_run']}轮；最佳{result['best_iteration']}轮。"),
                      f"应预测{result['counts']['predict']['selected']}，评分{result['counts']['score']['selected']}。", ""])
    lines.extend(["- " + item["code"] + "：" + item["detail"] for item in issues])
    lines.extend(["", "- [完整指标、逐日、分组和非平盘诊断](metrics.json)", "- [概率分组](calibration_bins.csv)",
                  "- [模型与训练参数](model.json)", "- [输入、哈希和执行结果](" + ("evaluation.json" if final else "validation.json") + ")", "",
                  "当前股票池有期末存续和价格筛选偏差，行业数据仍为unknown；未调整分类阈值或校准模型。", ""])
    return "\n".join(lines)
