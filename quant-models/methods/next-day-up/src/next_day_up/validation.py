"""一次只执行一个明确开发期折，回归与上涨分类保持独立目标和评分。"""

from dataclasses import dataclass
import numpy as np
from daily_return.experiments import ValidationError, check_matrix, fit_volatility_boundaries, FoldSource, RunTelemetry
from daily_return.execution import RunContext
from .domain import up_labels, classification_description
from .evaluation import evaluate_classification_fold
from .ports import ModelTrainer, ValidationArchive


VALIDATION_MODULES = (
    "serialization.py",
    "resources.py",
    "samples/__init__.py",
    "samples/domain.py",
    "samples/ports.py",
    "samples/application.py",
    "samples/prepared_reader.py",
    "samples/numpy_store.py",
    "samples/targets.py",
    "experiments/__init__.py",
    "experiments/domain.py",
    "experiments/ports.py",
    "experiments/application.py",
    "experiments/numpy_store.py",
    "experiments/evaluation_domain.py",
    "experiments/evaluation_ports.py",
    "experiments/validation_source.py",
    "methods/next-day-up/src/next_day_up/__init__.py",
    "methods/next-day-up/src/next_day_up/domain.py",
    "methods/next-day-up/src/next_day_up/ports.py",
    "methods/next-day-up/src/next_day_up/evaluation.py",
    "methods/next-day-up/src/next_day_up/lightgbm_model.py",
    "methods/next-day-up/src/next_day_up/archive.py",
    "methods/next-day-up/src/next_day_up/validation.py",
)


@dataclass(frozen=True)
class ValidationOutcome:
    result: dict
    status: str
    exit_code: int


def validate_fold(source: FoldSource, trainer: ModelTrainer, archive: ValidationArchive, telemetry: RunTelemetry,
                  settings, configuration, fold: str, context: RunContext):
    result = {"stage": "P3", "state": "failed", "fold": fold, "p3_complete": False,
              "model_trained": False, "holdout_evaluation_performed": False, "max_rounds": settings.rounds,
              "task": "binary_classification"}
    issues, receipt, changed = [], (), ()
    status, exit_code = "running", 0
    try:
        telemetry.checkpoint("started")
        description = classification_description(source.describe())
        result["provenance"] = dict(description.provenance)
        result["counts"] = dict(description.counts)
        archive.begin(description, configuration)
        context.progress({"stage": "fold_input_verified", "fold": fold, "features": len(description.feature_names),
                          "rows": {k: v["selected"] for k, v in description.counts.items()}})
        telemetry.checkpoint("input_verified")
        train_x, train_returns = source.training_data()
        valid_x, valid_returns = source.validation_data()
        train_y, valid_y = up_labels(train_returns), up_labels(valid_returns)
        check_matrix(train_x, train_y, len(description.feature_names))
        check_matrix(valid_x, valid_y, len(description.feature_names))
        prior = float(np.mean(train_y))
        result["training_prior"] = {"n": len(train_y), "up_count": int(train_y.sum()), "up_probability": prior,
                                    "majority_class": int(prior > 0.5)}
        boundaries = fit_volatility_boundaries(train_x[:, description.volatility_column])
        telemetry.checkpoint("matrices_selected")
        model = trainer.fit(train_x, train_y, valid_x, valid_y, description.feature_names, settings)
        result["model_trained"] = True
        telemetry.checkpoint("fit_completed")
        del train_x, train_y, valid_x, valid_y, train_returns, valid_returns
        telemetry.checkpoint("matrices_released")
        archive.fitted(model, boundaries)
        count = len(description.prediction_rows)
        predictions, volatility = np.empty(count), np.empty(count)
        trend_complete = np.empty(count, dtype=bool)
        written, reload_check = 0, None
        for batch in source.prediction_batches(settings.batch_size):
            end = batch.offset+len(batch.features)
            if batch.offset != written or end > count:
                raise ValidationError("预测批次有缺口、重复或越界")
            values = model.predict(batch.features)
            if values.shape != (len(batch.features),) or not np.isfinite(values).all():
                raise ValidationError("模型没有返回完整有限的数值预测")
            if np.any((values < 0) | (values > 1)):
                raise ValidationError("分类模型返回了[0,1]以外的概率")
            archive.predictions(batch.rows, values)
            predictions[batch.offset:end] = values
            volatility[batch.offset:end] = batch.features[:, description.volatility_column]
            trend_complete[batch.offset:end] = np.isfinite(batch.features[:, description.trend_columns]).all(axis=1)
            if reload_check is None:
                n = min(2048, len(values))
                reload_check = trainer.check_reload(model.export_text(), description.feature_names, batch.features[:n], values[:n])
            written = end
        if written != count:
            raise ValidationError("未覆盖全部应预测行")
        archive.predictions_finished()
        telemetry.checkpoint("prediction_completed")
        truth = source.prediction_targets()
        metrics = evaluate_classification_fold(description, truth, predictions, volatility, trend_complete, boundaries, prior)
        metric = settings.model_params["metric"]
        report_metric = "logloss"
        native, independent = model.summary["native_best_" + metric], metrics["overall"]["lightgbm"][report_metric]
        # 原生Dataset标签可能采用float32，报告仍使用P1保存的float64目标独立计算。
        if not np.isclose(native, independent, rtol=1e-6, atol=1e-8):
            raise ValidationError("原生最佳指标与独立评分不符，须检查轮数、键和单位")
        metrics["native_metric_check"] = {"metric": metric, "native": native, "independent": independent,
                                           "absolute_difference": abs(native-independent), "rtol": 1e-6, "atol": 1e-8}
        archive.evaluated(truth, predictions, metrics)
        telemetry.checkpoint("scoring_completed")
        receipt, changed = source.verify_unchanged()
        result["inputs_unchanged"] = not changed
        if changed:
            raise ValidationError("训练期间输入发生变化：" + ", ".join(changed))
        telemetry.checkpoint("input_reverified")
        result.update(state="fold-validated", completed_folds=[fold], model_trained=True, inputs_unchanged=True,
                      best_iteration=model.summary["best_iteration"], rounds_run=model.summary["rounds_run"],
                      reload_check=reload_check, metrics=metrics["overall"], label_availability="target day EOD; exact timestamp unverified")
        status = "completed"
    except KeyboardInterrupt:
        result["state"], status, exit_code = "interrupted", "interrupted", 130
        issues.append({"code": "interrupted", "detail": "单折中断，保留已写产物，不能作为完成模型复用"})
    except Exception as error:
        status, exit_code = "failed", 2 if isinstance(error, ValidationError) else 1
        issues.append({"code": type(error).__name__, "detail": str(error)})
    finally:
        source.close()
        if not receipt:
            receipt, changed = source.verify_unchanged()
            result["inputs_unchanged"] = not changed
            issues.extend({"code": "input_changed", "detail": p} for p in changed)
        telemetry.checkpoint("finished")
        archive.finish(result, receipt, issues, telemetry.records())
    return ValidationOutcome(result, status, exit_code)
