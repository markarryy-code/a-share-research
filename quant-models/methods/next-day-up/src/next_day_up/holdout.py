"""冻结最终分类模型、封存预测后才开启留出目标评分。"""

import numpy as np
from .validation import ValidationOutcome, VALIDATION_MODULES
from daily_return.experiments import (ValidationError, check_matrix, fit_volatility_boundaries,
                                      HoldoutSource, RunTelemetry, HoldoutUsage)
from daily_return.execution import RunContext
from .domain import up_labels, classification_description
from .ports import FinalTrainer, ValidationArchive
from .evaluation import evaluate_classification_fold


HOLDOUT_MODULES = (*VALIDATION_MODULES, "experiments/holdout_usage.py",
                   "methods/next-day-up/src/next_day_up/holdout.py",
                   "methods/next-day-up/src/next_day_up/development_source.py")


def evaluate_holdout(source: HoldoutSource, trainer: FinalTrainer, archive: ValidationArchive, usage: HoldoutUsage,
                     telemetry: RunTelemetry, settings, configuration, context: RunContext):
    result = {"stage":"P4", "state":"failed", "fold":"holdout", "task":"binary_classification",
              "model_trained":False, "holdout_evaluation_performed":False, "max_rounds":settings.rounds,
              "early_stopping_enabled":False, "class_threshold":0.5}
    receipt, issues, status, exit_code = (), [], "running", 0
    try:
        telemetry.checkpoint("started")
        description = classification_description(source.describe())
        for name in ("preparation_id", "split_id", "holdout_identity_sha256"):
            if description.provenance[name] != configuration["expected_"+name]:
                raise ValidationError("最终数据与开发期冻结身份不一致："+name)
        result.update(provenance=dict(description.provenance), counts=dict(description.counts))
        archive.begin(description, configuration)
        usage.reserve(description, configuration)
        context.progress({"stage":"holdout_plan_frozen", "rounds":settings.rounds,
                          "fit_cutoff":description.window["fit_cutoff_date"], "target_values_read":False})
        telemetry.checkpoint("input_verified")
        train_x, train_returns = source.training_data()
        train_y = up_labels(train_returns)
        check_matrix(train_x, train_y, len(description.feature_names))
        prior = float(np.mean(train_y))
        result["training_prior"] = {"n":len(train_y), "up_count":int(train_y.sum()), "up_probability":prior, "majority_class":int(prior > .5)}
        boundaries = fit_volatility_boundaries(train_x[:, description.volatility_column])
        telemetry.checkpoint("matrices_selected")
        model = trainer.fit_fixed(train_x, train_y, description.feature_names, settings)
        result["model_trained"] = True
        telemetry.checkpoint("fit_completed")
        del train_x, train_y, train_returns
        telemetry.checkpoint("matrices_released")
        archive.fitted(model, boundaries)
        count = len(description.prediction_rows)
        predictions, volatility, trend_complete = np.empty(count), np.empty(count), np.empty(count, dtype=bool)
        written, maximum = 0, 0.0
        for batch in source.prediction_batches(settings.batch_size):
            end = batch.offset+len(batch.features)
            if batch.offset != written or end > count:
                raise ValidationError("留出预测批次有缺口或重复")
            values = model.predict(batch.features)
            if values.shape != (len(batch.features),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValidationError("留出上涨概率不完整或越界")
            checked = trainer.check_reload(model.export_text(), description.feature_names, batch.features, values)
            maximum = max(maximum, checked["maximum_absolute_difference"])
            archive.predictions(batch.rows, values)
            predictions[batch.offset:end] = values
            volatility[batch.offset:end] = batch.features[:, description.volatility_column]
            trend_complete[batch.offset:end] = np.isfinite(batch.features[:, description.trend_columns]).all(axis=1)
            written = end
        if written != count:
            raise ValidationError("留出预测未覆盖全部应预测行")
        archive.predictions_finished()
        telemetry.checkpoint("prediction_completed")
        usage.mark_exposed()
        result["holdout_evaluation_performed"] = True
        source.enable_holdout_scoring()
        truth = source.prediction_targets()
        metrics = evaluate_classification_fold(description, truth, predictions, volatility, trend_complete, boundaries, prior)
        archive.evaluated(truth, predictions, metrics)
        telemetry.checkpoint("scoring_completed")
        receipt, changed = source.verify_unchanged()
        result["inputs_unchanged"] = not changed
        if changed:
            raise ValidationError("最终拟合或评价期间输入变化")
        telemetry.checkpoint("input_reverified")
        result.update(state="holdout-evaluated", best_iteration=model.summary["best_iteration"], rounds_run=model.summary["rounds_run"],
                      metrics=metrics["overall"], reload_check={"verified_rows":count, "maximum_absolute_difference":maximum},
                      label_availability="target day EOD; exact timestamp unverified")
        status = "completed"
    except KeyboardInterrupt:
        result["state"], status, exit_code = "interrupted", "interrupted", 130
        issues.append({"code":"interrupted", "detail":"保留留出使用状态与未完成产物"})
    except Exception as error:
        status, exit_code = "failed", 2 if isinstance(error, ValidationError) else 1
        issues.append({"code":type(error).__name__, "detail":str(error)})
    finally:
        source.close()
        if not receipt:
            receipt, changed = source.verify_unchanged()
            result["inputs_unchanged"] = not changed
        usage.finish(result["state"])
        telemetry.checkpoint("finished")
        archive.finish(result, receipt, issues, telemetry.records())
    return ValidationOutcome(result, status, exit_code)
