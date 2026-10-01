"""LightGBM CPU适配器；模型只取得既定矩阵，不能读资料或自行选择行。"""

import numpy as np
import lightgbm as lgb
from daily_return.experiments import ValidationError


class NativeModel:
    def __init__(self, booster, summary):
        self.booster, self.summary = booster, summary

    @classmethod
    def from_saved(cls, text, summary):
        """以已保存的轮数和列顺序恢复模型，不触发拟合。"""
        booster = lgb.Booster(model_str=text)
        if booster.feature_name() != summary["feature_names"] or booster.num_trees() != summary["best_iteration"]:
            raise ValidationError("保存模型的特征或树数与记录不一致")
        return cls(booster, summary)

    def predict(self, features):
        return np.asarray(self.booster.predict(features, num_iteration=self.summary["best_iteration"],
                                              num_threads=self.summary["model_parameters"]["num_threads"]), dtype=np.float64)

    def export_text(self):
        return self.booster.model_to_string(num_iteration=self.summary["best_iteration"])

    def feature_usage(self):
        best = self.summary["best_iteration"]
        gain = self.booster.feature_importance("gain", iteration=best)
        split = self.booster.feature_importance("split", iteration=best)
        return [{"name": name, "gain": float(gain[i]), "split": int(split[i])} for i, name in enumerate(self.booster.feature_name())]


class LightGBMTrainer:
    def __init__(self, telemetry, progress):
        self.telemetry, self.progress = telemetry, progress

    def fit_fixed(self, train_x, train_y, names, settings):
        """最终重训没有验证参数入口，不创建留出Dataset或早停回调。"""
        parameters = dict(settings.model_params)
        self.threads = parameters["num_threads"]
        train = lgb.Dataset(train_x, label=train_y, feature_name=list(names), params=parameters, free_raw_data=True)
        train.construct()
        self.telemetry.checkpoint("datasets_constructed")
        def progress_callback(event):
            if event.iteration == 0 or (event.iteration+1) % 25 == 0:
                self.telemetry.checkpoint("training_iteration")
                self.progress({"stage":"fixed_fit", "iteration":event.iteration+1, "requested_rounds":settings.rounds,
                               "validation_dataset_supplied":False})
        booster = lgb.train(parameters, train, num_boost_round=settings.rounds, callbacks=[progress_callback], keep_training_booster=True)
        actual = booster.current_iteration()
        if not 1 <= actual <= settings.rounds or booster.feature_name() != list(names):
            raise ValidationError("固定轮数重训的实际树数或特征顺序不符")
        summary = {"fit_mode":"fixed_rounds", "best_iteration":actual, "rounds_run":actual, "max_rounds":settings.rounds,
                   "iteration_selection":"development-fold median, not holdout optimization", "early_stopping_rounds":0,
                   "early_stopping_enabled":False, "validation_dataset_supplied":False,
                   "metric_name":parameters["metric"], "learning_curve":[], "model_parameters":dict(booster.params),
                   "lightgbm_version":lgb.__version__, "dataset_rows":{"train":train.num_data(), "validation":0}}
        booster.free_dataset()
        return NativeModel(booster, summary)

    def fit(self, train_x, train_y, valid_x, valid_y, names, settings):
        parameters = dict(settings.model_params)
        self.threads = parameters["num_threads"]
        train = lgb.Dataset(train_x, label=train_y, feature_name=list(names), params=parameters, free_raw_data=True)
        validation = lgb.Dataset(valid_x, label=valid_y, reference=train, feature_name=list(names), params=parameters, free_raw_data=True)
        train.construct()
        validation.construct()
        self.telemetry.checkpoint("datasets_constructed")
        history = {}
        metric = parameters["metric"]
        metric_field = "validation_" + metric

        def progress_callback(event):
            if event.iteration == 0 or (event.iteration+1) % 25 == 0:
                sample = self.telemetry.checkpoint("training_iteration")
                self.progress({"stage": "fit", "iteration": event.iteration+1,
                               metric_field: float(event.evaluation_result_list[0][2]),
                               "elapsed_seconds": round(sample["elapsed_seconds"], 2),
                               "working_set_bytes": sample["working_set_bytes"], "peak_working_set_bytes": sample["peak_working_set_bytes"]})
        progress_callback.order = 20
        booster = lgb.train(parameters, train, num_boost_round=settings.rounds, valid_sets=[validation], valid_names=["validation"],
                            callbacks=[lgb.record_evaluation(history), progress_callback,
                                       lgb.early_stopping(settings.patience, first_metric_only=True, verbose=False, min_delta=0.0)],
                            keep_training_booster=True)
        best = int(booster.best_iteration)
        if not 0 < best <= settings.rounds or booster.feature_name() != list(names):
            raise ValidationError("模型最佳轮数或特征顺序不符")
        curve = history["validation"][metric]
        summary = {"best_iteration": best, "rounds_run": len(curve), "max_rounds": settings.rounds,
                   "early_stopping_rounds": settings.patience, "metric_name": metric, "native_best_" + metric: float(curve[best-1]),
                   "learning_curve": [{"iteration": i+1, metric_field: float(value)} for i, value in enumerate(curve)],
                   "model_parameters": dict(booster.params), "lightgbm_version": lgb.__version__,
                   "dataset_rows": {"train": train.num_data(), "validation": validation.num_data()},
                   "validation_reference_is_train": validation.reference is train}
        booster.free_dataset()
        return NativeModel(booster, summary)

    def check_reload(self, text, names, features, expected):
        restored = lgb.Booster(model_str=text)
        if restored.feature_name() != list(names):
            raise ValidationError("保存模型的列顺序改变")
        prediction = restored.predict(features, num_threads=self.threads)
        if not np.allclose(prediction, expected, rtol=1e-12, atol=1e-12):
            raise ValidationError("模型保存加载后预测不一致")
        return {"verified_rows": len(expected), "rtol": 1e-12, "atol": 1e-12,
                "maximum_absolute_difference": float(np.max(np.abs(prediction-expected)))}
