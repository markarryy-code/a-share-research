"""原始次日收益回归的训练设置；不接受分类任务。"""

from dataclasses import dataclass
from typing import Mapping
from .evaluation_domain import ValidationError


@dataclass(frozen=True)
class ValidationSettings:
    split_run: str
    feature_set: str
    feature_count: int
    target_id: str
    rounds: int
    patience: int
    batch_size: int
    model_params: Mapping

    @classmethod
    def from_mapping(cls, config):
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValidationError("P3配置格式不符")
        required = {"schema_version", "split_run", "feature_set", "feature_count", "target_id", "num_boost_round",
                    "early_stopping_rounds", "prediction_batch_size", "model_params"}
        if set(config) != required:
            raise ValidationError("P3配置字段缺失或包含未声明字段")
        for name in ("feature_count", "num_boost_round", "early_stopping_rounds", "prediction_batch_size"):
            if type(config[name]) is not int or config[name] <= 0:
                raise ValidationError(f"{name}必须是正整数")
        for name in ("split_run", "feature_set", "target_id"):
            if not isinstance(config[name], str) or not config[name].strip():
                raise ValidationError(f"{name}必须明确指定")
        params = config["model_params"]
        fields = {"boosting", "objective", "metric", "learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
                  "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2", "seed", "data_random_seed",
                  "feature_fraction_seed", "bagging_seed", "device_type", "num_threads", "deterministic", "force_col_wise",
                  "max_bin", "use_missing", "zero_as_missing", "verbosity"}
        if not isinstance(params, dict) or set(params) != fields:
            raise ValidationError("模型参数必须使用已声明的唯一名称，轮数/早停不放入model_params")
        fixed = {"boosting": "gbdt", "device_type": "cpu",
                 "use_missing": True, "zero_as_missing": False, "deterministic": True, "force_col_wise": True}
        if (any(params[k] != v for k, v in fixed.items())
                or (params["objective"], params["metric"]) != ("regression", "rmse")):
            raise ValidationError("模型类型或缺失值/确定性契约不符")
        return cls(config["split_run"], config["feature_set"], config["feature_count"], config["target_id"],
                   config["num_boost_round"], config["early_stopping_rounds"], config["prediction_batch_size"], dict(params))
