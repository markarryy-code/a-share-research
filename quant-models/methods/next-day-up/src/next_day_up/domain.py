"""次日上涨分类的标签、训练设置和最终轮数规则；不读取文件。"""

from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np
from daily_return.experiments import ValidationError


UP_TARGET_SPEC = {
    "target_id": "next-market-day-up-v1",
    "source_target_id": "next-market-day-quote-return-v1",
    "positive": "observed quote return > 0",
    "negative": "observed quote return <= 0, including a genuine flat day",
    "missing": "not a class; preserve prediction eligibility and exclude from scoring",
    "unit": "binary_class", "labels": [0, 1],
}


def up_labels(returns):
    """只转换已观测目标；拒绝把NaN比较结果偷偷变成不上涨。"""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValidationError("上涨标签必须由一维有限涨跌幅生成，缺失目标不能记为0")
    return (values > 0).astype(np.uint8)


@dataclass(frozen=True)
class ClassificationSettings:
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
                or (params["objective"], params["metric"]) != ("binary", "binary_logloss")
                or config["target_id"] != UP_TARGET_SPEC["target_id"]):
            raise ValidationError("模型类型或缺失值/确定性契约不符")
        return cls(config["split_run"], config["feature_set"], config["feature_count"], config["target_id"],
                   config["num_boost_round"], config["early_stopping_rounds"], config["prediction_batch_size"], dict(params))


def freeze_classification_plan(records):
    if [r["fold"] for r in records] != [f"F{i:02}" for i in range(1, 8)]:
        raise ValidationError("最终轮数必须来自七个不同开发期，不能漏折或重复计数")
    first, iterations = records[0]["configuration"], []
    settings = ClassificationSettings.from_mapping(first)
    comparable = {k:v for k,v in first.items() if k != "early_stopping_rounds"}
    identity = records[0]["provenance"]
    for record in records:
        config, provenance = record["configuration"], record["provenance"]
        ClassificationSettings.from_mapping(config)
        if {k:v for k,v in config.items() if k != "early_stopping_rounds"} != comparable:
            raise ValidationError("开发期折的特征或模型参数不一致")
        if any(provenance[k] != identity[k] for k in ("preparation_id", "split_id", "holdout_identity_sha256")):
            raise ValidationError("开发期记录的样本或留出身份不一致")
        best = record["best_iteration"]
        if type(best) is not int or not 1 <= best <= config["num_boost_round"]:
            raise ValidationError("开发期最佳轮数不合法")
        iterations.append(best)
    rounds = sorted(iterations)[3]
    configuration = {**first, "phase":"holdout", "num_boost_round":rounds, "early_stopping_rounds":0,
                     "class_threshold":0.5, "threshold_comparison":">",
                     "round_selection_rule":"median of seven distinct completed classification folds",
                     "development_best_iterations":iterations,
                     "expected_preparation_id":identity["preparation_id"], "expected_split_id":identity["split_id"],
                     "expected_holdout_identity_sha256":identity["holdout_identity_sha256"]}
    return replace(settings, rounds=rounds, patience=0), configuration




def classification_description(description):
    """以共用原始收益输入建立本方法的标签说明，保留原数据身份。"""
    if description.provenance["target_id"] != UP_TARGET_SPEC["source_target_id"]:
        raise ValidationError("次日上涨分类必须接收匹配版本的原始次日收益")
    provenance = {**description.provenance, "target_id": UP_TARGET_SPEC["target_id"],
                  "source_target_id": UP_TARGET_SPEC["source_target_id"], "target_spec": dict(UP_TARGET_SPEC)}
    return replace(description, provenance=provenance)
