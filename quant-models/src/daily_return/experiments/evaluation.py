"""原始涨跌幅误差与分组；float64累计，同样本比较恒0参照。"""

import numpy as np
from .evaluation_domain import ValidationError


def metric_values(truth, prediction):
    if not len(truth):
        return {"n": 0, "squared_error_sum": 0.0, "absolute_error_sum": 0.0, "error_sum": 0.0,
                "rmse": None, "mae": None, "mean_error": None, "direction_accuracy": None}
    error = prediction - truth
    squared, absolute, signed = float(np.dot(error, error)), float(np.abs(error).sum()), float(error.sum())
    return {"n": len(truth), "squared_error_sum": squared, "absolute_error_sum": absolute, "error_sum": signed,
            "rmse": float(np.sqrt(squared/len(truth))), "mae": absolute/len(truth), "mean_error": signed/len(truth),
            "direction_accuracy": float(np.mean(np.sign(prediction) == np.sign(truth)))}


def compare_predictions(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValidationError("评分向量形状或有限性不符")
    model, zero = metric_values(truth, prediction), metric_values(truth, np.zeros(len(truth)))
    result = {"lightgbm": model, "zero": zero, "delta_rmse": None, "delta_mae": None, "relative_rmse_improvement": None}
    if len(truth):
        result.update(delta_rmse=model["rmse"]-zero["rmse"], delta_mae=model["mae"]-zero["mae"],
                      relative_rmse_improvement=1-model["rmse"]/zero["rmse"] if zero["rmse"] > 0 else None)
    return result


def evaluate_fold(description, truth, predictions, volatility, trend_complete, boundaries):
    score = description.score_mask
    if len(predictions) != len(score) or not np.isfinite(predictions).all():
        raise ValidationError("应预测集合没有完整有限预测")
    if not score.any() or not np.isfinite(truth[score]).all():
        raise ValidationError("本折没有完整可评分目标")
    comparison = compare_predictions(truth[score], predictions[score])
    rows, daily, groups = description.prediction_rows, [], []
    for index in range(description.calendar.index(description.window["first_target_date"]),
                       description.calendar.index(description.window["last_target_date"])+1):
        predicted = rows["target_idx"] == index
        selected = predicted & score
        daily.append({"target_date": description.calendar[index], "prediction_count": int(predicted.sum()),
                      "score_count": int(selected.sum()), **compare_predictions(truth[selected], predictions[selected])})
    categories = np.searchsorted(np.array(boundaries), volatility, side="right")
    finite = np.isfinite(volatility)
    for number in range(len(boundaries)+1):
        subset = finite & (categories == number)
        selected = score & subset
        groups.append({"grouping": "training_volatility", "group": f"Q{number+1}",
                       "prediction_count": int(subset.sum()), "unscorable_count": int((subset & ~score).sum()),
                       **compare_predictions(truth[selected], predictions[selected])})
    for grouping, name, subset in (("training_volatility", "missing", ~finite),
                                  ("trend_availability", "complete", trend_complete),
                                  ("trend_availability", "missing", ~trend_complete),
                                  ("industry", "unknown", np.ones(len(score), dtype=bool))):
        selected = score & subset
        groups.append({"grouping": grouping, "group": name, "prediction_count": int(subset.sum()),
                       "unscorable_count": int((subset & ~score).sum()), **compare_predictions(truth[selected], predictions[selected])})
    valid_days = [d for d in daily if d["score_count"]]
    day_equal = {name: {"rmse": float(np.sqrt(np.mean([d[name]["squared_error_sum"]/d["score_count"] for d in valid_days]))),
                        "mae": float(np.mean([d[name]["mae"] for d in valid_days]))} for name in ("lightgbm", "zero")}
    return {"unit": "return_fraction", "report_error_unit": "percentage_points", "overall": comparison,
            "daily": daily, "groups": groups, "day_equal": day_equal, "score_days": len(valid_days),
            "prediction_count": len(score), "score_count": int(score.sum()), "unscorable_count": int((~score).sum()),
            "volatility_boundaries": list(boundaries), "volatility_threshold_source": "this fold train only"}
