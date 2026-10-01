"""上涨二分类的纯评分：概率、类别、训练期基准及固定分组。"""

import numpy as np
from .domain import up_labels
from daily_return.experiments import ValidationError


THRESHOLD = 0.5
PROBABILITY_EPSILON = 1e-15
METRIC_NAMES = ("accuracy", "balanced_accuracy", "up_recall", "not_up_recall", "up_precision",
                "predicted_up_rate", "actual_up_rate", "logloss", "brier", "auc")


def roc_auc(labels, scores):
    """用并列分数的平均秩计算AUC；单类样本返回不可计算。"""
    n = len(labels)
    positives = int(np.count_nonzero(labels))
    if not positives or positives == n:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    positive_counts = np.add.reduceat(sorted_labels.astype(np.int64), starts)
    rank_sum = float(np.dot(positive_counts, (starts + ends + 1) / 2))
    return (rank_sum - positives * (positives + 1) / 2) / (positives * (n - positives))


def binary_metrics(labels, scores, *, probability=True, threshold=THRESHOLD):
    """回归值可作排序分数，但不能当概率计算Logloss/Brier。"""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, (0, 1)).all()
            or not np.isfinite(scores).all()):
        raise ValidationError("二分类评分要求同形一维0/1标签与有限分数")
    if probability and np.any((scores < 0) | (scores > 1)):
        raise ValidationError("上涨概率必须位于[0,1]")
    actual, predicted = labels.astype(bool), scores > threshold
    tp = int(np.count_nonzero(actual & predicted))
    tn = int(np.count_nonzero(~actual & ~predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    fn = int(np.count_nonzero(actual & ~predicted))
    n, up, not_up = len(labels), tp + fn, tn + fp
    result = {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
              **{name: None for name in METRIC_NAMES}}
    if not n:
        return result
    result.update(accuracy=(tp+tn)/n, up_recall=tp/up if up else None,
                  not_up_recall=tn/not_up if not_up else None,
                  up_precision=tp/(tp+fp) if tp+fp else None,
                  predicted_up_rate=(tp+fp)/n, actual_up_rate=up/n,
                  auc=roc_auc(actual, scores))
    if up and not_up:
        result["balanced_accuracy"] = (tp/up + tn/not_up)/2
    if probability:
        # 只为对数计算处理精确0/1，原概率和硬分类不改变。
        clipped = np.clip(scores, PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON)
        result["logloss"] = float(-np.mean(np.where(actual, np.log(clipped), np.log1p(-clipped))))
        result["brier"] = float(np.mean((scores-labels)**2))
    return result


def probability_bins(labels, probabilities):
    """事先固定十个等宽区间；空箱保留，不用验证分布选边界。"""
    assignments = np.minimum((probabilities * 10).astype(int), 9)
    bins = []
    for i in range(10):
        selected = assignments == i
        count = int(selected.sum())
        bins.append({"lower": i/10, "upper": (i+1)/10, "upper_inclusive": i == 9,
                     "n": count, "mean_probability": float(probabilities[selected].mean()) if count else None,
                     "observed_up_rate": float(labels[selected].mean()) if count else None})
    return bins


def evaluate_classification_fold(description, truth, predictions, volatility, trend_complete, boundaries, prior):
    score, rows = description.score_mask, description.prediction_rows
    if predictions.shape != score.shape or not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValidationError("全部应预测行必须具有有限的[0,1]概率")
    if not score.any() or not np.isfinite(truth[score]).all() or not 0 <= prior <= 1:
        raise ValidationError("二分类评分标签或训练期先验不完整")

    def compare(selected):
        labels = up_labels(truth[selected])
        return {"lightgbm": binary_metrics(labels, predictions[selected]),
                "training_prior": binary_metrics(labels, np.full(len(labels), prior))}

    daily, groups = [], []
    for index in range(description.calendar.index(description.window["first_target_date"]),
                       description.calendar.index(description.window["last_target_date"])+1):
        predicted = rows["target_idx"] == index
        selected = predicted & score
        daily.append({"target_date": description.calendar[index], "prediction_count": int(predicted.sum()),
                      "score_count": int(selected.sum()), **compare(selected)})
    categories = np.searchsorted(np.array(boundaries), volatility, side="right")
    finite = np.isfinite(volatility)
    subsets = [("training_volatility", f"Q{i+1}", finite & (categories == i)) for i in range(len(boundaries)+1)]
    subsets += [("training_volatility", "missing", ~finite), ("trend_availability", "complete", trend_complete),
                ("trend_availability", "missing", ~trend_complete), ("industry", "unknown", np.ones(len(score), dtype=bool))]
    for grouping, name, subset in subsets:
        groups.append({"grouping": grouping, "group": name, "prediction_count": int(subset.sum()),
                       "unscorable_count": int((subset & ~score).sum()), **compare(subset & score)})
    day_equal = {}
    for model in ("lightgbm", "training_prior"):
        day_equal[model] = {}
        for metric in METRIC_NAMES:
            values = [d[model][metric] for d in daily if d[model][metric] is not None]
            day_equal[model][metric] = float(np.mean(values)) if values else None
            day_equal[model][metric + "_days"] = len(values)
    return {"unit": "up_probability", "class_threshold": THRESHOLD, "threshold_comparison": ">",
            "prior_probability": prior, "prior_source": "this fold training labels only",
            "probability_log_epsilon": PROBABILITY_EPSILON, "overall": compare(score),
            "nonflat_diagnostic": compare(score & (truth != 0)), "actual_flat_count": int(np.count_nonzero(truth[score] == 0)),
            "daily": daily, "groups": groups, "day_equal": day_equal,
            "calibration_bins": probability_bins(up_labels(truth[score]), predictions[score]),
            "score_days": sum(d["score_count"] > 0 for d in daily), "prediction_count": len(score),
            "score_count": int(score.sum()), "unscorable_count": int((~score).sum()),
            "volatility_boundaries": list(boundaries), "volatility_threshold_source": "this fold train only"}
