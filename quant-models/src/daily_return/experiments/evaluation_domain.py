"""跨方法共用的样本输入、时间边界与评价分组规则；不包含分类标签或模型设置。"""

from dataclasses import dataclass
from typing import Mapping
import numpy as np


class ValidationError(ValueError):
    """训练输入、模型结果或实验阶段不符合契约。"""


@dataclass(frozen=True)
class FoldInputSpec:
    """只表达共用数据请求；具体预测目标由各方法拥有。"""
    split_run: str
    feature_set: str
    feature_count: int
    target_id: str


@dataclass(frozen=True)
class FoldDescription:
    fold: str
    window: Mapping
    feature_names: tuple[str, ...]
    trend_columns: tuple[int, ...]
    volatility_column: int
    calendar: tuple[str, ...]
    stock_ids: tuple[str, ...]
    prediction_rows: np.ndarray
    score_mask: np.ndarray
    counts: Mapping
    provenance: Mapping


@dataclass(frozen=True)
class PredictionBatch:
    offset: int
    features: np.ndarray
    rows: np.ndarray


def validate_fold_roles(rows, roles, window, calendar, month_start):
    """检验已给定角色的业务约束，不另建一套时间切分算法。"""
    if window["kind"] != "validation" or window["last_target_date"] >= month_start:
        raise ValidationError("validate只接受开发期折，不能拟合或评价留出月")
    return validate_temporal_roles(rows, roles, window, calendar)


def validate_temporal_roles(rows, roles, window, calendar):
    """共享时间角色不变量；开发期或留出期的准入由各自入口决定。"""
    fit = calendar.index(window["fit_cutoff_date"])
    first, last = calendar.index(window["first_target_date"]), calendar.index(window["last_target_date"])
    for name, indices in roles.items():
        if indices.dtype != np.dtype("int64") or indices.ndim != 1 or not len(indices):
            raise ValidationError(f"{name}索引格式不符或为空")
        if indices[0] < 0 or indices[-1] >= len(rows) or np.any(indices[1:] <= indices[:-1]):
            raise ValidationError(f"{name}行号越界、重复或无序")
        selected = rows[indices]
        if not selected["prediction_eligible"].all():
            raise ValidationError(f"{name}包含信息日不合格记录")
        if name == "train":
            good = ((selected["target_idx"] >= 0) & (selected["target_idx"] <= fit)
                    & (selected["target_status"] == 1) & (selected["label_available_idx"] >= 0)
                    & (selected["label_available_idx"] <= fit))
        else:
            good = ((selected["target_idx"] >= first) & (selected["target_idx"] <= last) & (selected["as_of_idx"] >= fit))
            if name == "score":
                good &= ((selected["target_status"] == 1) & (selected["label_available_idx"] >= 0)
                         & (selected["label_available_idx"] <= last))
        if not good.all():
            raise ValidationError(f"{name}违反目标期或标签成熟条件")
    positions = np.searchsorted(roles["predict"], roles["score"])
    if np.any(positions == len(roles["predict"])) or not np.array_equal(roles["predict"][positions], roles["score"]):
        raise ValidationError("评分行不是应预测行的子集")
    mask = np.zeros(len(roles["predict"]), dtype=bool)
    mask[positions] = True
    mask.flags.writeable = False
    return mask


def check_matrix(features, labels, columns):
    if features.ndim != 2 or features.shape[1] != columns or len(features) != len(labels) or not len(labels):
        raise ValidationError("拟合矩阵列数、行数或标签长度不符")
    if np.isinf(features).any() or not np.isfinite(labels).all():
        raise ValidationError("特征含无穷值或监督目标不是有限数值")


def validate_holdout_roles(rows, roles, window, calendar, month_start):
    if (window["kind"] != "holdout" or window["first_target_date"] != month_start
            or window["fit_cutoff_date"] >= month_start):
        raise ValidationError("留出评价必须使用冻结月边界，并在首个目标日前结束训练")
    return validate_temporal_roles(rows, roles, window, calendar)


def fit_volatility_boundaries(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return ()
    return tuple(float(v) for v in np.unique(np.quantile(finite.astype(np.float64), [.25, .5, .75], method="linear")))
