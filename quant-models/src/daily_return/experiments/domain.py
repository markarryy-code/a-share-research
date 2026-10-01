"""时间方案与样本角色：按目标日切分，在各截止日检验标签成熟。"""

from dataclasses import dataclass
from datetime import date
from typing import Mapping
import numpy as np


class SplitError(ValueError):
    """完成样本或时间方案不能支持此次实验。"""


@dataclass(frozen=True)
class SamplePopulation:
    parent_run: str
    preparation_id: str
    manifest_sha256: str
    source_fingerprint: str
    feature_set: str
    target_id: str
    calendar: tuple[str, ...]
    stock_ids: tuple[str, ...]
    period: Mapping
    limitations: Mapping
    rows: np.ndarray


@dataclass(frozen=True)
class EvaluationWindow:
    name: str
    kind: str
    train_through: str
    start: str
    end: str
    fit_index: int
    first_index: int
    last_index: int


@dataclass(frozen=True)
class SplitPlan:
    version: str
    windows: tuple[EvaluationWindow, ...]


@dataclass(frozen=True)
class RoleSelection:
    name: str
    indices: np.ndarray
    counts: Mapping
    daily: tuple[Mapping, ...]


@dataclass(frozen=True)
class FoldSelection:
    window: EvaluationWindow
    roles: tuple[RoleSelection, ...]


def build_split_plan(config, population):
    """日期声明必须明确；休市边界投影到市场日历，缺折、重叠或空窗直接失败。"""
    if not isinstance(config, dict) or config.get("version") != "expanding-target-date-v1":
        raise SplitError("缺少受支持的时间方案版本")
    if set(config) != {"version", "folds", "holdout"} or not isinstance(config["folds"], list) or not config["folds"]:
        raise SplitError("时间方案必须明确给出folds与holdout")
    days, windows = population.calendar, []
    holdout = config["holdout"]
    if not isinstance(holdout, dict) or set(holdout) != {"start", "end"}:
        raise SplitError("留出配置字段不符")
    if holdout != {"start": population.period["month_start"], "end": population.period["end"]}:
        raise SplitError("留出范围与完成样本冻结月份不符")

    def position(value, upper):
        try:
            valid = isinstance(value, str) and date.fromisoformat(value).isoformat() == value
        except ValueError:
            valid = False
        if not valid:
            raise SplitError("日期必须为YYYY-MM-DD")
        return int(np.searchsorted(days, value, side="right" if upper else "left")) - int(upper)

    for number, item in enumerate(config["folds"], 1):
        if not isinstance(item, dict) or set(item) != {"id", "train_through", "validation_start", "validation_end"} or item["id"] != f"F{number:02}":
            raise SplitError("验证折必须按F01开始连续编号并声明日期边界")
        start, end, cutoff = item["validation_start"], item["validation_end"], item["train_through"]
        first, last, fit = position(start, False), position(end, True), position(cutoff, True)
        if not population.period["start"] <= cutoff < start <= end < holdout["start"]:
            raise SplitError(f"{item['id']}训练、验证与留出日期关系不符")
        if fit < 0 or fit != first - 1 or first > last or last >= len(days):
            raise SplitError(f"{item['id']}为空窗口或拟合截止不是验证前一市场日")
        if windows and (first != windows[-1].last_index + 1 or start <= windows[-1].end):
            raise SplitError("验证目标区间重叠或有未声明间隔")
        windows.append(EvaluationWindow(item["id"], "validation", cutoff, start, end, fit, first, last))
    first, last = position(holdout["start"], False), position(holdout["end"], True)
    if first != windows[-1].last_index + 1 or first > last or last != len(days) - 1:
        raise SplitError("末折与留出月没有连续覆盖到样本末端")
    windows.append(EvaluationWindow("holdout", "holdout", days[first-1], holdout["start"], holdout["end"], first-1, first, last))
    return SplitPlan(config["version"], tuple(windows))


def plan_description(plan, population):
    return {"version": plan.version, "train_start": population.period["start"],
            "label_availability": "target day EOD; exact timestamp unverified",
            "windows": [{"id": w.name, "kind": w.kind, "train_through": w.train_through,
                         "validation_start": w.start, "validation_end": w.end,
                         "fit_cutoff_date": population.calendar[w.fit_index],
                         "first_target_date": population.calendar[w.first_index],
                         "last_target_date": population.calendar[w.last_index],
                         "score_cutoff_date": population.calendar[w.last_index],
                         "market_days": w.last_index-w.first_index+1} for w in plan.windows]}


def select_fold_rows(population, window):
    rows, days = population.rows, population.calendar
    target, available = rows["target_idx"], rows["label_available_idx"]
    eligible, observed = rows["prediction_eligible"], rows["target_status"] == 1
    selections = []
    for role, first, last, cutoff in (("train", 0, window.fit_index, window.fit_index),
                                     ("predict", window.first_index, window.last_index, window.last_index),
                                     ("score", window.first_index, window.last_index, window.last_index)):
        period = (target >= first) & (target <= last)
        candidates = period & eligible
        mature = observed & (available >= 0) & (available <= cutoff)
        selected = candidates if role == "predict" else candidates & mature
        if role != "train" and np.any(candidates & (rows["as_of_idx"] < window.fit_index)):
            raise SplitError(f"{window.name}预测信息日早于模型拟合截止")
        indices = np.flatnonzero(selected).astype("<i8", copy=False)
        if not len(indices):
            raise SplitError(f"{window.name}.{role}集合为空，不能跳过该折")
        indices.flags.writeable = False
        unavailable = candidates & ~observed
        immature = candidates & observed & ~mature
        counts = {"period_rows": int(period.sum()), "information_ineligible": int((period & ~eligible).sum()),
                  "prediction_eligible": int(candidates.sum()), "target_unavailable": int(unavailable.sum()),
                  "label_not_mature": int(immature.sum()), "selected": len(indices)}
        daily_arrays = [np.bincount(target[mask], minlength=len(days)) for mask in (period, candidates, selected, unavailable, immature)]
        daily = tuple({"target_date": days[i], "period_rows": int(daily_arrays[0][i]),
                       "information_ineligible": int(daily_arrays[0][i]-daily_arrays[1][i]),
                       "prediction_eligible": int(daily_arrays[1][i]), "selected": int(daily_arrays[2][i]),
                       "target_unavailable": int(daily_arrays[3][i]), "label_not_mature": int(daily_arrays[4][i])}
                      for i in range(first, last+1))
        selections.append(RoleSelection(role, indices, counts, daily))
    return FoldSelection(window, tuple(selections))


def population_coverage(population):
    rows = population.rows
    unknown = rows["target_idx"] < 0
    return {"grid_rows": len(rows), "prediction_eligible": int(rows["prediction_eligible"].sum()),
            "information_ineligible": int((~rows["prediction_eligible"]).sum()),
            "unknown_target_date": int(unknown.sum()),
            "eligible_without_target_date": int((unknown & rows["prediction_eligible"]).sum())}
