"""预测样本领域：日历、资格、目标与覆盖；数组是数值载荷，不是文件句柄。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping
from types import MappingProxyType
import numpy as np


class DataError(ValueError):
    """样本输入或结果不符合已冻结的研究契约。"""


class TradeState(IntEnum):
    PRE_LISTING = -1
    SUSPENDED = 0
    TRADED = 1


class TargetStatus(IntEnum):
    OBSERVED = 1
    SUSPENDED = 2
    OUTSIDE_CALENDAR = 3
    PRE_LISTING = 4


@dataclass(frozen=True)
class StockHistory:
    stock_id: str
    listing_date: str
    dates: np.ndarray
    trade_state: np.ndarray
    prices: dict[str, np.ndarray]
    quote_return: np.ndarray

    def prefix(self, length: int) -> StockHistory:
        """预测端也使用同一历史前缀，不能把未来标签传给特征函数。"""
        return StockHistory(self.stock_id, self.listing_date, self.dates[:length],
                            self.trade_state[:length], MappingProxyType({key: value[:length] for key, value in self.prices.items()}),
                            self.quote_return[:length])


@dataclass(frozen=True)
class FeatureBlock:
    names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class TargetBlock:
    values: np.ndarray
    target_indices: np.ndarray
    available_indices: np.ndarray
    statuses: np.ndarray

@dataclass(frozen=True)
class PreparationDescription:
    calendar: tuple[str, ...]
    stocks: tuple[Mapping, ...]
    frozen: Mapping
    audit_counts: Mapping
    input_sha256: Mapping[str, str]
    acceptance: Mapping


@dataclass(frozen=True)
class ReadReceipt:
    records: tuple[Mapping, ...]
    changed: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class PreparationChunk:
    values: np.ndarray
    labels: np.ndarray
    rows: np.ndarray
    coverage: Mapping


@dataclass(frozen=True)
class StoredSamples:
    preparation_id: str
    cache_key: str
    manifest_sha256: str
    metadata: Mapping


@dataclass(frozen=True)
class PreparedIndex:
    """完成样本的只读索引视图；P2不取得X或目标数值，行号定位原数组。"""
    stored: StoredSamples
    parent_run: str
    calendar: tuple[str, ...]
    stocks: tuple[Mapping, ...]
    period: Mapping
    limitations: Mapping
    rows: np.ndarray


def validate_prepared_index(index):
    """在完成产物的读取边界核对日期和身份，不重新计算特征。"""
    rows, size = index.rows, len(index.calendar)
    if not size or tuple(sorted(set(index.calendar))) != index.calendar or not len(rows):
        raise DataError("完成样本的日历或行索引为空、重复或无序")
    stock, current, target, available = (rows[key] for key in ("stock_idx", "as_of_idx", "target_idx", "label_available_idx"))
    if np.any((stock < 0) | (stock >= len(index.stocks)) | (current < 0) | (current >= size)):
        raise DataError("样本身份或信息日索引越界")
    if np.any(stock[1:] < stock[:-1]) or np.any((stock[1:] == stock[:-1]) & (current[1:] <= current[:-1])):
        raise DataError("完成样本键重复或无序")
    if not np.array_equal(target, np.where(current + 1 < size, current + 1, -1)):
        raise DataError("完成样本目标不是下一市场交易日")
    observed = rows["target_status"] == TargetStatus.OBSERVED
    if np.any(~np.isin(rows["trade_state"], (0, 1))) or not np.array_equal(rows["prediction_eligible"], rows["trade_state"] == 1):
        raise DataError("预测资格与信息日成交状态不符")
    if np.any(~np.isin(rows["target_status"], (1, 2, 3))) or not np.array_equal(rows["target_status"] == 3, target == -1):
        raise DataError("目标状态与目标日期不符")
    if np.any(observed & ((available < target) | (available >= size))) or np.any(~observed & (available != -1)):
        raise DataError("标签可得日期与目标状态不符")


ROW_DTYPE = np.dtype([
    ("stock_idx", "<i4"), ("as_of_idx", "<i4"), ("target_idx", "<i4"), ("label_available_idx", "<i4"),
    ("trade_state", "i1"), ("prediction_eligible", "?"), ("target_status", "u1"),
])
NUMERIC_TOLERANCE = {"float32_rtol": 2e-6, "float32_atol": 1e-7, "purpose": "手算与float64参考比较；不能容忍日期、列顺序或NaN位置变化"}



def require_same_dates(actual, expected, title):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise DataError(f"{title}日期不符：缺{len(missing)}，多{len(extra)}；缺样例{sorted(missing)[:3]}，多样例{sorted(extra)[:3]}")


def align_history(stock, calendar, tables):
    """仅处理已读取的值数据，所有偏移都按市场日历定位。"""
    dates = np.array(calendar, dtype="datetime64[D]")
    dates.flags.writeable = False
    locations = {current: index for index, current in enumerate(calendar)}
    symbol = stock["exchange"] + stock["code"]
    expected_dates = [current for current in calendar if current >= stock["list_date"]]
    require_same_dates(tables["state"], expected_dates, "上市后交易状态")
    state = np.full(len(dates), TradeState.PRE_LISTING, dtype=np.int8)
    for current, row in tables["state"].items():
        if row["trade_status"] not in ("0", "1"):
            raise DataError(f"未知交易状态：{symbol} {current}")
        state[locations[current]] = int(row["trade_status"])
    traded_dates = [current for current in calendar if state[locations[current]] == TradeState.TRADED]
    require_same_dates(tables["daily"], traded_dates, "成交与日线")
    require_same_dates(tables["label"], traded_dates, "成交与涨跌幅")
    positions = np.array([locations[current] for current in traded_dates], dtype=np.int32)
    prices = {}
    for field in ("open", "high", "low", "close", "volume_shares", "amount_yuan", "turnover_pct"):
        values = np.full(len(dates), np.nan)
        observed = np.array([float(tables["daily"][current][field]) for current in traded_dates], dtype=np.float64)
        if not np.all(np.isfinite(observed)):
            raise DataError(f"{symbol} {field}不能转换为有限float64")
        values[positions] = observed / 100 if field == "turnover_pct" else observed
        values.flags.writeable = False
        prices["turnover" if field == "turnover_pct" else field] = values
    returns = np.full(len(dates), np.nan)
    observed_returns = np.array([float(tables["label"][current]["provider_change_pct"]) / 100 for current in traded_dates])
    if not np.all(np.isfinite(observed_returns)):
        raise DataError(f"{symbol}涨跌幅不能转换为有限float64")
    returns[positions] = observed_returns
    returns.flags.writeable = state.flags.writeable = False
    return StockHistory(symbol, stock["list_date"], dates, state, MappingProxyType(prices), returns)


def market_closes(calendar, facts):
    result = {}
    for symbol, rows in facts.items():
        require_same_dates(rows, calendar, symbol)
        close = np.array([float(rows[current]["close"]) for current in calendar], dtype=np.float64)
        if not np.all(np.isfinite(close) & (close > 0)):
            raise DataError(f"市场收盘价无法转为有效数值：{symbol}")
        close.flags.writeable = False
        result[symbol] = close
    return result


def planned_rows(description):
    dates = np.array(description.calendar, dtype="datetime64[D]")
    count = sum(int(np.count_nonzero(dates >= np.datetime64(stock["list_date"], "D"))) for stock in description.stocks)
    if count != description.audit_counts["state_rows"]:
        raise DataError("上市后样本网格数量与P0状态验收不符")
    return count


def build_chunk(stock_index, history, block, target, expected_names):
    listed = history.trade_state != TradeState.PRE_LISTING
    as_of_indices = np.flatnonzero(listed).astype(np.int32)
    values, labels = block.values[listed], target.values[listed]
    status = target.statuses[listed]
    eligible = history.trade_state[listed] == TradeState.TRADED
    observed = status == TargetStatus.OBSERVED
    if block.names != expected_names or values.shape != (len(as_of_indices), len(expected_names)) or np.isinf(values).any():
        raise DataError(f"特征列、形状或有限性不符合规格：{history.stock_id}")
    if not np.all(np.isfinite(labels[observed])) or not np.all(np.isnan(labels[~observed])):
        raise DataError(f"目标状态与目标值不一致：{history.stock_id}")
    if np.any(target.target_indices[:-1] != np.arange(1, len(history.dates))) or target.target_indices[-1] != -1:
        raise DataError("目标不是下一市场交易日")
    if not np.array_equal(target.available_indices[listed][observed], target.target_indices[listed][observed]):
        raise DataError("日级标签可得时间与目标日期不一致")
    rows = np.empty(len(as_of_indices), dtype=ROW_DTYPE)
    rows["stock_idx"], rows["as_of_idx"] = stock_index, as_of_indices
    rows["target_idx"], rows["label_available_idx"] = target.target_indices[listed], target.available_indices[listed]
    rows["trade_state"], rows["prediction_eligible"], rows["target_status"] = history.trade_state[listed], eligible, status
    finite = np.isfinite(values)
    coverage = {
        "stock_id": history.stock_id, "grid_rows": len(values), "prediction_eligible": int(eligible.sum()),
        "information_day_paused": int((~eligible).sum()), "target_observed": int(observed.sum()),
        "target_suspended": int((status == TargetStatus.SUSPENDED).sum()),
        "target_calendar_unknown": int((status == TargetStatus.OUTSIDE_CALENDAR).sum()),
        "supervised_candidates": int((eligible & observed).sum()),
        "eligible_target_suspended": int((eligible & (status == TargetStatus.SUSPENDED)).sum()),
        "eligible_target_calendar_unknown": int((eligible & (status == TargetStatus.OUTSIDE_CALENDAR)).sum()),
        "observed_zero_labels": int((observed & (labels == 0)).sum()), "rows_with_nan_features": int((~finite).any(axis=1).sum()),
    }
    values.flags.writeable = labels.flags.writeable = rows.flags.writeable = False
    return PreparationChunk(values, labels, rows, MappingProxyType(coverage))


class PreparationSummary:
    """按股累积业务分母和列覆盖，不持有已写出的历史矩阵。"""
    def __init__(self, names):
        self.names, self.totals, self.coverage = names, Counter(), []
        self.finite = np.zeros(len(names), dtype=np.int64)
        self.eligible_finite = np.zeros(len(names), dtype=np.int64)
        self.minima, self.maxima = np.full(len(names), np.inf), np.full(len(names), -np.inf)

    def include(self, chunk):
        finite = np.isfinite(chunk.values)
        self.finite += finite.sum(axis=0)
        self.eligible_finite += finite[chunk.rows["prediction_eligible"]].sum(axis=0)
        self.minima = np.minimum(self.minima, np.where(finite, chunk.values, np.inf).min(axis=0))
        self.maxima = np.maximum(self.maxima, np.where(finite, chunk.values, -np.inf).max(axis=0))
        self.coverage.append(dict(chunk.coverage))
        self.totals.update({key: value for key, value in chunk.coverage.items() if key != "stock_id"})

    def complete(self, description):
        if self.totals["grid_rows"] != planned_rows(description) or self.totals["prediction_eligible"] != description.audit_counts["valid_trading_rows"]:
            raise DataError("样本或预测资格数量与P0不一致")
        if self.totals["supervised_candidates"] != description.audit_counts["next_day_pair_candidates"]:
            raise DataError("下一市场日候选对数量与P0不一致")
        return {"counts": dict(self.totals), "stock_count": len(description.stocks), "market_days": len(description.calendar),
                "shape": [self.totals["grid_rows"], len(self.names)]}

    def feature_quality(self):
        return [{"name": name, "finite_rows": int(self.finite[i]), "nan_rows": self.totals["grid_rows"] - int(self.finite[i]),
                 "eligible_finite_rows": int(self.eligible_finite[i]), "minimum": float(self.minima[i]) if self.finite[i] else "",
                 "maximum": float(self.maxima[i]) if self.finite[i] else ""} for i, name in enumerate(self.names)]
