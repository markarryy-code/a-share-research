"""目标单独计算：只连接下一市场日，暂停和未知日期不填零。"""

import numpy as np

from .domain import StockHistory, TargetBlock, TargetStatus, TradeState, DataError


TARGET_SPEC = {
    "target_id": "next-market-day-quote-return-v1", "source": "provider_change_pct",
    "conversion": "source / 100", "unit": "fraction", "dtype": "float64",
    "horizon": "next market calendar day", "availability_policy": "target day EOD data available; exact timestamp unverified",
    "missing_label": "NaN", "statuses": {status.name.lower(): int(status) for status in TargetStatus},
}

def build_next_day_targets(history: StockHistory) -> TargetBlock:
    length = len(history.dates)
    values = np.full(length, np.nan, dtype=np.float64)
    target_indices = np.arange(1, length + 1, dtype=np.int32)
    target_indices[-1] = -1
    available = np.full(length, -1, dtype=np.int32)
    statuses = np.full(length, TargetStatus.OUTSIDE_CALENDAR, dtype=np.uint8)
    next_state = history.trade_state[1:]
    observed = next_state == TradeState.TRADED
    if not np.all(np.isfinite(history.quote_return[1:][observed])):
        raise DataError("下一市场日有成交但缺少有限目标值")
    values[:-1][observed] = history.quote_return[1:][observed]
    available[:-1][observed] = target_indices[:-1][observed]
    statuses[:-1][observed] = TargetStatus.OBSERVED
    statuses[:-1][next_state == TradeState.SUSPENDED] = TargetStatus.SUSPENDED
    statuses[:-1][next_state == TradeState.PRE_LISTING] = TargetStatus.PRE_LISTING
    for array in (values, target_indices, available, statuses):
        array.flags.writeable = False
    return TargetBlock(values, target_indices, available, statuses)
