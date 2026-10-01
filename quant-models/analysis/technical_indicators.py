"""形态研究的因果指标：窗口按完整市场日历取此前观察，不接收未来收益。"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PLATFORM_WINDOWS = (20, 40, 60)
BOTTOM_WINDOWS = (60, 120, 250)
CHANNEL_WINDOWS = (20, 40, 60)


def _quote_index(quote_returns: np.ndarray, traded: np.ndarray) -> np.ndarray:
    """首观察成交日定基100；停牌保留累计水平，但不填造停牌日观察。"""
    if (quote_returns.ndim != 1 or traded.ndim != 1
            or quote_returns.shape != traded.shape):
        raise ValueError("涨跌幅和成交状态必须是一维、等长的完整市场日历数组")
    if quote_returns.dtype != np.float64 or traded.dtype != np.bool_:
        raise ValueError("涨跌幅须直接采用源行情float64，成交状态须为bool；不得从float32特征还原")

    level = np.full(quote_returns.size, np.nan, dtype=np.float64)
    positions = np.flatnonzero(traded)
    if not positions.size:
        return level
    observed_returns = quote_returns[positions]
    # 首日是任意基期，其涨跌幅不参与连乘；以后缺失或非正因子不能跨过去。
    if not np.all(np.isfinite(observed_returns[1:]) & (observed_returns[1:] > -1)):
        raise ValueError("首观察成交日之后的成交收益必须有限且大于-100%")
    factors = np.r_[1.0, 1.0 + observed_returns[1:]]
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            observed_levels = 100.0 * np.cumprod(factors, dtype=np.float64)
    except FloatingPointError as error:
        raise ValueError("累计行情指数超出float64有效数值范围") from error
    if not np.all(np.isfinite(observed_levels) & (observed_levels > 0)):
        raise ValueError("累计行情指数必须为正有限值")
    level[positions] = observed_levels
    return level


def compute_indicators(quote_returns: np.ndarray, traded: np.ndarray) -> dict[str, np.ndarray]:
    """返回等长float64指标；历史结构用[t-W,t)，今天只参与突破/通道位置。

    quote_returns必须来自原始provider_change_pct/100。所有结构窗口要求W个
    连续市场日都有有效Q，停牌和上市前不压缩。纯历史结构可在今天停牌时保留，
    涉及Q[t]的指标则为NaN；是否属于当日可分析样本由调用者按成交资格判定。
    """
    level = _quote_index(quote_returns, traded)
    result = {"quote_index": level}

    def empty(name: str) -> np.ndarray:
        values = np.full(level.size, np.nan, dtype=np.float64)
        result[name] = values
        return values

    for window in sorted(set(PLATFORM_WINDOWS + BOTTOM_WINDOWS + CHANNEL_WINDOWS)):
        fields = {}
        if window in PLATFORM_WINDOWS:
            for name in ("prior_width", "prior_drift", "breakout"):
                fields[name] = empty(f"{name}_{window}")
        if window in BOTTOM_WINDOWS:
            for name in ("bottom_position", "prior_drawdown", "bottom_low_distance"):
                fields[name] = empty(f"{name}_{window}")
        if window in CHANNEL_WINDOWS:
            for name in ("channel_gain", "channel_r2", "channel_width", "channel_offset", "channel_higher", "channel_sigma"):
                fields[name] = empty(f"{name}_{window}")
        if level.size <= window:
            continue

        # 最后一条滑窗无对应的今天；显式从level[:-1]取窗，排除Q[t]。
        views = sliding_window_view(level[:-1], window)
        valid = np.all(np.isfinite(views), axis=1)
        today = np.flatnonzero(valid) + window
        if not today.size:
            continue
        prior = views[valid]
        high, low = prior.max(axis=1), prior.min(axis=1)
        if window in PLATFORM_WINDOWS:
            fields["prior_width"][today] = high / low - 1
            fields["prior_drift"][today] = prior[:, -1] / prior[:, 0] - 1
            fields["breakout"][today] = level[today] / high - 1
        if window in BOTTOM_WINDOWS:
            distance = high - low
            position = np.full(today.size, np.nan, dtype=np.float64)
            np.divide(prior[:, -1] - low, distance, out=position, where=distance > 0)
            fields["bottom_position"][today] = position
            fields["prior_drawdown"][today] = prior[:, -1] / high - 1
            fields["bottom_low_distance"][today] = prior[:, -1] / low - 1
        if window in CHANNEL_WINDOWS:
            # 先中心化再拟合，避免用大数相减求斜率；窗口x固定为0..W-1。
            log_prior = np.log(prior)
            x = np.arange(window, dtype=np.float64)
            x_center = x - x.mean()
            y_mean = log_prior.mean(axis=1)
            y_center = log_prior - y_mean[:, None]
            slope = (y_center * x_center).sum(axis=1) / np.dot(x_center, x_center)
            residual = y_center - slope[:, None] * x_center
            sst = (y_center * y_center).sum(axis=1)
            sse = (residual * residual).sum(axis=1)
            r2 = np.full(today.size, np.nan, dtype=np.float64)
            varying = np.ptp(log_prior, axis=1) > 0
            np.divide(sse, sst, out=r2, where=varying)
            fields["channel_gain"][today] = np.expm1(slope * (window - 1))
            fields["channel_r2"][today] = np.clip(1 - r2, 0, 1)
            fields["channel_width"][today] = np.expm1(np.ptp(residual, axis=1))
            fields["channel_sigma"][today] = np.sqrt(sse / window)
            fields["channel_offset"][today] = np.log(level[today]) - (y_mean + slope * (window - x.mean()))
            half = window // 2
            higher = ((prior[:, half:].max(axis=1) > prior[:, :half].max(axis=1))
                      & (prior[:, half:].min(axis=1) > prior[:, :half].min(axis=1)))
            fields["channel_higher"][today] = higher.astype(np.float64)
    return result
