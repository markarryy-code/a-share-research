"""49项因果特征：原44项量价字段及5项累计行情指数趋势；不接收未来目标。"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .domain import DataError, FeatureBlock, StockHistory, TradeState


FEATURE_SET = "price-volume-v2"
WINDOWS = (5, 10, 20, 60)
LAGS = (0, 1, 2, 5, 10, 20, 60)
MARKETS = ("sh000001", "sz399001")


def feature_spec() -> dict:
    items = []

    def add(name, family, formula, unit, window=None):
        items.append(dict(name=name, family=family, formula=formula, unit=unit, window=window))

    for lag in LAGS:
        add(f"return_lag_{lag}", "return_lag", f"r[t-{lag}]", "return_fraction", lag)
    for kind in ("mean", "std", "valid_fraction"):
        for window in WINDOWS:
            add(f"return_{kind}_{window}", "return_" + kind, f"{kind}(r[t-{window-1}:t+1])", "ratio" if kind == "valid_fraction" else "return_fraction", window)
    for name, formula in (("candle_body", "close/open-1"), ("high_low_range", "high/low-1"),
                          ("close_position", "(close-low)/(high-low)"), ("upper_shadow", "(high-max(open,close))/close"),
                          ("lower_shadow", "(min(open,close)-low)/close")):
        add(name, "candle", formula, "ratio")
    for field in ("volume_shares", "amount_yuan"):
        for window in (5, 20):
            add(f"{field}_relative_{window}", "liquidity", f"{field}[t]/mean({field}[t-{window}:t])-1", "ratio", window)
    add("turnover", "turnover", "turnover_pct/100", "fraction")
    add("turnover_change_1", "turnover", "turnover[t]-turnover[t-1]", "fraction", 1)
    add("turnover_relative_20", "turnover", "turnover[t]/mean(turnover[t-20:t])-1", "ratio", 20)
    for symbol in MARKETS:
        add(f"{symbol}_return_0", "market", f"{symbol}.close[t]/{symbol}.close[t-1]-1", "return_fraction")
        for kind in ("mean", "std"):
            for window in (5, 20):
                add(f"{symbol}_return_{kind}_{window}", "market", f"{kind}({symbol}.return[t-{window-1}:t+1])", "return_fraction", window)
    add("listing_age_days", "history_state", "as_of_date-listing_date", "calendar_days")
    add("suspended_days_20", "history_state", "count(trade_state==SUSPENDED,t-19:t+1)", "market_days", 20)
    add("days_since_previous_trade", "history_state", "t-max(traded_index<t)", "market_days")
    for window in (20, 60):
        add(f"quote_index_ma_deviation_{window}", "quote_index_trend", f"Q[t]/MA_Q({window},t)-1", "fraction", window)
    add("quote_index_ma_change_20_5", "quote_index_trend", "MA_Q(20,t)/MA_Q(20,t-5)-1", "fraction", 25)
    add("quote_index_prior_high_distance_20", "quote_index_trend", "Q[t]/max(Q[t-20:t])-1", "fraction", 21)
    add("quote_index_prior_low_distance_20", "quote_index_trend", "Q[t]/min(Q[t-20:t])-1", "fraction", 21)
    return {
        "feature_set": FEATURE_SET, "count": len(items), "features": items,
        "compute_dtype": "float64", "storage_dtype": "float32", "return_source": "provider_change_pct/100",
        "window_axis": "market_calendar", "std_ddof": 0, "minimum_mean_observations": 1,
        "minimum_std_observations": 2, "incomplete_calendar_window": "NaN",
        "missing_observation": "NaN; no forward/backward filling", "zero_denominator": "NaN",
        "valid_fraction_denominator": "full market window length", "zero_value_is_missing": False,
        "previous_trade_before_source_window": "unknown -> NaN; no date inferred from adjusted prices",
        "trend_index": {
            "source": "provider_change_pct/100 at TRADED market dates",
            "definition": "Q[first observed traded date]=100; Q[next traded date]=Q[previous traded date]*(1+r[next traded date])",
            "meaning": "cumulative quoted-return index; not raw/adjusted currency price or total-return index",
            "first_return": "not compounded: the first observed close is the arbitrary base",
            "non_traded_dates": "Q is NaN; internal accumulated level is retained across suspensions, with no invented return",
            "MA_Q": "arithmetic mean including t; all requested market dates must have finite Q",
            "window_rule": "new trend fields require complete observations; overrides base fields' minimum_mean_observations=1",
            "direction": "MA_Q(20,t)/MA_Q(20,t-5)-1; a five-market-day change, not an annualized slope",
            "prior_extrema": "previous 20 market dates, excluding t; extrema of Q at closes, not intraday high/low",
            "incomplete_observations": "NaN; no forward/backward fill; does not change prediction eligibility",
        },
    }


FEATURE_NAMES = tuple(item["name"] for item in feature_spec()["features"])


def lagged(values, offset):
    result = np.full(len(values), np.nan, dtype=np.float64)
    if offset == 0:
        result[:] = values
    elif offset < len(values):
        result[offset:] = values[:-offset]
    return result


def ratio(numerator, denominator):
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result,
              where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0))
    return result


def rolling_stats(values, window):
    """窗口按日历完整性判断；中心化计算方差，避免相减丢失低波动精度。"""
    mean, std, count = (np.full(len(values), np.nan, dtype=np.float64) for _ in range(3))
    if len(values) < window:
        return mean, std, count
    view = sliding_window_view(values, window)
    finite = np.isfinite(view)
    observed = finite.sum(axis=1)
    average = np.full(len(view), np.nan)
    np.divide(np.where(finite, view, 0).sum(axis=1), observed, out=average, where=observed > 0)
    centered = np.where(finite, view - average[:, None], 0)
    variance = np.full(len(view), np.nan)
    np.divide((centered * centered).sum(axis=1), observed, out=variance, where=observed >= 2)
    mean[window-1:], std[window-1:], count[window-1:] = average, np.sqrt(variance), observed
    return mean, std, count


def build_market_features(closes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for symbol in MARKETS:
        returns = ratio(closes[symbol], lagged(closes[symbol], 1)) - 1
        result[f"{symbol}_return_0"] = returns
        for window in (5, 20):
            mean, std, _ = rolling_stats(returns, window)
            result[f"{symbol}_return_mean_{window}"] = mean
            result[f"{symbol}_return_std_{window}"] = std
    return result


def build_quote_index_trend(history: StockHistory) -> dict[str, np.ndarray]:
    """沿已核行情涨跌口径连乘；停牌只保留内部累计水平，不造当日观察。"""
    level = np.full(len(history.dates), np.nan, dtype=np.float64)
    positions = np.flatnonzero(history.trade_state == TradeState.TRADED)
    if len(positions):
        factors = 1 + history.quote_return[positions]
        factors[0] = 1.0  # 首个已观测收盘定基为100，不使用窗外或发行价起点。
        with np.errstate(over="raise", invalid="raise", under="raise"):
            observed_levels = 100 * np.cumprod(factors, dtype=np.float64)
        if not np.all(np.isfinite(observed_levels) & (observed_levels > 0)):
            raise DataError(f"累计行情指数不是正有限值：{history.stock_id}")
        level[positions] = observed_levels

    values, averages = {}, {}
    for window in (20, 60):
        average, _, count = rolling_stats(level, window)
        average[count != window] = np.nan
        averages[window] = average
        values[f"quote_index_ma_deviation_{window}"] = ratio(level, average) - 1
    values["quote_index_ma_change_20_5"] = ratio(averages[20], lagged(averages[20], 5)) - 1

    # 先求截至各日的20日极值，再后移一日，明确排除正在判断是否突破的今天。
    high, low = np.full(len(level), np.nan), np.full(len(level), np.nan)
    if len(level) >= 20:
        windows = sliding_window_view(level, 20)
        high[19:], low[19:] = np.max(windows, axis=1), np.min(windows, axis=1)
    values["quote_index_prior_high_distance_20"] = ratio(level, lagged(high, 1)) - 1
    values["quote_index_prior_low_distance_20"] = ratio(level, lagged(low, 1)) - 1
    return values


def build_stock_features(history: StockHistory, markets: dict[str, np.ndarray]) -> FeatureBlock:
    values = {f"return_lag_{lag}": lagged(history.quote_return, lag) for lag in LAGS}
    for window in WINDOWS:
        mean, std, count = rolling_stats(history.quote_return, window)
        values.update({f"return_mean_{window}": mean, f"return_std_{window}": std,
                       f"return_valid_fraction_{window}": count / window})
    prices = history.prices
    opening, high, low, close = (prices[field] for field in ("open", "high", "low", "close"))
    values.update(candle_body=ratio(close, opening) - 1, high_low_range=ratio(high, low) - 1,
                  close_position=ratio(close - low, high - low), upper_shadow=ratio(high - np.maximum(opening, close), close),
                  lower_shadow=ratio(np.minimum(opening, close) - low, close))
    for field in ("volume_shares", "amount_yuan"):
        for window in (5, 20):
            mean, _, _ = rolling_stats(prices[field], window)
            values[f"{field}_relative_{window}"] = ratio(prices[field], lagged(mean, 1)) - 1
    turnover = prices["turnover"]
    mean, _, _ = rolling_stats(turnover, 20)
    values.update(turnover=turnover, turnover_change_1=turnover - lagged(turnover, 1),
                  turnover_relative_20=ratio(turnover, lagged(mean, 1)) - 1)
    values.update(markets)
    age = (history.dates - np.datetime64(history.listing_date, "D")).astype(np.float64)
    age[history.trade_state == TradeState.PRE_LISTING] = np.nan
    paused = (history.trade_state == TradeState.SUSPENDED).astype(np.float64)
    pause_mean, _, _ = rolling_stats(paused, 20)
    gap = np.full(len(history.dates), np.nan)
    previous = None
    for index, state in enumerate(history.trade_state):
        if previous is not None:
            gap[index] = index - previous
        if state == TradeState.TRADED:
            previous = index
    values.update(listing_age_days=age, suspended_days_20=pause_mean * 20, days_since_previous_trade=gap)
    values.update(build_quote_index_trend(history))
    matrix = np.column_stack([values[name] for name in FEATURE_NAMES])
    with np.errstate(over="raise", invalid="raise"):
        result = matrix.astype(np.float32)
    result.flags.writeable = False
    return FeatureBlock(FEATURE_NAMES, result)
