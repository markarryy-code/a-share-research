"""独立手算参考核对49项特征与时间因果性，不借用实现中的滚动函数。"""

import math
import statistics
import sys
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from daily_return.samples.domain import DataError, StockHistory, TargetStatus
from daily_return.samples.features import FEATURE_NAMES, build_market_features, build_stock_features, feature_spec
from daily_return.samples.targets import build_next_day_targets


def make_history(length=100, listing_index=7, pauses=(9, 13, 30)):
    days, current = [], date(2026, 1, 1)
    while len(days) < length:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    dates = np.array(days, dtype="datetime64[D]")
    index = np.arange(length, dtype=np.float64)
    close = 20 + index / 10
    prices = {"open": close - .03, "high": close + .08, "low": close - .07, "close": close.copy(),
              "volume_shares": (index + 1) * 100, "amount_yuan": close * (index + 1) * 100,
              "turnover": (index % 8) / 1000}
    returns = np.resize(np.array([0, .02, -.01, .03]), length)
    state = np.ones(length, dtype=np.int8)
    state[list(pauses)] = 0
    state[:listing_index] = -1
    for values in [returns, *prices.values()]:
        values[state != 1] = np.nan
    history = StockHistory("SH600001", days[listing_index], dates, state, prices, returns)
    markets = {"sh000001": 100 + index, "sz399001": 200 + index * 3}
    return history, markets


def reference(history, closes, t):
    """使用Python的mean/pstdev和明示区间，独立于NumPy计算路径。"""
    def divide(left, right):
        return left / right if math.isfinite(left) and math.isfinite(right) and right > 0 else math.nan

    def statistic(values, end, width, kind):
        if end - width + 1 < 0:
            return math.nan
        window = [float(value) for value in values[end-width+1:end+1] if math.isfinite(value)]
        if kind == "valid_fraction":
            return len(window) / width
        if not window or (kind == "std" and len(window) < 2):
            return math.nan
        return statistics.mean(window) if kind == "mean" else statistics.pstdev(window)

    result = {}
    for k in (0, 1, 2, 5, 10, 20, 60):
        result[f"return_lag_{k}"] = float(history.quote_return[t-k]) if t >= k else math.nan
    for kind in ("mean", "std", "valid_fraction"):
        for width in (5, 10, 20, 60):
            result[f"return_{kind}_{width}"] = statistic(history.quote_return, t, width, kind)
    p = {field: float(values[t]) for field, values in history.prices.items()}
    result.update(candle_body=divide(p["close"], p["open"]) - 1, high_low_range=divide(p["high"], p["low"]) - 1,
                  close_position=divide(p["close"]-p["low"], p["high"]-p["low"]),
                  upper_shadow=divide(p["high"]-max(p["open"], p["close"]), p["close"]),
                  lower_shadow=divide(min(p["open"], p["close"])-p["low"], p["close"]))
    for field in ("volume_shares", "amount_yuan"):
        for width in (5, 20):
            result[f"{field}_relative_{width}"] = divide(p[field], statistic(history.prices[field], t-1, width, "mean")) - 1
    result["turnover"] = p["turnover"]
    result["turnover_change_1"] = p["turnover"] - float(history.prices["turnover"][t-1]) if t else math.nan
    result["turnover_relative_20"] = divide(p["turnover"], statistic(history.prices["turnover"], t-1, 20, "mean")) - 1
    for symbol in ("sh000001", "sz399001"):
        returns = [math.nan] + [float(closes[symbol][i]) / float(closes[symbol][i-1]) - 1 for i in range(1, len(history.dates))]
        result[f"{symbol}_return_0"] = returns[t]
        for kind in ("mean", "std"):
            for width in (5, 20):
                result[f"{symbol}_return_{kind}_{width}"] = statistic(returns, t, width, kind)
    result["listing_age_days"] = float((history.dates[t]-np.datetime64(history.listing_date, "D")) / np.timedelta64(1, "D")) if history.trade_state[t] >= 0 else math.nan
    result["suspended_days_20"] = sum(value == 0 for value in history.trade_state[t-19:t+1]) if t >= 19 else math.nan
    previous = [i for i in range(t) if history.trade_state[i] == 1]
    result["days_since_previous_trade"] = t - previous[-1] if previous else math.nan
    index_levels, previous_level = [], None
    for state, observed_return in zip(history.trade_state, history.quote_return, strict=True):
        if state != 1:
            index_levels.append(math.nan)
            continue
        previous_level = 100.0 if previous_level is None else previous_level * (1 + float(observed_return))
        index_levels.append(previous_level)

    def full_window(end, width):
        if end < width - 1:
            return []
        values = index_levels[end-width+1:end+1]
        return values if all(math.isfinite(value) for value in values) else []

    for width in (20, 60):
        values = full_window(t, width)
        result[f"quote_index_ma_deviation_{width}"] = divide(index_levels[t], statistics.mean(values)) - 1 if values else math.nan
    now, before = full_window(t, 20), full_window(t-5, 20)
    result["quote_index_ma_change_20_5"] = divide(statistics.mean(now), statistics.mean(before)) - 1 if now and before else math.nan
    previous_window = full_window(t-1, 20)
    for name, reducer in (("high", max), ("low", min)):
        result[f"quote_index_prior_{name}_distance_20"] = divide(index_levels[t], reducer(previous_window)) - 1 if previous_window else math.nan
    return result


class FeaturesTest(unittest.TestCase):
    def test_all_49_columns_against_independent_reference(self):
        history, closes = make_history()
        block = build_stock_features(history, build_market_features(closes))
        self.assertEqual(len(FEATURE_NAMES), 49)
        self.assertEqual(block.values.dtype, np.float32)
        for t in (0, 4, 7, 9, 15, 19, 20, 26, 27, 31, 49, 50, 51, 55, 59, 60, 79, 89, 90, 99):
            with self.subTest(index=t):
                expected = reference(history, closes, t)
                self.assertEqual(tuple(expected), FEATURE_NAMES)
                np.testing.assert_allclose(block.values[t], list(expected.values()), rtol=2e-6, atol=1e-7, equal_nan=True)

    def test_exact_first_available_dates_after_listing(self):
        history, closes = make_history(listing_index=7, pauses=())
        values = build_stock_features(history, build_market_features(closes)).values
        expected_first = {"quote_index_ma_deviation_20": 26, "quote_index_ma_deviation_60": 66,
                          "quote_index_ma_change_20_5": 31, "quote_index_prior_high_distance_20": 27,
                          "quote_index_prior_low_distance_20": 27}
        self.assertEqual(set(FEATURE_NAMES[44:]), set(expected_first))
        for name, first in expected_first.items():
            column = values[:, FEATURE_NAMES.index(name)]
            self.assertTrue(np.isnan(column[:first]).all(), name)
            self.assertTrue(np.isfinite(column[first:]).all(), name)
        self.assertEqual(feature_spec()["feature_set"], "price-volume-v2")
        self.assertEqual(feature_spec()["count"], 49)

    def test_trend_flat_series_keeps_zero_despite_raw_price_adjustment(self):
        history, closes = make_history(length=80, listing_index=0, pauses=())
        history = replace(history, quote_return=np.zeros(80))
        original = build_stock_features(history, build_market_features(closes)).values[:, 44:]
        changed_prices = {key: values.copy() for key, values in history.prices.items()}
        for field in ("open", "high", "low", "close"):
            changed_prices[field][40:] *= .5  # 模拟原价除权跳变，行情涨跌仍为0。
        actual = build_stock_features(replace(history, prices=changed_prices), build_market_features(closes)).values[:, 44:]
        np.testing.assert_array_equal(original, actual)
        np.testing.assert_array_equal(actual[59:], np.zeros((21, 5)))

    def test_prior_extrema_exclude_today_and_direction_is_a_five_day_change(self):
        history, closes = make_history(length=80, listing_index=0, pauses=())
        returns = np.zeros(80)
        returns[60], returns[61] = .1, -.2
        history = replace(history, quote_return=returns)
        values = build_stock_features(history, build_market_features(closes)).values
        self.assertAlmostEqual(values[60, FEATURE_NAMES.index("quote_index_ma_deviation_20")], 110 / 100.5 - 1, places=7)
        self.assertAlmostEqual(values[60, FEATURE_NAMES.index("quote_index_ma_change_20_5")], .005, places=7)
        self.assertAlmostEqual(values[60, FEATURE_NAMES.index("quote_index_prior_high_distance_20")], .1, places=7)
        self.assertAlmostEqual(values[60, FEATURE_NAMES.index("quote_index_prior_low_distance_20")], .1, places=7)
        self.assertAlmostEqual(values[61, FEATURE_NAMES.index("quote_index_prior_high_distance_20")], -.2, places=7)
        self.assertAlmostEqual(values[61, FEATURE_NAMES.index("quote_index_prior_low_distance_20")], -.12, places=7)

    def test_trend_pause_is_not_filled_and_windows_recover_on_exact_dates(self):
        history, closes = make_history(length=120, listing_index=0, pauses=(30,))
        values = build_stock_features(history, build_market_features(closes)).values
        self.assertTrue(np.isnan(values[30, 44:]).all())
        recovery = {"quote_index_ma_deviation_20": 50, "quote_index_ma_deviation_60": 90,
                    "quote_index_ma_change_20_5": 55, "quote_index_prior_high_distance_20": 51,
                    "quote_index_prior_low_distance_20": 51}
        for name, first in recovery.items():
            column = values[:, FEATURE_NAMES.index(name)]
            self.assertTrue(np.isnan(column[30:first]).all(), name)
            self.assertTrue(np.isfinite(column[first:]).all(), name)

    def test_trend_anchor_does_not_use_first_day_offering_return(self):
        history, closes = make_history(listing_index=7, pauses=())
        original = build_stock_features(history, build_market_features(closes)).values[:, 44:]
        returns = history.quote_return.copy()
        returns[7] = 3.0
        actual = build_stock_features(replace(history, quote_return=returns), build_market_features(closes)).values[:, 44:]
        np.testing.assert_array_equal(original, actual)

    def test_nonpositive_quote_index_cannot_be_silently_treated_as_missing(self):
        history, closes = make_history(listing_index=0, pauses=())
        returns = history.quote_return.copy()
        returns[10] = -1
        with self.assertRaisesRegex(DataError, "累计行情指数不是正有限值"):
            build_stock_features(replace(history, quote_return=returns), build_market_features(closes))

    def test_prefix_and_future_perturbation_preserve_past_features(self):
        history, closes = make_history()
        cutoff = 71
        original = build_stock_features(history, build_market_features(closes)).values
        prefix = build_stock_features(history.prefix(cutoff), build_market_features({key: values[:cutoff] for key, values in closes.items()})).values
        np.testing.assert_array_equal(original[:cutoff], prefix)
        changed_prices = {key: values.copy() for key, values in history.prices.items()}
        for values in changed_prices.values():
            values[cutoff:] *= 20
        changed_return = history.quote_return.copy()
        changed_return[cutoff:] += 2
        changed_market = {key: values.copy() for key, values in closes.items()}
        for values in changed_market.values():
            values[cutoff:] *= 5
        changed = replace(history, prices=changed_prices, quote_return=changed_return)
        actual = build_stock_features(changed, build_market_features(changed_market)).values
        np.testing.assert_array_equal(original[:cutoff], actual[:cutoff])

    def test_market_lag_does_not_skip_pause(self):
        history, closes = make_history(listing_index=0, pauses=(9,))
        block = build_stock_features(history, build_market_features(closes))
        self.assertTrue(np.isnan(block.values[10, FEATURE_NAMES.index("return_lag_1")]))
        self.assertEqual(block.values[10, FEATURE_NAMES.index("days_since_previous_trade")], 2)

    def test_empty_partial_and_single_observation_windows(self):
        history, closes = make_history(listing_index=7, pauses=())
        values = build_stock_features(history, build_market_features(closes)).values
        self.assertTrue(np.isnan(values[3, FEATURE_NAMES.index("return_valid_fraction_5")]))
        self.assertEqual(values[6, FEATURE_NAMES.index("return_valid_fraction_5")], 0)
        self.assertTrue(np.isnan(values[6, FEATURE_NAMES.index("return_mean_5")]))
        self.assertAlmostEqual(values[7, FEATURE_NAMES.index("return_valid_fraction_5")], .2)
        self.assertAlmostEqual(values[7, FEATURE_NAMES.index("return_mean_5")], history.quote_return[7])
        self.assertTrue(np.isnan(values[7, FEATURE_NAMES.index("return_std_5")]))
        self.assertEqual(values[19, FEATURE_NAMES.index("suspended_days_20")], 0)

    def test_flat_candle_zero_turnover_and_zero_return_preserve_meaning(self):
        history, closes = make_history(listing_index=0, pauses=())
        for field in ("open", "high", "low", "close"):
            history.prices[field][8] = 20
        block = build_stock_features(history, build_market_features(closes)).values
        self.assertEqual(block[8, FEATURE_NAMES.index("candle_body")], 0)
        self.assertEqual(block[8, FEATURE_NAMES.index("turnover")], 0)
        self.assertEqual(block[8, FEATURE_NAMES.index("return_lag_0")], 0)
        self.assertTrue(np.isnan(block[8, FEATURE_NAMES.index("close_position")]))

    def test_tiny_variance_is_not_lost_and_long_pause_keeps_origin(self):
        history, closes = make_history(listing_index=0, pauses=tuple(range(1, 70)))
        block = build_stock_features(history, build_market_features(closes)).values
        self.assertEqual(block[70, FEATURE_NAMES.index("days_since_previous_trade")], 70)
        self.assertEqual(block[69, FEATURE_NAMES.index("suspended_days_20")], 20)
        history, closes = make_history(listing_index=0, pauses=())
        returns = .1 + np.resize(np.array([-1e-12, 1e-12]), len(history.dates))
        history = replace(history, quote_return=returns)
        block = build_stock_features(history, build_market_features(closes)).values
        expected = statistics.pstdev(returns[-20:].tolist())
        self.assertGreater(block[-1, FEATURE_NAMES.index("return_std_20")], 0)
        np.testing.assert_allclose(block[-1, FEATURE_NAMES.index("return_std_20")], expected, rtol=2e-6, atol=0)

    def test_targets_follow_calendar_and_stay_independent_of_eligibility(self):
        history, _ = make_history(listing_index=7, pauses=(9,))
        targets = build_next_day_targets(history)
        self.assertEqual(targets.target_indices[8], 9)
        self.assertEqual(targets.statuses[8], TargetStatus.SUSPENDED)
        self.assertTrue(np.isnan(targets.values[8]))
        self.assertEqual(history.trade_state[9], 0)
        self.assertEqual(targets.statuses[9], TargetStatus.OBSERVED)
        self.assertEqual(targets.values[9], history.quote_return[10])
        self.assertEqual(targets.available_indices[9], 10)
        self.assertEqual(targets.statuses[-1], TargetStatus.OUTSIDE_CALENDAR)
        self.assertEqual(targets.target_indices[-1], -1)
        self.assertTrue(np.isnan(targets.values[-1]))


if __name__ == "__main__":
    unittest.main()
