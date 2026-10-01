"""以手算窗口和独立最小二乘核验形态指标，不读取正式标签或研究缓存。"""

from __future__ import annotations

import unittest

import numpy as np

from technical_indicators import compute_indicators


def from_levels(levels):
    levels = np.asarray(levels, dtype=np.float64)
    returns = np.r_[0.0, levels[1:] / levels[:-1] - 1]
    return compute_indicators(returns, np.ones(len(levels), dtype=bool))


class TechnicalIndicatorTests(unittest.TestCase):
    def test_interface_all_fields_are_float64_and_calendar_aligned(self):
        result = compute_indicators(np.zeros(300, dtype=np.float64), np.ones(300, dtype=bool))
        expected = {"quote_index"}
        for window in (20, 40, 60):
            expected.update(f"{name}_{window}" for name in ("prior_width", "prior_drift", "breakout"))
        for window in (60, 120, 250):
            expected.update(f"{name}_{window}" for name in ("bottom_position", "prior_drawdown", "bottom_low_distance"))
        for window in (20, 40, 60):
            expected.update(f"{name}_{window}" for name in (
                "channel_gain", "channel_r2", "channel_width", "channel_offset", "channel_higher", "channel_sigma"))
        self.assertEqual(set(result), expected)
        for values in result.values():
            self.assertEqual(values.shape, (300,))
            self.assertEqual(values.dtype, np.float64)

    def test_quote_index_ignores_first_return_and_keeps_suspension_gap(self):
        returns = np.array([np.nan, .90, .10, 999.0, -.20], dtype=np.float64)
        traded = np.array([False, True, True, False, True])
        actual = compute_indicators(returns, traded)["quote_index"]
        np.testing.assert_allclose(actual, [np.nan, 100, 110, np.nan, 88], equal_nan=True)
        # 输入不被改写，尤其首收益不会被原地替换成0。
        self.assertEqual(returns[1], .90)
        self.assertEqual(returns[3], 999.0)

    def test_platform_extrema_and_drift_exclude_today(self):
        levels = 100 + np.arange(62, dtype=np.float64)
        levels[-1] = 400
        actual = from_levels(levels)
        for window in (20, 40, 60):
            prior = levels[-1-window:-1]
            self.assertAlmostEqual(actual[f"prior_width_{window}"][-1], max(prior) / min(prior) - 1)
            self.assertAlmostEqual(actual[f"prior_drift_{window}"][-1], prior[-1] / prior[0] - 1)
            self.assertAlmostEqual(actual[f"breakout_{window}"][-1], 400 / max(prior) - 1)
            self.assertTrue(np.isnan(actual[f"breakout_{window}"][:window]).all())

    def test_bottom_position_and_drawdown_use_previous_close(self):
        levels = 100 + 20 * np.sin(np.arange(300, dtype=np.float64) / 13)
        levels[-1] = 300
        actual = from_levels(levels)
        for window in (60, 120, 250):
            prior = levels[-1-window:-1]
            expected_position = (prior[-1] - min(prior)) / (max(prior) - min(prior))
            self.assertAlmostEqual(actual[f"bottom_position_{window}"][-1], expected_position)
            self.assertAlmostEqual(actual[f"prior_drawdown_{window}"][-1], prior[-1] / max(prior) - 1)
            self.assertAlmostEqual(actual[f"bottom_low_distance_{window}"][-1], prior[-1] / min(prior) - 1)

    def test_channel_matches_independent_least_squares(self):
        levels = 100 * np.exp(.004 * np.arange(90) + .015 * np.sin(np.arange(90) / 3))
        actual = from_levels(levels)
        for window in (20, 40, 60):
            x = np.arange(window, dtype=np.float64)
            logs = np.log(levels[-1-window:-1])
            design = np.column_stack((np.ones(window), x))
            intercept, slope = np.linalg.lstsq(design, logs, rcond=None)[0]
            residual = logs - (intercept + slope * x)
            r2 = 1 - sum(residual ** 2) / sum((logs - logs.mean()) ** 2)
            self.assertAlmostEqual(actual[f"channel_gain_{window}"][-1], np.exp(slope * (window - 1)) - 1)
            self.assertAlmostEqual(actual[f"channel_r2_{window}"][-1], r2)
            self.assertAlmostEqual(actual[f"channel_width_{window}"][-1], np.exp(max(residual) - min(residual)) - 1)
            self.assertAlmostEqual(actual[f"channel_sigma_{window}"][-1], np.sqrt(sum(residual ** 2) / window))
            self.assertAlmostEqual(actual[f"channel_offset_{window}"][-1], np.log(levels[-1]) - intercept - slope * window)

    def test_future_perturbation_leaves_every_prefix_unchanged(self):
        returns = .01 * np.sin(np.arange(400) / 7)
        traded = np.ones(400, dtype=bool)
        before = compute_indicators(returns, traded)
        changed_returns, changed_traded = returns.copy(), traded.copy()
        changed_returns[301:] = .15
        changed_traded[320:325] = False
        after = compute_indicators(changed_returns, changed_traded)
        for name in before:
            np.testing.assert_array_equal(before[name][:301], after[name][:301], err_msg=name)

    def test_suspensions_and_prelisting_do_not_compress_calendar(self):
        returns = np.full(100, .01, dtype=np.float64)
        traded = np.ones(100, dtype=bool)
        traded[:5] = False
        traded[30] = False
        result = compute_indicators(returns, traded)
        self.assertTrue(np.isnan(result["prior_width_20"][:25]).all())
        self.assertTrue(np.isfinite(result["prior_width_20"][25:31]).all())
        self.assertTrue(np.isnan(result["breakout_20"][30]))
        self.assertTrue(np.isnan(result["channel_offset_20"][30]))
        for name in ("prior_width_20", "channel_r2_20", "channel_higher_20"):
            self.assertTrue(np.isnan(result[name][31:51]).all(), name)
            self.assertTrue(np.isfinite(result[name][51]), name)

    def test_constant_windows_keep_defined_metrics_but_not_r2_or_bottom_position(self):
        result = compute_indicators(np.zeros(300, dtype=np.float64), np.ones(300, dtype=bool))
        for window in (20, 40, 60):
            self.assertTrue(np.isnan(result[f"channel_r2_{window}"][window:]).all())
            for name in ("channel_gain", "channel_width", "channel_higher", "channel_offset", "channel_sigma"):
                np.testing.assert_allclose(result[f"{name}_{window}"][window:], 0, atol=1e-14)
        for window in (60, 120, 250):
            self.assertTrue(np.isnan(result[f"bottom_position_{window}"][window:]).all())
            np.testing.assert_array_equal(result[f"prior_drawdown_{window}"][window:], 0)

    def test_perfect_exponential_up_channel(self):
        result = compute_indicators(np.full(90, .01, dtype=np.float64), np.ones(90, dtype=bool))
        for window in (20, 40, 60):
            np.testing.assert_allclose(result[f"channel_gain_{window}"][window:], 1.01 ** (window - 1) - 1)
            np.testing.assert_allclose(result[f"channel_r2_{window}"][window:], 1)
            np.testing.assert_allclose(result[f"channel_width_{window}"][window:], 0, atol=1e-14)
            np.testing.assert_allclose(result[f"channel_sigma_{window}"][window:], 0, atol=1e-14)
            np.testing.assert_allclose(result[f"channel_offset_{window}"][window:], 0, atol=1e-14)
            np.testing.assert_array_equal(result[f"channel_higher_{window}"][window:], 1)

    def test_both_half_window_high_and_low_must_strictly_rise(self):
        for low, high, expected in ((125, 250, 1), (50, 250, 0), (125, 175, 0), (100, 250, 0), (125, 200, 0)):
            with self.subTest(low=low, high=high):
                levels = np.r_[np.tile([100., 200.], 5), np.tile([float(low), float(high)], 5), 225.]
                result = from_levels(levels)
                self.assertEqual(result["channel_higher_20"][-1], expected)

    def test_empty_and_untraded_histories_preserve_contract(self):
        for count in (0, 1, 19, 300):
            result = compute_indicators(np.full(count, np.nan, dtype=np.float64), np.zeros(count, dtype=bool))
            self.assertEqual(len(result), 37)
            self.assertTrue(all(np.isnan(values).all() for values in result.values()))

    def test_reject_bad_shape_precision_and_invalid_observed_returns(self):
        cases = [
            (np.zeros((2, 1), dtype=np.float64), np.ones(2, dtype=bool)),
            (np.zeros(3, dtype=np.float64), np.ones(2, dtype=bool)),
            (np.zeros(2, dtype=np.float32), np.ones(2, dtype=bool)),
            (np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.int8)),
        ]
        for returns, traded in cases:
            with self.assertRaises(ValueError):
                compute_indicators(returns, traded)
        for bad in (np.nan, np.inf, -1.0, -1.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compute_indicators(np.array([0., bad]), np.ones(2, dtype=bool))
        # 首观察成交日没有可用收益仍可以定基100。
        result = compute_indicators(np.array([np.nan]), np.ones(1, dtype=bool))
        self.assertEqual(result["quote_index"][0], 100)


if __name__ == "__main__":
    unittest.main()
