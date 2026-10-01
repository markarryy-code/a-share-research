"""独立小样本核对：以直接布尔筛选检验分箱研究统计，不读取正式研究缓存。"""

from __future__ import annotations

import unittest

import numpy as np

import feature_rules as rules


AMPLITUDES = [.01, .02, .03, .05]


def sample_stage():
    # 同日故意放不同数量的股票，检验股票日权重与日期等权不能混用。
    dates = np.repeat(np.arange(28), [1 + day % 6 for day in range(28)])
    stocks = np.concatenate([np.arange(1 + day % 6) for day in range(28)])
    labels = np.array([-.07, -.05, -.03, -.02, -.01, 0, .01, .02, .03, .05, .07])
    y = labels[(dates * 3 + stocks * 2) % len(labels)]
    values = np.array([-2., -1., 0., .5, 1., 2., np.nan], dtype=np.float32)
    values = values[(dates + stocks * 2) % len(values)]
    stage = rules.Stage(np.arange(len(y)), dates, stocks, y,
                        rules.event_matrix(y, AMPLITUDES), dates // 4,
                        [str(day) for day in range(28)], 6)
    return values, stage


def raw_mask(values, cuts, lo, hi):
    # 不复用assign_bins；直接按文字定义计算闭开区间。
    if lo == len(cuts) + 1:
        return np.isnan(values)
    selected = np.isfinite(values)
    if lo:
        selected &= values >= cuts[lo - 1]
    if hi <= len(cuts):
        selected &= values < cuts[hi - 1]
    return selected


class FeatureRuleTests(unittest.TestCase):
    def test_strict_integer_threshold_and_empty_denominator(self):
        np.testing.assert_array_equal(
            rules.over_65(np.array([0, 13, 14, 650_000, 650_001]),
                          np.array([0, 20, 20, 1_000_000, 1_000_000])),
            [False, False, True, False, True])
        self.assertIsNone(rules.rate(0, 0))
        self.assertEqual(rules.rate(13, 20), .65)

    def test_flat_returns_and_exact_amplitude_boundaries(self):
        labels = np.array([-.05, -.03, -.02, -.01, 0., .01, .02, .03, .05])
        actual = rules.event_matrix(labels, AMPLITUDES)
        self.assertEqual(actual.shape, (9, 10))
        self.assertFalse(actual[4].any())
        np.testing.assert_array_equal(actual.sum(axis=0), [4, 4, 4, 4, 3, 3, 2, 2, 1, 1])

    def test_half_open_intervals_nan_and_valid_zero(self):
        values = np.array([-1., 0., .5, 1., 2., np.nan])
        cuts = np.array([0., 1.])
        np.testing.assert_array_equal(rules.assign_bins(values, cuts), [0, 1, 1, 2, 2, 3])
        self.assertEqual(rules.conditions(cuts), [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)])
        self.assertEqual(rules.condition_text("x", cuts, 1, 2), "0 <= x < 1")
        self.assertEqual(rules.condition_text("x", cuts, 0, 1), "x < 0")
        self.assertEqual(rules.condition_text("x", cuts, 2, 3), "x >= 1")
        self.assertEqual(rules.condition_text("x", cuts, 0, 3), "x 为有限值")
        self.assertEqual(rules.condition_text("x", cuts, 3, 4), "x 为 NaN")

    def test_discrete_values_use_all_midpoints(self):
        values = np.array([0., 0., .5, 1., 1., np.nan], dtype=np.float32)
        config = dict(discrete_max_unique=64, quantiles=[.1, .9])
        actual = rules.make_cuts(values, np.arange(6),
                                 dict(family="return_valid_fraction", name="return_valid_fraction_5"), config)
        np.testing.assert_array_equal(actual, [.25, .75])

    def test_market_quantiles_weight_each_date_once(self):
        values = np.array([0., 10., 10., 10., 10., 20., 30.], dtype=np.float32)
        dates = np.array([0, 1, 1, 1, 1, 2, 3])
        config = dict(discrete_max_unique=1, quantiles=[.5])
        market = rules.make_cuts(values, dates, dict(family="market", name="index_return_0"), config)
        stock = rules.make_cuts(values, dates, dict(family="return_lag", name="return_lag_0"), config)
        self.assertIn(15., market)
        self.assertNotIn(10., market)
        self.assertIn(10., stock)
        self.assertNotIn(15., stock)

    def test_constant_and_all_nan_features_still_have_two_states(self):
        config = dict(discrete_max_unique=64, quantiles=[.5])
        for values in (np.array([0., 0., np.nan]), np.array([np.nan, np.nan, np.nan])):
            cuts = rules.make_cuts(values, np.arange(3), dict(family="candle", name="candle_body"), config)
            self.assertEqual(len(cuts), 0)
            self.assertEqual(rules.conditions(cuts), [(0, 1), (1, 2)])

    def test_natural_cuts_separate_exact_zero_and_one(self):
        cuts = np.array(sorted(set(rules.natural_cuts(dict(family="candle", name="close_position")))))
        values = np.array([-np.finfo(np.float32).tiny, 0., np.nextafter(np.float32(0), np.float32(1)),
                           np.nextafter(np.float32(1), np.float32(0)), 1., np.nextafter(np.float32(1), np.float32(2))])
        bins = rules.assign_bins(values, cuts)
        self.assertLess(bins[0], bins[1])
        self.assertLess(bins[1], bins[2])
        self.assertLess(bins[3], bins[4])
        self.assertLess(bins[4], bins[5])

    def test_histogram_all_intervals_match_direct_raw_selection(self):
        values, stage = sample_stage()
        # 空箱、精确边界、NaN与不同日期股票组合同时覆盖。
        cuts = np.array([-3., -1., 0., .25, .5, 1., 2., 3.])
        histogram = rules.Histogram(values, cuts, stage)
        for lo, hi in rules.conditions(cuts):
            with self.subTest(lo=lo, hi=hi):
                selected = raw_mask(values, cuts, lo, hi)
                actual = histogram.interval(lo, hi)
                count = int(selected.sum())
                expected_hits = stage.events[selected].sum(axis=0)
                expected_days = np.array([np.count_nonzero(selected & (stage.dates == day)) for day in range(stage.date_count)])
                expected_daily_hits = np.array([stage.events[selected & (stage.dates == day)].sum(axis=0) for day in range(stage.date_count)])
                expected_stocks = np.array([np.count_nonzero(selected & (stage.stocks == stock)) for stock in range(stage.stock_count)])
                self.assertEqual(actual["n"], count)
                np.testing.assert_array_equal(actual["hits"], expected_hits)
                np.testing.assert_array_equal(actual["daily_n"], expected_days)
                np.testing.assert_array_equal(actual["daily_hits"], expected_daily_hits)
                self.assertEqual(actual["dates"], len(set(stage.dates[selected])))
                self.assertEqual(actual["stocks"], len(set(stage.stocks[selected])))
                for fold in range(7):
                    folded = selected & (stage.folds == fold)
                    self.assertEqual(actual["fold_n"][fold], int(folded.sum()))
                    self.assertEqual(actual["fold_dates"][fold], len(set(stage.dates[folded])))
                    np.testing.assert_array_equal(actual["fold_hits"][fold], stage.events[folded].sum(axis=0))
                if count:
                    present = expected_days > 0
                    self.assertAlmostEqual(actual["mean_y"], float(stage.y[selected].mean()))
                    self.assertAlmostEqual(actual["max_day_share"], max(expected_days) / count)
                    self.assertAlmostEqual(actual["max_stock_share"], max(expected_stocks) / count)
                    expected_rates = (expected_daily_hits[present] / expected_days[present, None]).mean(axis=0)
                    np.testing.assert_allclose(actual["daily_rates"], expected_rates)
                    # 每个触发股票日使用其当日全池基线，独立验证匹配日期权重。
                    expected_matched = np.mean([stage.events[stage.dates == day].mean(axis=0) for day in stage.dates[selected]], axis=0)
                    np.testing.assert_allclose(actual["matched"], expected_matched)
                else:
                    self.assertIsNone(actual["mean_y"])
                    self.assertIsNone(actual["max_day_share"])
                    self.assertIsNone(actual["max_stock_share"])
                    self.assertTrue(np.isnan(actual["daily_rates"]).all())
                    self.assertTrue(np.isnan(actual["matched"]).all())

    def test_market_condition_matches_same_date_market_baseline(self):
        _, stage = sample_stage()
        values = stage.dates.astype(float)
        actual = rules.Histogram(values, np.array([7., 18.]), stage).interval(1, 2)
        np.testing.assert_allclose(actual["hits"] / actual["n"], actual["matched"])

    def test_correlations_exclude_missing_and_constant_series(self):
        self.assertAlmostEqual(rules.correlation(np.array([1., 2., np.nan, 3.]), np.array([2., 4., 99., 6.])), 1.)
        self.assertIsNone(rules.correlation(np.array([1., 1.]), np.array([1., 2.])))
        self.assertIsNone(rules.correlation(np.array([1., 2.]), np.array([1., 1.])))
        self.assertIsNone(rules.correlation(np.array([np.nan, 2.]), np.array([1., 1.])))

    def test_classification_requires_support_and_strict_fold_majority(self):
        config = dict(support=dict(discovery_rows=1000, discovery_dates=60, discovery_stocks=100,
                                   confirmation_rows=300, confirmation_dates=30, confirmation_stocks=50,
                                   fold_rows=100, fold_dates=10, minimum_supported_folds=3))
        discovery = dict(n=1000, hits=np.array([660]), dates=60, stocks=100)
        confirmation = dict(n=300, hits=np.array([198]), dates=30, stocks=50,
                            fold_n=np.array([100, 100, 100, 0, 0, 0, 0]),
                            fold_hits=np.array([[70], [70], [58], [0], [0], [0], [0]]),
                            fold_dates=np.array([10, 10, 10, 0, 0, 0, 0]))
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "支持较充分的跨期记录")
        # 三个有支持折里只有一个严格过线，不因另外两折恰好65%而放行。
        confirmation["fold_hits"] = np.array([[68], [65], [65], [0], [0], [0], [0]])
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "两期超过65%但支持或分期不足")
        confirmation["fold_hits"] = np.array([[70], [70], [58], [0], [0], [0], [0]])
        confirmation["fold_dates"][2] = 9
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "两期超过65%但支持或分期不足")
        confirmation["fold_dates"][2] = 10
        discovery["stocks"] = 99
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "两期超过65%但支持或分期不足")

    def test_classification_preserves_sparse_and_unreproduced_records(self):
        config = dict(support=dict(discovery_rows=1000, discovery_dates=60, discovery_stocks=100,
                                   confirmation_rows=300, confirmation_dates=30, confirmation_stocks=50,
                                   fold_rows=100, fold_dates=10, minimum_supported_folds=3))
        discovery = dict(n=3, hits=np.array([3]), dates=3, stocks=1)
        confirmation = dict(n=3, hits=np.array([3]), dates=3, stocks=1,
                            fold_n=np.array([3, 0, 0, 0, 0, 0, 0]),
                            fold_hits=np.array([[3], [0], [0], [0], [0], [0], [0]]),
                            fold_dates=np.array([3, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "两期超过65%但支持或分期不足")
        confirmation["hits"][0] = 0
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "发现期超过65%但后段未复现")
        confirmation["n"] = 0
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "发现期超过65%但后段无样本")
        discovery["hits"][0] = 0
        confirmation["n"], confirmation["hits"][0] = 3, 3
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "仅后段探索超过65%")
        confirmation["hits"][0] = 0
        self.assertEqual(rules.classify(discovery, confirmation, 0, config), "未超过65%")

    def test_bootstrap_zero_triggers_and_constant_hit_rate(self):
        config = dict(block_market_days=5, repetitions=100)
        dates = np.arange(12)
        self.assertEqual(rules.bootstrap_interval(np.zeros(12), np.zeros(12), dates, config, 1), [None, None])
        self.assertEqual(rules.bootstrap_interval(np.full(12, 10), np.full(12, 7), dates, config, 1), [.7, .7])

    def test_bootstrap_moves_entire_date_blocks_and_keeps_zero_days(self):
        # 手工按相同随机起点展开循环日期块；日期零触发也必须随块一起移动。
        counts = np.array([0, 2, 0, 3, 10, 1, 0, 4], dtype=np.int64)
        hits = np.array([0, 1, 0, 3, 2, 0, 0, 4], dtype=np.int64)
        config = dict(block_market_days=3, repetitions=80)
        rng = np.random.default_rng(23)
        starts = rng.integers(0, 8, size=(80, 3))
        estimates = []
        for row in starts:
            days = [(int(start) + offset) % 8 for start in row for offset in range(3)][:8]
            denominator = sum(int(counts[day]) for day in days)
            if denominator:
                estimates.append(sum(int(hits[day]) for day in days) / denominator)
        expected = np.quantile(estimates, [.025, .975])
        actual = rules.bootstrap_interval(counts, hits, np.repeat(np.arange(8), 2), config, 23)
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
