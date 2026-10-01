"""形态规则的独立行为核对；以手算边界和直接行筛选校验三态与统计。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

import numpy as np

import chart_patterns as patterns
from feature_rules import Stage, event_matrix
from technical_indicators import compute_indicators


def first_reference(known, hit, stocks, dates):
    # 按股票与市场日建字典，独立于生产实现的数组偏移方式。
    lookup = {(int(stock), int(day)): (bool(valid), bool(signal))
              for valid, signal, stock, day in zip(known, hit, stocks, dates)}
    valid, signal = [], []
    for current_known, current_hit, stock, day in zip(known, hit, stocks, dates):
        previous = [lookup.get((int(stock), int(day) - lag), (False, False)) for lag in range(1, 6)]
        allowed = bool(current_known) and all(item[0] for item in previous)
        valid.append(allowed)
        signal.append(allowed and bool(current_hit) and all(not item[1] for item in previous))
    return np.array(valid), np.array(signal)


class ChartPatternTests(unittest.TestCase):
    def test_frozen_grid_has_165_rules_and_three_canonical_conditions(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/chart-patterns-v1.json").read_text(encoding="utf-8"))
        rules = patterns.frozen_rules(config)
        self.assertEqual(len(rules), 165)
        self.assertEqual(Counter(r["family"] for r in rules), {"platform": 81, "bottom_volume": 48, "ascending_channel": 36})
        self.assertEqual(len({r["rule_id"] for r in rules}), 165)
        canonical = {r["family"]: r for r in rules if r["canonical"]}
        self.assertEqual(len(canonical), 3)
        self.assertEqual((canonical["platform"]["window"], canonical["platform"]["width"], canonical["platform"]["breakout"], canonical["platform"]["volume"]), (20, .1, 0., 1.5))
        self.assertEqual((canonical["bottom_volume"]["window"], canonical["bottom_volume"]["drawdown"], canonical["bottom_volume"]["low_distance"], canonical["bottom_volume"]["volume"], canonical["bottom_volume"]["rising"]), (120, .2, .08, 2., True))
        self.assertEqual((canonical["ascending_channel"]["window"], canonical["ascending_channel"]["gain"], canonical["ascending_channel"]["width"], canonical["ascending_channel"]["position"]), (60, .1, .2, "all"))

    def test_platform_inclusive_width_drift_and_strict_breakout(self):
        rule = dict(family="platform", window=20, width=.1, breakout=.01, volume=None)
        fields = dict(prior_width_20=np.array([.1, .100001, .1, .1, .1, .1, np.nan, .1]),
                      prior_drift_20=np.array([.05, .05, .050001, .05, -.05, 0, 0, 0]),
                      breakout_20=np.array([.01, .02, .02, .010001, .02, 0, .02, np.nan]),
                      volume_ratio_20=np.full(8, np.nan), volume_history_complete=np.zeros(8))
        known, hit = patterns.evaluate_rule(rule, fields, np.ones(8, dtype=bool))
        np.testing.assert_array_equal(known, [True, True, True, True, True, True, False, False])
        np.testing.assert_array_equal(hit, [False, False, False, True, True, False, False, False])
        # 不限量比版本不能受到不相关的成交量缺失影响。
        del fields["volume_ratio_20"], fields["volume_history_complete"]
        np.testing.assert_array_equal(patterns.evaluate_rule(rule, fields, np.ones(8, dtype=bool))[1], hit)

    def test_required_volume_needs_complete_history_and_inclusive_ratio(self):
        rule = dict(family="platform", window=20, width=.1, breakout=0., volume=1.5)
        fields = dict(prior_width_20=np.full(5, .05), prior_drift_20=np.zeros(5), breakout_20=np.full(5, .01),
                      volume_ratio_20=np.array([1.5, 1.49999, np.nan, 2., 2.]),
                      volume_history_complete=np.array([1, 1, 1, 0, 1]))
        known, hit = patterns.evaluate_rule(rule, fields, np.array([True, True, True, True, False]))
        np.testing.assert_array_equal(known, [True, True, False, False, False])
        np.testing.assert_array_equal(hit, [True, False, False, False, False])

    def test_bottom_uses_prior_drawdown_and_low_distance(self):
        rule = dict(family="bottom_volume", window=120, drawdown=.2, low_distance=.08, volume=2., rising=False)
        fields = dict(prior_drawdown_120=np.array([-.2, -.2, -.19, -.2, np.nan, -.3]),
                      bottom_low_distance_120=np.array([.08, .080001, .02, .01, .02, .08]),
                      quote_return=np.array([-.1, .1, .1, 0, .1, .1]),
                      volume_ratio_20=np.full(6, 2.), volume_history_complete=np.ones(6))
        known, hit = patterns.evaluate_rule(rule, fields, np.ones(6, dtype=bool))
        np.testing.assert_array_equal(known, [True, True, True, True, False, True])
        np.testing.assert_array_equal(hit, [True, False, False, True, False, True])
        rule["rising"] = True
        np.testing.assert_array_equal(patterns.evaluate_rule(rule, fields, np.ones(6, dtype=bool))[1], [False, False, False, False, False, True])

    def test_bottom_structure_excludes_today_and_future_low(self):
        # 构造过去120日从100跌至70的历史，今天及以后另作大幅扰动。
        prices = np.r_[np.linspace(100., 70., 120), np.full(30, 71.)]
        returns = np.r_[0., prices[1:] / prices[:-1] - 1].astype(np.float64)
        changed = returns.copy()
        changed[120:] = np.linspace(-.8, .3, 30)
        rule = dict(family="bottom_volume", window=120, drawdown=.2, low_distance=.08, volume=2., rising=False)
        outputs = []
        for values in (returns, changed):
            fields = compute_indicators(values, np.ones(150, dtype=bool))
            fields.update(quote_return=values, volume_ratio_20=np.full(150, 2.), volume_history_complete=np.ones(150))
            self.assertAlmostEqual(fields["prior_drawdown_120"][120], -.3)
            self.assertAlmostEqual(fields["bottom_low_distance_120"][120], 0.)
            known, hit = patterns.evaluate_rule(rule, fields, np.ones(150, dtype=bool))
            self.assertFalse(known[119])
            self.assertTrue(known[120])
            self.assertTrue(hit[120])
            outputs.append((known[:121], hit[:121]))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_channel_edges_center_and_sigma_tolerance(self):
        sigma = np.log1p(.2) / 4
        sigmas = np.array([sigma] * 6 + [1e-12, np.nextafter(1e-12, np.inf), 0., np.nan, sigma + 1e-8])
        u = np.array([-1., -.1, 0., 1., 1.00001, -1.00001, 0., 0., 0., 0., 0.])
        fields = dict(channel_gain_60=np.full(11, .1), channel_r2_60=np.full(11, .5),
                      channel_sigma_60=sigmas, channel_offset_60=2 * sigmas * u)
        rule = dict(family="ascending_channel", window=60, gain=.1, width=.2, position="all", r_squared=.5, sigma_zero_tolerance=1e-12)
        known, hit = patterns.evaluate_rule(rule, fields, np.ones(11, dtype=bool))
        np.testing.assert_array_equal(known, [True, True, True, True, True, True, False, True, False, False, True])
        np.testing.assert_array_equal(hit, [True, True, True, True, False, False, False, True, False, False, False])
        rule["position"] = "lower"
        np.testing.assert_array_equal(patterns.evaluate_rule(rule, fields, np.ones(11, dtype=bool))[1], [True, True, False, False, False, False, False, False, False, False, False])
        rule["position"] = "upper"
        np.testing.assert_array_equal(patterns.evaluate_rule(rule, fields, np.ones(11, dtype=bool))[1], [False, False, True, True, False, False, False, True, False, False, False])

    def test_channel_requires_gain_and_r_squared_thresholds(self):
        rule = dict(family="ascending_channel", window=20, gain=.05, width=.1, position="all", r_squared=.5, sigma_zero_tolerance=1e-12)
        fields = dict(channel_gain_20=np.array([.05, .049999, .05, np.nan]), channel_r2_20=np.array([.5, .5, .499999, .5]),
                      channel_sigma_20=np.full(4, .01), channel_offset_20=np.zeros(4))
        known, hit = patterns.evaluate_rule(rule, fields, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(known, [True, True, True, False])
        np.testing.assert_array_equal(hit, [True, False, False, False])

    def test_first_trigger_requires_five_known_clear_days(self):
        known = np.ones(13, dtype=bool)
        hit = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
        valid, first = patterns.first_trigger(known, hit, np.zeros(13, dtype=int), np.arange(13))
        np.testing.assert_array_equal(valid, [False] * 5 + [True] * 8)
        np.testing.assert_array_equal(np.flatnonzero(first), [5, 12])

    def test_first_trigger_unknown_requires_new_clear_history(self):
        known = np.ones(12, dtype=bool)
        known[5] = False
        hit = np.zeros(12, dtype=bool)
        hit[11] = True
        valid, first = patterns.first_trigger(known, hit, np.zeros(12, dtype=int), np.arange(12))
        self.assertFalse(valid[:11].any())
        self.assertTrue(valid[11])
        np.testing.assert_array_equal(np.flatnonzero(first), [11])
        # 未知位置意外传入真信号也不能变成有效首次触发。
        hit[5] = True
        self.assertFalse(patterns.first_trigger(known, hit, np.zeros(12, dtype=int), np.arange(12))[1][5])

    def test_first_trigger_keeps_unscored_and_prior_period_signals(self):
        known = np.ones(14, dtype=bool)
        hit = np.zeros(14, dtype=bool)
        hit[[5, 6, 12, 13]] = True
        _, first = patterns.first_trigger(known, hit, np.zeros(14, dtype=int), np.arange(14))
        # 第5日无次日标签、复核期从第6日开始；不能把第6日补成新的首次。
        score_ids = np.array([6, 7, 8, 9, 10, 11, 12, 13])
        np.testing.assert_array_equal(first[score_ids], [False, False, False, False, False, False, True, False])

    def test_first_trigger_does_not_cross_stock_or_calendar_gaps(self):
        stocks = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
        dates = np.r_[np.arange(12), np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12])]
        known = np.ones(24, dtype=bool)
        known[17] = False
        hit = np.zeros(24, dtype=bool)
        hit[[5, 11, 12, 18, 23]] = True
        actual = patterns.first_trigger(known, hit, stocks, dates)
        expected = first_reference(known, hit, stocks, dates)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertFalse(actual[1][12])

    def test_masked_statistics_matches_direct_rows_with_global_ids(self):
        dates = np.repeat(np.arange(14), [1 + day % 4 for day in range(14)])
        stocks = np.concatenate([np.arange(1 + day % 4) for day in range(14)])
        labels = np.array([-.05, -.02, -.01, 0., .01, .02, .03, .05])
        y = labels[(dates + 2 * stocks) % len(labels)]
        ids = 3 * np.arange(len(y)) + 2
        events = event_matrix(y, [.01, .02, .03, .05])
        stage = Stage(ids, dates, stocks, y, events, dates // 2, list(map(str, range(14))), 4)
        hit = np.zeros(int(ids.max()) + 1, dtype=bool)
        hit[ids] = (dates + stocks) % 3 == 0
        selected = hit[ids]
        actual = patterns.masked_statistics(hit, stage)
        self.assertEqual(actual["n"], int(selected.sum()))
        np.testing.assert_array_equal(actual["hits"], events[selected].sum(axis=0))
        self.assertEqual(actual["dates"], len(set(dates[selected])))
        self.assertEqual(actual["stocks"], len(set(stocks[selected])))
        self.assertAlmostEqual(actual["mean_y"], float(y[selected].mean()))
        self.assertAlmostEqual(actual["median_y"], float(np.median(y[selected])))
        daily_n = np.array([np.count_nonzero(selected & (dates == day)) for day in range(14)])
        daily_hits = np.array([events[selected & (dates == day)].sum(axis=0) for day in range(14)])
        np.testing.assert_array_equal(actual["daily_n"], daily_n)
        np.testing.assert_array_equal(actual["daily_hits"], daily_hits)
        present = daily_n > 0
        np.testing.assert_allclose(actual["daily_rates"], (daily_hits[present] / daily_n[present, None]).mean(axis=0))
        np.testing.assert_allclose(actual["matched"], np.mean([events[dates == day].mean(axis=0) for day in dates[selected]], axis=0))
        self.assertAlmostEqual(actual["max_day_share"], max(Counter(dates[selected]).values()) / selected.sum())
        self.assertAlmostEqual(actual["max_stock_share"], max(Counter(stocks[selected]).values()) / selected.sum())
        for fold in range(7):
            choose = selected & (dates // 2 == fold)
            self.assertEqual(actual["fold_n"][fold], int(choose.sum()))
            self.assertEqual(actual["fold_dates"][fold], len(set(dates[choose])))
            np.testing.assert_array_equal(actual["fold_hits"][fold], events[choose].sum(axis=0))
        empty = patterns.masked_statistics(np.zeros_like(hit), stage)
        self.assertEqual(empty["n"], 0)
        self.assertEqual(empty["dates"], 0)
        self.assertEqual(empty["stocks"], 0)
        self.assertIsNone(empty["mean_y"])
        self.assertIsNone(empty["median_y"])
        self.assertIsNone(empty["max_day_share"])
        self.assertTrue(np.isnan(empty["daily_rates"]).all())
        self.assertTrue(np.isnan(empty["matched"]).all())
        np.testing.assert_array_equal(empty["hits"], np.zeros(10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
