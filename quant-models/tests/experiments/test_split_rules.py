"""手工已知日期验证时间规则，不使用领域实现产生预期集合。"""

import copy
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from daily_return.samples.domain import ROW_DTYPE
from daily_return.experiments.domain import SamplePopulation, SplitError, build_split_plan, select_fold_rows, population_coverage


def small_population():
    days = tuple("2026-08-" + d for d in ("12", "13", "14", "17", "18", "19", "20", "21", "24", "25", "26", "27"))
    rows = np.zeros(len(days), dtype=ROW_DTYPE)
    rows["as_of_idx"] = np.arange(len(days))
    rows["target_idx"] = rows["label_available_idx"] = list(range(1, len(days))) + [-1]
    rows["trade_state"], rows["prediction_eligible"], rows["target_status"] = 1, True, 1
    rows["target_status"][-1] = 3
    population = SamplePopulation("fixture", "samples", "manifest", "source", "price-volume-v2", "next-market-day-quote-return-v1",
                                  days, ("SH600000",), {"start": days[0], "end": days[-1], "month_start": days[9]}, {}, rows)
    config = {"version": "expanding-target-date-v1", "folds": [
        {"id": "F01", "train_through": days[2], "validation_start": "2026-08-15", "validation_end": days[5]},
        {"id": "F02", "train_through": days[5], "validation_start": days[6], "validation_end": days[8]}],
        "holdout": {"start": days[9], "end": days[-1]}}
    return population, config


def selections(population, window):
    return {role.name: role for role in select_fold_rows(population, window).roles}


class SplitRulesTests(unittest.TestCase):
    def test_exact_expanding_roles_and_august_24_boundary(self):
        p, c = small_population()
        plan = build_split_plan(c, p)
        expected = [([0, 1], [2, 3, 4]), ([0, 1, 2, 3, 4], [5, 6, 7]), (list(range(8)), [8, 9, 10])]
        for window, (train, predict) in zip(plan.windows, expected, strict=True):
            role = selections(p, window)
            self.assertEqual(role["train"].indices.tolist(), train)
            self.assertEqual(role["predict"].indices.tolist(), predict)
            self.assertEqual(role["score"].indices.tolist(), predict)
        self.assertEqual(p.calendar[p.rows[8]["as_of_idx"]], "2026-08-24")
        self.assertEqual(p.calendar[p.rows[8]["target_idx"]], "2026-08-25")
        self.assertEqual(population_coverage(p)["eligible_without_target_date"], 1)

    def test_future_target_status_does_not_remove_prediction(self):
        p, c = small_population()
        p.rows[3]["target_status"], p.rows[3]["label_available_idx"] = 2, -1
        p.rows[4]["trade_state"], p.rows[4]["prediction_eligible"] = 0, False
        role = selections(p, build_split_plan(c, p).windows[0])
        self.assertEqual(role["predict"].indices.tolist(), [2, 3])
        self.assertEqual(role["score"].indices.tolist(), [2])
        self.assertEqual(role["score"].counts["information_ineligible"], 1)
        self.assertEqual(role["score"].counts["target_unavailable"], 1)

    def test_label_available_after_cutoff_is_excluded_until_later_fold(self):
        p, c = small_population()
        p.rows[1]["label_available_idx"] = 3
        p.rows[4]["label_available_idx"] = 6
        plan = build_split_plan(c, p)
        first, second = selections(p, plan.windows[0]), selections(p, plan.windows[1])
        self.assertEqual(first["train"].indices.tolist(), [0])
        self.assertEqual(first["train"].counts["label_not_mature"], 1)
        self.assertEqual(first["predict"].indices.tolist(), [2, 3, 4])
        self.assertEqual(first["score"].indices.tolist(), [2, 3])
        self.assertEqual(second["train"].indices.tolist(), [0, 1, 2, 3])

    def test_holdout_outcomes_do_not_change_development_indices(self):
        p, c = small_population()
        plan = build_split_plan(c, p)
        before = [selections(p, w) for w in plan.windows]
        p.rows[9]["target_status"], p.rows[9]["label_available_idx"] = 2, -1
        for window, original in zip(plan.windows[:-1], before[:-1], strict=True):
            changed = selections(p, window)
            for role in ("train", "predict", "score"):
                np.testing.assert_array_equal(original[role].indices, changed[role].indices)
        held = selections(p, plan.windows[-1])
        np.testing.assert_array_equal(before[-1]["predict"].indices, held["predict"].indices)
        self.assertEqual(held["score"].indices.tolist(), [8, 10])

    def test_daily_denominators_reconcile_and_indices_are_readonly(self):
        p, c = small_population()
        p.rows[3]["target_status"], p.rows[3]["label_available_idx"] = 2, -1
        for window in build_split_plan(c, p).windows:
            for role in select_fold_rows(p, window).roles:
                self.assertFalse(role.indices.flags.writeable)
                for key, value in role.counts.items():
                    self.assertEqual(sum(day[key] for day in role.daily), value)
                expected = role.counts["prediction_eligible"]
                if role.name != "predict":
                    expected -= role.counts["target_unavailable"] + role.counts["label_not_mature"]
                self.assertEqual(role.counts["selected"], expected)

    def test_empty_fit_prediction_or_scoring_set_is_not_skipped(self):
        for column, value in (("prediction_eligible", False), ("target_status", 2)):
            p, c = small_population()
            p.rows[column][:] = value
            with self.assertRaisesRegex(SplitError, "集合为空"):
                select_fold_rows(p, build_split_plan(c, p).windows[0])

    def test_overlap_gap_invalid_date_cutoff_and_changed_holdout_are_rejected(self):
        p, original = small_population()
        changes = [lambda c:c["folds"][1].update(validation_start="2026-08-19"),
                   lambda c:c["folds"][1].update(validation_start="2026-08-21"),
                   lambda c:c["folds"][0].update(train_through="2026-08-17"),
                   lambda c:c["folds"][0].update(validation_start="bad-date"),
                   lambda c:c["folds"][0].update(validation_end="2026-08-16"),
                   lambda c:c["folds"][1].update(id="F01"),
                   lambda c:c["holdout"].update(start="2026-08-24")]
        for change in changes:
            config = copy.deepcopy(original)
            change(config)
            with self.assertRaises((SplitError, ValueError)):
                build_split_plan(config, p)

    def test_repository_plan_keeps_the_seven_approved_periods(self):
        config = json.loads((Path(__file__).resolve().parents[2]/"configs/baseline.json").read_text(encoding="utf-8"))["split_plan"]
        self.assertEqual([(f["id"], f["train_through"], f["validation_start"], f["validation_end"]) for f in config["folds"]], [
            ("F01", "2024-12-31", "2025-01-01", "2025-03-31"), ("F02", "2025-03-31", "2025-04-01", "2025-06-30"),
            ("F03", "2025-06-30", "2025-07-01", "2025-09-30"), ("F04", "2025-09-30", "2025-10-01", "2025-12-31"),
            ("F05", "2025-12-31", "2026-01-01", "2026-03-31"), ("F06", "2026-03-31", "2026-04-01", "2026-06-30"),
            ("F07", "2026-06-30", "2026-07-01", "2026-08-24")])
        self.assertEqual(config["holdout"], {"start":"2026-08-25", "end":"2026-09-24"})
