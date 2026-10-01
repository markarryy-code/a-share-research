"""按冻结形态网格统计次日涨跌；前三类量价形态与核心题材数据状态分开记录。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from feature_rules import (ROOT, EVENT_NAMES, bootstrap_interval, classify, describe_stage,
                           digest, has_support, load_inputs, make_stage, over_65, rate,
                           read_json, write_csv, write_json)
from technical_indicators import compute_indicators


FAMILY_NAMES = {"platform": "平台突破", "bottom_volume": "底部放量", "ascending_channel": "上涨通道"}
VIEW_NAMES = {"all_qualifying_days": "全部满足日", "first_after_5_clear_days": "前5日无信号的新触发日"}


def frozen_rules(config):
    rules = []

    def append(family, window, canonical, **parameters):
        number = sum(row["family"] == family for row in rules) + 1
        prefix = {"platform": "P", "bottom_volume": "B", "ascending_channel": "C"}[family]
        rules.append(dict(rule_id=f"{prefix}{number:03d}", family=family, window=window,
                          canonical=canonical, **parameters))

    settings = config["platform"]
    for window, width, breakout, volume in itertools.product(settings["windows"], settings["maximum_widths"], settings["breakout_minimums"], settings["volume_minimums"]):
        canonical = (window, width, breakout, volume) == (20, .10, 0., 1.5)
        append("platform", window, canonical, width=width, breakout=breakout, volume=volume)
    settings = config["bottom_volume"]
    for window, drawdown, low, volume, rising in itertools.product(settings["windows"], settings["minimum_drawdowns"], settings["maximum_low_distances"], settings["volume_minimums"], settings["rising_today"]):
        canonical = (window, drawdown, low, volume, rising) == (120, .20, .08, 2., True)
        append("bottom_volume", window, canonical, drawdown=drawdown, low_distance=low, volume=volume, rising=rising)
    settings = config["ascending_channel"]
    for window, gain, width, position in itertools.product(settings["windows"], settings["minimum_gains"], settings["maximum_widths"], settings["positions"]):
        canonical = (window, gain, width, position) == (60, .10, .20, "all")
        append("ascending_channel", window, canonical, gain=gain, width=width, position=position,
               r_squared=settings["minimum_r_squared"], sigma_zero_tolerance=settings["sigma_zero_tolerance"])
    return rules


def rule_description(rule):
    window = rule["window"]
    if rule["family"] == "platform":
        volume = "不限量比" if rule["volume"] is None else f"量比≥{rule['volume']:g}"
        return f"此前{window}日平台宽≤{rule['width']:.0%}、首尾偏移绝对值≤{rule['width']/2:.1%}；今日突破>{rule['breakout']:.0%}；{volume}"
    if rule["family"] == "bottom_volume":
        direction = "且今日上涨" if rule["rising"] else "不限今日涨跌"
        return f"此前{window}日，昨日距高点回落≥{rule['drawdown']:.0%}、距低点≤{rule['low_distance']:.0%}；量比≥{rule['volume']:g}；{direction}"
    position = {"all": "全通道", "lower": "下半通道", "upper": "上半通道"}[rule["position"]]
    return f"此前{window}日拟合升幅≥{rule['gain']:.0%}、R²≥{rule['r_squared']:g}、宽度≤{rule['width']:.0%}；今日处于{position}"


def indicator_names():
    names = ["quote_index", "quote_return", "volume_ratio_20", "volume_history_complete"]
    for window in (20, 40, 60):
        names.extend(f"{name}_{window}" for name in ("prior_width", "prior_drift", "breakout", "channel_gain", "channel_r2", "channel_sigma", "channel_offset"))
    for window in (60, 120):
        names.extend(f"{name}_{window}" for name in ("prior_drawdown", "bottom_low_distance"))
    return names


def prepare_indicators(data, config, output):
    """只读原始涨跌CSV并核对P1指纹；不从标签或float32历史特征重建Q。"""
    source_config_path = ROOT / config["data_configuration"]
    source_config = read_json(source_config_path)
    if source_config["paths_relative_to"] != "module_root":
        raise ValueError("数据来源路径不是相对模型目录")
    history = (ROOT / source_config["history_directory"]).resolve()
    prepared = read_json(ROOT / data["dataset"]["cache_directory"] / "prepared.json")
    expected = prepared["identity"]["input_sha256"]
    rows, calendar = data["rows"], data["calendar"]
    date_indices = {day: i for i, day in enumerate(calendar)}
    development = np.r_[data["discovery_ids"], data["confirmation_ids"]]
    last_as_of = int(rows[development]["as_of_idx"].max())
    history_size = last_as_of + 1
    names = indicator_names()
    matrix = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float64, shape=(len(rows), len(names)))
    matrix[:] = np.nan
    existing_names = [item["name"] for item in data["features"]]
    r_column = existing_names.index("return_lag_0")
    volume_column = existing_names.index("volume_shares_relative_20")
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows["stock_idx"])) + 1, len(rows)]
    receipts = []
    for stock_index, stock in enumerate(data["stocks"]):
        start, stop = boundaries[stock_index:stock_index+2]
        ids = np.arange(start, stop)
        ids = ids[rows[ids]["as_of_idx"] <= last_as_of]
        days = rows[ids]["as_of_idx"]
        traded = np.zeros(history_size, dtype=bool)
        traded[days] = rows[ids]["trade_state"] == 1
        returns = np.full(history_size, np.nan, dtype=np.float64)
        seen = np.zeros(history_size, dtype=bool)
        path = history / "日线派生涨跌" / (stock["stock_id"] + ".csv")
        relative = Path(os.path.relpath(path, ROOT)).as_posix()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected.get(relative) != actual:
            raise ValueError(f"原始行情与P1冻结指纹不一致：{relative}")
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
            day = date_indices[row["date"]]
            if day > last_as_of:
                continue
            if seen[day] or not traded[day]:
                raise ValueError(f"行情日期重复或不是已核成交日：{stock['stock_id']} {row['date']}")
            seen[day] = True
            returns[day] = float(row["provider_change_pct"]) / 100
        if not np.array_equal(seen, traded) or not np.isfinite(returns[traded]).all():
            raise ValueError(f"原始行情日期与成交网格不一致：{stock['stock_id']}")
        if not np.array_equal(returns[days].astype(np.float32), data["x"][ids, r_column], equal_nan=True):
            raise ValueError(f"原始行情与现行收益列不一致：{stock['stock_id']}")
        indicators = compute_indicators(returns, traded)
        indicators["quote_return"] = returns
        volume_ratio = np.full(history_size, np.nan)
        volume_ratio[days] = data["x"][ids, volume_column].astype(np.float64) + 1
        indicators["volume_ratio_20"] = volume_ratio
        complete = np.zeros(history_size, dtype=np.float64)
        if history_size > 20:
            complete[20:] = np.lib.stride_tricks.sliding_window_view(traded[:-1], 20).all(axis=1)
        indicators["volume_history_complete"] = complete
        matrix[ids] = np.column_stack([indicators[name][days] for name in names])
        receipts.append(dict(path=relative, sha256=actual, rows=int(traded.sum()), last_as_of=calendar[last_as_of]))
        if (stock_index + 1) % 400 == 0 or stock_index == len(data["stocks"]) - 1:
            print(f"指标 {stock_index+1}/{len(data['stocks'])}", flush=True)
    matrix.flush()
    write_csv(output / "raw_return_sources.csv", receipts)
    write_json(output / "features.json", dict(feature_set="causal-chart-indicators-v1", names=names, dtype="float64",
               shape=list(matrix.shape), last_as_of=calendar[last_as_of], quote_return_source="original provider_change_pct/100",
               volume_ratio_source="existing float32 volume_shares_relative_20 + 1; requires 20 traded prior market days"))
    return matrix, names, receipts, last_as_of, source_config_path


def evaluate_rule(rule, fields, eligible):
    """先返回可判断性与命中两张布尔表，未知不等于不满足。"""
    window, family = rule["window"], rule["family"]
    if family == "platform":
        width, drift, breakout = (fields[f"{name}_{window}"] for name in ("prior_width", "prior_drift", "breakout"))
        known = eligible & np.isfinite(width) & np.isfinite(drift) & np.isfinite(breakout)
        hit = (width <= rule["width"]) & (np.abs(drift) <= rule["width"] / 2) & (breakout > rule["breakout"])
    elif family == "bottom_volume":
        drawdown, low = (fields[f"{name}_{window}"] for name in ("prior_drawdown", "bottom_low_distance"))
        known = eligible & np.isfinite(drawdown) & np.isfinite(low)
        hit = (drawdown <= -rule["drawdown"]) & (low <= rule["low_distance"])
        if rule["rising"]:
            known &= np.isfinite(fields["quote_return"])
            hit &= fields["quote_return"] > 0
    else:
        gain, r2, sigma, offset = (fields[f"{name}_{window}"] for name in ("channel_gain", "channel_r2", "channel_sigma", "channel_offset"))
        known = eligible & np.isfinite(gain) & np.isfinite(r2) & np.isfinite(sigma) & np.isfinite(offset) & (sigma > rule["sigma_zero_tolerance"])
        u = np.divide(offset, 2 * sigma, out=np.full(len(sigma), np.nan), where=known)
        hit = (gain >= rule["gain"]) & (r2 >= rule["r_squared"]) & (4 * sigma <= np.log1p(rule["width"])) & (u >= -1) & (u <= 1)
        if rule["position"] == "lower":
            hit &= u < 0
        elif rule["position"] == "upper":
            hit &= u >= 0
    if rule.get("volume") is not None:
        volume = fields["volume_ratio_20"]
        known &= np.isfinite(volume) & (fields["volume_history_complete"] == 1)
        hit &= volume >= rule["volume"]
    return known, hit & known


def first_trigger(known, hit, stock_indices, as_of_indices, clear_days=5):
    """在完整信息日网格检查此前连续可判断的无信号日，先于任何标签筛选。"""
    valid, clear = known.copy(), np.ones(len(known), dtype=bool)
    for lag in range(1, clear_days + 1):
        previous_known = np.zeros(len(known), dtype=bool)
        previous_clear = np.zeros(len(known), dtype=bool)
        contiguous = ((stock_indices[lag:] == stock_indices[:-lag]) & (as_of_indices[lag:] - as_of_indices[:-lag] == lag))
        previous_known[lag:] = known[:-lag] & contiguous
        previous_clear[lag:] = ~hit[:-lag] & contiguous
        valid &= previous_known
        clear &= previous_clear
    return valid, valid & hit & clear


def masked_statistics(hit, stage):
    selected = hit[stage.ids]
    n = int(selected.sum())
    dates, stocks, outcomes = stage.dates[selected], stage.stocks[selected], stage.events[selected]
    days = np.bincount(dates, minlength=stage.date_count)
    daily_hits = np.column_stack([np.bincount(dates, weights=event, minlength=stage.date_count) for event in outcomes.T])
    present = days > 0
    stock_counts = np.bincount(stocks, minlength=stage.stock_count)
    fold_n = np.bincount(stage.folds[selected], minlength=7)
    fold_hits = np.column_stack([np.bincount(stage.folds[selected], weights=event, minlength=7).astype(np.int64) for event in outcomes.T])
    return dict(n=n, hits=outcomes.sum(axis=0), dates=int(present.sum()), stocks=int(np.count_nonzero(stock_counts)),
                mean_y=float(stage.y[selected].mean()) if n else None, median_y=float(np.median(stage.y[selected])) if n else None,
                daily_rates=(daily_hits[present] / days[present, None]).mean(axis=0) if n else np.full(10, np.nan),
                matched=(days[:, None] * stage.daily_rates).sum(axis=0) / n if n else np.full(10, np.nan),
                max_day_share=rate(days.max(initial=0), n), max_stock_share=rate(stock_counts.max(initial=0), n),
                fold_n=fold_n, fold_hits=fold_hits, fold_dates=np.array([np.count_nonzero(days[indices]) for indices in stage.fold_days]),
                daily_n=days, daily_hits=daily_hits)


def write_records(path, records, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def run(config_path, output):
    started = time.monotonic()
    config = read_json(config_path)
    sample_config = read_json(ROOT / config["sample_configuration"])
    calculation_sources = ["analysis/chart_patterns.py", "analysis/technical_indicators.py", "analysis/feature_rules.py",
                           config["sample_configuration"], config["data_configuration"]]
    initial_source_hashes = {name: digest(ROOT/name) for name in calculation_sources}
    initial_config_hash = digest(config_path)
    output.mkdir(parents=True, exist_ok=False)
    rules = frozen_rules(config)
    if len(rules) != 165 or sum(rule["canonical"] for rule in rules) != 3:
        raise ValueError("本轮必须是165条规则及3条预设说明条件")
    write_json(output / "configuration.json", config)
    write_json(output / "sample_configuration.json", sample_config)
    write_json(output / "rules.json", dict(frozen_at=datetime.now(timezone.utc).isoformat(), target_values_read=False, rules=rules))
    data = load_inputs(sample_config)
    matrix, names, raw_sources, last_as_of, data_config_path = prepare_indicators(data, config, output)
    fields = {name: np.array(matrix[:, i]) for i, name in enumerate(names)}
    del matrix
    discovery, confirmation = (make_stage(data, phase, config["amplitudes"]) for phase in ("discovery", "confirmation"))
    rows = data["rows"]
    eligible = rows["prediction_eligible"] & (rows["as_of_idx"] <= last_as_of)
    records, coverage, fold_records, diagnostics = [], [], [], []
    old_columns = {item["name"]: column for column, item in enumerate(data["features"])}
    canonical_rows, discovery_choices, pairs = [], {}, {}
    for rule_number, rule in enumerate(rules):
        known, hit = evaluate_rule(rule, fields, eligible)
        first_known, first_hit = first_trigger(known, hit, rows["stock_idx"], rows["as_of_idx"], config["first_trigger_clear_days"])
        for view, valid, selected in ((config["views"][0], known, hit), (config["views"][1], first_known, first_hit)):
            d, c = masked_statistics(selected, discovery), masked_statistics(selected, confirmation)
            pairs[(rule["rule_id"], view)] = (d, c)
            coverage.append(dict(rule_id=rule["rule_id"], family=FAMILY_NAMES[rule["family"]], view=view,
                                 full_eligible_information_days=int(eligible.sum()), full_known=int(valid.sum()), full_signals=int(selected.sum()),
                                 discovery_known_scored=int(valid[discovery.ids].sum()), discovery_signals_scored=d["n"],
                                 confirmation_known_scored=int(valid[confirmation.ids].sum()), confirmation_signals_scored=c["n"],
                                 signal_targets_unscorable=int(np.count_nonzero(selected & (rows["target_status"] != 1)))))
            for phase, stage in (("discovery", discovery), ("confirmation", confirmation)):
                selected_ids = stage.ids[selected[stage.ids]]
                for feature, shift in (("volume_shares_relative_20", 1), ("amount_yuan_relative_20", 1), ("turnover", 0)):
                    values = data["x"][selected_ids, old_columns[feature]].astype(np.float64) + shift
                    finite_values = values[np.isfinite(values)]
                    diagnostics.append(dict(rule_id=rule["rule_id"], view=view, phase=phase,
                                            metric=feature + ("_plus_one" if shift else ""), n=len(values), finite_n=len(finite_values),
                                            mean=float(finite_values.mean()) if len(finite_values) else None,
                                            median=float(np.median(finite_values)) if len(finite_values) else None))
            for event, event_name in enumerate(EVENT_NAMES):
                state = classify(d, c, event, sample_config) if d["n"] + c["n"] else "两期均无可评分样本"
                record = dict(rule_id=rule["rule_id"], family=FAMILY_NAMES[rule["family"]], view=view, event=event_name,
                              condition=rule_description(rule), canonical=rule["canonical"], state=state)
                record.update({"discovery_" + key: value for key, value in describe_stage(d, event, discovery.base_rates).items()})
                record.update({"confirmation_" + key: value for key, value in describe_stage(c, event, confirmation.base_rates).items()})
                record.update(discovery_median_return=d["median_y"], confirmation_median_return=c["median_y"],
                              pooled_n=d["n"]+c["n"], pooled_hits=int(d["hits"][event]+c["hits"][event]),
                              pooled_rate=rate(d["hits"][event]+c["hits"][event], d["n"]+c["n"]))
                records.append(record)
                if rule["canonical"]:
                    canonical_rows.append(record)
                if has_support(d, sample_config, "discovery") and over_65(d["hits"][event], d["n"]):
                    key = (rule["family"], view, event)
                    if key not in discovery_choices or d["n"] > discovery_choices[key]["discovery_n"]:
                        discovery_choices[key] = record
                for fold in range(7):
                    fold_records.append(dict(rule_id=rule["rule_id"], family=FAMILY_NAMES[rule["family"]], view=view, event=event_name,
                                             fold=f"F{fold+1:02d}", n=int(c["fold_n"][fold]), hits=int(c["fold_hits"][fold,event]),
                                             dates=int(c["fold_dates"][fold]), hit_rate=rate(c["fold_hits"][fold,event], c["fold_n"][fold])))
        if (rule_number+1) % 20 == 0 or rule_number == len(rules)-1:
            print(f"形态规则 {rule_number+1}/{len(rules)}", flush=True)
    pooled = [row for row in records if over_65(row["pooled_hits"], row["pooled_n"])]
    empirical = [row for row in records if over_65(row["discovery_hits"], row["discovery_n"]) or over_65(row["confirmation_hits"], row["confirmation_n"])]
    supported = [row for row in records if row["state"] == "支持较充分的跨期记录"]
    representatives = {}
    for record in canonical_rows + list(discovery_choices.values()):
        representatives[(record["rule_id"], record["view"], record["event"])] = record
    selected_keys = {(r["rule_id"], r["view"], r["event"]) for r in discovery_choices.values()}
    representative_rows = []
    for key, record in representatives.items():
        event = EVENT_NAMES.index(record["event"])
        _, c = pairs[key[:2]]
        ci = bootstrap_interval(c["daily_n"], c["daily_hits"][:, event], confirmation.dates, sample_config["bootstrap"],
                                sample_config["bootstrap"]["seed"] + rules.index(next(r for r in rules if r["rule_id"] == key[0]))*20 + event)
        representative_rows.append({**record, "discovery_selected": key in selected_keys,
                                    "confirmation_block_ci_low": ci[0], "confirmation_block_ci_high": ci[1]})
    columns = list(records[0])
    for filename, content in (("all_statistics.csv", records), ("pooled_over_65.csv", pooled), ("empirical_over_65.csv", empirical),
                              ("supported_cross_period.csv", supported), ("canonical_rules.csv", canonical_rows),
                              ("discovery_selected.csv", list(discovery_choices.values()))):
        write_records(output / filename, content, columns)
    write_csv(output / "representatives.csv", representative_rows)
    write_csv(output / "coverage.csv", coverage)
    write_csv(output / "fold_statistics.csv", fold_records)
    write_csv(output / "liquidity_diagnostics.csv", diagnostics)
    family_summary = []
    for family in FAMILY_NAMES.values():
        relevant = [row for row in records if row["family"] == family]
        family_summary.append(dict(family=family, rules=len({r["rule_id"] for r in relevant}), rule_events=len(relevant),
                                   pooled_over_65=sum(bool(over_65(r["pooled_hits"], r["pooled_n"])) for r in relevant),
                                   both_period_over_65=sum(bool(over_65(r["discovery_hits"], r["discovery_n"]) and over_65(r["confirmation_hits"], r["confirmation_n"])) for r in relevant),
                                   supported=sum(r["state"] == "支持较充分的跨期记录" for r in relevant)))
    write_csv(output / "family_summary.csv", family_summary)
    baseline = []
    for phase, stage in (("discovery", discovery), ("confirmation", confirmation)):
        for event, label in enumerate(EVENT_NAMES):
            baseline.append(dict(phase=phase, event=label, n=len(stage.ids), hits=int(stage.events[:,event].sum()),
                                 hit_rate=float(stage.base_rates[event]), dates=int(np.unique(stage.dates).size)))
    write_csv(output / "baselines.csv", baseline)
    receipts = []
    for relative, original in data["records"].items():
        current = digest(ROOT / relative)
        receipts.append(dict(path=relative, sha256=original, end_sha256=current, unchanged=current == original))
    for record in raw_sources:
        current = digest(ROOT / record["path"])
        receipts.append(dict(path=record["path"], sha256=record["sha256"], end_sha256=current, unchanged=current == record["sha256"]))
    if not all(row["unchanged"] for row in receipts):
        raise ValueError("研究期间输入发生变化")
    if any(digest(ROOT/name) != original for name, original in initial_source_hashes.items()) or digest(config_path) != initial_config_hash:
        raise ValueError("研究期间计算代码或配置发生变化")
    write_csv(output / "input_files.csv", receipts)
    write_json(output / "summary.json", dict(state="computed", rules=len(rules), views=2, rule_events=len(records),
               pooled_over_65=len(pooled), empirical_over_65=len(empirical), supported_cross_period=len(supported),
               discovery_selected=len(discovery_choices), discovery_rows=len(discovery.ids), confirmation_rows=len(confirmation.ids),
               discovery_and_confirmation_are_reused_development_data=True, holdout_target_values_read=False,
               model_trained=False, core_theme_status=config["core_theme"]["status"], inputs_unchanged=True,
               sources_sha256=initial_source_hashes,
               config_sha256=initial_config_hash, features_sha256=digest(output/"features.npy"), elapsed_seconds=time.monotonic()-started))
    print(json.dumps(read_json(output/"summary.json"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/chart-patterns-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
