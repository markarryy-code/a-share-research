"""逐项研究冻结特征与次日涨跌的条件频率；独立研究入口，不训练模型。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVENT_NAMES = ["上涨", "下跌", "上涨至少1%", "下跌至少1%", "上涨至少2%", "下跌至少2%",
               "上涨至少3%", "下跌至少3%", "上涨至少5%", "下跌至少5%"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def over_65(hits, count):
    """整数判定严格超过65%，无样本与恰好65%均不通过。"""
    return (np.asarray(count) > 0) & (20 * np.asarray(hits) > 13 * np.asarray(count))


def rate(hits, count):
    return float(hits / count) if count else None


def event_matrix(y, amplitudes):
    result = [y > 0, y < 0]
    for amplitude in amplitudes:
        result.extend((y >= amplitude, y <= -amplitude))
    return np.column_stack(result)


def natural_cuts(feature):
    """看目标前按指标含义确定边界，使用实际float32口径。"""
    family, name = feature["family"], feature["name"]
    if family in ("return_lag", "return_mean", "market", "quote_index_trend") and "std" not in name:
        cuts = [-.20, -.10, -.099, -.098, -.095, -.05, -.03, -.02, -.01, 0,
                .01, .02, .03, .05, .095, .098, .099, .10, .20]
    elif family == "return_std" or "std" in name:
        cuts = [0, .005, .01, .02, .03, .05, .10]
    elif family == "liquidity" or "relative" in name:
        cuts = [-.9, -.75, -.5, 0, .5, 1, 2, 4, 9]
    elif name == "listing_age_days":
        cuts = [5, 10, 20, 30, 60, 90, 180, 365, 730, 1095, 1825, 3650, 7300]
    elif name == "close_position" or family == "return_valid_fraction":
        cuts = [0, .25, .5, .75, .9, .95, 1]
    elif name == "turnover":
        cuts = [0, .005, .01, .02, .03, .05, .10, .20, .30, .50]
    elif name == "turnover_change_1":
        cuts = [-.10, -.05, -.02, -.01, 0, .01, .02, .05, .10]
    elif family == "candle":
        cuts = [-.10, -.05, -.03, -.02, -.01, 0, .01, .02, .03, .05, .10, .20]
    else:
        cuts = [0, 1, 2, 5, 10, 20, 60]
    # 精确零实体、零影线、收于最高点等状态独立成箱。
    exact = [0] + ([1] if name == "close_position" or family == "return_valid_fraction" else [])
    values = [float(np.float32(value)) for value in cuts]
    values.extend(float(np.nextafter(np.float32(value), np.float32(np.inf))) for value in exact)
    return values


def make_cuts(values, dates, feature, config):
    if feature["family"] == "market":
        _, first = np.unique(dates, return_index=True)
        values = values[first]
    finite = values[np.isfinite(values)].astype(np.float64)
    unique = np.unique(finite)
    if len(unique) < 2:
        return np.array([], dtype=np.float64)
    if len(unique) <= config["discrete_max_unique"]:
        return (unique[:-1] + unique[1:]) / 2
    candidates = np.r_[np.quantile(finite, config["quantiles"], method="linear"), natural_cuts(feature)]
    return np.unique(candidates[(candidates > unique[0]) & (candidates <= unique[-1])])


def assign_bins(values, cuts):
    bins = np.searchsorted(cuts, values.astype(np.float64), side="right")
    bins[~np.isfinite(values)] = len(cuts) + 1
    return bins


def conditions(cuts):
    """枚举全部连续区间，含左右尾部、全部有限值和独立NaN状态。"""
    finite_bins = len(cuts) + 1
    result = [(lo, hi) for lo in range(finite_bins) for hi in range(lo + 1, finite_bins + 1)]
    return result + [(finite_bins, finite_bins + 1)]


def condition_text(feature, cuts, lo, hi):
    if lo == len(cuts) + 1:
        return f"{feature} 为 NaN"
    lower = format(float(cuts[lo - 1]), ".17g") if lo else None
    upper = format(float(cuts[hi - 1]), ".17g") if hi <= len(cuts) else None
    if lower is None and upper is None:
        return f"{feature} 为有限值"
    if lower is None:
        return f"{feature} < {upper}"
    if upper is None:
        return f"{feature} >= {lower}"
    return f"{lower} <= {feature} < {upper}"


def prefix(array):
    return np.concatenate((np.zeros((1,) + array.shape[1:], dtype=array.dtype), np.cumsum(array, axis=0)), axis=0)


@dataclass
class Stage:
    ids: np.ndarray
    dates: np.ndarray
    stocks: np.ndarray
    y: np.ndarray
    events: np.ndarray
    folds: np.ndarray
    calendar: list
    stock_count: int

    def __post_init__(self):
        self.date_count = len(self.calendar)
        self.daily_n = np.bincount(self.dates, minlength=self.date_count)
        self.daily_hits = np.column_stack([np.bincount(self.dates, weights=event, minlength=self.date_count) for event in self.events.T])
        self.daily_rates = np.divide(self.daily_hits, self.daily_n[:, None], out=np.zeros_like(self.daily_hits), where=self.daily_n[:, None] > 0)
        self.base_rates = self.events.mean(axis=0)
        self.fold_days = [np.unique(self.dates[self.folds == fold]) for fold in range(7)]


class Histogram:
    """按分箱聚合后用前缀差求区间，避免为每条规则重扫股票日。"""

    def __init__(self, values, cuts, stage):
        self.stage = stage
        self.bins = assign_bins(values, cuts)
        size = len(cuts) + 2
        self.n = prefix(np.bincount(self.bins, minlength=size))
        self.hits = prefix(np.column_stack([np.bincount(self.bins, weights=event, minlength=size).astype(np.int64) for event in stage.events.T]))
        self.total_y = prefix(np.bincount(self.bins, weights=stage.y, minlength=size))
        day_key = self.bins * stage.date_count + stage.dates
        self.daily_n = prefix(np.bincount(day_key, minlength=size * stage.date_count).reshape(size, stage.date_count))
        self.daily_hits = prefix(np.stack([np.bincount(day_key, weights=event, minlength=size * stage.date_count).reshape(size, stage.date_count) for event in stage.events.T], axis=2))
        stock_key = self.bins * stage.stock_count + stage.stocks
        self.stock_n = prefix(np.bincount(stock_key, minlength=size * stage.stock_count).reshape(size, stage.stock_count))
        fold_key = self.bins * 7 + stage.folds
        self.fold_n = prefix(np.bincount(fold_key, minlength=size * 7).reshape(size, 7))
        self.fold_hits = prefix(np.stack([np.bincount(fold_key, weights=event, minlength=size * 7).reshape(size, 7).astype(np.int64) for event in stage.events.T], axis=2))

    def interval(self, lo, hi):
        n = int(self.n[hi] - self.n[lo])
        days, stocks = self.daily_n[hi] - self.daily_n[lo], self.stock_n[hi] - self.stock_n[lo]
        daily_hits = self.daily_hits[hi] - self.daily_hits[lo]
        present = days > 0
        daily_rates = (daily_hits[present] / days[present, None]).mean(axis=0) if present.any() else np.full(10, np.nan)
        matched = (days[:, None] * self.stage.daily_rates).sum(axis=0) / n if n else np.full(10, np.nan)
        # 按真实触发日期计支持，不以同日的股票行数代替独立日期数。
        fold_dates = np.array([np.count_nonzero(days[indices]) for indices in self.stage.fold_days])
        return dict(n=n, hits=self.hits[hi] - self.hits[lo], dates=int(present.sum()), stocks=int(np.count_nonzero(stocks)),
                    mean_y=rate(self.total_y[hi] - self.total_y[lo], n), daily_rates=daily_rates, matched=matched,
                    max_day_share=rate(days.max(initial=0), n), max_stock_share=rate(stocks.max(initial=0), n),
                    fold_n=self.fold_n[hi] - self.fold_n[lo], fold_hits=self.fold_hits[hi] - self.fold_hits[lo],
                    fold_dates=fold_dates, daily_n=days, daily_hits=daily_hits)


def has_support(stats, config, phase):
    return all(stats[key] >= config["support"][phase + "_" + label] for key, label in (("n", "rows"), ("dates", "dates"), ("stocks", "stocks")))


def classify(discovery, confirmation, event, config):
    dpass = bool(over_65(discovery["hits"][event], discovery["n"]))
    cpass = bool(over_65(confirmation["hits"][event], confirmation["n"]))
    supported = ((confirmation["fold_n"] >= config["support"]["fold_rows"]) & (confirmation["fold_dates"] >= config["support"]["fold_dates"]))
    passing = supported & over_65(confirmation["fold_hits"][:, event], confirmation["fold_n"])
    stable = int(supported.sum()) >= config["support"]["minimum_supported_folds"] and int(passing.sum()) * 2 > int(supported.sum())
    if dpass and cpass:
        if has_support(discovery, config, "discovery") and has_support(confirmation, config, "confirmation") and stable:
            return "支持较充分的跨期记录"
        return "两期超过65%但支持或分期不足"
    if dpass:
        return "发现期超过65%但后段未复现" if confirmation["n"] else "发现期超过65%但后段无样本"
    return "仅后段探索超过65%" if cpass else "未超过65%"


def describe_stage(stats, event, base):
    output = dict(n=stats["n"], hits=int(stats["hits"][event]), hit_rate=rate(stats["hits"][event], stats["n"]), dates=stats["dates"], stocks=stats["stocks"],
                  mean_return=stats["mean_y"], daily_equal_rate=float(stats["daily_rates"][event]) if stats["n"] else None,
                  unconditional_rate=float(base[event]), matched_date_base=float(stats["matched"][event]) if stats["n"] else None,
                  max_day_share=stats["max_day_share"], max_stock_share=stats["max_stock_share"])
    output["lift"] = output["hit_rate"] - output["unconditional_rate"] if stats["n"] else None
    return output


def correlation(values, targets):
    finite = np.isfinite(values)
    x, y = values[finite].astype(np.float64), targets[finite]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_interval(daily_n, daily_hits, stage_dates, config, seed):
    """连续市场日期块重抽样，同日股票整体移动，保留未触发日期。"""
    days = np.unique(stage_dates)
    n, hits, length = daily_n[days], daily_hits[days], len(days)
    width, repetitions = config["block_market_days"], config["repetitions"]
    starts = np.random.default_rng(seed).integers(0, length, size=(repetitions, math.ceil(length / width)))
    sampled = ((starts[:, :, None] + np.arange(width)) % length).reshape(repetitions, -1)[:, :length]
    denominators = n[sampled].sum(axis=1)
    estimates = hits[sampled].sum(axis=1)[denominators > 0] / denominators[denominators > 0]
    return [float(v) for v in np.quantile(estimates, [.025, .975])] if len(estimates) else [None, None]


def load_inputs(config):
    dataset_path = ROOT / config["prepared_run"] / "dataset.json"
    split_path = ROOT / config["split_run"] / "splits.json"
    dataset, splits = read_json(dataset_path), read_json(split_path)
    cache, split_cache = ROOT / dataset["cache_directory"], ROOT / splits["cache_directory"]
    records = {}

    def checked(path, expected=None):
        actual = digest(path)
        if expected is not None and expected != actual:
            raise ValueError(f"输入哈希不匹配：{path}")
        records[path.relative_to(ROOT).as_posix()] = actual

    for path in (dataset_path, split_path):
        checked(path)
    checked(cache / "prepared.json", dataset["cache_manifest_sha256"])
    checked(split_cache / "split-ready.json", splits["cache_manifest_sha256"])
    prepared, split_ready = read_json(cache / "prepared.json"), read_json(split_cache / "split-ready.json")
    if (dataset["state"], splits["state"], splits["preparation_id"]) != ("prepared", "split-ready", dataset["preparation_id"]):
        raise ValueError("具名P1/P2状态或父身份不一致")
    for name in ("X.npy", "y.npy", "rows.npy", "calendar.json", "stocks.json", "features.json"):
        checked(cache / name, prepared["files"][name]["sha256"])
    roles = [config["discovery_role"], *config["confirmation_roles"], "holdout.train"]
    indices = {}
    for role in roles:
        name = role + ".npy"
        checked(split_cache / name, split_ready["files"][name]["sha256"])
        indices[role] = np.load(split_cache / name, allow_pickle=False)
    rows = np.load(cache / "rows.npy", allow_pickle=False)
    calendar, stocks, feature_doc = (read_json(cache / name) for name in ("calendar.json", "stocks.json", "features.json"))
    features = feature_doc["features"]
    if feature_doc["feature_set"] != "price-volume-v2" or len(features) != 49:
        raise ValueError("必须是现行49列特征")
    discovery_ids = indices[config["discovery_role"]]
    confirmation_ids = np.concatenate([indices[role] for role in config["confirmation_roles"]])
    combined = np.sort(np.r_[discovery_ids, confirmation_ids])
    if not np.array_equal(combined, indices["holdout.train"]) or np.any(np.diff(combined) <= 0):
        raise ValueError("两期必须互斥且完整覆盖既有开发全集")
    boundary = calendar.index(splits["holdout_identity"]["start"])
    for ids in (discovery_ids, confirmation_ids):
        selected = rows[ids]
        if not np.all(selected["prediction_eligible"] & (selected["target_status"] == 1) &
                      (selected["target_idx"] == selected["as_of_idx"] + 1) & (selected["target_idx"] < boundary) &
                      (selected["label_available_idx"] >= selected["target_idx"]) & (selected["label_available_idx"] < boundary)):
            raise ValueError("目标可评分性或留出隔离失败")
    if rows[discovery_ids]["target_idx"].max() >= rows[confirmation_ids]["target_idx"].min():
        raise ValueError("发现与复核未按目标日期隔离")
    x = np.load(cache / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(cache / "y.npy", mmap_mode="r", allow_pickle=False)
    if list(x.shape) != dataset["shape"] or x.shape[1] != 49 or len(y) != len(rows):
        raise ValueError("数组形状与清单不一致")
    folds = np.concatenate([np.full(len(indices[role]), fold, dtype=np.int8) for fold, role in enumerate(config["confirmation_roles"])])
    return dict(x=x, y=y, rows=rows, calendar=calendar, stocks=stocks, features=features, records=records,
                discovery_ids=discovery_ids, confirmation_ids=confirmation_ids, folds=folds,
                dataset=dataset, splits=splits)


def make_stage(data, phase, amplitudes):
    ids = data[phase + "_ids"]
    rows = data["rows"][ids]
    # 日期隔离通过后只取显式开发行号，不统计全量y或留出月标签。
    y = data["y"][ids]
    if not np.isfinite(y).all():
        raise ValueError("评分角色存在非有限目标")
    folds = data["folds"] if phase == "confirmation" else np.zeros(len(ids), dtype=np.int8)
    return Stage(ids, rows["target_idx"], rows["stock_idx"], y, event_matrix(y, amplitudes), folds, data["calendar"], len(data["stocks"]))


def write_csv(path, records):
    if not records:
        Path(path).write_text("\n", encoding="utf-8", newline="\n")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def run(config_path, output):
    started = time.monotonic()
    config = read_json(config_path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", config)
    data = load_inputs(config)
    frozen = []
    for column, feature in enumerate(data["features"]):
        values = data["x"][data["discovery_ids"], column]
        dates = data["rows"][data["discovery_ids"]]["target_idx"]
        cuts = make_cuts(values, dates, feature, config)
        frozen.append(dict(column=column, **feature, cuts=cuts.tolist(), conditions=len(conditions(cuts))))
    write_json(output / "frozen_rules.json", dict(frozen_at=datetime.now(timezone.utc).isoformat(), feature_set="price-volume-v2", target_values_read=False,
               interval_convention="lower inclusive, upper exclusive; all contiguous finite bins plus NaN", features=frozen))
    discovery, confirmation = (make_stage(data, phase, config["amplitudes"]) for phase in ("discovery", "confirmation"))
    baselines = []
    for phase, stage in (("discovery", discovery), ("confirmation", confirmation)):
        for event, name in enumerate(EVENT_NAMES):
            baselines.append(dict(phase=phase, event=name, n=len(stage.ids), hits=int(stage.events[:, event].sum()), hit_rate=float(stage.base_rates[event]),
                                  dates=int(np.unique(stage.dates).size), first_target_date=stage.calendar[int(stage.dates.min())],
                                  last_target_date=stage.calendar[int(stage.dates.max())], mean_return=float(stage.y.mean()),
                                  daily_equal_rate=float(stage.daily_rates[stage.daily_n > 0, event].mean())))
    write_csv(output / "baselines.csv", baselines)
    overview, empirical, supported_records, representatives, fold_records, bin_records = [], [], [], [], [], []
    all_count = 0
    with gzip.open(output / "all_statistics.csv.gz", "wt", encoding="utf-8", newline="") as all_stream, \
            (output / "representative_selection.jsonl").open("w", encoding="utf-8", newline="\n") as selection_stream:
        all_writer = None
        for item in frozen:
            column, name, cuts = item["column"], item["name"], np.array(item["cuts"], dtype=np.float64)
            print(f"[{column + 1:02d}/49] {name}", flush=True)
            dx, cx = data["x"][discovery.ids, column], data["x"][confirmation.ids, column]
            dh = Histogram(dx, cuts, discovery)
            pairs = conditions(cuts)
            ds = [dh.interval(lo, hi) for lo, hi in pairs]
            chosen = {}
            for event in range(10):
                candidates = [i for i, stats in enumerate(ds) if has_support(stats, config, "discovery") and over_65(stats["hits"][event], stats["n"])]
                if candidates:
                    chosen[event] = max(candidates, key=lambda i: (ds[i]["n"], -i))
            # 代表只按发现期选覆盖最大的条件，先落盘再计算该特征的后段统计。
            selection_stream.write(json.dumps(dict(feature=name, selected={EVENT_NAMES[k]: v for k, v in chosen.items()}), ensure_ascii=False) + "\n")
            selection_stream.flush()
            ch = Histogram(cx, cuts, confirmation)
            feature_empirical, feature_supported, cross_period, discovery_candidates = 0, 0, 0, 0
            best_record = None
            for condition_index, ((lo, hi), d) in enumerate(zip(pairs, ds)):
                c = ch.interval(lo, hi)
                if not d["n"] and not c["n"]:
                    continue
                rule_id = f"F{column+1:02d}-R{condition_index:04d}"
                condition = condition_text(name, cuts, lo, hi)
                atomic = hi - lo == 1
                for event, event_name in enumerate(EVENT_NAMES):
                    state = classify(d, c, event, config)
                    record = dict(rule_id=rule_id, feature=name, family=item["family"], event=event_name, condition=condition,
                                  bin_lo=lo, bin_hi=hi, state=state, representative=chosen.get(event) == condition_index)
                    record.update({"discovery_" + k: v for k, v in describe_stage(d, event, discovery.base_rates).items()})
                    record.update({"confirmation_" + k: v for k, v in describe_stage(c, event, confirmation.base_rates).items()})
                    record.update(pooled_n=d["n"] + c["n"], pooled_hits=int(d["hits"][event] + c["hits"][event]),
                                  pooled_rate=rate(d["hits"][event] + c["hits"][event], d["n"] + c["n"]))
                    if all_writer is None:
                        all_writer = csv.DictWriter(all_stream, fieldnames=list(record), lineterminator="\n")
                        all_writer.writeheader()
                    all_writer.writerow(record)
                    all_count += 1
                    dpass, cpass = bool(over_65(d["hits"][event], d["n"])), bool(over_65(c["hits"][event], c["n"]))
                    discovery_candidates += dpass
                    cross_period += dpass and cpass
                    if dpass or cpass:
                        empirical.append(record)
                        feature_empirical += 1
                        for fold in range(7):
                            fold_records.append(dict(rule_id=rule_id, feature=name, event=event_name, fold=f"F{fold+1:02d}",
                                                     n=int(c["fold_n"][fold]), hits=int(c["fold_hits"][fold, event]),
                                                     dates=int(c["fold_dates"][fold]), hit_rate=rate(c["fold_hits"][fold, event], c["fold_n"][fold])))
                    if state == "支持较充分的跨期记录":
                        supported_records.append(record)
                        feature_supported += 1
                    if atomic:
                        bin_records.append(record)
                    if record["representative"]:
                        selected = (ch.bins >= lo) & (ch.bins < hi)
                        interval = bootstrap_interval(c["daily_n"], c["daily_hits"][:, event], confirmation.dates, config["bootstrap"],
                                                      config["bootstrap"]["seed"] + column * 10 + event)
                        representative = {**record, "confirmation_median_return": float(np.median(confirmation.y[selected])) if selected.any() else None,
                                          "confirmation_block_ci_low": interval[0], "confirmation_block_ci_high": interval[1]}
                        representatives.append(representative)
                        if event < 2 and (best_record is None or record["discovery_n"] > best_record["discovery_n"]):
                            best_record = representative
            overview.append(dict(number=column+1, feature=name, family=item["family"], condition_count=len(pairs), event_count=10,
                                 discovery_finite=int(np.isfinite(dx).sum()), confirmation_finite=int(np.isfinite(cx).sum()),
                                 discovery_pearson=correlation(dx, discovery.y), confirmation_pearson=correlation(cx, confirmation.y),
                                 empirical_over_65=feature_empirical, discovery_over_65=discovery_candidates,
                                 both_periods_over_65=cross_period, supported_cross_period=feature_supported,
                                 representative_rule=best_record["rule_id"] if best_record else "", representative_event=best_record["event"] if best_record else "",
                                 representative_condition=best_record["condition"] if best_record else "",
                                 representative_discovery_rate=best_record["discovery_hit_rate"] if best_record else None,
                                 representative_confirmation_rate=best_record["confirmation_hit_rate"] if best_record else None,
                                 representative_state=best_record["state"] if best_record else "没有满足发现期支持门槛的方向规律"))
            del dh, ch, ds
    write_csv(output / "feature_overview.csv", overview)
    write_csv(output / "empirical_over_65.csv", empirical)
    write_csv(output / "supported_cross_period.csv", supported_records)
    write_csv(output / "representatives.csv", representatives)
    write_csv(output / "fold_statistics.csv", fold_records)
    write_csv(output / "bin_statistics.csv", bin_records)
    final_records = []
    for path, original in data["records"].items():
        current = digest(ROOT / path)
        final_records.append(dict(path=path, sha256=original, final_sha256=current, unchanged=original == current))
    if not all(row["unchanged"] for row in final_records):
        raise ValueError("研究期间输入发生变化")
    write_csv(output / "input_files.csv", final_records)
    summary = dict(state="completed", analysis_version=config["analysis_version"], features_checked=len(overview), condition_event_checks=all_count,
                   empirical_over_65=len(empirical), supported_cross_period=len(supported_records), representatives=len(representatives),
                   representative_supported=sum(r["state"] == "支持较充分的跨期记录" for r in representatives),
                   discovery_rows=len(discovery.ids), confirmation_rows=len(confirmation.ids), holdout_target_values_read=False,
                   model_trained=False, inputs_unchanged=True, source_preparation=data["dataset"]["preparation_id"], source_splits=data["splits"]["split_id"],
                   elapsed_seconds=time.monotonic() - started, python=sys.version, numpy=np.__version__, script_sha256=digest(Path(__file__)), config_sha256=digest(config_path))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/feature-rules-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
