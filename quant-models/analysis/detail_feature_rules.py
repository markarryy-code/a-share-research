"""为预先选定的代表条件补充年度、季度和逐日证据，不重新选择条件。"""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

from feature_rules import EVENT_NAMES, digest, load_inputs, make_stage, read_json, write_csv, write_json


def selected_values(values, item, lo, hi):
    values = values.astype(np.float64)
    cuts = item["cuts"]
    if lo == len(cuts) + 1:
        return np.isnan(values)
    selected = np.isfinite(values)
    if lo:
        selected &= values >= cuts[lo - 1]
    if hi <= len(cuts):
        selected &= values < cuts[hi - 1]
    return selected


def run(output):
    config = read_json(output / "configuration.json")
    data = load_inputs(config)
    frozen = {item["name"]: item for item in read_json(output / "frozen_rules.json")["features"]}
    with (output / "representatives.csv").open(encoding="utf-8", newline="") as stream:
        representatives = list(csv.DictReader(stream))
    with (output / "empirical_over_65.csv").open(encoding="utf-8", newline="") as stream:
        empirical = list(csv.DictReader(stream))
    # 合并开发期符合率是用户阈值的直接读法；分期失败仍保留在原表。
    pooled = [row for row in empirical if 20 * int(row["pooled_hits"]) > 13 * int(row["pooled_n"])]
    write_csv(output / "pooled_over_65.csv", pooled)
    write_csv(output / "pooled_feature_counts.csv", [dict(number=item["column"]+1, feature=item["name"],
              pooled_over_65=sum(row["feature"] == item["name"] for row in pooled)) for item in frozen.values()])
    temporal, daily, sensitivity = [], [], []
    calendar = np.asarray(data["calendar"])
    for phase in ("discovery", "confirmation"):
        stage = make_stage(data, phase, config["amplitudes"])
        labels = calendar[stage.dates]
        years = np.array([date[:4] for date in labels])
        quarters = np.array([f"{date[:4]}Q{(int(date[5:7])-1)//3+1}" for date in labels])
        for record in representatives:
            item = frozen[record["feature"]]
            selected = selected_values(data["x"][stage.ids, item["column"]], item, int(record["bin_lo"]), int(record["bin_hi"]))
            event = EVENT_NAMES.index(record["event"])
            common = dict(rule_id=record["rule_id"], feature=record["feature"], event=record["event"], phase=phase)
            for kind, groups in (("year", years), ("quarter", quarters)):
                for group in np.unique(groups):
                    group_mask = groups == group
                    mask = selected & group_mask
                    n = int(mask.sum())
                    temporal.append({**common, "period_kind": kind, "period": group, "n": n,
                                     "hits": int(stage.events[mask, event].sum()),
                                     "hit_rate": float(stage.events[mask, event].mean()) if n else None,
                                     "dates": int(np.unique(stage.dates[mask]).size), "stocks": int(np.unique(stage.stocks[mask]).size),
                                     "mean_return": float(stage.y[mask].mean()) if n else None,
                                     "median_return": float(np.median(stage.y[mask])) if n else None,
                                     "unconditional_rate": float(stage.events[group_mask, event].mean())})
            counts = np.bincount(stage.dates[selected], minlength=len(calendar))
            hits = np.bincount(stage.dates[selected], weights=stage.events[selected, event], minlength=len(calendar))
            returns = np.bincount(stage.dates[selected], weights=stage.y[selected], minlength=len(calendar))
            # 事后集中度诊断仅解释原记录，不改变条件、主分母或证据分层。
            largest = int(np.argmax(counts))
            remaining = selected & (stage.dates != largest)
            observed = data["x"][stage.ids[selected], item["column"]].astype(np.float64)
            finite = observed[np.isfinite(observed)]
            sensitivity.append({**common, "n": int(selected.sum()), "largest_target_date": calendar[largest],
                                "largest_date_n": int(counts[largest]), "largest_date_hits": int(hits[largest]),
                                "exclude_largest_date_n": int(remaining.sum()),
                                "exclude_largest_date_hits": int(stage.events[remaining, event].sum()),
                                "exclude_largest_date_rate": float(stage.events[remaining, event].mean()) if remaining.any() else None,
                                "observed_feature_min": float(finite.min()) if len(finite) else None,
                                "observed_feature_max": float(finite.max()) if len(finite) else None,
                                "observed_feature_unique_values": int(np.unique(finite).size)})
            for day in np.unique(stage.dates):
                n = int(counts[day])
                daily.append({**common, "target_date": calendar[day], "n": n, "hits": int(hits[day]),
                              "hit_rate": float(hits[day] / n) if n else None,
                              "mean_return": float(returns[day] / n) if n else None,
                              "same_date_base": float(stage.daily_rates[day, event])})
    write_csv(output / "representative_periods.csv", temporal)
    write_csv(output / "representative_sensitivity.csv", sensitivity)
    with gzip.open(output / "representative_days.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        if daily:
            writer = csv.DictWriter(stream, fieldnames=list(daily[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(daily)
    write_json(output / "detail_summary.json", dict(representatives=len(representatives), temporal_rows=len(temporal), daily_rows=len(daily),
               sensitivity_rows=len(sensitivity), pooled_over_65=len(pooled), pooled_features=len({row["feature"] for row in pooled}),
               selection_changed=False, holdout_target_values_read=False,
               script_sha256=digest(Path(__file__))))
    print(f"代表条件{len(representatives)}条；年度/季度记录{len(temporal)}行；逐日记录{len(daily)}行", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
