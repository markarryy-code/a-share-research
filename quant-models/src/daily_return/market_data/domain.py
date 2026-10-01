"""行情资料领域：值数据、准入与核验规则；不读取文件。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Mapping


def immutable(value):
    """冻结小型证据和描述；不携带读取器、Path或打开的文件。"""
    if isinstance(value, Mapping):
        return MappingProxyType({key: immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(item) for item in value)
    return value


class DataError(ValueError):
    """可定位的资料契约错误，不修改原始数据进行修复。"""


def day(value: str) -> str:
    """兼容本批日期列和网页的日期加零点格式，不推断模糊日期。"""
    text = value.removesuffix(" 00:00:00")
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise DataError(f"日期格式不是YYYY-MM-DD：{value}")
    return text


@dataclass(frozen=True)
class Table:
    fields: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]

    def indexed(self, date_field: str, identity: dict[str, str]) -> dict[str, dict[str, str]]:
        result = {}
        previous = ""
        for row in self.rows:
            if any(row.get(key) != value for key, value in identity.items()):
                raise DataError(f"证券身份不符：预期{identity}，实际日期{row.get(date_field)}")
            current = day(row[date_field])
            if current in result:
                raise DataError(f"重复证券日期：{current}")
            if current < previous:
                raise DataError(f"日期未按升序保存：{current}")
            result[current] = row
            previous = current
        return result



def stock_list(table: Table) -> tuple[Mapping[str, str], ...]:
    seen = set()
    for stock in table.rows:
        symbol = stock["exchange"] + stock["code"]
        if stock["exchange"] not in ("SH", "SZ") or len(stock["code"]) != 6 or not stock["code"].isascii() or not stock["code"].isdigit():
            raise DataError(f"证券代码或交易所无效：{symbol}")
        if symbol in seen:
            raise DataError(f"股票名单重复：{symbol}")
        day(stock["list_date"])
        seen.add(symbol)
    if not seen:
        raise DataError("股票名单为空")
    return table.rows


def require_same_dates(actual, expected, title: str):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise DataError(f"{title}日期不符：缺{len(missing)}，多{len(extra)}；缺样例{sorted(missing)[:3]}，多样例{sorted(extra)[:3]}")



@dataclass(frozen=True)
class DataIssue:
    code: str
    detail: str
    stock_id: str = ""
    date: str = ""
    source: str = ""
    severity: str = "error"


class IssueCollector:
    """只汇总问题值；应用按股取走问题，归档实现决定如何持久化。"""
    def __init__(self):
        self.pending = []
        self.counts = Counter()
        self.errors = 0

    def add(self, code, detail, stock_id="", date="", source="", severity="error"):
        self.pending.append(DataIssue(code, detail, stock_id, date, source, severity))
        self.counts[(severity, code)] += 1
        self.errors += severity == "error"

    def drain(self):
        result, self.pending = tuple(self.pending), []
        return result

    def summary(self):
        return [dict(severity=severity, code=code, count=count) for (severity, code), count in sorted(self.counts.items())]


@dataclass(frozen=True)
class ReadReceipt:
    records: tuple[Mapping, ...]
    changed: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class CoreData:
    frozen: Mapping
    calendar: tuple[str, ...]
    stocks: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class InspectionInputs:
    core: CoreData
    monthly_calendar: tuple[str, ...]
    master: Table
    monthly_master: Table
    daily_audit: Mapping
    derived_audit: Mapping
    provenance: Table
    clues: Table
    inventories: tuple[tuple[str, tuple[str, ...]], ...]
    history_key: str
    holdout_key: str


@dataclass(frozen=True)
class StockEvidence:
    tables: Mapping[str, Table]
    monthly: Mapping[str, Table]
    actual_hashes: Mapping[str, str]


@dataclass(frozen=True)
class MarketEvidence:
    symbol: str
    history: Table
    monthly: Table | None
    history_key: str
    monthly_key: str


@dataclass(frozen=True)
class AdmissionEvidence:
    primary: Mapping
    execution: Mapping
    run_id: str
    primary_key: str
    directories_match: bool
    manifest: tuple[Mapping, ...]
    supplements: tuple[Mapping, ...]
    prior_reads: Mapping[str, str]


@dataclass(frozen=True)
class AdmissionSnapshot:
    core: CoreData
    expected_hashes: Mapping[str, str]
    inputs: tuple[tuple[str, str], ...]
    audit_counts: Mapping[str, int]
    acceptance: Mapping


def number(row: dict, field: str) -> Decimal:
    try:
        value = Decimal(row[field])
    except (KeyError, InvalidOperation) as error:
        raise DataError(f"{field}不是有效数值：{row.get(field)!r}") from error
    if not value.is_finite():
        raise DataError(f"{field}不是有限数值：{row[field]}")
    return value


def daily_values(row: dict) -> dict[str, Decimal]:
    values = {key: number(row, key) for key in ("open", "high", "low", "close", "volume_lots", "volume_shares", "amount_yuan", "turnover_pct")}
    if min(values[key] for key in ("open", "high", "low", "close")) <= 0:
        raise DataError("实际成交OHLC必须为正")
    if not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
        raise DataError("OHLC高低关系矛盾")
    if min(values[key] for key in ("volume_lots", "volume_shares", "amount_yuan")) <= 0 or values["turnover_pct"] < 0:
        raise DataError("成交量额必须为正，换手率不能为负")
    if values["volume_shares"] != values["volume_lots"] * 100:
        raise DataError("手与股的换算不一致")
    return values


def review_stock(stock: dict, tables: dict[str, Table], calendar: list[str], month_start: str,
                 issues: IssueCollector, tolerance: Decimal, clues: dict, samples: list, sample_keys: set) -> dict:
    symbol = stock["exchange"] + stock["code"]
    identity = {"code": stock["code"], "exchange": stock["exchange"]}
    raw = tables["daily"].indexed("date", identity)
    labels = tables["label"].indexed("date", identity)
    states = tables["state"].indexed("date", identity)
    web = tables["web"].indexed("TRADE_DATE", {"SECURITY_CODE": stock["code"], "SECUCODE": stock["code"] + "." + stock["exchange"]})
    reference = tables["reference"].indexed("date", {"code": stock["exchange"].lower() + "." + stock["code"]}) if "reference" in tables else {}
    expected = [current for current in calendar if current >= stock["list_date"]]
    require_same_dates(states, expected, "上市后状态")
    require_same_dates(web, expected, "网页原值")
    require_same_dates(labels, raw, "日线与标签")
    traded = set()
    for current, state in states.items():
        if state["trade_status"] not in ("0", "1") or not state["evidence_type"] or not state["evidence_ref"]:
            raise DataError(f"交易状态或依据不完整：{current}")
        if state["trade_status"] == "1":
            traded.add(current)
    require_same_dates(raw, traded, "成交状态与实际日线")
    counts = Counter(daily_rows=len(raw), state_rows=len(states), suspended_rows=len(states) - len(raw),
                     web_rows=len(web), monthly_daily_rows=sum(current >= month_start for current in raw))
    valid = set()
    previous = None
    prior_day = {current: calendar[index - 1] if index else None for index, current in enumerate(calendar)}
    for current, row in raw.items():
        try:
            prices = daily_values(row)
            label, original = labels[current], web[current]
            percent = number(label, "provider_change_pct")
            if percent <= -100:
                raise DataError("正参考价口径的涨跌幅不能小于等于-100%")
            if number(original, "CLOSE_PRICE") != prices["close"] or number(original, "CHANGE_RATE") != percent:
                raise DataError("独立标签与原始网页的收盘或涨跌幅不一致")
            inferred = number(label, "inferred_reference_close_yuan")
            if inferred <= 0 or inferred != inferred.quantize(Decimal("0.01")):
                raise DataError("反推参考价不是正的分位价格")
            exact = (prices["close"] / inferred - 1) * 100
            if abs(exact - percent) > tolerance:
                raise DataError("参考价反推算术不符合来源声明的误差假设")
            for neighbor in (inferred - Decimal("0.01"), inferred + Decimal("0.01")):
                if neighbor > 0 and abs((prices["close"] / neighbor - 1) * 100 - percent) <= tolerance:
                    raise DataError("参考价在来源假设下并不唯一")
            if previous is not None and number(label, "previous_actual_close_yuan") != previous:
                raise DataError("上一实际成交收盘与本批日线不一致")
            reference_match = "not_available"
            observed = reference.get(current)
            if observed and observed["tradestatus"] == "1":
                if number(observed, "close") != prices["close"] or number(observed, "preclose") != inferred:
                    raise DataError("已有独立前收价子集与当前口径冲突")
                reference_match = "matched"
                counts["independent_reference_rows"] += 1
            clue = clues.get((symbol, current))
            if clue:
                if number(row, "change_pct") != number(clue, "existing_pct") or percent != number(clue, "web_pct"):
                    raise DataError("既有差异清单与实际来源值不一致")
                counts["legacy_difference_records_checked"] += 1
            categories = []
            if current == stock["list_date"]:
                categories.append("listing_day")
            if prior_day[current] in states and states[prior_day[current]]["trade_status"] == "0":
                categories.append("resumed_after_pause")
            if clue and clue["known_ex_dividend_date"] == "True":
                categories.append("known_ex_dividend_clue")
            if previous is not None and inferred != previous:
                categories.append("reference_differs_from_last_close")
            if not categories:
                categories.append("ordinary")
            for category in categories:
                counts["case_" + category] += 1
                # 每类最多五只不同股票，避免首只股票占满全部示例。
                sample_key = (category, symbol)
                if sample_key not in sample_keys and sum(key[0] == category for key in sample_keys) < 5:
                    samples.append({"case": category, "stock_id": symbol, "date": current,
                                    "provider_change_pct": str(percent), "label_fraction": str(percent / 100),
                                    "close": str(prices["close"]), "inferred_reference": str(inferred),
                                    "independent_reference": reference_match})
                    sample_keys.add(sample_key)
            valid.add(current)
        except DataError as error:
            issues.add("invalid_trading_row", str(error), symbol, current)
        # 后续行的连续性核验仍以源记录为准，不跳过坏记录伪造相邻日。
        try:
            previous = number(row, "close")
        except DataError:
            previous = None
    counts["valid_trading_rows"] = len(valid)
    next_day = dict(zip(calendar, calendar[1:]))
    for current in valid:
        target = next_day.get(current)
        if target is None:
            counts["calendar_tail_without_next_day"] += 1
        elif target in valid:
            counts["next_day_pair_candidates"] += 1
        elif target in states and states[target]["trade_status"] == "0":
            counts["next_day_suspended"] += 1
        else:
            counts["next_day_invalid_label"] += 1
    return {"stock_id": symbol, **dict(counts)}


def approve_evidence(evidence: AdmissionEvidence):
    primary = evidence.primary
    if primary.get("schema_version") != 1 or primary.get("stage") != "P0" or primary.get("admission_status") != "passed_with_limitations" or primary.get("inputs_unchanged") is not True:
        raise DataError("主验收不是完整通过的P0记录")
    if evidence.execution.get("execution_status") != "completed" or evidence.execution.get("run_id") != evidence.run_id:
        raise DataError("P0运行未完成或标识不一致")
    if any(item["severity"] == "error" for item in primary.get("issues", ())):
        raise DataError("P0存在未解决错误")
    if not evidence.directories_match:
        raise DataError("来源目录与指定P0验收不一致")
    expected = {}

    def merge(rows, allowed_new=None):
        seen = set()
        for row in rows:
            key = row["path"]
            if key in seen or row["unchanged"] != "True" or row["sha256"] != row["end_sha256"]:
                raise DataError("验收指纹清单不完整或重复")
            seen.add(key)
            if key in expected and expected[key] != row["sha256"]:
                raise DataError(f"验收记录之间存在指纹冲突：{key}")
            if allowed_new is not None and key not in expected and key not in allowed_new:
                raise DataError(f"补充验收超出已声明的市场文件范围：{key}")
            expected[key] = row["sha256"]

    merge(evidence.manifest)
    for supplement in evidence.supplements:
        record = supplement["record"]
        if record.get("previous_p0_run") != evidence.run_id or record.get("errors") != 0 or record.get("checked_inputs_unchanged") is not True or record.get("prior_p0_market_source_hashes_match") is not True:
            raise DataError("补充验收与主P0不匹配或没有通过")
        if any(record.get("markets", {}).get(symbol, {}).get("monthly_copy") != "verified" for symbol in ("sh000001", "sz399001")):
            raise DataError("市场月度副本补充验收不完整")
        merge(supplement["manifest"], supplement["allowed_new"])
    for key, digest in evidence.prior_reads.items():
        if key in expected and expected[key] != digest:
            raise DataError(f"已经读取的证据与准入指纹不符：{key}")
    return immutable(expected)


def admitted_snapshot(evidence, core, expected, inputs):
    calendar, frozen, primary = core.calendar, core.frozen, evidence.primary
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise DataError("交易日历为空、重复或无序")
    for current in calendar:
        if not frozen["start"] <= day(current) <= frozen["end"]:
            raise DataError("交易日期超出冻结范围")
    if any(frozen[key] != primary["range"][key] for key in ("start", "end", "month_start")):
        raise DataError("来源区间与P0区间不一致")
    if len(core.stocks) != primary["stock_count"] or len(core.stocks) != frozen["count"]:
        raise DataError("股票池数量与验收不一致")
    if any(key not in expected for key, _ in inputs):
        raise DataError("存在没有准入证据的必要输入文件")
    return AdmissionSnapshot(core, expected, inputs, immutable(primary["counts"]),
        immutable({"inspection_run": evidence.primary_key, "supplements": tuple(s["key"] for s in evidence.supplements),
                   "source_fingerprint": primary["source_fingerprint"]}))


def inspection_header(inputs, issues):
    frozen, calendar = inputs.core.frozen, inputs.core.calendar
    start, end, month_start = (day(frozen[key]) for key in ("start", "end", "month_start"))
    if not start <= month_start <= end:
        raise DataError("来源起止和月度边界矛盾")
    if not calendar or list(calendar) != sorted(set(calendar)) or any(not start <= day(current) <= end for current in calendar):
        raise DataError("交易日历为空、重复、无序或超出冻结区间")
    if inputs.monthly_calendar != tuple(current for current in calendar if current >= month_start):
        raise DataError("月度交易日历与历史日历切片不同")
    if inputs.master != inputs.monthly_master or len(inputs.core.stocks) != frozen["count"]:
        raise DataError("冻结股票数量或月度名单不一致")
    provenance = {}
    for row in inputs.provenance.rows:
        symbol = row["exchange"] + row["code"]
        if symbol in provenance:
            raise DataError(f"来源指纹表证券重复：{symbol}")
        provenance[symbol] = row
    if set(provenance) != {s["exchange"] + s["code"] for s in inputs.core.stocks}:
        raise DataError("标签来源指纹表与股票名单不同")
    tolerance = Decimal(inputs.derived_audit["assumed_max_percentage_error_points"])
    if not tolerance.is_finite() or tolerance <= 0 or Decimal(inputs.derived_audit["assumed_reference_price_unit_yuan"]) != Decimal("0.01"):
        raise DataError("反推参考价误差或单位假设不符合当前核验契约")
    clues = {(row["exchange"] + row["code"], day(row["date"])): row for row in inputs.clues.rows}
    if len(clues) != len(inputs.clues.rows):
        raise DataError("历史涨跌差异清单包含重复键")
    expected_names = {s["exchange"] + s["code"] for s in inputs.core.stocks}
    for key, inventory in inputs.inventories:
        actual = set(inventory)
        if actual != expected_names:
            issues.add("stock_file_inventory", f"逐股文件集合不同：缺{len(expected_names-actual)}，多{len(actual-expected_names)}", source=key)
    return tolerance, clues, provenance


def review_market(evidence, calendar, month_start, issues):
    symbol, table, copy = evidence.symbol, evidence.history, evidence.monthly
    observations = table.indexed("date", {"symbol": symbol})
    require_same_dates(observations, calendar, symbol)
    for current, row in observations.items():
        values = {key: number(row, key) for key in ("open", "close", "high", "low")}
        if min(values.values()) <= 0 or not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
            issues.add("invalid_market_price", "市场基准OHLC矛盾", symbol, current, evidence.history_key)
    sliced = tuple(row for row in table.rows if row["date"] >= month_start)
    if copy is None:
        issues.add("market_month_copy_missing", "没有独立市场行情月度副本；实际消费完整母表并按目标日隔离", symbol, source=evidence.monthly_key, severity="warning")
        status = "not_present"
    else:
        matched = copy.fields == table.fields and copy.rows == sliced
        if not matched:
            issues.add("market_month_mismatch", "市场基准月度副本与母表切片不一致", symbol, source=evidence.monthly_key)
        status = "verified" if matched else "mismatch"
    return {"rows": len(table.rows), "monthly_rows_in_history": len(sliced), "monthly_copy": status}


def review_stock_evidence(stock, evidence, calendar, month_start, issues, tolerance, clues, provenance, samples, sample_keys):
    symbol = stock["exchange"] + stock["code"]
    for group, monthly in evidence.monthly.items():
        table = evidence.tables[group]
        field = "TRADE_DATE" if group == "web" else "date"
        sliced = tuple(row for row in table.rows if day(row[field]) >= month_start)
        if monthly.fields != table.fields or monthly.rows != sliced:
            issues.add("monthly_slice_mismatch", f"{group}月度副本与历史表逐字段不一致", symbol)
        if group in ("daily", "web") and evidence.actual_hashes[group] != provenance[symbol][group + "_input_sha256"]:
            issues.add("provenance_hash_mismatch", f"{group}文件与派生标签时的输入指纹不同", symbol)
    return review_stock(stock, evidence.tables, calendar, month_start, issues, tolerance, clues, samples, sample_keys)
