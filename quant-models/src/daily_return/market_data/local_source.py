"""本地来源适配器：供应商文件格式、路径与实际读取账本。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from .domain import DataError, Table, CoreData, InspectionInputs, StockEvidence, MarketEvidence, ReadReceipt, immutable, stock_list

SOURCE_TABLES = {
    "daily": ("日线", ("date", "code", "exchange", "open", "high", "low", "close", "volume_lots", "volume_shares", "amount_yuan", "turnover_pct")),
    "label": ("日线派生涨跌", ("date", "code", "exchange", "provider_change_pct", "inferred_reference_close_yuan", "previous_actual_close_yuan")),
    "state": ("交易状态/逐日交易状态", ("date", "code", "exchange", "trade_status", "evidence_type", "evidence_ref")),
    "web": ("网页股本估值", ("SECURITY_CODE", "SECUCODE", "TRADE_DATE", "CLOSE_PRICE", "CHANGE_RATE")),
}
STOCK_FIELDS = ("code", "exchange", "list_date")
MARKET_FIELDS = ("date", "symbol", "open", "high", "low", "close")



REFERENCE_FIELDS = ("date", "code", "close", "preclose", "tradestatus")

class FileLedger:
    """每次读取冻结实际字节，结束时复核同一批文件；不保存原件副本。"""

    def __init__(self, module_root: Path):
        self.module_root = module_root
        self.entries: dict[Path, dict] = {}
        self.expected_hashes: dict[Path, str] | None = None

    def relative(self, path: Path) -> str:
        return os.path.relpath(path, self.module_root).replace(os.sep, "/")

    def read(self, path: Path, role: str) -> bytes:
        path = path.resolve()
        try:
            data = path.read_bytes()
        except OSError as error:
            raise DataError(f"无法读取来源：{path}；{error}") from error
        digest = hashlib.sha256(data).hexdigest()
        prior = self.entries.get(path)
        if prior and prior["sha256"] != digest:
            raise DataError(f"再次读取时来源已变化：{path}")
        self.entries[path] = {
            "path": self.relative(path), "role": role, "bytes": len(data),
            "sha256": digest, "end_sha256": "", "unchanged": "",
        }
        if self.expected_hashes is not None and self.expected_hashes.get(path) != digest:
            raise DataError(f"文件未获准入或内容已变化：{path}；实际SHA256={digest}")
        return data

    def json(self, path: Path, role: str) -> dict | list:
        return json.loads(self.read(path, role).decode("utf-8-sig"))

    def csv(self, path: Path, role: str, required: tuple[str, ...]) -> Table:
        reader = csv.reader(io.StringIO(self.read(path, role).decode("utf-8-sig"), newline=""))
        fields = next(reader, [])
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise DataError(f"字段重复或缺少必要列：{path}；要求{required}")
        rows = []
        for number, values in enumerate(reader, 2):
            if len(values) != len(fields):
                raise DataError(f"CSV列数不符：{path}:{number}")
            rows.append(dict(zip(fields, values)))
        return Table(tuple(fields), immutable(rows))

    def verify_unchanged(self) -> list[str]:
        changed = []
        for path, entry in self.entries.items():
            try:
                with path.open("rb") as stream:
                    current = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError:
                current = "unreadable"
            entry["end_sha256"] = current
            entry["unchanged"] = current == entry["sha256"]
            if not entry["unchanged"]:
                changed.append(entry["path"])
        return changed

    def fingerprint(self) -> str:
        pairs = sorted((entry["path"], entry["sha256"]) for entry in self.entries.values())
        return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()


    def receipt(self):
        changed = tuple(entry["path"] for entry in self.entries.values() if entry["unchanged"] is False)
        records = immutable(sorted(self.entries.values(), key=lambda row: row["path"]))
        return ReadReceipt(records, changed, self.fingerprint())


class LocalMarketSource:
    def __init__(self, module_root: Path, config: dict, ledger: FileLedger):
        self.root, self.ledger = module_root, ledger
        self.history = (module_root / config["history_directory"]).resolve()
        self.holdout = (module_root / config["holdout_directory"]).resolve()

    def read_core(self):
        frozen = immutable(self.ledger.json(self.history / "冻结配置.json", "source_metadata"))
        calendar = tuple(self.ledger.json(self.history / "交易日期.json", "calendar"))
        table = self.ledger.csv(self.history / "股票名单.csv", "stock_master", STOCK_FIELDS)
        stocks = tuple(sorted(stock_list(table), key=lambda s: s["exchange"] + s["code"]))
        return CoreData(frozen, calendar, stocks)

    def inspection_inputs(self):
        core = self.read_core()
        master = self.ledger.csv(self.history / "股票名单.csv", "stock_master", STOCK_FIELDS)
        # P0核验示例沿用源名单顺序；P1的read_core仍按证券身份稳定排序。
        core = CoreData(core.frozen, core.calendar, tuple(stock_list(master)))
        month_master = self.ledger.csv(self.holdout / "股票名单.csv", "holdout_stock_master", STOCK_FIELDS)
        month_calendar = tuple(self.ledger.json(self.holdout / "交易日期.json", "holdout_calendar"))
        daily = immutable(self.ledger.json(self.history / "日线验收结果.json", "source_metadata"))
        derived = immutable(self.ledger.json(self.history / "日线派生涨跌验收.json", "source_metadata"))
        self.ledger.read(self.history / "字段说明.md", "field_definitions")
        provenance = self.ledger.csv(self.history / "日线派生字段覆盖.csv", "source_provenance", ("code", "exchange", "daily_input_sha256", "web_input_sha256"))
        clues = self.ledger.csv(self.history / "网页与既有涨跌幅差异.csv", "label_difference_evidence", ("code", "exchange", "date", "existing_pct", "web_pct", "known_ex_dividend_date"))
        inventories = tuple((self.ledger.relative(root / folder), tuple(p.stem for p in (root / folder).glob("*.csv")))
                            for root in (self.history, self.holdout) for folder, _ in SOURCE_TABLES.values())
        return InspectionInputs(core, month_calendar, master, month_master, daily, derived, provenance, clues, inventories,
                                self.ledger.relative(self.history), self.ledger.relative(self.holdout))

    def read_market(self, symbol, monthly=False):
        path = self.history / "市场行情" / (symbol + ".csv")
        copy_path = self.holdout / "市场行情" / path.name
        table = self.ledger.csv(path, "market", MARKET_FIELDS)
        copy = self.ledger.csv(copy_path, "holdout_market", MARKET_FIELDS) if monthly and copy_path.exists() else None
        return MarketEvidence(symbol, table, copy, self.ledger.relative(path), self.ledger.relative(copy_path))

    def read_stock(self, stock, monthly=False):
        symbol = stock["exchange"] + stock["code"]
        tables, copies, hashes = {}, {}, {}
        groups = SOURCE_TABLES if monthly else ("daily", "label", "state")
        for group in groups:
            folder, fields = SOURCE_TABLES[group]
            path = self.history / folder / (symbol + ".csv")
            tables[group] = self.ledger.csv(path, "history_" + group, fields)
            hashes[group] = self.ledger.entries[path.resolve()]["sha256"]
            if monthly:
                copies[group] = self.ledger.csv(self.holdout / folder / path.name, "holdout_" + group, fields)
        reference = self.history / "状态估值" / (symbol + ".csv")
        if monthly and reference.exists():
            tables["reference"] = self.ledger.csv(reference, "independent_reference_evidence", REFERENCE_FIELDS)
        return StockEvidence(immutable(tables), immutable(copies), immutable(hashes))

    def enforce(self, hashes):
        self.ledger.expected_hashes = {(self.root / key).resolve(): value for key, value in hashes.items()}

    def input_refs(self, core):
        paths = [(self.history / name, role) for name, role in (("冻结配置.json", "source_metadata"), ("交易日期.json", "calendar"), ("股票名单.csv", "stock_master"))]
        for stock in core.stocks:
            symbol = stock["exchange"] + stock["code"]
            paths.extend((self.history / SOURCE_TABLES[group][0] / (symbol + ".csv"), "history_" + group) for group in ("daily", "label", "state"))
        paths.extend((self.history / "市场行情" / (symbol + ".csv"), "market") for symbol in ("sh000001", "sz399001"))
        return tuple((self.ledger.relative(path), role) for path, role in paths)

    def verify_current(self, snapshot):
        for key, role in snapshot.inputs:
            self.ledger.read(self.root / key, role)

    def verify_unchanged(self):
        self.ledger.verify_unchanged()
        return self.ledger.receipt()
