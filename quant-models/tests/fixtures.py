"""P0/P1共用的合成资料；价格和收益由常数构造，不是同名证券的真实行情。"""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from decimal import Decimal as D


SRC = Path(__file__).resolve().parents[1] / "src"
CALENDAR = ["2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


class DataFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.module, self.data = root / "module", root / "data"
        self.month = self.data / "month"
        self.module.mkdir()
        write_json(self.module / "configs/data_sources.json", {"paths_relative_to": "module_root", "history_directory": "../data", "holdout_directory": "../data/month"})
        write_json(self.data / "冻结配置.json", {"start": CALENDAR[0], "end": CALENDAR[-1], "month_start": CALENDAR[3], "count": 2,
                                                     "scope_complete": False, "survivorship_bias": True, "price_selection_bias": True})
        write_json(self.data / "交易日期.json", CALENDAR)
        write_json(self.month / "交易日期.json", CALENDAR[3:])
        (self.data / "字段说明.md").write_text("价格元；量为手与股；涨跌百分数。", encoding="utf-8")
        self.stocks = [{"code": "000001", "exchange": "SZ", "list_date": "2000-01-01"}, {"code": "600001", "exchange": "SH", "list_date": CALENDAR[1]}]
        write_csv(self.data / "股票名单.csv", self.stocks)
        write_csv(self.month / "股票名单.csv", self.stocks)
        total = month_total = 0
        for stock in self.stocks:
            symbol = stock["exchange"] + stock["code"]
            raw, labels, states, web, reference = [], [], [], [], []
            previous = None
            for index, current in enumerate(CALENDAR):
                if current < stock["list_date"]:
                    continue
                paused = symbol == "SZ000001" and index == 2
                states.append(dict(date=current, code=stock["code"], exchange=stock["exchange"], trade_status="0" if paused else "1", evidence_type="fixture", evidence_ref="fixture.csv"))
                close = D(("10", "10.2", "10.2", "9.9", "9.9", "10")[index]) if symbol == "SZ000001" else D("20")
                ref = D("10") if index in (0, 1, 3) else close
                percent = (close / ref - 1) * 100 if not paused else D(0)
                web.append(dict(SECURITY_CODE=stock["code"], SECUCODE=stock["code"] + "." + stock["exchange"], TRADE_DATE=current + " 00:00:00", CLOSE_PRICE=str(close), CHANGE_RATE=str(percent)))
                if paused:
                    continue
                raw.append(dict(date=current, code=stock["code"], exchange=stock["exchange"], open=str(close), high=str(close + 1), low=str(close - 1), close=str(close), volume_lots="1", volume_shares="100", amount_yuan=str(close * 100), turnover_pct="0",
                                change_pct="-2.94" if symbol == "SZ000001" and index == 3 else str(percent)))
                labels.append(dict(date=current, code=stock["code"], exchange=stock["exchange"], provider_change_pct=str(percent), inferred_reference_close_yuan=str(ref), previous_actual_close_yuan="" if previous is None else str(previous)))
                reference.append(dict(date=current, code="sz.000001", close=str(close), preclose=str(ref), tradestatus="1"))
                previous = close
            for folder, rows in (("日线", raw), ("日线派生涨跌", labels), ("交易状态/逐日交易状态", states), ("网页股本估值", web)):
                write_csv(self.data / folder / f"{symbol}.csv", rows)
                date_field = "TRADE_DATE" if folder == "网页股本估值" else "date"
                write_csv(self.month / folder / f"{symbol}.csv", [row for row in rows if row[date_field][:10] >= CALENDAR[3]])
            if symbol == "SZ000001":
                write_csv(self.data / "状态估值" / f"{symbol}.csv", reference)
            total += len(raw)
            month_total += sum(row["date"] >= CALENDAR[3] for row in raw)
        for symbol in ("sh000001", "sz399001"):
            rows = [dict(date=current, symbol=symbol, open="100", high="101", low="99", close="100") for current in CALENDAR]
            write_csv(self.data / "市场行情" / f"{symbol}.csv", rows)
            write_csv(self.month / "市场行情" / f"{symbol}.csv", rows[3:])
        write_json(self.data / "日线验收结果.json", {"daily_rows": total})
        write_json(self.data / "日线派生涨跌验收.json", {"monthly_rows": month_total, "assumed_max_percentage_error_points": "0.000001", "assumed_reference_price_unit_yuan": "0.01"})
        write_csv(self.data / "网页与既有涨跌幅差异.csv", [dict(code="000001", exchange="SZ", date=CALENDAR[3], existing_pct="-2.94", web_pct="-1", known_ex_dividend_date="True")])
        self.update_provenance()

    def update_provenance(self):
        rows = []
        for stock in self.stocks:
            symbol = stock["exchange"] + stock["code"]
            rows.append({"code": stock["code"], "exchange": stock["exchange"],
                         "daily_input_sha256": hashlib.sha256((self.data / "日线" / f"{symbol}.csv").read_bytes()).hexdigest(),
                         "web_input_sha256": hashlib.sha256((self.data / "网页股本估值" / f"{symbol}.csv").read_bytes()).hexdigest()})
        write_csv(self.data / "日线派生字段覆盖.csv", rows)

    def run_command(self):
        env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        process = subprocess.run([sys.executable, "-B", "-m", "daily_return", "inspect-data", "--module-root", str(self.module)], env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        result = json.loads(process.stdout.strip().splitlines()[-1])
        if "run_directory" not in result:
            return process, result, None
        run = Path(result["run_directory"])
        return process, json.loads((run / "dataset.json").read_text(encoding="utf-8")), run
