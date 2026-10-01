"""资料到样本的适配器；跨模块仅使用market_data公开能力。"""

from types import MappingProxyType
from ..market_data import MarketSource, AdmissionArchive, DataError as MarketDataError, resolve_admission, open_admitted_read
from .domain import DataError, PreparationDescription, ReadReceipt, align_history, market_closes


class AdmittedHistorySource:
    def __init__(self, source: MarketSource, archive: AdmissionArchive, selection):
        self._source, self._archive, self._selection = source, archive, selection
        self._reader = None
        self._description = None

    def describe(self):
        try:
            snapshot = resolve_admission(self._source, self._archive, self._selection)
            self._reader = open_admitted_read(snapshot, self._source)
        except MarketDataError as error:
            raise DataError(str(error)) from error
        self._description = PreparationDescription(
            snapshot.core.calendar, snapshot.core.stocks, snapshot.core.frozen, snapshot.audit_counts,
            MappingProxyType({key: snapshot.expected_hashes[key] for key, _ in snapshot.inputs}), snapshot.acceptance)
        return self._description

    def read_markets(self):
        try:
            facts = {symbol: self._reader.read_market_facts(symbol).indexed("date", {"symbol": symbol})
                     for symbol in ("sh000001", "sz399001")}
        except MarketDataError as error:
            raise DataError(str(error)) from error
        return market_closes(self._description.calendar, facts)

    def iter_stocks(self):
        try:
            for stock, evidence in self._reader.iter_stock_facts():
                identity = {"code": stock["code"], "exchange": stock["exchange"]}
                tables = {name: table.indexed("date", identity) for name, table in evidence.tables.items()}
                yield align_history(stock, self._description.calendar, tables)
        except MarketDataError as error:
            raise DataError(str(error)) from error

    def verify_current(self):
        try:
            self._reader.verify_current()
        except MarketDataError as error:
            raise DataError(str(error)) from error

    def verify_unchanged(self):
        receipt = self._source.verify_unchanged()
        return ReadReceipt(receipt.records, receipt.changed, receipt.fingerprint)
