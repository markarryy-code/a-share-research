"""实验到样本的适配器；只使用samples显式公开入口。"""

from ..samples import PreparedReader, read_prepared_index, DataError
from .domain import SamplePopulation, SplitError


class PreparedIndexSource:
    def __init__(self, reader: PreparedReader, reference: str, expected_versions: tuple[str, str]):
        self.reader, self.reference = reader, reference
        self.expected_versions = expected_versions

    def read_population(self):
        try:
            index = read_prepared_index(self.reader, self.reference)
        except DataError as error:
            raise SplitError(str(error)) from error
        meta = index.stored.metadata
        actual_versions = (meta["identity"]["features"]["feature_set"], meta["identity"]["target"]["target_id"])
        if actual_versions != self.expected_versions:
            raise SplitError("完成样本的特征或目标版本与实验配置不一致")
        return SamplePopulation(index.parent_run, index.stored.preparation_id, index.stored.manifest_sha256,
                                meta["source_fingerprint"], meta["identity"]["features"]["feature_set"],
                                meta["identity"]["target"]["target_id"], index.calendar,
                                tuple(s["stock_id"] for s in index.stocks), index.period, index.limitations, index.rows)

    def verify_unchanged(self):
        receipt = self.reader.verify_unchanged()
        if receipt.changed:
            raise SplitError("P2执行期间完成样本发生变化：" + ", ".join(receipt.changed))
        return receipt.records
