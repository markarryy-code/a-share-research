"""预测样本公开类型和用例；不导出具体文件实现。"""

from .domain import StockHistory, FeatureBlock, TargetBlock, PreparationDescription, StoredSamples, PreparedIndex, DataError, align_history, market_closes
from .features import build_market_features, build_stock_features, feature_spec
from .ports import PreparedReader, PreparedMatrices
from .application import prepare_samples, read_prepared_index, open_prepared_matrices

# 冻结模型评价新增日期时复用同一套纯计算，不在各研究方法内复制特征公式。
__all__ = ("StockHistory", "FeatureBlock", "TargetBlock", "PreparationDescription", "StoredSamples", "PreparedIndex", "PreparedReader", "PreparedMatrices", "DataError", "prepare_samples", "read_prepared_index", "open_prepared_matrices", "align_history", "market_closes", "build_market_features", "build_stock_features", "feature_spec")
