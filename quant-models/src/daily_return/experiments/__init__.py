"""时间方案及跨方法评价的公开契约；具体文件实现仅由组装入口选择。"""

from .application import prepare_splits
from .evaluation_domain import (ValidationError, FoldInputSpec, FoldDescription, PredictionBatch,
                                check_matrix, validate_holdout_roles, fit_volatility_boundaries)
from .evaluation_ports import FoldSource, HoldoutSource, HoldoutUsage, RunTelemetry

__all__ = ("prepare_splits", "ValidationError", "FoldInputSpec", "FoldDescription", "PredictionBatch",
           "check_matrix", "validate_holdout_roles", "fit_volatility_boundaries",
           "FoldSource", "HoldoutSource", "HoldoutUsage", "RunTelemetry")
