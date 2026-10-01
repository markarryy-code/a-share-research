"""样本准备用例：领域计算与I/O会话协作，流程止于prepared。"""

from dataclasses import dataclass
from .domain import DataError, PreparationSummary, build_chunk, planned_rows, ROW_DTYPE, NUMERIC_TOLERANCE, validate_prepared_index
from .features import FEATURE_SET, FEATURE_NAMES, feature_spec, build_market_features, build_stock_features
from .targets import TARGET_SPEC, build_next_day_targets
from .ports import HistorySource, PreparedStore, PreparedReader
from ..execution import RunContext


# 文件级保守闭包：格式转换、准入、对齐、计算与存储语义均进入身份。
CALCULATION_MODULES = (
    "serialization.py", "market_data/__init__.py", "market_data/domain.py", "market_data/ports.py",
    "market_data/application.py", "market_data/local_source.py", "market_data/archive.py",
    "samples/__init__.py", "samples/domain.py", "samples/ports.py", "samples/application.py",
    "samples/features.py", "samples/targets.py", "samples/history_reader.py", "samples/numpy_store.py",
)


def preparation_identity(description, context):
    return {
        "schema_version": 2, "input_sha256": dict(description.input_sha256), "features": feature_spec(), "target": TARGET_SPEC,
        "precision": {"X": "float32", "y": "float64", "rows": ROW_DTYPE.descr},
        "python": context.python_version, "numpy": context.numpy_version,
        "calculation_code_sha256": {name: context.code_sha256[name] for name in CALCULATION_MODULES},
    }


@dataclass(frozen=True)
class PreparationOutcome:
    result: dict
    status: str
    exit_code: int


def read_prepared_index(reader: PreparedReader, reference: str):
    index = reader.read_index(reference)
    validate_prepared_index(index)
    return index


def open_prepared_matrices(reader: PreparedReader, reference: str):
    """模型侧沿用同一完成产物边界，P2的索引读取仍不加载X/y。"""
    return reader.open_matrices(read_prepared_index(reader, reference))


def prepare_samples(source: HistorySource, store: PreparedStore, context: RunContext):
    manifest = {**feature_spec(), "target": TARGET_SPEC, "numerical_test_tolerance": NUMERIC_TOLERANCE}
    result = {"stage": "P1", "state": "failed", "splits_prepared": False, "model_trained": False}
    issues, writer, stored, receipt = [], None, None, None
    status, exit_code = "running", 0
    store.initialize(manifest)
    try:
        description = source.describe()
        identity = preparation_identity(description, context)
        stored = store.find_complete(identity)
        reused = stored is not None
        if reused:
            # 复用不执行逐股读取，因此显式核对当前来源；新构建由实际读取和结束复核覆盖。
            source.verify_current()
            receipt = source.verify_unchanged()
            if receipt.changed:
                raise DataError("缓存复用期间来源发生变化")
        else:
            planned_rows(description)
            markets = build_market_features(source.read_markets())
            writer = store.begin(identity, description, manifest)
            summary = PreparationSummary(FEATURE_NAMES)
            for index, history in enumerate(source.iter_stocks()):
                block = build_stock_features(history, markets)
                target = build_next_day_targets(history)
                chunk = build_chunk(index, history, block, target, FEATURE_NAMES)
                summary.include(chunk)
                writer.append(chunk)
                if (index + 1) % 100 == 0 or index + 1 == len(description.stocks):
                    context.progress({"stage": "prepare", "stocks": index + 1, "stocks_total": len(description.stocks),
                                      "grid_rows": summary.totals["grid_rows"]})
            summary.complete(description)
            context.progress({"stage": "verify_prepare_inputs"})
            receipt = source.verify_unchanged()
            if receipt.changed:
                raise DataError("准备期间来源发生变化，缓存不发布")
            stored = writer.publish(summary, receipt)
        metadata = stored.metadata
        result.update(
            state="prepared", preparation_id=stored.preparation_id, cache_directory=stored.cache_key, cache_reused=reused,
            cache_manifest_sha256=stored.manifest_sha256, counts=metadata["counts"], stock_count=metadata["stock_count"],
            market_days=metadata["market_days"], shape=metadata["shape"],
            arrays={name: metadata["files"][name] for name in ("X.npy", "y.npy", "rows.npy")},
            source_acceptance=dict(description.acceptance), source_fingerprint=metadata["source_fingerprint"],
            source_file_count=len(receipt.records), source_manifest="source_files.csv", inputs_unchanged=True,
            row_dtype=ROW_DTYPE.descr, target=TARGET_SPEC,
            range={key: description.frozen[key] for key in ("start", "end", "month_start")},
            survivorship_bias=description.frozen.get("survivorship_bias"), price_selection_bias=description.frozen.get("price_selection_bias"))
        status = "completed"
    except KeyboardInterrupt:
        result["state"], status, exit_code = "interrupted", "interrupted", 130
        issues.append({"code": "interrupted", "detail": "已中断，未完成缓存保留为partial"})
    except Exception as error:
        status = "failed"
        exit_code = 2 if isinstance(error, DataError) else 1
        issues.append({"code": type(error).__name__, "detail": str(error)})
    finally:
        if writer is not None:
            writer.close()
        if receipt is None or result["state"] != "prepared":
            receipt = source.verify_unchanged()
            result["inputs_unchanged"] = not receipt.changed
            issues.extend({"code": "source_changed", "detail": key} for key in receipt.changed)
        store.finish(result, receipt, issues, stored if result["state"] == "prepared" else None)
    return PreparationOutcome(result, status, exit_code)
