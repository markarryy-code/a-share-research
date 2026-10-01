"""唯一组装入口：路径配置、具体文件实现、运行记录与命令结果。"""

import json
from pathlib import Path
from .execution import RunRecord
from .market_data.application import inspect_market_data
from .market_data.local_source import FileLedger, LocalMarketSource
from .market_data.archive import LocalAdmissionArchive


def source_binding(module_root: Path, data_config: Path | None, experiment_config: Path | None = None):
    ledger = FileLedger(module_root)
    config = ledger.json((module_root / (data_config or Path("configs/data_sources.json"))).resolve(), "data_config")
    if not isinstance(config, dict) or config.get("paths_relative_to") != "module_root":
        raise ValueError("数据配置必须声明相对module_root的路径")
    sources = [(module_root / config[key]).resolve() for key in ("history_directory", "holdout_directory")]
    outputs = [(module_root / name).resolve() for name in ("runs", ".cache")]
    if any(output.is_relative_to(source) or source.is_relative_to(output) for output in outputs for source in sources):
        raise ValueError("输出目录与研究来源重叠")
    baseline = ledger.json((module_root / experiment_config).resolve(), "experiment_config") if experiment_config else None
    return ledger, LocalMarketSource(module_root, config, ledger), baseline


def inspect_data(module_root: Path, data_config: Path | None = None):
    try:
        ledger, source, _ = source_binding(module_root, data_config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"exit_code": 2, "error": str(error), "output_created": False}, ensure_ascii=False), flush=True)
        return 2
    run = RunRecord(module_root, "P0")
    archive = LocalAdmissionArchive(module_root, source.history, source.holdout, ledger, run.directory)
    try:
        outcome = inspect_market_data(source, archive, run.context)
    except BaseException:
        run.finish("failed")
        raise
    run.finish(outcome.status)
    print(json.dumps({"run_directory": str(run.directory), "admission_status": outcome.result["admission_status"], "exit_code": outcome.exit_code}, ensure_ascii=False), flush=True)
    return outcome.exit_code


def prepare_dataset(module_root: Path, data_config: Path | None = None, experiment_config: Path | None = None):
    # P0组装不能提前导入数值模块，NumPy只在prepare被实际调用时加载。
    from .samples.application import prepare_samples, FEATURE_SET, TARGET_SPEC
    from .samples.history_reader import AdmittedHistorySource
    from .samples.numpy_store import NumpyPreparedStore
    try:
        ledger, source, baseline = source_binding(module_root, data_config, experiment_config or Path("configs/baseline.json"))
        if not isinstance(baseline, dict) or baseline.get("schema_version") != 1:
            raise ValueError("配置格式或路径基准不符")
        if baseline.get("feature_set") != FEATURE_SET or baseline.get("target_id") != TARGET_SPEC["target_id"]:
            raise ValueError("不支持未定义的特征或目标版本")
        selection = baseline.get("source_acceptance")
        if not isinstance(selection, dict) or not isinstance(selection.get("inspection_run"), str) or not selection["inspection_run"].strip():
            raise ValueError("实验配置必须指定具名P0准入记录")
        supplements = selection.get("supplements", [])
        if not isinstance(supplements, list) or any(not isinstance(item, str) or not item.strip() for item in supplements):
            raise ValueError("补充准入记录必须是路径字符串列表")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"exit_code": 2, "error": str(error), "output_created": False}, ensure_ascii=False), flush=True)
        return 2
    run = RunRecord(module_root, "P1")
    archive = LocalAdmissionArchive(module_root, source.history, source.holdout, ledger)
    history = AdmittedHistorySource(source, archive, selection)
    store = NumpyPreparedStore(module_root, run.directory, run.context)
    try:
        outcome = prepare_samples(history, store, run.context)
    except BaseException:
        run.finish("failed")
        raise
    run.finish(outcome.status)
    print(json.dumps({"run_directory": str(run.directory), "state": outcome.result["state"], "exit_code": outcome.exit_code}, ensure_ascii=False), flush=True)
    return outcome.exit_code


def prepare_time_splits(module_root: Path, prepared_run: str, experiment_config: Path | None = None):
    from .experiments.application import prepare_splits, SPLIT_MODULES
    from .experiments.sample_source import PreparedIndexSource
    from .experiments.numpy_store import NumpySplitStore
    from .samples.prepared_reader import NumpyPreparedReader
    from .serialization import file_hash
    try:
        path = (module_root / (experiment_config or Path("configs/baseline.json"))).resolve()
        if not path.is_relative_to(module_root):
            raise ValueError("时间方案配置须位于模型目录内")
        digest = file_hash(path)
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if baseline.get("schema_version") != 1 or not isinstance(baseline.get("split_plan"), dict):
            raise ValueError("实验配置缺少P2时间方案")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({"exit_code": 2, "error": str(error), "output_created": False}, ensure_ascii=False), flush=True)
        return 2
    run = RunRecord(module_root, "P2")
    reader = NumpyPreparedReader(module_root)
    source = PreparedIndexSource(reader, prepared_run, (baseline.get("feature_set"), baseline.get("target_id")))
    store = NumpySplitStore(module_root, run.directory, run.context,
                           {name: run.context.code_sha256[name] for name in SPLIT_MODULES}, path, digest)
    try:
        outcome = prepare_splits(source, store, baseline["split_plan"], run.context)
    except BaseException:
        run.finish("failed")
        raise
    run.finish(outcome.status)
    print(json.dumps({"run_directory": str(run.directory), "state": outcome.result["state"], "exit_code": outcome.exit_code}, ensure_ascii=False), flush=True)
    return outcome.exit_code


def validate_development_fold(module_root: Path, fold: str, experiment_config: Path | None = None):
    from .experiments.validation_domain import ValidationSettings
    from .experiments.evaluation_domain import FoldInputSpec
    from .experiments.validation import validate_fold, VALIDATION_MODULES
    from .experiments.validation_source import PreparedFoldSource
    from .experiments.validation_archive import LocalValidationArchive
    from .experiments.lightgbm_model import LightGBMTrainer
    from .samples.prepared_reader import NumpyPreparedReader
    from .resources import ResourceMonitor
    from .serialization import file_hash
    try:
        path = (module_root / (experiment_config or Path("configs/p3-validation.json"))).resolve()
        if not path.is_relative_to(module_root):
            raise ValueError("训练配置须位于模型目录内")
        digest = file_hash(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        settings = ValidationSettings.from_mapping(config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"exit_code": 2, "error": str(error), "output_created": False}, ensure_ascii=False), flush=True)
        return 2
    run = RunRecord(module_root, "P3")
    telemetry = ResourceMonitor(module_root)
    request = FoldInputSpec(settings.split_run, settings.feature_set, settings.feature_count, settings.target_id)
    source = PreparedFoldSource(module_root, NumpyPreparedReader(module_root), request, fold, path, digest)
    archive = LocalValidationArchive(module_root, run.directory, run.context,
                                     {name: run.context.code_sha256[name] for name in VALIDATION_MODULES}, run.record["environment"])
    trainer = LightGBMTrainer(telemetry, run.context.progress)
    try:
        outcome = validate_fold(source, trainer, archive, telemetry, settings, config, fold, run.context)
    except BaseException:
        source.close()
        run.finish("failed")
        raise
    run.finish(outcome.status)
    print(json.dumps({"run_directory": str(run.directory), "state": outcome.result["state"], "fold": fold,
                      "best_iteration": outcome.result.get("best_iteration"), "exit_code": outcome.exit_code}, ensure_ascii=False), flush=True)
    return outcome.exit_code
