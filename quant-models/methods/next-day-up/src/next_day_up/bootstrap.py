"""二分类唯一组装入口；数据和登记以quant-models为根，新运行归本方法目录。"""

import json
from pathlib import Path
from daily_return.execution import RunRecord
from daily_return.resources import ResourceMonitor
from daily_return.serialization import file_hash
from daily_return.samples.prepared_reader import NumpyPreparedReader
from daily_return.experiments import FoldInputSpec, ValidationError
from daily_return.experiments.validation_source import PreparedFoldSource
from daily_return.experiments.holdout_usage import LocalHoldoutUsage
from .domain import ClassificationSettings, UP_TARGET_SPEC
from .validation import validate_fold, VALIDATION_MODULES
from .holdout import evaluate_holdout, HOLDOUT_MODULES
from .development_source import load_classification_plan
from .lightgbm_model import LightGBMTrainer
from .archive import LocalValidationArchive


def validate_development_fold(workspace_root: Path, fold: str, experiment_config: Path | None = None):
    method_root = workspace_root / "methods/next-day-up"
    try:
        path = (method_root / (experiment_config or Path("configs/p3-classification.json"))).resolve()
        if not path.is_relative_to(method_root):
            raise ValidationError("分类训练配置须位于本方法目录")
        configuration = json.loads(path.read_text(encoding="utf-8"))
        settings = ClassificationSettings.from_mapping(configuration)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"exit_code": 2, "error": str(error), "output_created": False}, ensure_ascii=False), flush=True)
        return 2
    run = RunRecord(workspace_root, "P3", output_root=method_root, method_source=Path(__file__).parent)
    telemetry = ResourceMonitor(workspace_root)
    request = FoldInputSpec(settings.split_run, settings.feature_set, settings.feature_count, UP_TARGET_SPEC["source_target_id"])
    source = PreparedFoldSource(workspace_root, NumpyPreparedReader(workspace_root), request, fold, path, file_hash(path))
    archive = LocalValidationArchive(workspace_root, run.directory, run.context,
                                     {name: run.context.code_sha256[name] for name in VALIDATION_MODULES},
                                     run.record["environment"])
    trainer = LightGBMTrainer(telemetry, run.context.progress)
    try:
        outcome = validate_fold(source, trainer, archive, telemetry, settings, configuration, fold, run.context)
    except BaseException:
        source.close()
        run.finish("failed")
        raise
    run.finish(outcome.status)
    print(json.dumps({"run_directory": str(run.directory), "state": outcome.result["state"], "fold": fold,
                      "best_iteration": outcome.result.get("best_iteration"), "exit_code": outcome.exit_code},
                     ensure_ascii=False), flush=True)
    return outcome.exit_code


def evaluate_classification_holdout(workspace_root: Path, development_selection: Path):
    method_root = workspace_root / "methods/next-day-up"
    try:
        path = (workspace_root / development_selection).resolve()
        settings, configuration, inputs = load_classification_plan(workspace_root, path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"exit_code": 2, "error": str(error), "output_created": False}, ensure_ascii=False), flush=True)
        return 2
    run = RunRecord(workspace_root, "P4", output_root=method_root, method_source=Path(__file__).parent)
    telemetry = ResourceMonitor(workspace_root)
    request = FoldInputSpec(settings.split_run, settings.feature_set, settings.feature_count, UP_TARGET_SPEC["source_target_id"])
    source = PreparedFoldSource(workspace_root, NumpyPreparedReader(workspace_root), request, "holdout", path, file_hash(path),
                                kind="holdout", input_records=inputs)
    archive = LocalValidationArchive(workspace_root, run.directory, run.context,
                                     {name: run.context.code_sha256[name] for name in HOLDOUT_MODULES},
                                     run.record["environment"])
    usage = LocalHoldoutUsage(workspace_root, run.directory)
    trainer = LightGBMTrainer(telemetry, run.context.progress)
    try:
        outcome = evaluate_holdout(source, trainer, archive, usage, telemetry, settings, configuration, run.context)
    except BaseException:
        source.close()
        usage.finish("failed")
        run.finish("failed")
        raise
    run.finish(outcome.status)
    print(json.dumps({"run_directory": str(run.directory), "state": outcome.result["state"],
                      "fixed_rounds": settings.rounds, "holdout_evaluation_performed": outcome.result["holdout_evaluation_performed"],
                      "exit_code": outcome.exit_code}, ensure_ascii=False), flush=True)
    return outcome.exit_code
