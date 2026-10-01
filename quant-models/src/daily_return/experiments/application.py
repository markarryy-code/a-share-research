"""P2用例：完成样本到时间角色索引，训练与效果评价尚不执行。"""

from dataclasses import dataclass
from .domain import SplitError, build_split_plan, select_fold_rows, population_coverage
from .ports import SampleSource, SplitStore
from ..execution import RunContext


# P2调用的读取/计算/序列化依赖；原P1数值由具名样本指纹绑定。
SPLIT_MODULES = (
    "serialization.py", "samples/__init__.py", "samples/domain.py", "samples/ports.py",
    "samples/application.py", "samples/numpy_store.py", "samples/prepared_reader.py", "samples/targets.py",
    "experiments/__init__.py", "experiments/domain.py", "experiments/ports.py",
    "experiments/evaluation_domain.py", "experiments/evaluation_ports.py",
    "experiments/application.py", "experiments/sample_source.py", "experiments/numpy_store.py",
)


@dataclass(frozen=True)
class SplitOutcome:
    result: dict
    status: str
    exit_code: int


def prepare_splits(source: SampleSource, store: SplitStore, config: dict, context: RunContext):
    result = {"stage": "P2", "state": "failed", "splits_prepared": False,
              "model_trained": False, "holdout_evaluation_performed": False}
    receipt, issues, status, exit_code = (), [], "running", 0
    try:
        population = source.read_population()
        plan = build_split_plan(config, population)
        identity = store.identity(population, plan)
        stored = store.find_complete(identity)
        reused = stored is not None
        if reused:
            receipt = source.verify_unchanged()
        else:
            writer = store.begin(population, plan, identity)
            for window in plan.windows:
                selection = select_fold_rows(population, window)
                writer.append(selection)
                context.progress({"stage": "split", "window": window.name,
                                  "rows": {role.name: len(role.indices) for role in selection.roles}})
            receipt = source.verify_unchanged()
            stored = writer.publish(receipt)
        result.update(stored)
        result.update(state="split-ready", splits_prepared=True, cache_reused=reused,
                      parent_run=population.parent_run, population=population_coverage(population),
                      limitations=dict(population.limitations))
        status = "completed"
    except KeyboardInterrupt:
        result["state"], status, exit_code = "interrupted", "interrupted", 130
        issues.append({"code": "interrupted", "detail": "P2中断，未完成索引保留为partial"})
    except Exception as error:
        status, exit_code = "failed", 2 if isinstance(error, SplitError) else 1
        issues.append({"code": type(error).__name__, "detail": str(error)})
    finally:
        if not receipt:
            try:
                receipt = source.verify_unchanged()
            except SplitError as error:
                issues.append({"code": "source_changed", "detail": str(error)})
        store.finish(result, receipt, issues)
    return SplitOutcome(result, status, exit_code)
