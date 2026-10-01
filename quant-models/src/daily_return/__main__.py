"""公开命令入口；P0保持标准库，P1按命令加载NumPy。"""

import argparse
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description="单日涨跌幅研究")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-data", help="只读检查P0资料并保存独立报告")
    inspect.add_argument("--module-root", type=Path, default=Path(__file__).resolve().parents[2])
    inspect.add_argument("--data-config", type=Path, help="相对模型目录的配置路径，默认configs/data_sources.json")
    prepare = commands.add_parser("prepare", help="准备P1样本，或从具名P1构建P2时间索引；不训练")
    prepare.add_argument("--module-root", type=Path, default=Path(__file__).resolve().parents[2])
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument("--data-config", type=Path, help="P1数据位置配置")
    selection.add_argument("--prepared-run", help="相对模型目录的具名P1运行；提供时只执行P2")
    prepare.add_argument("--experiment-config", type=Path, help="相对模型目录，默认configs/baseline.json")
    validate = commands.add_parser("validate", help="训练并评价一个明确的收益回归开发期折")
    validate.add_argument("--module-root", type=Path, default=Path(__file__).resolve().parents[2])
    validate.add_argument("--fold", required=True, choices=[f"F{i:02}" for i in range(1, 8)])
    validate.add_argument("--experiment-config", type=Path, help="相对模型目录，默认configs/p3-validation.json")
    args = parser.parse_args()
    if args.command == "inspect-data":
        from .bootstrap import inspect_data
        return inspect_data(args.module_root.resolve(), args.data_config)
    try:
        if args.command == "validate":
            from .bootstrap import validate_development_fold
            return validate_development_fold(args.module_root.resolve(), args.fold, args.experiment_config)
        if args.prepared_run:
            from .bootstrap import prepare_time_splits
            return prepare_time_splits(args.module_root.resolve(), args.prepared_run, args.experiment_config)
        from .bootstrap import prepare_dataset
        return prepare_dataset(args.module_root.resolve(), args.data_config, args.experiment_config)
    except ModuleNotFoundError as error:
        if error.name not in ("numpy", "lightgbm", "scipy", "narwhals"):
            raise
        required = "requirements-p3.txt" if args.command == "validate" else "requirements-p1.txt"
        parser.error("缺少必要依赖，请使用模型隔离环境并安装" + required)


if __name__ == "__main__":
    raise SystemExit(main())
