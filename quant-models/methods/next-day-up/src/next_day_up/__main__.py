"""次日上涨分类入口；共用daily_return入口只管理数据准备和收益回归。"""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="次日上涨二分类研究")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="训练并评价一个具名分类开发期折")
    validate.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[4])
    validate.add_argument("--fold", required=True, choices=[f"F{i:02}" for i in range(1, 8)])
    validate.add_argument("--experiment-config", type=Path, help="相对方法目录，默认configs/p3-classification.json")
    holdout = commands.add_parser("evaluate-holdout", help="按开发期冻结设置执行一次留出评价")
    holdout.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[4])
    holdout.add_argument("--development-selection", type=Path, required=True, help="相对quant-models的具名七折选用清单")
    args = parser.parse_args()
    from .bootstrap import validate_development_fold, evaluate_classification_holdout
    if args.command == "validate":
        return validate_development_fold(args.workspace_root.resolve(), args.fold, args.experiment_config)
    return evaluate_classification_holdout(args.workspace_root.resolve(), args.development_selection)


if __name__ == "__main__":
    raise SystemExit(main())
