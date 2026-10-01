"""本项目源码依赖守卫：拒绝未知位置、越界访问和动态加载，不作通用Python安全分析。"""

import ast
import sys
from pathlib import Path

PREFIX = "daily_return"
POLICY = {
    "": (),
    "__main__": ("bootstrap",),
    "bootstrap": ("execution", "serialization", "resources", "market_data", "market_data.domain", "market_data.application", "market_data.local_source", "market_data.archive", "samples", "samples.domain", "samples.application", "samples.history_reader", "samples.numpy_store", "samples.prepared_reader", "experiments.application", "experiments.sample_source", "experiments.numpy_store", "experiments.validation_domain", "experiments.evaluation_domain", "experiments.validation", "experiments.validation_source", "experiments.validation_archive", "experiments.lightgbm_model",),
    "execution": ("serialization",),
    "serialization": (),
    "resources": (),
    "market_data": ("market_data.domain", "market_data.ports", "market_data.application",),
    "market_data.domain": (),
    "market_data.ports": ("market_data.domain",),
    "market_data.application": ("market_data.domain", "market_data.ports", "execution",),
    "market_data.local_source": ("market_data.domain", "market_data.ports", "serialization",),
    "market_data.archive": ("market_data.domain", "market_data.ports", "market_data.local_source", "serialization",),
    "samples": ("samples.domain", "samples.features", "samples.ports", "samples.application",),
    "samples.domain": (),
    "samples.features": ("samples.domain",),
    "samples.targets": ("samples.domain",),
    "samples.ports": ("samples.domain",),
    "samples.application": ("samples.domain", "samples.features", "samples.targets", "samples.ports", "execution",),
    "samples.history_reader": ("samples.domain", "samples.ports", "market_data",),
    "samples.numpy_store": ("samples.domain", "samples.ports", "serialization", "execution",),
    "samples.prepared_reader": ("samples.domain", "samples.numpy_store", "serialization",),
    "experiments": ("experiments.application", "experiments.evaluation_domain", "experiments.evaluation_ports",),
    "experiments.domain": (),
    "experiments.ports": ("experiments.domain",),
    "experiments.application": ("experiments.domain", "experiments.ports", "execution",),
    "experiments.sample_source": ("experiments.domain", "samples",),
    "experiments.numpy_store": ("experiments.domain", "serialization",),
    "experiments.evaluation_domain": (),
    "experiments.evaluation_ports": ("experiments.evaluation_domain",),
    "experiments.validation_domain": ("experiments.evaluation_domain",),
    "experiments.evaluation": ("experiments.evaluation_domain",),
    "experiments.validation_ports": ("experiments.evaluation_domain", "experiments.validation_domain",),
    "experiments.validation_source": ("samples", "experiments.numpy_store", "experiments.evaluation_domain", "serialization",),
    "experiments.lightgbm_model": ("experiments.evaluation_domain",),
    "experiments.validation": ("experiments.evaluation_domain", "experiments.evaluation", "experiments.validation_ports", "experiments.evaluation_ports", "execution",),
    "experiments.validation_archive": ("serialization", "resources",),
    "experiments.holdout_usage": ("experiments.evaluation_domain", "serialization",),
    "next_day_up": (),
    "next_day_up.__main__": ("next_day_up.bootstrap",),
    "next_day_up.bootstrap": ("execution", "resources", "serialization", "samples.prepared_reader", "experiments", "experiments.validation_source", "experiments.holdout_usage", "next_day_up.domain", "next_day_up.validation", "next_day_up.holdout", "next_day_up.development_source", "next_day_up.lightgbm_model", "next_day_up.archive",),
    "next_day_up.domain": ("experiments",),
    "next_day_up.ports": ("experiments", "next_day_up.domain",),
    "next_day_up.evaluation": ("experiments", "next_day_up.domain",),
    "next_day_up.lightgbm_model": ("experiments",),
    "next_day_up.validation": ("experiments", "execution", "next_day_up.domain", "next_day_up.evaluation", "next_day_up.ports",),
    "next_day_up.holdout": ("next_day_up.validation", "experiments", "execution", "next_day_up.domain", "next_day_up.ports", "next_day_up.evaluation",),
    "next_day_up.development_source": ("next_day_up.domain", "experiments",),
    "next_day_up.archive": ("serialization", "resources",),
}
PURE = {"market_data.domain", "market_data.ports", "samples.domain", "samples.features", "samples.targets", "samples.ports", "experiments.domain", "experiments.ports", "experiments.evaluation_domain", "experiments.evaluation_ports", "experiments.validation_domain", "experiments.evaluation", "experiments.validation_ports", "next_day_up.domain", "next_day_up.ports", "next_day_up.evaluation"}
APPLICATIONS = {"market_data.application", "samples.application", "experiments.application", "experiments.validation", "next_day_up.validation", "next_day_up.holdout"}
PURE_LIBRARIES = {"__future__", "collections", "dataclasses", "datetime", "decimal", "enum", "math", "typing", "types"}
IO_CALLS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir", "unlink", "rename", "rglob", "glob", "stat", "memmap", "open_memmap", "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "fromfile", "tofile", "getattr", "globals", "locals", "eval", "exec", "compile"}
DYNAMIC = {"__import__", "import_module", "exec_module", "run_module", "run_path", "getattr", "globals", "locals", "eval", "exec", "compile"}


def policy_module(role):
    return role if role == "next_day_up" or role.startswith("next_day_up.") else PREFIX + ("." + role if role else "")


def module_name(path):
    parts = Path(path).with_suffix("").parts
    if parts[0] == "methods":
        if parts[1:4] != ("next-day-up", "src", "next_day_up"):
            return "unapproved." + ".".join(parts)
        parts = parts[3:]  # methods/<方法>/src/<包>/...
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if parts and parts[0] == "next_day_up":
        return ".".join(parts)
    return ".".join((PREFIX, *parts))


def qualified(node, bindings):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return qualified(node.value, bindings) + "." + node.attr
    return ""


def check_sources(sources, require_complete=False):
    trees = {module_name(path): (path, ast.parse(code)) for path, code in sources.items()}
    violations, graph, exports = [], {}, {}
    for module, (_, tree) in trees.items():
        exports[module] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                try:
                    exports[module] = set(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    violations.append(f"{module}: __all__必须是字面量清单")
    if require_complete:
        expected = {policy_module(key) for key in POLICY}
        violations.extend("缺少目标模块：" + name for name in sorted(expected - set(trees)))
    for module, (path, tree) in trees.items():
        role = module if module.startswith("next_day_up") else module.removeprefix(PREFIX).lstrip(".")
        if role not in POLICY:
            violations.append(f"{path}: 未获批准的模块位置（旧模块须删除）")
            continue
        allowed = {policy_module(key) for key in POLICY[role]}
        bindings, graph[module] = {}, set()
        package = module if Path(path).name == "__init__.py" else module.rpartition(".")[0]
        restricted = role in PURE | APPLICATIONS

        def report(node, reason):
            violations.append(f"{path}:{node.lineno}: {reason}")

        def dependency(node, target, imported=None):
            if target == PREFIX or target.startswith(PREFIX + ".") or target == "next_day_up" or target.startswith("next_day_up."):
                if target not in allowed:
                    report(node, f"依赖越界：{module} -> {target}")
                graph[module].add(target)
                public_access = (role.startswith("samples.") and target == PREFIX + ".market_data") or (role.startswith("experiments.") and target == PREFIX + ".samples") or (role.startswith("next_day_up.") and target == PREFIX + ".experiments")
                if public_access and imported is not None:
                    if imported not in exports.get(target, ()):
                        report(node, f"访问未公开符号：{target}.{imported}")
            else:
                library = target.split(".")[0]
                external = {"numpy"} if role.startswith(("samples", "experiments", "next_day_up")) else set()
                if role in ("experiments.lightgbm_model", "next_day_up.lightgbm_model"):
                    external.add("lightgbm")
                if library not in sys.stdlib_module_names | external:
                    report(node, "未批准的第三方依赖：" + target)
                permitted = PURE_LIBRARIES | external
                if restricted and library not in permitted:
                    report(node, f"规则或应用禁止技术依赖：{target}")

        # 包含函数内与TYPE_CHECKING导入；解析导入别名及简单引用转存。
        nodes = sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependency(node, alias.name)
                    bindings[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    target = ".".join(base[:len(base)-node.level+1] + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        report(node, "禁止通配符导入")
                    child = target + "." + alias.name
                    dependency(node, child if child in trees else target, alias.name)
                    bindings[alias.asname or alias.name] = child
                    if role in APPLICATIONS and target == PREFIX + ".execution" and alias.name != "RunContext":
                        report(node, "应用只可引用RunContext契约")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                reference = qualified(node.value, bindings)
                if reference:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id] = reference
            elif isinstance(node, ast.Call):
                name = qualified(node.func, bindings)
                leaf = name.rsplit(".", 1)[-1]
                if leaf in DYNAMIC:
                    report(node, "禁止动态加载：" + name)
                if restricted and leaf in IO_CALLS:
                    report(node, "规则或应用禁止I/O及反射：" + name)
            elif isinstance(node, ast.Attribute):
                name = qualified(node, bindings)
                if restricted and name.startswith("numpy.") and node.attr in IO_CALLS:
                    report(node, "规则或应用禁止引用数值文件I/O：" + name)
                if node.attr in {"__dict__", "__globals__", "__getattribute__"}:
                    report(node, "禁止反射穿透模块：" + name)
                for consumer, provider in (("samples.", "market_data"), ("experiments.", "samples"), ("next_day_up.", "experiments")):
                    gateway = PREFIX + "." + provider + "."
                    if role.startswith(consumer) and name.startswith(gateway):
                        member = name.removeprefix(gateway).split(".")[0]
                        if member not in exports.get(PREFIX + "." + provider, ()):
                            report(node, "穿透业务公开边界：" + name)
    visited = set()

    def visit(module, stack):
        if module in stack:
            violations.append("循环依赖：" + " -> ".join((*stack, module)))
        elif module not in visited:
            for target in graph.get(module, ()):
                visit(target, (*stack, module))
            visited.add(module)
    for module in graph:
        visit(module, ())
    return sorted(set(violations))
