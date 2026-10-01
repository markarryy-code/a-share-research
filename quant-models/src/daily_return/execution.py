"""运行生命周期和完整源码指纹；不决定业务准入或样本有效性。"""

import ctypes
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Callable, Mapping
import json
import time
import uuid
from .serialization import save_json


def environment(module_root: Path, *, method_source: Path | None = None) -> dict:
    packages = {}
    for name in ("numpy", "scipy", "narwhals", "pandas", "lightgbm", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = {"total_bytes": None, "available_bytes": None, "source": "unavailable"}
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "reserved")
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = {"total_bytes": status.total, "available_bytes": status.available, "source": "GlobalMemoryStatusEx"}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=module_root, capture_output=True, text=True, timeout=10)
        head = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        head = None
    source_root = Path(__file__).parent
    code_files = {path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_root.rglob("*.py"))}
    if method_source is not None:
        # 方法输出可写入临时工作区，代码指纹始终取实际执行的源码目录。
        prefix = method_source.relative_to(source_root.parents[1]).as_posix()
        code_files.update({prefix + "/" + path.relative_to(method_source).as_posix():
                           hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(method_source.rglob("*.py"))})
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "memory": memory, "installed_training_packages": packages,
        "p0_requires_third_party_packages": False, "git_head": head, "code_sha256": code_files,
    }


@dataclass(frozen=True)
class RunContext:
    run_id: str
    code_sha256: Mapping[str, str]
    python_version: str
    numpy_version: str | None
    progress: Callable[[dict], None]


class RunRecord:
    """技术执行记录与业务结果分离；文件实现由入口持有。"""
    def __init__(self, module_root: Path, stage: str, *, output_root: Path | None = None,
                 method_source: Path | None = None):
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + stage.lower() + "-" + uuid.uuid4().hex[:8]
        self.directory = (output_root or module_root) / "runs" / now.strftime("%Y-%m-%d") / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        env = environment(module_root, method_source=method_source)
        self.record = {"run_id": run_id, "stage": stage, "execution_status": "running",
                       "started_at_utc": now.isoformat(), "environment": env}
        self.context = RunContext(run_id, MappingProxyType(env["code_sha256"]), platform.python_version(),
                                  env["installed_training_packages"]["numpy"], self.progress)
        save_json(self.directory / "experiment.json", self.record)

    @staticmethod
    def progress(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def finish(self, status):
        self.record.update(execution_status=status, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=round(time.perf_counter() - self.started, 3))
        save_json(self.directory / "experiment.json", self.record)
