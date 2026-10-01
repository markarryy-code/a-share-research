"""进程资源观测：Windows原生峰值工作集，采样记录不干预模型参数。"""

import ctypes
import os
import shutil
import time
from datetime import datetime, timezone


def memory_sample():
    result = {"working_set_bytes": None, "peak_working_set_bytes": None, "private_bytes": None, "available_memory_bytes": None}
    if os.name != "nt":
        return {**result, "method": "unavailable"}
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
                   ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]
    class SystemMemory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
                   ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "reserved")]
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong)
    proc = ProcessMemory()
    proc.cb = ctypes.sizeof(proc)
    if not query(current(), ctypes.byref(proc), proc.cb):
        raise OSError("无法取得Windows进程内存观测")
    system = SystemMemory()
    system.length = ctypes.sizeof(system)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system)):
        raise OSError("无法取得Windows可用内存")
    return {"working_set_bytes": proc.working, "peak_working_set_bytes": proc.peak_working,
            "private_bytes": proc.private, "available_memory_bytes": system.available,
            "method": "GetProcessMemoryInfo/GlobalMemoryStatusEx; peak is cumulative process working set"}


class ResourceMonitor:
    def __init__(self, root):
        self.root, self.started, self.previous, self.samples = root, time.perf_counter(), None, []

    def checkpoint(self, phase):
        elapsed = time.perf_counter() - self.started
        record = {"phase": phase, "utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
                  "since_previous_seconds": elapsed-(self.previous or 0), **memory_sample(), "disk_free_bytes": shutil.disk_usage(self.root).free}
        self.previous = elapsed
        self.samples.append(record)
        return record

    def records(self):
        return list(self.samples)


def summarize_resources(records):
    marks = {r["phase"]: r["elapsed_seconds"] for r in records if r["phase"] != "training_iteration"}
    pairs = {"input_loading": ("started", "input_verified"), "matrix_selection": ("input_verified", "matrices_selected"),
             "dataset_construction": ("matrices_selected", "datasets_constructed"), "fit": ("datasets_constructed", "fit_completed"),
             "prediction": ("matrices_released", "prediction_completed"), "scoring": ("prediction_completed", "scoring_completed"),
             "input_recheck": ("scoring_completed", "input_reverified")}
    peaks = [r["peak_working_set_bytes"] for r in records if r["peak_working_set_bytes"] is not None]
    return {"phase_seconds": {k: marks[b]-marks[a] for k, (a, b) in pairs.items() if a in marks and b in marks},
            "elapsed_seconds": records[-1]["elapsed_seconds"] if records else None,
            "process_peak_working_set_bytes": max(peaks) if peaks else None,
            "method": "Windows OS cumulative process peak; checkpoints are not separate phase peaks", "checkpoints": records}
