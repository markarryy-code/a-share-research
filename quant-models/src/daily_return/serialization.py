"""无业务规则的JSON、CSV和摘要工具，内容统一使用UTF-8和LF。"""

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def json_value(value):
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"不支持的JSON值：{type(value).__name__}")


def save_json(path: Path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False, default=json_value) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, fields: tuple, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", restval=0)
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=json_value)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
