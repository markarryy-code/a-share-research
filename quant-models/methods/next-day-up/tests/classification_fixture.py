"""复用共用合成行情准备流程，通过独立分类入口运行测试，不使用真实行情。"""

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from fixtures import write_json
from validation_fixtures import ValidationFixture

METHOD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = METHOD_ROOT.parents[1]


def compressed_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ClassificationFixture(ValidationFixture):
    def setUp(self):
        super().setUp()
        self.method_root = self.module / "methods/next-day-up"
        self.config_path = self.method_root / "configs/p3-classification.json"
        self.config["target_id"] = "next-market-day-up-v1"
        self.config["model_params"].update(objective="binary", metric="binary_logloss")
        write_json(self.config_path, self.config)

    def method_command(self, command, *args):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(WORKSPACE_ROOT/"src"), str(METHOD_ROOT/"src"))),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        process = subprocess.run([sys.executable, "-B", "-m", "next_day_up", command,
                                  "--workspace-root", str(self.module), *args],
                                 env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)
        lines = process.stdout.strip().splitlines()
        last = json.loads(lines[-1]) if lines else {}
        return process, Path(last["run_directory"]) if "run_directory" in last else None

    def validate(self):
        process, run = self.method_command("validate", "--fold", "F01")
        result = json.loads((run/"validation.json").read_text(encoding="utf-8")) if run else None
        return process, result, run
