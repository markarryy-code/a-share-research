"""既检查实际源码，也用明确的违规样例证明守卫能拦截。"""

from pathlib import Path
import ast
import hashlib
import sys
import unittest
from architecture_policy import check_sources


class ArchitecturePolicyTests(unittest.TestCase):
    def test_samples_can_publish_pure_features_without_opening_private_module_access(self):
        public = "from .features import build_stock_features\n__all__ = ('build_stock_features',)"
        self.assertEqual(check_sources({"samples/__init__.py": public, "samples/features.py": ""}), [])
        self.assertEqual(check_sources({"samples/__init__.py": public,
                                       "experiments/sample_source.py": "from ..samples import build_stock_features"}), [])
        self.assertTrue(check_sources({"samples/__init__.py": public,
                                       "experiments/sample_source.py": "from ..samples.features import build_stock_features"}))

    def test_pure_calculation_and_same_business_import_are_allowed(self):
        self.assertEqual(check_sources({"samples/domain.py": "from dataclasses import dataclass\nimport numpy as np\nx = np.zeros(3)", "samples/features.py": "from .domain import X"}), [])

    def test_relative_absolute_delayed_and_type_only_imports_are_guarded(self):
        for code in ("from .numpy_store import X", "import daily_return.samples.numpy_store as s", "def f():\n from .numpy_store import X", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n from .numpy_store import X"):
            with self.subTest(code=code):
                self.assertTrue(check_sources({"samples/domain.py": code}))

    def test_io_alias_and_function_reference_are_guarded(self):
        for code in ("open('x')", "import numpy as n\nn.load('x')", "from numpy import load as read\nread('x')", "import numpy as n\nread=n.load\nread('x')", "import numpy as n\nread: object=n.load\nread('x')", "import numpy as n\ngetattr(n, 'load')('x')"):
            with self.subTest(code=code):
                self.assertTrue(check_sources({"samples/features.py": code}))

    def test_dynamic_import_and_star_import_are_rejected(self):
        for code in ("from .domain import *", "from importlib import import_module as load\nload('daily_return.samples.numpy_store')", "__import__('os')"):
            self.assertTrue(check_sources({"samples/application.py": code}))

    def test_cross_module_gateway_cannot_expose_infrastructure(self):
        public = "__all__ = ('AdmissionSnapshot',)"
        for code in ("from ..market_data.local_source import Reader", "from ..market_data import local_source", "import daily_return.market_data as md\nmd.local_source.Reader()", "import daily_return.market_data as md\ngetattr(md, 'local_source').Reader()"):
            self.assertTrue(check_sources({"market_data/__init__.py": public, "market_data/local_source.py": "", "samples/history_reader.py": code}))
        self.assertEqual(check_sources({"market_data/__init__.py": public, "samples/history_reader.py": "from ..market_data import AdmissionSnapshot"}), [])

    def test_p0_cannot_acquire_numpy_and_old_root_files_are_rejected(self):
        self.assertTrue(check_sources({"market_data/domain.py": "import numpy"}))
        self.assertTrue(check_sources({"dataset.py": ""}))
        self.assertTrue(check_sources({}, require_complete=True))

    def test_cycles_and_infrastructure_reexports_are_rejected(self):
        errors = check_sources({"samples/domain.py": "from .application import run", "samples/application.py": "from .domain import X"})
        self.assertTrue(any("循环依赖" in error for error in errors))
        self.assertTrue(check_sources({"market_data/__init__.py": "from .local_source import Reader\n__all__=('Reader',)", "market_data/local_source.py": ""}))

    def test_experiments_use_public_samples_and_pure_selection_only(self):
        public = "__all__ = ('read_prepared_index',)"
        self.assertEqual(check_sources({"samples/__init__.py": public, "experiments/sample_source.py": "from ..samples import read_prepared_index", "experiments/domain.py": "import numpy as np\nx = np.flatnonzero([True])"}), [])
        for code in ("from ..samples.prepared_reader import Reader", "from ..samples import prepared_reader", "import daily_return.samples as s\ns.prepared_reader.Reader()"):
            self.assertTrue(check_sources({"samples/__init__.py": public, "samples/prepared_reader.py": "", "experiments/sample_source.py": code}))
        for file, code in (("experiments/domain.py", "from .numpy_store import Writer"), ("experiments/application.py", "from .numpy_store import Writer"), ("experiments/domain.py", "import numpy as np\np = np.load\np('x')")):
            self.assertTrue(check_sources({file: code}))

    def test_lightgbm_is_limited_to_model_adapter_and_telemetry_to_composition(self):
        self.assertEqual(check_sources({"experiments/lightgbm_model.py": "import lightgbm as lgb"}), [])
        for path in ("experiments/validation_domain.py", "experiments/evaluation.py", "experiments/validation.py", "samples/domain.py"):
            self.assertTrue(check_sources({path: "import lightgbm"}))
        self.assertTrue(check_sources({"experiments/validation.py": "from ..resources import ResourceMonitor"}))
        self.assertTrue(check_sources({"experiments/validation_source.py": "from ..samples.prepared_reader import NumpyPreparedReader"}))

    def test_method_label_rules_use_public_data_contracts_without_loading_arrays(self):
        public = "__all__ = ('ValidationError',)"
        allowed = {"experiments/__init__.py": public,
                   "next_day_up/domain.py": "from daily_return.experiments import ValidationError"}
        self.assertEqual(check_sources(allowed), [])
        for code in ("from daily_return.samples.targets import up_labels",
                     "import numpy as np\nnp.load('y.npy')", "import lightgbm"):
            self.assertTrue(check_sources({"next_day_up/domain.py": code}))
        self.assertTrue(check_sources({"experiments/classification.py": ""}))

    def test_method_holdout_uses_public_ports_not_shared_registry_files(self):
        public = "__all__ = ('HoldoutUsage',)"
        self.assertEqual(check_sources({"experiments/__init__.py": public,
                         "next_day_up/holdout.py": "from daily_return.experiments import HoldoutUsage"}), [])
        for code in ("from daily_return.experiments.holdout_usage import LocalHoldoutUsage",
                     "import lightgbm", "open('usage.json')"):
            self.assertTrue(check_sources({"next_day_up/holdout.py": code}))
        self.assertTrue(check_sources({"next_day_up/domain.py": "from .development_source import load_classification_plan"}))

    def test_methods_cannot_bypass_shared_gateway_or_approved_source_location(self):
        public = "__all__ = ('FoldDescription',)"
        for code in ("from daily_return.experiments.validation_source import PreparedFoldSource",
                     "import daily_return.experiments as data\ndata.validation_source.PreparedFoldSource()",
                     "from daily_return.experiments import validation_source"):
            self.assertTrue(check_sources({"experiments/__init__.py": public, "next_day_up/evaluation.py": code}))
        self.assertTrue(check_sources({"methods/unapproved/src/next_day_up/domain.py": ""}))
        self.assertTrue(check_sources({"samples/domain.py": "from next_day_up.domain import up_labels"}))
        self.assertEqual(check_sources({"next_day_up/lightgbm_model.py": "import lightgbm"}), [])


class ProjectArchitectureTests(unittest.TestCase):
    def test_actual_dependency_graph(self):
        root = Path(__file__).resolve().parents[1]
        shared = root/"src/daily_return"
        sources = {p.relative_to(shared).as_posix(): p.read_text(encoding="utf-8") for p in shared.rglob("*.py")}
        for directory in (root/"methods").glob("*/src"):
            sources.update({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in directory.rglob("*.py")})
        self.assertEqual(check_sources(sources, require_complete=True), [])

    def test_recursive_code_fingerprint_and_calculation_inventory(self):
        root = Path(__file__).resolve().parents[1]
        method = root/"methods/next-day-up/src/next_day_up"
        sys.path.insert(0, str(root/"src"))
        sys.path.insert(0, str(method.parent))
        from daily_return.execution import environment
        code_root = root/"src/daily_return"
        shared = {p.relative_to(code_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in code_root.rglob("*.py")}
        owned = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in method.rglob("*.py")}
        self.assertEqual(environment(root)["code_sha256"], shared)
        self.assertEqual(environment(root, method_source=method)["code_sha256"], shared | owned)

        def declared(path, name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return set(next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))

        preparation = {p for p in shared if p == "serialization.py" or p.startswith(("market_data/", "samples/"))} - {"samples/prepared_reader.py"}
        self.assertEqual(declared(code_root/"samples/application.py", "CALCULATION_MODULES"), preparation)
        read_dependencies = {"serialization.py", "samples/__init__.py", "samples/domain.py", "samples/ports.py",
                             "samples/application.py", "samples/numpy_store.py", "samples/prepared_reader.py", "samples/targets.py"}
        split_files = {"experiments/__init__.py", "experiments/domain.py", "experiments/ports.py", "experiments/application.py",
                       "experiments/sample_source.py", "experiments/numpy_store.py",
                       "experiments/evaluation_domain.py", "experiments/evaluation_ports.py"}
        self.assertEqual(declared(code_root/"experiments/application.py", "SPLIT_MODULES"), read_dependencies | split_files)
        common = read_dependencies | (split_files - {"experiments/sample_source.py"}) | {"resources.py", "experiments/validation_source.py"}
        regression = common | {"experiments/validation_domain.py", "experiments/validation_ports.py", "experiments/validation.py",
                               "experiments/validation_archive.py", "experiments/lightgbm_model.py", "experiments/evaluation.py"}
        self.assertEqual(declared(code_root/"experiments/validation.py", "VALIDATION_MODULES"), regression)
        prefix = "methods/next-day-up/src/next_day_up/"
        classification = common | {prefix+name for name in ("__init__.py", "domain.py", "ports.py", "evaluation.py",
                                                            "lightgbm_model.py", "archive.py", "validation.py")}
        self.assertEqual(declared(method/"validation.py", "VALIDATION_MODULES"), classification)
        from next_day_up.holdout import HOLDOUT_MODULES
        self.assertEqual(set(HOLDOUT_MODULES), classification | {"experiments/holdout_usage.py",
                                                                prefix+"holdout.py", prefix+"development_source.py"})
