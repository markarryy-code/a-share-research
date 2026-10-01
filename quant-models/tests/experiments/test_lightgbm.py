"""以真实CPU训练检查分箱参考、早停、列顺序及最佳轮数保存。"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from daily_return.experiments.lightgbm_model import LightGBMTrainer
from daily_return.experiments.validation_domain import ValidationSettings


class Telemetry:
    def checkpoint(self,phase):
        return {"elapsed_seconds":0,"working_set_bytes":None,"peak_working_set_bytes":None}


class NativeModelTests(unittest.TestCase):
    def test_real_model_uses_training_bins_and_reloads_at_best_iteration(self):
        config=json.loads((Path(__file__).resolve().parents[2]/"configs/p3-validation.json").read_text(encoding="utf-8"))
        config.update(num_boost_round=30,early_stopping_rounds=5)
        config["model_params"].update(num_threads=1,min_data_in_leaf=5)
        rng=np.random.default_rng(4)
        x=rng.normal(size=(300,49)).astype(np.float32)
        y=x[:,0]*.02-x[:,1]*.01
        x[240:,0]=1000
        names=tuple(f"feature_{i}" for i in range(49))
        trainer=LightGBMTrainer(Telemetry(),lambda event:None)
        model=trainer.fit(x[:240],y[:240],x[240:],y[240:],names,ValidationSettings.from_mapping(config))
        self.assertTrue(model.summary["validation_reference_is_train"])
        self.assertEqual(model.summary["dataset_rows"],{"train":240,"validation":60})
        info=model.booster.dump_model()["feature_infos"][names[0]]
        self.assertLess(info["max_value"],1000)
        self.assertGreater(model.summary["best_iteration"],0)
        self.assertLessEqual(model.summary["rounds_run"],30)
        prediction=model.predict(x[240:])
        checked=trainer.check_reload(model.export_text(),names,x[240:],prediction)
        self.assertEqual(checked["maximum_absolute_difference"],0)
        self.assertEqual([r["name"] for r in model.feature_usage()],list(names))
