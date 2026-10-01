"""P3入口复用真实处理流程生成的小型合成产物，不读取正式行情或研究缓存。"""

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from fixtures import DataFixture, SRC, read_csv, write_csv, write_json


class ValidationFixture(DataFixture):
    def extend_training_history(self):
        """只扩充临时旧股历史，使0.8行采样有真实训练批量；不改正式参数。"""
        days=[]
        current=date(2026,8,19)
        while len(days)<40:
            if current.weekday()<5:days.append(current.isoformat())
            current-=timedelta(days=1)
        days.reverse()
        calendar=json.loads((self.data/"交易日期.json").read_text(encoding="utf-8"))
        write_json(self.data/"交易日期.json",days+calendar)
        frozen=json.loads((self.data/"冻结配置.json").read_text(encoding="utf-8"))
        frozen["start"]=days[0]
        write_json(self.data/"冻结配置.json",frozen)
        for folder in ("日线","日线派生涨跌","交易状态/逐日交易状态","网页股本估值","状态估值"):
            path=self.data/folder/"SZ000001.csv"
            rows=read_csv(path)
            earlier=[]
            for number,day in enumerate(days):
                row=rows[0].copy()
                row["TRADE_DATE" if folder=="网页股本估值" else "date"]=day+" 00:00:00" if folder=="网页股本估值" else day
                if folder=="日线派生涨跌":row["previous_actual_close_yuan"]="" if number==0 else "10"
                earlier.append(row)
            if folder=="日线派生涨跌":rows[0]["previous_actual_close_yuan"]="10"
            write_csv(path,earlier+rows)
        for symbol in ("sh000001","sz399001"):
            path=self.data/"市场行情"/(symbol+".csv")
            rows=read_csv(path)
            write_csv(path,[{**rows[0],"date":day} for day in days]+rows)
        admission=json.loads((self.data/"日线验收结果.json").read_text(encoding="utf-8"))
        admission["daily_rows"]+=len(days)
        write_json(self.data/"日线验收结果.json",admission)
        self.update_provenance()

    def command(self, command, *args):
        env=dict(os.environ,PYTHONPATH=str(SRC),PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")
        process=subprocess.run([sys.executable,"-B","-m","daily_return",command,"--module-root",str(self.module),*args],
                               env=env,capture_output=True,text=True,encoding="utf-8",timeout=45)
        lines=process.stdout.strip().splitlines()
        last=json.loads(lines[-1]) if lines else {}
        run=Path(last["run_directory"]) if "run_directory" in last else None
        return process,run

    def build_completed_inputs(self):
        process,_,p0=self.run_command()
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        config={"schema_version":1,"feature_set":"price-volume-v2","target_id":"next-market-day-quote-return-v1",
                "source_acceptance":{"inspection_run":str(p0.relative_to(self.module)),"supplements":[]},
                "split_plan":{"version":"expanding-target-date-v1","folds":[
                    {"id":"F01","train_through":"2026-08-21","validation_start":"2026-08-22","validation_end":"2026-08-24"}],
                    "holdout":{"start":"2026-08-25","end":"2026-08-27"}}}
        write_json(self.module/"configs/baseline.json",config)
        process,p1=self.command("prepare")
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        process,p2=self.command("prepare","--prepared-run",p1.relative_to(self.module).as_posix())
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        return p1,p2

    def setUp(self):
        super().setUp()
        self.extend_training_history()
        self.p1,self.p2=self.build_completed_inputs()
        self.config=json.loads((SRC.parent/"configs/p3-validation.json").read_text(encoding="utf-8"))
        self.config.update(split_run=self.p2.relative_to(self.module).as_posix(),num_boost_round=12,early_stopping_rounds=3,prediction_batch_size=1)
        self.config["model_params"].update(num_threads=1,min_data_in_leaf=1)
        write_json(self.module/"configs/p3-validation.json",self.config)

    def validate(self):
        process,run=self.command("validate","--fold","F01")
        result=json.loads((run/"validation.json").read_text(encoding="utf-8")) if run else None
        return process,result,run
