# Stock

股票研究的通用 Python 计算代码。按问题查找下列目录，再沿入口与测试理解实现。

## 目录导航

以下仅介绍发布源码中的目录。

| 目录 | 何时进入 |
|---|---|
| [.githooks](.githooks/) | 查本地研究仓库防止误推旧历史的钩子。 |
| [quant-models](quant-models/) | 查量化计算代码及依赖声明。 |
| [src](quant-models/src/) | 进入共用数据与收益回归源码。 |
| [src/daily_return](quant-models/src/daily_return/) | 从命令入口和组装代码追踪运行。 |
| [market_data](quant-models/src/daily_return/market_data/) | 查资料准入、行情核验和读取。 |
| [samples](quant-models/src/daily_return/samples/) | 查日期对齐、样本资格、特征与目标。 |
| [experiments](quant-models/src/daily_return/experiments/) | 查时间切分、收益回归、评价与留出登记。 |
| [methods](quant-models/methods/) | 查独立研究方法。 |
| [methods/next-day-up](quant-models/methods/next-day-up/) | 了解次日上涨分类的代码与测试。 |
| [next-day-up/src](quant-models/methods/next-day-up/src/) | 定位分类方法的 Python 包。 |
| [next_day_up](quant-models/methods/next-day-up/src/next_day_up/) | 查上涨标签、分类训练与概率评价。 |
| [next-day-up/tests](quant-models/methods/next-day-up/tests/) | 查分类行为和留出隔离的测试。 |
| [analysis](quant-models/analysis/) | 查技术指标、形态与特征条件计算及测试。 |
| [tests](quant-models/tests/) | 查合成夹具与架构边界检查。 |
| [tests/market_data](quant-models/tests/market_data/) | 查行情核验规则的测试。 |
| [tests/samples](quant-models/tests/samples/) | 查样本与特征计算的测试。 |
| [tests/experiments](quant-models/tests/experiments/) | 查时间切分、模型与评价的测试。 |
| [tests/integration](quant-models/tests/integration/) | 查资料检查、样本准备及训练流程的衔接。 |

## 架构阅读

先明确任务的输入、输出和证据要求，再沿目录说明找到实际代码与调用关系。上述目录是当前导航；新增能力按职责、复用需求和维护成本选择归属，边界变化时同步设计、实现、相关检查与文档，保留历史证据的身份。

## 版本管理

Git 仅保留已核查的通用 Python 源码、由常数和公式构造的合成测试、依赖声明、本说明及版本管理配置。新增源码仍需核查是否内嵌真实数据；实际范围见根目录与 `quant-models/` 下的 `.gitignore` 允许清单。

真实数据及其研究、模型和实验产物，账户资料，绑定真实批次的配置，以及未经单独核查的工具、技能和本机资料均只留本机，不纳入当前或后续提交。仅检出源码不具备完整研究环境，部分集成测试仍依赖本机配置；可运行示例需另备合成输入。

发布采用独立源码快照，不携带本地研究仓库的旧提交历史。本地研究仓库仍通过 `.githooks/pre-push` 拦截直接推送；后续发布需核对文件范围和内容，不能因忽略规则已生效就直接上传旧历史。
